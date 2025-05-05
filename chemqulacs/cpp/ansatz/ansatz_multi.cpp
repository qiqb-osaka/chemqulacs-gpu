/**
 * @file ansatz_multi.cpp
 * @brief Implementation of the MultiAnsatz class
 * @author Yusuke Teranishi
 */
#include <ansatz/ansatz_multi.hpp>
#include <ansatz/gatefabric.hpp>
#include <utils/execute_manager.hpp>
#include <utils/simulation_config.hpp>
#include <utils/timer.hpp>

#include <mpi.h>
#include <omp.h>

#include <custatevec.h>
template class MultiAnsatz<double>;
template class MultiAnsatz<float>;

/**
 * @brief Calculate the minimum number of devices for malloc distributed state
 * vector
 * @tparam FP floating point type
 * @tparam AnsatzType Ansatz type
 * @param ansatz Ansatz object(ex. GateFabric)
 */
template <typename FP>
template <class AnsatzType>
int MultiAnsatz<FP>::_getComputeUnitNum(AnsatzType *ansatz) {
    int mpi_size = _ExecuteManager.getMpiSize();
    int n_qubits = ansatz->getQubits();
    int n_global_qubits = 0;
    for (; n_global_qubits < n_qubits; ++n_global_qubits) {
        if (StateVector<FP>::isAllocatable(n_qubits - n_global_qubits)) {
            break;
        }
    }
    assert(n_global_qubits < n_qubits);
    std::vector<int> other_global_qubits(mpi_size);
    HANDLE_MPI(MPI_Allgather(&n_global_qubits, 1, MPI_INT, other_global_qubits.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int i = 0; i < mpi_size; ++i) {
        assert(n_global_qubits == other_global_qubits[i]);
    }

    // Increase parallelism in case of too large mpi_size
    int n_params = ansatz->getParamSize();
    while ((1 << (n_global_qubits + 1)) * (n_params + 1) <= mpi_size && n_global_qubits + 1 < n_qubits) {
        n_global_qubits++;
    }
    int n_compute_unit = (1 << n_global_qubits);
    return n_compute_unit;
}

/**
 * @brief Initialize the MultiAnsatz object
 * @param ansatz Ansatz object(ex. GateFabric)
 * @param n_compute_unit Number of compute units
 */
template <typename FP>
template <class AnsatzType>
void MultiAnsatz<FP>::_init(AnsatzType *ansatz, int n_compute_unit) {
    int mpi_rank = _ExecuteManager.getMpiRank();
    int mpi_size = _ExecuteManager.getMpiSize();
    assert(0 <= n_compute_unit && n_compute_unit <= mpi_size);
    assert(mpi_size % n_compute_unit == 0);  // n_compute_unit is power of 2

    if (mpi_rank == 0 && n_compute_unit > 1 && _SimulationConfig.getSkipParamThreshold() != 0) {
        printf("Warning: parameter skip is disabled.\n");
    }

    _local_mpi_size = n_compute_unit;
    _global_mpi_size = mpi_size / _local_mpi_size;
    _global_mpi_rank = mpi_rank / _local_mpi_size;
    _local_mpi_rank = mpi_rank % _local_mpi_size;
    HANDLE_MPI(MPI_Comm_split(MPI_COMM_WORLD, _global_mpi_rank, _local_mpi_rank, &_local_mpi_comm));
    HANDLE_MPI(MPI_Comm_split(MPI_COMM_WORLD, _local_mpi_rank, _global_mpi_rank, &_global_mpi_comm));

    _ansatz = std::make_unique<AnsatzType>(ansatz, _local_mpi_comm);
    custatevecSetMpiCommunicator(_local_mpi_comm);

    _grad_mode = _SimulationConfig.getGradientMode();
    _n_params = _ansatz->getParamSize();
    int n_computes = _n_params + 1;  // +1 for E_0
    if (_grad_mode == GradientMode::CENTRAL) {
        n_computes = _n_params * 2;
    }
    _computed_id = 0;
    _before_id = 0;

    int num_ranks_low = n_computes % _global_mpi_size;
    _chunk_size_low = n_computes / _global_mpi_size;
    _chunk_size_high = (n_computes + _global_mpi_size - 1) / _global_mpi_size;
    if (_global_mpi_rank < num_ranks_low) {
        _chunk_size = _chunk_size_high;
    } else {
        _chunk_size = _chunk_size_low;
    }
    _grad_chunk.resize(_chunk_size_high, 0);
    _grad_recv.resize(_chunk_size_high * _global_mpi_size);
    _grad.resize(_n_params);

    if (_grad_mode == GradientMode::CHECKPOINT) {
        if (n_compute_unit > 1) {
            _grad_mode = GradientMode::BACKWARD;
            if (mpi_rank == 0) {
                printf(
                    "Warning: Gradient computation by checkpoint is disabled "
                    "due to multiple distributed state vector simulation.\n");
                // パウリ順序入れ替えるためupdateStateRangeが対応していない
            }
        } else if (!StateVector<FP>::isAllocatable(_ansatz->getLocalQubits())) {
            _grad_mode = GradientMode::BACKWARD;
            if (mpi_rank == 0) {
                printf(
                    "Warning: Gradient computation by checkpoint is disabled "
                    "due to out-of-memory.\n");
            }
        } else if (_chunk_size_high <= 1) {
            _grad_mode = GradientMode::BACKWARD;
            if (mpi_rank == 0) {
                printf(
                    "Warning: Gradient computation by checkpoint is disabled "
                    "due to not enough chunk_size.\n");
            }
        } else {
            _checkpoint_sv = std::make_unique<StateVector<FP> >(_ansatz->getLocalQubits(), MPI_COMM_SELF);
        }
    }

    if (mpi_rank == 0) {
        printf("[LOGCPP]n_params=%d,grad_mode=%s\n", _n_params, getGradientName(_grad_mode));
    }

    _timer_cost_bcast = _TimerDict.addTimerHost("cost_bcast");
    _timer_grad_compute = _TimerDict.addTimerHost("grad_compute");
    _timer_grad_allgather = _TimerDict.addTimerHost("grad_allgather");
    _timer_barrier = _TimerDict.addTimerHost("barrier");
}

/**
 * @brief Calculate the gradient by central difference
 * @param params Parameters
 * @param dx Difference
 * @param init_state Initial state
 * @return Gradient
 */
template <typename FP>
std::vector<double> MultiAnsatz<FP>::_numericalGradCentral(const std::vector<double> &params, double dx,
                                                           long long init_state) {
    _timer_grad_compute->restart();
    _ansatz->updateParams(params);
    for (int i = 0; i < _chunk_size; ++i) {
        int pid2 = _cid2pid(i);
        int param_id = pid2 / 2;
        double old_param = params[param_id];
        double new_param;
        if (pid2 % 2 == 0) {
            new_param = old_param + dx;
        } else {
            new_param = old_param - dx;
        }
        _grad_chunk[i] = _ansatz->computeExpectationWithUpdate(param_id, new_param, init_state);
        _ansatz->updateParam(param_id, old_param);
    }
    _timer_grad_compute->stop();

    _timer_grad_allgather->restart();
    if (_local_mpi_rank == 0) {
        HANDLE_MPI(MPI_Allgather(_grad_chunk.data(), _chunk_size_high, MPI_DOUBLE, _grad_recv.data(), _chunk_size_high,
                                 MPI_DOUBLE, _global_mpi_comm));
    }
    HANDLE_MPI(MPI_Bcast(_grad_recv.data(), (int)_grad_recv.size(), MPI_DOUBLE, 0, _local_mpi_comm));
    _timer_grad_allgather->stop();
    for (int i = 0; i < _n_params; ++i) {
        _grad[i] = (_grad_recv[_pid2recvid(i * 2)] - _grad_recv[_pid2recvid(i * 2 + 1)]) / (2 * dx);
    }

    return _grad;
}

/**
 * @brief Calculate the gradient by forward difference
 * @param params Parameters
 * @param dx Difference
 * @param init_state Initial state
 * @return Gradient
 */
template <typename FP>
std::vector<double> MultiAnsatz<FP>::_numericalGradForward(const std::vector<double> &params, double dx,
                                                           long long init_state) {
    _timer_grad_compute->restart();
    _ansatz->updateParams(params);
    if (_computed_id == 0 && _cid2pid(0) == 0) {
        _ansatz->initState(init_state);
        _ansatz->updateState();
        _grad_chunk[0] = _ansatz->computeExpectation();
        _computed_id++;
        printf("Computed:1/%d\n", _chunk_size);
    }
    for (int i = _computed_id; i < _chunk_size; ++i) {
        int param_id = _cid2pid(i) - 1;
        assert(param_id >= 0);
        double old_param = params[param_id];
        double new_param = old_param + dx;
        _grad_chunk[i] = _ansatz->computeExpectationWithUpdate(param_id, new_param, init_state);
        _ansatz->updateParam(param_id, old_param);
    }
    _timer_grad_compute->stop();

    _timer_grad_allgather->restart();
    if (_local_mpi_rank == 0) {
        HANDLE_MPI(MPI_Allgather(_grad_chunk.data(), _chunk_size_high, MPI_DOUBLE, _grad_recv.data(), _chunk_size_high,
                                 MPI_DOUBLE, _global_mpi_comm));
    }
    HANDLE_MPI(MPI_Bcast(_grad_recv.data(), (int)_grad_recv.size(), MPI_DOUBLE, 0, _local_mpi_comm));
    _timer_grad_allgather->stop();
    for (int i = 0; i < _n_params; ++i) {
        _grad[i] = (_grad_recv[_pid2recvid(i + 1)] - _grad_recv[0]) / dx;
    }
    _computed_id = 0;
    _before_id = 0;

    return _grad;
}

/**
 * @brief Calculate the gradient by backward difference
 * @param params Parameters
 * @param dx Difference
 * @param init_state Initial state
 * @return Gradient
 */
template <typename FP>
std::vector<double> MultiAnsatz<FP>::_numericalGradBackward(const std::vector<double> &params, double dx,
                                                            long long init_state) {
    _timer_grad_compute->restart();
    _ansatz->updateParams(params);
    if (_computed_id == 0 && _cid2pid(0) == 0) {
        _ansatz->initState(init_state);
        _ansatz->updateState();
        _grad_chunk[0] = _ansatz->computeExpectation();
        _computed_id++;
        printf("Computed:1/%d\n", _chunk_size);
    }
    for (int i = _computed_id; i < _chunk_size; ++i) {
        int param_id = _cid2pid(i) - 1;
        assert(param_id >= 0);
        double old_param = params[param_id];
        double new_param = old_param - dx;

        // std::time_t end_time = std::chrono::system_clock::to_time_t(
        //     std::chrono::system_clock::now());
        // std::string now_str = std::ctime(&end_time);
        // printf("rank=%d,before_compute:%s\n", _global_mpi_rank,
        //        now_str.c_str());

        _grad_chunk[i] = _ansatz->computeExpectationWithUpdate(param_id, new_param, init_state);

        // end_time = std::chrono::system_clock::to_time_t(
        //     std::chrono::system_clock::now());
        // now_str = std::ctime(&end_time);
        // printf("rank=%d,after_compute:%s\n", _global_mpi_rank,
        // now_str.c_str());

        _ansatz->updateParam(param_id, old_param);
        // if (_global_mpi_rank == 0 && _local_mpi_rank == 0) {
        //     printf("Computed:%d/%d\n", i + 1, _chunk_size);
        //     fflush(stdout);
        // }
    }
    _timer_grad_compute->stop();

    if (_local_mpi_rank == 0) {
        _timer_barrier->restart();
        HANDLE_MPI(MPI_Barrier(_global_mpi_comm));
        _timer_barrier->stop();

        // std::time_t end_time = std::chrono::system_clock::to_time_t(
        //     std::chrono::system_clock::now());
        // std::string now_str = std::ctime(&end_time);
        // printf("rank=%d,before_allgather:%s\n", _global_mpi_rank,
        //        now_str.c_str());
        _timer_grad_allgather->restart();
        HANDLE_MPI(MPI_Allgather(_grad_chunk.data(), _chunk_size_high, MPI_DOUBLE, _grad_recv.data(), _chunk_size_high,
                                 MPI_DOUBLE, _global_mpi_comm));

        // end_time = std::chrono::system_clock::to_time_t(
        //     std::chrono::system_clock::now());
        // now_str = std::ctime(&end_time);
        // printf("rank=%d,after_allgather:%s\n", _global_mpi_rank,
        //        now_str.c_str());
    }

    // std::time_t end_time =
    //     std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // std::string now_str = std::ctime(&end_time);
    // printf("rank=%d,before_bcast:%s\n", _global_mpi_rank, now_str.c_str());
    HANDLE_MPI(MPI_Bcast(_grad_recv.data(), (int)_grad_recv.size(), MPI_DOUBLE, 0, _local_mpi_comm));
    // end_time =
    //     std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // now_str = std::ctime(&end_time);
    // printf("rank=%d,after_bcast:%s\n", _global_mpi_rank, now_str.c_str());
    if (_local_mpi_rank == 0) {
        _timer_grad_allgather->stop();
    }
    for (int i = 0; i < _n_params; ++i) {
        _grad[i] = (_grad_recv[0] - _grad_recv[_pid2recvid(i + 1)]) / dx;
    }
    if (_local_mpi_rank == 0) {
        if (_local_mpi_rank == 0) {
            printf("[LOGCPP]grad=");
            for (int i = 0; i < _n_params; ++i) {
                printf("%f ", _grad[i]);
            }
            printf("\n");
        }
    }
    _computed_id = 0;
    _before_id = 0;

    return _grad;
}

/**
 * @brief Calculate the gradient by backward difference with checkpoint
 * @param params Parameters
 * @param dx Difference
 * @param init_state Initial state
 * @return Gradient
 */
template <typename FP>
std::vector<double> MultiAnsatz<FP>::_numericalGradCheckpoint(const std::vector<double> &params, double dx,
                                                              long long init_state) {
    assert(_checkpoint_sv);
    _timer_grad_compute->restart();
    _ansatz->updateParams(params);
    if (_computed_id == 0 && _cid2pid(0) == 0) {
        _ansatz->initState(init_state);
        _ansatz->updateState();
        _grad_chunk[0] = _ansatz->computeExpectation();
        _computed_id++;
        printf("Computed:1/%d\n", _chunk_size);
    }
    for (int i = _computed_id; i < _chunk_size; ++i) {
        int param_id = _cid2pid(i) - 1;
        double old_param = params[param_id];
        double new_param = old_param - dx;

        // std::time_t end_time = std::chrono::system_clock::to_time_t(
        //     std::chrono::system_clock::now());
        // std::string now_str = std::ctime(&end_time);
        // printf("rank=%d,before_compute:%s,i=%d,chunk_size=%d\n",
        //        _global_mpi_rank, now_str.c_str(), i, _chunk_size);

        _ansatz->updateParam(param_id, new_param);
        if (_before_id > 0) {
            _ansatz->loadState(_checkpoint_sv.get());
        } else {
            _ansatz->initState(init_state);
        }
        if (param_id > 0 && i + 1 < _chunk_size) {
            _ansatz->updateStateRange(_before_id, param_id);
            _ansatz->storeState(_checkpoint_sv.get());
            _ansatz->updateStateRange(param_id, _n_params);
            _before_id = param_id;
        } else {  // param_id==0 or last loop
            _ansatz->updateStateRange(_before_id, _n_params);
        }
        _grad_chunk[i] = _ansatz->computeExpectation();
        _ansatz->updateParam(param_id, old_param);

        // end_time = std::chrono::system_clock::to_time_t(
        //     std::chrono::system_clock::now());
        // now_str = std::ctime(&end_time);
        // printf("rank=%d,after_compute:%s\n", _global_mpi_rank,
        // now_str.c_str());

        // if (_global_mpi_rank == 0 && _local_mpi_rank == 0) {
        //     printf("Computed:%d/%d\n", i + 1, _chunk_size);
        //     fflush(stdout);
        // }
    }
    _timer_grad_compute->stop();

    if (_local_mpi_rank == 0) {
        _timer_barrier->restart();
        HANDLE_MPI(MPI_Barrier(_global_mpi_comm));
        _timer_barrier->stop();

        // std::time_t end_time = std::chrono::system_clock::to_time_t(
        //     std::chrono::system_clock::now());
        // std::string now_str = std::ctime(&end_time);
        // printf("rank=%d,before_allgather:%s\n", _global_mpi_rank,
        //    now_str.c_str());
        _timer_grad_allgather->restart();
        HANDLE_MPI(MPI_Allgather(_grad_chunk.data(), _chunk_size_high, MPI_DOUBLE, _grad_recv.data(), _chunk_size_high,
                                 MPI_DOUBLE, _global_mpi_comm));

        // end_time = std::chrono::system_clock::to_time_t(
        //     std::chrono::system_clock::now());
        // now_str = std::ctime(&end_time);
        // printf("rank=%d,after_allgather:%s\n", _global_mpi_rank,
        //        now_str.c_str());
    }
    // std::time_t end_time =
    //     std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // std::string now_str = std::ctime(&end_time);
    // printf("rank=%d,before_bcast:%s\n", _global_mpi_rank, now_str.c_str());
    HANDLE_MPI(MPI_Bcast(_grad_recv.data(), (int)_grad_recv.size(), MPI_DOUBLE, 0, _local_mpi_comm));
    // end_time =
    //     std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // now_str = std::ctime(&end_time);
    // printf("rank=%d,after_bcast:%s\n", _global_mpi_rank, now_str.c_str());
    if (_local_mpi_rank == 0) {
        _timer_grad_allgather->stop();
    }
    for (int i = 0; i < _n_params; ++i) {
        _grad[i] = (_grad_recv[0] - _grad_recv[_pid2recvid(i + 1)]) / dx;
    }
    _computed_id = 0;
    _before_id = 0;

    return _grad;
}

/**
 * @brief Calculate the gradient by numerical method
 * @param params Parameters
 * @param dx Difference
 * @param init_state Initial state
 * @return Gradient
 */
template <typename FP>
std::vector<double> MultiAnsatz<FP>::numericalGrad(const std::vector<double> &params, double dx, long long init_state) {
    switch (_grad_mode) {
        case GradientMode::CENTRAL:
            return _numericalGradCentral(params, dx, init_state);
        case GradientMode::FORWARD:
            return _numericalGradForward(params, dx, init_state);
        case GradientMode::BACKWARD:
            return _numericalGradBackward(params, dx, init_state);
        case GradientMode::CHECKPOINT:
            return _numericalGradCheckpoint(params, dx, init_state);
        default:
            return std::vector<double>();
    }
}

/**
 * @brief Calculate the expectation value
 * @param params Parameters
 * @param dx Difference
 * @param init_state Initial state
 * @return Expectation value
 */
template <typename FP>
double MultiAnsatz<FP>::computeExpectation(const std::vector<double> &params, double dx, long long init_state) {
    _ansatz->initState(init_state);
    _ansatz->updateParams(params);
    double exp = 0;
    if (_global_mpi_rank == 0) {
        if (_checkpoint_sv && _chunk_size > 1) {
            int next_param_id = _cid2pid(1) - 1;
            _timer.restart();
            _ansatz->updateStateRange(0, next_param_id);
            _timer.stop();
            _ansatz->storeState(_checkpoint_sv.get());
            _timer.start();
            _ansatz->updateStateRange(next_param_id, _n_params);
            _timer.stop();
            _before_id = next_param_id;
        } else {
            _ansatz->updateState();
            _before_id = 0;
        }
        _grad_chunk[0] = _ansatz->computeExpectation();
        exp = _grad_chunk[0];
        if (_checkpoint_sv && _chunk_size > 1) {
            _TimerDict.registerTime("update", _timer.getTime());
            _TimerDict.registerTime("update_compute", _timer.getTime());
        }
    } else if (_chunk_size > 0) {
        int param_id = _cid2pid(0) - 1;
        double old_param = params[param_id];
        double new_param = old_param - dx;
        _ansatz->updateParam(param_id, new_param);
        if (_checkpoint_sv && param_id > 0 && _chunk_size > 1) {
            _ansatz->updateStateRange(0, param_id);
            _ansatz->storeState(_checkpoint_sv.get());
            _ansatz->updateStateRange(param_id, _n_params);
            _before_id = param_id;
        } else {
            _ansatz->updateState();
            _before_id = 0;
        }
        _grad_chunk[0] = _ansatz->computeExpectation();
    }
    // else {
    //     printf("Warning: too many computing resources.\n");
    // }
    _timer_cost_bcast->restart();
    HANDLE_MPI(MPI_Bcast(&exp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    _timer_cost_bcast->stop();
    _computed_id = 1;
    // if (_global_mpi_rank == 0 && _local_mpi_rank == 0) {
    //     printf("Computed:%d/%d\n", 1, _chunk_size);
    //     fflush(stdout);
    // }
    return exp;
}

/**
 * @brief Calculate the gradient by parameter shift
 * @param params Parameters
 * @param init_state Initial state
 * @return Gradient
 */
template <typename FP>
std::vector<double> MultiAnsatz<FP>::parameterShift(const std::vector<double> &params, long long init_state) {
    int n_params = params.size();
    int n_computes = n_params * 2;
    int num_ranks_low = n_computes % _global_mpi_size;
    int chunk_size_low = n_computes / _global_mpi_size;
    int chunk_size_high = chunk_size_low + (num_ranks_low == 0 ? 0 : 1);
    int chunk_size = 0;
    if (_global_mpi_rank < num_ranks_low) {
        chunk_size = chunk_size_high;
    } else {
        chunk_size = chunk_size_low;
    }

    std::vector<double> recv_data(n_computes);
    std::vector<double> chunk_data(chunk_size_high);

    _ansatz->updateParams(params);
    for (int i = 0; i < chunk_size; ++i) {
        int idx = _cid2pid(i);
        int param_id = idx / 2;
        double old_param = params[param_id];
        double new_param;
        if (idx % 2 == 0) {
            new_param = old_param + M_PI / 2;
        } else {
            new_param = old_param - M_PI / 2;
        }
        chunk_data[i] = _ansatz->computeExpectationWithUpdate(param_id, new_param, init_state);
        _ansatz->updateParam(param_id, old_param);
    }
    if (_local_mpi_rank == 0) {
        HANDLE_MPI(MPI_Allgather(chunk_data.data(), chunk_size_high, MPI_DOUBLE, recv_data.data(), chunk_size_high,
                                 MPI_DOUBLE, _global_mpi_comm));
    }
    HANDLE_MPI(MPI_Bcast(recv_data.data(), (int)recv_data.size(), MPI_DOUBLE, 0, _local_mpi_comm));
    for (int i = 0; i < _n_params; ++i) {
        _grad[i] = (recv_data[_pid2recvid(i * 2)] - recv_data[_pid2recvid(i * 2 + 1)]) / 2;
    }

    return _grad;
}

/**
 * @brief Calculate the gradient by NFT(Notch Fourier Transform)
 * @param params Parameters
 * @param dx Difference
 * @param init_state Initial state
 * @return Gradient
 */
template <typename FP>
std::vector<double> MultiAnsatz<FP>::computeNFT(const std::vector<double> &params, long long init_state) {
    int n_params = params.size();
    int n_computes = n_params * 2;
    int num_ranks_low = n_computes % _global_mpi_size;
    int chunk_size_low = n_computes / _global_mpi_size;
    int chunk_size_high = chunk_size_low + (num_ranks_low == 0 ? 0 : 1);
    int chunk_size = 0;
    if (_global_mpi_rank < num_ranks_low) {
        chunk_size = chunk_size_high;
    } else {
        chunk_size = chunk_size_low;
    }

    std::vector<double> recv_data(n_computes);
    std::vector<double> chunk_data(chunk_size_high);

    _params = params;

    double z0 = _ansatz->computeExpectationWithUpdate(params, init_state);
    _ansatz->updateParams(params);
    for (int i = 0; i < chunk_size; ++i) {
        int idx = _cid2pid(i);
        int param_id = idx / 2;
        double old_param = params[param_id];
        double new_param;
        if (idx % 2 == 0) {
            new_param = old_param + M_PI / 2;
        } else {
            new_param = old_param - M_PI / 2;
        }
        _params[param_id] = new_param;
        chunk_data[i] = _ansatz->computeExpectationWithUpdate(_params, init_state);
        _params[param_id] = old_param;
        // chunk_data[i] = _ansatz->computeExpectationWithUpdate(
        //     param_id, new_param, init_state);
        printf("i=%d,new_param=%.16lf,value=%.16lf\n", i, new_param, chunk_data[i]);
        // _ansatz->updateParam(param_id, old_param);
    }
    if (_local_mpi_rank == 0) {
        HANDLE_MPI(MPI_Allgather(chunk_data.data(), chunk_size_high, MPI_DOUBLE, recv_data.data(), chunk_size_high,
                                 MPI_DOUBLE, _global_mpi_comm));
    }
    HANDLE_MPI(MPI_Bcast(recv_data.data(), (int)recv_data.size(), MPI_DOUBLE, 0, _local_mpi_comm));
    for (int i = 0; i < _n_params; ++i) {
        double z1 = recv_data[_pid2recvid(i * 2)];  // warning: chunk_size
        double z3 = recv_data[_pid2recvid(i * 2 + 1)];
        double z2 = z1 + z3 - z0;
        double sign = 0;
        if (z0 - z2 > 0) {
            sign = 1;
        } else if (z0 - z2 < 0) {
            sign = -1;
        }
        double b = params[i] + atan((z1 - z3) / (z0 - z2)) + 0.5 * M_PI + 0.5 * M_PI * sign;
        printf("i=%d,z1=%.10lf,z2=%.10lf,z3=%.10lf,b=%.10lf\n", i, z1, z2, z3, b);
        _grad[i] = b;
    }
    return _grad;
}
