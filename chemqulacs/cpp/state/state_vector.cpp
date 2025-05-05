/**
 * @file state_vector.cpp
 * @brief Implementation of the StateVector class
 * @author Yusuke Teranishi
 */
#include <pauli/pauli_product.hpp>
#include <state/state_vector.hpp>
#include <utils/execute_manager.hpp>
#include <utils/precision.hpp>
#include <utils/timer.hpp>

#include <mpi.h>
#include <nvtx3/nvToolsExt.h>
#include <omp.h>

#include <cassert>
#include <iostream>

template class StateVector<double>;
template class StateVector<float>;

/**
 * @brief malloc StateVector
 */
template <typename FP>
void StateVector<FP>::_mallocStateVector() {
    HANDLE_CUDA(cudaMalloc(&_d_sv_ptr, sizeof(cuFpComplex<FP>) * _sv_len_per_device));
    if (_n_devices > 1) {
        _distributedStateVectorCreate();
    }
}

/**
 * @brief free StateVector
 */
template <typename FP>
void StateVector<FP>::_freeStateVector() {
    if (_n_devices > 1) {
        _distributedStateVectorDestroy();
    }
    HANDLE_CUDA(cudaFree(_d_sv_ptr));
}

/**
 * @brief create distributed StateVector
 */
template <typename FP>
void StateVector<FP>::_distributedStateVectorCreate() {
    size_t cusv_extra_workspace_size;
    size_t cusv_min_transfer_workspace_size;
    HANDLE_CUSV(custatevecSVSwapWorkerCreate(
        _ExecuteManager.getCusvHandle(), &_cusv_swap_worker, _ExecuteManager.getCusvCommunicator(), _d_sv_ptr,
        _mpi_local_rank, _ExecuteManager.getCudaEvent(), CUDA_C_<FP>, _ExecuteManager.getCudaStream(),
        &cusv_extra_workspace_size, &cusv_min_transfer_workspace_size));

    // プロセスごとに異なりバグる
    size_t available_device_mem_size = _ExecuteManager.getAvailableDeviceMemSize();
    size_t max_workspace_size_my_gpu = sizeof(cuFpComplex<FP>);
    for (int i_qubits = 0; i_qubits <= this->_n_local_qubits - 1; ++i_qubits) {
        if ((max_workspace_size_my_gpu << 1) > available_device_mem_size) {
            break;
        }
        max_workspace_size_my_gpu <<= 1;
    }
    max_workspace_size_my_gpu = std::max(max_workspace_size_my_gpu, cusv_min_transfer_workspace_size);

    size_t cusv_transfer_workspace_size;
    HANDLE_MPI(MPI_Allreduce(&max_workspace_size_my_gpu, &cusv_transfer_workspace_size, 1, MPI_UNSIGNED_LONG_LONG,
                             MPI_MIN, _mpi_comm));
    // printf(
    //     "available_device_mem_size=%lu, "
    //     "cusv_min_transfer_workspace_size=%lu, "
    //     "cusv_transfer_workspace_size=%lu\n",
    //     available_device_mem_size, cusv_min_transfer_workspace_size,
    //     cusv_transfer_workspace_size);

    HANDLE_CUDA(cudaMalloc(&_d_cusv_extra_workspace, cusv_extra_workspace_size));
    HANDLE_CUDA(cudaMalloc(&_d_cusv_transfer_workspace, cusv_transfer_workspace_size));
    HANDLE_CUSV(custatevecSVSwapWorkerSetExtraWorkspace(_ExecuteManager.getCusvHandle(), _cusv_swap_worker,
                                                        _d_cusv_extra_workspace, cusv_extra_workspace_size));
    HANDLE_CUSV(custatevecSVSwapWorkerSetTransferWorkspace(_ExecuteManager.getCusvHandle(), _cusv_swap_worker,
                                                           _d_cusv_transfer_workspace, cusv_transfer_workspace_size));
    HANDLE_CUSV(custatevecDistIndexBitSwapSchedulerCreate(_ExecuteManager.getCusvHandle(), &_cusv_swap_scheduler,
                                                          _n_global_qubits, _n_local_qubits));

    if (_is_p2p) {
        cudaIpcMemHandle_t ipc_mem_handle;
        std::vector<cudaIpcMemHandle_t> ipc_mem_handles(_n_devices);
        cudaIpcEventHandle_t ipc_event_handle;
        std::vector<cudaIpcEventHandle_t> ipc_event_handles(_n_devices);
        int mpi_global_rank = _ExecuteManager.getMpiRank();
        std::vector<int> mpi_ranks(_n_devices);
        HANDLE_CUDA(cudaIpcGetMemHandle(&ipc_mem_handle, _d_sv_ptr));
        HANDLE_CUDA(cudaIpcGetEventHandle(&ipc_event_handle, _ExecuteManager.getCudaEvent()));

        HANDLE_MPI(MPI_Allgather(&ipc_mem_handle, sizeof(ipc_mem_handle), MPI_UINT8_T, ipc_mem_handles.data(),
                                 sizeof(ipc_mem_handle), MPI_UINT8_T, _mpi_comm));
        HANDLE_MPI(MPI_Allgather(&ipc_event_handle, sizeof(ipc_event_handle), MPI_UINT8_T, ipc_event_handles.data(),
                                 sizeof(ipc_event_handle), MPI_UINT8_T, _mpi_comm));
        HANDLE_MPI(MPI_Allgather(&mpi_global_rank, 1, MPI_INT, mpi_ranks.data(), 1, MPI_INT, _mpi_comm));

        std::vector<int> remote_sv_indexes;
        int my_node = mpi_global_rank / _ExecuteManager.getDevicesPerNode();
        for (int i_sv = 0; i_sv < _n_devices; ++i_sv) {
            if (i_sv == _mpi_local_rank) {
                continue;
            }
            int sv_node = mpi_ranks[i_sv] / _ExecuteManager.getDevicesPerNode();
            if (sv_node == my_node) {
                void *remote_ipc_mem_handle;
                const auto &dst_mem_handle = ipc_mem_handles[i_sv];
                HANDLE_CUDA(
                    cudaIpcOpenMemHandle(&remote_ipc_mem_handle, dst_mem_handle, cudaIpcMemLazyEnablePeerAccess));

                _remote_ipc_mem_handles.push_back(remote_ipc_mem_handle);
                cudaEvent_t remote_event;
                HANDLE_CUDA(cudaIpcOpenEventHandle(&remote_event, ipc_event_handles[i_sv]));
                _remote_events.push_back(remote_event);
                remote_sv_indexes.push_back(i_sv);
            }
        }
        HANDLE_CUSV(custatevecSVSwapWorkerSetSubSVsP2P(_ExecuteManager.getCusvHandle(), _cusv_swap_worker,
                                                       _remote_ipc_mem_handles.data(), remote_sv_indexes.data(),
                                                       _remote_events.data(), remote_sv_indexes.size()));
    }
}

/**
 * @brief destroy distributed StateVector
 */
template <typename FP>
void StateVector<FP>::_distributedStateVectorDestroy() {
    if (_is_p2p) {
        for (void *remote_ipc_mem_handle : _remote_ipc_mem_handles) {
            HANDLE_CUDA(cudaIpcCloseMemHandle(remote_ipc_mem_handle));
        }
        for (cudaEvent_t remote_event : _remote_events) {
            HANDLE_CUDA(cudaEventDestroy(remote_event));
        }
    }
    HANDLE_CUSV(custatevecDistIndexBitSwapSchedulerDestroy(_ExecuteManager.getCusvHandle(), _cusv_swap_scheduler));
    HANDLE_CUDA(cudaFree(_d_cusv_transfer_workspace));
    HANDLE_CUDA(cudaFree(_d_cusv_extra_workspace));
    HANDLE_CUSV(custatevecSVSwapWorkerDestroy(_ExecuteManager.getCusvHandle(), _cusv_swap_worker));
}

/**
 * @brief get Host pointer
 */
template <typename FP>
cuFpComplex<FP> *StateVector<FP>::getHostPtr() {
    if (!_h_sv_ptr) {
        _h_sv_ptr = std::make_unique<cuFpComplex<FP>>(1LL << _n_local_qubits);
    }

    HANDLE_CUDA(cudaMemcpyAsync(_h_sv_ptr.get(), _d_sv_ptr, sizeof(cuFpComplex<FP>) * _sv_len_per_device,
                                cudaMemcpyDeviceToHost, _ExecuteManager.getCudaStream()));
    HANDLE_CUDA(cudaStreamSynchronize(_ExecuteManager.getCudaStream()));
    return _h_sv_ptr.get();
}

/**
 * @brief Initialize the state vector by binary string
 */
template <typename FP>
void StateVector<FP>::initState(const long long state_bit) {
    HANDLE_CUDA(
        cudaMemsetAsync(_d_sv_ptr, 0, sizeof(cuFpComplex<FP>) * _sv_len_per_device, _ExecuteManager.getCudaStream()));
    int target_rank = state_bit / _sv_len_per_device;
    if (_mpi_local_rank == target_rank) {
        cuFpComplex<FP> v = make_cuFpComplex<FP>(1, 0);
        cuFpComplex<FP> *d_sv_ptr = reinterpret_cast<cuFpComplex<FP> *>(_d_sv_ptr);
        HANDLE_CUDA(cudaMemcpyAsync(d_sv_ptr + state_bit % _sv_len_per_device, &v, sizeof(cuFpComplex<FP>),
                                    cudaMemcpyHostToDevice, _ExecuteManager.getCudaStream()));
    }
}

/**
 * @brief Initialize the state vector to zero
 */
template <typename FP>
void StateVector<FP>::zeroState() {
    HANDLE_CUDA(
        cudaMemsetAsync(_d_sv_ptr, 0, sizeof(cuFpComplex<FP>) * _sv_len_per_device, _ExecuteManager.getCudaStream()));
}

/**
 * @brief Copy the state vector from another state vector
 */
template <typename FP>
void StateVector<FP>::copyStateFrom(StateVector<FP> *state_vector) {
    assert(_sv_len_per_device == state_vector->getLenPerDevice());
    HANDLE_CUDA(cudaMemcpyAsync(_d_sv_ptr, state_vector->getDevicePtr(), sizeof(cuFpComplex<FP>) * _sv_len_per_device,
                                cudaMemcpyDeviceToDevice));
}

/**
 * @brief Calculate the dot product of two state vectors with double precision
 * @param[in] state_vector StateVector object
 * @return Dot product
 */
template <>
cuDoubleComplex StateVector<double>::dotProduct(StateVector<double> *state_vector) {
    assert(_sv_len_per_device == state_vector->getLenPerDevice());
    cuDoubleComplex val;
    HANDLE_CUBLAS(cublasZdotc(_ExecuteManager.getCublasHandle(), _sv_len_per_device,
                              reinterpret_cast<cuDoubleComplex *>(_d_sv_ptr), 1,
                              reinterpret_cast<cuDoubleComplex *>(state_vector->getDevicePtr()), 1, &val));
    return val;
}

/**
 * @brief Calculate the dot product of two state vectors with single precision
 * @param[in] state_vector StateVector object
 * @return Dot product
 */
template <>
cuFloatComplex StateVector<float>::dotProduct(StateVector<float> *state_vector) {
    assert(_sv_len_per_device == state_vector->getLenPerDevice());
    cuFloatComplex val;
    HANDLE_CUBLAS(cublasCdotc(_ExecuteManager.getCublasHandle(), _sv_len_per_device,
                              reinterpret_cast<cuFloatComplex *>(_d_sv_ptr), 1,
                              reinterpret_cast<cuFloatComplex *>(state_vector->getDevicePtr()), 1, &val));
    return val;
}

/**
 * @brief Add a state vector multiplied by a coefficient to the current state
 * vector
 * @param[in] state_vector StateVector object
 * @param[in] coef Coefficient
 */
template <>
void StateVector<double>::add(StateVector<double> *state_vector, std::complex<double> coef) {
    assert(_sv_len_per_device == state_vector->getLenPerDevice());
    cuDoubleComplex alpha = make_cuDoubleComplex(coef.real(), coef.imag());
    HANDLE_CUBLAS(cublasZaxpy(_ExecuteManager.getCublasHandle(), _sv_len_per_device, &alpha,
                              reinterpret_cast<cuDoubleComplex *>(state_vector->getDevicePtr()), 1,
                              reinterpret_cast<cuDoubleComplex *>(_d_sv_ptr), 1));
}

/**
 * @brief Calculate the L2 norm of the state vector
 * @param[in] state_vector StateVector object
 * @return L2 norm
 */
template <>
double StateVector<double>::getL2Norm(StateVector<double> *state_vector) {
    add(state_vector, -1.0);
    cuDoubleComplex val;
    HANDLE_CUBLAS(cublasZdotc(_ExecuteManager.getCublasHandle(), _sv_len_per_device,
                              reinterpret_cast<cuDoubleComplex *>(_d_sv_ptr), 1,
                              reinterpret_cast<cuDoubleComplex *>(_d_sv_ptr), 1, &val));
    add(state_vector);
    return val.x;
}

/**
 * @brief Check if the state vector is equal to another state vector
 * @param[in] state_vector StateVector object
 * @param[in] eps Threshold for equality
 * @return True if the state vectors are equal within the threshold
 */
template <>
bool StateVector<double>::isEqual(StateVector<double> *state_vector, double eps) {
    double norm = getL2Norm(state_vector);
    return norm < eps;
}

/**
 * @brief Print the state vector
 * @warning Multiple calls to this function may cause bugs
 */
template <>
void StateVector<double>::printSv() {
    // if (_mpi_local_rank != 0) return;
    cuDoubleComplex *h_sv_ptr = getHostPtr();
    for (int i = 0; i < _sv_len_per_device; ++i) {
        if (abs(h_sv_ptr[i].x) > 1e-5 || abs(h_sv_ptr[i].y) > 1e-5) {
            std::cout << "local_rank: " << _mpi_local_rank << std::endl;
            std::cout << "i: " << i << " "
                      << "x: " << h_sv_ptr[i].x << " y: " << h_sv_ptr[i].y << "\n";
        }
    }
}

/**
 * @brief Apply a Pauli rotation to the state vector
 * @param[in] pauli_product PauliProduct object
 * @param[in] theta Rotation angle
 * @param[in] controls Control qubits
 */
template <typename FP>
void StateVector<FP>::ApplyPauliRotation(PauliProduct &pauli_product, double theta, std::vector<int> &controls,
                                         std::vector<int> &targets) {
    if (pauli_product.getPauliOperator().empty()) {
        return;
    }
    nvtxRangePush("ApplyPauliRotation");
    HANDLE_CUSV(custatevecApplyPauliRotation(_ExecuteManager.getCusvHandle(), _d_sv_ptr, CUDA_C_<FP>, _n_local_qubits,
                                             theta, pauli_product.getPauliOperator().data(), targets.data(),
                                             targets.size(), controls.data(), nullptr, controls.size()));
    nvtxRangePop();
}

/**
 * @brief Swap qubits positions in the state vector from local to local
 * @param[in] swap_qubits Qubits to swap
 */
template <typename FP>
void StateVector<FP>::swapLocalQubits(std::vector<int2> &swap_qubits) {
    nvtxRangePush("swapLocalQubits");
    HANDLE_CUSV(custatevecSwapIndexBits(_ExecuteManager.getCusvHandle(), _d_sv_ptr, CUDA_C_<FP>, _n_local_qubits,
                                        swap_qubits.data(), swap_qubits.size(), nullptr, nullptr, 0));
    nvtxRangePop();
}

/**
 * @brief Swap qubits positions in the state vector from local to Global or
 * Global to local
 * @brief swap_qubits Qubits to swap
 */
template <typename FP>
void StateVector<FP>::swapGlobalQubits(std::vector<int2> &swap_qubits) {
    if (_is_p2p) {
        HANDLE_CUDA(cudaStreamSynchronize(_ExecuteManager.getCudaStream()));
        HANDLE_MPI(MPI_Barrier(_mpi_comm));
    }
    nvtxRangePush("swapGlobalQubits");
    unsigned int n_swap_batches;
    HANDLE_CUSV(custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(
        _ExecuteManager.getCusvHandle(), _cusv_swap_scheduler, swap_qubits.data(), swap_qubits.size(), nullptr, nullptr,
        0, &n_swap_batches));
    for (unsigned int i_swap_batch = 0; i_swap_batch < n_swap_batches; ++i_swap_batch) {
        custatevecSVSwapParameters_t parameters;
        HANDLE_CUSV(custatevecDistIndexBitSwapSchedulerGetParameters(
            _ExecuteManager.getCusvHandle(), _cusv_swap_scheduler, i_swap_batch, _mpi_local_rank, &parameters));
        HANDLE_CUSV(custatevecSVSwapWorkerSetParameters(_ExecuteManager.getCusvHandle(), _cusv_swap_worker, &parameters,
                                                        parameters.dstSubSVIndex));
        HANDLE_CUSV(custatevecSVSwapWorkerExecute(_ExecuteManager.getCusvHandle(), _cusv_swap_worker, 0,
                                                  parameters.transferSize));
    }
    nvtxRangePop();
}

/**
 * @brief Swap qubits positions of global qubits and local first bits
 */
template <typename FP>
void StateVector<FP>::swapGlobalQubits() {
    std::vector<int2> swap_qubits(_n_global_qubits);
    for (int i = 0; i < _n_global_qubits; ++i) {
        swap_qubits[i] = {i, i + _n_local_qubits};
    }
    swapGlobalQubits(swap_qubits);
}
