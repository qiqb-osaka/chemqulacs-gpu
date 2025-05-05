/**
 * @file ansatz_multi.hpp
 * @brief Header of the MultiAnsatz class
 * @author Yusuke Teranishi
 */
#pragma once

#include <ansatz/ansatz.hpp>
#include <ansatz/gatefabric.hpp>
#include <ansatz/pauli_exp.hpp>
#include <utils/simulation_config.hpp>
#include <utils/timer.hpp>

#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

/**
 * @brief Class for the multi ansatz
 * @tparam FP floating point type
 */
template <typename FP>
class MultiAnsatz {
   private:
    std::unique_ptr<Ansatz<FP>> _ansatz;
    std::unique_ptr<StateVector<FP>> _checkpoint_sv;
    int _n_params;

    MPI_Comm _local_mpi_comm;
    MPI_Comm _global_mpi_comm;

    GradientMode _grad_mode;

    int _global_mpi_rank;
    int _global_mpi_size;
    int _local_mpi_rank;
    int _local_mpi_size;
    int _chunk_idx;
    int _chunk_size;
    int _chunk_size_low;
    int _chunk_size_high;
    int _computed_id;
    int _before_id;
    std::vector<double> _grad_chunk;  // chunk_size
    std::vector<double> _grad_recv;   // chunk_size * mpi_size
    std::vector<double> _grad;        // _n_params
    std::vector<double> _params;      // _n_params

    Timer *_timer_cost_bcast;
    Timer *_timer_grad_compute;
    Timer *_timer_grad_allgather;
    Timer *_timer_barrier;

    TimerDevice _timer;
    double _time_init;
    double _time_copy;
    double _time_update_avg;

    template <class AnsatzType>
    int _getComputeUnitNum(AnsatzType *ansatz);

    template <class AnsatzType>
    void _init(AnsatzType *ansatz, int n_compute_unit);

    std::vector<double> _numericalGradCentral(const std::vector<double> &params, double dx, long long init_state);

    std::vector<double> _numericalGradForward(const std::vector<double> &params, double dx, long long init_state);

    std::vector<double> _numericalGradBackward(const std::vector<double> &params, double dx, long long init_state);

    std::vector<double> _numericalGradCheckpoint(const std::vector<double> &params, double dx, long long init_state);

    int _pid2recvid(int param_id) {
        return (param_id % _global_mpi_size) * _chunk_size_high + (param_id / _global_mpi_size);
    }  ///< Convert parameter id to receive id
    int _cid2pid(int chunk_id) {
        return _global_mpi_rank + _global_mpi_size * chunk_id;
    }  ///< Convert chunk id to parameter id

   public:
    MultiAnsatz(GateFabric<FP> *ansatz, int n_compute_unit) { _init(ansatz, n_compute_unit); }
    MultiAnsatz(GateFabric<FP> *ansatz) : MultiAnsatz(ansatz, _getComputeUnitNum(ansatz)) {}

    MultiAnsatz(PauliExp<FP> *ansatz, int n_compute_unit) { _init(ansatz, n_compute_unit); }
    MultiAnsatz(PauliExp<FP> *ansatz) : MultiAnsatz(ansatz, _getComputeUnitNum(ansatz)) {}

    std::vector<double> numericalGrad(const std::vector<double> &params, double dx, long long init_state);

    double computeExpectation(const std::vector<double> &params, double dx, long long init_state);

    std::vector<double> parameterShift(const std::vector<double> &params, long long init_state);

    std::vector<double> computeNFT(const std::vector<double> &params, long long init_state);
};
