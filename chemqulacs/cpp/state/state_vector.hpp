/**
 * @file state_vector.hpp
 * @brief Header file for StateVector class
 * @author Yusuke Teranishi
 */
#pragma once

#include <pauli/pauli_product.hpp>
#include <utils/execute_manager.hpp>
#include <utils/precision.hpp>

#include <cublas_v2.h>
#include <custatevec.h>

#include <cassert>
#include <complex>
#include <memory>
#include <vector>

/**
 * @brief StateVector class
 */
template <typename FP>
class StateVector {
   private:
    int _n_qubits;                 ///< Number of qubits
    int _n_local_qubits;           ///< Number of local qubits
    int _n_global_qubits;          ///< Number of global qubits
    long long _sv_len_per_device;  ///< Length of state vector per device

    MPI_Comm _mpi_comm;   ///< Type of MPI communicator
    int _mpi_local_rank;  ///< Local rank
    int _n_devices;       ///< Number of devices

    void *_d_sv_ptr;                             ///< Device pointer
    void *_d_cusv_extra_workspace;               ///< Extra workspace for cuStateVec
    void *_d_cusv_transfer_workspace;            ///< Transfer workspace for cuStateVec
    std::unique_ptr<cuFpComplex<FP>> _h_sv_ptr;  ///< Host pointer

    custatevecSVSwapWorkerDescriptor_t _cusv_swap_worker;                  ///< Worker descriptor
    custatevecDistIndexBitSwapSchedulerDescriptor_t _cusv_swap_scheduler;  ///< Scheduler descriptor

    bool _is_p2p;                                 ///< Whether P2P communication is enabled
    std::vector<void *> _remote_ipc_mem_handles;  ///< Remote IPC memory handles
    std::vector<cudaEvent_t> _remote_events;      ///< Remote events

    void _mallocStateVector();
    void _freeStateVector();

    void _distributedStateVectorCreate();
    void _distributedStateVectorDestroy();

   public:
    /**
     * @brief Constructor
     * @param[in] n_qubits Number of qubits
     * @param[in] mpi_comm Type of MPI communicator
     * @param[in] is_p2p Whether P2P communication is enabled
     */
    StateVector(int n_qubits, MPI_Comm mpi_comm = MPI_COMM_WORLD, bool is_p2p = true) {
        _n_qubits = n_qubits;
        _mpi_comm = mpi_comm;
        _is_p2p = is_p2p;

        HANDLE_MPI(MPI_Comm_rank(_mpi_comm, &_mpi_local_rank));
        HANDLE_MPI(MPI_Comm_size(_mpi_comm, &_n_devices));
        assert((_n_devices & (_n_devices - 1)) == 0);
        _n_global_qubits = int(log2(_n_devices));
        _n_local_qubits = _n_qubits - _n_global_qubits;
        assert(_n_local_qubits > 0);
        _sv_len_per_device = (1LL << _n_qubits) / _n_devices;
        _mallocStateVector();
    }
    /**
     * @brief Constructor
     * @param[in] state_vector StateVector object
     * @param[in] is_copy Whether to copy the state vector
     */
    StateVector(StateVector *state_vector, bool is_copy = false)
        : StateVector(state_vector->getQubits(), state_vector->getMpiCommunicator(), state_vector->isP2P()) {
        if (is_copy) {
            this->copyStateFrom(state_vector);
        }
    }

    ~StateVector() { _freeStateVector(); }
    /**
     * @brief Allocate memory for the state vector
     * @param[in] n_qubits Number of qubits
     * @return Whether memory allocation is possible
     */
    static bool isAllocatable(int n_qubits) {
        unsigned long long memsize = sizeof(cuFpComplex<FP>) * (1LL << n_qubits);
        return (memsize < _ExecuteManager.getAvailableDeviceMemSize());
    }

    int getQubits() const { return _n_qubits; }                       ///< Get the number of qubits
    int getLocalQubits() const { return _n_local_qubits; }            ///< Get the number of local qubits
    int getGlobalQubits() const { return _n_global_qubits; }          ///< Get the number of global qubits
    long long getLenPerDevice() const { return _sv_len_per_device; }  ///< Get the length of the state vector per device
    int getLocalRank() const { return _mpi_local_rank; }              ///< Get the local rank
    int getDevices() const { return _n_devices; }                     ///< Get the number of devices
    MPI_Comm getMpiCommunicator() const { return _mpi_comm; }         ///< Get the MPI communicator
    bool isP2P() const { return _is_p2p; }                            ///< Whether P2P communication is enabled

    void *getDevicePtr() const {
        return _d_sv_ptr;
    };  // Get the device pointer
    cuFpComplex<FP> *getHostPtr();

    void initState(const long long state_bit = 0);
    void zeroState();
    void copyStateFrom(StateVector<FP> *state_vector);

    cuFpComplex<FP> dotProduct(StateVector<FP> *state_vector);
    void add(StateVector<FP> *state_vector, std::complex<double> coef = {1.0, 0.0});
    double getL2Norm(StateVector<FP> *state_vector);
    bool isEqual(StateVector<FP> *state_vector, double eps = 1e-9);
    void printSv();
    void ApplyPauliRotation(PauliProduct &pauli_product, double theta, std::vector<int> &controls,
                            std::vector<int> &targets);
    void swapLocalQubits(std::vector<int2> &swap_qubits);
    void swapGlobalQubits(std::vector<int2> &swap_qubits);
    void swapGlobalQubits();
};
