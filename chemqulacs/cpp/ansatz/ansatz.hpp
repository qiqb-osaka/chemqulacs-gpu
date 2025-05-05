/**
 * @file ansatz.hpp
 * @brief Header of the Ansatz class
 * @author Yusuke Teranishi
 */
#pragma once

#include <circuit/circuit_element.hpp>
#include <expectation/expectation_computer.hpp>
#include <pauli/pauli_string.hpp>
#include <state/state_vector.hpp>
#include <utils/execute_manager.hpp>
#include <utils/simulation_config.hpp>
#include <utils/timer.hpp>

#include <custatevec.h>
#include <mpi.h>
#include <nvtx3/nvToolsExt.h>
#include <omp.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
/**
 * @brief Class for the ansatz
 * @tparam FP floating point type
 */
template <typename FP>
class Ansatz {
   protected:
    int _n_qubits;         ///< Number of qubits
    int _n_local_qubits;   ///< Number of local qubits
    int _n_global_qubits;  ///< Number of global qubits

    MPI_Comm _mpi_comm;   ///< Type of MPI communication
    int _mpi_local_rank;  ///< MPI local rank
    int _n_devices;       ///< Number of devices

    std::unique_ptr<StateVector<FP>> _state_vector;  ///< State vector

    std::vector<double> _params;                                       ///< Parameters
    std::vector<std::vector<CircuitElement<FP> *>> _param_to_circuit;  ///< Parameter mapping to circuit elements
    std::vector<CircuitElement<FP> *> _circuit_elements;               ///< Execution order of circuit elements

    std::shared_ptr<PauliString> _pauli_string;                      ///< Pauli string
    std::unique_ptr<ExpectationComputer<FP>> _expectation_computer;  ///< Expectation value computer

    Timer *_timer_init;    ///< Timer for initialization
    Timer *_timer_param;   ///< Timer for parameter updates
    Timer *_timer_update;  ///< Timer for circuit updates
    Timer *_timer_exp;     ///< Timer for expectation computation

    virtual void buildCircuit() = 0;

    void buildUpdateQR(std::vector<int> &wire, UpdateQRalgorithm update_qr_algo);

    void buildExpectationQR(std::vector<int> &wire, ExpectationQRalgorithm exp_qr_algo);

    void buildQR(UpdateQRalgorithm update_qr_algo, ExpectationQRalgorithm exp_qr_algo);

    void buildQR();

   public:
    /**
     * @brief Constructor for Ansatz class
     * @param n_qubits Number of qubits
     * @param pauli_string Pauli string
     * @param mpi_comm Type of MPI communication
     * @param is_init Flag for initialization
     */
    Ansatz(int n_qubits, std::shared_ptr<PauliString> pauli_string, MPI_Comm mpi_comm, bool is_init = true) {
        _n_qubits = n_qubits;
        _pauli_string = pauli_string;
        _mpi_comm = mpi_comm;
        HANDLE_MPI(MPI_Comm_rank(_mpi_comm, &_mpi_local_rank));
        HANDLE_MPI(MPI_Comm_size(_mpi_comm, &_n_devices));
        if (is_init) {
            _state_vector = std::make_unique<StateVector<FP>>(_n_qubits, _mpi_comm);
            _n_local_qubits = _state_vector->getLocalQubits();
            _n_global_qubits = _state_vector->getGlobalQubits();
        }
        _timer_init = _TimerDict.addTimerDevice("init");
        _timer_param = _TimerDict.addTimerHost("param");
        _timer_update = _TimerDict.addTimerDevice("update");
        _timer_exp = _TimerDict.addTimerDevice("exp");
    }

    virtual ~Ansatz() {
        for (CircuitElement<FP> *ptr : _circuit_elements) {
            delete ptr;
        }
    }

    // Getter methods
    int getQubits() { return _n_qubits; }                                             ///< Get number of qubits
    int getLocalQubits() const { return _n_local_qubits; }                            ///< Get number of local qubits
    int getGlobalQubits() const { return _n_global_qubits; }                          ///< Get number of global qubits
    std::shared_ptr<PauliString> getPauliStringSharedPtr() { return _pauli_string; }  ///< Get Pauli string
    int getCircuitElementsSize() { return _circuit_elements.size(); }  ///< Get number of circuit elements
    cuFpComplex<FP> *getStateVectorHostPtr() {
        return _state_vector->getHostPtr();
    };  ///< Get host pointer of state vector
    StateVector<FP> *getStateVector() { return _state_vector.get(); }  ///< Get state vector
    int getLocalRank() { return _mpi_local_rank; }                     ///< Get local rank
    // Parametric methods
    std::vector<double> getParams();
    int getParamSize();
    void updateParams(const std::vector<double> &params);
    void updateParam(const int param_id, const double param);
    void updateCircuit();

    // State methods
    void initState(const long long state_bit = 0) {
        _timer_init->restart();
        _state_vector->initState(state_bit);
        _timer_init->stop();
    }  ///< Initialize the state vector
    void loadState(StateVector<FP> *state_vector) {
        _state_vector->copyStateFrom(state_vector);
    }  ///< Load the current state vector
    void storeState(StateVector<FP> *state_vector) {
        state_vector->copyStateFrom(_state_vector.get());
    }  ///< Store the saved state vector
    void updateStateRange(int start_param, int end_param);
    void updateState();
    void updateState2(bool redece_ancila_QR = false);
    void updateStateMulti(bool redece_ancila_QR = false);

    // Expectation methods
    double computeExpectation();
    double computeExpectationWithUpdate(const std::vector<double> &params, const long long init_state);
    double computeExpectationWithUpdate(const int param_id, const double param, const long long init_state);
};
