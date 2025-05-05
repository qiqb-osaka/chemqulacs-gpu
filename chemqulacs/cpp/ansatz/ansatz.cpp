/**
 * @file ansatz.cpp
 * @brief Implementation of the Ansatz class
 * @author Yusuke Teranishi
 */
#include <ansatz/ansatz.hpp>
#include <ansatz/library/gate.hpp>
#include <circuit/optimize_update_qr.hpp>
#include <expectation/expectation_computer.hpp>
#include <utils/simulation_config.hpp>
#include <utils/timer.hpp>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <custatevec.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <complex>
#include <iostream>
#include <numeric>
#include <stdexcept>

template class Ansatz<double>;
template class Ansatz<float>;

/**
 * @brief Build Qubit Reordering (QR) for the update
 * @param wire Wires for the QR
 * @param update_qr_algo Algorithm for the update QR(UNORDER, TILING,
 * TILING_INTERCONNECT)
 */
template <typename FP>
void Ansatz<FP>::buildUpdateQR(std::vector<int> &wire, UpdateQRalgorithm update_qr_algo) {
    bool is_takeover_wire = true;
    OptimizeUpdateQR<FP> optimizer(_n_local_qubits, _n_global_qubits, is_takeover_wire, update_qr_algo);
    optimizer.optimize(_circuit_elements, wire);
}

/**
 * @brief Build Qubit Reordering (QR) for the expectation
 * @param wire Wires for the QR
 * @param exp_qr_algo Algorithm for the expectation QR(UNORDER, OFFLINE,
 * DIAGONAL, DIAGONAL_INTERCONNECT, ALLDOT)
 */
template <typename FP>
void Ansatz<FP>::buildExpectationQR(std::vector<int> &wire, ExpectationQRalgorithm exp_qr_algo) {
    _expectation_computer = std::make_unique<ExpectationComputer<FP>>(_state_vector.get(), _pauli_string.get());
    _expectation_computer->build(wire, exp_qr_algo);
}

/**
 * @brief Build Qubit Reordering (QR) for the update and expectation
 * @param update_qr_algo Algorithm for the update QR(UNORDER, TILING,
 * TILING_INTERCONNECT)
 * @param exp_qr_algo Algorithm for the expectation QR(UNORDER, OFFLINE,
 * DIAGONAL, DIAGONAL_INTERCONNECT, ALLDOT)
 */
template <typename FP>
void Ansatz<FP>::buildQR(UpdateQRalgorithm update_qr_algo, ExpectationQRalgorithm exp_qr_algo) {
    std::vector<int> wire(_n_qubits);
    std::iota(wire.begin(), wire.end(), 0);
    buildUpdateQR(wire, update_qr_algo);
    buildExpectationQR(wire, exp_qr_algo);
}

template <typename FP>
void Ansatz<FP>::buildQR() {
    buildQR(_SimulationConfig.getUpdateQRalgorithm(), _SimulationConfig.getExpectationQRalgorithm());
}

/**
 * @brief Get parameters
 * @return parameters
 */
template <typename FP>
std::vector<double> Ansatz<FP>::getParams() {
    return _params;
}

/**
 * @brief Get the number of parameters
 * @return the number of parameters
 */
template <typename FP>
int Ansatz<FP>::getParamSize() {
    return _params.size();
}

/**
 * @brief Update all parameters
 * @param params New parameters
 */
template <typename FP>
void Ansatz<FP>::updateParams(const std::vector<double> &params) {
    assert(params.size() == _params.size());
    _timer_param->restart();
    std::copy(params.begin(), params.end(), _params.begin());
    updateCircuit();
    _timer_param->stop();
}

/**
 * @brief Update a parameter
 * @param param_id Index of the parameter
 * @param param New parameter
 */
template <typename FP>
void Ansatz<FP>::updateParam(const int param_id, const double param) {
    assert(0 <= param_id && param_id < (int)_param_to_circuit.size());
    _timer_param->restart();
    _params[param_id] = param;
    for (CircuitElement<FP> *circuit : _param_to_circuit[param_id]) {
        circuit->updateCircuit();
    }
    _timer_param->stop();
}

/**
 * @brief change parameters of parametric gates
 */
template <typename FP>
void Ansatz<FP>::updateCircuit() {
    for (CircuitElement<FP> *circuit_element : _circuit_elements) {
        circuit_element->updateCircuit();
    }
}

/**
 * @brief apply gates from start_param to end_param
 * @param start_param start index of the parameter
 * @param end_param end index of the parameter
 */
template <typename FP>
void Ansatz<FP>::updateStateRange(int start_param, int end_param) {
    assert(_n_devices == 1);
    double skip_param_threshold = _SimulationConfig.getSkipParamThreshold();
    for (int i_param = start_param; i_param < end_param; ++i_param) {
        if (fabs(_params[i_param]) < skip_param_threshold) {
            continue;
        }
        for (CircuitElement<FP> *circuit : _param_to_circuit[i_param]) {
            circuit->updateState(_state_vector.get());
        }
    }
    // HANDLE_CUDA(cudaStreamSynchronize(_ExecuteManager.getCudaStream()));
}

/**
 * @brief apply all gates (multi)
 */
template <typename FP>
void Ansatz<FP>::updateStateMulti(bool redece_ancila_QR) {
    for (CircuitElement<FP> *circuit_element : _circuit_elements) {
        const auto type = circuit_element->getType();
        if (redece_ancila_QR and type == CircuitElementType::PAULI_ROTATION and
            this->_mpi_local_rank < this->_n_devices / 2)
            continue;
        circuit_element->updateState(_state_vector.get());
    }
}

/**
 * @brief apply all gates
 */
template <typename FP>
void Ansatz<FP>::updateState() {
    _timer_update->restart();
    _timer_update_compute.restart();
    _timer_update_communicate.reset();
    if (_n_devices == 1) {
        updateStateRange(0, _params.size());
    } else {
        updateStateMulti();
    }
    _timer_update_compute.stop();
    _timer_update->stop();
}

/**
 * @brief apply all gates
 */
template <typename FP>
void Ansatz<FP>::updateState2(bool redece_ancila_QR) {
    _timer_update->restart();
    _timer_update_compute.restart();
    _timer_update_communicate.reset();
    updateStateMulti(redece_ancila_QR);
    _timer_update_compute.stop();
    _timer_update->stop();
}

/**
 * @brief Compute the expectation value
 * @return expectation value
 */
template <typename FP>
double Ansatz<FP>::computeExpectation() {
    assert(static_cast<bool>(_expectation_computer));
    _timer_exp->restart();
    double e = _expectation_computer->computeExpectation();
    _timer_exp->stop();

    // delay evaluation
    _TimerDict.registerTime("update_compute", _timer_update_compute.getTime());
    _TimerDict.registerTime("update_communicate", _timer_update_communicate.getTime());

    return e;
}

/**
 * @brief Compute the expectation value with the updated parameters
 * @param params New parameters
 * @param init_state Initial state
 * @return expectation value
 */
template <typename FP>
double Ansatz<FP>::computeExpectationWithUpdate(const std::vector<double> &params, const long long init_state) {
    initState(init_state);
    updateParams(params);
    updateState();
    double exp = computeExpectation();
    return exp;
}

/**
 * @brief Compute the expectation value with the updated parameter
 * @param param_id Index of the parameter
 * @param param New parameter
 * @param init_state Initial state
 * @return expectation value
 */
template <typename FP>
double Ansatz<FP>::computeExpectationWithUpdate(const int param_id, const double param, const long long init_state) {
    initState(init_state);
    updateParam(param_id, param);
    updateState();
    double exp = computeExpectation();
    return exp;
}
