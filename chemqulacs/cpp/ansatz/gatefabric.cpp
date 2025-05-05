/**
 * @file gatefabric.cpp
 * @brief Implementation of the GateFabric class
 */
#include <ansatz/gatefabric.hpp>
#include <ansatz/library/double_excitation.hpp>
#include <ansatz/library/single_excitation.hpp>
#include <circuit/apply_matrix.hpp>
#include <circuit/swap_global_qubits.hpp>
#include <utils/precision.hpp>

#include <cassert>
#include <cmath>
#include <vector>

template class GateFabric<double>;
template class GateFabric<float>;

/**
 * @brief Build the circuit for the ansatz
 * @details push all the possible consecutive 4-qubits Gate to CircuitElement
 */
template <typename FP>
void GateFabric<FP>::buildCircuit() {
    const int n_wires = 4;
    assert(this->_n_qubits >= n_wires);

    std::vector<std::vector<int>> wire_pattern;
    std::vector<int> wire_tmp(n_wires);
    for (int i = 0; i <= this->_n_qubits - 4; i += 4) {
        for (int j = 0; j < 4; ++j) {
            wire_tmp[j] = i + j;
        }
        wire_pattern.push_back(wire_tmp);
    }
    for (int i = 2; i <= this->_n_qubits - 4; i += 4) {
        for (int j = 0; j < 4; ++j) {
            wire_tmp[j] = i + j;
        }
        wire_pattern.push_back(wire_tmp);
    }

    int n_params = _n_layers * wire_pattern.size() * 2;
    int param_id = 0;
    this->_params.resize(n_params);
    this->_param_to_circuit.resize(n_params);
    for (int i = 0; i < _n_layers; ++i) {
        for (std::vector<int>& wires : wire_pattern) {
            _fourQubitGate(wires, param_id);
        }
    }
}

/**
 * GateFabric<FP>::_fourQubitGate
 * @brief Add a four-qubit gate to the circuit
 * @param wires Wires for the gate
 * @param param_id Index of the parameter
 */
template <typename FP>
void GateFabric<FP>::_fourQubitGate(std::vector<int>& wires, int& param_id) {
    int start_id = this->_circuit_elements.size();
    if (_include_pi) {
        _orbitalRotation(wires, _PI);
    }
    _doubleExcitation(wires, &this->_params[param_id]);
    this->_param_to_circuit[param_id++]
        .assign(this->_circuit_elements.begin() + start_id, this->_circuit_elements.end());

    start_id = this->_circuit_elements.size();
    _orbitalRotation(wires, &this->_params[param_id]);
    this->_param_to_circuit[param_id++]
        .assign(this->_circuit_elements.begin() + start_id, this->_circuit_elements.end());
}

/**
 * GateFabric<FP>::_orbitalRotation
 * @brief Add an
 * OrbitalRotation(https://docs.pennylane.ai/en/stable/code/api/pennylane.OrbitalRotation.html?highlight=orbital#pennylane.OrbitalRotation)
 * to the circuit
 * @param wires Wires for the gate
 * @param phi Parameter for the orbital rotation
 */
template <typename FP>
void GateFabric<FP>::_orbitalRotation(std::vector<int>& wires, double* phi) {
    assert(wires.size() == 4);
    _singleExcitation({wires[0], wires[2]}, phi);
    _singleExcitation({wires[1], wires[3]}, phi);
}

/**
 * GateFabric<FP>::_doubleExcitation
 * @brief Add a
 * DoubleExcitation(https://docs.pennylane.ai/en/stable/code/api/pennylane.DoubleExcitation.html?highlight=doubleexcitation#pennylane.DoubleExcitation)
 * to the circuit
 * @param wires Wires for the gate
 * @param theta Parameter for the double excitation
 */
template <typename FP>
void GateFabric<FP>::_doubleExcitation(std::vector<int> wires, double* theta) {
    this->_circuit_elements.push_back(new ApplyMatrix<FP>(new DoubleExcitation(theta), wires));
}

/**
 * GateFabric<FP>::_singleExcitation
 * @brief Add a Single
 * Excitation(https://docs.pennylane.ai/en/stable/code/api/pennylane.SingleExcitation.html?highlight=singleexcitation#pennylane.SingleExcitation)
 * to the circuit
 * @param wires Wires for the gate
 * @param theta Parameter for the Single Excitation
 */
template <typename FP>
void GateFabric<FP>::_singleExcitation(std::vector<int> wires, double* theta) {
    this->_circuit_elements.push_back(new ApplyMatrix<FP>(new SingleExcitation(theta), wires));
}
