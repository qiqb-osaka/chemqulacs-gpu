/**
 * @file swap_local_qubits.hpp
 * @brief Header file for SwapLocalQubits class
 */
#pragma once

#include <circuit/circuit_element.hpp>
#include <utils/precision.hpp>

#include <custatevec.h>

#include <iostream>
#include <vector>

/**
 * @brief SwapLocalQubits class
 */
template <typename FP>
class SwapLocalQubits : public CircuitElement<FP> {
   private:
    std::vector<int2> _swap_qubits;

   public:
    /**
     * @brief Constructor
     * @param swap_qubits List of swap qubits(Both are local)
     */
    SwapLocalQubits(const std::vector<int2>& swap_qubits) : CircuitElement<FP>(CircuitElementType::SWAP_LOCAL_QUBITS) {
        _swap_qubits = swap_qubits;
    }

    void updateCircuit() {}

    /**
     * @brief Apply swap LocalQubits using custatevec lib when build circuit
     * @param state_vector StateVector object
     */
    void updateState(StateVector<FP>* state_vector) { state_vector->swapLocalQubits(_swap_qubits); }

    void print() {
        std::cout << "SwapLocalQubits" << std::endl;
        for (auto qubit : _swap_qubits) {
            std::cout << qubit.x << " " << qubit.y << std::endl;
        }
    }
    virtual CircuitElement<FP>* clone() const { return new SwapLocalQubits<FP>(*this); }
};
