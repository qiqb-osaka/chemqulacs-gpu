/**
 * @file swap_global_qubits.hpp
 * @brief Header file for SwapGlobalQubits class
 * @author Yusuke Teranishi
 */
#pragma once

#include <circuit/circuit_element.hpp>
#include <utils/execute_manager.hpp>
#include <utils/precision.hpp>
#include <utils/timer.hpp>

#include <custatevec.h>
#include <mpi.h>

#include <iostream>
#include <vector>

/**
 * @brief SwapGlobalQubits class
 * @details Swap gate for G2G or G2L(Sub class of CircuitElement)
 */
template <typename FP>
class SwapGlobalQubits : public CircuitElement<FP> {
   private:
    std::vector<int2> _swap_qubits;

   public:
    /**
     * @brief Constructor
     * @param swap_qubits List of swap qubits(At least one of two is global)
     */
    SwapGlobalQubits(const std::vector<int2>& swap_qubits)
        : CircuitElement<FP>(CircuitElementType::SWAP_GLOBAL_QUBITS) {
        _swap_qubits = swap_qubits;
    }

    void updateCircuit() {}

    /**
     * @brief Apply swap Qubits using custatevec lib when build circuit
     * @param state_vector StateVector object
     */
    void updateState(StateVector<FP>* state_vector) {
        _timer_update_compute.stop();
        _timer_update_communicate.start();
        state_vector->swapGlobalQubits(_swap_qubits);
        _timer_update_communicate.stop();
        _timer_update_compute.start();
    }

    void print() {
        std::cout << "SwapGlobalQubits" << std::endl;
        for (auto qubit : _swap_qubits) {
            std::cout << qubit.x << " " << qubit.y << std::endl;
        }
    }
    virtual CircuitElement<FP>* clone() const { return new SwapGlobalQubits<FP>(*this); }
};
