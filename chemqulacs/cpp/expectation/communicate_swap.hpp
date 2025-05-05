/**
 * @file communicate_swap.hpp
 * @brief Header file for CommunicateSwap class
 * @author Yusuke Teranishi
 */
#pragma once

#include <expectation/expectation_element.hpp>
#include <utils/precision.hpp>
#include <utils/timer.hpp>

#include <custatevec.h>

#include <vector>

/**
 * @brief CommunicateSwap class
 */
template <typename FP>
class CommunicateSwap : public ExpectationElement<FP> {
   private:
    std::vector<int2> _swap_qubits;

   public:
    /**
     * @brief Constructor
     * @param swap_qubits List of swap qubits(one is global and the other is local)
     */
    CommunicateSwap(const std::vector<int2>& swap_qubits)
        : ExpectationElement<FP>(ExpectationElementType::QUBIT_REORDERING) {
        _swap_qubits = swap_qubits;
    }

    /**
     * @brief Apply swap Qubits using custatevec lib when computing expectation
     * @param state_vector StateVector object
     */
    double computeExpectation(StateVector<FP>* state_vector) {
        _timer_exp_compute.stop();
        _timer_exp_communicate.start();
        state_vector->swapGlobalQubits(_swap_qubits);
        _timer_exp_communicate.stop();
        _timer_exp_compute.start();
        return 0.0;
    }
};
