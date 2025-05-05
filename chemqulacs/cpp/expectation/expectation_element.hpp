/**
 * @file expectation_element.hpp
 * @brief Header file for ExpectationElement class
 * @author Yusuke Teranishi
 */
#pragma once

#include <state/state_vector.hpp>

/**
 * @brief ExpectationElementType enum class
 * - PAULI_BASIS: Compute expectation value in the Pauli basis
 * - DOT_PRODUCT: Compute expectation value in the dot
 * - QUBIT_REORDERING: Swap qubits
 */
enum class ExpectationElementType {
    PAULI_BASIS,
    DOT_PRODUCT,
    QUBIT_REORDERING
};

template <typename FP>
class ExpectationElement {
   private:
    ExpectationElementType _type;

   public:
    ExpectationElement(ExpectationElementType type) : _type(type) {}

    virtual ~ExpectationElement() {}

    virtual double computeExpectation(StateVector<FP> *state_vector) = 0;

    ExpectationElementType getType() { return _type; }
};
