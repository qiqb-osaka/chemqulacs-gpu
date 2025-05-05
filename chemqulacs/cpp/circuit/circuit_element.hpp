/**
 * @file circuit_element.hpp
 * @brief Header file for CircuitElement class
 * @author Yusuke Teranishi
 */
#pragma once

#include <state/state_vector.hpp>
#include <utils/precision.hpp>

#include <custatevec.h>

#include <vector>

/**
 * @enum CircuitElementType
 * @brief Enum class for circuit element type
 * @details
 * - APPLY_MATRIX : Gate as a matrix
 * - PAULI_ROTATION : Gate as a \f$ e^{iP\theta} \f$
 * - SWAP_GLOBAL_QUBITS : Swap gate for G2G or G2L
 * - SWAP_LOCAL_QUBITS : Swap gate for L2L
 */
enum class CircuitElementType {
    APPLY_MATRIX,
    PAULI_ROTATION,
    SWAP_GLOBAL_QUBITS,
    SWAP_LOCAL_QUBITS
};

/**
 * @brief CircuitElement class
 */
template <typename FP>
class CircuitElement {
   private:
    CircuitElementType _type;

   protected:
    std::vector<int> _controls;
    std::vector<int> _targets;

   public:
    CircuitElement(CircuitElementType type) : _type(type) {}
    CircuitElement(CircuitElementType type, std::vector<int> &controls, std::vector<int> &targets)
        : _type(type), _controls(controls), _targets(targets) {}
    virtual ~CircuitElement() {}
    virtual CircuitElement *clone() const = 0;
    CircuitElementType getType() { return _type; }        ///< Get the type of the circuit element
    std::vector<int> getControls() { return _controls; }  ///< Get the control qubits
    std::vector<int> getTargets() { return _targets; }    ///< Get the target qubits
    std::vector<int> getQubits() {
        std::vector<int> tmp = _controls;  // non sort
        tmp.insert(tmp.end(), _targets.begin(), _targets.end());
        return tmp;
    }                                                                       ///< Get the control + target qubits
    void setControls(std::vector<int> &controls) { _controls = controls; }  ///< Set the control qubits
    void setTargets(std::vector<int> &targets) { _targets = targets; }      ///< Set the target qubits

    virtual void updateCircuit() = 0;
    virtual void updateState(StateVector<FP> *state_vector) = 0;
    virtual void print() = 0;
};
