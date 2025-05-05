/**
 * @file pauli_rotation.hpp
 * @brief Header file for PauliRotation class
 * @author Yusuke Teranishi
 */
#pragma once

#include <circuit/circuit_element.hpp>
#include <pauli/pauli_product.hpp>
#include <state/state_vector.hpp>
#include <utils/execute_manager.hpp>
#include <utils/precision.hpp>

#include <custatevec.h>
#include <nvtx3/nvToolsExt.h>

#include <iostream>
#include <vector>

/**
 * @brief PauliRotation class
 * @details Add Gate as a \f$ e^{iP\theta} \f$( Sub class of CircuitElement)
 */
template <typename FP>
class PauliRotation : public CircuitElement<FP> {
   private:
    PauliProduct _pauli_product;
    std::vector<custatevecPauli_t> _paulis;
    double *_theta;
    double _coef;

   public:
    PauliRotation(std::vector<custatevecPauli_t> paulis, double *theta, double coef, std::vector<int> controls,
                  std::vector<int> targets)
        : CircuitElement<FP>(CircuitElementType::PAULI_ROTATION, controls, targets) {
        _pauli_product = PauliProduct(paulis, targets, coef);
        _paulis = paulis;
        _theta = theta;
        _coef = coef;
    }

    PauliRotation(std::vector<custatevecPauli_t> paulis, double *theta, double coef, std::vector<int> targets)
        : PauliRotation(paulis, theta, coef, std::vector<int>(), targets) {}

    PauliRotation(custatevecPauli_t pauli, double *theta, double coef, std::vector<int> targets)
        : PauliRotation(std::vector<custatevecPauli_t>(targets.size(), pauli), theta, coef, targets) {}
    PauliRotation(PauliProduct pauli_product, double *theta, std::vector<int> controls = std::vector<int>())
        : PauliRotation(pauli_product.getPauliOperator(), theta, pauli_product.getPauliCoefReal(), controls,
                        pauli_product.getBasisQubit()) {}
    PauliRotation(PauliProduct pauli_product, double *theta, double coef,
                  std::vector<int> controls = std::vector<int>())
        : PauliRotation(pauli_product.getPauliOperator(), theta, coef, controls, pauli_product.getBasisQubit()) {}
    void updateCircuit() {}

    void updateState(StateVector<FP> *state_vector) {
        nvtxRangePush("updateStateUnit");
        double theta = *_theta * _coef;
        state_vector->ApplyPauliRotation(_pauli_product, theta, this->_controls, this->_targets);
    }

    void print() {
        std::cout << "PauliRotation" << std::endl;
        std::cout << "Controls: ";
        for (auto control : this->_controls) std::cout << control << " ";
        std::cout << std::endl;
        std::cout << "Targets: ";
        for (auto target : this->_targets) std::cout << target << " ";
        std::cout << std::endl;
        double theta = *_theta * _coef;
        std::cout << "theta: " << theta << std::endl;
        std::cout << "coeff: " << _pauli_product.getPauliCoef().real() << " " << _pauli_product.getPauliCoef().imag()
                  << std::endl;
        int len = (int)_pauli_product.getBasisQubit().size();
        std::cout << "len: " << len << std::endl;
        for (int i = 0; i < len; i++) {
            char op = 'I';
            if (_pauli_product.getPauliOperator()[i] == 1)
                op = 'X';
            else if (_pauli_product.getPauliOperator()[i] == 2)
                op = 'Y';
            else if (_pauli_product.getPauliOperator()[i] == 3)
                op = 'Z';
            std::cout << op << _pauli_product.getBasisQubit()[i] << " ";
        }
        std::cout << std::endl;
    }
    virtual CircuitElement<FP> *clone() const {
        return new PauliRotation<FP>(_paulis, _theta, _coef, this->_controls, this->_targets);
    }
};
