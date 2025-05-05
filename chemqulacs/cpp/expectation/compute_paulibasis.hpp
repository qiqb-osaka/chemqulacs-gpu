/**
 * @file compute_paulibasis.hpp
 * @brief Header file for ComputePauliBasis class
 * @author Yusuke Teranishi
 */
#pragma once

#include <expectation/expectation_element.hpp>
#include <pauli/pauli_string.hpp>
#include <state/state_vector.hpp>
#include <utils/execute_manager.hpp>
#include <utils/precision.hpp>

#include <custatevec.h>

#include <cassert>
#include <vector>

/**
 * @brief ComputePauliBasis class
 */
template <typename FP>
class ComputePauliBasis : public ExpectationElement<FP> {
   private:
    int _n_operators;                                              ///< Number of Pauli operators
    std::vector<std::vector<custatevecPauli_t>> _pauli_operators;  ///< Pauli operators
    std::vector<custatevecPauli_t *> _pauli_operators_ptrs;        ///< Pointers to Pauli operators
    std::vector<std::vector<int>> _basis_qubits;                   ///< Basis qubits
    std::vector<int *> _basis_qubits_ptrs;                         ///< Pointers to basis qubits
    std::vector<unsigned int> _n_basis_qubits;                     ///< Number of basis qubits
    std::vector<double> _pauli_coefs;                              ///< Coefficients
    std::vector<double> _exp_terms;                                ///< Expectation values

   public:
    /**
     * @brief Constructor
     * @param[in] pauli_products PauliProduct objects
     */
    ComputePauliBasis(const std::vector<PauliProduct> &pauli_products)
        : ExpectationElement<FP>(ExpectationElementType::PAULI_BASIS) {
        _n_operators = (int)pauli_products.size();
        _pauli_operators.resize(_n_operators);
        _basis_qubits.resize(_n_operators);
        _pauli_operators_ptrs.resize(_n_operators);
        _basis_qubits_ptrs.resize(_n_operators);
        _n_basis_qubits.resize(_n_operators);
        _pauli_coefs.resize(_n_operators);
        _exp_terms.resize(_n_operators);
        for (int i = 0; i < _n_operators; ++i) {
            _pauli_operators[i] = pauli_products[i].getPauliOperator();
            _basis_qubits[i] = pauli_products[i].getBasisQubit();
            _pauli_coefs[i] = pauli_products[i].getPauliCoefReal();
            _pauli_operators_ptrs[i] = _pauli_operators[i].data();
            _basis_qubits_ptrs[i] = _basis_qubits[i].data();
            _n_basis_qubits[i] = _pauli_operators[i].size();
        }
    }

    /**
     * @brief Constructor
     * @param[in] pauli_product PauliProduct object
     */
    ComputePauliBasis(const PauliProduct &pauli_product)
        : ComputePauliBasis(std::vector<PauliProduct>(1, pauli_product)) {}

    /**
     * @brief Constructor
     * @param[in] pauli_string PauliString object
     */
    ComputePauliBasis(const PauliString *pauli_string) : ComputePauliBasis(pauli_string->getPauliProducts()) {}

    /**
     * @brief Compute the expectation value
     * @param[in] state_vector StateVector object
     */
    double computeExpectation(StateVector<FP> *state_vector) {
        HANDLE_CUSV(custatevecComputeExpectationsOnPauliBasis(
            _ExecuteManager.getCusvHandle(), state_vector->getDevicePtr(), CUDA_C_<FP>, state_vector->getLocalQubits(),
            _exp_terms.data(), const_cast<const custatevecPauli_t **>(_pauli_operators_ptrs.data()), _n_operators,
            const_cast<const int **>(_basis_qubits_ptrs.data()), _n_basis_qubits.data()));
        double exp = 0;
        for (int i = 0; i < _n_operators; ++i) {
            exp += _pauli_coefs[i] * _exp_terms[i];
        }
        return exp;
    }
};
