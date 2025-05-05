/**
 * @file pauli_product.hpp
 * @brief Header file for PauliProduct class
 * @author Shoma Hiraoka
 */
#pragma once

#include <custatevec.h>
#include <mpi.h>

#include <cassert>
#include <complex>
#include <vector>

/**
 * @brief PauliProduct class
 * @details Main component of PauliString
 */
class PauliProduct {
   private:
    std::vector<custatevecPauli_t> _pauli_operator;
    std::vector<int> _basis_qubit;
    std::complex<double> _pauli_coef;
    double _pauli_coef_real;
    int _pauli_index;  ///< The position of itself within the PauliString
    int _id;           ///< ID

   public:
    /**
     * @brief Constructor
     * @param pauli_operator Pauli operator(int)(1:X, 2:Y, 3:Z)
     * @param basis_qubit Basis qubit(0, 1, 2, ...)
     * @param pauli_coef Coefficient
     */
    PauliProduct(const std::vector<int> &pauli_operator, const std::vector<int> &basis_qubit,
                 const std::complex<double> &pauli_coef) {
        int orig_size = pauli_operator.size();
        assert((int)basis_qubit.size() == orig_size);
        for (int i = 0; i < orig_size; ++i) {
            assert(pauli_operator[i] >= 0 && pauli_operator[i] <= 3);
            assert(basis_qubit[i] >= 0);
            _pauli_operator.push_back((custatevecPauli_t)pauli_operator[i]);
        }
        _basis_qubit = basis_qubit;
        _pauli_coef = pauli_coef;
        _pauli_coef_real = _pauli_coef.real();
    }
    /**
     * @brief Constructor
     * @param pauli_operator Pauli operator(custatevecPauli_t)(X, Y, Z)
     * @param basis_qubit Basis qubit(0, 1, 2, ...)
     * @param pauli_coef Coefficient
     */
    PauliProduct(const std::vector<custatevecPauli_t> &pauli_operator, const std::vector<int> &basis_qubit,
                 const std::complex<double> &pauli_coef) {
        int orig_size = pauli_operator.size();
        assert((int)basis_qubit.size() == orig_size);
        for (int i = 0; i < orig_size; ++i) {
            assert(basis_qubit[i] >= 0);
        }
        _pauli_operator = pauli_operator;
        _basis_qubit = basis_qubit;
        _pauli_coef = pauli_coef;
        _pauli_coef_real = _pauli_coef.real();
    }
    PauliProduct() {}
    std::vector<custatevecPauli_t> getPauliOperator() const {
        return _pauli_operator;
    }  ///< Get Pauli operator(custatevecPauli_t)
    std::vector<int> getBasisQubit() const { return _basis_qubit; }    ///< Get basis qubit
    std::complex<double> getPauliCoef() const { return _pauli_coef; }  ///< Get coefficient
    double getPauliCoefReal() const { return _pauli_coef_real; }       ///< Get real part of coefficient
    int getPauliIndex() const { return _pauli_index; }                 ///< Get Pauli index
    int getID() const { return _id; }                                  ///< Get ID
    void setPauliOperator(const std::vector<int> &pauli_operator) {
        int orig_size = pauli_operator.size();
        assert((int)_basis_qubit.size() == orig_size);
        _pauli_operator.clear();
        for (int i = 0; i < orig_size; ++i) {
            assert(pauli_operator[i] >= 0 && pauli_operator[i] <= 3);
            _pauli_operator.push_back((custatevecPauli_t)pauli_operator[i]);
        }
    }  ///< Set Pauli operator(int)
    void setBasisQubit(const std::vector<int> &basis_qubit) {
        int orig_size = basis_qubit.size();
        assert((int)_pauli_operator.size() == orig_size);
        _basis_qubit = basis_qubit;
    }  ///< Set basis qubit
    void setPauliCoef(const std::complex<double> &pauli_coef) {
        _pauli_coef = pauli_coef;
        _pauli_coef_real = _pauli_coef.real();
    }  ///< Set coefficient
    void setPauliCoef(const double &pauli_coef_real) {
        _pauli_coef_real = pauli_coef_real;
        _pauli_coef = std::complex<double>(_pauli_coef_real, 0);
    }  ///< Set real part of coefficient
    void setPauliIndex(const int &pauli_index) { _pauli_index = pauli_index; }  ///< Set Pauli index
    void setID(const int &id) { _id = id; }                                     ///< Set ID
    void addElement(const custatevecPauli_t &pauli_operator, const int &basis_qubit) {
        _pauli_operator.push_back(pauli_operator);
        _basis_qubit.push_back(basis_qubit);
    }  ///< Add one pauli operator and basis qubit to the end of the list
};