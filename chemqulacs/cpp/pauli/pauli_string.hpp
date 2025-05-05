/**
 * @file pauli_string.hpp
 * @brief Header file for PauliString class
 * @authors Yusuke Teranishi, Shoma Hiraoka
 */
#pragma once

#include <pauli/pauli_product.hpp>
#include <state/state_vector.hpp>

#include <custatevec.h>
#include <mpi.h>

#include <cassert>
#include <complex>
#include <memory>
#include <unordered_map>
#include <vector>

/**
 * @brief PauliString class
 */
class PauliString {
   private:
    int _n_qubits;
    std::vector<PauliProduct> _pauli_products;

    void _recursiveCombinationBitsRange(int n, int m, std::vector<long long> &comb_bits, int &comb_idx, long long bits,
                                        int depth, int i_n, long long start, long long size, long long range_start,
                                        const std::vector<std::vector<long long>> &combination);
    std::vector<long long> _getDistributedCombinationBits(int n, int m, int mpi_rank, int mpi_size);

    void _recursiveCombinationBits(int n, int m, std::vector<long long> &comb_bits, int &comb_idx, long long bits,
                                   int depth, int i_n);
    std::vector<long long> _getCombinationBits(int n_bits, int one_bits);
    void _setIDs() {
        for (int i = 0; i < getOperatorsNum(); i++) {
            _pauli_products[i].setID(i);
        }
    }

   public:
    /**
     * @brief Constructor
     * @param n_qubits Number of qubits
     * @param pauli_operators List of Pauli operators
     * @param basis_qubits List of basis qubits
     * @param pauli_coefs List of coefficients
     */
    PauliString(const int n_qubits, const std::vector<std::vector<int>> &pauli_operators,
                const std::vector<std::vector<int>> &basis_qubits,
                const std::vector<std::complex<double>> &pauli_coefs) {
        int orig_size = (int)pauli_operators.size();
        constexpr double EPS = 1e-10;
        assert((int)basis_qubits.size() == orig_size);
        assert((int)pauli_coefs.size() == orig_size);
        _n_qubits = n_qubits;

        int n_operators = 0;
        for (int i = 0; i < orig_size; ++i) {
            if (abs(pauli_coefs[i]) < EPS) continue;
            n_operators++;
        }
        _pauli_products.resize(n_operators);
        for (int i = 0, idx = 0; i < orig_size; ++i) {
            if (abs(pauli_coefs[i]) < EPS) continue;
            PauliProduct pauli_product(pauli_operators[i], basis_qubits[i], pauli_coefs[i]);
            _pauli_products[idx++] = pauli_product;
        }
        _setIDs();
        assert((int)_pauli_products.size() == n_operators);
    }

    PauliString() { _n_qubits = 0; };
    /**
     * @brief Add PauliProduct
     * @param pauli_product PauliProduct to be added
     */
    void addPauliProduct(const PauliProduct &pauli_product) { _pauli_products.push_back(pauli_product); }
    int getOperatorsNum() { return (int)_pauli_products.size(); }                   ///< Get the number of PauliProducts
    std::vector<PauliProduct> getPauliProducts() const { return _pauli_products; }  ///< Get the list of PauliProducts
    PauliProduct getPauliProduct(int idx) const { return _pauli_products[idx]; }    ///< Get PauliProduct @ idx
    void clear() { _pauli_products.clear(); }                                       ///< Clear the list of PauliProducts
    std::unordered_map<long long, PauliString> timeSpaceTiling(int n_global_qubits, MPI_Comm mpi_comm);
    std::unordered_map<long long, PauliString> timeSpaceTilingWithControls(int n_global_qubits,
                                                                           std::vector<int> &controls,
                                                                           MPI_Comm mpi_comm);
};
