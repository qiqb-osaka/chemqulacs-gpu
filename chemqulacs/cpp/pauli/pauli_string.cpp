/**
 * @file pauli_string.cpp
 * @brief Source file for PauliString class
 * @authors Yusuke Teranishi, Shoma Hiraoka
 */
#include <pauli/pauli_string.hpp>

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>

/**
 * @brief Split the Paulistring so that it can be processed with as few
 * QRs as possible.
 * @param n_global_qubits Number of global qubits
 * @param mpi_comm Type of MPI communicator
 */
std::unordered_map<long long, PauliString> PauliString::timeSpaceTiling(int n_global_qubits, MPI_Comm mpi_comm) {
    int mpi_rank;
    int mpi_size;
    HANDLE_MPI(MPI_Comm_rank(mpi_comm, &mpi_rank));
    HANDLE_MPI(MPI_Comm_size(mpi_comm, &mpi_size));
    assert((1 << n_global_qubits) == mpi_size);
    std::unordered_map<long long, PauliString> local_cover;
    int n_operators = getOperatorsNum();

    // convert pauli term to bits
    std::vector<long long> global_bits(n_operators);
    std::vector<std::vector<int>> pauli_classify(_n_qubits + 1);
    for (int i = 0; i < n_operators; ++i) {
        const std::vector<custatevecPauli_t> &pauli_operator = _pauli_products[i].getPauliOperator();
        const std::vector<int> &basis_qubit = _pauli_products[i].getBasisQubit();
        int n_basis = (int)pauli_operator.size();
        long long identity_bits = (1LL << _n_qubits) - 1;
        long long pauli_z_bits = 0;
        for (int j = 0; j < n_basis; ++j) {
            if (pauli_operator[j] == CUSTATEVEC_PAULI_Z) {
                pauli_z_bits |= (1LL << basis_qubit[j]);
            }
            identity_bits ^= (1LL << basis_qubit[j]);
        }
        long long bits = identity_bits | pauli_z_bits;
        int n_bits = __builtin_popcountll(bits);
        if (n_bits < n_global_qubits) {
            throw std::invalid_argument("Too large parallelism.");
        }
        global_bits[i] = bits;
        pauli_classify[n_bits].push_back(i);
    }

    // create pauli tree
    std::vector<std::vector<int>> pauli_child(n_operators);
    std::list<int> no_parent;
    for (int n_bits = _n_qubits; n_bits > 0; --n_bits) {
        // before no parent
        for (auto it = no_parent.begin(); it != no_parent.end();) {
            int c_pauli = *it;
            long long c_bits = global_bits[c_pauli];
            bool is_parent = false;
            for (int p_pauli : pauli_classify[n_bits - 1]) {
                long long p_bits = global_bits[p_pauli];
                if ((c_bits & p_bits) == p_bits) {
                    pauli_child[p_pauli].push_back(c_pauli);
                    is_parent = true;
                    break;
                }
            }
            if (is_parent) {
                it = no_parent.erase(it);
            } else {
                ++it;
            }
        }
        // current pauli
        for (int i = 0; i < (int)pauli_classify[n_bits].size(); ++i) {
            int c_pauli = pauli_classify[n_bits][i];
            long long c_bits = global_bits[c_pauli];
            bool is_parent = false;
            // sibling
            for (int j = i + 1; j < (int)pauli_classify[n_bits].size(); ++j) {
                int s_pauli = pauli_classify[n_bits][j];
                long long s_bits = global_bits[s_pauli];
                if (c_bits == s_bits) {
                    pauli_child[s_pauli].push_back(c_pauli);
                    is_parent = true;
                    break;
                }
            }
            if (is_parent) {
                continue;
            }
            // parent
            for (int p_pauli : pauli_classify[n_bits - 1]) {
                long long p_bits = global_bits[p_pauli];
                if ((c_bits & p_bits) == p_bits) {
                    pauli_child[p_pauli].push_back(c_pauli);
                    is_parent = true;
                    break;
                }
            }
            if (!is_parent) {
                no_parent.push_back(c_pauli);
            }
        }
    }

    // merge represent pauli
    std::list<int> represent_pauli;
    represent_pauli.insert(represent_pauli.end(), pauli_classify[0].begin(), pauli_classify[0].end());
    represent_pauli.insert(represent_pauli.end(), no_parent.begin(), no_parent.end());

    // greedy pauli cover
    std::vector<long long> comb_bits_set =
        _getDistributedCombinationBits(_n_qubits, n_global_qubits, mpi_rank, mpi_size);
    std::vector<long long> send_maxbuf(2);
    std::vector<long long> recv_maxbuf(mpi_size * 2);  // odd:max_cnt,even:max_bits per process
    while (!represent_pauli.empty()) {
        long long max_cnt = 0;
        long long max_bits = -1;
        for (long long comb_bits : comb_bits_set) {
            long long cnt = 0;
            for (int i_pauli : represent_pauli) {
                long long bits = global_bits[i_pauli];
                if (__builtin_popcountll(bits & comb_bits) >= n_global_qubits) {
                    cnt++;
                }
            }
            if (max_cnt < cnt) {
                max_cnt = cnt;
                max_bits = comb_bits;
            }
        }
        send_maxbuf[0] = max_cnt;
        send_maxbuf[1] = max_bits;
        HANDLE_MPI(MPI_Allgather(send_maxbuf.data(), 2, MPI_LONG_LONG, recv_maxbuf.data(), 2, MPI_LONG_LONG, mpi_comm));
        max_cnt = 0;
        for (int i = 0; i < mpi_size; ++i) {
            long long cnt = recv_maxbuf[i * 2];
            long long bits = recv_maxbuf[i * 2 + 1];
            if (max_cnt < cnt) {
                max_cnt = cnt;
                max_bits = bits;
            }
        }
        assert(max_cnt > 0);
        assert(max_bits > 0);

        PauliString pauli_string;
        auto getTreePauliString = [&](auto && func, int parent)->void {
            _pauli_products[parent].setPauliIndex(parent);
            pauli_string.addPauliProduct(_pauli_products[parent]);
            for (int child : pauli_child[parent]) {
                func(func, child);
            }
        };
        for (auto it = represent_pauli.begin(); it != represent_pauli.end();) {
            int i_pauli = *it;
            long long bits = global_bits[i_pauli];
            if (__builtin_popcountll(bits & max_bits) >= n_global_qubits) {
                getTreePauliString(getTreePauliString, i_pauli);
                it = represent_pauli.erase(it);
            } else {
                ++it;
            }
        }
        local_cover[max_bits] = pauli_string;
    }
    return local_cover;
}

std::unordered_map<long long, PauliString> PauliString::timeSpaceTilingWithControls(int n_global_qubits,
                                                                                    std::vector<int> &controls,
                                                                                    MPI_Comm mpi_comm) {
    int mpi_rank;
    int mpi_size;
    HANDLE_MPI(MPI_Comm_rank(mpi_comm, &mpi_rank));
    HANDLE_MPI(MPI_Comm_size(mpi_comm, &mpi_size));
    assert((1 << n_global_qubits) == mpi_size);
    std::unordered_map<long long, PauliString> local_cover;
    int n_operators = getOperatorsNum();

    // convert pauli term to bits
    std::vector<long long> global_bits(n_operators);
    std::vector<std::vector<int>> pauli_classify(_n_qubits + 1);
    for (int i = 0; i < n_operators; ++i) {
        const std::vector<custatevecPauli_t> &pauli_operator = _pauli_products[i].getPauliOperator();
        const std::vector<int> &basis_qubit = _pauli_products[i].getBasisQubit();
        int n_basis = (int)pauli_operator.size();
        long long identity_bits = (1LL << _n_qubits) - 1;
        long long pauli_z_bits = 0;
        for (int j = 0; j < n_basis; ++j) {
            // if (pauli_operator[j] == CUSTATEVEC_PAULI_Z) {
            //     pauli_z_bits |= (1LL << basis_qubit[j]);
            // }
            identity_bits ^= (1LL << basis_qubit[j]);
        }
        for (auto j : controls) {
            identity_bits ^= (1LL << j);
        }

        long long bits = identity_bits | pauli_z_bits;
        int n_bits = __builtin_popcountll(bits);
        if (n_bits < n_global_qubits) {
            throw std::invalid_argument("Too large parallelism.");
        }
        global_bits[i] = bits;
        pauli_classify[n_bits].push_back(i);
    }

    // create pauli tree
    std::vector<std::vector<int>> pauli_child(n_operators);
    std::list<int> no_parent;
    for (int n_bits = _n_qubits; n_bits > 0; --n_bits) {
        // before no parent
        for (auto it = no_parent.begin(); it != no_parent.end();) {
            int c_pauli = *it;
            long long c_bits = global_bits[c_pauli];
            bool is_parent = false;
            for (int p_pauli : pauli_classify[n_bits - 1]) {
                long long p_bits = global_bits[p_pauli];
                if ((c_bits & p_bits) == p_bits) {
                    pauli_child[p_pauli].push_back(c_pauli);
                    is_parent = true;
                    break;
                }
            }
            if (is_parent) {
                it = no_parent.erase(it);
            } else {
                ++it;
            }
        }
        // current pauli
        for (int i = 0; i < (int)pauli_classify[n_bits].size(); ++i) {
            int c_pauli = pauli_classify[n_bits][i];
            long long c_bits = global_bits[c_pauli];
            bool is_parent = false;
            // sibling
            for (int j = i + 1; j < (int)pauli_classify[n_bits].size(); ++j) {
                int s_pauli = pauli_classify[n_bits][j];
                long long s_bits = global_bits[s_pauli];
                if (c_bits == s_bits) {
                    pauli_child[s_pauli].push_back(c_pauli);
                    is_parent = true;
                    break;
                }
            }
            if (is_parent) {
                continue;
            }
            // parent
            for (int p_pauli : pauli_classify[n_bits - 1]) {
                long long p_bits = global_bits[p_pauli];
                if ((c_bits & p_bits) == p_bits) {
                    pauli_child[p_pauli].push_back(c_pauli);
                    is_parent = true;
                    break;
                }
            }
            if (!is_parent) {
                no_parent.push_back(c_pauli);
            }
        }
    }

    // merge represent pauli
    std::list<int> represent_pauli;
    represent_pauli.insert(represent_pauli.end(), pauli_classify[0].begin(), pauli_classify[0].end());
    represent_pauli.insert(represent_pauli.end(), no_parent.begin(), no_parent.end());

    // greedy pauli cover
    std::vector<long long> comb_bits_set =
        _getDistributedCombinationBits(_n_qubits, n_global_qubits, mpi_rank, mpi_size);
    std::vector<long long> send_maxbuf(2);
    std::vector<long long> recv_maxbuf(mpi_size * 2);  // odd:max_cnt,even:max_bits per process
    while (!represent_pauli.empty()) {
        long long max_cnt = 0;
        long long max_bits = -1;
        for (long long comb_bits : comb_bits_set) {
            long long cnt = 0;
            for (int i_pauli : represent_pauli) {
                long long bits = global_bits[i_pauli];
                if (__builtin_popcountll(bits & comb_bits) >= n_global_qubits) {
                    cnt++;
                }
            }
            if (max_cnt < cnt) {
                max_cnt = cnt;
                max_bits = comb_bits;
            }
        }
        send_maxbuf[0] = max_cnt;
        send_maxbuf[1] = max_bits;
        HANDLE_MPI(MPI_Allgather(send_maxbuf.data(), 2, MPI_LONG_LONG, recv_maxbuf.data(), 2, MPI_LONG_LONG, mpi_comm));
        max_cnt = 0;
        for (int i = 0; i < mpi_size; ++i) {
            long long cnt = recv_maxbuf[i * 2];
            long long bits = recv_maxbuf[i * 2 + 1];
            if (max_cnt < cnt) {
                max_cnt = cnt;
                max_bits = bits;
            }
        }
        assert(max_cnt > 0);
        assert(max_bits > 0);

        PauliString pauli_string;
        auto getTreePauliString = [&](auto && func, int parent)->void {
            _pauli_products[parent].setPauliIndex(parent);
            pauli_string.addPauliProduct(_pauli_products[parent]);
            for (int child : pauli_child[parent]) {
                func(func, child);
            }
        };
        for (auto it = represent_pauli.begin(); it != represent_pauli.end();) {
            int i_pauli = *it;
            long long bits = global_bits[i_pauli];
            if (__builtin_popcountll(bits & max_bits) >= n_global_qubits) {
                getTreePauliString(getTreePauliString, i_pauli);
                it = represent_pauli.erase(it);
            } else {
                ++it;
            }
        }
        local_cover[max_bits] = pauli_string;
    }
    return local_cover;
}

/**
 * @brief Internal function for _getCombinationBits
 * @param[in] n Number of bits
 * @param[in] m Number of one bits
 * @param[out] comb_bits All pair of n choose m
 * @param[in] comb_work Work array
 * @param[in] comb_idx Index of comb_bits
 * @param[in] depth Depth of recursive
 * @param[in] i_n Start index of recursive
 */
void PauliString::_recursiveCombinationBits(int n, int m, std::vector<long long> &comb_bits, int &comb_idx,
                                            long long bits, int depth, int i_n) {
    if (depth == m) {
        comb_bits[comb_idx++] = bits;
        return;
    }
    for (int i = i_n; i < n; ++i) {
        _recursiveCombinationBits(n, m, comb_bits, comb_idx, bits | (1LL << i), depth + 1, i + 1);
    }
}

/**
 * @brief Calcrate all pair of n_bits choose zero_bits
 * @param[in] n_bits Number of bits
 * @param[in] one_bits Number of one bits
 * @return All pair of n_bits choose zero_bits
 */

std::vector<long long> PauliString::_getCombinationBits(int n_bits, int one_bits) {
    long long n_comb = 1;
    for (int i = 0; i < one_bits; ++i) {
        n_comb *= (n_bits - i);
    }
    for (int i = 0; i < one_bits; ++i) {
        n_comb /= (one_bits - i);
    }
    std::vector<long long> comb_bits(n_comb);
    int comb_idx = 0;
    _recursiveCombinationBits(n_bits, one_bits, comb_bits, comb_idx, 0, 0, 0);
    return comb_bits;
}

/**
 * @brief Internal function for _getDistributedCombinationBits
 */
void PauliString::_recursiveCombinationBitsRange(int n, int m, std::vector<long long> &comb_bits, int &comb_idx,
                                                 long long bits, int depth, int i_n, long long start, long long size,
                                                 long long range_start,
                                                 const std::vector<std::vector<long long>> &combination) {
    long long range_size = combination[n - i_n][m - depth];
    if ((start + size <= range_start) || (range_start + range_size <= start)) {
        return;
    }
    if (depth == m) {
        assert(comb_idx < size);
        comb_bits[comb_idx++] = bits;
        return;
    }
    for (int i = i_n; i < n; ++i) {
        _recursiveCombinationBitsRange(n, m, comb_bits, comb_idx, bits | (1LL << i), depth + 1, i + 1, start, size,
                                       range_start, combination);
        range_start += combination[n - i - 1][m - depth - 1];
    }
}

/**
 * @brief Calcrate all pair of n choose m parallel
 * @param[in] n Number of bits
 * @param[in] m Number of one bits
 * @param[in] mpi_rank Rank of MPI
 * @param[in] mpi_size Size of MPI
 */
std::vector<long long> PauliString::_getDistributedCombinationBits(int n, int m, int mpi_rank, int mpi_size) {
    std::vector<std::vector<long long>> combination(n + 1, std::vector<long long>(n + 1, 0));
    for (int i = 0; i <= n; ++i) {
        combination[i][0] = 1;
        for (int j = 1; j < i; ++j) {
            combination[i][j] = combination[i - 1][j - 1] + combination[i - 1][j];
        }
        combination[i][i] = 1;
    }
    long long n_combi = combination[n][m];
    long long size_low = n_combi / mpi_size;
    long long size_high = (n_combi + mpi_size - 1) / mpi_size;
    long long size;
    long long start;
    int num_ranks_high = n_combi % mpi_size;
    if (mpi_rank < num_ranks_high) {
        size = size_high;
        start = size_high * mpi_rank;
    } else {
        size = size_low;
        start = size_high * num_ranks_high + (mpi_rank - num_ranks_high) * size_low;
    }

    assert(start + size <= combination[n][m]);
    std::vector<long long> comb_bits(size);
    int comb_idx = 0;
    _recursiveCombinationBitsRange(n, m, comb_bits, comb_idx, 0, 0, 0, start, size, 0, combination);
    assert(comb_idx == size);
    return comb_bits;
}
