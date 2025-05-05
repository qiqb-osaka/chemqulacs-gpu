/**
 * @file expectation_computer.cpp
 * @brief Implementation of ExpectationComputer class
 * @author Yusuke Teranishi
 */
#include <ansatz/library/gate.hpp>
#include <circuit/parametric_matrix.hpp>
#include <expectation/communicate_swap.hpp>
#include <expectation/compute_dotproduct.hpp>
#include <expectation/compute_paulibasis.hpp>
#include <expectation/expectation_computer.hpp>
#include <expectation/expectation_element.hpp>
#include <pauli/pauli_string.hpp>
#include <utils/execute_manager.hpp>
#include <utils/precision.hpp>

#include <custatevec.h>
#include <nvtx3/nvToolsExt.h>
#include <omp.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template class ExpectationComputer<double>;
template class ExpectationComputer<float>;

/**
 * @brief　Add QR sequentially
 * @param[in] wire Wires for the QR
 */
template <typename FP>
void ExpectationComputer<FP>::buildExpectationElementsUnorder(std::vector<int> &wire) {
    assert((int)wire.size() == _n_qubits);
    for (int i = 0; i < _n_qubits; ++i) {
        assert(wire[i] == i);
    }
    const std::vector<PauliProduct> &pauli_products = _pauli_string->getPauliProducts();
    int n_operators = _pauli_string->getOperatorsNum();
    for (int i = 0; i < n_operators; ++i) {
        const PauliProduct &pauli_product = pauli_products[i];
        const std::vector<custatevecPauli_t> &pauli_operator = pauli_product.getPauliOperator();
        const std::vector<int> &basis_qubit = pauli_product.getBasisQubit();
        const double pauli_coef = pauli_product.getPauliCoefReal();
        int n_basis = (int)pauli_operator.size();
        if (n_basis > _n_local_qubits) {
            if (!_state_vector_work) {
                _state_vector_work = std::make_unique<StateVector<FP>>(_state_vector);
            }
            _expectation_elements.push_back(new ComputeDotProduct<FP>(pauli_product, _state_vector_work.get()));
        } else if (n_basis == 0 || basis_qubit[n_basis - 1] < _n_local_qubits) {
            _expectation_elements.push_back(new ComputePauliBasis<FP>(pauli_product));
        } else {
            long long apply_bits = 0;
            for (int j = 0; j < n_basis; ++j) {
                apply_bits |= (1LL << basis_qubit[j]);
            }
            for (int j = 0; j < _n_qubits; ++j) {
                wire[j] = j;
            }
            std::vector<int2> swap_qubits;
            for (int j = 0; j < n_basis; ++j) {
                int target = basis_qubit[j];
                if (target >= _n_local_qubits) {
                    int zero_qubit = 0;
                    while ((apply_bits & (1LL << zero_qubit)) > 0) zero_qubit++;
                    apply_bits |= (1LL << zero_qubit);
                    assert(zero_qubit < _n_local_qubits);
                    swap_qubits.emplace_back(zero_qubit, target);
                    std::swap(wire[zero_qubit], wire[target]);
                }
            }
            assert((int)swap_qubits.size() > 0);
            assert((int)swap_qubits.size() <= _n_global_qubits);
            _expectation_elements.push_back(new CommunicateSwap<FP>(swap_qubits));
            std::vector<int> basis_qubits_ = basis_qubit;
            for (int j = 0; j < n_basis; ++j) {
                basis_qubits_[j] = wire[basis_qubits_[j]];
            }
            PauliProduct new_pauli_product(pauli_operator, basis_qubits_, pauli_coef);
            _expectation_elements.push_back(new ComputePauliBasis<FP>(new_pauli_product));
            _expectation_elements.push_back(new CommunicateSwap<FP>(swap_qubits));
        }
    }
}

/**
 * @brief Reorder PauliProducts in a sequence that can be computed
 * simultaneously using the minimum clique cover
 * @param[in] wire Wires for the QR
 */
template <typename FP>
void ExpectationComputer<FP>::buildExpectationElementsOffline(std::vector<int> &wire) {
    assert((int)wire.size() == _n_qubits);
    for (int i = 0; i < _n_qubits; ++i) {
        assert(wire[i] == i);
    }

    const std::vector<PauliProduct> &pauli_products = _pauli_string->getPauliProducts();
    int n_operators = _pauli_string->getOperatorsNum();
    for (int i = 0; i < n_operators; ++i) {
        const PauliProduct &pauli_product = pauli_products[i];
        const std::vector<custatevecPauli_t> &pauli_operator = pauli_product.getPauliOperator();
        const std::vector<int> &basis_qubit = pauli_product.getBasisQubit();
        int n_basis = (int)pauli_operator.size();
        if (n_basis > _n_local_qubits) {
            if (!_state_vector_work) {
                _state_vector_work = std::make_unique<StateVector<FP>>(_state_vector);
            }
            _expectation_elements.push_back(new ComputeDotProduct<FP>(pauli_product, _state_vector_work.get()));
        }
    }

    // compute expectation by pauli on basis

    int n_local_operators = 0;
    std::vector<long long> idx_to_bits(n_operators);  // idx -> pauli binary bits
    std::vector<std::unordered_set<long long>> bits_per_n_basis(_n_local_qubits + 1);
    std::unordered_map<long long, std::vector<int>> bits_to_idx;
    std::unordered_map<long long, std::vector<long long>> bits_tree;

    // classify bits per n_basis
    for (int i = 0; i < n_operators; ++i) {
        const PauliProduct &pauli_product = pauli_products[i];
        const std::vector<custatevecPauli_t> &pauli_operator = pauli_product.getPauliOperator();
        const std::vector<int> &basis_qubit = pauli_product.getBasisQubit();
        int n_basis = (int)pauli_operator.size();
        if (n_basis <= _n_local_qubits) {
            long long bits = 0;
            for (int j = 0; j < n_basis; ++j) {
                bits |= (1LL << basis_qubit[j]);
            }
            bits_to_idx[bits].push_back(i);
            bits_per_n_basis[n_basis].insert(bits);
            n_local_operators++;
        }
    }

    // compression pauli bits tree
    for (int i = 0; i < _n_local_qubits; ++i) {
        std::vector<long long> not_merge_bits;
        for (long long child_bits : bits_per_n_basis[i]) {
            bool is_merge = false;
            for (long long parent_bits : bits_per_n_basis[i + 1]) {
                if ((child_bits & parent_bits) == child_bits) {
                    bits_tree[parent_bits].push_back(child_bits);
                    is_merge = true;
                    break;
                }
            }
            if (!is_merge) {
                not_merge_bits.push_back(child_bits);
            }
        }
        bits_per_n_basis[i + 1].insert(not_merge_bits.begin(), not_merge_bits.end());
    }

    // greedy min clique cover
    std::unordered_set<long long> represent_bits = bits_per_n_basis[_n_local_qubits];
    std::vector<long long> comb_bits_set = _getCombinationBits(_n_qubits, _n_global_qubits);
    std::map<long long, std::vector<long long>> qubits_combination;  // require sort
    while (represent_bits.size() > 0) {
        int max_cnt = 0;
        int max_idx = -1;
        for (int i = 0; i < (int)comb_bits_set.size(); ++i) {
            long long comb_bits = comb_bits_set[i];
            if (comb_bits < 0) {
                continue;
            }
            int cnt = 0;
            for (long long bits : represent_bits) {
                if ((bits & comb_bits) == bits) {
                    cnt++;
                }
            }
            if (max_cnt < cnt) {
                max_cnt = cnt;
                max_idx = i;
            }
        }
        assert(max_idx >= 0);
        long long comb_bits = comb_bits_set[max_idx];
        comb_bits_set[max_idx] = -1;
        std::vector<long long> bits_elem;
        for (auto it = represent_bits.begin(); it != represent_bits.end();) {
            long long bits = *it;
            if ((bits & comb_bits) == bits) {
                bits_elem.push_back(bits);
                it = represent_bits.erase(it);
            } else {
                ++it;
            }
        }
        qubits_combination[comb_bits] = bits_elem;
    }

    // build expectation elements on pauli basis
    int sum_pauli = 0;
    std::vector<bool> is_current_global(_n_qubits, false);
    for (int i = 0; i < _n_global_qubits; ++i) {
        is_current_global[_n_local_qubits + i] = true;
    }
    for (auto it : qubits_combination) {
        // calculate global qubit
        long long next_qubits = it.first;
        std::vector<bool> is_next_global(_n_qubits, false);
        for (int i = 0; i < _n_qubits; ++i) {
            if ((next_qubits & (1LL << i)) == 0) {  // next global
                is_next_global[i] = true;
            }
        }

        // check necessity of swap qubits
        std::vector<int2> swap_qubits;
        for (int i = 0, j = 0; i < _n_qubits; ++i) {
            if (is_current_global[i] && !is_next_global[i]) {
                bool is_swap = false;
                for (; j < _n_qubits; ++j) {
                    if (is_next_global[j] && !is_current_global[j]) {
                        int min_wire = std::min(wire[i], wire[j]);
                        int max_wire = std::max(wire[i], wire[j]);
                        assert(min_wire < _n_local_qubits);
                        assert(max_wire >= _n_local_qubits);
                        swap_qubits.emplace_back(min_wire, max_wire);
                        std::swap(wire[i], wire[j]);
                        is_swap = true;
                        j++;
                        break;
                    }
                }
                assert(is_swap);
            }
        }
        assert((int)swap_qubits.size() <= _n_global_qubits);
        if ((int)swap_qubits.size() > 0) {
            _expectation_elements.push_back(new CommunicateSwap<FP>(swap_qubits));
        }
        is_current_global = is_next_global;

        // decompression pauli bits
        std::vector<int> pauli_indexes;
        for (long long bits : it.second) {
            _getTreePauliIndexes(pauli_indexes, bits_tree, bits_to_idx, bits);
        }
        int n_pauli = pauli_indexes.size();
        sum_pauli += n_pauli;
        std::vector<PauliProduct> pauli_products_;
        for (int i = 0; i < n_pauli; ++i) {
            PauliProduct pauli_product_;
            int pauli_index = pauli_indexes[i];
            const PauliProduct &pauli_product = _pauli_string->getPauliProduct(pauli_index);
            const std::vector<custatevecPauli_t> &pauli_operator = pauli_product.getPauliOperator();
            const std::vector<int> &basis_qubit = pauli_product.getBasisQubit();
            const double pauli_coef = pauli_product.getPauliCoefReal();
            int n_basis = pauli_operator.size();
            for (int j = 0; j < n_basis; ++j) {
                int target = wire[basis_qubit[j]];
                assert(target < _n_local_qubits);
                pauli_product_.addElement(pauli_operator[j], target);
            }
            pauli_product_.setPauliCoef(pauli_coef);
            pauli_products_.push_back(pauli_product_);
        }
        _expectation_elements.push_back(new ComputePauliBasis<FP>(pauli_products_));
    }
    assert(sum_pauli == n_local_operators);
}

/**
 * @brief Precompute PauliZ and Reorder PauliProducts while disregarding PauliZ
 * @param[in] wire Wires for the QR
 */
template <typename FP>
void ExpectationComputer<FP>::buildExpectationElementsDiagonalZ(std::vector<int> &wire) {
    assert((int)wire.size() == _n_qubits);

    const std::vector<PauliProduct> &pauli_products = _pauli_string->getPauliProducts();
    int n_operators = _pauli_string->getOperatorsNum();

    // prepare compute sign pauli z tensor
    std::vector<int> sign_pauli_z(1 << _n_global_qubits);
    for (int i = 0; i < (1 << _n_global_qubits); ++i) {
        int j = _state_vector->getLocalRank();
        int parity = __builtin_popcountll(i & j) % 2;
        sign_pauli_z[i] = 1 - 2 * parity;
    }

    std::unordered_map<long long, PauliString> pauli_cover =
        _pauli_string->timeSpaceTiling(_n_global_qubits, _state_vector->getMpiCommunicator());

    // build expectation elements on pauli basis
    int sum_pauli = 0;
    long long current_global = 0;
    for (int q = 0; q < _n_qubits; ++q) {
        if (wire[q] >= _n_local_qubits) {
            current_global |= (1LL << q);
        }
    }

    int n_qr_nv = 0;
    int n_qr_ib = 0;
    int external_wire_threshold = _n_local_qubits + __builtin_ctz(_ExecuteManager.getDevicesPerNode());
    for (auto it = pauli_cover.begin(); it != pauli_cover.end(); ++it) {
        long long next_global = it->first;
        long long current_swap_qubits = current_global ^ (current_global & next_global);
        long long next_swap_qubits = next_global ^ (current_global & next_global);
        std::vector<int2> swap_qubits;
        bool is_qr_ib = false;
        while (current_swap_qubits > 0) {
            int q_current = __builtin_ctzll(current_swap_qubits);  // LSB
            int q_next = __builtin_ctzll(next_swap_qubits);        // LSB
            current_swap_qubits ^= (1LL << q_current);
            next_swap_qubits ^= (1LL << q_next);
            int min_wire = std::min(wire[q_current], wire[q_next]);
            int max_wire = std::max(wire[q_current], wire[q_next]);
            assert(min_wire < _n_local_qubits);
            assert(max_wire >= _n_local_qubits);
            if (max_wire >= external_wire_threshold) {
                is_qr_ib = true;
            }
            swap_qubits.emplace_back(min_wire, max_wire);
            std::swap(wire[q_current], wire[q_next]);
        }
        assert((int)swap_qubits.size() <= _n_global_qubits);
        if ((int)swap_qubits.size() > 0) {
            _expectation_elements.push_back(new CommunicateSwap<FP>(swap_qubits));
        }
        current_global = next_global;
        if (is_qr_ib) {
            n_qr_ib++;
        } else {
            n_qr_nv++;
        }
        int n_pauli = it->second.getOperatorsNum();
        sum_pauli += n_pauli;
        std::vector<PauliProduct> pauli_products_;
        for (PauliProduct &pauli_product : it->second.getPauliProducts()) {
            PauliProduct pauli_product_;
            const std::vector<custatevecPauli_t> &pauli_operator = pauli_product.getPauliOperator();
            const std::vector<int> &basis_qubit = pauli_product.getBasisQubit();
            const double pauli_coef = pauli_product.getPauliCoefReal();
            int n_basis = (int)pauli_operator.size();
            int z_mask = 0;  // _n_global_qubits binary bits
            for (int j = 0; j < n_basis; ++j) {
                int target = wire[basis_qubit[j]];
                if (target >= _n_local_qubits) {  // global pauli z
                    z_mask |= (1 << (target - _n_local_qubits));
                    assert(pauli_operator[j] == CUSTATEVEC_PAULI_Z);
                } else {
                    pauli_product_.addElement(pauli_operator[j], target);
                }
            }
            assert(z_mask < (1LL << _n_global_qubits));
            pauli_product_.setPauliCoef(sign_pauli_z[z_mask] * pauli_coef);
            pauli_products_.push_back(pauli_product_);
        }
        _expectation_elements.push_back(new ComputePauliBasis<FP>(pauli_products_));
    }
    assert(sum_pauli == n_operators);

    if (_ExecuteManager.getMpiRank() == 0) {
        printf("[Exp]n_qr=%d,n_qr_nv=%d,n_qr_ib=%d\n", n_qr_nv + n_qr_ib, n_qr_nv, n_qr_ib);
    }
}

/**
 * @brief Execute DiagonalZ and reduce swap between slower GPUs
 * @param[in] wire Wires for the QR
 */
template <typename FP>
void ExpectationComputer<FP>::buildExpectationElementsDiagonalzGreedy(std::vector<int> &wire) {
    assert((int)wire.size() == _n_qubits);

    const std::vector<PauliProduct> &pauli_products = _pauli_string->getPauliProducts();
    int n_operators = _pauli_string->getOperatorsNum();

    // prepare compute sign pauli z tensor
    std::vector<int> sign_pauli_z(1 << _n_global_qubits);
    for (int i = 0; i < (1 << _n_global_qubits); ++i) {
        int j = _state_vector->getLocalRank();
        int parity = __builtin_popcountll(i & j) % 2;
        sign_pauli_z[i] = 1 - 2 * parity;
    }

    std::unordered_map<long long, PauliString> pauli_cover =
        _pauli_string->timeSpaceTiling(_n_global_qubits, _state_vector->getMpiCommunicator());

    // build expectation elements on pauli basis
    int sum_pauli = 0;
    long long current_global = 0;
    for (int q = 0; q < _n_qubits; ++q) {
        if (wire[q] >= _n_local_qubits) {
            current_global |= (1LL << q);
        }
    }

    int n_qr_nv = 0;
    int n_qr_ib = 0;
    int external_wire_threshold = _n_local_qubits + __builtin_ctz(_ExecuteManager.getDevicesPerNode());
    while (pauli_cover.size() > 0) {
        int min_swap_nv = _n_global_qubits + 1;
        int min_swap_ib = _n_global_qubits + 1;
        auto it_min_qr = pauli_cover.begin();
        for (auto it = pauli_cover.begin(); it != pauli_cover.end(); ++it) {
            long long next_global = it->first;
            long long swap_qubits_bin = current_global ^ (current_global & next_global);
            int n_swap_ib = 0;
            int n_swap_nv = 0;
            while (swap_qubits_bin > 0) {
                int q_current = __builtin_ctzll(swap_qubits_bin);
                swap_qubits_bin ^= (1LL << q_current);
                if (wire[q_current] >= external_wire_threshold) {
                    n_swap_ib++;
                } else {
                    n_swap_nv++;
                }
            }
            if (min_swap_ib > n_swap_ib) {
                min_swap_ib = n_swap_ib;
                min_swap_nv = n_swap_nv;
                it_min_qr = it;
            } else if (min_swap_ib == n_swap_ib) {
                if (min_swap_nv > n_swap_nv) {
                    min_swap_ib = n_swap_ib;
                    min_swap_nv = n_swap_nv;
                    it_min_qr = it;
                }
            }
        }
        assert(min_swap_nv + min_swap_ib <= _n_global_qubits);
        if (min_swap_ib == 0) {
            n_qr_nv++;
        } else {
            n_qr_ib++;
        }

        long long next_global = it_min_qr->first;
        long long current_swap_qubits = current_global ^ (current_global & next_global);
        long long next_swap_qubits = next_global ^ (current_global & next_global);
        std::vector<int2> swap_qubits;
        while (current_swap_qubits > 0) {
            int q_current = __builtin_ctzll(current_swap_qubits);  // LSB
            int q_next = __builtin_ctzll(next_swap_qubits);        // LSB
            current_swap_qubits ^= (1LL << q_current);
            next_swap_qubits ^= (1LL << q_next);
            int min_wire = std::min(wire[q_current], wire[q_next]);
            int max_wire = std::max(wire[q_current], wire[q_next]);
            assert(min_wire < _n_local_qubits);
            assert(max_wire >= _n_local_qubits);
            swap_qubits.emplace_back(min_wire, max_wire);
            std::swap(wire[q_current], wire[q_next]);
        }
        assert((int)swap_qubits.size() <= _n_global_qubits);
        if ((int)swap_qubits.size() > 0) {
            _expectation_elements.push_back(new CommunicateSwap<FP>(swap_qubits));
        }
        current_global = next_global;
        int n_pauli = it_min_qr->second.getOperatorsNum();
        sum_pauli += n_pauli;

        std::vector<PauliProduct> pauli_products_;
        for (PauliProduct &pauli_product : it_min_qr->second.getPauliProducts()) {
            PauliProduct pauli_product_;
            const std::vector<custatevecPauli_t> &pauli_operator = pauli_product.getPauliOperator();
            const std::vector<int> &basis_qubit = pauli_product.getBasisQubit();
            const double pauli_coef = pauli_product.getPauliCoefReal();
            int n_basis = (int)pauli_operator.size();
            int z_mask = 0;  // _n_global_qubits binary bits
            for (int j = 0; j < n_basis; ++j) {
                int target = wire[basis_qubit[j]];
                if (target >= _n_local_qubits) {  // global pauli z
                    z_mask |= (1 << (target - _n_local_qubits));
                    assert(pauli_operator[j] == CUSTATEVEC_PAULI_Z);
                } else {
                    pauli_product_.addElement(pauli_operator[j], target);
                }
            }
            assert(z_mask < (1LL << _n_global_qubits));
            pauli_product_.setPauliCoef(sign_pauli_z[z_mask] * pauli_coef);
            pauli_products_.push_back(pauli_product_);
        }
        _expectation_elements.push_back(new ComputePauliBasis<FP>(pauli_products_));
        pauli_cover.erase(it_min_qr);
    }
    assert(sum_pauli == n_operators);

    if (_ExecuteManager.getMpiRank() == 0) {
        printf("[Exp]n_qr=%d,n_qr_nv=%d,n_qr_ib=%d\n", n_qr_nv + n_qr_ib, n_qr_nv, n_qr_ib);
    }
}

/**
 * @brief Compute all elements' dot
 */
template <typename FP>
void ExpectationComputer<FP>::buildExpectationElementsAllDot(std::vector<int> &wire) {
    assert((int)wire.size() == _n_qubits);
    for (int i = 0; i < _n_qubits; ++i) {
        assert(wire[i] == i);
    }

    if (!_state_vector_work) {
        _state_vector_work = std::make_unique<StateVector<FP>>(_state_vector);
    }
    int n_operators = _pauli_string->getOperatorsNum();
    for (int i = 0; i < n_operators; ++i) {
        _expectation_elements.push_back(
            new ComputeDotProduct<FP>(_pauli_string->getPauliProduct(i), _state_vector_work.get()));
    }
}

/**
 * @brief Internal function for _getCombinationBits
 * @param[in] n Number of bits
 * @param[in] m Number of zero bits
 * @param[out] comb_bits All pair of n choose m
 * @param[in] comb_work Work array
 * @param[in] comb_idx Index of comb_bits
 * @param[in] depth Depth of recursive
 * @param[in] i_n Start index of recursive
 */
template <typename FP>
void ExpectationComputer<FP>::_recursiveCombinationBits(int n, int m, std::vector<long long> &comb_bits,
                                                        std::vector<int> &comb_work, int &comb_idx, int depth,
                                                        int i_n) {
    if (depth == m) {
        long long bits = (1LL << n) - 1;
        for (int i = 0; i < m; ++i) {
            bits ^= (1LL << comb_work[i]);
        }
        comb_bits[comb_idx++] = bits;
        return;
    }
    for (int i = i_n; i < n; ++i) {
        comb_work[depth] = i;
        _recursiveCombinationBits(n, m, comb_bits, comb_work, comb_idx, depth + 1, i + 1);
    }
}

/**
 * @brief Calcrate all pair of n_bits choose zero_bits
 * @param[in] n_bits Number of bits
 * @param[in] zero_bits Number of zero bits
 * @return All pair of n_bits choose zero_bits
 */
template <typename FP>
std::vector<long long> ExpectationComputer<FP>::_getCombinationBits(int n_bits, int zero_bits) {
    long long n_comb = 1;
    for (int i = 0; i < zero_bits; ++i) {
        n_comb *= (n_bits - i);
    }
    for (int i = 0; i < zero_bits; ++i) {
        n_comb /= (zero_bits - i);
    }
    std::vector<long long> comb_bits(n_comb);
    std::vector<int> comb_work(zero_bits);
    int comb_idx = 0;
    _recursiveCombinationBits(n_bits, zero_bits, comb_bits, comb_work, comb_idx, 0, 0);
    return comb_bits;
}

template <typename FP>
void ExpectationComputer<FP>::_recursiveCombinationBitsNew(int n, int m, std::vector<long long> &comb_bits,
                                                           int &comb_idx, long long bits, int depth, int i_n) {
    if (depth == m) {
        comb_bits[comb_idx++] = bits;
        return;
    }
    for (int i = i_n; i < n; ++i) {
        _recursiveCombinationBitsNew(n, m, comb_bits, comb_idx, bits | (1LL << i), depth + 1, i + 1);
    }
}

template <typename FP>
std::vector<long long> ExpectationComputer<FP>::_getCombinationBitsNew(int n_bits, int one_bits) {
    long long n_comb = 1;
    for (int i = 0; i < one_bits; ++i) {
        n_comb *= (n_bits - i);
    }
    for (int i = 0; i < one_bits; ++i) {
        n_comb /= (one_bits - i);
    }
    std::vector<long long> comb_bits(n_comb);
    int comb_idx = 0;
    _recursiveCombinationBitsNew(n_bits, one_bits, comb_bits, comb_idx, 0, 0, 0);
    return comb_bits;
}

/**
 * @brief Find the Paulis that can be computed simultaneously based on the bits
 * @param[out] pauli_indexes Pauli indexes that can be computed simultaneously
 * @param[in] bits_tree Tree structure of Pauli bits
 * @param[in] bits_to_idx Mapping from Pauli bits to Pauli index
 * @param[in] current_bits Current Pauli bits
 */
template <typename FP>
void ExpectationComputer<FP>::_getTreePauliIndexes(std::vector<int> &pauli_indexes,
                                                   std::unordered_map<long long, std::vector<long long>> &bits_tree,
                                                   std::unordered_map<long long, std::vector<int>> &bits_to_idx,
                                                   long long current_bits) {
    for (int idx : bits_to_idx[current_bits]) {
        pauli_indexes.push_back(idx);
    }
    for (long long child_bits : bits_tree[current_bits]) {
        _getTreePauliIndexes(pauli_indexes, bits_tree, bits_to_idx, child_bits);
    }
}

template <typename FP>
void ExpectationComputer<FP>::_getTreePauliIndexesNew(std::vector<int> &pauli_indexes,
                                                      const std::vector<std::vector<int>> &pauli_child, int parent) {
    pauli_indexes.push_back(parent);
    for (int child : pauli_child[parent]) {
        _getTreePauliIndexesNew(pauli_indexes, pauli_child, child);
    }
}

/**
 * @brief Apply each expectation element to the state vector and compute the
 * expectation value
 * @return Expectation value
 */
template <typename FP>
double ExpectationComputer<FP>::computeExpectation() {
    _timer_exp_compute.restart();
    _timer_exp_communicate.reset();
    double exp_local_sum = 0;
    double exp;
    for (int i = 0; i < (int)_expectation_elements.size(); ++i) {
        exp = _expectation_elements[i]->computeExpectation(_state_vector);
        exp_local_sum += exp;
    }
    _timer_exp_compute.stop();
    _TimerDict.registerTime("exp_compute", _timer_exp_compute.getTime());
    _TimerDict.registerTime("exp_communicate", _timer_exp_communicate.getTime());

    _timer_exp_reduce->restart();
    double exp_sum = 0;
    HANDLE_MPI(MPI_Reduce(&exp_local_sum, &exp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, _state_vector->getMpiCommunicator()));
    // All process or only local root process handle exp value.
    // HANDLE_MPI(MPI_Allreduce(&exp_local_sum, &exp_sum, 1, MPI_DOUBLE,
    // MPI_SUM, _state_vector->getMpiCommunicator()));
    _timer_exp_reduce->stop();
    return exp_sum;
}
