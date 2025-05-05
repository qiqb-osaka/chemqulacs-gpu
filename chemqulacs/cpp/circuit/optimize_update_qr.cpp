/**
 * @file optimize_update_qr.cpp
 * @brief Implementation of the OptimizeUpdateQR class
 * @author Yusuke Teranishi
 */
#include <circuit/circuit_element.hpp>
#include <circuit/optimize_update_qr.hpp>
#include <circuit/swap_global_qubits.hpp>
#include <circuit/swap_local_qubits.hpp>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

template class OptimizeUpdateQR<double>;
template class OptimizeUpdateQR<float>;

/**
 * @param circuit_elements List of circuit elements
 * @param wire List of qubit wire
 * @brief Add QR sequentially whenever necessary
 */
template <typename FP>
void OptimizeUpdateQR<FP>::optimizeUnorder(std::vector<CircuitElement<FP> *> &circuit_elements,
                                           std::vector<int> &wire) {
    std::vector<CircuitElement<FP> *> optimized_circuit;
    std::vector<int2> swap_qubits(this->_n_global_qubits);
    for (int i = 0; i < this->_n_global_qubits; ++i) {
        swap_qubits[i] = {i, i + this->_n_local_qubits};
    }
    for (CircuitElement<FP> *element : circuit_elements) {
        std::vector<int> controls = element->getControls();
        std::vector<int> targets = element->getTargets();
        bool possible_apply = true;
        for (int q : controls) {
            if (q >= this->_n_local_qubits) {
                possible_apply = false;
            }
        }
        for (int q : targets) {
            if (q >= this->_n_local_qubits) {
                possible_apply = false;
            }
        }
        if (possible_apply) {
            optimized_circuit.push_back(element);
        } else {
            optimized_circuit.push_back(new SwapGlobalQubits<FP>(swap_qubits));
            for (int &q : controls) {
                if (q >= this->_n_local_qubits) {
                    q -= this->_n_local_qubits;
                }
            }
            for (int &q : targets) {
                if (q >= this->_n_local_qubits) {
                    q -= this->_n_local_qubits;
                }
            }
            element->setControls(controls);
            element->setTargets(targets);
            optimized_circuit.push_back(element);
            optimized_circuit.push_back(new SwapGlobalQubits<FP>(swap_qubits));
        }
    }
    circuit_elements = optimized_circuit;
}

/**
 * @param circuit_elements List of circuit elements
 * @param wire List of qubit wire
 * @brief Add QR prioritizing qubits with slower gate
 * appearances to minimize the number of QR
 */
template <typename FP>
void OptimizeUpdateQR<FP>::optimizeTimeSpaceTiling(std::vector<CircuitElement<FP> *> &circuit_elements,
                                                   std::vector<int> &wire) {
    std::vector<CircuitElement<FP> *> optimized_circuit;
    int n_elements = circuit_elements.size();
    std::vector<int> element_depth(n_elements);
    std::vector<std::vector<int>> targets(n_elements);
    std::vector<std::vector<int>> controls(n_elements);
    std::vector<long long> apply_qubits(n_elements);  // binary
    std::vector<int> depth_per_qubit(this->_n_qubits, -1);
    for (int i = 0; i < n_elements; ++i) {
        controls[i] = circuit_elements[i]->getControls();
        targets[i] = circuit_elements[i]->getTargets();
        int max_depth = -1;
        long long bin = 0;
        for (int q : controls[i]) {
            max_depth = std::max(max_depth, depth_per_qubit[q]);
            bin |= (1LL << q);
        }
        for (int q : targets[i]) {
            max_depth = std::max(max_depth, depth_per_qubit[q]);
            bin |= (1LL << q);
        }
        element_depth[i] = max_depth + 1;
        apply_qubits[i] = bin;
        for (int q : controls[i]) {
            depth_per_qubit[q] = max_depth + 1;
        }
        for (int q : targets[i]) {
            depth_per_qubit[q] = max_depth + 1;
        }
    }
    int circuit_depth = *std::max_element(depth_per_qubit.begin(), depth_per_qubit.end()) + 1;
    std::vector<std::set<int>> element_per_depth(circuit_depth);
    for (int i = 0; i < n_elements; ++i) {
        element_per_depth[element_depth[i]].insert(i);
    }

    long long current_local_qubits = (1LL << this->_n_local_qubits) - 1;
    int nv_comm_cnt = 0;
    int ib_comm_cnt = 0;
    int external_wire_threshold = this->_n_local_qubits + __builtin_ctz(_ExecuteManager.getDevicesPerNode());

    int applyed_depth = 0;
    while (applyed_depth < circuit_depth) {
        long long allowed_qubits = current_local_qubits;
        for (int d = applyed_depth; d < circuit_depth; ++d) {
            for (auto it = element_per_depth[d].begin(); it != element_per_depth[d].end();) {
                int element_id = *it;
                if ((apply_qubits[element_id] & allowed_qubits) == apply_qubits[element_id]) {
                    for (int &control : controls[element_id]) {
                        control = wire[control];
                    }
                    for (int &target : targets[element_id]) {
                        target = wire[target];
                    }
                    circuit_elements[element_id]->setControls(controls[element_id]);
                    circuit_elements[element_id]->setTargets(targets[element_id]);
                    optimized_circuit.push_back(circuit_elements[element_id]);
                    it = element_per_depth[d].erase(it);
                } else {
                    allowed_qubits &= ~apply_qubits[element_id];
                    it++;
                }
            }
        }
        while (applyed_depth < circuit_depth && (int)element_per_depth[applyed_depth].size() == 0) {
            applyed_depth++;
        }
        if (applyed_depth >= circuit_depth) {
            break;
        }
        long long next_local_qubits = 0;
        bool is_finish = false;
        for (int d = applyed_depth; d < circuit_depth && !is_finish; ++d) {
            for (int element_id : element_per_depth[d]) {
                if (__builtin_popcountll(next_local_qubits | apply_qubits[element_id]) >= this->_n_local_qubits) {
                    long long add_qubits = (~next_local_qubits & apply_qubits[element_id]);
                    int n_reduce =
                        __builtin_popcountll(next_local_qubits | apply_qubits[element_id]) - this->_n_local_qubits;
                    for (int k = 0; k < n_reduce; ++k) {
                        add_qubits ^= (add_qubits & -add_qubits);  // reduce LSB
                    }
                    next_local_qubits |= add_qubits;
                    is_finish = true;
                    break;
                } else {
                    next_local_qubits |= apply_qubits[element_id];
                }
            }
        }
        if (__builtin_popcountll(next_local_qubits) < this->_n_local_qubits) {
            int n_add = this->_n_local_qubits - __builtin_popcountll(next_local_qubits);
            for (int k = 0; k < n_add; ++k) {
                int zero_pos = 0;
                while ((next_local_qubits & (1LL << zero_pos))) zero_pos++;
                next_local_qubits |= (1LL << zero_pos);
            }
        }
        assert(__builtin_popcountll(next_local_qubits) == this->_n_local_qubits);

        bool is_comm_nv = true;

        long long current_swap_qubits = current_local_qubits ^ (current_local_qubits & next_local_qubits);
        long long next_swap_qubits = next_local_qubits ^ (current_local_qubits & next_local_qubits);
        std::vector<int2> swap_qubits;
        while (current_swap_qubits > 0) {
            int q_current = __builtin_ctzll(current_swap_qubits);  // LSB
            int q_next = __builtin_ctzll(next_swap_qubits);        // LSB
            current_swap_qubits ^= (1LL << q_current);
            next_swap_qubits ^= (1LL << q_next);
            int min_wire = std::min(wire[q_current], wire[q_next]);
            int max_wire = std::max(wire[q_current], wire[q_next]);
            assert(min_wire < this->_n_local_qubits);
            assert(max_wire >= this->_n_local_qubits);
            if (max_wire >= external_wire_threshold) {
                is_comm_nv = false;
            }
            swap_qubits.emplace_back(min_wire, max_wire);
            std::swap(wire[q_current], wire[q_next]);
        }
        if (is_comm_nv) {
            nv_comm_cnt++;
        } else {
            ib_comm_cnt++;
        }
        assert((int)swap_qubits.size() > 0);
        assert((int)swap_qubits.size() <= this->_n_global_qubits);
        optimized_circuit.push_back(new SwapGlobalQubits<FP>(swap_qubits));
        current_local_qubits = next_local_qubits;
    }
    circuit_elements = optimized_circuit;

    if (_ExecuteManager.getMpiRank() == 0) {
        printf("[Update]n_qr=%d,n_qr_nv=%d,n_qr_ib=%d\n", nv_comm_cnt + ib_comm_cnt, nv_comm_cnt, ib_comm_cnt);
    }
}

/**
 * @param circuit_elements List of circuit elements
 * @param wire List of qubit wire
 * @brief Excecute TimeSpaceTiling while minimizing the communication frequency
 * between GPUs with slower communication speeds
 */
template <typename FP>
void OptimizeUpdateQR<FP>::optimizeTimeSpaceTilingAwareInterconnect(std::vector<CircuitElement<FP> *> &circuit_elements,
                                                                    std::vector<int> &wire) {
    std::vector<CircuitElement<FP> *> optimized_circuit;
    int n_elements = circuit_elements.size();
    std::vector<int> element_depth(n_elements);
    std::vector<std::vector<int>> targets(n_elements);
    std::vector<std::vector<int>> controls(n_elements);
    std::vector<long long> apply_qubits(n_elements);  // binary
    std::vector<int> depth_per_qubit(this->_n_qubits, -1);
    for (int i = 0; i < n_elements; ++i) {
        controls[i] = circuit_elements[i]->getControls();
        targets[i] = circuit_elements[i]->getTargets();
        int max_depth = -1;
        long long bin = 0;
        for (int q : controls[i]) {
            max_depth = std::max(max_depth, depth_per_qubit[q]);
            bin |= (1LL << q);
        }
        for (int q : targets[i]) {
            max_depth = std::max(max_depth, depth_per_qubit[q]);
            bin |= (1LL << q);
        }
        element_depth[i] = max_depth + 1;
        apply_qubits[i] = bin;
        for (int q : controls[i]) {
            depth_per_qubit[q] = max_depth + 1;
        }
        for (int q : targets[i]) {
            depth_per_qubit[q] = max_depth + 1;
        }
    }
    int circuit_depth = *std::max_element(depth_per_qubit.begin(), depth_per_qubit.end()) + 1;
    std::vector<std::set<int>> element_per_depth(circuit_depth);
    for (int i = 0; i < n_elements; ++i) {
        element_per_depth[element_depth[i]].insert(i);
    }

    long long current_local_qubits = (1LL << this->_n_local_qubits) - 1;
    int external_wire_threshold = this->_n_local_qubits + __builtin_ctz(_ExecuteManager.getDevicesPerNode());

    int nv_comm_cnt = 0;
    int ib_comm_cnt = 0;

    int applyed_depth = 0;
    while (applyed_depth < circuit_depth) {
        long long allowed_qubits = current_local_qubits;
        for (int d = applyed_depth; d < circuit_depth; ++d) {
            for (auto it = element_per_depth[d].begin(); it != element_per_depth[d].end();) {
                int element_id = *it;
                if ((apply_qubits[element_id] & allowed_qubits) == apply_qubits[element_id]) {
                    for (int &control : controls[element_id]) {
                        control = wire[control];
                    }
                    for (int &target : targets[element_id]) {
                        target = wire[target];
                    }
                    circuit_elements[element_id]->setControls(controls[element_id]);
                    circuit_elements[element_id]->setTargets(targets[element_id]);
                    optimized_circuit.push_back(circuit_elements[element_id]);
                    it = element_per_depth[d].erase(it);
                } else {
                    allowed_qubits &= ~apply_qubits[element_id];
                    it++;
                }
            }
        }
        while (applyed_depth < circuit_depth && (int)element_per_depth[applyed_depth].size() == 0) {
            applyed_depth++;
        }
        if (applyed_depth >= circuit_depth) {
            break;
        }

        long long ng_qubits = 0;
        for (int q = 0; q < this->_n_qubits; ++q) {
            if (wire[q] >= external_wire_threshold) {
                ng_qubits |= (1LL << q);
            }
        }
        long long next_local_qubits = 0;
        for (int step = 0; step < 2; ++step) {
            bool is_finish = false;
            for (int d = applyed_depth; d < circuit_depth && !is_finish; ++d) {
                for (int element_id : element_per_depth[d]) {
                    if (ng_qubits & apply_qubits[element_id]) {
                        ng_qubits |= apply_qubits[element_id];
                    } else if (__builtin_popcountll(next_local_qubits | apply_qubits[element_id]) <
                               this->_n_local_qubits) {
                        next_local_qubits |= apply_qubits[element_id];
                    } else {
                        long long add_qubits = (~next_local_qubits & apply_qubits[element_id]);
                        int n_reduce =
                            __builtin_popcountll(next_local_qubits | apply_qubits[element_id]) - this->_n_local_qubits;
                        for (int k = 0; k < n_reduce; ++k) {
                            add_qubits ^= (add_qubits & -add_qubits);  // reduce LSB
                                                                       // 同じdepthの中で削減できるかも
                        }
                        next_local_qubits |= add_qubits;
                        is_finish = true;
                        break;
                    }
                }
            }
            if (step == 0) {
                if (next_local_qubits > 0) {
                    nv_comm_cnt++;
                    break;
                } else {
                    ib_comm_cnt++;
                }
            }
            ng_qubits = 0;
        }
        if (__builtin_popcountll(next_local_qubits) < this->_n_local_qubits) {
            int n_add = this->_n_local_qubits - __builtin_popcountll(next_local_qubits);
            for (int k = 0; k < n_add; ++k) {
                int zero_pos = 0;
                while ((next_local_qubits & (1LL << zero_pos))) zero_pos++;
                next_local_qubits |= (1LL << zero_pos);
            }
        }
        assert(__builtin_popcountll(next_local_qubits) == this->_n_local_qubits);

        long long current_swap_qubits = current_local_qubits ^ (current_local_qubits & next_local_qubits);
        long long next_swap_qubits = next_local_qubits ^ (current_local_qubits & next_local_qubits);
        std::vector<int2> swap_qubits;
        while (current_swap_qubits > 0) {
            int q_current = __builtin_ctzll(current_swap_qubits);  // LSB
            int q_next = __builtin_ctzll(next_swap_qubits);        // LSB
            current_swap_qubits ^= (1LL << q_current);
            next_swap_qubits ^= (1LL << q_next);
            int min_wire = std::min(wire[q_current], wire[q_next]);
            int max_wire = std::max(wire[q_current], wire[q_next]);
            assert(min_wire < this->_n_local_qubits);
            assert(max_wire >= this->_n_local_qubits);
            swap_qubits.emplace_back(min_wire, max_wire);
            std::swap(wire[q_current], wire[q_next]);
        }
        assert((int)swap_qubits.size() > 0);
        assert((int)swap_qubits.size() <= this->_n_global_qubits);
        optimized_circuit.push_back(new SwapGlobalQubits<FP>(swap_qubits));
        current_local_qubits = next_local_qubits;
    }
    circuit_elements = optimized_circuit;

    if (_ExecuteManager.getMpiRank() == 0) {
        printf("[Update]n_qr=%d,n_qr_nv=%d,n_qr_ib=%d\n", nv_comm_cnt + ib_comm_cnt, nv_comm_cnt, ib_comm_cnt);
    }
}

/**
 * @param circuit_elements List of circuit elements
 * @param wire List of qubit wire
 * @brief Reorder qubits without SWAP
 */
template <typename FP>
void OptimizeUpdateQR<FP>::reorderingStarndardQubits(std::vector<CircuitElement<FP> *> &circuit_elements,
                                                     std::vector<int> &start_wire, std::vector<int> goal_wire) {
    if (goal_wire.empty()) {
        goal_wire.resize(this->_n_qubits);
        std::iota(goal_wire.begin(), goal_wire.end(), 0);
    }

    std::vector<int> pos_start(this->_n_qubits);
    for (int i = 0; i < this->_n_qubits; ++i) {
        pos_start[start_wire[i]] = i;
    }
    std::vector<int> pos_goal(this->_n_qubits);
    for (int i = 0; i < this->_n_qubits; ++i) {
        pos_goal[goal_wire[i]] = i;
    }
    long long sorted_global = 0;
    auto add = [&](long long &bits, int q) { bits |= (1LL << q); };
    while (sorted_global != (1LL << this->_n_global_qubits) - 1) {
        long long used = 0;
        auto is_used = [&](int q) { return (used & (1LL << q)); };
        std::vector<int2> swap_qubits;
        for (int i = this->_n_local_qubits; i < this->_n_qubits; ++i) {
            if (start_wire[pos_goal[i]] == i) {
                add(sorted_global, i - this->_n_local_qubits);
            } else if (!is_used(i) and !is_used(start_wire[pos_goal[i]])) {
                swap_qubits.emplace_back(i, start_wire[pos_goal[i]]);
                add(used, i), add(used, start_wire[pos_goal[i]]);
                int q = pos_start[i];
                pos_start[start_wire[pos_goal[i]]] = q;
                start_wire[q] = start_wire[pos_goal[i]];
                pos_start[i] = i;
                start_wire[pos_goal[i]] = i;
                add(sorted_global, i - this->_n_local_qubits);
            }
        }
        assert(std::ssize(swap_qubits) <= this->_n_global_qubits);
        if (!swap_qubits.empty()) {
            circuit_elements.emplace_back(new SwapGlobalQubits<FP>(swap_qubits));
        }
    }
    long long sorted_local = 0;
    while (sorted_local != (1LL << this->_n_local_qubits) - 1) {
        long long used = 0;
        auto is_used = [&](int q) { return (used & (1LL << q)); };
        std::vector<int2> swap_qubits;
        for (int i = 0; i < this->_n_local_qubits; ++i) {
            if (start_wire[pos_goal[i]] == i) {
                add(sorted_local, i);
            } else if (!is_used(i) and !is_used(start_wire[pos_goal[i]])) {
                swap_qubits.emplace_back(i, start_wire[pos_goal[i]]);
                add(used, i), add(used, start_wire[pos_goal[i]]);
                int q = pos_start[i];
                pos_start[start_wire[pos_goal[i]]] = q;
                start_wire[q] = start_wire[pos_goal[i]];
                pos_start[i] = i;
                start_wire[pos_goal[i]] = i;
                add(sorted_local, i);
            }
        }
        if (!swap_qubits.empty()) {
            circuit_elements.emplace_back(new SwapLocalQubits<FP>(swap_qubits));
        }
    }
    for (int i = 0; i < this->_n_qubits; ++i) {
        assert(start_wire[i] == goal_wire[i]);
    }
}