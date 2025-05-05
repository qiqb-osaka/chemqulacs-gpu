/**
 * @file pauli_exp.cpp
 * @brief Implementation of the PauliExp class
 * @author Yusuke Teranishi
 */
#include <ansatz/library/gate.hpp>
#include <ansatz/pauli_exp.hpp>
#include <circuit/pauli_rotation.hpp>
#include <circuit/swap_global_qubits.hpp>
#include <utils/precision.hpp>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <list>
#include <vector>

template class PauliExp<double>;
template class PauliExp<float>;

/**
 * @brief build the circuit for the ansatz
 * @tparam FP floating point type
 * @details push Gate that generate from pauli string(\f[e^{iP\theta}\f]) to
 * CircuitElement
 */
template <typename FP>
void PauliExp<FP>::buildCircuit() {
    int n_params = _param_ranges.size() - 1;
    this->_params.resize(n_params);
    this->_param_to_circuit.resize(n_params);

    for (int i_param = 0; i_param < n_params; ++i_param) {
        for (int i_op = _param_ranges[i_param]; i_op < _param_ranges[i_param + 1]; ++i_op) {
            const PauliProduct &pauli_product = _pauli_string->getPauliProduct(i_op);
            CircuitElement<FP> *element = new PauliRotation<FP>(pauli_product, &this->_params[i_param]);
            this->_circuit_elements.push_back(element);
            this->_param_to_circuit[i_param].push_back(element);
        }
    }
}

/**
 * @brief Build Qubit Reordering (QR) for the update
 * @param wire Wires for the QR
 */
template <typename FP>
void PauliExp<FP>::_buildPauliRotationsQR(std::vector<int> &wire) {
    std::vector<CircuitElement<FP> *> circuit_elements;
    int n_params = _param_ranges.size() - 1;
    int n_operators = _pauli_string->getOperatorsNum();
    std::vector<int> operator_to_paramid(n_operators);

    for (int i_param = 0; i_param < n_params; ++i_param) {
        for (int i_op = _param_ranges[i_param]; i_op < _param_ranges[i_param + 1]; ++i_op) {
            operator_to_paramid[i_op] = i_param;
        }
    }

    // prepare compute sign pauli z tensor
    std::vector<int> sign_pauli_z(1 << this->_n_global_qubits);
    for (int i = 0; i < (1 << this->_n_global_qubits); ++i) {
        int j = this->_mpi_local_rank;
        int parity = __builtin_popcountll(i & j) % 2;
        sign_pauli_z[i] = 1 - 2 * parity;
    }

    Timer *timer = _TimerDict.addTimerHost("timeSpaceTiling");
    timer->start();
    // std::unordered_map<long long, std::vector<int>> pauli_cover =
    //     _pauli_string->getMinLocalCover(this->_n_global_qubits);
    std::unordered_map<long long, PauliString> pauli_cover =
        _pauli_string->timeSpaceTiling(this->_n_global_qubits, this->_mpi_comm);
    timer->stop();

    // build expectation elements on pauli basis
    int sum_pauli = 0;
    int n_qr_nv = 0;
    int n_qr_ib = 0;
    long long current_global = 0;
    for (int q = 0; q < this->_n_qubits; ++q) {
        if (wire[q] >= this->_n_local_qubits) {
            current_global |= (1LL << q);
        }
    }
    int external_wire_threshold = this->_n_local_qubits + __builtin_ctz(_ExecuteManager.getDevicesPerNode());
    while (!pauli_cover.empty()) {
        int min_swap_nv = this->_n_global_qubits + 1;
        int min_swap_ib = this->_n_global_qubits + 1;
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
        assert(min_swap_nv + min_swap_ib <= this->_n_global_qubits);

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
            assert(min_wire < this->_n_local_qubits);
            assert(max_wire >= this->_n_local_qubits);
            swap_qubits.emplace_back(min_wire, max_wire);
            std::swap(wire[q_current], wire[q_next]);
        }
        assert((int)swap_qubits.size() <= this->_n_global_qubits);
        if ((int)swap_qubits.size() > 0) {
            circuit_elements.push_back(new SwapGlobalQubits<FP>(swap_qubits));
            if (min_swap_ib > 0) {
                n_qr_ib++;
            } else {
                n_qr_nv++;
            }
        }
        current_global = next_global;

        // decompression pauli bits
        PauliString pauli_string(it_min_qr->second);
        int n_pauli = pauli_string.getOperatorsNum();
        sum_pauli += n_pauli;
        for (PauliProduct &pauli_product : pauli_string.getPauliProducts()) {
            int n_basis = (int)pauli_product.getPauliOperator().size();
            int z_mask = 0;  // _n_global_qubits binary bits
            PauliProduct new_pauli_product;
            for (int j = 0; j < n_basis; ++j) {
                int target = wire[pauli_product.getBasisQubit()[j]];
                if (target >= this->_n_local_qubits) {  // global pauli z
                    assert(pauli_product.getPauliOperator()[j] == CUSTATEVEC_PAULI_Z);
                    z_mask |= (1 << (target - this->_n_local_qubits));
                } else {
                    new_pauli_product.addElement(pauli_product.getPauliOperator()[j], target);
                }
            }
            assert((int)new_pauli_product.getPauliOperator().size() <= this->_n_local_qubits);
            assert(z_mask < (1LL << this->_n_global_qubits));
            std::complex<double> new_coef = {sign_pauli_z[z_mask] * pauli_product.getPauliCoef().real(),
                                             sign_pauli_z[z_mask] * pauli_product.getPauliCoef().imag()};
            new_pauli_product.setPauliCoef(new_coef);
            circuit_elements.push_back(new PauliRotation<FP>(
                new_pauli_product, &this->_params[operator_to_paramid[pauli_product.getPauliIndex()]]));
        }
        pauli_cover.erase(it_min_qr);
    }
    assert(sum_pauli == n_operators);
    this->_circuit_elements = circuit_elements;
    if (_ExecuteManager.getMpiRank() == 0) {
        printf("[LOGCPP]PauliExp:n_qr=%d,n_qr_nv=%d,n_qr_ib=%d\n", n_qr_nv + n_qr_ib, n_qr_nv, n_qr_ib);
    }
}
