/**
 * @file compute_dotproduct.hpp
 * @brief Header file for ComputeDotProduct class
 * @author Yusuke Teranishi
 */
#pragma once

#include <ansatz/library/gate.hpp>
#include <circuit/parametric_matrix.hpp>
#include <expectation/expectation_element.hpp>
#include <state/state_vector.hpp>
#include <utils/execute_manager.hpp>
#include <utils/precision.hpp>

#include <custatevec.h>

#include <cassert>
#include <stdexcept>
#include <vector>

/**
 * @brief ComputeDotProduct class
 */
template <typename FP>
class ComputeDotProduct : public ExpectationElement<FP> {
   private:
    std::vector<ParametricMatrix *> _local_gates;
    std::vector<int> _local_targets;
    std::vector<ParametricMatrix *> _global_gates;
    std::vector<int> _global_targets;

    StateVector<FP> *_state_vector_work;

    double _pauli_coef;

    X_gate *x_gate;
    Y_gate *y_gate;
    Z_gate *z_gate;
    // 上：ConstantMatrixにしてグローバルに確保したい

   public:
    ComputeDotProduct(const PauliProduct &pauli_product, StateVector<FP> *state_vector_work)
        : ExpectationElement<FP>(ExpectationElementType::DOT_PRODUCT) {
        const std::vector<custatevecPauli_t> &paulis = pauli_product.getPauliOperator();
        const std::vector<int> &basis_qubits = pauli_product.getBasisQubit();
        const double pauli_coef = pauli_product.getPauliCoefReal();
        int n_paulis = (int)paulis.size();

        _pauli_coef = pauli_coef;
        _state_vector_work = state_vector_work;
        int n_local_qubits = state_vector_work->getLocalQubits();

        ParametricMatrix *pauli_gate;
        x_gate = new X_gate();
        y_gate = new Y_gate();
        z_gate = new Z_gate();

        for (int i = 0; i < n_paulis; ++i) {
            if (paulis[i] == CUSTATEVEC_PAULI_X) {
                pauli_gate = x_gate;
            } else if (paulis[i] == CUSTATEVEC_PAULI_Y) {
                pauli_gate = y_gate;
            } else if (paulis[i] == CUSTATEVEC_PAULI_Z) {
                pauli_gate = z_gate;
            } else {
                throw std::invalid_argument("unknown pauli type");
            }
            if (basis_qubits[i] < n_local_qubits) {
                _local_gates.push_back(pauli_gate);
                _local_targets.push_back(basis_qubits[i]);
            } else {
                _global_gates.push_back(pauli_gate);
                _global_targets.push_back(basis_qubits[i] - n_local_qubits);
            }
        }
    }

    ~ComputeDotProduct() {
        delete x_gate;
        delete y_gate;
        delete z_gate;
    }

    double computeExpectation(StateVector<FP> *state_vector) {
        _state_vector_work->copyStateFrom(state_vector);
        for (int i = 0; i < (int)_local_gates.size(); ++i) {
            HANDLE_CUSV(custatevecApplyMatrix(_ExecuteManager.getCusvHandle(), _state_vector_work->getDevicePtr(),
                                              CUDA_C_<FP>, _state_vector_work->getLocalQubits(),
                                              _local_gates[i]->getMatrix(), CUDA_C_<FP>, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                                              0, &_local_targets[i], 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_<FP>,
                                              nullptr, 0));
        }
        if (_global_gates.size() > 0) {
            _state_vector_work->swapGlobalQubits();
            for (int i = 0; i < (int)_global_gates.size(); ++i) {
                HANDLE_CUSV(custatevecApplyMatrix(_ExecuteManager.getCusvHandle(), _state_vector_work->getDevicePtr(),
                                                  CUDA_C_<FP>, _state_vector_work->getLocalQubits(),
                                                  _global_gates[i]->getMatrix(), CUDA_C_<FP>,
                                                  CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, &_global_targets[i], 1, nullptr,
                                                  nullptr, 0, CUSTATEVEC_COMPUTE_<FP>, nullptr, 0));
            }
            _state_vector_work->swapGlobalQubits();
        }
        cuFpComplex<FP> dot_val = state_vector->dotProduct(_state_vector_work);
        double exp = _pauli_coef * dot_val.x;
        return exp;
    }
};
