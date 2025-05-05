/**
 * @file apply_matrix.hpp
 * @brief Header file for ApplyMatrix class
 * @author Yusuke Teranishi
 */
#pragma once

#include <circuit/circuit_element.hpp>
#include <circuit/parametric_matrix.hpp>
#include <utils/execute_manager.hpp>
#include <utils/precision.hpp>

#include <custatevec.h>
#include <nvtx3/nvToolsExt.h>

#include <iostream>
#include <vector>

/**
 * @brief ApplyMatrix class(Sub class of CircuitElement)
 */
template <typename FP>
class ApplyMatrix : public CircuitElement<FP> {
   private:
    ParametricMatrix *_parametric_matrix;

   public:
    /**
     * @brief Constructor
     * @param parametric_matrix ParametricMatrix object
     * @param controls List of control qubits
     * @param targets List of target qubits
     */
    ApplyMatrix(ParametricMatrix *parametric_matrix, std::vector<int> controls, std::vector<int> targets)
        : CircuitElement<FP>(CircuitElementType::APPLY_MATRIX, controls, targets) {
        _parametric_matrix = parametric_matrix;
    }
    ApplyMatrix(ParametricMatrix *parametric_matrix, std::vector<int> targets)
        : ApplyMatrix(parametric_matrix, std::vector<int>(), targets) {}

    /**
     * @brief apply updateMatrix method of ParametricMatrix
     */
    void updateCircuit() { _parametric_matrix->updateMatrix(); }

    /**
     * @brief apply custatevecApplyMatrix method to update state vector
     * @param state_vector StateVector object
     */
    void updateState(StateVector<FP> *state_vector) {
        nvtxRangePush("updateStateUnit");
        HANDLE_CUSV(custatevecApplyMatrix(_ExecuteManager.getCusvHandle(), state_vector->getDevicePtr(), CUDA_C_<FP>,
                                          state_vector->getLocalQubits(), _parametric_matrix->getMatrix(), CUDA_C_64F,
                                          CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, this->_targets.data(), this->_targets.size(),
                                          this->_controls.data(), nullptr, this->_controls.size(),
                                          CUSTATEVEC_COMPUTE_<FP>, nullptr, 0));
        nvtxRangePop();
    }

    void print() {
        std::cout << "ApplyMatrix" << std::endl;
        std::cout << "Controls: ";
        for (auto control : this->_controls) std::cout << control << " ";
        std::cout << std::endl;
        std::cout << "Targets: ";
        for (auto target : this->_targets) std::cout << target << " ";
        std::cout << std::endl;
    }

    virtual CircuitElement<FP> *clone() const {
        return new ApplyMatrix<FP>(_parametric_matrix, this->_controls, this->_targets);
    }
};
