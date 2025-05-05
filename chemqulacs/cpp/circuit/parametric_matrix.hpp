/**
 * @file parametric_matrix.hpp
 * @brief Header file for ParametricMatrix class
 * @author Yusuke Teranishi
 */
#pragma once

#include <utils/precision.hpp>

/**
 * @brief ParametricMatrix class(cuDoubleComplex list)
 */
class ParametricMatrix {
   protected:
    cuDoubleComplex *_matrix;

   public:
    cuDoubleComplex *getMatrix() { return _matrix; }

    virtual void updateMatrix() = 0;
};
