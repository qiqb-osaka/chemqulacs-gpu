/**
 * @file precision.hpp
 * @brief Header file for precision utilities
 */
#pragma once

#include <cuComplex.h>
#include <custatevec.h>

/**
 * @brief Selects the appropriate cuFpComplex type based on the input type.
 * cuDoubleComplex is selected for double, and cuFloatComplex is selected for
 * float.
 */
template <typename FP>
using cuFpComplex = typename std::conditional<std::is_same<FP, double>::value, cuDoubleComplex, cuFloatComplex>::type;

/**
 * @brief Selects the appropriate CUDA data type based on the input type.
 * CUDA_C_64F is selected for double, and CUDA_C_32F is selected for float.
 */
template <typename FP>
constexpr cudaDataType_t CUDA_C_ = (std::is_same<FP, double>::value ? CUDA_C_64F : CUDA_C_32F);

/**
 * @brief Selects the appropriate cuStateVecComputeType_t based on the input
 * type. CUSTATEVEC_COMPUTE_64F is selected for double, and
 * CUSTATEVEC_COMPUTE_32F is selected for float.
 */
template <typename FP>
constexpr custatevecComputeType_t CUSTATEVEC_COMPUTE_ =
    (std::is_same<FP, double>::value ? CUSTATEVEC_COMPUTE_64F : CUSTATEVEC_COMPUTE_32F);

/**
 * @brief Makes a cuFpComplex object with the given real and imaginary parts.
 * @param[in] x Real part
 * @param[in] y Imaginary part
 * @return cuFpComplex object
 */
template <typename FP>
cuFpComplex<FP> make_cuFpComplex(FP x, FP y) {
    cuFpComplex<FP> c;
    c.x = x;
    c.y = y;
    return c;
}
