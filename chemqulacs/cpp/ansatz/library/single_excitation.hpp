/**
 * @file single_excitation.hpp
 * @brief Implementation of the SingleExcitation class
 */
#pragma once

#include <circuit/parametric_matrix.hpp>
#include <cstring>
#include <utils/precision.hpp>

/**
 * @brief SingleExcitation
 * @details \f{bmatrix}{
 * 1 & 0 & 0 & 0\\
 * 0 & cos(\theta/2) & -sin(\theta/2) & 0\\
 * 0 & sin(\theta/2) & cos(\theta/2) & 0\\
 * 0 & 0 & 0 & 1\\
 * }\f
 */
class SingleExcitation : public ParametricMatrix {
   private:
    double *_theta;

   public:
    SingleExcitation(double *theta) {
        _theta = theta;

        this->_matrix = new cuDoubleComplex[16];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 16);
        for (int i = 0; i < 16; i += 5) {
            this->_matrix[i].x = 1.0;
        }
    }

    ~SingleExcitation() { delete[] this->_matrix; }

    void updateMatrix() {
        double s = sin(*_theta / 2);
        double c = cos(*_theta / 2);
        this->_matrix[5].x = c;
        this->_matrix[6].x = -s;
        this->_matrix[9].x = s;
        this->_matrix[10].x = c;
    }
};
