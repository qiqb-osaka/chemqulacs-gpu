/**
 * @file double_excitation.hpp
 * @brief Implementation of the DoubleExcitation class
 */
#pragma once

#include <circuit/parametric_matrix.hpp>
#include <cstring>
#include <utils/precision.hpp>

/**
 * @brief Double excitation gate
 */
class DoubleExcitation : public ParametricMatrix {
   private:
    double *_theta;

   public:
    DoubleExcitation(double *theta) {
        _theta = theta;

        this->_matrix = new cuDoubleComplex[256];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 256);
        for (int i = 0; i < 256; i += 17) {
            this->_matrix[i].x = 1.0;
        }
    }

    ~DoubleExcitation() { delete[] this->_matrix; }

    void updateMatrix() {
        double s = sin(*_theta / 2);
        double c = cos(*_theta / 2);
        this->_matrix[16 * 3 + 3].x = c;
        this->_matrix[16 * 3 + 12].x = -s;
        this->_matrix[16 * 12 + 3].x = s;
        this->_matrix[16 * 12 + 12].x = c;
    }
};
