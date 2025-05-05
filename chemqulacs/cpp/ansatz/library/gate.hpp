/**
 * @file gate.hpp
 * @brief Implementation of the gate classes
 */
#pragma once

#include <circuit/parametric_matrix.hpp>
#include <utils/precision.hpp>

#include <complex>
#include <cstring>
#include <iostream>

/**
 * @brief RX gate
 * @details \f{bmatrix}{
 * \cos(\theta/2) & -i\sin(\theta/2)\\
 * -i\sin(\theta/2) & \cos(\theta/2)\\
 * }\f
 */
class RX_gate : public ParametricMatrix {
   private:
    double _theta;
    double _coef;

   public:
    RX_gate(double theta, double coef) {
        _theta = theta;
        _coef = coef;
        this->_matrix = new cuDoubleComplex[4];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 4);
    }

    ~RX_gate() { delete[] this->_matrix; }

    void updateMatrix() {
        double theta = _theta * _coef;
        double s = sin(theta / 2);
        double c = cos(theta / 2);
        this->_matrix[0].x = c;
        this->_matrix[1].y = -s;
        this->_matrix[2].y = -s;
        this->_matrix[3].x = c;
    }
};

/**
 * @brief RZ gate
 * @details \f{bmatrix}{
 * e^{-i\theta/2} & 0\\
 * 0 & e^{i\theta/2}\\
 * }\f
 */
class RZ_gate : public ParametricMatrix {
   private:
    double _theta;
    double _coef;

   public:
    RZ_gate(double theta, double coef) {
        _theta = theta;
        _coef = coef;
        this->_matrix = new cuDoubleComplex[4];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 4);
    }

    ~RZ_gate() { delete[] this->_matrix; }

    void updateMatrix() {
        double theta = _theta * _coef;
        double s = sin(theta / 2);
        double c = cos(theta / 2);
        this->_matrix[0].x = c;
        this->_matrix[0].y = -s;
        this->_matrix[3].x = c;
        this->_matrix[3].y = s;
    }
};

/**
 * @brief P gate
 * @details \f{bmatrix}{
 * 1 & 0\\
 * 0 & e^{i\theta}\\
 * }\f
 */
class P_gate : public ParametricMatrix {
   private:
    double _theta;
    double _coef;

   public:
    P_gate(double theta, double coef) {
        _theta = theta;
        _coef = coef;
        this->_matrix = new cuDoubleComplex[4];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 4);
    }

    ~P_gate() { delete[] this->_matrix; }

    void updateMatrix() {
        double theta = _theta * _coef;
        double s = sin(theta);
        double c = cos(theta);
        this->_matrix[0].x = 1;
        this->_matrix[0].y = 0;
        this->_matrix[3].x = c;
        this->_matrix[3].y = s;
    }
};

/**
 * @brief H gate
 * @details \f{bmatrix}{
 * \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\
 * \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}\\
 * }\f
 */
class H_gate : public ParametricMatrix {
   public:
    H_gate() {
        this->_matrix = new cuDoubleComplex[4];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 4);
        this->_matrix[0].x = 1.0 / sqrt(2);
        this->_matrix[1].x = 1.0 / sqrt(2);
        this->_matrix[2].x = 1.0 / sqrt(2);
        this->_matrix[3].x = -1.0 / sqrt(2);
    }

    ~H_gate() { delete[] this->_matrix; }

    void updateMatrix() {}
};

/**
 * @brief V gate
 * @details \f{bmatrix}{
 * \frac{1}{\sqrt{2}} & -i\frac{1}{\sqrt{2}}\\
 * -i\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\
 * }\f
 */
class V_gate : public ParametricMatrix {
   private:
    bool _is_dagger;

   public:
    V_gate(bool is_dagger) {
        _is_dagger = is_dagger;
        this->_matrix = new cuDoubleComplex[4];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 4);
        this->_matrix[0].x = 1.0 / sqrt(2);
        this->_matrix[3].x = 1.0 / sqrt(2);
        if (_is_dagger) {
            this->_matrix[1].y = 1.0 / sqrt(2);
            this->_matrix[2].y = 1.0 / sqrt(2);
        } else {
            this->_matrix[1].y = -1.0 / sqrt(2);
            this->_matrix[2].y = -1.0 / sqrt(2);
        }
    }

    ~V_gate() { delete[] this->_matrix; }

    void updateMatrix() {}
};

/**
 * @brief X gate
 * @details \f{bmatrix}{
 * 0 & 1\\
 * 1 & 0\\
 * }\f
 */
class X_gate : public ParametricMatrix {
   public:
    X_gate() {
        this->_matrix = new cuDoubleComplex[4];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 4);
        this->_matrix[1].x = 1.0;
        this->_matrix[2].x = 1.0;
    }

    virtual ~X_gate() { delete[] this->_matrix; }

    void updateMatrix() {}
};

/**
 * @brief Y gate
 * @details \f{bmatrix}{
 * 0 & -i\\
 * i & 0\\
 * }\f
 */
class Y_gate : public ParametricMatrix {
   public:
    Y_gate() {
        this->_matrix = new cuDoubleComplex[4];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 4);
        this->_matrix[1].y = -1.0;
        this->_matrix[2].y = 1.0;
    }

    virtual ~Y_gate() { delete[] this->_matrix; }

    void updateMatrix() {}
};

/**
 * @brief Z_gate
 * @details \f{bmatrix}{
 * 1 & 0\\
 * 0 & -1\\
 * }\f
 */
class Z_gate : public ParametricMatrix {
   public:
    Z_gate() {
        this->_matrix = new cuDoubleComplex[4];
        std::memset(this->_matrix, 0, sizeof(cuDoubleComplex) * 4);
        this->_matrix[0].x = 1.0;
        this->_matrix[3].x = -1.0;
    }

    virtual ~Z_gate() { delete[] this->_matrix; }

    void updateMatrix() {}
};
