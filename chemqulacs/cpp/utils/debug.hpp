#pragma once

#include <complex>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <custatevec.h>
#include <mpi.h>
#include <circuit/circuit_element.hpp>
#include <pauli/pauli_product.hpp>

class Debug {
   private:
    int _rank = 0;
    bool _is_print = false;

    bool _check() const { return (_rank == 0) && _is_print; }

    template <typename T>
    void _printData(const T& data, char separator) const {
        std::cout << data << separator;
    }

    void _printData(const PauliProduct& pp, char separator) const {
        std::cout << "\ncoeff: " << pp.getPauliCoef().real() << " " << pp.getPauliCoef().imag() << "\n";
        for (size_t i = 0; i < pp.getBasisQubit().size(); ++i) {
            char op = 'I';
            switch (pp.getPauliOperator()[i]) {
                case 1:
                    op = 'X';
                    break;
                case 2:
                    op = 'Y';
                    break;
                case 3:
                    op = 'Z';
                    break;
                default:
                    break;
            }
            std::cout << op << pp.getBasisQubit()[i] << separator;
        }
    }

    void _printData(const std::vector<std::vector<int>>& vec, char separator) const {
        for (const auto& v : vec) {
            for (const auto& vv : v) {
                std::cout << vv << separator;
            }
        }
    }

    void _printData(const std::vector<int2>& vec, char separator) const {
        for (const auto& v : vec) {
            std::cout << v.x << " " << v.y << separator;
        }
    }

    void _printData(const std::vector<std::complex<double>>& vec, char separator) const {
        for (const auto& v : vec) {
            std::cout << v.real() << " " << v.imag() << separator;
        }
    }

    void _printData(const std::vector<double>& vec, char separator) const {
        for (const auto& v : vec) {
            std::cout << v << separator;
        }
    }

    void _printData(const std::vector<int>& vec, char separator) const {
        for (const auto& v : vec) {
            std::cout << v << separator;
        }
    }

   public:
    Debug() = default;

    explicit Debug(int rank) : _rank(rank) {}

    template <typename T>
    void print(const std::string& msg, const T& data, char separator = ' ') const {
        if (!_check()) return;
        std::cout << msg;
        _printData(data, separator);
        std::cout << std::endl;
    }

    void print(const std::string& msg) const {
        if (!_check()) return;
        std::cout << msg << std::endl;
    }

    void setPrint(bool is_print) { _is_print = is_print; }
};
