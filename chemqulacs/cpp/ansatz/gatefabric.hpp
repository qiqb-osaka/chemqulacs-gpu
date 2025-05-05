/**
 * @file gatefabric.hpp
 * @brief Header of the GateFabric class
 * @details This file, originally from PennyLane's
 * pennylane.templates.subroutines.uccsd, has been modified for chemqulacs-gpu
 * by Yusuke Teranishi
 *
 *          Copyright 2018-2021 Xanadu Quantum Technologies Inc.
 *
 *          Licensed under the Apache License, Version 2.0 (the "License");
 *          you may not use this file except in compliance with the License.
 *          You may obtain a copy of the License at
 *
 *              http://www.apache.org/licenses/LICENSE-2.0
 *
 *          Unless required by applicable law or agreed to in writing, software
 *          distributed under the License is distributed on an "AS IS" BASIS,
 *          WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 *          limitations under the License.
 */
#pragma once
#include <ansatz/ansatz.hpp>
#include <pauli/pauli_string.hpp>
#include <utils/precision.hpp>

#include <mpi.h>

/**
 * @brief Class for the ansatz using the gate fabric
 * @details This class is for the ansatz using the gate fabric.
 */
template <typename FP = double>
class GateFabric : public Ansatz<FP>
{
private:
    int _n_layers;    ///< Number of layers
    bool _include_pi; ///< Include Pi flag

    // Constant value
    double *_PI; ///< Pi

    void buildCircuit(); ///< Build the circuit

    void _fourQubitGate(std::vector<int> &wires, int &param_id);   ///< Add a four-qubit gate
    void _orbitalRotation(std::vector<int> &wires, double *phi);   ///< Perform orbital rotation
    void _doubleExcitation(std::vector<int> wires, double *theta); ///< Perform a double excitation
    void _singleExcitation(std::vector<int> wires, double *theta); ///< Perform a single excitation

public:
    /**
     * @brief Constructor
     * @param n_qubits Number of qubits
     * @param n_layers Number of layers
     * @param include_pi Include Pi flag
     * @param pauli_string Pauli string
     * @param mpi_comm Type of MPI communication
     * @param is_init Flag for initialization
     */
    GateFabric(int n_qubits, int n_layers, bool include_pi, std::shared_ptr<PauliString> pauli_string,
               MPI_Comm mpi_comm, bool is_init = true)
        : Ansatz<FP>(n_qubits, pauli_string, mpi_comm, is_init)
    {
        _n_layers = n_layers;
        _include_pi = include_pi;
        _PI = new double(acos(-1));
        buildCircuit();

        if (is_init)
        {
            this->buildQR();
        }
    }

    /**
     * @brief Constructor
     * @param n_qubits Number of qubits
     * @param n_layers Number of layers
     * @param include_pi Include Pi flag
     * @param pauli_string Pauli string
     * @param is_init Flag for initialization
     */
    GateFabric(int n_qubits, int n_layers, bool include_pi, std::shared_ptr<PauliString> pauli_string,
               bool is_init = true)
        : GateFabric<FP>(n_qubits, n_layers, include_pi, pauli_string, MPI_COMM_WORLD, is_init) {}

    /**
     * @brief Constructor
     * @param gf GateFabric pointer
     * @param mpi_comm Type of MPI communication
     */
    GateFabric(GateFabric<FP> *gf, MPI_Comm mpi_comm)
        : GateFabric<FP>(gf->getQubits(), gf->_n_layers, gf->_include_pi, gf->getPauliStringSharedPtr(), mpi_comm) {}

    ~GateFabric() { delete _PI; }
};
