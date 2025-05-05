/**
 * @file pauli_exp.hpp
 * @brief Header file for PauliExp class
 * @author Yusuke Teranishi
 */
#pragma once

#include <ansatz/ansatz.hpp>
#include <pauli/pauli_string.hpp>
#include <utils/precision.hpp>

#include <mpi.h>

#include <numeric>

/**
 * @brief Class for the ansatz of the PauliExp
 * @tparam FP floating point type
 */
template <typename FP = double>
class PauliExp : public Ansatz<FP> {
   private:
    std::shared_ptr<PauliString> _pauli_string;
    std::vector<int> _param_ranges;

    void buildCircuit();

    void _buildPauliRotationsQR(std::vector<int> &wire);

    // Future work: refactoring with <circuit/optimize_update_qr.hpp>
    void _logQRinfo() {
        if (_ExecuteManager.getMpiRank() != 0) {
            return;
        }
        int n_qr = 0;
        for (CircuitElement<FP> *element : this->_circuit_elements) {
            CircuitElementType type = element->getType();
            if (type == CircuitElementType::SWAP_GLOBAL_QUBITS) {
                n_qr++;
            }
        }
        printf(
            "[LOGCPP]Update:n_qubits=%d,n_devices=%d,qr_"
            "algo=%s,n_apply=%d,n_qr=%d\n",
            this->_n_qubits, this->_n_devices, "PAULI_ROTATION", (int)this->_circuit_elements.size() - n_qr, n_qr);
    }

    /**
     * @brief Calculate QR order
     */
    void _build() {
        std::vector<int> wire(this->_n_qubits);
        std::iota(wire.begin(), wire.end(), 0);
        if (this->_n_devices > 1) {
            Timer *timer = _TimerDict.addTimerHost("buildPauliRotationsQR");
            timer->start();
            _buildPauliRotationsQR(wire);
            timer->stop();
        }
        _logQRinfo();
        this->buildExpectationQR(wire, _SimulationConfig.getExpectationQRalgorithm());
    }

   public:
    /**
     * @brief Constructor
     * @param n_qubits Number of qubits
     * @param pauli_string Pauli string
     * @param param_ranges Parameter ranges
     * @param hamiltonian Hamiltonian
     * @param mpi_comm Type of MPI communication
     * @param is_init Flag for initialization
     */
    PauliExp(int n_qubits, std::shared_ptr<PauliString> pauli_string, const std::vector<int> &param_ranges,
             std::shared_ptr<PauliString> hamiltonian, MPI_Comm mpi_comm, bool is_init = true)
        : Ansatz<FP>(n_qubits, hamiltonian, mpi_comm, is_init) {
        _pauli_string = pauli_string;
        _param_ranges = param_ranges;
        buildCircuit();
        if (is_init) {
            _build();
        }
    }

    PauliExp(int n_qubits, std::shared_ptr<PauliString> pauli_string, const std::vector<int> &param_ranges,
             std::shared_ptr<PauliString> hamiltonian, bool is_init = true)
        : PauliExp<FP>(n_qubits, pauli_string, param_ranges, hamiltonian, MPI_COMM_WORLD, is_init) {}

    PauliExp(PauliExp<FP> *PauliExp, MPI_Comm mpi_comm)
        : PauliExp<FP>(PauliExp->getQubits(), PauliExp->_pauli_string, PauliExp->_param_ranges,
                       PauliExp->getPauliStringSharedPtr(), mpi_comm) {}

    ~PauliExp() {}
};
