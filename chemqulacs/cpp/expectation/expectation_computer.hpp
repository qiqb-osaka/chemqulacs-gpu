/**
 * @file expectation_computer.hpp
 * @brief Header file for ExpectationComputer class
 * @author Yusuke Teranishi
 */
#pragma once

#include <expectation/compute_paulibasis.hpp>
#include <expectation/expectation_element.hpp>
#include <pauli/pauli_string.hpp>
#include <state/state_vector.hpp>
#include <utils/execute_manager.hpp>
#include <utils/simulation_config.hpp>
#include <utils/timer.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

/**
 * @brief ExpectationComputer class
 */
template <typename FP>
class ExpectationComputer {
   private:
    int _n_qubits;           ///< Number of qubits
    int _n_local_qubits;     ///< Number of local qubits
    int _n_global_qubits;    ///< Number of global qubits
    int _sv_len_per_device;  ///< Length of state vector per device
    int _n_devices;          ///< Number of devices

    StateVector<FP> *_state_vector;                       ///< State vector
    std::unique_ptr<StateVector<FP>> _state_vector_work;  ///< State vector work

    PauliString *_pauli_string;                                   ///< Pauli string
    std::vector<ExpectationElement<FP> *> _expectation_elements;  ///< Expectation elements

    Timer *_timer_exp_reduce;

    void buildExpectationElementsUnorder(std::vector<int> &wire);
    void buildExpectationElementsOffline(std::vector<int> &wire);
    void buildExpectationElementsDiagonalZ(std::vector<int> &wire);
    void buildExpectationElementsDiagonalzGreedy(std::vector<int> &wire);
    void buildExpectationElementsAllDot(std::vector<int> &wire);

    void _recursiveCombinationBits(int n, int m, std::vector<long long> &comb_bits, std::vector<int> &comb_work,
                                   int &comb_idx, int depth, int i_n);
    std::vector<long long> _getCombinationBits(int n_bits, int one_bits);

    void _recursiveCombinationBitsNew(int n, int m, std::vector<long long> &comb_bits, int &comb_idx, long long bits,
                                      int depth, int i_n);
    std::vector<long long> _getCombinationBitsNew(int n_bits, int one_bits);

    void _getTreePauliIndexes(std::vector<int> &pauli_indexes,
                              std::unordered_map<long long, std::vector<long long>> &bits_tree,
                              std::unordered_map<long long, std::vector<int>> &bits_to_idx, long long current_bits);
    void _getTreePauliIndexesNew(std::vector<int> &pauli_indexes, const std::vector<std::vector<int>> &pauli_child,
                                 int parent);

    /**
     * @brief Log expectation information
     * @param[in] algo ExpectationQRalgorithm
     */
    void _logExpectationInfo(ExpectationQRalgorithm algo) {
        if (_ExecuteManager.getMpiRank() != 0) {
            return;
        }
        int n_compute_pauli = 0;
        int n_compute_dot = 0;
        int n_qr = 0;
        for (ExpectationElement<FP> *element : _expectation_elements) {
            ExpectationElementType type = element->getType();
            if (type == ExpectationElementType::PAULI_BASIS) {
                n_compute_pauli++;
            } else if (type == ExpectationElementType::DOT_PRODUCT) {
                n_compute_dot++;
            } else if (type == ExpectationElementType::QUBIT_REORDERING) {
                n_qr++;
            }
        }
        printf(
            "[LOGCPP]EXP:n_qubits=%d,n_devices=%d,n_pauli=%d,exp_algo=%s,n_"
            "compute_pauli=%d,n_compute_dot=%d,n_qr=%d\n",
            _n_qubits, _n_devices, (int)_pauli_string->getOperatorsNum(), getAlgorithmName(algo), n_compute_pauli,
            n_compute_dot, n_qr);
    }

   public:
    /**
     * @brief Constructor
     * @param[in] state_vector StateVector object
     * @param[in] pauli_string PauliString object
     */
    ExpectationComputer(StateVector<FP> *state_vector, PauliString *pauli_string) {
        _state_vector = state_vector;
        _pauli_string = pauli_string;
        _n_qubits = _state_vector->getQubits();
        _n_local_qubits = _state_vector->getLocalQubits();
        _n_global_qubits = _state_vector->getGlobalQubits();
        _sv_len_per_device = _state_vector->getLenPerDevice();
        _n_devices = _state_vector->getDevices();
        _timer_exp_reduce = _TimerDict.addTimerHost("exp_reduce");
    }

    ~ExpectationComputer() {
        for (ExpectationElement<FP> *ptr : _expectation_elements) {
            delete ptr;
        }
    }

    /**
     * @brief Build expectation elements
     * @param[in] wire Wire for the QR
     * @param[in] algo ExpectationQRalgorithm
     */
    void build(std::vector<int> &wire, ExpectationQRalgorithm algo = ExpectationQRalgorithm::DIAGONAL) {
        Timer *_timer_exp_optimize = _TimerDict.addTimerHost("exp_optimize");
        if (_n_devices == 1) {
            _expectation_elements.push_back(new ComputePauliBasis<FP>(_pauli_string));
            _logExpectationInfo(algo);
            return;
        }
        _timer_exp_optimize->start();
        switch (algo) {
            case ExpectationQRalgorithm::UNORDER:
                buildExpectationElementsUnorder(wire);
                break;
            case ExpectationQRalgorithm::OFFLINE:
                buildExpectationElementsOffline(wire);
                break;
            case ExpectationQRalgorithm::DIAGONAL:
                buildExpectationElementsDiagonalZ(wire);
                break;
            case ExpectationQRalgorithm::DIAGONAL_INTERCONNECT:
                buildExpectationElementsDiagonalzGreedy(wire);
                break;
            case ExpectationQRalgorithm::ALLDOT:
                buildExpectationElementsAllDot(wire);
                break;
            default:
                throw std::invalid_argument("unknown ExpectationQRalgorithm");
        }
        _timer_exp_optimize->stop();
        _logExpectationInfo(algo);
    }

    double computeExpectation();
};
