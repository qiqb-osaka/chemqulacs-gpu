/**
 * @file optimize_update_qr.hpp
 * @brief Header file for OptimizeUpdateQR class
 * @author Yusuke Teranishi
 */
#pragma once

#include <circuit/circuit_element.hpp>
#include <utils/execute_manager.hpp>
#include <utils/simulation_config.hpp>
#include <utils/timer.hpp>

/**
 * @brief OptimizeUpdateQR class
 */
template <typename FP>
class OptimizeUpdateQR {
   private:
    int _n_qubits;
    int _n_local_qubits;
    int _n_global_qubits;
    bool _is_takeover_wire;
    UpdateQRalgorithm _algo;

    void optimizeUnorder(std::vector<CircuitElement<FP> *> &circuit_elements, std::vector<int> &wire);
    void optimizeTimeSpaceTiling(std::vector<CircuitElement<FP> *> &circuit_elements, std::vector<int> &wire);
    void optimizeTimeSpaceTilingAwareInterconnect(std::vector<CircuitElement<FP> *> &circuit_elements,
                                                  std::vector<int> &wire);

    /**
     * @brief Log information of the QR update
     * @param circuit_elements List of circuit elements
     */
    void _logQRinfo(const std::vector<CircuitElement<FP> *> &circuit_elements) {
        if (_ExecuteManager.getMpiRank() != 0) {
            return;
        }
        int n_qr = 0;
        for (CircuitElement<FP> *element : circuit_elements) {
            CircuitElementType type = element->getType();
            if (type == CircuitElementType::SWAP_GLOBAL_QUBITS) {
                n_qr++;
            }
        }
        printf(
            "[LOGCPP]Update:n_qubits=%d,n_devices=%d,is_takeover_wire=%d,qr_"
            "algo=%s,n_apply=%d,n_qr=%d\n",
            _n_qubits, 1 << _n_global_qubits, (int)_is_takeover_wire, getAlgorithmName(_algo),
            (int)circuit_elements.size() - n_qr, n_qr);
    }

   public:
    OptimizeUpdateQR(int n_local_qubits, int n_global_qubits, bool is_takeover_wire,
                     UpdateQRalgorithm algo = UpdateQRalgorithm::TILING)
        : _n_qubits(n_local_qubits + n_global_qubits),
          _n_local_qubits(n_local_qubits),
          _n_global_qubits(n_global_qubits),
          _is_takeover_wire(is_takeover_wire),
          _algo(algo) {}

    void optimize(std::vector<CircuitElement<FP> *> &circuit_elements, std::vector<int> &wire) {
        Timer *_timer_update_optimize = _TimerDict.addTimerHost("update_optimize");
        if (_n_global_qubits == 0) {
            _logQRinfo(circuit_elements);
            return;
        }
        _timer_update_optimize->start();
        switch (_algo) {
            case UpdateQRalgorithm::UNORDER:
                optimizeUnorder(circuit_elements, wire);
                break;
            case UpdateQRalgorithm::TILING:
                optimizeTimeSpaceTiling(circuit_elements, wire);
                break;
            case UpdateQRalgorithm::TILING_INTERCONNECT:
                optimizeTimeSpaceTilingAwareInterconnect(circuit_elements, wire);
                break;
            default:
                throw std::invalid_argument("unknown UpdateQRalgorithm");
        }
        if (!_is_takeover_wire) {
            reorderingStarndardQubits(circuit_elements, wire);
        }
        _timer_update_optimize->stop();
        _logQRinfo(circuit_elements);
    }
    void reorderingStarndardQubits(std::vector<CircuitElement<FP> *> &circuit_elements, std::vector<int> &start_wire,
                                   std::vector<int> goal_wire = std::vector<int>());
};
