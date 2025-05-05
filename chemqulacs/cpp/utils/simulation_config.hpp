/**
 * @file simulation_config.hpp
 * @brief Header file for SimulationConfig class
 * @author Yusuke Teranishi
 */
#pragma once

#include <string>
#include <vector>

/**
 * @brief GradientMode enum class
 */
enum class GradientMode {
    CENTRAL,
    FORWARD,
    BACKWARD,
    CHECKPOINT
};
const char* getGradientName(GradientMode mode);
GradientMode convertNameToGradient(std::string name);

/**
 * @brief UpdateQRalgorithm enum class
 */
enum class UpdateQRalgorithm {
    UNORDER,
    TILING,
    TILING_INTERCONNECT
};
const char* getAlgorithmName(UpdateQRalgorithm algo);
UpdateQRalgorithm convertNameToUpdateQRalgorithm(std::string name);

/**
 * @brief ExpectationQRalgorithm enum class
 */
enum class ExpectationQRalgorithm {
    UNORDER,
    OFFLINE,
    DIAGONAL,
    DIAGONAL_INTERCONNECT,
    ALLDOT
};
const char* getAlgorithmName(ExpectationQRalgorithm algo);
ExpectationQRalgorithm convertNameToExpectationQRalgorithm(std::string name);

/**
 * @brief SimulationConfig class
 */
class SimulationConfig {
   private:
    GradientMode _grad_mode;  ///< Gradient mode
    bool _is_checkpoint;      ///< Use checkpoint

    UpdateQRalgorithm _update_algo;    ///< Update QR algorithm
    ExpectationQRalgorithm _exp_algo;  ///< Expectation QR algorithm

    double _skip_param_threshold;  ///< Threshold for skipping parameters

   public:
    /**
     * @brief Constructor
     */
    SimulationConfig() {
        setGradientMode(GradientMode::CENTRAL);
        setCheckpoint(true);
        setUpdateQRalgorithm(UpdateQRalgorithm::TILING_INTERCONNECT);
        setExpectationQRalgorithm(ExpectationQRalgorithm::DIAGONAL_INTERCONNECT);
        setSkipParamThreshold(1e-6);
    }

    void setGradientMode(GradientMode mode) {
        _grad_mode = mode;
    }  ///< Set the gradient mode from the GradientMode enum class
    void setGradientMode(std::string name) {
        setGradientMode(convertNameToGradient(name));
    }                                                                          ///< Set the gradient mode from the name
    GradientMode getGradientMode() { return _grad_mode; }                      ///< Get the gradient mode
    const char* getGradientModeName() { return getGradientName(_grad_mode); }  ///< Get the name of the gradient mode

    void setCheckpoint(bool is_checkpoint) { _is_checkpoint = is_checkpoint; }  ///< Set whether to use checkpoint
    bool getCheckpoint() { return _is_checkpoint; }                             ///< Get whether to use checkpoint

    void setUpdateQRalgorithm(UpdateQRalgorithm update_algo) {
        _update_algo = update_algo;
    }  ///< Set the update QR algorithm from the UpdateQRalgorithm enum class
    void setUpdateQRalgorithm(std::string name) {
        setUpdateQRalgorithm(convertNameToUpdateQRalgorithm(name));
    }                                                                  ///< Set the update QR algorithm from the name
    UpdateQRalgorithm getUpdateQRalgorithm() { return _update_algo; }  ///< Get the update QR algorithm
    const char* getUpdateQRalgorithmName() {
        return getAlgorithmName(_update_algo);
    }  ///< Get the name of the update QR algorithm

    void setExpectationQRalgorithm(ExpectationQRalgorithm exp_algo) {
        _exp_algo = exp_algo;
    }  ///< Set the expectation QR algorithm from the ExpectationQRalgorithm
       ///< enum class
    void setExpectationQRalgorithm(std::string name) {
        setExpectationQRalgorithm(convertNameToExpectationQRalgorithm(name));
    }  ///< Set the expectation QR algorithm from the name
    ExpectationQRalgorithm getExpectationQRalgorithm() { return _exp_algo; }  ///< Get the expectation QR algorithm
    const char* getExpectationQRalgorithmName() {
        return getAlgorithmName(_exp_algo);
    }  ///< Get the name of the expectation QR algorithm

    void setSkipParamThreshold(double threshold) {
        _skip_param_threshold = threshold;
    }                                                                 ///< Set the threshold for skipping parameters
    double getSkipParamThreshold() { return _skip_param_threshold; }  ///< Get the threshold for skipping parameters
};

extern SimulationConfig _SimulationConfig;
