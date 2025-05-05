/**
 * @file simulation_config.cpp
 * @brief Implementation file for SimulationConfig class
 * @author Yusuke Teranishi
 */
#include <utils/simulation_config.hpp>

#include <stdexcept>
#include <string>

/**
 * @brief Get the name of the gradient mode
 * @param[in] mode Gradient mode
 * @return Name of the gradient mode
 */
const char* getGradientName(GradientMode mode) {
    switch (mode) {
        case GradientMode::CENTRAL:
            return "CENTRAL";
        case GradientMode::FORWARD:
            return "FORWARD";
        case GradientMode::BACKWARD:
            return "BACKWARD";
        case GradientMode::CHECKPOINT:
            return "CHECKPOINT";
        default:
            return "";
    }
}

/**
 * @brief Convert the name of the gradient mode to the GradientMode enum class
 * @param[in] name Name of the gradient mode
 * @return GradientMode enum class
 */
GradientMode convertNameToGradient(std::string name) {
    if (name == "CENTRAL") {
        return GradientMode::CENTRAL;
    } else if (name == "FORWARD") {
        return GradientMode::FORWARD;
    } else if (name == "BACKWARD") {
        return GradientMode::BACKWARD;
    } else if (name == "CHECKPOINT") {
        return GradientMode::CHECKPOINT;
    }
    throw std::invalid_argument("unknown GradientModeName");
}

/**
 * @brief Get the name of the update QR algorithm
 * @param[in] algo Update QR algorithm
 * @return Name of the update QR algorithm
 */
const char* getAlgorithmName(UpdateQRalgorithm algo) {
    switch (algo) {
        case UpdateQRalgorithm::UNORDER:
            return "UNORDER";
        case UpdateQRalgorithm::TILING:
            return "TILING";
        case UpdateQRalgorithm::TILING_INTERCONNECT:
            return "TILING_INTERCONNECT";
        default:
            return "";
    }
}

/**
 * @brief Convert the name of the update QR algorithm to the UpdateQRalgorithm
 * enum class
 * @param[in] name Name of the update QR algorithm
 * @return UpdateQRalgorithm enum class
 */
UpdateQRalgorithm convertNameToUpdateQRalgorithm(std::string name) {
    if (name == "UNORDER") {
        return UpdateQRalgorithm::UNORDER;
    } else if (name == "TILING") {
        return UpdateQRalgorithm::TILING;
    } else if (name == "TILING_INTERCONNECT") {
        return UpdateQRalgorithm::TILING_INTERCONNECT;
    }
    throw std::invalid_argument("unknown UpdateQRalgorithmName");
}

/**
 * @brief Get the name of the expectation QR algorithm
 * @param[in] algo Expectation QR algorithm
 * @return Name of the expectation QR algorithm
 */
const char* getAlgorithmName(ExpectationQRalgorithm algo) {
    switch (algo) {
        case ExpectationQRalgorithm::UNORDER:
            return "UNORDER";
        case ExpectationQRalgorithm::OFFLINE:
            return "OFFLINE";
        case ExpectationQRalgorithm::DIAGONAL:
            return "DIAGONAL";
        case ExpectationQRalgorithm::DIAGONAL_INTERCONNECT:
            return "DIAGONAL_INTERCONNECT";
        case ExpectationQRalgorithm::ALLDOT:
            return "ALLDOT";
        default:
            return "";
    }
}

/**
 * @brief Convert the name of the expectation QR algorithm to the
 * ExpectationQRalgorithm enum class
 * @param[in] name Name of the expectation QR algorithm
 * @return ExpectationQRalgorithm enum class
 */
ExpectationQRalgorithm convertNameToExpectationQRalgorithm(std::string name) {
    if (name == "UNORDER") {
        return ExpectationQRalgorithm::UNORDER;
    } else if (name == "OFFLINE") {
        return ExpectationQRalgorithm::OFFLINE;
    } else if (name == "DIAGONAL") {
        return ExpectationQRalgorithm::DIAGONAL;
    } else if (name == "DIAGONAL_INTERCONNECT") {
        return ExpectationQRalgorithm::DIAGONAL_INTERCONNECT;
    } else if (name == "ALLDOT") {
        return ExpectationQRalgorithm::ALLDOT;
    }
    throw std::invalid_argument("unknown ExpectationQRalgorithmName");
}

SimulationConfig _SimulationConfig;
