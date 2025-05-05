#include <ansatz/library/gate.hpp>
#include <circuit/apply_matrix.hpp>
#include <circuit/circuit_element.hpp>
#include <state/state_vector.hpp>

#include <vector>

void addCNOTGate(int control, int target, std::vector<CircuitElement<double> *> &circuit_elements) {
    circuit_elements.emplace_back(new ApplyMatrix<double>(new X_gate(), {control}, {target}));
}

void addRXGate(int target, double theta, double coef, std::vector<CircuitElement<double> *> &circuit_elements) {
    circuit_elements.emplace_back(new ApplyMatrix<double>(new RX_gate(theta, coef), {target}));
}

void addRZGate(int target, double theta, double coef, std::vector<CircuitElement<double> *> &circuit_elements) {
    circuit_elements.emplace_back(new ApplyMatrix<double>(new RZ_gate(theta, coef), {target}));
}

void addHGate(int target, std::vector<CircuitElement<double> *> &circuit_elements) {
    circuit_elements.emplace_back(new ApplyMatrix<double>(new H_gate(), {target}));
}