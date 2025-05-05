#include <ansatz/ansatz.hpp>
#include <ansatz/ansatz_multi.hpp>
#include <ansatz/gatefabric.hpp>
#include <ansatz/pauli_exp.hpp>
#include <circuit/optimize_update_qr.hpp>
#include <expectation/expectation_computer.hpp>
#include <pauli/pauli_string.hpp>
#include <utils/precision.hpp>
#include <utils/simulation_config.hpp>
#include <utils/timer.hpp>

#include <pybind11/complex.h>
// #include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

namespace py = pybind11;

template <typename FP>
void defineModule(py::module_ &m) {
    py::class_<Ansatz<FP>>(m, "Ansatz");

    py::class_<GateFabric<FP>, Ansatz<FP>>(m, "GateFabric")
        .def(py::init<int, int, bool, std::shared_ptr<PauliString>>(), "Constructor", py::arg("n_qubits"),
             py::arg("n_layers"), py::arg("include_pi"), py::arg("pauli_string"))
        .def(py::init<int, int, bool, std::shared_ptr<PauliString>, bool>(), "Constructor", py::arg("n_qubits"),
             py::arg("n_layers"), py::arg("include_pi"), py::arg("pauli_string"), py::arg("is_init"))
        .def("getCircuitElementsSize", &GateFabric<FP>::getCircuitElementsSize, "Get circuit elements size")
        .def("getParamSize", &GateFabric<FP>::getParamSize, "Get params size of parametric circuit")
        .def("updateParams", &GateFabric<FP>::updateParams, "Update param of parametric circuit", py::arg("params"))
        .def("initState", &GateFabric<FP>::initState, "Init state vector by state bit", py::arg("state_bit"))
        .def("updateState", &GateFabric<FP>::updateState, "Update state vector")
        .def("computeExpectation", &GateFabric<FP>::computeExpectation, "Compute expectation from pauli string")
        .def("computeExpectationWithUpdate",
             pybind11::overload_cast<const std::vector<double> &, const long long>(
                 &GateFabric<FP>::computeExpectationWithUpdate),
             "Compute expectation from pauli string", py::arg("param"), py::arg("init_state"))
        .def("computeExpectationWithUpdate",
             pybind11::overload_cast<const int, const double, const long long>(
                 &GateFabric<FP>::computeExpectationWithUpdate),
             "Compute expectation from pauli string", py::arg("param_id"), py::arg("param"), py::arg("init_state"));

    py::class_<PauliExp<FP>, Ansatz<FP>>(m, "PauliExp")
        .def(py::init<int, std::shared_ptr<PauliString>, const std::vector<int> &, std::shared_ptr<PauliString>>(),
             "Constructor", py::arg("n_qubits"), py::arg("circuit"), py::arg("param_ranges"), py::arg("hamiltonian"))
        .def(
            py::init<int, std::shared_ptr<PauliString>, const std::vector<int> &, std::shared_ptr<PauliString>, bool>(),
            "Constructor", py::arg("n_qubits"), py::arg("circuit"), py::arg("param_ranges"), py::arg("hamiltonian"),
            py::arg("is_init"))
        .def("getCircuitElementsSize", &PauliExp<FP>::getCircuitElementsSize, "Get circuit elements size")
        .def("getParamSize", &PauliExp<FP>::getParamSize, "Get params size of parametric circuit")
        .def("updateParams", &PauliExp<FP>::updateParams, "Update param of parametric circuit", py::arg("params"))
        .def("initState", &PauliExp<FP>::initState, "Init state vector by state bit", py::arg("state_bit"))
        .def("updateState", &PauliExp<FP>::updateState, "Update state vector")
        .def("computeExpectation", &PauliExp<FP>::computeExpectation, "Compute expectation from pauli string")
        .def("computeExpectationWithUpdate",
             pybind11::overload_cast<const std::vector<double> &, const long long>(
                 &PauliExp<FP>::computeExpectationWithUpdate),
             "Compute expectation from pauli string", py::arg("param"), py::arg("init_state"))
        .def("computeExpectationWithUpdate",
             pybind11::overload_cast<const int, const double, const long long>(
                 &PauliExp<FP>::computeExpectationWithUpdate),
             "Compute expectation from pauli string", py::arg("param_id"), py::arg("param"), py::arg("init_state"));

    py::class_<MultiAnsatz<FP>>(m, "MultiAnsatz")
        .def(py::init<GateFabric<FP> *>(), "Constructor", py::arg("ansatz"))
        .def(py::init<PauliExp<FP> *>(), "Constructor", py::arg("ansatz"))
        .def(py::init<GateFabric<FP> *, int>(), "Constructor", py::arg("ansatz"), py::arg("n_compute_unit"))
        .def(py::init<PauliExp<FP> *, int>(), "Constructor", py::arg("ansatz"), py::arg("n_compute_unit"))
        .def("numericalGrad", &MultiAnsatz<FP>::numericalGrad, "Get numerical gradient", py::arg("params"),
             py::arg("dx"), py::arg("init_state"))
        .def("parameterShift", &MultiAnsatz<FP>::parameterShift, "Get numerical gradient by parameter shift method",
             py::arg("params"), py::arg("init_state"))
        .def("computeNFT", &MultiAnsatz<FP>::computeNFT, "Compute NFT params", py::arg("params"), py::arg("init_state"))
        .def("computeExpectation", &MultiAnsatz<FP>::computeExpectation, "Compute expectation on single unit",
             py::arg("params"), py::arg("dx"), py::arg("init_state"));
}

PYBIND11_MODULE(chemqulacs_cpp, m) {
    m.doc() = "chemqulacs c++ interface";
    defineModule<double>(m);
    // defineModule<float>(m);

    py::class_<PauliString, std::shared_ptr<PauliString>>(m, "PauliString")
        .def(py::init<const int, const std::vector<std::vector<int>> &, const std::vector<std::vector<int>> &,
                      const std::vector<std::complex<double>> &>(),
             "Constructor", py::arg("n_qubits"), py::arg("pauli_operators"), py::arg("basis_qubits"),
             py::arg("pauli_coefs"));

    m.def("setGradientMode", [](const std::string name) { _SimulationConfig.setGradientMode(name); });
    m.def("setCheckPoint", [](const bool is_checkpoint) { _SimulationConfig.setCheckpoint(is_checkpoint); });
    m.def("setUpdateQRalgorithm", [](const std::string name) { _SimulationConfig.setUpdateQRalgorithm(name); });
    m.def("setExpectationQRalgorithm",
          [](const std::string name) { _SimulationConfig.setExpectationQRalgorithm(name); });
    m.def("setSkipParamThreshold", [](const double threshold) { _SimulationConfig.setSkipParamThreshold(threshold); });
    m.def("getTimeDict", []() { return _TimerDict.getTimeDict(); });
}
