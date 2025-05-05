"""
chemqulacs c++ interface
"""

from __future__ import annotations

import typing

__all__ = [
    "Ansatz",
    "GateFabric",
    "MultiAnsatz",
    "PauliExp",
    "PauliString",
    "getTimeDict",
    "setCheckPoint",
    "setExpectationQRalgorithm",
    "setGradientMode",
    "setSkipParamThreshold",
    "setUpdateQRalgorithm",
]

class Ansatz:
    pass

class GateFabric(Ansatz):
    @typing.overload
    def __init__(
        self, n_qubits: int, n_layers: int, include_pi: bool, pauli_string: PauliString
    ) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        include_pi: bool,
        pauli_string: PauliString,
        is_init: bool,
    ) -> None:
        """
        Constructor
        """
    def computeExpectation(self) -> float:
        """
        Compute expectation from pauli string
        """
    @typing.overload
    def computeExpectationWithUpdate(
        self, param: list[float], init_state: int
    ) -> float:
        """
        Compute expectation from pauli string
        """
    @typing.overload
    def computeExpectationWithUpdate(
        self, param_id: int, param: float, init_state: int
    ) -> float:
        """
        Compute expectation from pauli string
        """
    def getCircuitElementsSize(self) -> int:
        """
        Get circuit elements size
        """
    def getParamSize(self) -> int:
        """
        Get params size of parametric circuit
        """
    def initState(self, state_bit: int) -> None:
        """
        Init state vector by state bit
        """
    def updateParams(self, params: list[float]) -> None:
        """
        Update param of parametric circuit
        """
    def updateState(self) -> None:
        """
        Update state vector
        """

class MultiAnsatz:
    @typing.overload
    def __init__(self, ansatz: GateFabric) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, ansatz: PauliExp) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, ansatz: GateFabric, n_compute_unit: int) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, ansatz: PauliExp, n_compute_unit: int) -> None:
        """
        Constructor
        """
    def computeExpectation(
        self, params: list[float], dx: float, init_state: int
    ) -> float:
        """
        Compute expectation on single unit
        """
    def computeNFT(self, params: list[float], init_state: int) -> list[float]:
        """
        Compute NFT params
        """
    def numericalGrad(
        self, params: list[float], dx: float, init_state: int
    ) -> list[float]:
        """
        Get numerical gradient
        """
    def parameterShift(self, params: list[float], init_state: int) -> list[float]:
        """
        Get numerical gradient by parameter shift method
        """

class PauliExp(Ansatz):
    @typing.overload
    def __init__(
        self,
        n_qubits: int,
        circuit: PauliString,
        param_ranges: list[int],
        hamiltonian: PauliString,
    ) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(
        self,
        n_qubits: int,
        circuit: PauliString,
        param_ranges: list[int],
        hamiltonian: PauliString,
        is_init: bool,
    ) -> None:
        """
        Constructor
        """
    def computeExpectation(self) -> float:
        """
        Compute expectation from pauli string
        """
    @typing.overload
    def computeExpectationWithUpdate(
        self, param: list[float], init_state: int
    ) -> float:
        """
        Compute expectation from pauli string
        """
    @typing.overload
    def computeExpectationWithUpdate(
        self, param_id: int, param: float, init_state: int
    ) -> float:
        """
        Compute expectation from pauli string
        """
    def getCircuitElementsSize(self) -> int:
        """
        Get circuit elements size
        """
    def getParamSize(self) -> int:
        """
        Get params size of parametric circuit
        """
    def initState(self, state_bit: int) -> None:
        """
        Init state vector by state bit
        """
    def updateParams(self, params: list[float]) -> None:
        """
        Update param of parametric circuit
        """
    def updateState(self) -> None:
        """
        Update state vector
        """

class PauliString:
    def __init__(
        self,
        n_qubits: int,
        pauli_operators: list[list[int]],
        basis_qubits: list[list[int]],
        pauli_coefs: list[complex],
    ) -> None:
        """
        Constructor
        """

def getTimeDict() -> dict[str, float]: ...
def setCheckPoint(arg0: bool) -> None: ...
def setExpectationQRalgorithm(arg0: str) -> None: ...
def setGradientMode(arg0: str) -> None: ...
def setSkipParamThreshold(arg0: float) -> None: ...
def setUpdateQRalgorithm(arg0: str) -> None: ...
