import pennylane as qml
from openfermion.ops import QubitOperator
from pennylane import numpy as np
from quket.lib.openfermion import QubitOperator as QuketQubitOperator

OPERATOR_MAP = {
    "Identity": "I",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
}


def convertHamiltonianOpenfermionToPennylane(hamiltonian):
    """convert Hamiltonian of Openfermion to one of Pennylane

    Args:
        hamiltonian: Hamiltonian of Openfermion

    Returns:
        _type_: Hamiltonian of Pennylane
    """
    return qml.qchem.import_operator(hamiltonian)


def convertHamiltonianOpenfermionToQuket(hamiltonian) -> QuketQubitOperator:
    """convert Hamiltonian of Openfermion to one of Quket

    Args:
        hamiltonian: Hamiltonian of Openfermion

    Returns:
        _type_: Hamiltonian of Quket
    """
    qubit_operator = QuketQubitOperator()
    for term, coef in hamiltonian.terms.items():
        qubit_operator += QuketQubitOperator(term, coef)
    return qubit_operator


def convertHamiltonianPennylaneToOpenFermion(
    hamiltonian, map_wires=None, Operator=QubitOperator
):
    """convert Hamiltonian of Pennylane to one of OpenFermion

    Args:
        hamiltonian: Hamiltonian of Pennylane
        map_wires:  Defaults to None.
        Operator: Defaults to QubitOperator.

    Returns:
        _type_: Hamiltonian of OpenFermion
    """
    if map_wires is None:
        map_wires = {q: q for q in hamiltonian.wires}
    qubit_operator = Operator()
    for coef, term in zip(hamiltonian.coeffs, hamiltonian.ops, strict=True):
        if isinstance(coef, np.tensor):
            coef = coef.unwrap()
        if term.name == "Identity":
            qubit_operator += Operator("", coef)
        elif type(term.name) is str:
            op = OPERATOR_MAP[term.name]
            idx = map_wires[term.wires.labels[0]]
            qubit_operator += Operator(f"{op}{idx}", coef)
        else:
            term_str = ""
            for i in range(len(term.name)):
                if term.name[i] == "Identity":
                    continue
                if term_str != "":
                    term_str += " "
                op = OPERATOR_MAP[term.name[i]]
                idx = map_wires[term.wires.labels[i]]
                term_str += f"{op}{idx}"
            qubit_operator += Operator(term_str, coef)
    return qubit_operator


def convertHamiltonianPennylaneToQuket(hamiltonian, map_wires=None):
    """convert Hamiltonian of Pennylane to one of Quket

    Args:
        hamiltonian: Hamiltonian of Pennylane
        map_wires: Defaults to None.

    Returns:
        _type_: Hamiltonian of : Quket
    """
    return convertHamiltonianPennylaneToOpenFermion(
        hamiltonian, map_wires, QuketQubitOperator
    )
