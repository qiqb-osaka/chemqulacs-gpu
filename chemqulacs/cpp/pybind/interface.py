import chemqulacs_cpp
import pennylane as qml
from pennylane import numpy as np

CUSTATEVEC_PAULI_X = 1
CUSTATEVEC_PAULI_Y = 2
CUSTATEVEC_PAULI_Z = 3

PAULI_PENNY_TO_CUSV = {
    "PauliX": CUSTATEVEC_PAULI_X,
    "PauliY": CUSTATEVEC_PAULI_Y,
    "PauliZ": CUSTATEVEC_PAULI_Z,
}
PAULI_OPFE_TO_CUSV = {
    "X": CUSTATEVEC_PAULI_X,
    "Y": CUSTATEVEC_PAULI_Y,
    "Z": CUSTATEVEC_PAULI_Z,
}

EPS_ZERO = 1e-9


def ParsePauliString(n_qubits, pauli_string):
    pauli_operators = []
    basis_qubits = []
    pauli_coefs = list(pauli_string.terms.values())  # list[numpy.complex128]
    pauli_terms = list(pauli_string.terms.keys())

    def get_first_unused_qubit(key):
        factor_set = set(key)
        for i in range(n_qubits):
            if not any((i, pauli) in factor_set for pauli in ("X", "Y", "Z")):
                return i
        return i + 1

    def get_last_unused_qubit(key):
        factor_set = set(key)
        for i in range(n_qubits - 2, -1, -1):
            if not any((i, pauli) in factor_set for pauli in ("X", "Y", "Z")):
                return i
        return i + 1

    # # zipで2つのリストを組み合わせ、基準リストでソート
    sorted_pairs = sorted(
        # zip(pauli_terms, pauli_coefs, strict=False), key=lambda pair: -get_last_unused_qubit(pair[0])
        zip(pauli_terms, pauli_coefs, strict=False),
        key=lambda pair: get_first_unused_qubit(pair[0]),
    )

    # ソート後のリストに分解
    pauli_terms, pauli_coefs = zip(*sorted_pairs, strict=False)

    for term in pauli_terms:
        pauli_operator = []
        basis_qubit = []
        for target, pauli in term:
            basis_qubit.append(target)
            pauli_operator.append(PAULI_OPFE_TO_CUSV[pauli])
        basis_qubits.append(basis_qubit)
        pauli_operators.append(pauli_operator)
    return chemqulacs_cpp.PauliString(
        n_qubits, pauli_operators, basis_qubits, pauli_coefs
    )


def ParsePennylaneCircuit(circuit, map_wires=None):
    pauli_operators = []
    basis_qubits = []
    pauli_coefs = []
    param_ranges = [0]
    for param_ops in circuit:
        ops = param_ops(1.0)
        is_append = False
        for op in ops:
            if isinstance(op.coeff, np.complex128):
                coef = np.real(op.coeff / 1j)
            elif isinstance(op.coeff, qml.numpy.tensor):
                coef = np.real(op.coeff.unwrap() / 1j)
            if np.abs(coef) < EPS_ZERO:
                continue
            if isinstance(op.base, qml.operation.Tensor):
                pauli_product = op.base.obs
            else:
                pauli_product = [op.base]
            paulis = []
            targets = []
            for pauli in pauli_product:
                paulis.append(PAULI_PENNY_TO_CUSV[pauli.name])
                if map_wires is None:
                    target = pauli.wires[0]
                else:
                    target = map_wires[pauli.wires[0]]
                targets.append(target)
            pauli_coefs.append(coef)
            pauli_operators.append(paulis)
            basis_qubits.append(targets)
            is_append = True
        if is_append:
            param_ranges.append(len(pauli_coefs))
    return (
        chemqulacs_cpp.PauliString(pauli_operators, basis_qubits, pauli_coefs),
        param_ranges,
    )


def ParsePauliListToCircuit(n_qubits, pauli_list):
    pauli_operators = []
    basis_qubits = []
    pauli_coefs = []
    param_ranges = [0]
    for pauli_exps in pauli_list:
        is_append = False
        for term, coef_complex in pauli_exps.terms.items():
            if coef_complex.real:
                coef = coef_complex.real
            elif coef_complex.imag:
                coef = coef_complex.imag
            else:
                print("Warning: invalid complex coef.")
                continue
            if np.abs(coef) < EPS_ZERO:
                continue
            paulis = []
            targets = []
            for target, pauli in term:
                paulis.append(PAULI_OPFE_TO_CUSV[pauli])
                targets.append(target)
            pauli_coefs.append(coef)
            pauli_operators.append(paulis)
            basis_qubits.append(targets)
            is_append = True
        if is_append:
            param_ranges.append(len(pauli_coefs))
    return (
        chemqulacs_cpp.PauliString(
            n_qubits, pauli_operators, basis_qubits, pauli_coefs
        ),
        param_ranges,
    )
