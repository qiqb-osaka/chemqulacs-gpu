from quket.lib.openfermion import QubitOperator as QuketQubitOperator
from quket.pauli import get_allowed_pauli_list
from quket.quket_data.chemical import get_pointgroup_character_table
from quket.tapering.tapering import Z2tapering
from quket.utils.utils import set_initial_det

from chemqulacs.cpp.vqe.converter import (
    convertHamiltonianOpenfermionToQuket,
)


class Tapering:
    """Tapering class"""

    def __init__(self, hamiltonian, mf, n_qubits, n_electrons):
        ##
        # @brief Constructor
        # @param hamiltonian: Hamiltonian
        # @param mf: mean-field object
        # @param n_qubits: number of qubits
        # @param n_electrons: number of electrons
        self.hamiltonian = convertHamiltonianOpenfermionToQuket(hamiltonian)
        self.mf = mf
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons

        noa = nob = n_electrons // 2
        det = set_initial_det(noa, nob)
        self.tapering = Z2tapering(
            self.hamiltonian,
            self.n_qubits,
            det=det,
            PGS=self.getPGS(),
            Sz_symmetry=True,
        )
        self.tapering.run()
        self.redundant_bits = self.tapering.redundant_bits
        self.map_wires = {}
        idx = 0
        for q in range(self.n_qubits):
            if q in self.redundant_bits:
                continue
            self.map_wires[q] = idx
            idx += 1

    def getPGS(self):
        ##
        # @brief Get point group symmetry information
        irrep_name = self.mf.mol.irrep_name
        symm_orb = self.mf.mol.symm_orb
        groupname = self.mf.mol.groupname
        mo_coeff = self.mf.mo_coeff
        symm_operations, irrep_list, character_list = get_pointgroup_character_table(
            self.mf.mol, groupname, irrep_name, symm_orb, mo_coeff
        )
        if symm_operations is None:
            return None
        n_frozen_orbitals = 0
        n_core_orbitals = 0
        n_active_orbitals = self.n_qubits // 2
        nfrozen = n_frozen_orbitals * 2
        ncore = n_core_orbitals * 2
        nact = n_active_orbitals * 2
        pgs_head, pgs_tail = nfrozen, nfrozen + ncore + nact
        pgs = (
            symm_operations,
            irrep_list[pgs_head:pgs_tail],
            [x[pgs_head:pgs_tail] for x in character_list],
        )
        return pgs

    def taperHamiltonian(self):
        self.tapered_hamiltonian = self.tapering.transform_operator(self.hamiltonian)
        return self.tapered_hamiltonian

    def taperHFstate(self):
        hf_bit = 2**self.n_electrons - 1
        tapered_hf_bit, parity = self.tapering.transform_bit(hf_bit)
        n_tapered_qubits = self.n_qubits - len(self.redundant_bits)
        return tapered_hf_bit, n_tapered_qubits

    def getAllowedPauliList(self, pauli_list):
        return get_allowed_pauli_list(self.tapering, pauli_list)

    def taperMapQubitPauliOperator(self, pauli_operator):
        mapped_pauli_operator = QuketQubitOperator()
        for term, coef in pauli_operator.terms.items():
            term_str = ""
            for target, pauli in term:
                new_target = self.map_wires[target]
                term_str += f"{pauli}{new_target}"
            mapped_pauli_operator += QuketQubitOperator(term_str, coef)
        return mapped_pauli_operator

    def taperMapQubitPauliList(self, pauli_list):
        mapped_pauli_list = []
        for pauli_operator in pauli_list:
            mapped_pauli_operator = self.taperMapQubitPauliOperator(pauli_operator)
            mapped_pauli_list.append(mapped_pauli_operator)
        return mapped_pauli_list

    def taperPauliList(self, pauli_list):
        return self.tapering.transform_pauli_list(pauli_list)
