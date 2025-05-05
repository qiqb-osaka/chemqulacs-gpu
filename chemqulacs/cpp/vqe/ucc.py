##
# @file ucc.py
# @brief UCC classes for VQE calculation
# @authors Yusuke Teranishi, Shoma Hiraoka
from abc import ABCMeta, abstractmethod

from quket.lib.openfermion import QubitOperator as QuketQubitOperator
from quket.utils.utils import Gdoubles_list


class UCCbase(metaclass=ABCMeta):
    """UCC base class"""

    LAYERS_SINGLE = [
        [1, 0],
        [0, 1],
    ]
    LAYERS_DOUBLE = [
        [0, 0, 1, 0],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
    ]

    def __init__(self) -> None:
        """Constructor"""
        ##
        # @brief Constructor
        self.Operator = QuketQubitOperator
        self.rho = 8  # operator's coef
        self.excitations = self.getExcitations()
        self.pauli_list = []

    @abstractmethod
    def getExcitations(self):
        raise NotImplementedError()

    def getSinglePauliOperator(self, wire):
        ##
        # @brief Get SingleExcitationOperator
        # @param wire: wire
        # @return SingleExcitationOperator
        assert len(wire) == 2
        operator = self.Operator()
        for i, layer in enumerate(self.LAYERS_SINGLE):
            op = []
            for j in range(len(layer)):
                if layer[j] == 1:
                    op.append((wire[j], "Y"))
                elif layer[j] == 0:
                    op.append((wire[j], "X"))
            for j in range(min(wire) + 1, max(wire)):
                op.append((j, "Z"))
            coef = 0.5
            if i >= 1:
                coef *= -1
            operator += self.Operator(op, 1j * coef)
        return operator

    def getDoublePauliOperator(self, wire):
        ##
        # @brief Get DoubleExcitationOperator
        # @param wire: wire
        # @return DoubleExcitationOperator
        assert len(wire) == 4
        if int(wire[0] < wire[1]) + int(wire[2] < wire[3]) == 1:
            sign = -1
        else:
            sign = 1
        operator = self.Operator()
        for i, layer in enumerate(self.LAYERS_DOUBLE):
            op = []
            for j in range(4):
                if layer[j] == 1:
                    op.append((wire[j], "Y"))
                elif layer[j] == 0:
                    op.append((wire[j], "X"))
            for j in range(min(wire[0], wire[1]), max(wire[0], wire[1])):
                op.append((j, "Z"))
            for j in range(min(wire[2], wire[3]), max(wire[2], wire[3])):
                op.append((j, "Z"))
            coef = 0.125
            if i >= 4:
                coef *= -1
            operator += self.Operator(op, 1j * coef)
        return sign * operator

    def getPauliOperator(self, wire):
        ##
        # @brief Get PauliOperator
        # @param wire: wire
        # @return PauliOperator
        if len(wire) == 2:
            return self.getSinglePauliOperator(wire)
        elif len(wire) == 4:
            return self.getDoublePauliOperator(wire)
        else:
            raise NotImplementedError()

    def getPauliList(self):
        ##
        # @brief Get PauliList
        # @return PauliList
        if len(self.pauli_list) > 0:
            return self.pauli_list
        if len(self.excitations) == 0:
            self.excitations = self.getExcitations()
        self.pauli_list = []
        for excitation_group in self.excitations:
            pauli_operator = self.Operator()
            for excitation in excitation_group:
                pauli_operator += self.getPauliOperator(excitation)
            self.pauli_list.append((1 / self.rho) * pauli_operator)
        return self.pauli_list

    def getParamSize(self):
        ##
        # @brief Get number of parameters
        # @return number of parameters
        if len(self.excitations) == 0:
            self.excitations = self.getExcitations()
        return len(self.excitations)

    def getRho(self):
        return self.rho

    def getTaperedExcitations(self, taper):
        ##
        # @brief Get tapered excitations
        # @param taper: tapering object
        # @return tapered excitations
        if len(self.pauli_list) == 0:
            self.getPauliList()
        allowed_pauli_list = taper.getAllowedPauliList(self.pauli_list)
        tapered_excitations = []
        for allowed, excitation in zip(
            allowed_pauli_list, self.excitations, strict=True
        ):
            if allowed:
                tapered_excitations.append(excitation)
        return tapered_excitations

    def getTaperedPauliList(self, taper):
        if len(self.pauli_list) == 0:
            self.getPauliList()
        return taper.taperPauliList(self.pauli_list)


class UCCSD(UCCbase):
    """UCCSD class"""

    def __init__(self, n_qubits, n_electrons):
        """Constructor

        Args:
            n_qubits: number of qubits
            n_electrons: number of electrons
        """
        ##
        # @brief Constructor
        # @param n_qubits: number of qubits
        # @param n_electrons: number of electrons
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        super().__init__()

    def getExcitations(self):
        ##
        # @brief Get excitations.
        #   We are considering only the cases where delta_sz equals 0.
        # @return excitations
        sz = [1 - 2 * (i % 2) for i in range(self.n_qubits)]
        excitations = []
        for r in range(self.n_electrons):
            for p in range(self.n_electrons, self.n_qubits):
                if sz[r] - sz[p] == 0:  # even=even or odd=odd
                    excitations.append([[r, p]])
        for s in range(self.n_electrons):
            for r in range(s + 1, self.n_electrons):
                for q in range(self.n_electrons, self.n_qubits):
                    for p in range(q + 1, self.n_qubits):
                        if sz[s] + sz[r] - sz[q] - sz[p] == 0:
                            excitations.append([[s, r, q, p]])
        return excitations


class SAUCCSD(UCCbase):
    """SAUCCSD class"""

    def __init__(self, n_qubits, n_electrons):
        """Constructor

        Args:
            n_qubits: number of qubits
            n_electrons: umber of electrons
        """
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        super().__init__()

    def getExcitations(self):
        ##
        # @brief Get excitations.
        #   We are considering only the cases where delta_sz equals 0.
        # @return excitations
        sz = [1 - 2 * (i % 2) for i in range(self.n_qubits)]
        excitations = []
        for r in range(self.n_electrons):
            for p in range(self.n_electrons, self.n_qubits):
                if sz[r] + sz[p] == 2:  # even and even
                    excitations.append([[r, p], [r + 1, p + 1]])
        for s in range(self.n_electrons):
            for r in range(self.n_electrons):
                for q in range(self.n_electrons, self.n_qubits):
                    for p in range(q, self.n_qubits):
                        if s > r and not q < p:
                            continue
                        if sz[s] + sz[r] + sz[q] + sz[p] == 4:
                            excitations.append(
                                [
                                    [s, r, q, p],
                                    [s + 1, r + 1, q + 1, p + 1],
                                    [s + 1, r, q, p + 1],
                                    [s, r + 1, q + 1, p],
                                ]
                            )
        return excitations


class KupCCGSD(UCCbase):
    """KupCCGSD class"""

    def __init__(self, n_qubits) -> None:
        """Constructor

        Args:
            n_qubits: number of qubits
        """
        self.n_qubits = n_qubits
        super().__init__()

    def getExcitations(self):
        ##
        # @brief Get excitations.
        # @return excitations
        sz = [1 - 2 * (i % 2) for i in range(self.n_qubits)]
        excitations = []
        for r in range(self.n_qubits):
            for p in range(self.n_qubits):
                if r != p and sz[r] == sz[p]:
                    excitations.append([[r, p]])
        for r in range(0, self.n_qubits - 1, 2):
            for p in range(0, self.n_qubits - 1, 2):
                if r != p:
                    excitations.append([[r, r + 1, p, p + 1]])
        return excitations


class UCCGSD(UCCbase):
    """UCCGSD class"""

    def __init__(self, n_qubits) -> None:
        """Constructor

        Args:
            n_qubits: number of qubits
        """
        ##
        # @brief Constructor
        # @param n_qubits: number of qubits
        self.n_qubits = n_qubits
        super().__init__()

    def getExcitations(self):
        ##
        # @brief Get excitations.
        # @return excitations
        excitations = []
        for p in range(self.n_orbitals):
            for q in range(p):
                excitations.append([[p * 2, q * 2]])
                excitations.append([[p * 2 + 1, q * 2 + 1]])
        _, u_list, _ = Gdoubles_list(self.n_orbitals)
        for ilist in range(len(u_list)):
            for b, a, j, i in u_list[ilist]:
                excitations.append([[b, a, j, i]])
        return excitations


# https://github.com/quket/quket/blob/b343f0fbd0818fe5cc1d3c6d98228b4595967904/quket/pauli/pauli.py
class SAUCCGSD(UCCbase):
    """SAUCCGSD class
    https://github.com/quket/quket/blob/b343f0fbd0818fe5cc1d3c6d98228b4595967904/quket/pauli/pauli.py
    """

    def __init__(self, n_orbitals) -> None:
        """Constructor

        Args:
            n_orbitals: _description_
        """
        self.n_orbitals = n_orbitals
        super().__init__()

    def getExcitations(self):
        ##
        # @brief Get excitations.
        # @return excitations
        excitations = []
        for p in range(self.n_orbitals):
            for q in range(p):
                excitations.append([[p * 2, q * 2], [p * 2 + 1, q * 2 + 1]])
        _, u_list, _ = Gdoubles_list(self.n_orbitals)
        for u in u_list:
            if len(u) == 6:
                group = []
                for i in range(4):
                    group.append([u[i][0], u[i][1], u[i][2], u[i][3]])
                excitations.append(group)
                group = []
                for i in [0, 1, 4, 5]:
                    group.append([u[i][0], u[i][1], u[i][2], u[i][3]])
                excitations.append(group)
            elif len(u) <= 2:
                group = []
                for i in range(len(u)):
                    group.append([u[i][0], u[i][1], u[i][2], u[i][3]])
                excitations.append(group)
        return excitations
