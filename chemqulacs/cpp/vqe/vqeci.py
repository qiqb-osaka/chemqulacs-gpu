##
# @file vqeci.py
# @brief VQE class
# type:ignore
import datetime
import json
import time

import chemqulacs_cpp
import numpy as np
import openfermion
from openfermion.ops import InteractionOperator
from openfermion.transforms import get_fermion_operator
from pyscf import ao2mo
from scipy.optimize import minimize

from chemqulacs.cpp.pybind.interface import (
    ParsePauliListToCircuit,
    ParsePauliString,
)
from chemqulacs.cpp.vqe.ccsd import CCSD
from chemqulacs.cpp.vqe.mp2 import MP2
from chemqulacs.cpp.vqe.tapering import Tapering
from chemqulacs.cpp.vqe.ucc import SAUCCGSD, SAUCCSD, UCCGSD, UCCSD, KupCCGSD
from chemqulacs.cpp.vqe.utils import ndjson_dump


class VQECI(object):
    ##
    # @brief VQE class
    def __init__(
        self,
        mf=None,
        ansatz_name: str = "uccsd",
        n_layers: int = 1,
        include_pi: bool = False,
        k: int = 1,
        init_param=None,
        run_nft: bool = False,
        random_seed: int = 0,
        is_tapering: bool = False,
        is_debug: bool = False,
        maxiter: int = 1000,
        n_compute_unit=None,
        dump_filename=None,
        load_filename=None,
        comm=None,
    ):
        ##
        # @brief Constructor
        # @param mf: mean-field object
        # @param ansatz_name: ansatz name(["uccsd", "sauccsd", "uccgsd", "sauccgsd", "kupccgsd", "gatefabric"])
        # @param n_layers: number of layers
        # @param include_pi: include pi(for gate fabric)
        # @param k: number of qubits for tapering(for k-upCCGSD)
        # @param init_param: initial parameters
        # @param run_nft: whether to run NFT
        # @param random_seed: random seed
        # @param is_tapering: whether to tapering
        # @param is_debug: whether to debug(only one iteration)
        # @param maxiter: maximum number of iterations
        # @param n_compute_unit: number of compute unit
        # @param dump_filename: dump filename
        # @param load_filename: load filename
        # @param comm: type of MPI communicator
        self.mf = mf
        self.ansatz_name = ansatz_name
        self.n_layers = n_layers
        self.include_pi = include_pi
        self.k = k
        self.init_param = init_param
        self.run_nft = run_nft
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.is_tapering = is_tapering
        self.is_debug = is_debug
        self.maxiter = maxiter
        self.n_compute_unit = n_compute_unit

        self.comm = comm
        if self.comm is None:
            self.mpi_rank = 0
            self.mpi_size = 1
        else:
            self.mpi_rank = comm.Get_rank()
            self.mpi_size = comm.Get_size()
        self.is_root = True if self.mpi_rank == 0 else False

        self.is_dump = True if dump_filename is not None else False
        self.is_load = True if load_filename is not None else False
        self.load_cost_list = []
        self.load_grad_list = []
        self.iteration = 0
        if self.is_load:
            with open(load_filename, "r") as f:
                lines = f.readlines()
                is_init_param = False
                for line in lines:
                    data = json.loads(line)
                    if data["type"] == "cost":
                        self.load_cost_list.append(data["cost"])
                        if not is_init_param:
                            self.init_param = data["param"]
                            is_init_param = True
                    elif data["type"] == "grad":
                        self.load_grad_list.append(data["grad"])
        if self.is_dump and self.is_root:
            self.fdump = open(dump_filename, "a")

    def __del__(self):
        if self.is_dump and self.is_root:
            self.fdump.close()

    def get_active_hamiltonian(self, h1, h2, norb, nelec, ecore):
        ##
        # @brief Get active space Hamiltonian
        # @param h1: one electron integrals
        # @param h2: two electron integrals
        # @param norb: number of orbitals
        # @param nelec: number of electrons
        # @param ecore: core energy
        # @return active space Hamiltonian
        n_orbitals = h1.shape[0]
        n_qubits = 2 * n_orbitals

        self.n_orbitals = n_orbitals
        self.n_qubits = n_qubits
        self.n_electrons = nelec[0] + nelec[1]

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
        # Set MO one and two electron-integrals
        # according to OpenFermion conventions
        one_body_integrals = h1
        h2_ = ao2mo.restore(
            1, h2.copy(), n_orbitals
        )  # no permutation see two_body_integrals of _pyscf_molecular_data.py
        two_body_integrals = np.asarray(h2_.transpose(0, 2, 3, 1), order="C")

        # Taken from OpenFermion
        # Loop through integrals.
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):
                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
                one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_integrals[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):
                        # Mixed spin
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.0
                        )
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.0
                        )

                        # Same spin
                        two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.0
                        )
                        two_body_coefficients[
                            2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                        ] = two_body_integrals[p, q, r, s] / 2.0

        # Get Hamiltonian in OpenFermion format
        active_hamiltonian = InteractionOperator(
            ecore, one_body_coefficients, two_body_coefficients
        )
        return active_hamiltonian

    def kernel(self, h1, h2, norb, nelec, ecore=0, **kwargs):
        ##
        # @brief VQE kernel
        # @param h1: one electron integrals
        # @param h2: two electron integrals
        # @param norb: number of orbitals
        # @param nelec: number of electrons
        # @param ecore: core energy
        # @return energy and wavefunction
        # Get the active space Hamiltonian
        active_hamiltonian = self.get_active_hamiltonian(h1, h2, norb, nelec, ecore)
        # Convert the Hamiltonian using Jordan Wigner
        fermionic_hamiltonian = get_fermion_operator(active_hamiltonian)
        hamiltonian = openfermion.transforms.jordan_wigner(fermionic_hamiltonian)

        hf_state = 2**self.n_electrons - 1

        if self.is_dump and self.is_root:
            ndjson_dump({"ansatz_name": self.ansatz_name}, self.fdump, type="info")

        # set given ansatz
        is_init = False  # init by MultiAnsatz
        if self.ansatz_name in ["uccsd", "sauccsd", "uccgsd", "sauccgsd", "kupccgsd"]:
            if self.ansatz_name == "uccsd":
                ucc_ansatz = UCCSD(self.n_qubits, self.n_electrons)
            elif self.ansatz_name == "sauccsd":
                ucc_ansatz = SAUCCSD(self.n_qubits, self.n_electrons)
            elif self.ansatz_name == "uccgsd":
                ucc_ansatz = UCCGSD(self.n_orbitals)
            elif self.ansatz_name == "sauccgsd":
                ucc_ansatz = SAUCCGSD(self.n_orbitals)
            elif self.ansatz_name == "kupccgsd":
                ucc_ansatz = KupCCGSD(self.n_qubits)
            pauli_list = ucc_ansatz.getPauliList()
            if self.is_tapering:
                tapering = Tapering(
                    hamiltonian, self.mf, self.n_qubits, self.n_electrons
                )
                hamiltonian = tapering.taperHamiltonian()
                hf_state, n_qubits_tapered = tapering.taperHFstate()
                tapered_pauli_list = ucc_ansatz.getTaperedPauliList(tapering)
                pauli_circuit, param_ranges = ParsePauliListToCircuit(
                    n_qubits_tapered, tapered_pauli_list
                )
                ansatz_cpp = chemqulacs_cpp.PauliExp(
                    n_qubits_tapered,
                    pauli_circuit,
                    param_ranges,
                    ParsePauliString(n_qubits_tapered, hamiltonian),
                    is_init,
                )
                if self.is_root:
                    print(f"[INFO]n_qubits={self.n_qubits}->{n_qubits_tapered}")
                    print(
                        f"[INFO]n_params={len(pauli_list)}->{len(tapered_pauli_list)}",
                        flush=True,
                    )
                    if self.is_dump:
                        ndjson_dump(
                            {
                                "n_qubits": self.n_qubits,
                                "n_qubits_tapered": n_qubits_tapered,
                                "n_params": len(pauli_list),
                                "n_params_tapered": len(tapered_pauli_list),
                            },
                            self.fdump,
                            type="info",
                        )
            else:
                pauli_circuit, param_ranges = ParsePauliListToCircuit(
                    self.n_qubits, pauli_list
                )
                ansatz_cpp = chemqulacs_cpp.PauliExp(
                    self.n_qubits,
                    pauli_circuit,
                    param_ranges,
                    ParsePauliString(self.n_qubits, hamiltonian),
                    is_init,
                )
            if self.init_param in [
                "ccsd",
                "mp2",
            ] and self.ansatz_name in ["uccsd", "sauccsd"]:
                if self.is_tapering:
                    excitations = ucc_ansatz.getTaperedExcitations(tapering)
                else:
                    excitations = ucc_ansatz.getExcitations()
                if self.init_param == "ccsd":
                    ccsd = CCSD(self.mf)
                    self.init_param = (
                        ccsd.getAmplitudesFromExcitations(excitations)
                        * ucc_ansatz.getRho()
                    )
                    if self.is_dump and self.is_root:
                        ndjson_dump(
                            {"ccsd": {"e_tot": ccsd.e_tot, "time": ccsd.duration}},
                            self.fdump,
                            type="info",
                        )
                elif self.init_param == "mp2":
                    mp2 = MP2(self.mf)
                    self.init_param = (
                        mp2.getAmplitudesFromExcitations(excitations)
                        * ucc_ansatz.getRho()
                    )
                    if self.is_dump and self.is_root:
                        ndjson_dump(
                            {"mp2": {"e_tot": mp2.e_tot, "time": mp2.duration}},
                            self.fdump,
                            type="info",
                        )
        else:
            if self.ansatz_name == "gatefabric":
                ansatz_cpp = chemqulacs_cpp.GateFabric(
                    self.n_qubits,
                    self.n_layers,
                    self.include_pi,
                    ParsePauliString(self.n_qubits, hamiltonian),
                    is_init,
                )
            else:
                raise ValueError("given ansatz are not supported.")
        if not self.is_tapering and self.is_root:
            print(f"[INFO]n_qubits={self.n_qubits}")
            print(
                f"[INFO]n_params={ansatz_cpp.getParamSize()}",
                flush=True,
            )
            if self.is_dump:
                ndjson_dump(
                    {"n_qubits": self.n_qubits, "n_params": ansatz_cpp.getParamSize()},
                    self.fdump,
                    type="info",
                )

        # def bench():
        #     t = time.time()
        #     exp = ansatz_cpp.computeExpectationWithUpdate(self.init_param, hf_state)
        #     dt = time.time() - t
        #     for i in range(self.mpi_size):
        #         if i == self.mpi_rank:
        #             print(
        #                 f"host={MPI.Get_processor_name()},rank={self.mpi_rank},exp={exp},time={dt}", flush=True
        #             )
        #         self.comm.barrier()
        #     exit(0)

        # bench()

        if self.n_compute_unit is not None:
            multi_ansatz_cpp = chemqulacs_cpp.MultiAnsatz(
                ansatz_cpp, self.n_compute_unit
            )
        else:
            multi_ansatz_cpp = chemqulacs_cpp.MultiAnsatz(ansatz_cpp)

        if self.is_root:
            time_dict = chemqulacs_cpp.getTimeDict()
            print(f"{time_dict=}")
            print("Ansatz:", self.ansatz_name)
            print("----VQE-----", flush=True)

        self.predict_iteration = 40
        self.time_cost = 0

        def cost(param):
            ##
            # @brief Cost function for VQE
            # @param param: parameters
            # @return expectation value
            if self.iteration < len(self.load_cost_list):
                exp = self.load_cost_list[self.iteration]
                if self.is_root:
                    print(f"Iteration {self.iteration}: E={exp}", flush=True)
                return exp

            t0 = time.time()
            exp = multi_ansatz_cpp.computeExpectation(param, 1e-6, hf_state)
            time_cost = time.time() - t0

            if self.is_root:
                print(f"Iteration {self.iteration}: E={exp}")
                time_dict = chemqulacs_cpp.getTimeDict()
                time_init = time_dict["init"]
                time_param = time_dict["param"]
                time_update = time_dict["update"]
                time_exp = time_dict["exp"]
                time_bcast = time_dict["cost_bcast"]
                print(
                    f"\t[time]cost:{time_cost:.5f}(init:{time_init:.5f},param:{time_param:.5f},update:{time_update:.5f},exp:{time_exp:.5f},bcast:{time_bcast:.5f})"
                )
                time_update_compute = time_dict["update_compute"]
                time_update_comm = time_dict["update_communicate"]
                print(
                    f"\t\t[time]update(compute:{time_update_compute:.5f},comm:{time_update_comm:.5f})"
                )
                time_exp_compute = time_dict["exp_compute"]
                time_exp_comm = time_dict["exp_communicate"]
                time_exp_reduce = time_dict["exp_reduce"]
                print(
                    f"\t\t[time]exp(compute:{time_exp_compute:.5f},comm:{time_exp_comm:.5f},reduce:{time_exp_reduce:.5f})",
                    flush=True,
                )
                time_predict_grad = (time_cost - time_bcast) * np.ceil(
                    len(param) / self.mpi_size
                ) + time_bcast
                print(
                    f"\t\t[time]predict_grad:{datetime.timedelta(seconds=time_predict_grad)}"
                )
                self.time_cost = time_cost
                if self.is_dump:
                    ndjson_dump(
                        {
                            "iteration": self.iteration,
                            "cost": exp,
                            "time": {
                                "cost": time_cost,
                                "init": time_init,
                                "param": time_param,
                                "update": {
                                    "total": time_update,
                                    "compute": time_update_compute,
                                    "communicate": time_update_comm,
                                },
                                "exp": {
                                    "total": time_exp,
                                    "compute": time_exp_compute,
                                    "communicate": time_exp_comm,
                                    "reduce": time_exp_reduce,
                                },
                                "bcast": time_bcast,
                            },
                            "param": param.tolist(),
                        },
                        self.fdump,
                        type="cost",
                    )
            return exp

        def grad_cost(param):
            ##
            # @brief Calculate gradient
            # @param param: parameters
            # @return gradient
            if self.iteration < len(self.load_grad_list):
                grad = self.load_grad_list[self.iteration]
                self.iteration += 1
                return grad

            t = time.time()
            # grad = multi_ansatz_cpp.parameterShift(param, hf_state)
            grad = multi_ansatz_cpp.numericalGrad(param, 1e-6, hf_state)
            time_grad = time.time() - t

            if self.is_root:
                time_dict = chemqulacs_cpp.getTimeDict()
                time_compute = time_dict["grad_compute"]
                time_allgather = time_dict["grad_allgather"]
                time_barrier = time_dict["barrier"]
                print(
                    f"\t[time]grad:{time_grad:.5f}(compute:{time_compute:.5f},barrier:{time_barrier:.5f},allgather:{time_allgather:.5f})",
                    flush=True,
                )
                if self.iteration > 0:
                    if self.iteration >= self.predict_iteration:
                        self.predict_iteration += 10
                    time_predict_left = (self.predict_iteration - self.iteration) * (
                        time_grad + self.time_cost
                    )
                    t_current = time.time() - self.t0
                    t_finish = t_current + time_predict_left
                    print(
                        f"\t[time]to{self.predict_iteration}iteration->{datetime.timedelta(seconds=t_current)}/{datetime.timedelta(seconds=t_finish)}[s]({t_current / t_finish * 100:.2f}[%])"
                    )
                if self.is_dump:
                    ndjson_dump(
                        {
                            "iteration": self.iteration,
                            "time": {
                                "grad": time_grad,
                                "compute": time_compute,
                                "allgather": time_allgather,
                            },
                            "grad": grad,
                        },
                        self.fdump,
                        type="grad",
                    )
            self.iteration += 1
            return grad

        init_theta_list = self.__set_init_theta_list(ansatz_cpp)
        if self.run_nft:
            init_theta_list = self.__descentNFT(
                multi_ansatz_cpp, init_theta_list, hf_state
            )

        if self.is_debug:
            cost(init_theta_list)
            cost(init_theta_list)
            cost(init_theta_list)
            return 0, None

        method = "BFGS"
        disp = True if self.mpi_rank == 0 else False
        options = {"disp": disp, "maxiter": self.maxiter, "gtol": 1e-5}

        self.t0 = time.time()
        opt = minimize(
            cost,
            init_theta_list,
            # jac=jac_ana,
            method=method,
            options=options,
        )
        dt = time.time() - self.t0

        if self.is_root:
            print("[time]optimize:", dt, flush=True)

        e = opt.fun.real
        return e, None

    def __set_init_theta_list(self, ansatz_cpp):
        ##
        # @brief Set initial parameters
        # @param ansatz_cpp: ansatz object
        # @return initial parameters
        n_params = ansatz_cpp.getParamSize()
        if isinstance(self.init_param, (list, np.ndarray)):
            assert len(self.init_param) == n_params
            init_theta_list = self.init_param
        elif isinstance(self.init_param, str):
            if self.init_param == "random":
                init_theta_list = np.random.random(size=n_params)
            else:
                init_theta_list = np.zeros(n_params)
        else:
            init_theta_list = np.zeros(n_params)
        return init_theta_list

    def __descentNFT(self, multi_ansatz, init_theta_list, hf_state):
        ##
        # @brief Descent NFT
        # @param multi_ansatz: multi ansatz object
        # @param init_theta_list: initial parameters
        # @param hf_state: Hartree-Fock state
        # @return optimized parameters
        params = np.copy(init_theta_list)
        pre_params = np.copy(init_theta_list)
        pre_exp = np.inf
        while True:
            for i in range(len(params)):
                old_param = params[i]
                z0 = multi_ansatz.computeExpectation(params, hf_state)
                params[i] = old_param + np.pi / 2
                z1 = multi_ansatz.computeExpectation(params, hf_state)
                params[i] = old_param - np.pi / 2
                z3 = multi_ansatz.computeExpectation(params, hf_state)
                z2 = z1 + z3 - z0
                params[i] = (
                    old_param
                    + np.arctan((z1 - z3) / (z0 - z2))
                    + 0.5 * np.pi
                    + 0.5 * np.pi * np.sign((z0 - z2))
                )
            exp = multi_ansatz.computeExpectation(params, hf_state)
            print("NFT:", exp)
            if pre_exp < exp:
                break
            pre_exp = exp
            pre_params = np.copy(params)
        return pre_params
