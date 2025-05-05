##
# @file vqemcscf.py
# @brief VQE MCSCF class
# type:ignore
from pyscf.mcscf import casci

from chemqulacs.cpp.vqe import vqeci


class VQECASCI(casci.CASCI):
    ##
    # @brief VQE MCSCF class
    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        ncore=None,
        ansatz_name: str = "sauccsd",
        n_layers: int = 2,
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
        casci.CASCI.__init__(self, mf, ncas, nelecas, ncore)
        self.canonicalization = False
        self.fcisolver = vqeci.VQECI(
            mf=mf,
            ansatz_name=ansatz_name,
            n_layers=n_layers,
            include_pi=include_pi,
            k=k,
            init_param=init_param,
            run_nft=run_nft,
            random_seed=random_seed,
            is_tapering=is_tapering,
            is_debug=is_debug,
            maxiter=maxiter,
            n_compute_unit=n_compute_unit,
            dump_filename=dump_filename,
            load_filename=load_filename,
            comm=comm,
        )
