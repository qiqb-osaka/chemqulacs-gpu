# type:ignore
import argparse
import json
import time

import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm.barrier()
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
is_root = mpi_rank == 0
if is_root:
    print(f"Number of GPUs: {mpi_size}", flush=True)

import chemqulacs_cpp
from pyscf import cc, gto, mcscf, scf

from chemqulacs.cpp.vqe import vqemcscf
from chemqulacs.cpp.vqe.utils import get_molecule, ndjson_dump

# chemqulacs_cpp.setSkipParamThreshold(0)
chemqulacs_cpp.setSkipParamThreshold(1e-7)
chemqulacs_cpp.setGradientMode("CHECKPOINT")
# chemqulacs_cpp.setGradientMode("BACKWARD")
# chemqulacs_cpp.setGradientMode("CENTRAL")
# chemqulacs_cpp.setUpdateQRalgorithm("TILING")
# chemqulacs_cpp.setExpectationQRalgorithm("DIAGONAL")

parser = argparse.ArgumentParser()
parser.add_argument("--mol_name", default="h2o")
parser.add_argument("--ansatz", default="sauccsd")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--dump")
parser.add_argument("--load")
args = parser.parse_args()
mol_name = args.mol_name
ansatz_name = args.ansatz
is_debug = args.debug
dump_filename = args.dump
load_filename = args.load
is_dump = dump_filename is not None and is_root
is_load = load_filename is not None

if is_dump:
    if is_load:
        with open(load_filename, "r") as f:
            buf = f.read()
        with open(dump_filename, "a") as f:
            f.write(buf)
    with open(dump_filename, "a") as f:
        ndjson_dump(
            {"mpi_size": mpi_size},
            f,
            type="info",
        )

is_load_casci = False
if is_load:
    with open(load_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if "mol_name" in data:
                mol_name = data["mol_name"]
            if "geometry" in data:
                geometry = data["geometry"]
            if "n_orbitals" in data:
                n_orbitals = data["n_orbitals"]
            if "n_electrons" in data:
                n_electrons = data["n_electrons"]
            if "charge" in data:
                charge = data["charge"]
            if "symmetry" in data:
                symmetry = data["symmetry"]
            if "basis" in data:
                basis = data["basis"]
            if "casci" in data:
                casci_e = data["casci"]["e_tot"]
                time_casci = data["casci"]["time"]
                is_load_casci = True
else:
    geometry, symmetry, charge = get_molecule(mol_name)
    basis = "sto-3g"

if is_root:
    print("Mol Name:", mol_name, flush=True)

mol = gto.M(atom=geometry, basis=basis, symmetry=symmetry, charge=charge)
mol.verbose = 3 if is_root else 0
mol.build()
if not is_load:
    n_orbitals = int(mol.nao)  # np.int64 -> int
    n_electrons = mol.nelectron

if is_dump:
    with open(dump_filename, "a") as f:
        ndjson_dump(
            {
                "mol_name": mol_name,
                "geometry": geometry,
                "n_orbitals": n_orbitals,
                "n_electrons": n_electrons,
                "charge": charge,
                "symmetry": symmetry,
                "basis": basis,
            },
            f,
            type="info",
        )

t = time.time()
mf = scf.RHF(mol).run()
time_hf = time.time() - t
hf_e = mf.e_tot
if is_root:
    print(f"[time]mf.run:{time_hf}", flush=True)

if not is_load_casci:
    t = time.time()
    refmc = mcscf.CASCI(mf, n_orbitals, n_electrons)
    if not is_debug:
        refmc.run()
    elif is_root:
        print("CASCI is skipped.", flush=True)
    casci_e = refmc.e_tot
    time_casci = time.time() - t
    if is_root:
        print(f"[time]refmc.run:{time_casci}", flush=True)
else:
    if is_root:
        print(f"load_casci:{casci_e}", flush=True)

if is_dump:
    with open(dump_filename, "a") as f:
        ndjson_dump(
            {
                "hf": {"e_tot": hf_e, "time": time_hf},
                "casci": {"e_tot": casci_e, "time": time_casci},
            },
            f,
            type="info",
        )

mc = vqemcscf.VQECASCI(
    mf,
    n_orbitals,
    n_electrons,
    ansatz_name=ansatz_name,
    is_tapering=True,
    init_param="ccsd",
    is_debug=is_debug,
    dump_filename=dump_filename,
    load_filename=load_filename,
    comm=comm,
    n_compute_unit=2,
)
mc.kernel()
comm.barrier()

if is_root:
    print("VQE Energy,   CASCI Energy")
    print(mc.e_tot, casci_e, flush=True)
