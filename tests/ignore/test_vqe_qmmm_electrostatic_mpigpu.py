import numpy as np
import torch
from mpi4py import MPI
from pyscf import gto, qmmm, scf

from chemqulacs.qmmm import electrostatic
from chemqulacs.util import utils
from chemqulacs.vqe import vqemcscf

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()


coords = np.array(
    [
        [5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    ]
)

ref_array = np.array(
    [
        0.0158443480475567,
        -0.0209835214742796,
        0.0202941142094810,
        -0.0131361984358331,
        -0.0116259752588906,
        0.0096787003994477,
    ]
)

mc_e_tot_true = -74.965695110031


geom_water = utils.get_geometry_from_pubchem("water")

mol = gto.M(atom=geom_water, basis="sto-3g", unit="angstrom")
mf = scf.RHF(mol)
mf.run()
mf_qmmm = qmmm.mm_charge(mf, coords, np.ones_like(coords[:, 0]), unit="Bohr")


def run_mc_single():
    mc = vqemcscf.VQECASCI(mf, 2, 2, use_grad=True)
    mc.kernel()
    return mc


def run_mc_mpi():
    mc = vqemcscf.VQECASCI(mf, 2, 2, use_grad=True, comm=comm)
    mc.kernel()
    return mc


def run_mc_mpigpu():
    # distribute gpuid
    numGpuPerNode = torch.cuda.device_count()
    gpuid = mpirank % numGpuPerNode
    mc = vqemcscf.VQECASCI(mf, 2, 2, use_grad=True, comm=comm, gpuid=gpuid)
    mc.kernel()
    return mc


def assert_mc_almost_equal(mc):
    assert utils.almost_equal(mc.e_tot, mc_e_tot_true)
    vpot = electrostatic.vpot_pyscf(mf_qmmm, mc)
    assert utils.almost_equal(vpot, ref_array, threshold=1.0e-6).prod()


def test_vpot_vqe_single():
    if mpirank == 0:
        mc_single = run_mc_single()
        assert_mc_almost_equal(mc_single)
    comm.barrier()


def test_vpot_vqe_mpi():
    mc_mpi = run_mc_mpi()
    if mpirank == 0:
        assert_mc_almost_equal(mc_mpi)
    comm.barrier()


def test_vpot_vqe_mpigpu():
    mc_mpigpu = run_mc_mpigpu()
    if mpirank == 0:
        assert_mc_almost_equal(mc_mpigpu)
    comm.barrier()


if __name__ == "__main__":
    test_vpot_vqe_single()
    test_vpot_vqe_mpi()
    test_vpot_vqe_mpigpu()
