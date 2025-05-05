# type:ignore
import os

import numpy as np
from pyscf import gto, qmmm, scf

from chemqulacs.qmmm import electrostatic
from chemqulacs.util import utils
from chemqulacs.vqe import vqemcscf

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
geom_water = utils.get_geometry_from_pubchem("water")


mol = gto.M(atom=geom_water, basis="sto-3g", unit="angstrom")
mf = scf.RHF(mol)
mf.run()
mc = vqemcscf.VQECASCI(mf, 2, 2)
mc.kernel()
mf_qmmm = qmmm.mm_charge(mf, coords, np.ones_like(coords[:, 0]), unit="Bohr")


def test_vpot_vqe():
    vpot = electrostatic.vpot_pyscf(mf_qmmm, mc)
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
    assert utils.almost_equal(mf.e_tot, -74.964447583028)
    assert utils.almost_equal(mc.e_tot, -74.965695110031)
    assert utils.almost_equal(vpot, ref_array, threshold=1.0e-6).prod()


test_vpot_vqe()


def test_write_vpot():
    fname = "vpot.out"
    if os.path.isfile(fname):
        os.remove(fname)
    electrostatic.write_vpot(mf_qmmm, mc)
    assert os.path.isfile(fname)
    os.remove(fname)
