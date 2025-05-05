import os

from pyscf import gto

from chemqulacs.util import utils


def test_almost_equal():
    assert utils.almost_equal(1.000000001, 1.000000000)


def test_get_geometry_from_pubchem():
    geom_water = utils.get_geometry_from_pubchem("water")
    from pyscf import gto, scf

    mol = gto.M(atom=geom_water, basis="sto-3g")
    mf = scf.RHF(mol).density_fit()
    mf.run()
    energy = mf.energy_tot()
    assert utils.almost_equal(energy, -74.9645350309677)


def test_write_xyz():
    geom_water = utils.get_geometry_from_pubchem("water")

    mol = gto.M(atom=geom_water, basis="sto-3g")
    fname = "test_water.xyz"

    if os.path.isfile(fname):
        os.remove(fname)
    utils.write_xyz(fname, mol)
    os.remove(fname)
