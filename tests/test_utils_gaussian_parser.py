# from chemqulacs.util import gaussian_parser, utils

# fname = "./sample_gaussian_input.inp"


# def test_gaussian_parser():
#     qm_atoms, mm_atoms = gaussian_parser.read_gaussian_in(open(fname, "r"))
#     print(qm_atoms)
#     print(mm_atoms)
#     assert len(qm_atoms[0]) == 3
#     assert len(mm_atoms[0]) == 6


# def test_get_coords_from_gaussian_input():
#     qm_geometry, mm_pos, mm_charges = gaussian_parser.get_coords_from_gaussian_input(
#         open(fname, "r")
#     )
#     from pyscf import gto, qmmm, scf

#     mol = gto.M(atom=qm_geometry, basis="cc-pVDZ")
#     mf = scf.RHF(mol)
#     mf_qmmm = qmmm.mm_charge(mf, mm_pos, mm_charges, unit="Agnstrom")
#     mf_qmmm.kernel()
#     energy = mf_qmmm.energy_tot()

#     assert utils.almost_equal(energy, -75.9857526101466, 1.0e-06)
