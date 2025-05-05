import numpy as np

from pyscf import gto, scf, qmmm, mcscf

from chemqulacs.util import gaussian_parser
from chemqulacs.util import utils
from chemqulacs.vqe import vqemcscf
from chemqulacs.qmmm import electrostatic
from chemqulacs.resp import resp


if __name__ == "__main__":
    fname = "sample_gaussian_input.inp"
    qm_geometry, mm_pos, mm_charges = gaussian_parser.get_coords_from_gaussian_input(fname)
    mol = gto.M(atom=qm_geometry, basis="cc-pVDZ")
    # do RHF without MM charges using RI approximation
    #    mf = scf.RHF(mol).density_fit()
    mf = scf.RHF(mol)
    mf.run()
    # do RHF with MM charges
    mf_qmmm = qmmm.mm_charge(mf, mm_pos, mm_charges, unit="Agnstrom")
    mf_qmmm.kernel()
    # do VQE with MM charges
    mc = vqemcscf.VQECASCI(mf_qmmm, 2, 2)
    mc = mcscf.CASCI(mf_qmmm, 4, 4)
    mc.kernel()

    # compute the ESP charges
    q_vqe = resp.compute_resp(mf, mc)  # do not use mf_qmmm

    # compute the electrostatic potentials at mm charges
    vpot = electrostatic.vpot_pyscf(mf_qmmm, cwf=None)  # MM regions
    vpot_qm = electrostatic.vpot_qm_pyscf(mf_qmmm, cwf=None)  # QM regions

    # save electrostatic potentials
    electrostatic.write_vpot(mf_qmmm, mc)
    electrostatic.write_vpot(mf_qmmm)

    # compute_electric_field
    field = electrostatic.compute_electric_field(mf_qmmm, mc)
    field = electrostatic.compute_electric_field(mf_qmmm)

    # compute self-energy of the chages
    mm_self_energy = electrostatic.compute_mm_selfenergy(mf_qmmm)

    # get dipole moment in atomic unit
    dip = mf_qmmm.dip_moment(unit="AU")

    # write down solute's coordinates in xyz format
    utils.write_xyz("solute.xyz", mol)

    # comptue grad
    mf.nuc_grad_method().run()
    mf_qmmm.nuc_grad_method().run()
    mf_g_scanner = mf_qmmm.nuc_grad_method().as_scanner()
    mf_g = mf_g_scanner(mol)
    mc_g_scanner = mc.nuc_grad_method().as_scanner()
    mc_g = mc_g_scanner(mol)
    mc_g = mf_g

    vpot_all = np.concatenate([vpot_qm, vpot])
    print("\nDipole in au", dip)
    print("\nESP chages", q_vqe[:-1])
    print("Self energy of the charges =  %15.10f a.u. " % mm_self_energy)
    print("    Center     Electric         -------- Electric Field --------")
    print("               Potential          X             Y             Z")
    print("-----------------------------------------------------------------")

    for idx, ivpot in enumerate(vpot_qm):
        print(
            "%5i Atom %13.6f%14.6f%14.6f%14.6f"
            % (idx + 1, ivpot, field[idx][0], field[idx][1], field[idx][2])
        )
    for jdx, ivpot in enumerate(vpot):
        print(
            "%5i     %14.6f%14.6f%14.6f%14.6f"
            % (
                jdx + 30001,
                ivpot,
                field[idx + jdx + 1][0],
                field[idx + jdx + 1][1],
                field[idx + jdx + 1][2],
            )
        )
    print("-----------------------------------------------------------------")

    print(mm_self_energy + mf_qmmm.e_tot)
    tot_ene = mm_self_energy + mf_qmmm.e_tot
    print("Total Energy                               R    %23.15E\n" % tot_ene)
    nesp = len(q_vqe[:-1])
    print("ESP Charges                                R   N=%12i" % nesp)
    nline = nesp // 5
    for iline in range(nline):
        ioff = iline * 5
        print(
            "%16.8E%16.8E%16.8E%16.8E"
            % (q_vqe[ioff], q_vqe[ioff + 1], q_vqe[ioff + 2], q_vqe[ioff + 3], q_vqe[ioff + 4])
        )
    lastline = ""
    for iterm in range(np.mod(nesp, 5)):
        ioff = nline * 5
        lastline += "%16.8E" % (q_vqe[ioff + iterm])
    print(lastline)

    mc_g = mc_g[1].flatten()
    ngrad = len(mc_g)
    print("Cartesian Gradient                         R   N=%12i" % ngrad)
    nline = ngrad // 5
    for iline in range(nline):
        ioff = iline * 5
        print(
            "%16.8E%16.8E%16.8E%16.8E%16.8E"
            % (mc_g[ioff], mc_g[ioff + 1], mc_g[ioff + 2], mc_g[ioff + 3], mc_g[ioff + 4])
        )
    lastline = ""
    for iterm in range(np.mod(ngrad, 5)):
        ioff = nline * 5
        lastline += "%16.8E" % (mc_g[ioff + iterm])
    print(lastline)

    print("Dipole Moment                              R   N=%12i" % 3)
    lastline = ""
    for iterm in range(3):
        lastline += "%16.8E" % (dip[iterm])
    print(lastline)
