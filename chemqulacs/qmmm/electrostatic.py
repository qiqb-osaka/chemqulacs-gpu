import numpy as np
from pyscf import df, gto, lib


def vpot_nuc(mf_qmmm):
    """vpot_nuc

    Compute electrostatic potentials from QM nuclei at specified points
    where +1 charges are put.

    Args:
      mf_qmmm: mean-field wavefunction object of PySCF

    Returs:
      1D numpy arrays: electrostatic potentials from QM-region's nuclei

    Example:

      >>> vpot_nuc(mf_qmmm)

    """
    coords = mf_qmmm.mm_mol.atom_coords()
    charges = np.ones_like(mf_qmmm.mm_mol.atom_charges())
    vnuc = np.zeros_like(charges)
    # Mine
    # for j in range(mf_qmmm.mol.natm):
    #    q2, r2 = mf_qmmm.mol.atom_charge(j), mf_qmmm.mol.atom_coord(j)
    #    r = lib.norm(r2-coords, axis=1)
    #    vnuc += q2*(charges/r)

    # From chubegen.py
    # Nuclear potential at given points
    mol = mf_qmmm.mol
    for i in range(mol.natm):
        r = mol.atom_coord(i)
        Z = mol.atom_charge(i)
        rp = r - coords
        vnuc += Z / np.einsum("xi,xi->x", rp, rp) ** 0.5

    return vnuc


def vpot_elec(mf_qmmm, cwf=None):
    """vpot_elec

    Compute electrostatic potentials from QM-region's electron density at specified points

    Args:
      mf_qmmm: mean-field wavefunction object of PySCF with MM charges
      cwf: correlated wavefunction object of PySCF


    Returs:
      1D numpy arrays: electrostatic potentials from QM electron density

    Example:

      >>> vpot_elec(mf_qmmm)

    """
    coords = mf_qmmm.mm_mol.atom_coords()
    charges = np.ones_like(mf_qmmm.mm_mol.atom_charges(), dtype="float64")
    vpot_elec = np.zeros_like(charges, dtype="float64")
    if cwf is None:
        dm = mf_qmmm.make_rdm1()
    else:
        dm = cwf.make_rdm1()
    # Mine
    # for i, q in enumerate(charges):
    #    with mf_qmmm.mol.with_rinv_origin(coords[i]):
    #        v = mf_qmmm.mol.intor('int1e_rinv')
    #    f = np.einsum('ij,ij', dm, v) * -q
    #    vpot_elec[i] = f

    # From cubgen.py
    mol = mf_qmmm.mol
    #    for p0, p1 in lib.prange(0, vpot_elec.size, 600):
    #        fakemol = gto.fakemol_for_charges(coords[p0:p1])
    #        ints = df.incore.aux_e2(mol, fakemol)
    #        vpot_elec[p0:p1] = - np.einsum('ijp,ij->p', ints, dm)
    fakemol = gto.fakemol_for_charges(coords)
    ints = df.incore.aux_e2(mol, fakemol)
    vpot_elec = -np.einsum("ijp,ij->p", ints, dm)

    return vpot_elec


def vpot_qm_nuc(mf):
    """vpot_qm_nuc

    Compute electrostatic potentials at QM nuclei

    Args:
      mf: mean-field wavefunction object of PySCF

    Returs:
      1D numpy arrays: electrostatic potentials from QM-region's nuclei

    Example:

      >>> vpot_nuc(mf)

    """
    coords = mf.mol.atom_coords()
    charges = np.ones_like(mf.mol.atom_charges(), dtype="float64")
    vnuc_qm = np.zeros_like(charges, dtype="float64")
    for j in range(mf.mol.natm):
        q2, r2 = mf.mol.atom_charge(j), mf.mol.atom_coord(j)
        r = lib.norm(r2 - coords[:j], axis=1)
        vnuc_qm[:j] += q2 * (charges[:j] / r)
        r = lib.norm(r2 - coords[j + 1 :], axis=1)
        vnuc_qm[j + 1 :] += q2 * (charges[j + 1 :] / r)
    return vnuc_qm


def vpot_qm_elec(mf, cwf=None):
    """vpot_qm_elec

    Compute electrostatic potentials from QM-region's electron density at QM atoms

    Args:
      mf: mean-field wavefunction object of PySCF
      cwf: correlated wavefunction object of PySCF

    Returs:
      1D numpy arrays: electrostatic potentials from QM electron density

    Example:

      >>> vpot_elec(mf)

    """
    coords = mf.mol.atom_coords()
    charges = np.ones_like(mf.mol.atom_charges(), dtype="float64")
    vpot_qm_elec = np.zeros_like(charges, dtype="float64")
    if cwf is None:
        dm = mf.make_rdm1()
    else:
        dm = cwf.make_rdm1()
    for i, q in enumerate(charges):
        with mf.mol.with_rinv_origin(coords[i]):
            v = mf.mol.intor("int1e_rinv")
        f = np.einsum("ij,ij", dm, v) * -q
        vpot_qm_elec[i] = f
    return vpot_qm_elec


def vpot_qm_pyscf(mf, cwf=None):
    """vpot_elec

    Compute electrostatic potentials at QM atoms

    Args:
      mf: mean-field wavefunction object of PySCF
      cwf: correlated wavefunction object of PySCF

    Returs:
      1D numpy arrays: electrostatic potentials from QM

    Example:

      >>> vpot_qm_pyscf(mf)

    """
    return vpot_qm_nuc(mf) + vpot_qm_elec(mf, cwf)


def vpot_pyscf(mf, cwf=None):
    """vpot_elec

    Compute electrostatic potentials from QM-region at specified points

    Args:
      mf: mean-field wavefunction object of PySCF
      cwf: correlated wavefunction object of PySCF

    Returs:
      1D numpy arrays: electrostatic potentials from QM

    Example:

      >>> vpot_pyscf(mf)

    """
    return vpot_nuc(mf) + vpot_elec(mf, cwf)


def write_vpot(mf_qmmm, cwf=None, filename="vpot.out", fmt="%22.15f"):
    """write_vpot

    Write a electrostatic potentials into a file. ORCA's vpot style is used.

    Args:
      mf_qmmm: qmmm object of pyscf
      cwf: correlated wavefunction object of PySCF
      filename: name of a file where electrostatic potentials is written.
      Default is 'vpot'
      fmt: a string specify how many digits are written in the file.


    Returs:


    Example:

      >>> write_vpot(mf_qmmm)

    """
    coords = mf_qmmm.mm_mol.atom_coords()
    vpot = vpot_pyscf(mf_qmmm, cwf)
    fileobj = open("vpot.out", "w")
    for (x, y, z), q in zip(coords, vpot, strict=True):
        fileobj.write("%-2s %s %s %s\n" % (fmt % x, fmt % y, fmt % z, fmt % q))


def field_mm(mf, cwf=None):
    # This routine is originally comes from pyscf
    # ./pyscf/examples/qmmm/30-force_on_mm_particles.py
    if cwf is None:
        dm = mf.make_rdm1()
    else:
        dm = cwf.make_rdm1()
    coords = mf.mm_mol.atom_coords()
    charges = np.ones_like(mf.mm_mol.atom_charges(), dtype="float64")
    # The interaction between QM atoms and MM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
    qm_coords = mf.mol.atom_coords()
    qm_charges = mf.mol.atom_charges()
    dr = qm_coords[:, None, :] - coords
    r = np.linalg.norm(dr, axis=2)
    g = np.einsum("r,R,rRx,rR->Rx", qm_charges, charges, dr, r**-3)

    # The interaction between electron density and MM particles
    # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
    #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
    for i, q in enumerate(charges):
        with mf.mol.with_rinv_origin(coords[i]):
            v = mf.mol.intor("int1e_iprinv")
        f = (np.einsum("ij,xji->x", dm, v) + np.einsum("ij,xij->x", dm, v.conj())) * -q
        g[i] += f

    # Electric Field
    return -g


def field_qm(mf, cwf=None):
    if cwf is None:
        dm = mf.make_rdm1()
    else:
        dm = cwf.make_rdm1()
    # The interaction between QM atoms and QM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
    qm_coords = mf.mol.atom_coords()
    qm_charges = mf.mol.atom_charges()
    charges = np.ones_like(mf.mol.atom_charges(), dtype="float64")
    g = np.zeros_like(mf.mol.atom_coords())
    for i, qi in enumerate(charges):
        ri = qm_coords[i]
        for j, qj in enumerate(qm_charges):
            rj = qm_coords[j]
            if i != j:
                dr = rj - ri
                r = np.linalg.norm(dr)
                g[i, :] += qi * qj * dr * (r**-3)

    # The interaction between electron density and MM particles
    # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
    #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
    for i, q in enumerate(charges):
        with mf.mol.with_rinv_origin(qm_coords[i]):
            v = mf.mol.intor("int1e_iprinv")
        f = (np.einsum("ij,xji->x", dm, v) + np.einsum("ij,xij->x", dm, v.conj())) * -q
        g[i] += f

    # Electric Field
    return -g


def compute_electric_field(mf, cwf=None):
    return np.vstack((field_qm(mf, cwf), field_mm(mf, cwf)))


def compute_mm_selfenergy(mf):
    coords = mf.mm_mol.atom_coords()
    charges = mf.mm_mol.atom_charges()
    vnuc = 0.0
    for j in range(len(coords)):
        q1, r1 = charges[j], coords[j]
        r = lib.norm(r1 - coords[j + 1 :], axis=1)
        vnuc += np.sum(q1 * (charges[j + 1 :] / r))
    return vnuc
