import copy

import numpy as np
from pyscf import gto, mcscf, qmmm, scf

from chemqulacs.cpp.vqe import vqemcscf
from chemqulacs.qmmm import electrostatic
from chemqulacs.resp.resp_helper import helper_VDW_surface
from chemqulacs.util import utils

bohr_to_angstrom = 0.52917721092


def generate_surface(mol, doDebug=False):
    # vdw surface generation parameters
    n_vdw_layers = 4  # the number of vdW shells
    vdw_scale_factor = 1.4  # scaling factor
    vdw_increment = 0.2  # (1.4, 1.6, 1.8, 2.0)*(vdW radii)
    # vdw_point_density = 1.75 # This might reproduce the number of points of GAMESS.
    vdw_point_density = 1.00  #
    # for debug
    if doDebug:
        n_vdw_layers = 1  # the number of vdW shells
        vdw_scale_factor = 1.4  # scaling factor
        vdw_increment = 0.2  # (1.4, 1.6, 1.8, 2.0)*(vdW radii)
        vdw_point_density = 1.0

    # generate grid points on the vdw surface shells
    points = []
    surface = helper_VDW_surface()
    radii = {}
    for i in range(n_vdw_layers):
        scale_factor = vdw_scale_factor + i * vdw_increment
        surface.vdw_surface(
            mol.atom_coords() * bohr_to_angstrom,
            mol.elements,
            scale_factor,
            vdw_point_density,
            radii,
        )
        points.append(surface.shell)
    radii = surface.radii
    points = np.concatenate(points)  # in angstrom
    # print ("#ESP fitting points:", len(points))
    # quit()
    return points / bohr_to_angstrom  # in bohr


def esp_solve(a, b):
    """Solves for point charges: A*q = B

    Parameters
    ----------
    a : np.array
        array of matrix A
    b : np.array
        array of matrix B

    Return
    ------
    q : np.array
        array of charges
    """
    q = np.linalg.solve(a, b)
    # Warning for near singular matrix
    # in case np.linalg.solve does not detect singularity
    if np.linalg.cond(a) > 1 / np.finfo(a.dtype).eps:
        print("Warning, possible fit problem; singular matrix")
    return q


def compute_inv_riJ(points, coordinates, doDebug=False):
    # Build a matrix of the inverse distance from each ESP point to each nucleus
    inv_riJ = np.zeros((len(points), len(coordinates)))
    for i in range(inv_riJ.shape[0]):
        for j in range(inv_riJ.shape[1]):
            inv_riJ[i, j] = 1 / np.linalg.norm(points[i] - coordinates[j])
            if doDebug:
                print(np.linalg.norm(points[i] - coordinates[j]) * bohr_to_angstrom)
    return inv_riJ  # in bohr(esp_fit, nucleus)


def compute_A_without_restraint(inv_riJ):
    """compute_Aij

    Args:
      riJ: 1/r_{iJ} in bohr(esp_fit, nucleus)

    """
    natm = inv_riJ.shape[1]
    A = np.zeros((natm + 1, natm + 1))
    inv = inv_riJ.reshape((1, inv_riJ.shape[0], inv_riJ.shape[1]))
    A[:natm, :natm] = np.einsum("iwj, iwk -> jk", inv, inv)
    A[:natm, natm] = 1.0
    A[natm, :natm] = 1.0
    return A


def add_restraint(q, AwoR, resp_a, resp_b, esp_values):
    kai2_rst_der = kai2_rst_derivatives(q, resp_a, resp_b)
    natm = q.shape[0] - 1
    A = AwoR.copy()
    for i in range(natm):
        A[i, i] += kai2_rst_der[i]
    return A


def compute_B(inv_riJ, esp_values, q_tot):
    natm = inv_riJ.shape[1]
    B = np.zeros((natm + 1))
    B[:natm] = np.dot(esp_values, inv_riJ)
    B[natm] = q_tot
    return B


def espfit(inv_riJ, esp_values, q_tot):
    maxiter = 25
    toler = 1e-8
    resp_a = 0.0005
    resp_b = 0.1
    #
    niter = 0
    difm = 1.0
    natm = inv_riJ.shape[1]
    q = np.zeros((natm + 1))
    q_old = np.zeros((natm + 1))
    AwoR = compute_A_without_restraint(inv_riJ)
    B = compute_B(inv_riJ, esp_values, q_tot)
    while difm > toler and niter < maxiter:
        niter += 1
        difm = 0.0
        # print (q)
        A = add_restraint(q, AwoR, resp_a, resp_b, esp_values)
        q = esp_solve(A, B)
        #
        for i in range(len(q) - 1):
            dif = (q[i] - q_old[i]) ** 2
            if difm < dif:
                difm = dif
        q_old = copy.deepcopy(q)
        difm = np.sqrt(difm)
    # print ("niter", niter)
    if difm > toler:
        print(
            "Warning: Not Converged!! Try increasing the maximum number of iterations"
        )
    return q


def kai2_rst(q, resp_a, resp_b):
    kai2 = resp_a * (np.sqrt(q * q + resp_b * resp_b) - resp_b)
    return kai2


def kai2_rst_derivatives(q, resp_a, resp_b):
    kai2_der = resp_a / np.sqrt(q**2 + resp_b**2)
    return kai2_der


def compute_resp(mf, cwf=None):
    """compute_resp

    Args:
      mf: the pyscf wave function object
      cwf: correlaed wave function object

    Returns:
      charges: numpy 1D array of ESP charges

    """

    # get molecular object
    mol = mf.mol

    # Total molecular charges
    qtot = mol.charge

    # Generate vdW surface points
    points = generate_surface(mol)

    # Compute electrostatic potentials at each grid point
    # print (points.shape)
    mf_qmmm = qmmm.mm_charge(mf, points, np.zeros_like(points[:, 0]), unit="Bohr")
    esp_values = electrostatic.vpot_pyscf(mf_qmmm, cwf)

    # 'esp_values'->esp_values, 'inv_riJ'->riJ, 'coordinates'->points
    # 'mol_charge'->qtot

    # r_{iJ}^{-1}
    inv_riJ = compute_inv_riJ(points, mol.atom_coords())

    # Do ESP fit
    q_esp = espfit(inv_riJ, esp_values, qtot)

    return q_esp


if __name__ == "__main__":
    geom_water = utils.get_geometry_from_pubchem("water")
    mol = gto.M(atom=geom_water, basis="sto-3g")

    # compute at Hartree-Fock level
    mf = scf.RHF(mol)
    mf.run()
    q_hf = compute_resp(mf)

    # compute at CASCI level
    refmc = mcscf.CASCI(mf, 2, 2)
    refmc.run()
    q_cas = compute_resp(mf, refmc)

    # compute at VQE level
    mc = vqemcscf.VQECASCI(mf, 2, 2)
    mc.kernel()
    q_vqe = compute_resp(mf, mc)

    # check the ESP chareges
    print("RESP       HF/STO-3G", q_hf)
    print("RESP CAS(2,2)/STO-3G", q_cas)
    print("RESP VQE(2,2)/STO-3G", q_vqe)
