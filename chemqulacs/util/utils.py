import json

import numpy as np
import pubchempy

bohr_to_angstrom = 0.52917721092


def read_vpot_xyz(fileobj):
    """read_vpot_xyz

    Read ORCA's vpot style file

    Args:
      fileobj: name of file where electrostatic information is written
    in the ORCA's-vpot-style.

    Returns:
      tuple(numpy array, numpy array): the first numpy array contains coordinates in bohr;
    the second numpy array contains potentials at each coordinate

    Examples:

    """
    lines = fileobj.readlines()
    coords = []
    vpots = []
    ncharges = int(lines.pop(0))
    for _ in range(ncharges):
        line = lines.pop(0)
        x, y, z, vpot = line.split()[:4]
        coords.append([float(x), float(y), float(z)])
        vpots.append(float(vpot))
    return np.array(coords), np.array(vpots)


def read_coords(fileobj):
    """read_coords

    Read a xyz-like-format file

    Args:
      fileobj: the name of a file where coordiante information is written in xyz-like format

    Returns:
      numpy array: two-dimensional arrays of coordinates

    Examples:

    """
    lines = fileobj.readlines()
    coords = []
    ncharges = int(lines.pop(0))
    for _ in range(ncharges):
        line = lines.pop(0)
        x, y, z = line.split()[:3]
        coords.append([float(x), float(y), float(z)])
    return np.array(coords)


def get_geometry_from_pubchem(name):
    """get_geometry_from_pubchem

    Get a geometry file from PubChem using pubchempy

    Args:
      name: name of molecule

    Returns:
      numpy array: two-dimensional arrays of coordinates

    Examples:

      >>> geom_water = get_geometry_from_pubchem('water')
      >>> mol = gto.M(atom=geom_water,basis='sto-3g')

    """
    pubchempy_molecule = pubchempy.get_compounds(name, "name", record_type="3d")
    pubchempy_geometry = pubchempy_molecule[0].to_dict(properties=["atoms"])["atoms"]
    geometry = [
        (atom["element"], (atom["x"], atom["y"], atom.get("z", 0)))
        for atom in pubchempy_geometry
    ]
    return geometry


def almost_equal(x, y, threshold=0.00000001):
    return abs(x - y) < threshold


def read_xyz(fileobj):
    """read_xyz

    Read a xyz-format file

    Args:
      fileobj: the name of a file where coordiante information is written in the xyz format

    Returns:
      list: list of cartesian coordinates with symbols

    Examples:

    """
    lines = fileobj.readlines()
    coords = []
    ncharges = int(lines.pop(0))
    lines.pop(0)
    for _ in range(ncharges):
        line = lines.pop(0)
        s, x, y, z = line.split()[:4]
        coords.append([s, float(x), float(y), float(z)])
    return coords


def write_xyz(fname, mol, comments="", fmt="%22.15f"):
    """write_xyz

    Write a geometry file in the xyz format

    Args:
      fname: the name of the file where coordinate information stored.
      mol: the PySCF molecule object
      comments: comments for the file
      fmt: this defines the format.

    Returns:

    Examples:
    """
    fileobj = open(fname, "w")
    natoms = mol.natm
    fileobj.write("%d\n%s\n" % (natoms, comments))
    for s, (x, y, z) in zip(
        mol.elements, mol.atom_coords() * bohr_to_angstrom, strict=True
    ):
        fileobj.write("%-2s %s %s %s\n" % (s, fmt % x, fmt % y, fmt % z))


def get_geometric_center(mol):
    centroid = np.einsum("ij->j", mol.atom_coords()) / mol.atom_coords().shape[0]
    return centroid


def set_centroid(mol):
    r_new = mol.atom_coords() - get_geometric_center(mol)
    mol.set_geom_(r_new, unit="Bohr")
    return mol


def numerical_grad(f, x, first_idx=None, last_idx=None, dx=1e-05):
    numgrad = np.zeros_like(x)
    x_new1 = np.copy(x)
    x_new2 = np.copy(x)
    for i in range(first_idx, last_idx):
        x_new1[i] += dx
        x_new2[i] -= dx
        numgrad[i] = (f(x_new1) - f(x_new2)) / (2 * dx)
        x_new1[i] -= dx
        x_new2[i] += dx
    return numgrad


def param_dump(param, filename):
    with open(filename, "w") as f:
        json.dump(param, f)


def param_load(filename):
    with open(filename, "w") as f:
        param = json.load(f)
        return param
