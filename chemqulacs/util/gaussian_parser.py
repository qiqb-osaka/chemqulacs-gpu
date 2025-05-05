import re

import numpy as np

# Based on gaussina.py of the ASE package
_re_chgmult = re.compile(r"^\s*[+-]?\d+(?:,\s*|\s+)[+-]?\d+\s*$")


def read_gaussian_in(fd):
    qm_symbols = []
    qm_positions = []
    mm_charges = []
    mm_positions = []
    # We're looking for charge and multiplicity
    for line in fd:
        # QM Regions
        if _re_chgmult.match(line) is not None:
            tokens = fd.readline().split()
            while tokens:
                qm_symbol = tokens[0]
                qm_pos = list(map(float, tokens[1:4]))
                qm_symbols.append(qm_symbol)
                qm_positions.append(qm_pos)
                tokens = fd.readline().split()
            # MM Regions
            tokens = fd.readline().split()
            while tokens:
                mm_charge = float(tokens[3])
                mm_pos = list(map(float, tokens[0:3]))
                mm_charges.append(mm_charge)
                mm_positions.append(mm_pos)
                tokens = fd.readline().split()
            # Return the information
            qm_atoms = (qm_symbols, qm_positions)
            mm_atoms = (mm_charges, mm_positions)
            return qm_atoms, mm_atoms


def get_coords_from_gaussian_input(fd):
    qm_atoms, mm_atoms = read_gaussian_in(fd)
    qm_geometry = [
        (symb, (pos[0], pos[1], pos[2]))
        for symb, pos in zip(qm_atoms[0], qm_atoms[1], strict=True)
    ]
    mm_charges = np.array(mm_atoms[0])
    mm_coords = np.array(mm_atoms[1])
    return qm_geometry, mm_coords, mm_charges


if __name__ == "__main__":
    import sys

    fname = sys.argv[1]
    qm_atoms, mm_atoms = read_gaussian_in(open(fname, "r"))
    print(qm_atoms)
    print(mm_atoms)
