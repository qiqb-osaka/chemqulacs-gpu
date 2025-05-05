##
# @file mp2.py
# @author Yusuke Teranishi
import time

import numpy as np
from pyscf import mp


class MP2:
    """MP2 Class"""

    def __init__(self, mf) -> None:
        """Constructor

        Args:
            mf: mean-field object
        """
        t = time.time()
        mymp = mp.MP2(mf).run()
        self.duration = time.time() - t

        self.n_ele = mf.mol.nelectron
        self.t2 = mymp.t2
        self.e_tot = mymp.e_tot

    def getAmplitude(self, exc):
        if len(exc) == 2:
            return 0.0
        elif len(exc) == 4:
            return (
                -self.t2[exc[0] // 2][exc[1] // 2][(exc[2] - self.n_ele) // 2][
                    (exc[3] - self.n_ele) // 2
                ]
                / 2
            )
        else:
            raise ValueError("exc must be a list of length 2 or 4.")

    def getAmplitudesFromExcitations(self, excitations):
        amplitudes = []
        for excitation_group in excitations:
            excitation = excitation_group[0]
            amplitudes.append(self.getAmplitude(excitation))
        return np.array(amplitudes)
