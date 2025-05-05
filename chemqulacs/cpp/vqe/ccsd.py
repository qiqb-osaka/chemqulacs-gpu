##
# @file ccsd.py
# @author Yusuke Teranishi
import time

import numpy as np
from pyscf import cc


class CCSD:
    """CCSD class"""

    def __init__(self, mf) -> None:
        """Constructor

        Args:
            mf (_type_): mean-field object
        """
        t = time.time()
        mycc = cc.CCSD(mf).run()
        self.duration = time.time() - t

        self.n_ele = mf.mol.nelectron
        self.t1 = mycc.t1
        self.t2 = mycc.t2
        self.e_tot = mycc.e_tot

    def getAmplitude(self, exc):
        ##
        # @brief Get amplitude
        # @param exc: excitation
        # @return amplitude
        if len(exc) == 2:
            return -self.t1[exc[0] // 2][(exc[1] - self.n_ele) // 2]
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
        ##
        # @brief Get amplitudes from excitations
        # @param excitations: list of excitations
        # @return amplitudes
        amplitudes = []
        for excitation_group in excitations:
            excitation = excitation_group[0]
            amplitudes.append(self.getAmplitude(excitation))
        return np.array(amplitudes)
