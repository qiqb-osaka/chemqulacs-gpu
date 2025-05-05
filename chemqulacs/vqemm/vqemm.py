from chemqulacs.cpp.vqe import vqemcscf


class VQECASCIMM(vqemcscf.VQECASCI):
    def __init__(self, mf, ncas, nelecas, ncore=None):
        vqemcscf.VQECASCI(mf, ncas, nelecas, ncore=None)
        # self._scf is mf
