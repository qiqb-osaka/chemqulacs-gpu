# type:ignore
import time

from pyscf import cc, gto, mcscf, mp, scf

from chemqulacs.util import utils
from chemqulacs.cpp.vqe import vqeci, vqemcscf
# from chemqulacs.vqe import vqeci, vqemcscf

# 水の分子構造をパブケムから取得
geom_water = utils.get_geometry_from_pubchem("water")
print(geom_water)
# Molecular class のオブジェクトを生成。基底関数や構造の情報を持っている。
mol = gto.M(atom=geom_water, basis="sto-3g")

# SCF波動関数のオブジェクトを生成。ここではRHFを使用する。
mf = scf.RHF(mol)
# SCF計算の実行, エネルギーが得られる(-74.96444758277)
mf.run()
ele = 6
orb = 10
# CASCIのオブジェクトを生成。ここではRHF波動関数を読み込ませて、その分子軌道を使用。
# 活性軌道にはCAS(2e,2o)を使用する
refmc = mcscf.CASCI(mf, ele, orb)
refmc.run()

s = time.time()
print(s)
mc = vqemcscf.VQECASCI(
    mf, ele, orb, ansatz_name="gatefabric", is_init_random=True, layers=ele
)
mc.kernel()
e = time.time()
print("time:", e - s)
print("VQE Energy,   CASCI Energy")
# print(mc.e_tot, refmc.e_tot)
print(mc.e_tot)
# print(refmc.e_tot)
