from pyscf import gto, scf, mcscf, mp, cc
from chemqulacs.util import utils

# 水の分子構造をパブケムから取得
geom_water = utils.get_geometry_from_pubchem("water")
# Molecular class のオブジェクトを生成。基底関数や構造の情報を持っている。
mol = gto.M(atom=geom_water, basis="sto-3g")
# SCF波動関数のオブジェクトを生成。ここではRHFを使用する。
mf = scf.RHF(mol)
# SCF計算の実行, エネルギーが得られる(-74.96444758277)
mf.run()

# MP2のオブジェクトを生成。ここではRHF波動関数を読み込ませて、その分子軌道を使用。
mymp2 = mp.MP2(mf)
mymp2.run()

# CCSDのオブジェクトを生成。ここではRHF波動関数を読み込ませて、その分子軌道を使用。
mycc = cc.CCSD(mf)
mycc.run()

# CASCIのオブジェクトを生成。ここではRHF波動関数を読み込ませて、その分子軌道を使用。
# 活性軌道にはCAS(2e,2o)を使用する
mymc = mcscf.CASCI(mf, 2, 2)
mymc.run()
