import os
import numpy as np
from pyscf import gto, scf, qmmm
from chemqulacs.util import utils
from chemqulacs.qmmm import electrostatic

# 水の分子構造をパブケムから取得
geom_water = utils.get_geometry_from_pubchem("water")
# Molecular class のオブジェクトを生成。基底関数や構造の情報を持っている。
mol = gto.M(atom=geom_water, basis="sto-3g")
# SCF波動関数のオブジェクトを生成。ここではRHFを使用する。
mf = scf.RHF(mol)
mf.verbose = 5
# SCF計算の実行, エネルギーが得られる(-74.96444758277)
mf.run()


# 静電ポテンシャルを計算する点の座標
coords = np.array(
    [
        [5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    ]
)
# 静電ポテンシャルを計算する点に、電荷１のチャージを挿入した、水分子を用意
# 電荷はなんでも良い
mf_qmmm = qmmm.mm_charge(mf, coords, np.ones_like(coords[:, 0]), unit="Bohr")
# 静電ポテンシャルの計算
vpot = electrostatic.vpot_pyscf(mf_qmmm)
# 参照値
ref_vpot = np.array([0.01588951, -0.02104672, 0.02036143, -0.01317634, -0.01166455, 0.00970652])
print("Computed Electrostatic potential")
print(vpot)
print("Reference Electrostatic potential")
print(ref_vpot)

# ORCAの形式で静電ポテンシャルを書き出す
fname = "vpot.out"
if os.path.isfile(fname):
    os.remove(fname)
electrostatic.write_vpot(mf_qmmm)
