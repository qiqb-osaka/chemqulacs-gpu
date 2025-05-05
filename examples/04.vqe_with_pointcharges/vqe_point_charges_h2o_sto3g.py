import numpy as np
from pyscf import gto, scf, mcscf, qmmm
from chemqulacs.vqe import vqemcscf
from chemqulacs.util import utils

# 水の分子構造をパブケムから取得
geom_water = utils.get_geometry_from_pubchem("water")
# Molecular class のオブジェクトを生成。基底関数や構造の情報を持っている。
mol = gto.M(atom=geom_water, basis="sto-3g")
# SCF波動関数のオブジェクトを生成。ここではRHFを使用する。
mf = scf.RHF(mol)
# SCF計算の実行, エネルギーが得られる(-74.96444758277)
print("\n without point charges")
mf.run()


# 静電ポテンシャルを計算する点の座標
coords = np.array(
    [
        [3.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -3.0],
    ]
)
# 静電ポテンシャルを計算する点に、電荷1のチャージを挿入した、水分子を用意
mf_qmmm = qmmm.mm_charge(mf, coords, np.ones_like(coords[:, 0]), unit="Bohr")
# qmmmオブジェクトを使ってポイントチャージを考慮したSCF計算を実行する。
print("\n with point charges")
mf_qmmm.run()

# CASCIのオブジェクトを生成。qmmmオブジェクトを読み込ませる。
# 活性軌道にはCAS(2e,2o)を使用する
refmc = mcscf.CASCI(mf_qmmm, 2, 2)
refmc.run()


# 量子古典混合アルゴリズムであるVQEを使ったCASCIを実行する
# （より正確にはactive space disentangled unitary coupled cluster)
# 活性軌道にはCAS(2e,2o)を使用する
mc = vqemcscf.VQECASCI(mf_qmmm, 2, 2)
mc.kernel()

print("\nVQE Energy,   CASCI Energy")
print(mc.e_tot, refmc.e_tot)
