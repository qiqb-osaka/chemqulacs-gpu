import numpy as np
from pyscf import gto, scf, qmmm, mcscf

from chemqulacs.util import utils
from chemqulacs.resp import resp
from chemqulacs.vqe import vqemcscf

if __name__ == "__main__":
    # 水の構造をPubchemからとってくる
    geom_water = utils.get_geometry_from_pubchem("water")
    mol = gto.M(atom=geom_water, basis="sto-3g")

    # Hartree-Fock計算を実行
    mf = scf.RHF(mol)
    mf.run()
    # Hartree-Fockのオブジェクトmfからrespを計算
    # 注意：このときのmfは、mmチャージの情報を持っていてはいけない
    q_hf = resp.compute_resp(mf)

    # CASCI(2,2)のオブジェクトrefmfからrespを計算
    # この時、compute_respは、mfとrefmfの双方が必要
    # ＊mfは、mmチャージの情報を持っていてはいけない
    refmc = mcscf.CASCI(mf, 2, 2)
    refmc.run()
    q_cas = resp.compute_resp(mf, refmc)

    # VQE(2e,2o)のオブジェクトmcからrespを計算
    # この時、compute_respは、mfとmcの双方が必要
    # ＊mfは、mmチャージの情報を持っていてはいけない
    mc = vqemcscf.VQECASCI(mf, 2, 2)
    mc.kernel()
    q_vqe = resp.compute_resp(mf, mc)

    # 手法ごとのRESPの値をチェック
    print("RESP       HF/STO-3G", q_hf)
    print("RESP CAS(2,2)/STO-3G", q_cas)
    print("RESP VQE(2,2)/STO-3G", q_vqe)
