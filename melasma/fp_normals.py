"""
fp_normals.py -- Do ti le BAO NHAM tren tap anh 'normal' cua Roboflow da GIU LAI (khong mo hinh nao
tung dung de huan luyen), cho bat ky mo hinh nao -- ke ca mo hinh goc (chua thay anh normal nao).
Dung de so sanh cong bang mo hinh goc va mo hinh v2 (train_unet_neg.py).

Chay (cung cach chia nhu luc huan luyen; mac dinh seed 42, giu lai 30%):
  python fp_normals.py --neg-dir Pigmentation_Melasma_Dark_Spots_v1i_folder --ckpt runs/unet_r34/fold0/best.pt runs/unet_r34/fold1/best.pt runs/unet_r34/fold2/best.pt runs/unet_r34/fold3/best.pt runs/unet_r34/fold4/best.pt
  python fp_normals.py --neg-dir Pigmentation_Melasma_Dark_Spots_v1i_folder --ckpt runs/unet_neg/fold0/best.pt ...  (5 fold cua v2)

Moi mo hinh duoc do RIENG, roi lay trung binh. Chi so: % dien tich nam du doan tren anh normal
(cang thap cang tot), ti le anh bi bao >1% va >5%.
"""
import argparse
import json

import numpy as np
import torch

import neg_utils as N
from predict import load_model, predict_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--neg-heldout", type=float, default=0.3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save", default=None, help="luu ket qua ra file json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, held = N.load_normals(a.neg_dir, seed=42, heldout_frac=a.neg_heldout)
    print(f">>> {len(held)} anh normal giu lai | {len(a.ckpt)} mo hinh | {device}")

    per_model = []
    for c in a.ckpt:
        fwd, cfg = load_model(c, device)
        pcts = []
        for img, _, info in N.to_arrays(held, cfg["width"], cfg["height"]):
            pr = predict_prob(fwd, img, device, False)
            pcts.append(N.content_pct(pr > a.thr, info))
        s = N.fp_summary(pcts)
        per_model.append(s)
        print(f"  {c}: TB {s['mean']:.2f}% | trung vi {s['median']:.2f}% | >1%: {s['pct_gt1']:.0f}% | >5%: {s['pct_gt5']:.0f}%")

    keys = ["mean", "median", "pct_gt1", "pct_gt5"]
    avg = {k: float(np.mean([m[k] for m in per_model])) for k in keys}
    print(f"\nTRUNG BINH {len(per_model)} mo hinh: TB {avg['mean']:.2f}% | trung vi {avg['median']:.2f}% | "
          f"anh >1%: {avg['pct_gt1']:.0f}% | anh >5%: {avg['pct_gt5']:.0f}%")
    if a.save:
        json.dump({"average": avg, "per_model": per_model}, open(a.save, "w"), indent=1)


if __name__ == "__main__":
    main()
