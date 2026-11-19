"""
compare_models.py -- So sanh cong bang nhieu mo hinh da chay tren CUNG 5 fold.

Chay (mo hinh dau tien la mo hinh goc de so sanh):
  python compare_models.py --runs runs/segformer_b0 runs/unet_r34 runs/yolo11n_seg --split memi_processed/split.csv

Doc  : <run>/fold*/per_image_test.csv
Ghi  : compare_per_image.csv (bang tung anh, moi mo hinh mot cot Dice)

Vi sao khong chi nhin trung binh: tap rat nho va cac anh cua cung 1 benh nhan khong doc lap.
Nen khoang tin cay cua CHENH LECH Dice duoc tinh bang bootstrap THEO BENH NHAN (resample ca nguoi).
Neu khoang tin cay 95% chua so 0 thi chua the ket luan mo hinh nao tot hon.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(path):
    files = sorted(glob.glob(str(Path(path) / "fold*" / "per_image_test.csv")))
    if not files:
        raise SystemExit(f"Khong thay per_image_test.csv trong {path}/fold*/")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d["fold"] = Path(f).parent.name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def cluster_bootstrap(diff, groups, n=5000, seed=0):
    """Khoang tin cay 95% cua trung binh chenh lech, resample theo benh nhan."""
    g = pd.Series(np.asarray(diff)).groupby(np.asarray(groups))
    sums, cnts = g.sum().values, g.count().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="memi_processed/split.csv")
    a = ap.parse_args()

    labels = [Path(r).name for r in a.runs]
    runs = [load_run(r) for r in a.runs]
    common = set(runs[0].name)
    for r in runs[1:]:
        common &= set(r.name)
    runs = [r[r.name.isin(common)].sort_values("name").reset_index(drop=True) for r in runs]
    print(f"So anh chung giua cac mo hinh: {len(common)}")

    meta = pd.read_csv(a.split)[["name", "group", "type", "view"]]
    base = runs[0][["name", "gt_pct"]].merge(meta, on="name", how="left")
    base["kich_thuoc"] = pd.cut(base.gt_pct, [0, 3, 8, 15, 100], labels=["<3%", "3-8%", "8-15%", ">15%"])
    for lab, r in zip(labels, runs):
        base[f"dice_{lab}"] = r.dice.values
        base[f"pred_{lab}"] = r.pred_pct.values
    base.to_csv("compare_per_image.csv", index=False)

    print("\n--- Tong quan (trung binh theo anh) ---")
    rows = []
    for lab, r in zip(labels, runs):
        fold_means = r.groupby("fold").dice.mean()
        rows.append({"mo_hinh": lab, "dice": r.dice.mean(), "std_anh": r.dice.std(),
                     "dice_fold_tb": fold_means.mean(), "std_fold": fold_means.std(ddof=0),
                     "iou": r.iou.mean(), "precision": r.precision.mean(), "recall": r.recall.mean(),
                     "du_doan/that": r.pred_pct.sum() / max(r.gt_pct.sum(), 1e-9)})
    print(pd.DataFrame(rows).set_index("mo_hinh").round(3).to_string())

    for col, title in (("type", "loai nam"), ("view", "goc chup"), ("kich_thuoc", "kich thuoc vung nam")):
        t = base.groupby(col, observed=True)[[f"dice_{l}" for l in labels]].mean()
        t.columns = labels
        t.insert(0, "n", base.groupby(col, observed=True).size())
        print(f"\n--- Dice theo {title} ---")
        print(t.round(3).to_string())

    print(f"\n--- So voi mo hinh goc ({labels[0]}): chenh lech Dice, bootstrap theo benh nhan ---")
    for lab in labels[1:]:
        diff = base[f"dice_{lab}"] - base[f"dice_{labels[0]}"]
        lo, hi = cluster_bootstrap(diff.values, base.group.values)
        per_pat = diff.groupby(base.group.values).mean()
        verdict = "CHUA ket luan duoc (khoang tin cay chua 0)" if lo <= 0 <= hi else \
                  ("tot hon co y nghia" if lo > 0 else "kem hon co y nghia")
        print(f"{lab:>20}: chenh {diff.mean():+.3f}  KTC95% [{lo:+.3f}, {hi:+.3f}]  "
              f"tot hon o {(per_pat > 0).mean() * 100:.0f}% benh nhan -> {verdict}")
    print("\nDa luu bang tung anh: compare_per_image.csv")


if __name__ == "__main__":
    main()
