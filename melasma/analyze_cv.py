"""
analyze_cv.py -- Gop ket qua 5-fold (moi anh duoc test dung 1 lan) va xem mo hinh sai o dau.

Chay:
  python analyze_cv.py --runs runs/segformer_b0 --split memi_processed/split.csv

Doc  : <runs>/fold*/per_image_test.csv  (do train_segformer.py --all-folds tao ra)
Ghi  : <runs>/cv_analysis.csv  (bang tung anh da gop thong tin type/view/fold)
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def show(title, df):
    print(f"\n--- {title} ---")
    print(df.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/segformer_b0")
    ap.add_argument("--split", default="memi_processed/split.csv")
    a = ap.parse_args()

    files = sorted(glob.glob(str(Path(a.runs) / "fold*" / "per_image_test.csv")))
    if not files:
        raise SystemExit(f"Khong thay per_image_test.csv trong {a.runs}/fold*/")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d["fold"] = Path(f).parent.name
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    meta = pd.read_csv(a.split)[["name", "group", "type", "view"]]
    d = d.merge(meta, on="name", how="left")
    d.to_csv(Path(a.runs) / "cv_analysis.csv", index=False)

    print(f"Tong so anh da test (moi anh 1 lan): {len(d)} | so fold: {d['fold'].nunique()}")
    print(f"Dice trung binh theo anh: {d.dice.mean():.4f} +/- {d.dice.std():.4f} | "
          f"trung vi {d.dice.median():.4f}")
    print(f"IoU {d.iou.mean():.4f} | Precision {d.precision.mean():.4f} | Recall {d.recall.mean():.4f}")

    ratio = d.pred_pct.sum() / max(d.gt_pct.sum(), 1e-9)
    print(f"Tong dien tich du doan / tong dien tich nhan that = {ratio:.2f}  "
          f"({'du doan THUA (to hon nhan)' if ratio > 1.1 else 'du doan THIEU (nho hon nhan)' if ratio < 0.9 else 'xap xi nhau'})")

    agg = {"n": ("dice", "size"), "dice": ("dice", "mean"), "iou": ("iou", "mean"),
           "precision": ("precision", "mean"), "recall": ("recall", "mean"), "gt_pct": ("gt_pct", "mean")}
    show("Theo nhom benh (Flaky = mang, Punctate = cham)", d.groupby("type").agg(**agg))
    show("Theo goc chup (F = chinh dien, L/R = nghieng)", d.groupby("view").agg(**agg))
    show("Theo fold", d.groupby("fold").agg(**agg))

    d["kich_thuoc_nam"] = pd.cut(d.gt_pct, [0, 3, 8, 15, 100], labels=["<3%", "3-8%", "8-15%", ">15%"])
    show("Theo kich thuoc vung nam that (% anh)", d.groupby("kich_thuoc_nam", observed=True).agg(**agg))

    corr = d.gt_pct.rank().corr(d.dice.rank())
    print(f"\nTuong quan (hang) giua kich thuoc vung nam va Dice: {corr:.2f}  "
          "(duong lon = vung nho thi Dice thap)")

    worst = d.sort_values("dice").head(10)[["name", "dice", "precision", "recall", "pred_pct", "gt_pct"]]
    show("10 anh Dice thap nhat (hay mo anh overlay cua cac anh nay)", worst.set_index("name"))
    print(f"\nDa luu bang gop: {Path(a.runs) / 'cv_analysis.csv'}")


if __name__ == "__main__":
    main()
