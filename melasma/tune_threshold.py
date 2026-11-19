"""
tune_threshold.py -- Chon nguong phan vung tot nhat (va thu TTA) cho cac mo hinh 5-fold.

Cach lam dung (khong gian lan): voi moi fold, nguong duoc CHON tren cac anh val (benh nhan
khac voi fold test), roi moi BAO CAO tren cac anh test. Khong dung test de chon nguong.

Chay:
  python tune_threshold.py --runs runs/segformer_b0 --data memi_processed
  python tune_threshold.py --runs runs/segformer_b0 --data memi_processed --tta

Ghi: <runs>/thr_sweep[_tta].csv va <runs>/best_threshold[_tta].json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

import memi_common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/segformer_b0")
    p.add_argument("--data", default="memi_processed")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--lo", type=float, default=0.30)
    p.add_argument("--hi", type=float, default=0.80)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--tta", action="store_true", help="trung binh voi anh lat ngang")
    return p.parse_args()


@torch.no_grad()
def prob_map(model, img, device, tta):
    x = torch.from_numpy(C.normalize(img)).unsqueeze(0).to(device)

    def run(t):
        lg = model(pixel_values=t).logits
        lg = torch.nn.functional.interpolate(lg, size=t.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(lg.float())[0, 0]

    p = run(x)
    if tta:
        p = 0.5 * (p + torch.flip(run(torch.flip(x, dims=[-1])), dims=[-1]))
    return p.cpu().numpy()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = np.round(np.arange(a.lo, a.hi + 1e-9, a.step), 3)
    print(f">>> Thiet bi {device} | luoi nguong {grid.tolist()} | TTA={a.tta}")

    val_d, test_d, test_p, test_r = [], [], [], []   # moi phan tu: mang theo nguong, 1 anh
    for k in range(a.n_folds):
        fd = Path(a.runs) / f"fold{k}"
        if not (fd / "best_model").exists():
            raise SystemExit(f"Thieu {fd / 'best_model'} -- hay chay train_segformer.py --all-folds truoc")
        cfg = json.loads((fd / "run_config.json").read_text())
        W, H = cfg["width"], cfg["height"]
        _, va_n, te_n = C.load_split(Path(a.data) / "split.csv", k, a.n_folds)
        data = C.load_arrays(a.data, va_n + te_n, W, H)
        model = SegformerForSemanticSegmentation.from_pretrained(fd / "best_model").to(device).eval()
        print(f"    fold {k}: {len(va_n)} anh val, {len(te_n)} anh test")

        for names, is_test in ((va_n, False), (te_n, True)):
            for n in names:
                img, gt = data[n]
                prob = prob_map(model, img, device, a.tta)
                ms = [C.seg_metrics(prob > t, gt) for t in grid]
                if is_test:
                    test_d.append([m["dice"] for m in ms])
                    test_p.append([m["precision"] for m in ms])
                    test_r.append([m["recall"] for m in ms])
                else:
                    val_d.append([m["dice"] for m in ms])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_d, test_d, test_p, test_r = map(np.array, (val_d, test_d, test_p, test_r))
    vm, tm, pm, rm = val_d.mean(0), test_d.mean(0), test_p.mean(0), test_r.mean(0)
    best = int(np.argmax(vm))
    i05 = int(np.argmin(np.abs(grid - 0.5)))

    print(f"\n{'nguong':>7} {'val Dice':>9} {'test Dice':>10} {'test Prec':>10} {'test Rec':>9}")
    for i, t in enumerate(grid):
        mark = "  <- chon tren val" if i == best else ("  (mac dinh)" if i == i05 else "")
        print(f"{t:7.2f} {vm[i]:9.4f} {tm[i]:10.4f} {pm[i]:10.4f} {rm[i]:9.4f}{mark}")
    print(f"\nNguong chon tren val: {grid[best]:.2f}")
    print(f"Test Dice: {tm[best]:.4f} (nguong da chon) so voi {tm[i05]:.4f} (nguong 0.5) "
          f"-> chenh {tm[best] - tm[i05]:+.4f}  [{len(test_d)} anh]")
    print("Luu y: chenh duoi ~0.01 la nam trong nhieu do (tap test nho), khong nen doi nguong vi no.")

    tag = "_tta" if a.tta else ""
    with open(Path(a.runs) / f"thr_sweep{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["thr", "val_dice", "test_dice", "test_precision", "test_recall"])
        for i, t in enumerate(grid):
            w.writerow([t, f"{vm[i]:.5f}", f"{tm[i]:.5f}", f"{pm[i]:.5f}", f"{rm[i]:.5f}"])
    (Path(a.runs) / f"best_threshold{tag}.json").write_text(
        json.dumps({"thr": float(grid[best]), "tta": a.tta, "test_dice_at_thr": float(tm[best]),
                    "test_dice_at_0.5": float(tm[i05])}, indent=1))


if __name__ == "__main__":
    main()
