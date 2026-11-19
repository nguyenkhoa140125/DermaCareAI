"""
train_yolo_seg.py -- Baseline YOLO-seg (Ultralytics) tren MEMI-DS, cung 5 fold, cung chi so.

YOLO-seg la mo hinh phan vung THEO TUNG VUNG (instance). De so sanh cong bang, ta GOP moi vung
du doan thanh 1 mask nhi phan roi tinh Dice/IoU y het SegFormer/U-Net. Nguong tin cay (conf)
duoc chon tren tap val, bao cao tren tap test.

Can cai them:  pip install ultralytics      (xong kiem tra lai: python -c "import torch;print(torch.cuda.is_available())")
Can da chay:   prepare_memi.py --yolo  (de co thu muc memi_processed/yolo_labels_tmp)

Vi du:
  python train_yolo_seg.py --smoke
  python train_yolo_seg.py --data memi_processed --all-folds --out runs/yolo11n_seg
  python train_yolo_seg.py --data memi_processed --all-folds --out runs/yolo11n_seg --resume

Mac dinh tat mosaic (ghep 4 anh) vi no pha vo bo cuc khuon mat; co the thu --mosaic 1.0 de doi chieu.
GTX 10xx khong co tensor core: fp16 co the khong nhanh hon; thu --no-amp neu thay cham.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import memi_common as C

CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="memi_processed")
    p.add_argument("--out", default="runs/yolo11n_seg")
    p.add_argument("--model", default="yolo11n-seg.pt", help="vd yolo11n-seg.pt | yolo11s-seg.pt")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def build_fold_dataset(data, fold, n_folds, root, W, H, subset=None):
    """Tao thu muc dung dinh dang YOLO cho 1 fold (anh da resize ve W x H, nhan da chuan hoa 0..1)."""
    tr, va, te = C.load_split(Path(data) / "split.csv", fold, n_folds)
    if subset:
        tr, va, te = tr[:subset], va[:max(2, subset // 2)], te[:max(2, subset // 2)]
    lab_src = Path(data) / "yolo_labels_tmp"
    if not lab_src.exists():
        raise SystemExit(f"Thieu {lab_src}. Chay lai: python prepare_memi.py --raw ... --out {data} --yolo")
    d = Path(root) / (f"fold{fold}" if fold >= 0 else "fixed")
    if d.exists():
        shutil.rmtree(d)
    for split, names in (("train", tr), ("val", va), ("test", te)):
        (d / "images" / split).mkdir(parents=True)
        (d / "labels" / split).mkdir(parents=True)
        for n in names:
            img = cv2.imread(str(Path(data) / "images" / f"{n}.jpg"))
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / "images" / split / f"{n}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy(lab_src / f"{n}.txt", d / "labels" / split / f"{n}.txt")
    yaml = d / "data.yaml"
    yaml.write_text(f"path: {d.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                    f"test: images/test\nnames:\n  0: melasma\n")
    return yaml, tr, va, te


def raw_predict(model, img_rgb, imgsz, device):
    """Tra ve (masks bool [n,H,W], conf [n]) o conf rat thap de sau do loc theo nhieu nguong."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    r = model.predict(bgr, imgsz=imgsz, conf=min(CONF_GRID), retina_masks=True, verbose=False, device=device)[0]
    H, W = img_rgb.shape[:2]
    if r.masks is None or len(r.masks.data) == 0:
        return np.zeros((0, H, W), bool), np.zeros(0, np.float32)
    ms = r.masks.data.cpu().numpy() > 0.5
    if ms.shape[1:] != (H, W):
        ms = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for m in ms])
    return ms, r.boxes.conf.cpu().numpy()


def union_at(ms, cf, conf, shape):
    keep = cf >= conf
    if ms.shape[0] == 0 or not keep.any():
        return np.zeros(shape, bool)
    return ms[keep].any(0)


def run_fold(args, fold, all_data, device, imgsz):
    from ultralytics import YOLO

    out = (Path(args.out) / (f"fold{fold}" if fold >= 0 else "fixed")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    yaml, tr_n, va_n, te_n = build_fold_dataset(args.data, fold, args.n_folds, out / "dataset",
                                                args.width, args.height, subset=6 if args.smoke else None)
    print(f"\n=== YOLO-seg {out.name}: train {len(tr_n)} | val {len(va_n)} | test {len(te_n)} anh ===")

    wdir = out / "train" / "weights"
    kw = dict(data=str(yaml), imgsz=imgsz, epochs=args.epochs, batch=args.batch, patience=args.patience,
              device=device, workers=args.workers, project=str(out.resolve()), name="train", exist_ok=True,
              seed=args.seed, amp=not args.no_amp, plots=False, mosaic=args.mosaic)
    if args.resume and (wdir / "last.pt").exists():
        try:
            YOLO(str(wdir / "last.pt")).train(resume=True)
        except Exception as e:  # run da xong -> khong resume duoc, dung best.pt co san
            print(f">>> Khong resume duoc ({str(e)[:90]}...). Dung best.pt co san neu co.")
    elif not (args.resume and (wdir / "best.pt").exists()):
        YOLO(args.model).train(**kw)
    (out / "run_config.json").write_text(json.dumps(
        {"arch": "yolo-seg", "model": args.model, "width": args.width, "height": args.height, "imgsz": imgsz}, indent=1))

    model = YOLO(str(wdir / "best.pt"))
    # chon conf tren val
    preds = {n: raw_predict(model, all_data[n][0], imgsz, device) for n in va_n}
    val_curve = []
    for c in CONF_GRID:
        val_curve.append(np.mean([C.seg_metrics(union_at(*preds[n], c, all_data[n][1].shape), all_data[n][1])["dice"] for n in va_n]))
    best_i = int(np.argmax(val_curve))
    conf = CONF_GRID[best_i]
    print(f"    val Dice theo conf: " + ", ".join(f"{c:.2f}:{v:.3f}" for c, v in zip(CONF_GRID, val_curve)))
    print(f"    -> chon conf = {conf:.2f} (val Dice {val_curve[best_i]:.4f})")

    ov_dir = out / "test_overlays"
    ov_dir.mkdir(exist_ok=True)
    timg = []
    with open(out / "per_image_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "dice", "iou", "precision", "recall", "pred_pct", "gt_pct"])
        for n in te_n:
            img, gt = all_data[n]
            ms, cf = raw_predict(model, img, imgsz, device)
            pred = union_at(ms, cf, conf, gt.shape)
            m = C.seg_metrics(pred, gt)
            timg.append(m)
            w.writerow([n] + [f"{m[k]:.4f}" for k in ["dice", "iou", "precision", "recall", "pred_pct", "gt_pct"]])
            cv2.imwrite(str(ov_dir / f"{n}.jpg"), cv2.cvtColor(C.make_overlay(img, pred, gt), cv2.COLOR_RGB2BGR))
    ts = C.summarize(timg)
    print(f"\n[yolo {out.name}] TEST ({len(timg)} anh): Dice {ts['dice'][0]:.4f} +/- {ts['dice'][1]:.4f} | "
          f"IoU {ts['iou'][0]:.4f} | Precision {ts['precision'][0]:.4f} | Recall {ts['recall'][0]:.4f}")
    (out / "test_summary.json").write_text(json.dumps({k: list(v) for k, v in ts.items()}, indent=1))
    return {"fold": fold, "val_dice": float(val_curve[best_i]), **{f"test_{k}": v[0] for k, v in ts.items()},
            "n_test": len(timg), "conf": conf}


def main():
    args = parse_args()
    import torch
    if args.smoke:
        args.model, args.width, args.height, args.epochs = "yolo11n-seg.yaml", 256, 384, 1
        args.batch, args.workers, args.no_amp, args.out = 2, 0, True, "runs/smoke_yolo"
    device = 0 if torch.cuda.is_available() else "cpu"
    imgsz = args.height  # canh dai; Ultralytics letterbox ve vuong imgsz x imgsz
    print(f">>> Thiet bi: {device} | imgsz {imgsz}")

    folds = list(range(args.n_folds)) if args.all_folds else [args.fold]
    split_csv = Path(args.data) / "split.csv"
    names = set()
    for f in folds:
        names |= set(sum(map(list, C.load_split(split_csv, f, args.n_folds)), []))
    print(f">>> Doc {len(names)} anh vao RAM o {args.width}x{args.height} (de danh gia)...")
    all_data = C.load_arrays(args.data, sorted(names), args.width, args.height)

    results = []
    for f in folds:
        results.append(run_fold(args, f, all_data, device, imgsz))

    if len(results) > 1:
        out = Path(args.out)
        with open(out / "cv_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        d = np.array([r["test_dice"] for r in results]); i = np.array([r["test_iou"] for r in results])
        print(f"\n=========== YOLO-SEG {len(results)}-FOLD CROSS-VALIDATION ===========")
        print(f"Dice: {d.mean():.4f} +/- {d.std():.4f}   IoU: {i.mean():.4f} +/- {i.std():.4f}")


if __name__ == "__main__":
    main()
