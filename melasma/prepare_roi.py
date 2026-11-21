"""
prepare_roi.py -- Huong 1, chuan bi du lieu cho GIAI DOAN 2: cat tung ROI (tran, 2 ma, mui, cam) tu
anh + mask MEMI-DS da xu ly (output cua prepare_memi.py) bang CUNG giai doan 1 se dung luc suy luan.

Moi ROI duoc letterbox vao khung VUONG (--size, mac dinh 768, vien den) va luu theo dung dinh dang
cua prepare_memi.py (images/*.jpg, masks/*.png, split.csv) nen cac script train dung lai duoc:

  python melasma/prepare_roi.py --data data/memi_processed --out data/memi_roi \
      --face-weights models/face/yolov8n-face.pt --qa 20
  python melasma/train_unet.py --data data/memi_roi --all-folds --out runs/unet_roi --width 512 --height 512
  python melasma/train_segformer.py --data data/memi_roi --all-folds --out runs/segformer_roi --width 512 --height 512

Chia tap van THEO BENH NHAN (giu nguyen cot split/fold cua anh goc), cac ROI cua cung mot anh
cung nam trong mot fold. Tu "name" cua ROI la  <ten_anh_goc>__<ten_roi>.

Cac chi so can doc trong MANIFEST_ROI.txt:
  * so anh khong tim thay mat (bi bo)
  * "do bao phu nam": ti le pixel nam cua anh goc nam trong cac ROI. Neu thap, ROI_SPECS chua khop
    du lieu -> xem anh trong qa/ roi chinh --roi-json truoc khi train.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import hg1_roi as R
import memi_common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/memi_processed")
    p.add_argument("--out", default="data/memi_roi")
    p.add_argument("--size", type=int, default=768, help="canh khung vuong luu moi ROI")
    p.add_argument("--face-weights", default=None)
    p.add_argument("--face-conf", type=float, default=0.15, help="thap hon mac dinh vi mat bi che mat")
    p.add_argument("--roi-json", default=None)
    p.add_argument("--views", nargs="+", default=None,
                   help="chi lay cac goc chup nay (cot 'view' cua split.csv; F = chinh dien). Mac dinh: tat ca")
    p.add_argument("--rois", nargs="+", default=R.ROI_NAMES, choices=R.ROI_NAMES)
    p.add_argument("--min-fg", type=float, default=0.0,
                   help="bo ROI co ti le nam < nguong (%% ROI); mac dinh giu ca ROI khong co nam (mau am)")
    p.add_argument("--qa", type=int, default=20, help="so anh luu ve ROI de kiem tra")
    p.add_argument("--no-haar", action="store_true", help="khong dung Haar khi YOLO khong thay mat")
    p.add_argument("--device", default=None)
    return p.parse_args()


def letterbox_mask(mask01, size):
    m3 = np.dstack([mask01 * 255] * 3).astype(np.uint8)
    canvas, _ = C.letterbox(m3, size, size)
    return (canvas[:, :, 0] > 127).astype(np.uint8)


def main():
    a = parse_args()
    data, out = Path(a.data), Path(a.out)
    for d in ("images", "masks", "qa"):
        (out / d).mkdir(parents=True, exist_ok=True)
    specs = R.load_roi_specs(a.roi_json)
    rows = list(csv.DictReader(open(data / "split.csv", encoding="utf-8")))
    views = defaultdict(int)
    for r in rows:
        views[r["view"]] += 1
    print(">>> Goc chup trong du lieu:", dict(views), "(F = chinh dien; L/R = nghieng ~45 do)")
    if a.views:
        rows = [r for r in rows if r["view"] in set(a.views)]
    if not rows:
        raise SystemExit("Khong con anh nao sau khi loc --views")
    print(f">>> {len(rows)} anh. Dang cat ROI...")

    out_rows, no_face, backends, coverage, qa_done = [], [], defaultdict(int), [], 0
    for i, r in enumerate(rows, 1):
        name = r["name"]
        img = cv2.imread(str(data / "images" / f"{name}.jpg"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(data / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(f"Thieu anh/mask cua {name} trong {data}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m01 = (msk > 127).astype(np.uint8)
        face = R.detect_face(rgb, a.face_weights, a.face_conf, a.device, allow_haar=not a.no_haar)
        if face is None:
            no_face.append(name)
            continue
        backends[face["backend"]] += 1
        if r["view"].upper() not in R.VIEWS:
            print(f"    [!] {name}: goc chup '{r['view']}' khong ro, dung hinh hoc chinh dien (F)")
        rois = R.derive_rois(face["box"], rgb.shape[:2], specs, R.norm_view(r["view"]))
        union = np.zeros_like(m01)
        for rn in a.rois:
            if rn not in rois:
                continue
            x0, y0, x1, y1 = rois[rn]["box"]
            cm = m01[y0:y1, x0:x1]
            union[y0:y1, x0:x1] = np.maximum(union[y0:y1, x0:x1], cm)
            fg = 100.0 * float(cm.mean())
            if fg < a.min_fg:
                continue
            canvas, _ = C.letterbox(rgb[y0:y1, x0:x1], a.size, a.size)
            roi_name = f"{name}__{rn}"
            cv2.imwrite(str(out / "images" / f"{roi_name}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / "masks" / f"{roi_name}.png"), letterbox_mask(cm, a.size) * 255)
            out_rows.append({"name": roi_name, "group": r["group"], "type": r["type"], "view": r["view"],
                             "split": r["split"], "fold": r["fold"], "fg_percent": round(fg, 2),
                             "region_ids": "", "roi": rn})
        if m01.sum() > 0:
            coverage.append(float(union.sum()) / float(m01.sum()))
        if qa_done < a.qa:
            qa_done += 1
            ov = R.draw_rois(C.make_overlay(rgb, m01), face["box"], rois)
            cv2.imwrite(str(out / "qa" / f"{name}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        if i % 20 == 0:
            print(f"    {i}/{len(rows)}")

    if not out_rows:
        raise SystemExit("Khong tao duoc ROI nao (khong tim thay mat o anh nao?). Kiem tra --face-weights.")
    with open(out / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "group", "type", "view", "split", "fold", "fg_percent",
                                           "region_ids", "roi"])
        w.writeheader()
        w.writerows(out_rows)

    cnt = defaultdict(lambda: defaultdict(int))
    for r in out_rows:
        cnt[r["split"]]["roi"] += 1
    lines = ["BIEN BAN DU LIEU ROI (Huong 1)",
             f"Anh dau vao: {len(rows)} | khong tim thay mat (bi bo): {len(no_face)} | tong ROI: {len(out_rows)}",
             f"Giai doan 1 dung: {dict(backends)}",
             f"Khung luu: {a.size}x{a.size} | ROI: {', '.join(a.rois)}"]
    if coverage:
        lines.append(f"Do bao phu nam (ti le pixel nam cua anh goc nam trong cac ROI): "
                     f"trung binh {100 * np.mean(coverage):.1f}%, thap nhat {100 * np.min(coverage):.1f}%")
    lines += [""] + [f"  {s:5s}: {cnt[s]['roi']:4d} ROI" for s in ("train", "val", "test")]
    if no_face:
        lines += ["", "Anh khong tim thay mat:"] + [f"  - {n}" for n in no_face]
    (out / "MANIFEST_ROI.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nHOAN TAT. Xem anh ve ROI trong {out / 'qa'} TRUOC khi train.")


if __name__ == "__main__":
    main()
