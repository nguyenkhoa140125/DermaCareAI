"""
prepare_memi.py  --  Chuan bi bo MEMI-DS de huan luyen phan vung nam da.

Dau vao  (sau khi giai nen 4 file zip vao CUNG 1 thu muc RAW):
    RAW/Flaky_Image/*.jpg      RAW/Flaky_Annotation/*.png
    RAW/Punctate_Image/*.jpg   RAW/Punctate_Annotation/*.png

Dau ra (OUT):
    images/            anh RGB da resize (mac dinh 1024x1536, giu ti le 2:3)
    masks/             mask NHI PHAN 0/255 (moi vung nam = 255)
    masks_region/      mask goc co chi so vung 0..5 (de tham khao, KHONG dung de train nhi phan)
    qa/                anh overlay de kiem tra bang mat
    yolo_seg/          (tuy chon --yolo) nhan polygon dinh dang YOLO-seg + data.yaml
    split.csv          bang chia tap: name, group, type, view, split, fold, fg_percent
    MANIFEST.txt       bien ban du lieu (so anh, so nguoi, ket qua chia...)

Cach chay:
    python prepare_memi.py --raw memi_raw --out memi_processed
    python prepare_memi.py --raw memi_raw --out memi_processed --yolo

Luu y quan trong:
  * Chia tap THEO MA SO BENH NHAN (phan truoc dau '_' cua ten file), gop ca 2 nhom
    Flaky va Punctate: cung ma so o hai nhom nhin giong cung 1 nguoi, nen KHONG de
    ma so do nam o ca train lan test.
  * Chi so vung trong mask goc co the khac nghia giua anh F (chinh dien) va L/R (nghieng),
    nen khi train nhi phan ta gop moi chi so > 0 thanh 1.
"""
import argparse
import csv
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TYPES = ["Flaky", "Punctate"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="memi_raw")
    p.add_argument("--out", default="memi_processed")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1536)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--yolo", action="store_true", help="xuat them nhan YOLO-seg")
    return p.parse_args()


def find_pairs(raw: Path):
    pairs, orphans = [], []
    for t in TYPES:
        img_dir, ann_dir = raw / f"{t}_Image", raw / f"{t}_Annotation"
        if not img_dir.exists() or not ann_dir.exists():
            print(f"[!] Thieu thu muc {img_dir} hoac {ann_dir}")
            continue
        imgs = {p.stem: p for p in img_dir.glob("*.jpg")}
        anns = {p.stem: p for p in ann_dir.glob("*.png")}
        for stem in sorted(imgs):
            if stem in anns:
                pairs.append((t, stem, imgs[stem], anns[stem]))
            else:
                orphans.append(f"{t}/{stem}: co anh, thieu mask")
        for stem in sorted(set(anns) - set(imgs)):
            orphans.append(f"{t}/{stem}: co mask, thieu anh (bo qua)")
    return pairs, orphans


def process_one(job):
    t, stem, img_path, ann_path, out, size, yolo = job
    name = f"{t}_{stem}"
    W, H = size

    im = Image.open(img_path)
    im.draft("RGB", (im.width // 2, im.height // 2))  # giai ma JPEG nhanh hon
    im = im.convert("RGB").resize((W, H), Image.LANCZOS)
    im.save(out / "images" / f"{name}.jpg", quality=95)

    idx = np.array(Image.open(ann_path))  # chi so palette goc
    idx_small = np.array(Image.fromarray(idx).resize((W, H), Image.NEAREST))
    binary = ((idx_small > 0).astype(np.uint8)) * 255
    Image.fromarray(binary).save(out / "masks" / f"{name}.png")
    Image.fromarray(idx_small.astype(np.uint8)).save(out / "masks_region" / f"{name}.png")

    # anh overlay de kiem tra bang mat
    ov = np.array(im).astype(np.float32)
    m = binary > 0
    ov[m] = 0.55 * ov[m] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    qa = Image.fromarray(ov.astype(np.uint8)).resize((W // 2, H // 2), Image.BILINEAR)
    qa.save(out / "qa" / f"{name}.jpg", quality=80)

    # nhan YOLO-seg: moi thanh phan lien thong = 1 polygon, toa do chuan hoa 0..1
    if yolo:
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in cnts:
            if cv2.contourArea(c) < 150:  # bo cham nho li ti
                continue
            c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True)
            if len(c) < 3:
                continue
            pts = " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in c.reshape(-1, 2))
            lines.append(f"0 {pts}")
        (out / "yolo_labels_tmp").mkdir(exist_ok=True)
        (out / "yolo_labels_tmp" / f"{name}.txt").write_text("\n".join(lines))

    return {
        "name": name,
        "group": stem.split("_")[0],
        "type": t,
        "view": stem.split("_")[1],
        "fg_percent": round(100.0 * float(m.mean()), 2),
        "region_ids": " ".join(str(i) for i in np.unique(idx_small) if i > 0),
    }


def split_groups(groups, val, test, folds, seed):
    groups = sorted(groups, key=int)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test, n_val = round(n * test), round(n * val)
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])
    split = {g: ("test" if g in test_g else "val" if g in val_g else "train") for g in groups}
    fold = {g: i % folds for i, g in enumerate(groups)}
    return split, fold


def main():
    a = parse_args()
    raw, out = Path(a.raw), Path(a.out)
    for d in ["images", "masks", "masks_region", "qa"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    pairs, orphans = find_pairs(raw)
    if not pairs:
        raise SystemExit(f"Khong tim thay cap anh+mask nao trong {raw.resolve()}")
    print(f">>> {len(pairs)} cap anh+mask. Dang xu ly (chi mat vai phut)...")

    jobs = [(t, s, ip, ap, out, (a.width, a.height), a.yolo) for t, s, ip, ap in pairs]
    with Pool(a.workers) as pool:
        rows = pool.map(process_one, jobs)

    groups = {r["group"] for r in rows}
    split, fold = split_groups(groups, a.val, a.test, a.folds, a.seed)
    for r in rows:
        r["split"], r["fold"] = split[r["group"]], fold[r["group"]]

    with open(out / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "group", "type", "view", "split", "fold",
                                           "fg_percent", "region_ids"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (int(r["group"]), r["name"])))

    if a.yolo:
        for s in ["train", "val", "test"]:
            (out / "yolo_seg" / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / "yolo_seg" / "labels" / s).mkdir(parents=True, exist_ok=True)
        for r in rows:
            s = r["split"]
            (out / "yolo_seg" / "images" / s / f"{r['name']}.jpg").write_bytes(
                (out / "images" / f"{r['name']}.jpg").read_bytes())
            (out / "yolo_seg" / "labels" / s / f"{r['name']}.txt").write_text(
                (out / "yolo_labels_tmp" / f"{r['name']}.txt").read_text())
        (out / "yolo_seg" / "data.yaml").write_text(
            f"path: {(out / 'yolo_seg').resolve()}\ntrain: images/train\nval: images/val\n"
            "test: images/test\nnames:\n  0: melasma\n")

    # bien ban du lieu
    cnt = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cnt[r["split"]]["anh"] += 1
    for s in cnt:
        cnt[s]["nguoi"] = len({r["group"] for r in rows if r["split"] == s})
    fg = [r["fg_percent"] for r in rows]
    lines = [
        "BIEN BAN DU LIEU MEMI-DS",
        f"Tong cap anh+mask: {len(rows)} | tong so ma benh nhan (gop 2 nhom): {len(groups)}",
        f"Kich thuoc luu: {a.width}x{a.height} | seed={a.seed}",
        "Ti le vung nam / toan anh (%): "
        f"min={min(fg):.2f}  trung binh={sum(fg)/len(fg):.2f}  max={max(fg):.2f}",
        "",
    ]
    for s in ["train", "val", "test"]:
        lines.append(f"  {s:5s}: {cnt[s]['anh']:3d} anh / {cnt[s]['nguoi']:2d} nguoi")
    if orphans:
        lines += ["", "Muc bi bo qua:"] + [f"  - {o}" for o in orphans]
    (out / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nHOAN TAT. Mo thu muc {out / 'qa'} va xem het anh overlay truoc khi train.")


if __name__ == "__main__":
    main()
