"""
memi_common.py -- cac ham dung chung (KHONG can PyTorch): doc du lieu, chia fold,
augmentation, chi so danh gia, ve overlay, letterbox, cat mat.
"""
import csv
import math
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------
# Chia tap
# ----------------------------------------------------------------------------
def load_split(csv_path, fold=-1, n_folds=5):
    """fold < 0 : dung cot 'split' (train/val/test) trong split.csv.
    fold = k   : test = fold k, val = fold (k+1)%n, train = cac fold con lai.
    Tra ve (train_names, val_names, test_names). Chia theo benh nhan nen khong ro ri."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if fold < 0:
        pick = lambda s: [r["name"] for r in rows if r["split"] == s]
        return pick("train"), pick("val"), pick("test")
    val_fold = (fold + 1) % n_folds
    tr, va, te = [], [], []
    for r in rows:
        f = int(r["fold"])
        (te if f == fold else va if f == val_fold else tr).append(r["name"])
    return tr, va, te


def load_arrays(root, names, width, height):
    """Doc anh + mask nhi phan, resize ve (width, height). Tra ve dict name -> (img_rgb uint8, mask uint8 {0,1})."""
    root = Path(root)
    out = {}
    for n in names:
        img = cv2.imread(str(root / "images" / f"{n}.jpg"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(root / "masks" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(f"Khong doc duoc anh/mask cua {n} trong {root}")
        img = cv2.cvtColor(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        msk = cv2.resize(msk, (width, height), interpolation=cv2.INTER_LINEAR)
        out[n] = (img, (msk > 127).astype(np.uint8))
    return out


# ----------------------------------------------------------------------------
# Augmentation (chi dung OpenCV + numpy, khong phu thuoc phien ban thu vien ngoai)
# Chu y: chi bien doi mau NHE vi mau sac chinh la dau hieu cua nam.
# ----------------------------------------------------------------------------
def augment(img, mask, rng):
    h, w = mask.shape
    img = img.copy()
    mask = mask.copy()

    if rng.random() < 0.5:  # lat ngang
        img, mask = img[:, ::-1], mask[:, ::-1]

    if rng.random() < 0.8:  # xoay / phong to / dich chuyen
        ang = rng.uniform(-15, 15)
        scale = rng.uniform(0.85, 1.20)
        tx, ty = rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.08, 0.08) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    img = np.ascontiguousarray(img).astype(np.float32)
    mask = np.ascontiguousarray(mask)

    if rng.random() < 0.8:  # do sang / tuong phan / gamma
        img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
        img = np.clip(img, 0, 255)
        g = rng.uniform(0.8, 1.25)
        img = 255.0 * (img / 255.0) ** g

    if rng.random() < 0.5:  # lech sac / bao hoa nhe (gia lap anh sang phong khac nhau)
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    if rng.random() < 0.3:  # lam mo nhe
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.3:  # nhieu hat
        img = img + rng.normal(0, rng.uniform(1, 6), img.shape)

    return np.clip(img, 0, 255).astype(np.uint8), mask


def normalize(img_uint8):
    """HxWx3 uint8 -> 3xHxW float32 chuan ImageNet."""
    x = (img_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ----------------------------------------------------------------------------
# Chi so danh gia (tinh tren tung anh roi lay trung binh)
# ----------------------------------------------------------------------------
def seg_metrics(pred, gt, eps=1e-7):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "pred_pct": 100.0 * pred.mean(),
        "gt_pct": 100.0 * gt.mean(),
    }


def summarize(per_image):
    keys = ["dice", "iou", "precision", "recall"]
    return {k: (float(np.mean([m[k] for m in per_image])), float(np.std([m[k] for m in per_image])))
            for k in keys}


# ----------------------------------------------------------------------------
# Ve overlay
# ----------------------------------------------------------------------------
def make_overlay(img_rgb, pred, gt=None, alpha=0.45):
    """Phu do len vung du doan. Neu co gt: ve vien xanh la quanh vung nam that."""
    ov = img_rgb.astype(np.float32).copy()
    m = pred.astype(bool)
    ov[m] = (1 - alpha) * ov[m] + alpha * np.array([255, 0, 0], dtype=np.float32)
    ov = ov.astype(np.uint8)
    if gt is not None:
        cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), max(2, ov.shape[1] // 300))
    return ov


# ----------------------------------------------------------------------------
# Suy luan tren anh bat ky: cat mat (tuy chon) + letterbox ve ti le W:H cua mo hinh
# ----------------------------------------------------------------------------
def crop_face(img_rgb, target_aspect=2 / 3, margin=1.9):
    """Cat quanh khuon mat bang Haar Cascade cua OpenCV (khong can tai file).
    Tra ve (crop, (x0, y0, x1, y1)) hoac (img, None) neu khong tim thay mat.
    target_aspect = W/H cua anh huan luyen (MEMI-DS: 2:3)."""
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        det = cv2.CascadeClassifier(path)
        if det.empty():
            return img_rgb, None
        H, W = img_rgb.shape[:2]
        s = 800.0 / max(H, W) if max(H, W) > 800 else 1.0
        gray = cv2.cvtColor(cv2.resize(img_rgb, None, fx=s, fy=s), cv2.COLOR_RGB2GRAY)
        faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return img_rgb, None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = (x + w / 2) / s, (y + h / 2) / s
        bw = w / s * margin
        bh = bw / target_aspect
        x0, x1 = int(max(0, cx - bw / 2)), int(min(W, cx + bw / 2))
        y0, y1 = int(max(0, cy - bh / 2 + 0.05 * bh)), int(min(H, cy + bh / 2 + 0.05 * bh))
        if x1 - x0 < 50 or y1 - y0 < 50:
            return img_rgb, None
        return img_rgb[y0:y1, x0:x1], (x0, y0, x1, y1)
    except Exception:
        return img_rgb, None


def letterbox(img_rgb, width, height):
    """Resize giu ti le, them vien den cho vua (width, height). Tra ve (anh, (nw, nh, left, top))."""
    h0, w0 = img_rgb.shape[:2]
    s = min(width / w0, height / h0)
    nw, nh = max(1, int(round(w0 * s))), max(1, int(round(h0 * s)))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left, top = (width - nw) // 2, (height - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, (nw, nh, left, top)


def unletterbox(prob, info, orig_hw):
    """prob (height x width, float) -> xac suat tren kich thuoc anh goc (bo vien den)."""
    nw, nh, left, top = info
    crop = prob[top:top + nh, left:left + nw]
    return cv2.resize(crop, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)


def cosine_warmup(step, warmup, total, floor=0.01):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return max(floor, 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
