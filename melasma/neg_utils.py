"""
neg_utils.py -- Lay anh 'normal' cua bo Roboflow lam MAU AM (mask toan nen) cho huan luyen,
dong thoi GIU LAI mot phan de do ti le bao nham tren da binh thuong cho moi mo hinh.

Quy tac chong ro ri / nhieu:
  * Phat hien anh gan trung tren TOAN bo 819 anh roi bo moi anh thuoc nhom co nhan mau thuan
    (vd cung 1 anh vua 'normal' vua 'melasma') va chi giu 1 dai dien moi nhom trung.
  * Tap giu lai (held-out) KHONG BAO GIO dung de huan luyen o bat ky mo hinh nao -> co the so sanh
    cong bang mo hinh goc (chua thay anh normal nao) va mo hinh v2 (da thay phan huan luyen).
  * Chia xac dinh theo seed => train_unet_neg.py va fp_normals.py luon thay cung mot cach chia.
"""
import random

import cv2
import numpy as np

import memi_common as C
from eval_roboflow import collect, dup_groups


def load_normals(neg_dir, seed=42, heldout_frac=0.3):
    rows = collect(neg_dir)
    if not rows:
        raise SystemExit(f"Khong thay anh trong {neg_dir}")
    gid, conflict, keep = dup_groups(rows)
    normals = [r for r, c, k in zip(rows, conflict, keep) if r["cls"] == "normal" and k and not c]
    normals = sorted(normals, key=lambda r: r["path"])
    random.Random(seed).shuffle(normals)
    n_held = max(1, round(len(normals) * heldout_frac))
    return normals[n_held:], normals[:n_held]   # (train_neg, heldout)


def to_arrays(items, width, height):
    """Anh normal -> (anh letterbox WxH, mask toan 0, thong tin letterbox) giong cach eval_roboflow dua anh vao."""
    out = []
    for r in items:
        img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        canvas, info = C.letterbox(img, width, height)
        out.append((canvas, np.zeros((height, width), np.uint8), info))
    return out


def content_pct(mask_bool, info):
    """% pixel nam chi tinh trong vung co noi dung (bo vien den cua letterbox)."""
    nw, nh, left, top = info
    return 100.0 * float(mask_bool[top:top + nh, left:left + nw].mean())


def fp_summary(pcts):
    p = np.asarray(pcts, dtype=float)
    return {"n": len(p), "mean": float(p.mean()), "median": float(np.median(p)),
            "pct_gt1": float(100 * (p > 1).mean()), "pct_gt5": float(100 * (p > 5).mean())}
