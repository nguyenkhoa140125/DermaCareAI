"""
eval_roboflow.py -- Kiem tra TONG QUAT HOA cua mo hinh tren bo Roboflow "Pigmentation Melasma Dark Spots"
(dinh dang FOLDER = chi co nhan CAP ANH, KHONG co mask) nen KHONG tinh duoc Dice.

Cau hoi duoc tra loi: mo hinh (huan luyen tren MEMI-DS) co phan biet duoc anh melasma voi da binh
thuong / dom do khac hay khong, tren anh co nguon va hoan canh chup rat khac (anh web, nen khac,
mat khong bi che)?  Diem so cua moi anh = % dien tich nam ma mo hinh du doan.

Chay:
  python eval_roboflow.py --data Pigmentation_Melasma_Dark_Spots_v1i_folder --out rf_eval --ckpt runs/unet_r34/fold0/best.pt
  python eval_roboflow.py --data ... --out rf_eval --ckpt runs/unet_r34/fold0/best.pt runs/unet_r34/fold1/best.pt ... --tta
  (them --limit 20 de chay thu nhanh: moi lop 20 anh)

Ghi: <out>/per_image.csv, <out>/summary.txt, <out>/montage_*.jpg  (xem bang mat cac anh overlay!)

Luu y khi doc ket qua:
  * Lop "normal" gom nhieu anh vai/lung/tay (khong phai mat) => de, AUC melasma-vs-normal bi LAC QUAN.
  * Nhan la do nguoi dung Roboflow gan, khong qua bac si; nhieu anh bi lap va co anh nhan mau thuan.
  * Anh da bi resize 640x640 kieu STRETCH (meo ti le) truoc khi tai ve.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ----------------------------------------------------------------------------
# Cac ham thuan numpy/OpenCV (khong can PyTorch)
# ----------------------------------------------------------------------------
def collect(root):
    rows = []
    for s in ["train", "valid", "test"]:
        sd = Path(root) / s
        if not sd.is_dir():
            continue
        for cd in sorted(p for p in sd.iterdir() if p.is_dir()):
            for f in sorted(cd.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"split": s, "cls": cd.name, "file": f.name, "path": str(f)})
    return rows


def dhash(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def dup_groups(rows, max_ham=6):
    """Gom anh gan trung nhau (dHash). Tra ve (id_nhom, co_nhan_mau_thuan, giu_lai) cho tung anh."""
    n = len(rows)
    H = np.array([dhash(r["path"]) for r in rows])
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.nonzero(d <= max_ham)[0]:
            par[find(i)] = find(i + 1 + j)
    gid = [find(i) for i in range(n)]
    members = {}
    for i, g in enumerate(gid):
        members.setdefault(g, []).append(i)
    conflict, keep = [False] * n, [False] * n
    for g, idx in members.items():
        conf = len({rows[i]["cls"] for i in idx}) > 1
        for i in idx:
            conflict[i] = conf
        keep[idx[0]] = True
    return gid, conflict, keep


def auc(pos, neg):
    """AUC = xac suat diem cua anh duong > diem cua anh am (Mann-Whitney)."""
    s = np.concatenate([pos, neg])
    r = pd.Series(s).rank().values
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def montage(items, out_path, cols=6, size=256):
    """items: list (anh RGB, chu thich). Ghep thanh luoi de xem nhanh."""
    if not items:
        return
    tiles = []
    for img, text in items:
        t = cv2.resize(img, (size, size))
        cv2.rectangle(t, (0, 0), (size, 22), (0, 0, 0), -1)
        cv2.putText(t, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * size, cols * size, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // cols) * size:(i // cols + 1) * size, (i % cols) * size:(i % cols + 1) * size] = t
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def summarize(df, thr_list=(1.0, 5.0)):
    u = df[df.keep]
    lines = [f"So anh: {len(df)} | sau khi bo anh gan trung: {len(u)} "
             f"({len(df) - len(u)} ban sao) | nhom anh nhan mau thuan: "
             f"{df[df.conflict].dup_group.nunique()} nhom"]
    lines.append("\n--- % dien tich nam du doan, theo lop (anh duy nhat) ---")
    t = u.groupby("cls").pred_pct.agg(n="size", mean="mean", median="median",
                                      p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75))
    for th in thr_list:
        t[f"%anh>{th:g}%"] = u.groupby("cls").pred_pct.apply(lambda x, th=th: 100 * (x > th).mean())
    lines.append(t.round(2).to_string())

    def comp(a, b):
        if a in set(u.cls) and b in set(u.cls):
            pos, neg = u[u.cls == a].pred_pct.values, u[u.cls == b].pred_pct.values
            lo, hi = auc_ci(pos, neg)
            return f"  AUC {a} > {b}: {auc(pos, neg):.3f}  (KTC95% {lo:.3f}-{hi:.3f}, n={len(pos)} vs {len(neg)})"
        return None

    lines.append("\n--- Kha nang PHAN BIET (AUC; 0.5 = ngau nhien, 1.0 = hoan hao) ---")
    for a, b in [("melasma", "normal"), ("melasma", "dark spots"), ("melasma", "hyperpigmentation"),
                 ("melasma", "pigmentation")]:
        r = comp(a, b)
        if r:
            lines.append(r)
    lines.append("\nLuu y: 'normal' gom nhieu anh vai/lung (khong phai mat) nen AUC melasma-vs-normal de bi LAC QUAN;"
                 "\nso sanh melasma-vs-dark spots/hyperpigmentation moi phan anh kha nang phan biet kho that.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="rf_eval")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--mode", choices=["letterbox", "stretch"], default="letterbox",
                    help="cach dua anh 640x640 vao mo hinh 512x768")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="chi lay N anh moi lop (chay thu)")
    ap.add_argument("--n-overlay", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch  # import muon de cac ham o tren thu duoc khi khong co PyTorch
    import memi_common as C
    from predict import load_model, predict_prob

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(a.data)
    if not rows:
        raise SystemExit(f"Khong thay anh trong {a.data} (can cau truc train|valid|test/<ten_lop>/*.jpg)")
    if a.limit:
        cnt, sel = {}, []
        for r in rows:
            cnt[r["cls"]] = cnt.get(r["cls"], 0) + 1
            if cnt[r["cls"]] <= a.limit:
                sel.append(r)
        rows = sel
    print(f">>> {len(rows)} anh. Dang tim anh gan trung...")
    gid, conflict, keep = dup_groups(rows)
    for r, g, c, k in zip(rows, gid, conflict, keep):
        r.update(dup_group=g, conflict=c, keep=k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(c, device) for c in a.ckpt]
    print(f">>> {len(models)} mo hinh | che do {a.mode} | nguong {a.thr} | TTA={a.tta} | {device}")

    overlays = {}
    for i, r in enumerate(rows):
        bgr = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]
        probs = []
        for fwd, cfg in models:
            W, H = cfg["width"], cfg["height"]
            if a.mode == "letterbox":
                canvas, info = C.letterbox(rgb, W, H)
                pr = C.unletterbox(predict_prob(fwd, canvas, device, a.tta), info, (H0, W0))
            else:
                canvas = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                pr = cv2.resize(predict_prob(fwd, canvas, device, a.tta), (W0, H0))
            probs.append(pr)
        mask = (np.mean(probs, 0) > a.thr).astype(np.uint8)
        r["pred_pct"] = 100.0 * float(mask.mean())
        overlays[i] = (rgb, mask)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(rows)}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "path"} for r in rows])
    df.to_csv(out / "per_image.csv", index=False)
    text = summarize(df)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    rnd = random.Random(a.seed)

    def item(i):
        rgb, mask = overlays[i]
        return C.make_overlay(rgb, mask), f"{rows[i]['cls'][:12]} {rows[i]['pred_pct']:.1f}%"

    uniq = df[df.keep]
    for cls in sorted(uniq.cls.unique()):
        idx = list(uniq[uniq.cls == cls].index)
        pick = rnd.sample(idx, min(a.n_overlay, len(idx)))
        montage([item(i) for i in pick], out / f"montage_{cls.replace(' ', '_')}.jpg")
    if "normal" in set(uniq.cls):
        top = uniq[uniq.cls == "normal"].sort_values("pred_pct", ascending=False).head(12).index
        montage([item(i) for i in top], out / "montage_normal_DU_DOAN_NAM_NHIEU_NHAT.jpg")
    if "melasma" in set(uniq.cls):
        low = uniq[uniq.cls == "melasma"].sort_values("pred_pct").head(12).index
        montage([item(i) for i in low], out / "montage_melasma_DU_DOAN_IT_NHAT.jpg")
    print(f"\nHOAN TAT. Xem cac file montage_*.jpg trong {out.resolve()}")


if __name__ == "__main__":
    main()
