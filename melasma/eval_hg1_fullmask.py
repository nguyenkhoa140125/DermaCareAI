import sys, csv, numpy as np, cv2
sys.path.insert(0, "melasma")
import hg1_roi as R, pipeline_hg1 as PH
from pathlib import Path
DATA = Path("data/memi_processed"); RUN = "runs/unet_roi"
FACE = "models/face/yolov8n-face.pt"; BASE = Path("runs/unet_r34")
img_fold = {}
for r in csv.DictReader(open("data/memi_roi/split.csv", encoding="utf-8")):
    img_fold[r["name"].split("__")[0]] = (int(r["fold"]), r["view"])
specs = R.load_roi_specs(None); fns = {}; rows = []
for img, (k, view) in sorted(img_fold.items()):
    im = cv2.imread(str(DATA / "images" / f"{img}.jpg")); gt = cv2.imread(str(DATA / "masks" / f"{img}.png"), 0)
    if im is None or gt is None: continue
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); g = gt > 127
    face = R.detect_face(rgb, FACE, 0.15, None, allow_haar=True)
    if face is None: continue
    if k not in fns: fns[k] = PH.make_segment_fn([f"{RUN}/fold{k}/best.pt"], tta=False)
    seg, thr = fns[k]
    res, full, rois = PH.analyse_image(rgb, face, seg, specs=specs, thr=thr, use_skin=True, view=view)
    p = full.astype(bool); inter = int((p & g).sum())
    dice = 2 * inter / (p.sum() + g.sum() + 1e-9); iou = inter / (int((p | g).sum()) + 1e-9)
    rows.append((img, k, view, dice, iou))
import pandas as pd
df = pd.DataFrame(rows, columns=["name", "fold", "view", "dice", "iou"]); df.to_csv("hg1_fullmask_per_image.csv", index=False)
print("HUONG 1 toan mat (held-out):", len(df), "anh")
fm = df.groupby("fold").dice.mean()
print("  Dice trung binh cac fold %.4f +/- %.4f | IoU %.4f" % (fm.mean(), fm.std(ddof=0), df.groupby("fold").iou.mean().mean()))
print("  Dice trung binh theo anh %.4f +/- %.4f" % (df.dice.mean(), df.dice.std()))
bp = sorted(BASE.glob("fold*/per_image_test.csv"))
if bp:
    b = pd.concat([pd.read_csv(f) for f in bp])[["name", "dice"]].rename(columns={"dice": "dice_base"})
    m = df.merge(b, on="name")
    print("BASELINE (cung", len(m), "anh): Dice trung binh theo anh %.4f +/- %.4f" % (m.dice_base.mean(), m.dice_base.std()))
    d = m.dice - m.dice_base
    print("CHENH LECH (Huong 1 - baseline): trung binh %.4f | Huong 1 cao hon o %d / %d anh" % (d.mean(), (d > 0).sum(), len(m)))
