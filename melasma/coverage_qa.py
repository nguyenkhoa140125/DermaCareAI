import sys, cv2, numpy as np
sys.path.insert(0, "melasma")
import hg1_roi as R, memi_common as C
from pathlib import Path
data = Path("data/memi_processed"); out = Path("qa_low"); out.mkdir(exist_ok=True)
rows = [l.split(",") for l in open(data / "split.csv", encoding="utf-8").read().splitlines()]
head = rows[0]; iname, iview = head.index("name"), head.index("view")
specs = R.load_roi_specs(None); res = []
for r in rows[1:]:
    name, view = r[iname], r[iview]
    img = cv2.imread(str(data / "images" / f"{name}.jpg")); msk = cv2.imread(str(data / "masks" / f"{name}.png"), 0)
    if img is None or msk is None: continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); m01 = (msk > 127).astype(np.uint8)
    face = R.detect_face(rgb, "models/face/yolov8n-face.pt", 0.15, None, allow_haar=True)
    if face is None or m01.sum() == 0: continue
    rois = R.derive_rois(face["box"], rgb.shape[:2], specs, R.norm_view(view))
    union = np.zeros_like(m01)
    for v in rois.values():
        x0, y0, x1, y1 = v["box"]; union[y0:y1, x0:x1] = np.maximum(union[y0:y1, x0:x1], m01[y0:y1, x0:x1])
    res.append((float(union.sum()) / float(m01.sum()), name, view, face["backend"], rgb, m01, face, rois))
res.sort(key=lambda t: t[0])
with open("coverage.csv", "w", encoding="utf-8") as f:
    f.write("name,view,backend,coverage\n")
    for c, n, v, b, *_ in res: f.write(f"{n},{v},{b},{c:.3f}\n")
pick = res[:4] + [[t for t in res if t[2].upper() == vw][-1] for vw in "FLR"]
for c, n, v, b, rgb, m01, face, rois in pick:
    ov = R.draw_rois(C.make_overlay(rgb, m01), face["box"], rois)
    cv2.imwrite(str(out / f"{n}_cov{int(c * 100)}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
print("xong:", len(res), "anh; xem thu muc qa_low va file coverage.csv")
