"""
test_hg1.py -- kiem tra nhanh phan Huong 1 bang du lieu gia lap (khong can PyTorch, du lieu, GPU).
Chay:  python melasma/test_hg1.py
Chi kiem tra LOGIC (hinh hoc ROI, cong thuc MASI, ghep noi pipeline, tao du lieu ROI),
KHONG kiem tra do chinh xac cua YOLO hay mo hinh phan vung.
"""
import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hg1_masi as M
import hg1_roi as R
import pipeline_hg1 as P
import prepare_roi as PR

FACE = {"box": (300.0, 350.0, 700.0, 850.0), "conf": 0.9, "backend": "yolo"}
SKIN = (205, 150, 125)  # RGB mau da


def fake_face(w=1000, h=1200):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.ellipse(img, (500, 650), (240, 330), 0, 0, 360, SKIN, -1)
    return img


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        check.failed += 1


check.failed = 0


def test_area_score():
    pairs = [(0, 0), (0.1, 1), (9.99, 1), (10, 2), (29.9, 2), (30, 3), (49, 3), (50, 4), (69, 4),
             (70, 5), (89, 5), (90, 6), (100, 6)]
    check(all(M.area_score(p) == s for p, s in pairs), "diem dien tich A theo dung bang 0-6")


def test_masi_formula():
    full = {"A": 2, "D": 2, "H": 2}
    r = M.compute_masi({n: dict(full) for n in M.MASI_ROIS})
    check(abs(r["masi"] - 8.0) < 1e-6 and r["complete"], f"MASI A=2,D=2,H=2 moi vung = 8.0 (got {r['masi']})")
    worst = M.compute_masi({n: {"A": 6, "D": 4, "H": 4} for n in M.MASI_ROIS})
    check(abs(worst["masi"] - 48.0) < 1e-6, f"MASI toi da = 48 (got {worst['masi']})")
    part = M.compute_masi({"forehead": dict(full)})
    check(not part["complete"] and set(part["missing_rois"]) == {"right_malar", "left_malar", "chin"},
          "thieu vung -> complete=False va liet ke vung thieu")


def test_local_reference_and_reliability():
    base = np.array((205, 150, 125), np.float32)

    def make(lesion_factor, gradient=False):
        h = w = 300
        f = np.linspace(1.0, 0.6, h)[:, None, None] if gradient else np.ones((h, 1, 1))
        crop = np.clip(base[None, None, :] * f * np.ones((h, w, 1)), 0, 255).astype(np.float32)
        les = np.zeros((h, w), np.uint8)
        les[60:130, 80:220] = 1                                  # vung nam o phan tren (sang)
        crop[les > 0] *= lesion_factor
        return np.clip(crop, 0, 255).astype(np.uint8), les

    c, les = make(0.80, gradient=True)
    m = M.roi_metrics(c, les, ref_L=None)
    check(m["ref_source"] == "ring" and m["d_reliable"] and m["D"] >= 2 and m["delta_L"] > 0,
          f"nam toi hon da sat ben, du anh sang giam dan -> D dang tin (D={m['D']}, delta {m['delta_L']})")
    c, les = make(1.00, gradient=True)
    m = M.roi_metrics(c, les, ref_L=None)
    check(not m["d_reliable"] and any("KHONG khac da sat ben" in w for w in m["warnings"]),
          "vung phan vung khong khac da xung quanh -> canh bao duong tinh gia, D khong dang tin")
    c, les = make(1.12, gradient=False)
    m = M.roi_metrics(c, les, ref_L=None)
    check(not m["d_reliable"] and m["D"] == 1 and m["delta_L"] < 0 and any("KHONG toi hon" in w for w in m["warnings"]),
          f"nam SANG hon da sat ben -> D khong dang tin, D=1 (delta {m['delta_L']})")
    c, les = make(0.80, gradient=True)
    good = M.roi_metrics(c, les, ref_L=None)
    r = M.compute_masi({n: dict(m) for n in M.MASI_ROIS})
    check(not r["reliable"] and set(r["unreliable_D"]) == set(M.MASI_ROIS), "MASI danh dau khong dang tin khi D khong dang tin")
    r2 = M.compute_masi({n: dict(good) for n in M.MASI_ROIS})
    check(r2["reliable"], "MASI dang tin khi moi vung deu dang tin")
    check(not R.view_has_suffix("anh.jpg") and R.view_has_suffix("1_R.jpg"), "nhan biet ten file co/khong co goc chup")


def test_roi_geometry():
    rois = R.derive_rois(FACE["box"], (1200, 1000))
    check({"forehead", "right_malar", "left_malar", "nose", "chin"} <= set(rois), "du 5 ROI voi hop khuon mat o giua anh")
    check(rois["right_malar"]["box"][0] < rois["left_malar"]["box"][0],
          "ma phai cua benh nhan nam ben TRAI anh")
    check(rois["forehead"]["box"][3] <= rois["nose"]["box"][3] and rois["chin"]["box"][1] > rois["nose"]["box"][1],
          "tran o tren, cam o duoi mui")
    ok = all(0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1200 for b in (r["box"] for r in rois.values()))
    check(ok, "moi ROI nam trong anh")
    near_top = R.derive_rois((300.0, 5.0, 700.0, 305.0), (1200, 1000))
    check("forehead" not in near_top, "tran nam ngoai anh (mat sat mep tren) thi bi bo")
    L = R.derive_rois(FACE["box"], (1200, 1000), view="L")
    Rr = R.derive_rois(FACE["box"], (1200, 1000), view="R")
    check("left_malar" in L and "right_malar" not in L, "view L: chi co ma TRAI")
    check("right_malar" in Rr and "left_malar" not in Rr, "view R: chi co ma PHAI")
    check(R.view_from_name("12_L_xyz.jpg") == "L" and R.view_from_name("anh.jpg") == "F", "doan view tu ten file")
    check(R.derive_rois(FACE["box"], (1200, 1000), view="??").keys() == R.derive_rois(FACE["box"], (1200, 1000)).keys(),
          "view la -> dung hinh hoc chinh dien")


def test_haar_fallback_no_crash():
    check(R.detect_face(np.zeros((300, 300, 3), np.uint8)) is None, "anh den khong co mat -> None, khong loi")


def lesion_stub(rgb_box_fn):
    """segment_fn gia lap: to nam trong nua duoi cua moi ROI (xac suat 1)."""
    def seg(crop):
        p = np.zeros(crop.shape[:2], np.float32)
        h = crop.shape[0]
        p[h // 2:, :] = 1.0
        return p
    return seg


def test_pipeline():
    rgb = fake_face()
    # lam toi vung nam that tren anh de D > 0
    dark = rgb.copy()
    rois = R.derive_rois(FACE["box"], rgb.shape[:2])
    for n, r in rois.items():
        x0, y0, x1, y1 = r["box"]
        dark[(y0 + y1) // 2:y1, x0:x1] = np.where(dark[(y0 + y1) // 2:y1, x0:x1].sum(-1, keepdims=True) > 0,
                                                    np.array((150, 100, 80), np.uint8), 0)
    res, mask, _ = P.analyse_image(dark, FACE, lesion_stub(None), use_skin=True)
    check(mask.shape == rgb.shape[:2] and mask.sum() > 0, "mask day du cung kich thuoc anh goc")
    fh = res["rois"]["forehead"]
    check(fh["A"] >= 1 and fh["D"] >= 1 and fh["H"] >= 1, f"forehead co A,D,H >= 1 (got {fh['A']},{fh['D']},{fh['H']})")
    check(res["masi"]["masi"] > 0, f"MASI > 0 khi co nam (got {res['masi']['masi']})")
    check(res["masi"]["complete"], "du 4 vung MASI")
    ob, _, _ = P.analyse_image(dark, FACE, lesion_stub(None), view="L")
    check(not ob["masi"]["complete"] and any("nghieng" in w for w in ob["warnings"]),
          "anh nghieng -> MASI khong day du + co canh bao")
    clean, _, _ = P.analyse_image(rgb, FACE, lambda c: np.zeros(c.shape[:2], np.float32))
    check(clean["masi"]["masi"] == 0 and all(m["A"] == 0 for m in clean["rois"].values()),
          "khong co nam -> MASI = 0")
    # hinh dang: nam rai rac phai co H cao hon nam dac
    def blob(crop):
        p = np.zeros(crop.shape[:2], np.float32)
        h, w = crop.shape[:2]
        p[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
        return p
    def speckle(crop):
        p = np.zeros(crop.shape[:2], np.float32)
        h, w = crop.shape[:2]
        s = max(12, int(0.12 * min(h, w)))  # du lon de khong bi loc nhieu
        for yy in range(h // 8, h - s, h // 4):
            for xx in range(w // 8, w - s, w // 4):
                p[yy:yy + s, xx:xx + s] = 1
        return p
    rb, _, _ = P.analyse_image(dark, FACE, blob)
    rs, _, _ = P.analyse_image(dark, FACE, speckle)
    check(rs["rois"]["forehead"]["H"] > rb["rois"]["forehead"]["H"],
          f"nam rai rac co H cao hon nam dac ({rs['rois']['forehead']['H']} > {rb['rois']['forehead']['H']})")


def test_prepare_roi():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data, out = td / "proc", td / "roi"
        for d in ("images", "masks"):
            (data / d).mkdir(parents=True)
        rows = []
        for i, (split, fold) in enumerate([("train", 0), ("val", 1), ("test", 2)]):
            name = f"Flaky_{i}_F"
            img = fake_face()
            m = np.zeros(img.shape[:2], np.uint8)
            m[700:800, 420:580] = 255
            cv2.imwrite(str(data / "images" / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(data / "masks" / f"{name}.png"), m)
            rows.append({"name": name, "group": str(i), "type": "Flaky", "view": "F", "split": split,
                         "fold": fold, "fg_percent": 1.0, "region_ids": ""})
        with open(data / "split.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        orig = R.detect_face
        R.detect_face = lambda *a, **k: dict(FACE)
        try:
            sys.argv = ["prepare_roi.py", "--data", str(data), "--out", str(out), "--size", "256", "--qa", "1"]
            PR.main()
        finally:
            R.detect_face = orig
        res = list(csv.DictReader(open(out / "split.csv", encoding="utf-8")))
        check(len(res) == 15, f"3 anh x 5 ROI = 15 dong split.csv (got {len(res)})")
        check({r["split"] for r in res} == {"train", "val", "test"}, "giu nguyen chia tap theo benh nhan")
        img = cv2.imread(str(out / "images" / f"{res[0]['name']}.jpg"))
        msk = cv2.imread(str(out / "masks" / f"{res[0]['name']}.png"), cv2.IMREAD_GRAYSCALE)
        check(img.shape[:2] == (256, 256) and msk.shape == (256, 256), "anh/mask ROI la khung vuong 256x256")
        check((out / "MANIFEST_ROI.txt").exists(), "co MANIFEST_ROI.txt")
        from memi_common import load_split
        tr, va, te = load_split(out / "split.csv", fold=-1)
        check(len(tr) == 5 and len(va) == 5 and len(te) == 5, "load_split cua script train doc duoc split.csv moi")


if __name__ == "__main__":
    for t in (test_area_score, test_masi_formula, test_local_reference_and_reliability, test_roi_geometry, test_haar_fallback_no_crash,
              test_pipeline, test_prepare_roi):
        print(f"\n[{t.__name__}]")
        t()
    print("\nTAT CA OK" if not check.failed else f"\n{check.failed} KIEM TRA LOI")
    sys.exit(1 if check.failed else 0)
