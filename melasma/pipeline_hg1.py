"""
pipeline_hg1.py -- Huong 1 (Coarse-to-Fine): anh mat -> YOLO tim khuon mat -> ROI (tran, 2 ma, mui, cam)
-> crop o DO PHAN GIAI GOC -> phan vung nam tren tung ROI (U-Net / SegFormer) -> % dien tich + MASI.

Vi du:
  python melasma/pipeline_hg1.py --image anh.jpg --out ket_qua_hg1 \
      --face-weights models/face/yolov8n-face.pt \
      --ckpt runs/unet_roi/fold0/best.pt runs/unet_roi/fold1/best.pt --tta

  # chi xem ROI co cat dung cho khong (khong can mo hinh phan vung):
  python melasma/pipeline_hg1.py --image anh.jpg --out qa_roi --qa

Mo hinh phan vung (--ckpt) nen la mo hinh train tren ROI (xem prepare_roi.py). Dung mo hinh train tren
ca khuon mat cung chay duoc nhung lech ti le (ROI phong to) nen ket qua kem tin cay hon.

Dau ra trong --out:  <ten>_hg1.jpg (ROI + vung nam), <ten>_mask.png (cung kich thuoc anh goc),
                     results.json (tung ROI + MASI).
Luu y: D va H cua MASI duoc UOC LUONG tu dong bang quy tac thu cong, chua kiem chung lam sang.
Day la cong cu nghien cuu, khong phai chan doan.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import hg1_masi as M
import hg1_roi as R
import memi_common as C

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISCLAIMER = ("Cong cu nghien cuu. A tinh tu mask; D va H uoc luong tu dong theo quy tac thu cong "
              "(chua kiem chung voi bac si). Khong phai chan doan.")


def analyse_image(rgb, face, segment_fn, specs=None, thr=0.5, use_skin=True, view="F"):
    """Chay giai doan 1 (da co 'face') -> crop -> segment_fn -> MASI.
    face: dict {'box':..., 'conf':..., 'backend':...}; segment_fn(crop_rgb) -> xac suat HxW (0..1) cung kich thuoc crop.
    Tra ve (result_dict, full_mask uint8 {0,1} cung kich thuoc rgb, rois)."""
    H, W = rgb.shape[:2]
    view = R.norm_view(view)
    rois = R.derive_rois(face["box"], (H, W), specs, view)
    full = np.zeros((H, W), np.uint8)
    crops, lesions = {}, {}
    for name, r in rois.items():
        crop = R.crop_box(rgb, r["box"])
        prob = segment_fn(crop)
        les = (prob > thr).astype(np.uint8)
        x0, y0, x1, y1 = r["box"]
        full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], les)
        crops[name], lesions[name] = crop, les

    skins = {n: (M.skin_mask(c) if use_skin else np.ones(lesions[n].shape, bool)) for n, c in crops.items()}
    ref_L = M.global_reference_L([(crops[n], lesions[n].astype(bool) & skins[n], skins[n]) for n in crops])
    metrics, warnings = {}, []
    for name in rois:
        m = M.roi_metrics(crops[name], lesions[name], ref_L, use_skin=use_skin)
        m["box"] = list(rois[name]["box"])
        m["roi_inside_image"] = rois[name]["inside"]
        metrics[name] = m
        warnings += [f"{name}: {w}" for w in m["warnings"]]
    masi = M.compute_masi(metrics)
    if masi["missing_rois"]:
        warnings.append("thieu vung MASI: " + ", ".join(masi["missing_rois"]) + " -> MASI co the thap hon that")
    if face["backend"] != "yolo":
        warnings.append("khuon mat duoc tim bang Haar (khong co trong so YOLO hoac YOLO khong thay) -> ROI kem chinh xac")
    if view != "F":
        warnings.append(f"anh nghieng ({view}): chi co ma ben nhin thay, MASI cua anh nay khong day du")
    result = {"view": view, "face": {"box": [round(v, 1) for v in face["box"]], "conf": round(face["conf"], 3),
                       "backend": face["backend"]},
              "ref_L": None if ref_L is None else round(ref_L, 2),
              "rois": metrics, "masi": masi, "warnings": warnings, "disclaimer": DISCLAIMER}
    return result, full, rois


def make_segment_fn(ckpts, device=None, tta=False, thr=None):
    """Nap cac mo hinh (U-Net best.pt hoac SegFormer best_model) va tra ve (segment_fn, nguong)."""
    import torch
    import predict as P  # nap tre vi can torch + transformers
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = [P.load_model(c, device) for c in ckpts]
    use_thr = thr if thr is not None else float(np.mean([cfg.get("thr", 0.5) for _, cfg in models]))

    def segment(crop_rgb):
        probs = []
        for fwd, cfg in models:
            canvas, info = C.letterbox(crop_rgb, cfg["width"], cfg["height"])
            pr = P.predict_prob(fwd, canvas, device, tta)
            probs.append(C.unletterbox(pr, info, crop_rgb.shape[:2]))
        return np.mean(probs, axis=0)

    return segment, use_thr


def render(rgb, face, rois, full_mask, result):
    texts = {n: f"{m['area_percent']:.1f}%" for n, m in result["rois"].items()}
    ov = C.make_overlay(rgb, full_mask) if full_mask.any() else rgb.copy()
    ov = R.draw_rois(ov, face["box"], rois, texts)
    label = f"MASI~{result['masi']['masi']:.1f}" + ("" if result["masi"]["complete"] else " (thieu vung)") \
        + ("" if result["masi"]["reliable"] or not result["masi"]["complete"] else " (D?)")
    cv2.putText(ov, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, rgb.shape[1] / 1200),
                (255, 255, 255), 3, cv2.LINE_AA)
    return ov


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="anh hoac thu muc anh")
    p.add_argument("--out", default="ket_qua_hg1")
    p.add_argument("--face-weights", default=None, help="trong so YOLO nhan dien khuon mat (.pt)")
    p.add_argument("--face-conf", type=float, default=0.25)
    p.add_argument("--ckpt", nargs="+", default=None, help="mo hinh phan vung (best.pt / best_model), nhieu = ghep")
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--roi-json", default=None, help="JSON ghi de ti le ROI (xem hg1_roi.ROI_SPECS)")
    p.add_argument("--view", default="auto", choices=["auto", "F", "L", "R"],
                   help="goc chup: F chinh dien, L/R nghieng. auto = doan tu duoi ten file (_F/_L/_R), khong co thi F")
    p.add_argument("--no-skin-mask", action="store_true", help="tinh % tren toan ROI, khong loc da")
    p.add_argument("--qa", action="store_true", help="chi ve ROI de kiem tra, khong phan vung")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    if not a.qa and not a.ckpt:
        raise SystemExit("Can --ckpt (mo hinh phan vung) hoac dung --qa de chi xem ROI.")
    specs = R.load_roi_specs(a.roi_json)
    segment, thr = (None, 0.5) if a.qa else make_segment_fn(a.ckpt, a.device, a.tta, a.thr)
    if a.qa:
        segment = lambda crop: np.zeros(crop.shape[:2], np.float32)

    src = Path(a.image)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for f in files:
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[!] Khong doc duoc {f}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face = R.detect_face(rgb, a.face_weights, a.face_conf, a.device)
        if face is None:
            print(f"[!] {f.name}: khong tim thay khuon mat -> bo qua")
            all_res[f.name] = {"error": "no_face"}
            continue
        view = R.view_from_name(f.name) if a.view == "auto" else a.view
        res, mask, rois = analyse_image(rgb, face, segment, specs, thr, not a.no_skin_mask, view)
        if a.view == "auto" and not R.view_has_suffix(f.name):
            res["warnings"].insert(0, "goc chup khong ro tu ten file -> dung chinh dien (F). Neu anh nghieng, "
                                      "chay lai voi --view L hoac --view R, neu khong ROI se sai")
        ov = render(rgb, face, rois, mask, res) if not a.qa else R.draw_rois(rgb, face["box"], rois)
        cv2.imwrite(str(out / f"{f.stem}_hg1.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        if not a.qa:
            cv2.imwrite(str(out / f"{f.stem}_mask.png"), mask * 255)
            m = res["masi"]
            print(f"    {f.name} [view {view}]: MASI~{m['masi']:.2f}"
                  + "".join(f" | {n} {r['area_percent']:.1f}% (A{r['A']} D{r['D']} H{r['H']})"
                            for n, r in res["rois"].items() if n in M.MASI_ROIS))
            for w in res["warnings"]:
                print(f"      [!] {w}")
        else:
            print(f"    {f.name} [view {view}]: {len(rois)} ROI, backend={face['backend']} conf={face['conf']:.2f}")
        all_res[f.name] = res
    (out / "results.json").write_text(json.dumps(all_res, indent=1, ensure_ascii=False))
    print(f"\nHOAN TAT. Ket qua trong {out.resolve()}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
