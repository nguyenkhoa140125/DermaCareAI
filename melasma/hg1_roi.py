"""
hg1_roi.py -- Huong 1, GIAI DOAN 1: tim khuon mat bang YOLO (pretrained) roi suy ra cac vung ROI
bang hinh hoc (tran, 2 ma, mui, cam).  KHONG can PyTorch khi import (ultralytics chi nap khi can).

Quy uoc trai/phai: theo BENH NHAN. Anh chinh dien -> ma phai cua benh nhan nam o ben TRAI anh.
Goc chup (view): F = chinh dien, L = nghieng trai (thay ma TRAI, mui huong sang trai anh),
                 R = nghieng phai (thay ma PHAI, mui huong sang phai anh).

ROI duoc tinh tu hop khuon mat (fx0, fy0, fx1, fy1), w = fx1 - fx0:
    x  = fx0 + a*w .. fx0 + b*w          (a, b: ti le theo chieu rong hop)
    y  = fy1 - up_top*w .. fy1 - up_bot*w (do cao tinh LEN TREN tu day hop, don vi = w)
Neo vao DAY hop va CHIEU RONG vi tren du lieu MEMI-DS (mat bi che bang mosaic) canh tren cua hop
YOLO nhay theo mosaic, con day hop (cam) va chieu rong on dinh hon nhieu.
Cac ti le duoc do tren 3 anh mau (F, L, R) cua mot benh nhan: hay chay --qa tren nhieu anh hon va
chinh ROI_SPECS (hoac truyen --roi-json) truoc khi tin vao ket qua.
Voi anh nghieng, ma o phia xa khong thay duoc nen khong co ROI do (MASI se bao thieu vung).
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROI_NAMES = ["forehead", "right_malar", "left_malar", "nose", "chin"]
MASI_ROIS = ["forehead", "right_malar", "left_malar", "chin"]  # MASI khong tinh mui

# (a, b, up_top, up_bot): x tu a*w den b*w ke tu mep trai hop; y tu (day hop - up_top*w) den (day hop - up_bot*w)
ROI_SPECS = {
    "F": {
        "forehead":    (0.10, 0.90, 1.33, 1.05),
        "right_malar": (0.02, 0.38, 0.66, 0.18),
        "left_malar":  (0.64, 0.98, 0.66, 0.18),
        "nose":        (0.38, 0.64, 0.66, 0.43),
        "chin":        (0.28, 0.72, 0.17, 0.00),
    },
    "L": {  # thay ma TRAI (ben phai anh); ma phai o phia xa khong thay
        "forehead":    (0.10, 0.90, 1.33, 1.05),
        "left_malar":  (0.38, 0.99, 0.66, 0.18),
        "nose":        (0.06, 0.38, 0.66, 0.43),
        "chin":        (0.18, 0.55, 0.17, 0.00),
    },
    "R": {  # thay ma PHAI (ben trai anh); ma trai o phia xa khong thay
        "forehead":    (0.10, 0.90, 1.33, 1.05),
        "right_malar": (0.01, 0.63, 0.66, 0.18),
        "nose":        (0.64, 0.95, 0.66, 0.43),
        "chin":        (0.45, 0.82, 0.17, 0.00),
    },
}
VIEWS = tuple(ROI_SPECS)


def norm_view(v):
    v = (v or "F").strip().upper()
    return v if v in ROI_SPECS else "F"


def view_from_name(name):
    """Doan goc chup tu ten file kieu 1_F.jpg / 12_L_xxx.png; khong co thi 'F'."""
    for part in Path(name).stem.upper().split("_")[1:]:
        if part in ROI_SPECS:
            return part
    return "F"


def view_has_suffix(name):
    return any(part in ROI_SPECS for part in Path(name).stem.upper().split("_")[1:])


def load_roi_specs(path=None):
    """Doc ROI_SPECS tu JSON {view: {ten_roi: [a,b,up_top,up_bot]}} (tuy chon); thieu thi dung mac dinh."""
    specs = {v: dict(d) for v, d in ROI_SPECS.items()}
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for v, d in data.items():
            if v not in specs:
                raise ValueError(f"view khong hop le trong {path}: {v}")
            for k, t in d.items():
                if k not in ROI_NAMES or len(t) != 4:
                    raise ValueError(f"ROI khong hop le trong {path}: {v}/{k}")
                specs[v][k] = tuple(float(x) for x in t)
    return specs


_YOLO_CACHE = {}
_YUNET_CACHE = {}


# ----------------------------------------------------------------------------
# Phat hien khuon mat
# ----------------------------------------------------------------------------
def _detect_yolo(rgb, weights, conf, device):
    from ultralytics import YOLO  # nap tre: chi can khi dung YOLO
    if weights not in _YOLO_CACHE:
        _YOLO_CACHE[weights] = YOLO(weights)
    res = _YOLO_CACHE[weights].predict(rgb[:, :, ::-1], conf=conf, verbose=False, device=device)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    cf = res.boxes.conf.cpu().numpy()
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(np.argmax(cf * np.sqrt(area)))  # uu tien hop vua chac chan vua lon
    x0, y0, x1, y1 = xyxy[i]
    return (float(x0), float(y0), float(x1), float(y1)), float(cf[i])


def _detect_yunet(rgb, onnx_path=None, conf=0.4):
    """Phat hien khuon mat bang OpenCV YuNet ONNX (ho tro 5 landmarks: mat, mui, mieng)."""
    if not hasattr(cv2, "FaceDetectorYN"):
        return None
    if onnx_path is None:
        p = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"
        if p.exists():
            onnx_path = str(p)
        else:
            p2 = Path(__file__).parent.parent / "models" / "face" / "face_detection_yunet_2023mar.onnx"
            if p2.exists():
                onnx_path = str(p2)
            else:
                return None
    if not Path(onnx_path).exists():
        return None
    H, W = rgb.shape[:2]
    try:
        if onnx_path not in _YUNET_CACHE:
            _YUNET_CACHE[onnx_path] = cv2.FaceDetectorYN.create(onnx_path, "", (W, H), score_threshold=conf)
        detector = _YUNET_CACHE[onnx_path]
        detector.setInputSize((W, H))
        detector.setScoreThreshold(conf)
        bgr = rgb[:, :, ::-1]
        _, faces = detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[2] * f[3] * f[14])
        x, y, w, h = float(best[0]), float(best[1]), float(best[2]), float(best[3])
        conf_val = float(best[14])
        landmarks = {
            "right_eye": (float(best[4]), float(best[5])),
            "left_eye": (float(best[6]), float(best[7])),
            "nose_tip": (float(best[8]), float(best[9])),
            "right_mouth": (float(best[10]), float(best[11])),
            "left_mouth": (float(best[12]), float(best[13])),
        }
        return (x, y, x + w, y + h), conf_val, landmarks
    except Exception:
        return None


def _detect_haar(rgb):
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data") or not hasattr(cv2.data, "haarcascades"):
        return None
    try:
        det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if det.empty():
            return None
        H, W = rgb.shape[:2]
        s = 800.0 / max(H, W) if max(H, W) > 800 else 1.0
        gray = cv2.cvtColor(cv2.resize(rgb, None, fx=s, fy=s), cv2.COLOR_RGB2GRAY)
        faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x / s, y / s, (x + w) / s, (y + h) / s), 1.0
    except Exception:
        return None


def detect_face(rgb, weights=None, conf=0.25, device=None, allow_haar=True, yunet_path=None, allow_yunet=True):
    """Tra ve dict {box:(x0,y0,x1,y1), conf, backend, [landmarks]} hoac None neu khong thay mat.
    Thu tu: YOLO (neu co weights) -> YuNet ONNX (co landmark) -> Haar Cascade (fallback)."""
    if weights and Path(weights).exists():
        r = _detect_yolo(rgb, str(weights), conf, device)
        if r is not None:
            res = {"box": r[0], "conf": r[1], "backend": "yolo"}
            if allow_yunet:
                yr = _detect_yunet(rgb, yunet_path, conf=0.3)
                if yr is not None:
                    res["landmarks"] = yr[2]
            return res
    if allow_yunet:
        yr = _detect_yunet(rgb, yunet_path, conf=conf)
        if yr is not None:
            return {"box": yr[0], "conf": yr[1], "backend": "yunet", "landmarks": yr[2]}
    if allow_haar:
        r = _detect_haar(rgb)
        if r is not None:
            return {"box": r[0], "conf": r[1], "backend": "haar"}
    return None


# ----------------------------------------------------------------------------
# Suy ra ROI tu hop khuon mat
# ----------------------------------------------------------------------------
def derive_rois(face_box, img_hw, specs=None, view="F", min_side=24, min_inside=0.5):
    """Tra ve dict ten -> {box:(x0,y0,x1,y1) nguyen, inside:ti le dien tich ROI nam trong anh}.
    Chi tra ve ROI co trong view (anh nghieng khong co ma phia xa). ROI bi cat qua nua
    (inside < min_inside) hoac qua be thi bi bo."""
    specs = specs or ROI_SPECS
    H, W = img_hw
    fx0, fy0, fx1, fy1 = face_box
    w = fx1 - fx0
    out = {}
    for name, (a, b, up_top, up_bot) in specs[norm_view(view)].items():
        x0, x1 = fx0 + a * w, fx0 + b * w
        y0, y1 = fy1 - up_top * w, fy1 - up_bot * w
        full = max(1e-6, (x1 - x0) * (y1 - y0))
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 - cx0 < min_side or cy1 - cy0 < min_side:
            continue
        inside = (cx1 - cx0) * (cy1 - cy0) / full
        if inside < min_inside:
            continue
        out[name] = {"box": (int(round(cx0)), int(round(cy0)), int(round(cx1)), int(round(cy1))),
                     "inside": round(float(inside), 3)}
    return out


def crop_box(arr, box):
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def draw_rois(rgb, face_box, rois, texts=None):
    """Ve hop khuon mat (xanh la) va cac ROI (mau theo vung) len ban sao cua anh, tra ve RGB."""
    colors = {"forehead": (255, 200, 0), "right_malar": (0, 180, 255), "left_malar": (255, 0, 200),
              "nose": (160, 160, 160), "chin": (0, 220, 120)}
    ov = rgb.copy()
    t = max(2, rgb.shape[1] // 400)
    scale = max(0.5, rgb.shape[1] / 1600)
    fx0, fy0, fx1, fy1 = [int(v) for v in face_box]
    cv2.rectangle(ov, (fx0, fy0), (fx1, fy1), (0, 255, 0), t)
    for name, r in rois.items():
        x0, y0, x1, y1 = r["box"]
        cv2.rectangle(ov, (x0, y0), (x1, y1), colors[name], t)
        label = name if not texts or name not in texts else f"{name} {texts[name]}"
        cv2.putText(ov, label, (x0 + 4, y0 + int(26 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale,
                    colors[name], max(1, t - 1), cv2.LINE_AA)
    return ov
