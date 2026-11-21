"""
melasma_seg.py -- Module khoanh vùng nghi ngờ nám và đánh giá lâm sàng DermaCare AI.

Hỗ trợ 3 chức năng chính:
1. Đánh giá nguy cơ nám từ bảng khảo sát dịch tễ học (XGBoost / Clinical Risk) theo nghiên cứu BV Phong - Da liễu Quy Hòa.
2. Pipeline Hướng 1 (Coarse-to-Fine): YOLO / YuNet phát hiện khuôn mặt & cắt 5 ROI giải phẫu ở độ phân giải gốc
   -> U-Net ResNet34 phân vùng trên từng ROI -> Tự động tính chỉ số diện tích và mức độ nặng MASI (0 - 48).
3. Pipeline toàn mặt (Baseline): U-Net 5-fold ensemble phân vùng nhanh toàn bộ khung ảnh.

Endpoint:
  GET  /api/melasma/status        kiểm tra trạng thái các mô hình
  POST /api/melasma/risk_survey   tiếp nhận dữ liệu khảo sát và trả về % nguy cơ + phân tích
  POST /api/melasma/pipeline_hg1  chạy quy trình 2 giai đoạn (ROI + MASI)
  POST /api/melasma/segment       hỗ trợ cả mode="hg1" (mặc định) và mode="baseline"

Quyền riêng tư: ảnh được xử lý hoàn toàn trong bộ nhớ RAM, không lưu ảnh xuống đĩa.
"""
import base64
import io
import json
import logging
import os
import sys
import threading
import time
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps

log = logging.getLogger("DermaCareAI.melasma")

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent
MELASMA_DIR = REPO_ROOT / "melasma"
RUNS_ROI_DIR = REPO_ROOT / "runs" / "unet_roi"
FACE_MODEL_PATH = REPO_ROOT / "models" / "face" / "yolov8n-face.pt"
YUNET_MODEL_PATH = MELASMA_DIR / "face_detection_yunet_2023mar.onnx"

# Thêm melasma/ vào sys.path để gọi hg1_roi, hg1_masi, pipeline_hg1
if str(MELASMA_DIR) not in sys.path:
    sys.path.insert(0, str(MELASMA_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

try:
    import melasma_risk
except ImportError:
    melasma_risk = None

MODEL_DIR = Path(os.environ.get("MELASMA_MODEL_DIR",
                                Path(__file__).resolve().parent.parent / "models" / "melasma"))
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_SIDE = 1600                # cạnh dài tối đa khi xử lý
WARN_BIG_AREA = 40.0           # % khung ảnh

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DISCLAIMER = ("Công cụ tham khảo và hỗ trợ nghiên cứu, không phải chẩn đoán y khoa và không thay thế bác sĩ da liễu. "
              "Kết quả ước lượng MASI dựa trên quy tắc tự động; hãy đến khám chuyên khoa nếu bạn lo lắng về làn da.")
ACCURACY_NOTE = ("Được huấn luyện và kiểm định chéo 5-fold theo bệnh nhân trên bộ dữ liệu MEMI-DS. "
                 "Pipeline Hướng 1 cắt 5 vùng giải phẫu (trán, má phải, má trái, mũi, cằm) ở độ phân giải gốc.")


class ModelUnavailable(Exception):
    pass


# ----------------------------------------------------------------------------
# Tiện ích xử lý ảnh
# ----------------------------------------------------------------------------
def decode_image(data):
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise ValueError("Không đọc được ảnh. Hãy chọn tệp JPG, PNG hoặc WebP hợp lệ.")
    arr = np.asarray(im)
    h, w = arr.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr)


def normalize(img_uint8):
    x = (img_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def letterbox(img_rgb, width, height):
    h0, w0 = img_rgb.shape[:2]
    s = min(width / w0, height / h0)
    nw, nh = max(1, int(round(w0 * s))), max(1, int(round(h0 * s)))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left, top = (width - nw) // 2, (height - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, (nw, nh, left, top)


def unletterbox(prob, info, orig_hw):
    nw, nh, left, top = info
    crop = prob[top:top + nh, left:left + nw]
    return cv2.resize(crop, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)


def make_overlay(img_rgb, mask, alpha=0.45):
    ov = img_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    ov[m] = (1 - alpha) * ov[m] + alpha * np.array([255, 30, 30], dtype=np.float32)
    return np.clip(ov, 0, 255).astype(np.uint8)


def to_data_url(img_rgb, quality=88):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Không mã hóa được ảnh kết quả")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def build_warnings(rgb, pct):
    h, w = rgb.shape[:2]
    out = []
    if min(h, w) < 300:
        out.append({"level": "warn", "text": "Ảnh có độ phân giải thấp nên kết quả kém tin cậy. Hãy chụp ảnh lớn hơn."})
    if w / h > 1.25:
        out.append({"level": "warn", "text": "Ảnh nằm ngang. Khuyến khích chụp dọc thấy rõ toàn bộ khuôn mặt."})
    if pct > WARN_BIG_AREA:
        out.append({"level": "warn", "text": "Vùng được khoanh rất lớn so với khung ảnh. Cần kiểm tra xem ảnh có bị chụp cận cảnh quá mức không."})
    if pct < 0.5:
        out.append({"level": "info", "text": "Mô hình không thấy vùng sắc tố rõ rệt (dưới 0.5%). Sắc tố nhạt lan tỏa có thể cần kiểm tra thêm."})
    return out


# ----------------------------------------------------------------------------
# 1. Baseline Full-Face Segmenter
# ----------------------------------------------------------------------------
class BaselineSegmenter:
    def __init__(self):
        self._lock = threading.Lock()
        self._models = None
        self._torch = None
        self._device = None
        self._thr = 0.5

    def available(self):
        return (find_spec("torch") is not None and find_spec("segmentation_models_pytorch") is not None
                and any(MODEL_DIR.glob("fold*/best.pt")))

    def load(self):
        if self._models is not None:
            return
        with self._lock:
            if self._models is not None:
                return
            try:
                import torch
                import segmentation_models_pytorch as smp
            except ImportError:
                raise ModelUnavailable("Máy chủ chưa cài PyTorch hoặc segmentation-models-pytorch.")
            ckpts = sorted(MODEL_DIR.glob("fold*/best.pt"))
            if not ckpts:
                raise ModelUnavailable(f"Chưa có mô hình trong {MODEL_DIR}.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models, thrs = [], []
            for p in ckpts:
                cfg = json.loads((p.parent / "run_config.json").read_text())
                m = smp.Unet(encoder_name=cfg["encoder"], encoder_weights=None, in_channels=3, classes=1)
                try:
                    sd = torch.load(p, map_location=device, weights_only=True)
                except TypeError:
                    sd = torch.load(p, map_location=device)
                m.load_state_dict(sd)
                models.append((m.to(device).eval(), cfg))
                thrs.append(float(cfg.get("thr", 0.5)))
            self._torch, self._device, self._thr = torch, device, float(np.mean(thrs))
            self._models = models
            log.info("Da nap %d mo hinh U-Net Full-Face tren %s", len(models), device)

    def predict(self, rgb, tta=True):
        self.load()
        torch = self._torch
        probs = []
        with self._lock, torch.no_grad():
            for m, cfg in self._models:
                canvas, info = letterbox(rgb, cfg["width"], cfg["height"])
                x = torch.from_numpy(normalize(canvas)).unsqueeze(0).to(self._device)
                p = torch.sigmoid(m(x).float())[0, 0]
                if tta:
                    p2 = torch.sigmoid(m(torch.flip(x, dims=[-1])).float())[0, 0]
                    p = 0.5 * (p + torch.flip(p2, dims=[-1]))
                probs.append(unletterbox(p.cpu().numpy(), info, rgb.shape[:2]))
        return np.mean(probs, axis=0)


# ----------------------------------------------------------------------------
# 2. Hướng 1 Coarse-to-Fine (ROI + MASI) Segmenter
# ----------------------------------------------------------------------------
class HG1Segmenter:
    def __init__(self):
        self._lock = threading.Lock()
        self._seg_fn = None
        self._thr = 0.5
        self._device = None
        self._ckpts = []

    def available(self):
        roi_ckpts = sorted(list(RUNS_ROI_DIR.glob("*/best.pt")))
        has_face = FACE_MODEL_PATH.exists() or YUNET_MODEL_PATH.exists()
        return len(roi_ckpts) > 0 and has_face

    def load(self, tta=False):
        if self._seg_fn is not None:
            return
        with self._lock:
            if self._seg_fn is not None:
                return
            roi_ckpts = sorted(list(RUNS_ROI_DIR.glob("*/best.pt")))
            if not roi_ckpts:
                raise ModelUnavailable(f"Chưa có mô hình ROI trong {RUNS_ROI_DIR}.")
            try:
                import torch
                import pipeline_hg1 as P
            except ImportError as e:
                raise ModelUnavailable(f"Thiếu thư viện để chạy Hướng 1: {e}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            seg_fn, thr = P.make_segment_fn(roi_ckpts, device=device, tta=tta)
            self._seg_fn = seg_fn
            self._thr = thr
            self._device = device
            self._ckpts = roi_ckpts
            log.info("Da nap %d mo hinh U-Net ROI cho Huong 1 tren %s", len(roi_ckpts), device)


SEG_BASELINE = BaselineSegmenter()
SEG_HG1 = HG1Segmenter()


def analyse_baseline(rgb, tta=True):
    t0 = time.time()
    prob = SEG_BASELINE.predict(rgb, tta=tta)
    thr = SEG_BASELINE._thr
    mask = (prob > thr).astype(np.uint8)
    pct = 100.0 * float(mask.mean())
    return {
        "success": True,
        "mode": "baseline",
        "area_percent": round(pct, 1),
        "original": to_data_url(rgb),
        "overlay": to_data_url(make_overlay(rgb, mask)),
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "warnings": build_warnings(rgb, pct),
        "n_models": len(SEG_BASELINE._models or []),
        "tta": bool(tta),
        "seconds": round(time.time() - t0, 2),
        "accuracy_note": ACCURACY_NOTE,
        "disclaimer": DISCLAIMER,
    }


def analyse_hg1(rgb, tta=False, view="auto"):
    t0 = time.time()
    SEG_HG1.load(tta=tta)
    import hg1_roi as R
    import pipeline_hg1 as P

    face_weights = str(FACE_MODEL_PATH) if FACE_MODEL_PATH.exists() else None
    yunet_path = str(YUNET_MODEL_PATH) if YUNET_MODEL_PATH.exists() else None

    face = R.detect_face(rgb, weights=face_weights, yunet_path=yunet_path)
    if face is None:
        raise ValueError("Không phát hiện được khuôn mặt trong ảnh. Hãy chụp thẳng hoặc nghiêng rõ khuôn mặt.")

    result, full_mask, rois = P.analyse_image(rgb, face, SEG_HG1._seg_fn, thr=SEG_HG1._thr, view=view)
    overlay_rgb = P.render(rgb, face, rois, full_mask, result)

    # Tạo ảnh crop chi tiết từng ROI
    roi_cards = {}
    roi_vn_names = {
        "forehead": "Trán (Forehead)",
        "right_malar": "Gò má phải (Right Malar)",
        "left_malar": "Gò má trái (Left Malar)",
        "nose": "Vùng mũi (Nose)",
        "chin": "Vùng cằm (Chin)",
    }
    for name, r in rois.items():
        crop = R.crop_box(rgb, r["box"])
        x0, y0, x1, y1 = r["box"]
        c_mask = full_mask[y0:y1, x0:x1]
        c_ov = make_overlay(crop, c_mask)
        m = result["rois"][name]
        roi_cards[name] = {
            "title": roi_vn_names.get(name, name),
            "crop_url": to_data_url(c_ov),
            "area_percent": round(m["area_percent"], 1),
            "A": m["A"],
            "D": m["D"],
            "H": m["H"],
            "delta_L": round(m["delta_L"], 2) if m.get("delta_L") is not None else None,
            "contrast_dE": round(m["contrast_dE"], 2) if m.get("contrast_dE") is not None else None,
            "d_reliable": m.get("d_reliable", True),
            "warnings": m.get("warnings", []),
        }

    masi_val = result["masi"]["masi"]
    if masi_val < 10.0:
        masi_severity = "Nhẹ (Mild)"
        masi_class = "mild"
    elif masi_val <= 20.0:
        masi_severity = "Trung bình (Moderate)"
        masi_class = "moderate"
    else:
        masi_severity = "Nặng (Severe)"
        masi_class = "severe"

    pct = 100.0 * float(full_mask.mean())
    return {
        "success": True,
        "mode": "hg1",
        "area_percent": round(pct, 1),
        "masi": {
            "score": round(masi_val, 1),
            "severity": masi_severity,
            "severity_class": masi_class,
            "complete": result["masi"]["complete"],
            "reliable": result["masi"]["reliable"],
            "contributions": {k: round(v, 2) for k, v in result["masi"]["contributions"].items()},
            "unreliable_D": result["masi"]["unreliable_D"],
        },
        "roi_cards": roi_cards,
        "rois": result["rois"],
        "face": result["face"],
        "view": result["view"],
        "original": to_data_url(rgb),
        "overlay": to_data_url(overlay_rgb),
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "warnings": result["warnings"],
        "n_models": len(SEG_HG1._ckpts),
        "seconds": round(time.time() - t0, 2),
        "accuracy_note": ACCURACY_NOTE,
        "disclaimer": DISCLAIMER,
    }


# ----------------------------------------------------------------------------
# Flask Blueprint & Endpoints
# ----------------------------------------------------------------------------
bp = Blueprint("melasma", __name__)
_rate_limit = None


@bp.route("/api/melasma/status")
def status():
    hg1_ok = SEG_HG1.available()
    base_ok = SEG_BASELINE.available()
    return jsonify({
        "available": hg1_ok or base_ok,
        "hg1_available": hg1_ok,
        "baseline_available": base_ok,
        "risk_survey_available": melasma_risk is not None,
        "message": "Hệ thống sẵn sàng (Hỗ trợ Coarse-to-Fine Hướng 1, Toàn mặt, và Khảo sát nguy cơ)."
    })


@bp.route("/api/melasma/risk_survey", methods=["POST"])
def risk_survey():
    if _rate_limit is not None and not _rate_limit():
        return jsonify({"success": False, "error": "Bạn gửi quá nhiều yêu cầu. Hãy đợi một chút."}), 429
    if melasma_risk is None:
        return jsonify({"success": False, "error": "Chưa nạp được module khảo sát lâm sàng."}), 500
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        res = melasma_risk.evaluate_risk(data)
        return jsonify(res)
    except Exception as e:
        log.exception("Loi khi danh gia khao sat nguy co")
        return jsonify({"success": False, "error": f"Lỗi xử lý khảo sát: {e}"}), 400


@bp.route("/api/melasma/segment", methods=["POST"])
@bp.route("/api/melasma/pipeline_hg1", methods=["POST"])
def segment():
    if _rate_limit is not None and not _rate_limit():
        return jsonify({"success": False, "error": "Bạn gửi quá nhiều yêu cầu. Hãy đợi một phút rồi thử lại."}), 429
    f = request.files.get("image")
    if f is None:
        return jsonify({"success": False, "error": "Chưa có ảnh. Hãy chọn một ảnh khuôn mặt."}), 400
    data = f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return jsonify({"success": False, "error": "Ảnh lớn hơn 10 MB. Hãy chọn ảnh nhỏ hơn."}), 413
    try:
        rgb = decode_image(data)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    if min(rgb.shape[:2]) < 128:
        return jsonify({"success": False, "error": "Ảnh quá nhỏ (dưới 128 điểm ảnh). Hãy chọn ảnh lớn hơn."}), 400

    # Chế độ phân tích: hg1 (mặc định nếu có) hoặc baseline
    mode = request.form.get("mode", "hg1" if SEG_HG1.available() else "baseline")
    tta = request.form.get("tta", "0") in ("1", "true", "True")
    view = request.form.get("view", "auto")

    try:
        if mode == "hg1" and SEG_HG1.available():
            result = analyse_hg1(rgb, tta=tta, view=view)
        elif SEG_BASELINE.available():
            result = analyse_baseline(rgb, tta=tta)
        elif SEG_HG1.available():
            result = analyse_hg1(rgb, tta=tta, view=view)
        else:
            return jsonify({"success": False, "error": "Chưa có mô hình nào sẵn sàng để phân tích."}), 503
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ModelUnavailable as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.error("Hết bộ nhớ GPU khi phân tích ảnh %sx%s", rgb.shape[1], rgb.shape[0])
            return jsonify({"success": False, "error": "Hết bộ nhớ đồ họa. Hãy thử ảnh nhỏ hơn."}), 503
        log.exception("Lỗi khi phân tích ảnh")
        return jsonify({"success": False, "error": f"Phân tích thất bại: {e}"}), 500
    except Exception as e:
        log.exception("Lỗi khi phân tích ảnh")
        return jsonify({"success": False, "error": f"Phân tích thất bại: {e}"}), 500

    log.info("melasma (%s): %sx%s, %.1f%% (%.2fs)", result.get("mode"), rgb.shape[1], rgb.shape[0],
             result.get("area_percent", 0), result.get("seconds", 0))
    return jsonify(result)


def register(app, rate_limit=None):
    global _rate_limit
    _rate_limit = rate_limit
    app.register_blueprint(bp)
