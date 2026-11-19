"""
melasma_seg.py -- Khoanh vung nghi ngo nam tren anh khuon mat (U-Net, ghep cac mo hinh cac fold).

Gan vao DermaCare: them 3 dong vao server.py (xem HUONG_DAN_DEMO.md):
    from melasma_seg import register
    register(app, rate_limit_check)

Endpoint:
    GET  /api/melasma/status    kiem tra thu vien + mo hinh
    POST /api/melasma/segment   multipart, truong 'image' -> JSON (anh goc, anh phu mask, % dien tich, canh bao)

Quyen rieng tu: anh duoc xu ly trong bo nho va KHONG duoc luu; log chi ghi kich thuoc, thoi gian, %.
Mo hinh duoc nap luc co yeu cau dau tien, nen DermaCare van chay binh thuong neu chua cai torch.
"""
import base64
import io
import json
import logging
import os
import threading
import time
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps

log = logging.getLogger("DermaCareAI.melasma")

MODEL_DIR = Path(os.environ.get("MELASMA_MODEL_DIR",
                                Path(__file__).resolve().parent.parent / "models" / "melasma"))
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_SIDE = 1600                # canh dai toi da khi xu ly
WARN_BIG_AREA = 40.0           # % khung anh: tren muc nay thuong la anh can canh / khong phai mat

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DISCLAIMER = ("Công cụ tham khảo, không phải chẩn đoán y khoa và không thay thế bác sĩ da liễu. "
              "Kết quả có thể sai; hãy đến khám nếu bạn lo lắng về làn da.")
ACCURACY_NOTE = ("Trên 130 ảnh chụp lâm sàng của bộ MEMI-DS, một mô hình đơn lẻ ước lượng sai lệch trung bình "
                 "khoảng 3–4 điểm phần trăm diện tích. Bản ghép 5 mô hình dùng ở đây chưa được kiểm định riêng, "
                 "và độ chính xác trên ảnh điện thoại chưa được đo.")


class ModelUnavailable(Exception):
    pass


# ----------------------------------------------------------------------------
# Xu ly anh (numpy / OpenCV / Pillow, khong can torch)
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
    ov[m] = (1 - alpha) * ov[m] + alpha * np.array([255, 0, 0], dtype=np.float32)
    return ov.astype(np.uint8)


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
        out.append({"level": "warn", "text": "Ảnh nằm ngang. Mô hình học trên ảnh khuôn mặt chụp dọc; "
                                             "hãy chụp dọc và để thấy cả khuôn mặt."})
    if pct > WARN_BIG_AREA:
        out.append({"level": "warn", "text": "Vùng được khoanh rất lớn so với khung ảnh. Điều này thường gặp khi ảnh "
                                             "chụp quá gần hoặc không phải khuôn mặt; hãy chụp lại từ xa hơn."})
    if pct < 0.5:
        out.append({"level": "info", "text": "Mô hình không thấy vùng sắc tố rõ rệt. Sắc tố nhạt và lan tỏa "
                                             "có thể bị bỏ sót, nên đây chưa phải kết luận là da không có nám."})
    return out


# ----------------------------------------------------------------------------
# Mo hinh
# ----------------------------------------------------------------------------
class Segmenter:
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
                raise ModelUnavailable("Máy chủ chưa cài PyTorch. Chạy: pip install torch "
                                       "segmentation-models-pytorch (xem HUONG_DAN_DEMO.md).")
            ckpts = sorted(MODEL_DIR.glob("fold*/best.pt"))
            if not ckpts:
                raise ModelUnavailable(f"Chưa có mô hình trong {MODEL_DIR}. Chạy export_demo_models.py "
                                       "để chép các tệp best.pt vào đó.")
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
            log.info("Da nap %d mo hinh U-Net tren %s", len(models), device)

    def info(self):
        return {"n_models": len(self._models or []), "device": str(self._device), "thr": self._thr}

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


SEG = Segmenter()


def analyse(rgb, tta=True):
    t0 = time.time()
    prob = SEG.predict(rgb, tta=tta)
    thr = SEG._thr
    mask = (prob > thr).astype(np.uint8)
    pct = 100.0 * float(mask.mean())
    info = SEG.info()
    return {
        "success": True,
        "area_percent": round(pct, 1),
        "original": to_data_url(rgb),
        "overlay": to_data_url(make_overlay(rgb, mask)),
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "warnings": build_warnings(rgb, pct),
        "n_models": info["n_models"],
        "tta": bool(tta),
        "seconds": round(time.time() - t0, 2),
        "accuracy_note": ACCURACY_NOTE,
        "disclaimer": DISCLAIMER,
    }


# ----------------------------------------------------------------------------
# Flask
# ----------------------------------------------------------------------------
bp = Blueprint("melasma", __name__)
_rate_limit = None


@bp.route("/api/melasma/status")
def status():
    ok = SEG.available()
    return jsonify({"available": ok, "model_dir": str(MODEL_DIR),
                    "message": "Sẵn sàng." if ok else
                    "Chưa sẵn sàng: cần cài torch, segmentation-models-pytorch và chép mô hình vào models/melasma."})


@bp.route("/api/melasma/segment", methods=["POST"])
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
    try:
        result = analyse(rgb, tta=request.form.get("tta", "1") != "0")
    except ModelUnavailable as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.error("Het bo nho GPU khi phan tich anh %sx%s", rgb.shape[1], rgb.shape[0])
            return jsonify({"success": False, "error": "Hết bộ nhớ đồ họa. Hãy thử ảnh nhỏ hơn."}), 503
        log.exception("Loi khi phan tich anh")
        return jsonify({"success": False, "error": "Phân tích thất bại. Xem nhật ký máy chủ để biết chi tiết."}), 500
    except Exception:
        log.exception("Loi khi phan tich anh")
        return jsonify({"success": False, "error": "Phân tích thất bại. Xem nhật ký máy chủ để biết chi tiết."}), 500
    log.info("melasma: anh %sx%s, %.1f%% (%.2fs, %d mo hinh)", rgb.shape[1], rgb.shape[0],
             result["area_percent"], result["seconds"], result["n_models"])
    return jsonify(result)


def register(app, rate_limit=None):
    global _rate_limit
    _rate_limit = rate_limit
    app.register_blueprint(bp)
