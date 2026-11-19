"""
predict.py -- Dua anh khuon mat vao, nhan lai anh phu mask vung nam + % dien tich nam.
Ho tro SegFormer (thu muc best_model), U-Net (file best.pt) va GHEP nhieu mo hinh
(trung binh xac suat).

Vi du:
  # 1 mo hinh SegFormer
  python predict.py --ckpt runs/segformer_b0/fold0/best_model --image anh.jpg
  # 1 mo hinh U-Net
  python predict.py --ckpt runs/unet_r34/fold0/best.pt --image anh.jpg --tta
  # ghep 5 mo hinh U-Net cua 5 fold (khuyen dung cho demo)
  python predict.py --ckpt runs/unet_r34/fold0/best.pt runs/unet_r34/fold1/best.pt runs/unet_r34/fold2/best.pt runs/unet_r34/fold3/best.pt runs/unet_r34/fold4/best.pt --image thu_muc_anh --out ket_qua --tta

Dau ra (trong thu muc --out):
  <ten>_overlay.jpg   anh goc + vung nam to mau do
  <ten>_mask.png      mask nhi phan (255 = nam), CUNG kich thuoc anh goc
  results.json        % dien tich nam cho tung anh

Luu y:
  * % nam = so pixel nam / so pixel trong khung anh duoc phan tich. Day la chi so tham khao,
    KHONG phai MASI, chua duoc kiem dinh lam sang. Sai so do tren MEMI-DS khoang 3-4 diem phan tram.
  * Ghep 5 mo hinh cac fold KHONG the danh gia cong bang tren MEMI-DS (moi mo hinh da thay
    anh cua cac fold khac), nen khong dung de bao cao Dice.
  * Mo hinh hoc tren anh chup lam sang (nen den, mat bi che). Anh dien thoai co the kem hon.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

import memi_common as C

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="thu muc best_model (SegFormer) va/hoac file best.pt (U-Net); nhieu duong dan = ghep")
    p.add_argument("--image", required=True, help="duong dan anh hoac thu muc anh")
    p.add_argument("--out", default="predictions")
    p.add_argument("--thr", type=float, default=None, help="nguong (mac dinh lay tu luc train)")
    p.add_argument("--tta", action="store_true", help="trung binh voi anh lat ngang")
    p.add_argument("--face-crop", action="store_true", help="cat quanh khuon mat truoc khi phan vung")
    return p.parse_args()


def load_model(path, device):
    """Tra ve (ham_logits, cau_hinh). ham_logits(x) -> logits cung kich thuoc x."""
    p = Path(path)
    if p.is_dir():  # SegFormer
        cfg = json.loads((p / "memi_config.json").read_text())
        model = SegformerForSemanticSegmentation.from_pretrained(p).to(device).eval()

        def fwd(x):
            lg = model(pixel_values=x).logits
            return F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)
    else:  # U-Net (best.pt) -- cau hinh nam canh file
        import segmentation_models_pytorch as smp
        cfg = json.loads((p.parent / "run_config.json").read_text())
        model = smp.Unet(encoder_name=cfg["encoder"], encoder_weights=None, in_channels=3, classes=1)
        try:
            sd = torch.load(p, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(p, map_location=device)
        model.load_state_dict(sd)
        model = model.to(device).eval()
        fwd = model
    return fwd, cfg


@torch.no_grad()
def predict_prob(fwd, canvas_rgb, device, tta):
    x = torch.from_numpy(C.normalize(canvas_rgb)).unsqueeze(0).to(device)
    p = torch.sigmoid(fwd(x).float())[0, 0]
    if tta:
        p2 = torch.sigmoid(fwd(torch.flip(x, dims=[-1])).float())[0, 0]
        p = 0.5 * (p + torch.flip(p2, dims=[-1]))
    return p.cpu().numpy()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(c, device) for c in a.ckpt]
    thr = a.thr if a.thr is not None else float(np.mean([cfg.get("thr", 0.5) for _, cfg in models]))
    print(f">>> {len(models)} mo hinh | nguong {thr:.2f} | TTA={a.tta} | thiet bi {device}")

    src = Path(a.image)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    for f in files:
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)  # OpenCV tu xoay theo EXIF
        if bgr is None:
            print(f"[!] Khong doc duoc {f}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        box, work = None, rgb
        if a.face_crop:
            work, box = C.crop_face(rgb, target_aspect=models[0][1]["width"] / models[0][1]["height"])
            if box is None:
                print(f"    [{f.name}] khong tim thay mat, dung ca anh")

        probs = []
        for fwd, cfg in models:
            canvas, info = C.letterbox(work, cfg["width"], cfg["height"])
            pr = predict_prob(fwd, canvas, device, a.tta)
            probs.append(C.unletterbox(pr, info, work.shape[:2]))
        mask_work = (np.mean(probs, axis=0) > thr).astype(np.uint8)

        full_mask = np.zeros(rgb.shape[:2], np.uint8)
        if box is None:
            full_mask = mask_work
        else:
            x0, y0, x1, y1 = box
            full_mask[y0:y1, x0:x1] = mask_work

        pct = 100.0 * float(mask_work.mean())
        ov = C.make_overlay(rgb, full_mask)
        if box is not None:
            cv2.rectangle(ov, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), max(2, rgb.shape[1] // 400))
        cv2.imwrite(str(out / f"{f.stem}_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{f.stem}_mask.png"), full_mask * 255)
        results[f.name] = {"melasma_area_percent": round(pct, 2), "face_cropped": box is not None,
                           "n_models": len(models)}
        print(f"    {f.name}: vung nam = {pct:.2f}% dien tich phan tich")

    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\nHOAN TAT. Ket qua trong {out.resolve()}")
    print("Luu y: day la cong cu tham khao, khong thay the chan doan cua bac si da lieu.")


if __name__ == "__main__":
    main()
