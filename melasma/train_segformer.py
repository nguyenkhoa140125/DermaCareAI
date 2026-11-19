"""
train_segformer.py -- Huan luyen SegFormer phan vung nam da tren MEMI-DS, CHAY LOCAL.

Dac diem:
  * Loss Dice + BCE (vung nam chi ~9% anh).
  * Mixed precision (AMP) + tich luy gradient -> vua GPU ~4GB.
  * Chia theo benh nhan; ho tro 5-fold cross-validation (--all-folds).
  * Tu dong luu checkpoint moi epoch, CHAY TIEP duoc neu bi ngat (--resume).
  * Early stopping theo Dice tren tap val; danh gia tap test mot lan o cuoi.

Vi du:
  python train_segformer.py --smoke                      # thu nhanh 1-2 phut, kiem tra moi thu chay duoc
  python train_segformer.py --data memi_processed        # huan luyen 1 lan (chia co dinh)
  python train_segformer.py --data memi_processed --all-folds --out runs/segformer_b0
  python train_segformer.py --data memi_processed --resume --out runs/segformer_b0

Neu bao "out of memory": giam --batch (vd 2) va tang --accum (vd 2), hoac giam --width/--height.
Neu loi DataLoader tren Windows: them --workers 0.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

import memi_common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="memi_processed")
    p.add_argument("--out", default="runs/segformer_b0")
    p.add_argument("--model", default="nvidia/mit-b0",
                   help="nvidia/mit-b0 (nhe nhat). Co the thu nvidia/mit-b1 neu du VRAM")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=1, help="so buoc tich luy gradient")
    p.add_argument("--lr", type=float, default=6e-5, help="lr encoder; decoder head dung x10")
    p.add_argument("--patience", type=int, default=15, help="early stopping (so epoch)")
    p.add_argument("--fold", type=int, default=-1, help="-1 = chia co dinh trong split.csv")
    p.add_argument("--all-folds", action="store_true", help="chay lan luot 5 fold")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--smoke", action="store_true", help="chay thu cuc nhanh, khong tai model")
    return p.parse_args()


# ----------------------------------------------------------------------------
class SegDS(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, mask = self.items[i]
        if self.train:
            img, mask = C.augment(img, mask, np.random.default_rng())
        x = torch.from_numpy(C.normalize(img))
        y = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return x, y


def build_model(args):
    if args.smoke:
        return SegformerForSemanticSegmentation(SegformerConfig(num_labels=1))
    return SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True)


def logits_full(model, x):
    lg = model(pixel_values=x).logits  # (B,1,H/4,W/4)
    return F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)


def dice_bce_loss(logits, target):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1.0 - (2 * inter + 1.0) / (den + 1.0)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, loader, device, use_amp, thr):
    model.eval()
    per_img, losses = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            lg = logits_full(model, x)
        losses.append(dice_bce_loss(lg, y).item())
        pred = (torch.sigmoid(lg.float()) > thr).cpu().numpy()[:, 0]
        gt = y.cpu().numpy()[:, 0]
        for p, g in zip(pred, gt):
            per_img.append(C.seg_metrics(p, g))
    return float(np.mean(losses)), per_img


def make_scaler(use_amp):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


# ----------------------------------------------------------------------------
def run_fold(args, fold, all_data, device):
    out = Path(args.out) / (f"fold{fold}" if fold >= 0 else "fixed")
    out.mkdir(parents=True, exist_ok=True)
    tr_n, va_n, te_n = C.load_split(Path(args.data) / "split.csv", fold, args.n_folds)
    if args.smoke:
        tr_n, va_n, te_n = tr_n[:6], va_n[:4], te_n[:4]
    print(f"\n=== {out.name}: train {len(tr_n)} | val {len(va_n)} | test {len(te_n)} anh ===")

    mk = lambda names, train: SegDS([all_data[n] for n in names], train)
    pin = device.type == "cuda"
    train_ld = DataLoader(mk(tr_n, True), batch_size=args.batch, shuffle=True, drop_last=len(tr_n) > args.batch,
                          num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
    val_ld = DataLoader(mk(va_n, False), batch_size=2, shuffle=False, num_workers=0)
    test_ld = DataLoader(mk(te_n, False), batch_size=2, shuffle=False, num_workers=0)

    model = build_model(args).to(device)
    enc = [p for n, p in model.named_parameters() if n.startswith("segformer.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("segformer.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": head, "lr": args.lr * 10}],
                            weight_decay=0.01)
    steps_per_epoch = max(1, len(train_ld) // args.accum)
    total = args.epochs * steps_per_epoch
    warm = min(2 * steps_per_epoch, max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: C.cosine_warmup(s, warm, total))
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = make_scaler(use_amp)

    cfg = {"width": args.width, "height": args.height, "model": args.model, "thr": args.thr}
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1))

    start, best, bad, history = 1, -1.0, 0, []
    last = out / "last.pt"
    if args.resume and last.exists():
        try:
            ck = torch.load(last, map_location=device, weights_only=False)
        except TypeError:  # PyTorch cu khong co tham so weights_only
            ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, bad, history = ck["epoch"] + 1, ck["best"], ck["bad"], ck["history"]
        print(f">>> Tiep tuc tu epoch {start} (best val Dice = {best:.4f})")

    log_path = out / "training_log.csv"
    if not (args.resume and log_path.exists()):
        log_path.write_text("epoch,train_loss,val_loss,val_dice,val_iou,lr,seconds\n")

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        run, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_ld):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                lg = logits_full(model, x)
            loss = dice_bce_loss(lg, y)
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            run += loss.item(); n += 1

        vloss, vimg = evaluate(model, val_ld, device, use_amp, args.thr)
        vs = C.summarize(vimg)
        vd, vi = vs["dice"][0], vs["iou"][0]
        lr_now = opt.param_groups[0]["lr"]
        sec = time.time() - t0
        print(f"[{out.name}] epoch {epoch:3d}/{args.epochs}  train_loss={run / max(1, n):.4f}  "
              f"val_loss={vloss:.4f}  val_Dice={vd:.4f}  val_IoU={vi:.4f}  ({sec:.0f}s)")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{run / max(1, n):.5f},{vloss:.5f},{vd:.5f},{vi:.5f},{lr_now:.3e},{sec:.1f}\n")

        if vd > best:
            best, bad = vd, 0
            model.save_pretrained(out / "best_model")
            (out / "best_model" / "memi_config.json").write_text(json.dumps({**cfg, "epoch": epoch}, indent=1))
            print(f"    -> luu best_model (val Dice {vd:.4f})")
        else:
            bad += 1
        history.append([epoch, vd])
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "bad": bad, "history": history}, last)
        if bad >= args.patience:
            print(f">>> Early stopping: {args.patience} epoch khong cai thien.")
            break

    # ---------- danh gia tap test bang mo hinh tot nhat (CHI MOT LAN) ----------
    best_model = SegformerForSemanticSegmentation.from_pretrained(out / "best_model").to(device)
    _, timg = evaluate(best_model, test_ld, device, use_amp, args.thr)
    ts = C.summarize(timg)
    ov_dir = out / "test_overlays"
    ov_dir.mkdir(exist_ok=True)
    with open(out / "per_image_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "dice", "iou", "precision", "recall", "pred_pct", "gt_pct"])
        best_model.eval()
        for name, m in zip(te_n, timg):
            w.writerow([name] + [f"{m[k]:.4f}" for k in ["dice", "iou", "precision", "recall", "pred_pct", "gt_pct"]])
            img, gt = all_data[name]
            x = torch.from_numpy(C.normalize(img)).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                lg = logits_full(best_model, x)
            pred = (torch.sigmoid(lg.float())[0, 0] > args.thr).cpu().numpy()
            cv2.imwrite(str(ov_dir / f"{name}.jpg"),
                        cv2.cvtColor(C.make_overlay(img, pred, gt), cv2.COLOR_RGB2BGR))
    print(f"\n[{out.name}] TEST ({len(timg)} anh): Dice {ts['dice'][0]:.4f} +/- {ts['dice'][1]:.4f} | "
          f"IoU {ts['iou'][0]:.4f} | Precision {ts['precision'][0]:.4f} | Recall {ts['recall'][0]:.4f}")
    (out / "test_summary.json").write_text(json.dumps({k: list(v) for k, v in ts.items()}, indent=1))
    return {"fold": fold, "val_dice": best, **{f"test_{k}": v[0] for k, v in ts.items()}, "n_test": len(timg)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.smoke:
        args.width, args.height, args.epochs, args.workers, args.batch = 256, 384, 2, 0, 2
        args.out = "runs/smoke"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Thiet bi: {device}", end="")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" | {torch.cuda.get_device_name(0)} | {vram:.1f} GB VRAM")
        if vram < 5 and args.batch > 2:
            print("    Goi y: VRAM thap, neu bi OOM hay dung --batch 2 --accum 2")
    else:
        print("\n    CANH BAO: khong thay GPU CUDA -> chay CPU se RAT cham (xem huong dan cai PyTorch CUDA).")

    folds = list(range(args.n_folds)) if args.all_folds else [args.fold]
    split_csv = Path(args.data) / "split.csv"
    names = set()
    for f in folds:
        names |= set(sum(map(list, C.load_split(split_csv, f, args.n_folds)), []))
    print(f">>> Doc {len(names)} anh vao RAM o {args.width}x{args.height}...")
    all_data = C.load_arrays(args.data, sorted(names), args.width, args.height)

    results = []
    try:
        for f in folds:
            results.append(run_fold(args, f, all_data, device))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise SystemExit("\n[!] Het VRAM. Thu: --batch 2 --accum 2  hoac --width 384 --height 576")
        raise

    if len(results) > 1:
        out = Path(args.out)
        with open(out / "cv_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        d = np.array([r["test_dice"] for r in results]); i = np.array([r["test_iou"] for r in results])
        print(f"\n=========== {len(results)}-FOLD CROSS-VALIDATION ===========")
        print(f"Dice: {d.mean():.4f} +/- {d.std():.4f}   IoU: {i.mean():.4f} +/- {i.std():.4f}")
        print(f"Chi tiet: {out / 'cv_summary.csv'}")


if __name__ == "__main__":
    main()
