"""
train_unet.py -- Baseline U-Net (encoder ResNet34 pretrained ImageNet) tren MEMI-DS.

Dung CHUNG du lieu, augmentation, loss (Dice+BCE), 5 fold, chi so va dinh dang dau ra voi
train_segformer.py -> co the dung analyze_cv.py / compare_models.py de so sanh cong bang.

Can cai them:  pip install segmentation-models-pytorch

Vi du:
  python train_unet.py --smoke
  python train_unet.py --data memi_processed --all-folds --out runs/unet_r34
  python train_unet.py --data memi_processed --all-folds --out runs/unet_r34 --resume

Mac dinh hop GPU 4GB: batch 2 x accum 2 (= batch hieu dung 4). Het VRAM: --encoder resnet18 hoac
giam --width/--height (nhung khi do khong con so sanh cong bang voi SegFormer).
GTX 10xx khong co tensor core, fp16 co the khong nhanh hon: thu 1 epoch voi va khong co --no-amp.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

import memi_common as C
from train_segformer import SegDS, dice_bce_loss, make_scaler  # dung lai phan da chay on


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="memi_processed")
    p.add_argument("--out", default="runs/unet_r34")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4, help="lr encoder; decoder x10")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def build_model(args, pretrained=True):
    weights = "imagenet" if (pretrained and not args.smoke and not args.no_pretrained) else None
    return smp.Unet(encoder_name=args.encoder, encoder_weights=weights, in_channels=3, classes=1)


@torch.no_grad()
def evaluate(model, loader, device, use_amp, thr):
    model.eval()
    per_img, losses = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            lg = model(x)
        losses.append(dice_bce_loss(lg, y).item())
        pred = (torch.sigmoid(lg.float()) > thr).cpu().numpy()[:, 0]
        gt = y.cpu().numpy()[:, 0]
        per_img += [C.seg_metrics(p, g) for p, g in zip(pred, gt)]
    return float(np.mean(losses)), per_img


def run_fold(args, fold, all_data, device):
    out = Path(args.out) / (f"fold{fold}" if fold >= 0 else "fixed")
    out.mkdir(parents=True, exist_ok=True)
    tr_n, va_n, te_n = C.load_split(Path(args.data) / "split.csv", fold, args.n_folds)
    if args.smoke:
        tr_n, va_n, te_n = tr_n[:6], va_n[:4], te_n[:4]
    print(f"\n=== U-Net {out.name}: train {len(tr_n)} | val {len(va_n)} | test {len(te_n)} anh ===")

    mk = lambda names, train: SegDS([all_data[n] for n in names], train)
    train_ld = DataLoader(mk(tr_n, True), batch_size=args.batch, shuffle=True,
                          drop_last=len(tr_n) > args.batch, num_workers=args.workers,
                          pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_ld = DataLoader(mk(va_n, False), batch_size=2, shuffle=False)
    test_ld = DataLoader(mk(te_n, False), batch_size=2, shuffle=False)

    model = build_model(args).to(device)
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": rest, "lr": args.lr * 10}],
                            weight_decay=0.01)
    steps_per_epoch = max(1, len(train_ld) // args.accum)
    total = args.epochs * steps_per_epoch
    warm = min(2 * steps_per_epoch, max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: C.cosine_warmup(s, warm, total))
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = make_scaler(use_amp)

    cfg = {"arch": "unet", "encoder": args.encoder, "width": args.width, "height": args.height, "thr": args.thr}
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1))

    start, best, bad = 1, -1.0, 0
    last = out / "last.pt"
    if args.resume and last.exists():
        try:
            ck = torch.load(last, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
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
                lg = model(x)
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
        sec = time.time() - t0
        print(f"[unet {out.name}] epoch {epoch:3d}/{args.epochs}  train_loss={run / max(1, n):.4f}  "
              f"val_loss={vloss:.4f}  val_Dice={vd:.4f}  val_IoU={vi:.4f}  ({sec:.0f}s)")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{run / max(1, n):.5f},{vloss:.5f},{vd:.5f},{vi:.5f},"
                    f"{opt.param_groups[0]['lr']:.3e},{sec:.1f}\n")

        if vd > best:
            best, bad = vd, 0
            torch.save(model.state_dict(), out / "best.pt")
            print(f"    -> luu best.pt (val Dice {vd:.4f})")
        else:
            bad += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "bad": bad}, last)
        if bad >= args.patience:
            print(f">>> Early stopping: {args.patience} epoch khong cai thien.")
            break

    # ---------- danh gia test bang mo hinh tot nhat (mot lan) ----------
    best_model = build_model(args, pretrained=False).to(device)
    try:
        sd = torch.load(out / "best.pt", map_location=device, weights_only=True)
    except TypeError:  # PyTorch cu khong co weights_only
        sd = torch.load(out / "best.pt", map_location=device)
    best_model.load_state_dict(sd)
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
                lg = best_model(x)
            pred = (torch.sigmoid(lg.float())[0, 0] > args.thr).cpu().numpy()
            cv2.imwrite(str(ov_dir / f"{name}.jpg"), cv2.cvtColor(C.make_overlay(img, pred, gt), cv2.COLOR_RGB2BGR))
    print(f"\n[unet {out.name}] TEST ({len(timg)} anh): Dice {ts['dice'][0]:.4f} +/- {ts['dice'][1]:.4f} | "
          f"IoU {ts['iou'][0]:.4f} | Precision {ts['precision'][0]:.4f} | Recall {ts['recall'][0]:.4f}")
    (out / "test_summary.json").write_text(json.dumps({k: list(v) for k, v in ts.items()}, indent=1))
    return {"fold": fold, "val_dice": best, **{f"test_{k}": v[0] for k, v in ts.items()}, "n_test": len(timg)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.smoke:
        args.width, args.height, args.epochs, args.workers, args.batch, args.accum = 256, 384, 2, 0, 2, 1
        args.out = "runs/smoke_unet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Thiet bi: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

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
            raise SystemExit("\n[!] Het VRAM. Thu: --encoder resnet18  hoac --batch 1 --accum 4")
        raise

    if len(results) > 1:
        out = Path(args.out)
        with open(out / "cv_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        d = np.array([r["test_dice"] for r in results]); i = np.array([r["test_iou"] for r in results])
        print(f"\n=========== U-NET {len(results)}-FOLD CROSS-VALIDATION ===========")
        print(f"Dice: {d.mean():.4f} +/- {d.std():.4f}   IoU: {i.mean():.4f} +/- {i.std():.4f}")


if __name__ == "__main__":
    main()
