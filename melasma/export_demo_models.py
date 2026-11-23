"""
export_demo_models.py -- Chep cac mo hinh U-Net da huan luyen vao thu muc DermaCare de chay demo.

Chay (trong thu muc du an nghien cuu, noi co 'runs'):
  python melasma/export_demo_models.py --runs runs/unet_r34 --dest DermaCareAI/models/melasma

Moi fold chep 2 tep: best.pt (trong so) va run_config.json (kich thuoc dau vao, encoder, nguong).
Dung runs/unet_r34 (mo hinh GOC). KHONG dung runs/unet_neg (da loai vi hoc 'duong tat').
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/unet_r34")
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    src, dest = Path(a.runs), Path(a.dest)
    folds = sorted(p for p in src.glob("fold*") if (p / "best.pt").exists() and (p / "run_config.json").exists())
    if not folds:
        raise SystemExit(f"Khong thay fold*/best.pt + run_config.json trong {src.resolve()}")
    total = 0
    for f in folds:
        cfg = json.loads((f / "run_config.json").read_text())
        if cfg.get("arch") != "unet":
            raise SystemExit(f"{f} khong phai mo hinh U-Net (arch={cfg.get('arch')}). Chi ho tro U-Net.")
        out = dest / f.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("best.pt", "run_config.json"):
            shutil.copy2(f / name, out / name)
        size = (out / "best.pt").stat().st_size
        total += size
        print(f"  {f.name}: {cfg['encoder']} {cfg['width']}x{cfg['height']}  {size / 1e6:.0f} MB")
    print(f"\nDa chep {len(folds)} mo hinh ({total / 1e6:.0f} MB) vao {dest.resolve()}")
    if len(folds) != 5:
        print(f"[!] Mong doi 5 fold, chi thay {len(folds)}.")


if __name__ == "__main__":
    main()
