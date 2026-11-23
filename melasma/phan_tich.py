import sys, glob, pandas as pd

run = sys.argv[1] if len(sys.argv) > 1 else "runs/unet_roi"
files = sorted(glob.glob(run + "/fold*/per_image_test.csv"))
if not files:
    sys.exit("Khong tim thay per_image_test.csv trong " + run)
df = pd.concat([pd.read_csv(f).assign(fold=f) for f in files])
df["roi"] = df["name"].str.split("_").str[-1]
print("So file:", len(files), "| So ROI:", len(df), "| Ten bi trung:", df["name"].duplicated().sum())
print(df.groupby("roi")[["dice", "recall", "precision", "gt_pct", "pred_pct"]].mean().round(3))
print("ROI gt<1%:", (df.gt_pct < 1).sum(), "/", len(df),
      "| Dice ROI gt>=5%:", round(df[df.gt_pct >= 5].dice.mean(), 3),
      "| Dice tat ca:", round(df.dice.mean(), 4))
