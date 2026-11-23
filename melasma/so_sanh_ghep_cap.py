import sys, numpy as np, pandas as pd
from pathlib import Path
base = Path(sys.argv[1])
df = pd.read_csv("hg1_fullmask_per_image.csv")
files = sorted(base.glob("fold*/per_image_test.csv"))
print("tim thay", len(files), "file baseline")
b = pd.concat([pd.read_csv(f) for f in files])[["name", "dice"]].rename(columns={"dice": "dice_base"})
sp = pd.read_csv("data/memi_roi/split.csv"); sp["img"] = sp["name"].str.split("__").str[0]
grp = sp.drop_duplicates("img").set_index("img")["group"]
m = df.merge(b, on="name"); m["group"] = m["name"].map(grp); m["d"] = m["dice"] - m["dice_base"]
print("so anh ghep cap:", len(m), "| Huong 1 %.4f | baseline %.4f" % (m.dice.mean(), m.dice_base.mean()))
print("chenh lech (Huong 1 - baseline): trung binh %.4f | Huong 1 cao hon o %d / %d anh" % (m.d.mean(), (m.d > 0).sum(), len(m)))
rng = np.random.default_rng(42); g = {k: v.d.values for k, v in m.groupby("group")}; keys = list(g)
bs = [np.concatenate([g[k] for k in rng.choice(keys, len(keys))]).mean() for _ in range(5000)]
print("khoang tin cay 95%% (bootstrap theo benh nhan) cua chenh lech: [%.3f, %.3f]" % tuple(np.percentile(bs, [2.5, 97.5])))
