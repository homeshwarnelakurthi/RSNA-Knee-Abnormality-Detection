"""
Why is Medial Meniscus at 0.476 when the literature pools it at 0.925?

Our train-v1 per-target gold AUCs are inverted against the literature: medial 0.476,
lateral 0.704, where Botnari 2024 pools medial 0.925 ABOVE lateral 0.844. Medial is the
easier meniscus everywhere else. The suspicion is our left/right canonicalisation.

A BLIND SWAP TEST WOULD BE WEAK, and it is worth saying why. A *consistent* global flip
does not break learning at all - the model simply learns the mirrored convention and
predicts correctly. Only an INCONSISTENT flip hurts, i.e. laterality assigned wrongly on
some subset, which scrambles medial and lateral against each other.

So the decisive test is side-stratified, with a control:

  right knees -> no flip was applied
  left knees  -> a flip WAS applied

  A laterality bug can ONLY damage side-specific labels. If left-knee studies score worse
  on MCL / Medial Meniscus / Lateral Meniscus / Medial OA / Lateral OA but the SAME on
  ACL / PF OA / Effusion / Synovitis / Baker's / Contusion / Fracture, the flip is the
  cause. If left-knee studies are worse across the board, it is not laterality at all and
  we should stop looking here.

Runs on CPU: ~900 studies x 6 slots through a resnet34 costs no GPU quota.

Outputs predictions.csv so later analyses need no re-run - train-v1 saved none, which is
why this kernel has to exist.
"""
import os, json, time, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import timm

warnings.filterwarnings("ignore")
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# A laterality error can only affect labels that name a side.
SIDE_SPECIFIC = ["MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
SIDE_AGNOSTIC = [l for l in LABELS if l not in SIDE_SPECIFIC]
PAIRS = [("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")]
FOLD, N_FOLDS, SEED = 0, 5, 42


def find(*needles):
    for depth in range(1, 5):
        stack = [("/kaggle/input", 0)]
        while stack:
            p, d = stack.pop()
            if d == depth:
                if all(os.path.exists(os.path.join(p, n)) for n in needles):
                    return p
                continue
            try:
                stack.extend((e.path, d + 1) for e in os.scandir(p) if e.is_dir())
            except OSError:
                pass
    raise SystemExit(f"missing mount for {needles}")


C = find("cache.npy", "index.csv")
A = find("weak_labels_v1.csv", "series_meta.csv")
W_ = find("best.pt")
G_ = find("train.csv")
log(f"cache={C}\nartifacts={A}\nweights={W_}\ncomp={G_}")

arr = np.load(f"{C}/cache.npy", mmap_mode="r")
idx = pd.read_csv(f"{C}/index.csv")
mask = np.load(f"{C}/slot_mask.npy")
N_SLOT, N_SL, IMG = arr.shape[1], arr.shape[2], arr.shape[3]

W = pd.read_csv(f"{A}/weak_labels_v1.csv")
meta = pd.read_csv(f"{A}/series_meta.csv", low_memory=False)
lang = pd.read_csv(f"{A}/report_lang.csv")
lat = pd.read_csv(f"{A}/laterality_sources.csv")


def norm_manu(v):
    s = str(v).upper()
    for k in ["SIEMENS", "PHILIPS", "TOSHIBA", "CANON", "FUJI", "HITACHI"]:
        if k in s:
            return k
    return "GE" if "GE" in s else "OTHER"


sm = (meta[meta.split == "train"].groupby("StudyInstanceUID")
      .agg(manu=("Manufacturer", "first"), model=("ManufacturerModelName", "first")))
sm["manu"] = sm["manu"].map(norm_manu)

df = (idx.merge(W, on="StudyInstanceUID", how="left")
         .merge(lang, on="StudyInstanceUID", how="left")
         .merge(sm, on="StudyInstanceUID", how="left")
         .merge(lat[["StudyInstanceUID", "tag", "geo"]], on="StudyInstanceUID", how="left"))
df["group"] = df["lang"].astype(str) + "|" + df["manu"].astype(str) + "|" + df["model"].astype(str)
# Which source decided this study's side? tag is authoritative, geo is the 92.6-98.9% rule.
df["lat_src"] = np.where(df["tag"].isin(["L", "R"]), "tag", "geo")

SEV = [f"sev::{l}" for l in LABELS]
df[SEV] = df[SEV].fillna(0.05)

G = pd.read_csv(f"{G_}/train.csv")
gold_ids = set(G.loc[G[LABELS].notna().all(axis=1), "StudyInstanceUID"])
df["is_gold"] = df["StudyInstanceUID"].isin(gold_ids)
gold_truth = G[G.StudyInstanceUID.isin(gold_ids)].set_index("StudyInstanceUID")[LABELS]

pool = df[~df["is_gold"]].reset_index(drop=True)
_, va_i = list(GroupKFold(n_splits=N_FOLDS).split(pool, groups=pool["group"]))[FOLD]
val = pool.iloc[va_i].reset_index(drop=True)
gold = df[df.is_gold].reset_index(drop=True)
log(f"val {len(val)}  gold {len(gold)}   side counts val: {val['side'].value_counts().to_dict()}")


class DS(Dataset):
    def __init__(self, sub):
        self.rows = sub["row"].values
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = np.asarray(arr[r], dtype=np.float32) / 255.0
        return torch.from_numpy(x), torch.from_numpy(mask[r].astype(np.float32))


class Net(nn.Module):
    """Must match train_v1.py exactly or the state dict will not load."""
    def __init__(self, name="resnet34", n_slot=N_SLOT, n_sl=N_SL, n_out=len(LABELS)):
        super().__init__()
        self.bb = timm.create_model(name, pretrained=False, in_chans=n_sl, num_classes=0)
        d = self.bb.num_features
        self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.slot_emb = nn.Parameter(torch.zeros(n_slot, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, n_out))
    def forward(self, x, m):
        B, S = x.shape[0], x.shape[1]
        f = self.bb(x.flatten(0, 1)).view(B, S, -1) + self.slot_emb
        a = self.att(f).squeeze(-1).masked_fill(m < 0.5, float("-inf"))
        a = torch.nan_to_num(torch.softmax(a, dim=1))
        return self.head((f * a.unsqueeze(-1)).sum(1))


dev = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(dev)
sd = torch.load(f"{W_}/best.pt", map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
log(f"device {dev} | missing keys {len(missing)} | unexpected {len(unexpected)}")
if missing or unexpected:
    raise SystemExit(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
model.eval()


@torch.no_grad()
def predict(sub):
    dl = DataLoader(DS(sub), batch_size=16, shuffle=False, num_workers=2)
    out = []
    for x, m in dl:
        out.append(torch.sigmoid(model(x.to(dev), m.to(dev)).float()).cpu().numpy())
    return np.concatenate(out)


log("predicting val ...")
Pv = predict(val)
log("predicting gold ...")
Pg = predict(gold)
log("done")

Yv = (val[SEV].values > 0.5).astype(int)
Yg = gold_truth.loc[gold["StudyInstanceUID"]].values.astype(int)


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def table(y, P, rows, title):
    print("\n" + "=" * 76); print(title); print("=" * 76)
    print(f"{'label':<18}{'n':>6}{'n_pos':>7}{'AUC':>9}")
    res = {}
    for j, l in enumerate(LABELS):
        a = auc(y[rows, j], P[rows, j])
        res[l] = a
        print(f"{l:<18}{len(rows):>6}{int(y[rows, j].sum()):>7}{a:>9.3f}" if not np.isnan(a)
              else f"{l:<18}{len(rows):>6}{int(y[rows, j].sum()):>7}{'  n/a':>9}")
    return res


# ---------------------------------------------------------------- THE MAIN TEST
print("\n" + "#" * 76)
print("# TEST 1 (decisive): side-stratified AUC vs weak labels on the fold-0 val set")
print("# Right knees had NO flip applied. Left knees DID.")
print("# A laterality bug can only hurt SIDE-SPECIFIC labels.")
print("#" * 76)
rowsR = np.where(val["side"].values == "R")[0]
rowsL = np.where(val["side"].values == "L")[0]
log(f"val right={len(rowsR)}  left={len(rowsL)}")
rr = table(Yv, Pv, rowsR, "RIGHT knees (no flip)")
ll = table(Yv, Pv, rowsL, "LEFT knees (flipped)")

print("\n" + "-" * 76)
print(f"{'label':<18}{'right':>9}{'left':>9}{'L-R':>9}   group")
print("-" * 76)
d_spec, d_agn = [], []
for l in LABELS:
    if np.isnan(rr[l]) or np.isnan(ll[l]):
        continue
    d = ll[l] - rr[l]
    g = "side-specific" if l in SIDE_SPECIFIC else "side-agnostic"
    (d_spec if l in SIDE_SPECIFIC else d_agn).append(d)
    print(f"{l:<18}{rr[l]:>9.3f}{ll[l]:>9.3f}{d:>+9.3f}   {g}")
print("-" * 76)
ms = float(np.mean(d_spec)) if d_spec else float("nan")
ma = float(np.mean(d_agn)) if d_agn else float("nan")
print(f"mean (left - right), side-specific : {ms:+.4f}")
print(f"mean (left - right), side-agnostic : {ma:+.4f}")
print(f"DIFFERENCE-IN-DIFFERENCES          : {ms - ma:+.4f}")
print("""
Read it this way:
  clearly negative DiD  -> left knees are selectively worse on side-specific labels.
                           The flip is the problem. Fix canonicalisation.
  DiD near zero         -> laterality is NOT the cause, whatever the two means are.
                           Stop looking here; suspect slice sampling or resolution.
""")

# ---------------------------------------------------------------- swap test
print("\n" + "#" * 76)
print("# TEST 2: cross-prediction. Does the LATERAL head predict the MEDIAL label better")
print("#         than the medial head does? A systematic swap would show that.")
print("#" * 76)
for ds_name, y, P in [("val / weak labels", Yv, Pv), ("gold-58 / true labels", Yg, Pg)]:
    print(f"\n--- {ds_name} ---")
    print(f"{'':<22}{'pred=medial':>13}{'pred=lateral':>14}")
    for a_, b_ in PAIRS:
        ia, ib = LABELS.index(a_), LABELS.index(b_)
        for lbl, il in [(a_, ia), (b_, ib)]:
            v1, v2 = auc(y[:, il], P[:, ia]), auc(y[:, il], P[:, ib])
            star = "   <-- swapped wins" if (not np.isnan(v1) and not np.isnan(v2)
                                             and ((il == ia and v2 > v1) or (il == ib and v1 > v2))) else ""
            print(f"label={lbl:<16}{v1:>13.3f}{v2:>14.3f}{star}")
        print()

# ---------------------------------------------------------------- laterality source
print("\n" + "#" * 76)
print("# TEST 3: does it matter whether the side came from the DICOM tag or from geometry?")
print("# The tag is authoritative; geometry is 92.6-98.9% accurate depending on vendor.")
print("#" * 76)
for src in ["tag", "geo"]:
    rows = np.where(val["lat_src"].values == src)[0]
    if len(rows) < 50:
        print(f"\n{src}: only {len(rows)} studies, skipped")
        continue
    r = table(Yv, Pv, rows, f"side from {src.upper()} (n={len(rows)})")
    print(f"  side-specific mean: {np.nanmean([r[l] for l in SIDE_SPECIFIC]):.4f}")
    print(f"  side-agnostic mean: {np.nanmean([r[l] for l in SIDE_AGNOSTIC]):.4f}")

# ---------------------------------------------------------------- persist
out = pd.DataFrame(Pv, columns=[f"pred::{l}" for l in LABELS])
out.insert(0, "StudyInstanceUID", val["StudyInstanceUID"].values)
out["side"] = val["side"].values
out["lat_src"] = val["lat_src"].values
out["split"] = "val"
og = pd.DataFrame(Pg, columns=[f"pred::{l}" for l in LABELS])
og.insert(0, "StudyInstanceUID", gold["StudyInstanceUID"].values)
og["side"] = gold["side"].values
og["lat_src"] = gold["lat_src"].values
og["split"] = "gold"
pd.concat([out, og], ignore_index=True).to_csv("/kaggle/working/predictions.csv", index=False)
log("saved predictions.csv")
log(f"RUNTIME {time.time()-T0:.0f}s")
