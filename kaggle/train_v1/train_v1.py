"""
Phase 1 - first model. Close the loop end to end; do not chase a score.

Design notes that are decisions rather than defaults:

  no horizontal flip   Mirroring swaps medial and lateral, which are DISTINCT labels in
                       5 of the 12 targets. The cache is already canonicalised to one
                       side; flipping now would undo that and corrupt the labels.
  grouped folds        Group on language|manufacturer|model. Random folds inflate AUC by
                       ~0.053 through scanner memorisation (measured, FINDINGS.md 3).
                       Random CV is reported too - the GAP measures how much the model
                       leans on site rather than anatomy.
  soft targets         Severity in [0,1] from the report labeler, not binary presence.
                       AUC is rank-based, so a model that learns the severity continuum
                       ranks correctly under any threshold - which is the point, since
                       ground truth uses stricter thresholds than the reports do.
  confidence weights   Per-sample loss weight. Low where the report was short or hedged.
                       Uncertainty belongs here, NOT in the target value: inflating a
                       target to express doubt corrupts the ranking (FINDINGS.md 9).
  gold held out        The 58 gold studies never enter training. They are the only truth
                       we have, so they are worth more as an honest eval than as 58
                       extra weak-labelled rows.
"""
import os, json, time, math, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
import timm

warnings.filterwarnings("ignore")
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

BACKBONE = "resnet34"
EPOCHS = 12
BATCH = 16          # studies; each carries 6 slot images
LR_HEAD, LR_BB = 1e-3, 1e-4
WD = 1e-2
FOLD = 0
N_FOLDS = 5
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


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
log(f"cache {C}\nartifacts {A}")

cfg = json.load(open(f"{C}/cache_config.json"))
arr = np.load(f"{C}/cache.npy", mmap_mode="r")
idx = pd.read_csv(f"{C}/index.csv")
mask = np.load(f"{C}/slot_mask.npy")
N_SLOT, N_SL, IMG = arr.shape[1], arr.shape[2], arr.shape[3]
log(f"cache {arr.shape}  slots={cfg['slots']}")

W = pd.read_csv(f"{A}/weak_labels_v1.csv")
meta = pd.read_csv(f"{A}/series_meta.csv", low_memory=False)
lang = pd.read_csv(f"{A}/report_lang.csv")

# ---- fold groups: language | manufacturer | model  (FINDINGS.md 3)
def norm_manu(v):
    s = str(v).upper()
    for k in ["SIEMENS", "PHILIPS", "TOSHIBA", "CANON", "FUJI", "HITACHI"]:
        if k in s:
            return k
    return "GE" if "GE" in s else "OTHER"


sm = (meta[meta.split == "train"]
      .groupby("StudyInstanceUID")
      .agg(manu=("Manufacturer", "first"), model=("ManufacturerModelName", "first")))
sm["manu"] = sm["manu"].map(norm_manu)
df = idx.merge(W, on="StudyInstanceUID", how="left") \
        .merge(lang, on="StudyInstanceUID", how="left") \
        .merge(sm, on="StudyInstanceUID", how="left")
df["group"] = (df["lang"].astype(str) + "|" + df["manu"].astype(str) + "|"
               + df["model"].astype(str))
log(f"rows {len(df)}   groups {df['group'].nunique()}")

SEV = [f"sev::{l}" for l in LABELS]
CONF = [f"conf::{l}" for l in LABELS]
df[SEV] = df[SEV].fillna(0.05)
df[CONF] = df[CONF].fillna(0.3)

# ---- gold studies: held out of training entirely
gold = find("train.csv")
G = pd.read_csv(f"{gold}/train.csv")
gold_ids = set(G.loc[G[LABELS].notna().all(axis=1), "StudyInstanceUID"])
df["is_gold"] = df["StudyInstanceUID"].isin(gold_ids)
gold_truth = G[G.StudyInstanceUID.isin(gold_ids)].set_index("StudyInstanceUID")[LABELS]
log(f"gold studies held out: {df['is_gold'].sum()}")

pool = df[~df["is_gold"]].reset_index(drop=True)
gk = GroupKFold(n_splits=N_FOLDS)
tr_i, va_i = list(gk.split(pool, groups=pool["group"]))[FOLD]
rk = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
_, va_rand = list(rk.split(pool))[FOLD]
log(f"grouped fold {FOLD}: train {len(tr_i)}  val {len(va_i)}  "
    f"({pool.iloc[va_i]['group'].nunique()} held-out groups)")


class KneeDS(Dataset):
    def __init__(self, sub, train):
        self.rows = sub["row"].values
        self.y = sub[SEV].values.astype(np.float32)
        self.w = sub[CONF].values.astype(np.float32)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = np.asarray(arr[r], dtype=np.float32) / 255.0     # (slots, slices, H, W)
        m = mask[r].astype(np.float32)
        if self.train:
            # Rigid jitter only. No flips - see module docstring.
            if np.random.rand() < 0.7:
                sh = np.random.randint(-12, 13, size=2)
                x = np.roll(x, tuple(sh), axis=(2, 3))
            x = np.clip(x * np.random.uniform(0.9, 1.1) +
                        np.random.uniform(-0.05, 0.05), 0, 1)
            # slot dropout: teaches the model to survive a missing sequence
            if np.random.rand() < 0.25:
                d = np.random.randint(N_SLOT)
                if m.sum() > 1:
                    x[d] = 0; m[d] = 0
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(m),
                torch.from_numpy(self.y[i]), torch.from_numpy(self.w[i]))


class Net(nn.Module):
    """Per-slot 2D encoder over slices-as-channels, then masked attention over slots."""

    def __init__(self, name=BACKBONE, n_slot=N_SLOT, n_sl=N_SL, n_out=len(LABELS)):
        super().__init__()
        self.bb = timm.create_model(name, pretrained=True, in_chans=n_sl, num_classes=0)
        d = self.bb.num_features
        self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.slot_emb = nn.Parameter(torch.zeros(n_slot, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, n_out))

    def forward(self, x, m):
        B, S = x.shape[0], x.shape[1]
        f = self.bb(x.flatten(0, 1)).view(B, S, -1) + self.slot_emb
        a = self.att(f).squeeze(-1)
        a = a.masked_fill(m < 0.5, float("-inf"))
        a = torch.softmax(a, dim=1)
        a = torch.nan_to_num(a)                       # a study with zero slots
        return self.head((f * a.unsqueeze(-1)).sum(1))


dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device {dev}  {torch.cuda.get_device_name(0) if dev=='cuda' else ''}")
model = Net().to(dev)
head_p = [p for n, p in model.named_parameters() if not n.startswith("bb.")]
bb_p = [p for n, p in model.named_parameters() if n.startswith("bb.")]
opt = torch.optim.AdamW([{"params": bb_p, "lr": LR_BB},
                         {"params": head_p, "lr": LR_HEAD}], weight_decay=WD)

dl_tr = DataLoader(KneeDS(pool.iloc[tr_i], True), batch_size=BATCH, shuffle=True,
                   num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)
dl_va = DataLoader(KneeDS(pool.iloc[va_i], False), batch_size=BATCH, shuffle=False,
                   num_workers=2, pin_memory=True)
dl_rand = DataLoader(KneeDS(pool.iloc[va_rand], False), batch_size=BATCH, shuffle=False,
                     num_workers=2)
dl_gold = DataLoader(KneeDS(df[df.is_gold].reset_index(drop=True), False),
                     batch_size=BATCH, shuffle=False, num_workers=2)

sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[LR_BB, LR_HEAD], total_steps=EPOCHS * len(dl_tr), pct_start=0.25)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")


@torch.no_grad()
def predict(dl):
    model.eval()
    P, Y = [], []
    for x, m, y, w in dl:
        with torch.amp.autocast("cuda", enabled=dev == "cuda"):
            p = model(x.to(dev, non_blocking=True), m.to(dev))
        P.append(torch.sigmoid(p.float()).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def macro_auc(y, p, thr=0.5):
    aucs = []
    for j in range(len(LABELS)):
        t = (y[:, j] > thr).astype(int)
        if 0 < t.sum() < len(t):
            aucs.append(roc_auc_score(t, p[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan"), aucs


best = -1
for ep in range(EPOCHS):
    model.train(); tot = n = 0
    for x, m, y, w in dl_tr:
        x, m, y, w = x.to(dev, non_blocking=True), m.to(dev), y.to(dev), w.to(dev)
        with torch.amp.autocast("cuda", enabled=dev == "cuda"):
            out = model(x, m)
            loss = (F.binary_cross_entropy_with_logits(out, y, reduction="none") * w).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(x); n += len(x)

    pv, yv = predict(dl_va)
    a_grp, _ = macro_auc(yv, pv)
    pr, yr = predict(dl_rand)
    a_rnd, _ = macro_auc(yr, pr)
    pg, _ = predict(dl_gold)
    gt = gold_truth.loc[df[df.is_gold]["StudyInstanceUID"]].values
    a_gold, per_gold = macro_auc(gt.astype(float), pg, thr=0.5)
    log(f"ep {ep+1:2d}  loss {tot/n:.4f}  grouped {a_grp:.4f}  random {a_rnd:.4f}  "
        f"gap {a_rnd-a_grp:+.4f}  GOLD {a_gold:.4f}")
    if a_gold > best:
        best = a_gold
        torch.save(model.state_dict(), "/kaggle/working/best.pt")

log(f"\nbest gold macro AUC {best:.4f}")

pg, _ = predict(dl_gold)
gt = gold_truth.loc[df[df.is_gold]["StudyInstanceUID"]].values.astype(float)
print("\nper-label AUC on the 58 held-out gold studies:")
for j, l in enumerate(LABELS):
    t = (gt[:, j] > 0.5).astype(int)
    if 0 < t.sum() < len(t):
        print(f"  {l:<18} n_pos {int(t.sum()):>3}   AUC {roc_auc_score(t, pg[:, j]):.3f}")

json.dump({"backbone": BACKBONE, "epochs": EPOCHS, "fold": FOLD,
           "best_gold_auc": best, "runtime_s": time.time() - T0},
          open("/kaggle/working/run.json", "w"), indent=2)
log(f"RUNTIME {time.time()-T0:.0f}s  (log this - it is half the Efficiency metric)")
