"""
B4 - the direct test of the thesis.

Score the presence extractor (control) and the severity extractor (thesis) against the
58 gold-labelled studies.

Read this as a DIRECTION check, not a scoreboard. n=58, the gold set is enriched, and
every one of the 58 carries at least one positive - it is not a random sample. It cannot
resolve differences below roughly 0.02-0.03 on a single label. What it CAN show is
whether most of the twelve labels move the same way.
"""
import sys, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from report_labeler import label_report, extract_side, LABELS, MAGNITUDE_LABELS

D = r"H:\RSNA Knee Abnormality Detection\data"
tr = pd.read_csv(f"{D}/train.csv")
lang = pd.read_csv(r"H:\RSNA Knee Abnormality Detection\eda\report_lang.csv")
tr = tr.merge(lang, on="StudyInstanceUID", how="left")

print(f"scoring {len(tr)} reports ...", flush=True)
res = [label_report(t) for t in tr["Report"].astype(str)]
tr["side_from_report"] = [extract_side(t) for t in tr["Report"].astype(str)]

pres = pd.DataFrame([{l: r[l]["presence"] for l in LABELS} for r in res])
sev = pd.DataFrame([{l: r[l]["severity"] for l in LABELS} for r in res])
conf = pd.DataFrame([{l: r[l]["confidence"] for l in LABELS} for r in res])
ment = pd.DataFrame([{l: r[l]["mentioned"] for l in LABELS} for r in res])
for df in (pres, sev, conf, ment):
    df.index = tr.index

gold_mask = tr[LABELS].notna().all(axis=1)
G = tr[gold_mask]
print(f"gold studies: {len(G)}")

# ------------------------------------------------------------------ head to head
print("\n" + "=" * 86)
print("B4  PRESENCE vs SEVERITY on the 58 gold studies")
print("=" * 86)
rows = []
for l in LABELS:
    y = G[l].astype(int).values
    p = pres.loc[G.index, l].values.astype(float)
    s = sev.loc[G.index, l].values.astype(float)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        rows.append({"label": l, "n_pos": n_pos, "auc_presence": np.nan,
                     "auc_severity": np.nan, "delta": np.nan,
                     "acc_presence": np.nan, "group": ""})
        continue
    a_p = roc_auc_score(y, p)
    a_s = roc_auc_score(y, s)
    rows.append({
        "label": l, "n_pos": n_pos,
        "auc_presence": round(a_p, 3), "auc_severity": round(a_s, 3),
        "delta": round(a_s - a_p, 3),
        "acc_presence": round((p == y).mean(), 3),
        "group": "magnitude" if l in MAGNITUDE_LABELS else "categorical",
    })
R = pd.DataFrame(rows)
print("\n" + R.to_string(index=False))

mp, ms = R["auc_presence"].mean(), R["auc_severity"].mean()
print(f"\nMACRO AUC   presence {mp:.4f}   severity {ms:.4f}   delta {ms-mp:+.4f}")
win = (R["delta"] > 0).sum(); lose = (R["delta"] < 0).sum(); tie = (R["delta"] == 0).sum()
print(f"labels improved by severity: {win}   worsened: {lose}   unchanged: {tie}")

print("\nBy rubric family (the split predicted in FINDINGS.md section 8):")
for gname in ["magnitude", "categorical"]:
    sub = R[R.group == gname]
    if len(sub):
        print(f"  {gname:<12} n={len(sub)}  presence {sub['auc_presence'].mean():.4f}  "
              f"severity {sub['auc_severity'].mean():.4f}  "
              f"delta {sub['auc_severity'].mean()-sub['auc_presence'].mean():+.4f}  "
              f"({(sub['delta']>0).sum()}/{len(sub)} improved)")

# ------------------------------------------------------------------ why it moves
print("\n" + "=" * 86)
print("WHY: ties. AUC cannot rank inside a tied block, and binary output is all ties.")
print("=" * 86)
for l in LABELS[:6]:
    p = pres.loc[G.index, l]; s = sev.loc[G.index, l]
    print(f"  {l:<18} presence distinct values: {p.nunique():>2}   "
          f"severity distinct values: {s.nunique():>2}")

# ------------------------------------------------------------------ bootstrap
print("\n" + "=" * 86)
print("HOW MUCH OF THIS IS NOISE?  bootstrap over the 58 studies, 2000 resamples")
print("=" * 86)
rng = np.random.default_rng(0)
idx = np.arange(len(G))
deltas = []
for _ in range(2000):
    b = rng.choice(idx, len(idx), replace=True)
    ds = []
    for l in LABELS:
        y = G[l].astype(int).values[b]
        if y.sum() in (0, len(y)):
            continue
        ds.append(roc_auc_score(y, sev.loc[G.index, l].values[b]) -
                  roc_auc_score(y, pres.loc[G.index, l].values[b]))
    if ds:
        deltas.append(np.mean(ds))
deltas = np.array(deltas)
lo, hi = np.percentile(deltas, [2.5, 97.5])
print(f"\nmacro delta (severity - presence): {deltas.mean():+.4f}")
print(f"95% CI: [{lo:+.4f}, {hi:+.4f}]   P(delta > 0) = {(deltas>0).mean():.1%}")
print("\nIf the CI straddles zero the effect is unproven at n=58 - which is expected.")
print("The direction and the win/loss count carry more information than the interval.")

# ------------------------------------------------------------------ coverage
print("\n" + "=" * 86)
print("EXTRACTOR COVERAGE OVER ALL 4407 STUDIES")
print("=" * 86)
cov = pd.DataFrame({
    "mentioned_pct": (ment.mean() * 100).round(1),
    "presence_pos_pct": (pres.mean() * 100).round(1),
    "sev_mean": sev.mean().round(3),
    "sev_gt_0.5_pct": ((sev > 0.5).mean() * 100).round(1),
    "mean_conf": conf.mean().round(3),
})
gold_prev = (G[LABELS].mean() * 100).round(1)
cov["gold_prev_pct"] = gold_prev
print("\n" + cov.to_string())
print("\nNote: gold prevalence is from an enriched 58-study sample, so it should sit")
print("ABOVE the corpus rate. Where the extractor exceeds it, suspect over-calling.")

# ------------------------------------------------------------------ laterality
print("\n" + "=" * 86)
print("SIDE FROM REPORT  (closes the GE blind spot - FINDINGS.md section 1)")
print("=" * 86)
sv = tr["side_from_report"].value_counts(dropna=False)
print(sv.to_string())
print(f"\nresolved to a single side: {tr['side_from_report'].isin(['L','R']).mean():.1%}")
print("by language:")
t = tr.assign(ok=tr["side_from_report"].isin(["L", "R"])).groupby("lang")["ok"].agg(["size", "mean"])
print(t.assign(mean=(t["mean"] * 100).round(1)).sort_values("size", ascending=False).to_string())

# ------------------------------------------------------------------ save
out = tr[["StudyInstanceUID"]].copy()
out["side_from_report"] = tr["side_from_report"]
for l in LABELS:
    out[f"sev::{l}"] = sev[l]
    out[f"pres::{l}"] = pres[l]
    out[f"conf::{l}"] = conf[l]
dst = pathlib.Path(r"H:\RSNA Knee Abnormality Detection\artifacts")
dst.mkdir(exist_ok=True)
out.to_csv(dst / "weak_labels_v1.csv", index=False)
R.to_csv(dst / "b4_eval_v1.csv", index=False)
print(f"\n[saved artifacts/weak_labels_v1.csv  {out.shape}]")
print("[saved artifacts/b4_eval_v1.csv]")
