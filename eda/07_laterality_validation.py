"""
Close decision 1 from FINDINGS.md: is the geometry sign rule safe on the vendors that
carry no Laterality tag?

The DICOM tag validated the rule at 98.5% - but almost entirely on Siemens and Philips.
GE MEDICAL SYSTEMS (4,914 series) carries no tag at all. Report-derived side is a third,
independent source available for every training study, including every GE one.

Three questions:
  1. Where the tag exists, is report-derived side correct? (validates the text extractor)
  2. For studies with NO tag, does the report supply the side? (coverage)
  3. Does the geometry sign rule agree with report-side, per vendor? (the actual answer)
"""
import numpy as np, pandas as pd

M = pd.read_csv(r"H:\RSNA Knee Abnormality Detection\kaggle\out\series_meta.csv", low_memory=False)
M = M[M.split == "train"].copy()
W = pd.read_csv(r"H:\RSNA Knee Abnormality Detection\artifacts\weak_labels_v1.csv")


def clean_lat(v):
    if not isinstance(v, str):
        return None
    s = v.strip().upper()
    if s in ("", "NONE", "NAN"):
        return None
    return "L" if s.startswith("L") else "R" if s.startswith("R") else None


def center_x(r):
    try:
        iop = [float(v) for v in str(r["IOP"]).split(",")]
        ipp = [float(v) for v in str(r["IPP"]).split(",")]
        ps, rows, cols = float(r["PixelSpacing"]), float(r["Rows"]), float(r["Columns"])
        return ipp[0] + (cols / 2) * ps * iop[0] + (rows / 2) * ps * iop[3]
    except Exception:
        return np.nan


M["lat"] = M["Laterality"].map(clean_lat)
M["cx"] = M.apply(center_x, axis=1)


def norm_manu(v):
    s = str(v).upper()
    for k in ["SIEMENS", "PHILIPS", "TOSHIBA", "CANON", "FUJI", "HITACHI"]:
        if k in s:
            return k
    return "GE" if "GE" in s else "OTHER"


M["manu"] = M["Manufacturer"].map(norm_manu)

# Study level: majority tag, median centre-x, vendor.
S = M.groupby("StudyInstanceUID").agg(
    manu=("manu", "first"),
    tag=("lat", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
    cx=("cx", "median"),
    n_series=("lat", "size"))
S = S.join(W.set_index("StudyInstanceUID")[["side_from_report"]])
S["rep"] = S["side_from_report"].where(S["side_from_report"].isin(["L", "R"]))
S["geo"] = np.where(S["cx"] > 0, "L", "R")
S.loc[S["cx"].isna(), "geo"] = None

print("=" * 80)
print("SOURCE AVAILABILITY  (4,407 studies)")
print("=" * 80)
print(f"  DICOM tag          {S['tag'].notna().mean():6.1%}")
print(f"  report-derived     {S['rep'].notna().mean():6.1%}")
print(f"  geometry           {S['geo'].notna().mean():6.1%}")
print(f"  tag OR report      {(S['tag'].notna() | S['rep'].notna()).mean():6.1%}")
print(f"  NEITHER tag nor report: {((S['tag'].isna()) & (S['rep'].isna())).sum()} studies")

print("\n" + "=" * 80)
print("Q1  Where the tag exists, is the REPORT extractor right?")
print("=" * 80)
both = S[S["tag"].notna() & S["rep"].notna()]
print(f"  overlap: {len(both)} studies")
if len(both):
    print(f"  agreement: {(both['tag'] == both['rep']).mean():6.1%}")
    print("\n  confusion (rows tag, cols report):")
    print(pd.crosstab(both["tag"], both["rep"]).to_string())
    print("\n  by vendor:")
    t = both.assign(ok=both["tag"] == both["rep"]).groupby("manu")["ok"].agg(["size", "mean"])
    print(t.assign(mean=(t["mean"] * 100).round(1)).sort_values("size", ascending=False).to_string())

print("\n" + "=" * 80)
print("Q2  For studies with NO tag, does the report supply the side?")
print("=" * 80)
notag = S[S["tag"].isna()]
print(f"  studies with no tag: {len(notag)}")
print(f"  of those, report gives a side: {notag['rep'].notna().mean():6.1%}")
print("\n  by vendor:")
t = notag.assign(ok=notag["rep"].notna()).groupby("manu")["ok"].agg(["size", "mean"])
print(t.assign(mean=(t["mean"] * 100).round(1)).sort_values("size", ascending=False).to_string())

print("\n" + "=" * 80)
print("Q3  *** Does GEOMETRY agree with the REPORT, per vendor? ***")
print("=" * 80)
print("This is the question the DICOM tag could not answer for GE/Toshiba/Canon.\n")
g = S[S["rep"].notna() & S["geo"].notna()].copy()
g["ok"] = g["geo"] == g["rep"]
t = g.groupby("manu")["ok"].agg(["size", "mean"])
t["mean"] = (t["mean"] * 100).round(1)
print(t.sort_values("size", ascending=False).to_string())
print(f"\n  overall: {g['ok'].mean():.1%} over {len(g)} studies")
print("\n  ~100% -> sign convention holds, ship the geometry rule for that vendor")
print("  ~0%   -> convention is INVERTED, flip the sign for that vendor")
print("  ~50%  -> unusable, fall back to another source")

# how confident can we be per vendor, given sample size
print("\n  Wilson 95% interval per vendor:")
for m, r in t.iterrows():
    n, p = int(r["size"]), r["mean"] / 100
    if n == 0:
        continue
    z = 1.96
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    print(f"    {m:<9} n={n:<5} {p:6.1%}  [{max(0,c-h):.1%}, {min(1,c+h):.1%}]")

print("\n" + "=" * 80)
print("COVERAGE OF THE FINAL RULE  (what runs at test time)")
print("=" * 80)
print("At test time only geometry exists - no report, and the tag is missing half the")
print("time. So the shipped rule is: use the tag when present, else calibrated geometry.")
avail = S["tag"].notna() | S["geo"].notna()
print(f"\n  studies resolvable by tag-or-geometry: {avail.mean():.1%}")
disagree = S[S["tag"].notna() & S["geo"].notna()]
print(f"  tag vs geometry agreement where both exist: "
      f"{(disagree['tag'] == disagree['geo']).mean():.1%} over {len(disagree)} studies")

out = S.reset_index()[["StudyInstanceUID", "manu", "tag", "rep", "geo", "cx"]]
out.to_csv(r"H:\RSNA Knee Abnormality Detection\artifacts\laterality_sources.csv", index=False)
print("\n[saved artifacts/laterality_sources.csv]")
