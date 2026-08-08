"""
Follow-up on Job A. Three things the in-kernel report got wrong or left open:

  1. Laterality missingness was undercounted - blank strings are missing too.
  2. Can slice geometry recover laterality where the tag is absent? Validate the
     rule against the series that DO carry the tag.
  3. The 3262 scanner fingerprints are unusable for grouped folds (2668 are
     singletons). Find a grouping that actually yields folds.
"""
import numpy as np, pandas as pd, re

M = pd.read_csv(r"H:\RSNA Knee Abnormality Detection\kaggle\out\series_meta.csv", low_memory=False)
M = M[M.split == "train"].copy()
lang = pd.read_csv(r"H:\RSNA Knee Abnormality Detection\eda\report_lang.csv")
M = M.merge(lang, on="StudyInstanceUID", how="left")
print("series:", len(M), " studies:", M.StudyInstanceUID.nunique())

# ------------------------------------------------------------------ 1. laterality
def clean_lat(v):
    if not isinstance(v, str):
        return None
    s = v.strip().upper()
    if s in ("", "NONE", "NAN"):
        return None
    if s.startswith("L"):
        return "L"
    if s.startswith("R"):
        return "R"
    if s == "B":
        return "B"
    return None

M["lat"] = M["Laterality"].map(clean_lat)
print("\n" + "=" * 74)
print("A2 (corrected)  LATERALITY MISSINGNESS")
print("=" * 74)
print(M["lat"].value_counts(dropna=False).to_string())
print(f"\nSeries missing laterality: {M['lat'].isna().sum()} ({M['lat'].isna().mean():.1%})")
print("  (the in-kernel report said 20.9% - it counted only NaN, not blank strings)")

g = M.groupby("StudyInstanceUID")["lat"]
has_any = g.apply(lambda s: s.notna().any())
n_uniq = g.nunique(dropna=True)
print(f"\nStudies: {len(has_any)}")
print(f"  with >=1 series carrying laterality : {has_any.sum()} ({has_any.mean():.1%})")
print(f"  with NO laterality anywhere         : {(~has_any).sum()} ({(~has_any).mean():.1%})  <- need geometry")
print("\nDistinct laterality values per study:")
print(n_uniq.value_counts().sort_index().to_string())
print(f"\nBilateral (>=2 distinct sides): {(n_uniq >= 2).sum()} ({(n_uniq>=2).mean():.2%})")

print("\nLaterality presence by report language:")
sl = M.groupby("StudyInstanceUID").agg(lang=("lang", "first"))
sl["has_lat"] = has_any
t = sl.groupby("lang")["has_lat"].agg(["size", "mean"])
print(t.assign(mean=(t["mean"] * 100).round(1)).sort_values("size", ascending=False).to_string())

# ------------------------------------------------------------------ 2. geometry fallback
print("\n" + "=" * 74)
print("CAN GEOMETRY RECOVER THE SIDE?  (validated on series that carry the tag)")
print("=" * 74)

def center_x(r):
    """x of the image centre in patient coordinates (DICOM LPS: +x = patient Left)."""
    try:
        iop = [float(v) for v in str(r["IOP"]).split(",")]
        ipp = [float(v) for v in str(r["IPP"]).split(",")]
        ps = float(r["PixelSpacing"]); rows = float(r["Rows"]); cols = float(r["Columns"])
        if len(iop) != 6 or len(ipp) != 3:
            return np.nan
        return ipp[0] + (cols / 2.0) * ps * iop[0] + (rows / 2.0) * ps * iop[3]
    except Exception:
        return np.nan

M["cx"] = M.apply(center_x, axis=1)
print(f"centre-x computable for {M['cx'].notna().mean():.1%} of series")

known = M[M["lat"].isin(["L", "R"]) & M["cx"].notna()]
print(f"\nvalidation set: {len(known)} series with both tag and geometry")
print("\ncentre-x distribution by tagged side:")
print(known.groupby("lat")["cx"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_string())

for thr in [0, 5, 10, 20, 30, 40]:
    d = known[known["cx"].abs() >= thr]
    if not len(d):
        continue
    pred = np.where(d["cx"] > 0, "L", "R")
    acc = (pred == d["lat"]).mean()
    print(f"  |x| >= {thr:2d} mm : covers {len(d)/len(known):6.1%} of series, "
          f"sign rule accuracy {acc:6.1%}")

# does the sign convention hold per manufacturer?
print("\nsign-rule accuracy by manufacturer (|x| >= 20 mm):")
d = known[known["cx"].abs() >= 20].copy()
d["pred"] = np.where(d["cx"] > 0, "L", "R")
d["ok"] = d["pred"] == d["lat"]
mm = d.groupby("Manufacturer")["ok"].agg(["size", "mean"])
print(mm.assign(mean=(mm["mean"] * 100).round(1)).sort_values("size", ascending=False).to_string())
print("\n-> if accuracy is near 50% the convention is inverted or unusable;")
print("   near 100% it is a reliable fallback for the studies with no tag.")

# ------------------------------------------------------------------ 3. grouping
print("\n" + "=" * 74)
print("A4 (revisited)  A GROUPING THAT ACTUALLY YIELDS FOLDS")
print("=" * 74)

def norm_manu(v):
    s = str(v).upper()
    for k in ["SIEMENS", "PHILIPS", "GE", "TOSHIBA", "CANON", "FUJI", "HITACHI"]:
        if k in s:
            return "GE" if k == "GE" else k
    return "OTHER"

S = M.groupby("StudyInstanceUID").agg(
    manu=("Manufacturer", "first"), model=("ManufacturerModelName", "first"),
    sw=("SoftwareVersions", "first"), coil=("ReceiveCoilName", "first"),
    field=("MagneticFieldStrength", "first"), lang=("lang", "first"))
S["manu_n"] = S["manu"].map(norm_manu)

schemes = {
    "manufacturer": S["manu_n"],
    "manu|model": S["manu_n"] + "|" + S["model"].astype(str),
    "manu|model|sw": S["manu_n"] + "|" + S["model"].astype(str) + "|" + S["sw"].astype(str),
    "language": S["lang"].astype(str),
    "lang|manu": S["lang"].astype(str) + "|" + S["manu_n"],
    "lang|manu|model": S["lang"].astype(str) + "|" + S["manu_n"] + "|" + S["model"].astype(str),
    "lang|manu|model|coil": (S["lang"].astype(str) + "|" + S["manu_n"] + "|" +
                             S["model"].astype(str) + "|" + S["coil"].astype(str)),
}
print(f"\n{'scheme':<24} {'groups':>7} {'largest':>9} {'singletons':>11} {'>=50 studies':>13}")
print("-" * 70)
for name, key in schemes.items():
    vc = key.value_counts()
    print(f"{name:<24} {len(vc):>7} {vc.iloc[0]/len(S):>8.1%} "
          f"{(vc==1).sum():>11} {(vc>=50).sum():>13}")

print("\nA usable scheme needs: enough groups to build 5 folds, no group so large")
print("it dominates a fold, and few singletons. Detail of the two best candidates:")
for name in ["manu|model", "lang|manu"]:
    vc = schemes[name].value_counts()
    print(f"\n--- {name}: {len(vc)} groups ---")
    print(vc.head(12).to_string())
    print(f"top-5 share {vc.head(5).sum()/len(S):.1%}; "
          f"studies in singleton groups {(vc[vc==1].sum())}")

print("\nlanguage x manufacturer (studies) - is language a site proxy?")
print(pd.crosstab(S["lang"], S["manu_n"]).to_string())

# ------------------------------------------------------------------ extras
print("\n" + "=" * 74)
print("SERIES DESCRIPTION USABILITY")
print("=" * 74)
sd = M["SeriesDescription"].fillna("")
dummy = sd.str.contains("DummySeriesDesc", case=False)
print(f"null: {M['SeriesDescription'].isna().mean():.1%}   "
      f"placeholder 'DummySeriesDesc!': {dummy.mean():.1%}   "
      f"usable: {(~dummy & M['SeriesDescription'].notna()).mean():.1%}")
ds = M.groupby("StudyInstanceUID").apply(lambda d: dummy.loc[d.index].all(), include_groups=False)
print(f"studies where EVERY series is a placeholder: {ds.sum()} ({ds.mean():.1%})")
print("\nplaceholder rate by manufacturer:")
pm = M.assign(d=dummy).groupby("Manufacturer")["d"].agg(["size", "mean"])
print(pm.assign(mean=(pm["mean"] * 100).round(1)).sort_values("size", ascending=False).to_string())
print("\n-> a placeholder concentrated in one vendor is a site marker, not a description.")
