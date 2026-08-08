"""
Job A — DICOM metadata scan (Phase 0 research).

Header-only pass over the competition DICOMs. No pixel data is decoded, so this runs
on a CPU session and costs zero GPU quota.

Answers RESEARCH_AGENDA.md A1-A8:
  A1 bilateral studies      A2 Laterality completeness   A3 PatientID repeats
  A4 site/scanner groups    A5 physical scale            A6 slices per slot
  A7 transfer syntaxes      A8 SeriesDescription vs CSV columns

Outputs /kaggle/working/series_meta.csv (one row per series) and prints a full report.
"""
import os, sys, time, warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings("ignore")
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def find_root():
    """Locate the competition mount rather than assuming its path.

    A hardcoded path that is merely absent fails silently three screens later, as the
    first version of this script did: every series scan returned nothing and the error
    surfaced as a missing DataFrame column.
    """
    base = "/kaggle/input"
    if not os.path.isdir(base):
        raise SystemExit(f"{base} does not exist - is this running on Kaggle?")
    log(f"/kaggle/input contains: {sorted(os.listdir(base))}")

    # Depth is not fixed: a plain competition slug mounts at
    # /kaggle/input/<slug>, while a "competitions/<slug>" source mounts one level
    # deeper at /kaggle/input/competitions/<slug>. Search rather than assume.
    seen = []
    for depth in range(1, 4):
        stack = [(base, 0)]
        while stack:
            path, d = stack.pop()
            if d == depth:
                seen.append(path)
                if os.path.isdir(os.path.join(path, "train_series")):
                    log(f"competition mount found (depth {depth}): {path}")
                    return path
                continue
            try:
                for e in os.scandir(path):
                    if e.is_dir():
                        stack.append((e.path, d + 1))
            except OSError:
                pass

    for p in seen[:40]:
        try:
            log(f"  {p} -> {sorted(os.listdir(p))[:12]}")
        except OSError:
            pass
    raise SystemExit("no directory containing train_series/ found under /kaggle/input")


ROOT = find_root()


# Tags worth pulling. Read once per series from a middle slice.
TAGS = [
    "PatientID", "PatientSex", "Laterality", "SeriesDescription", "BodyPartExamined",
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "MagneticFieldStrength",
    "ImagingFrequency", "ReceiveCoilName", "Rows", "Columns", "SliceThickness",
    "SpacingBetweenSlices", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
    "EchoTrainLength", "PixelBandwidth", "ScanningSequence", "SequenceVariant",
    "ScanOptions", "MRAcquisitionType", "PatientPosition", "SeriesNumber", "ImageType",
]


def scan_split(split):
    """Walk <split>_series/ and read one header per series."""
    base = os.path.join(ROOT, f"{split}_series")
    if not os.path.isdir(base):
        log(f"{base} missing - skipped")
        return []

    log(f"scanning {split}_series/ ...")
    studies = sorted(e.name for e in os.scandir(base) if e.is_dir())
    log(f"  {len(studies)} studies")

    rows, n_files_total, errors = [], 0, 0
    for si, study in enumerate(studies):
        spath = os.path.join(base, study)
        for series in sorted(e.name for e in os.scandir(spath) if e.is_dir()):
            dpath = os.path.join(spath, series)
            try:
                files = sorted(e.name for e in os.scandir(dpath) if e.name.endswith(".dcm"))
            except OSError:
                errors += 1
                continue
            n_files_total += len(files)
            row = {"split": split, "StudyInstanceUID": study,
                   "SeriesInstanceUID": series, "n_slices": len(files)}
            if not files:
                rows.append(row)
                continue
            try:
                # Middle slice: most representative, and avoids localiser oddities at the ends.
                ds = pydicom.dcmread(os.path.join(dpath, files[len(files) // 2]),
                                     stop_before_pixels=True, force=True)
                for t in TAGS:
                    v = getattr(ds, t, None)
                    row[t] = str(v) if v is not None else None
                ps = getattr(ds, "PixelSpacing", None)
                row["PixelSpacing"] = float(ps[0]) if ps else None
                iop = getattr(ds, "ImageOrientationPatient", None)
                row["IOP"] = ",".join(f"{float(x):.4f}" for x in iop) if iop else None
                ipp = getattr(ds, "ImagePositionPatient", None)
                row["IPP"] = ",".join(f"{float(x):.2f}" for x in ipp) if ipp else None
                try:
                    row["TransferSyntax"] = str(ds.file_meta.TransferSyntaxUID.name)
                except Exception:
                    row["TransferSyntax"] = "UNKNOWN"
            except Exception as e:
                errors += 1
                row["read_error"] = str(e)[:120]
            rows.append(row)
        if (si + 1) % 500 == 0:
            log(f"  {si+1}/{len(studies)} studies, {len(rows)} series, {n_files_total} files")

    log(f"{split}: {len(rows)} series, {n_files_total} files, {errors} errors")
    return rows


def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


rows = scan_split("train") + scan_split("test")
df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/series_meta.csv", index=False)
log(f"saved series_meta.csv  shape={df.shape}")

tr = df[df.split == "train"].copy()
ts_csv = pd.read_csv(f"{ROOT}/train_series.csv")
tr = tr.merge(ts_csv, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")

# ----------------------------------------------------------------- A1 + A2
sec("A1/A2  LATERALITY  -> bilateral studies, and is the tag always there?")
print("Laterality value counts (per series):")
print(tr["Laterality"].value_counts(dropna=False).to_string())
print(f"\nSeries with NO Laterality: {tr['Laterality'].isna().sum()} "
      f"({tr['Laterality'].isna().mean():.1%})")

g = tr.groupby("StudyInstanceUID")["Laterality"]
n_lat = g.nunique(dropna=True)
print(f"\nStudies total: {len(n_lat)}")
print("Distinct Laterality values per study:")
print(n_lat.value_counts().sort_index().to_string())
bilat = n_lat[n_lat >= 2].index
print(f"\n*** BILATERAL studies (>=2 distinct Laterality): {len(bilat)} "
      f"({len(bilat)/len(n_lat):.2%}) ***")

miss_any = tr.groupby("StudyInstanceUID")["Laterality"].apply(lambda s: s.isna().any())
miss_all = tr.groupby("StudyInstanceUID")["Laterality"].apply(lambda s: s.isna().all())
print(f"Studies with ANY series missing Laterality: {miss_any.sum()} ({miss_any.mean():.2%})")
print(f"Studies with ALL series missing Laterality: {miss_all.sum()} ({miss_all.mean():.2%})")
print("\n-> if ALL-missing is non-zero we need a geometry fallback from IOP/IPP.")

# ----------------------------------------------------------------- A3
sec("A3  PATIENT ID REPEATS  -> do folds need to group on patient?")
pid = tr.groupby("StudyInstanceUID")["PatientID"].first()
print(f"Studies: {len(pid)}   distinct PatientIDs: {pid.nunique()}   "
      f"null PatientID: {pid.isna().sum()}")
vc = pid.value_counts()
print("\nStudies per PatientID:")
print(vc.value_counts().sort_index().to_string())
rep = (vc > 1).sum()
print(f"\n*** PatientIDs appearing in >1 study: {rep} "
      f"(covering {vc[vc>1].sum()} studies) ***")
print("-> non-zero means GroupKFold must group on patient as well as scanner.")

# ----------------------------------------------------------------- A4
sec("A4  SITE / SCANNER FINGERPRINTS  -> how do we build grouped folds?")
for col in ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"]:
    print(f"\n{col}:")
    print(tr.groupby("StudyInstanceUID")[col].first().value_counts(dropna=False).head(15).to_string())

fp_cols = ["Manufacturer", "ManufacturerModelName", "SoftwareVersions",
           "ImagingFrequency", "ReceiveCoilName"]
stud = tr.groupby("StudyInstanceUID")[fp_cols].first()
stud["fingerprint"] = stud[fp_cols].astype(str).agg("|".join, axis=1)
fpc = stud["fingerprint"].value_counts()
print(f"\n*** distinct scanner fingerprints: {len(fpc)} ***")
print(f"largest group: {fpc.iloc[0]} studies ({fpc.iloc[0]/len(stud):.1%})")
print(f"top 5 cover {fpc.head(5).sum()/len(stud):.1%}, "
      f"top 20 cover {fpc.head(20).sum()/len(stud):.1%}")
print(f"singleton fingerprints: {(fpc==1).sum()}")
print("\nTop 15 fingerprint sizes:"); print(fpc.head(15).to_string())

# a coarser grouping, likelier to match true institutions
stud["coarse"] = stud["Manufacturer"].astype(str) + "|" + stud["ManufacturerModelName"].astype(str)
cc = stud["coarse"].value_counts()
print(f"\nCoarse (Manufacturer|Model) groups: {len(cc)}; largest {cc.iloc[0]/len(stud):.1%}")
print("-> pick the coarsest grouping that still yields >=5 usable folds.")

# ----------------------------------------------------------------- A5
sec("A5  PHYSICAL SCALE  -> what should CROP_MM and target resolution be?")
tr["fov_mm"] = tr["Rows"].astype(float) * tr["PixelSpacing"].astype(float)
for c in ["PixelSpacing", "Rows", "Columns", "SliceThickness", "SpacingBetweenSlices", "fov_mm"]:
    s = pd.to_numeric(tr[c], errors="coerce").dropna()
    if len(s):
        q = s.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).round(3)
        print(f"\n{c}:  n={len(s)}")
        print("  " + "  ".join(f"p{int(k*100)}={v}" for k, v in q.items()))

fov = pd.to_numeric(tr["fov_mm"], errors="coerce").dropna()
print("\nFraction of series whose field of view is at least X mm:")
for mm in [100, 110, 120, 130, 140, 150, 160, 180]:
    print(f"  >= {mm:3d} mm : {(fov>=mm).mean():6.2%}")
ps = pd.to_numeric(tr["PixelSpacing"], errors="coerce").dropna()
print(f"\nPixelSpacing spread (p99/p01): {ps.quantile(.99)/ps.quantile(.01):.2f}x")
print("-> CROP_MM should sit just below the FOV of ~99% of series.")

# ----------------------------------------------------------------- A6
sec("A6  SLICES PER SERIES / SLOT  -> how many slices to cache?")
print("n_slices overall:")
print(pd.to_numeric(tr["n_slices"]).describe(percentiles=[.05,.25,.5,.75,.95,.99]).round(1).to_string())
tr["slot"] = (tr["Anatomical_Plane"].astype(str) + "|" +
              np.where(tr["Fluid_Sensitive"] == 1, "FLUID", "STRUCT"))
print("\nn_slices by slot:")
print(tr.groupby("slot")["n_slices"].describe(percentiles=[.05,.5,.95]).round(1).to_string())
print("\nSeries count by slot:"); print(tr["slot"].value_counts().to_string())
cov = tr.groupby("slot")["StudyInstanceUID"].nunique() / tr["StudyInstanceUID"].nunique()
print("\nStudy coverage per slot:"); print(cov.round(3).sort_values(ascending=False).to_string())

# ----------------------------------------------------------------- A7
sec("A7  TRANSFER SYNTAXES  -> decode cost for cache build and inference")
print(tr["TransferSyntax"].value_counts(dropna=False).to_string())
print("\nSeries share by syntax:")
print((tr["TransferSyntax"].value_counts(normalize=True)*100).round(2).to_string())

log("timing a decode of one slice per transfer syntax ...")
for syn, sub in tr.groupby("TransferSyntax"):
    r = sub.iloc[0]
    d = os.path.join(ROOT, "train_series", r["StudyInstanceUID"], r["SeriesInstanceUID"])
    try:
        f = sorted(x for x in os.listdir(d) if x.endswith(".dcm"))[0]
        t = time.time()
        for _ in range(5):
            _ = pydicom.dcmread(os.path.join(d, f)).pixel_array
        ms = (time.time()-t)/5*1000
        print(f"  {syn:<45} {ms:7.1f} ms/slice")
    except Exception as e:
        print(f"  {syn:<45} decode FAILED: {str(e)[:70]}")
print("\n-> multiply ms/slice by ~423,000 to estimate the full cache build.")

# ----------------------------------------------------------------- A8
sec("A8  SeriesDescription vs THE PROVIDED CSV COLUMNS")
print(f"SeriesDescription null: {tr['SeriesDescription'].isna().mean():.1%}")
print(f"distinct values: {tr['SeriesDescription'].nunique()}")
print("\nTop 30 SeriesDescription values:")
print(tr["SeriesDescription"].value_counts().head(30).to_string())

sd = tr["SeriesDescription"].fillna("").str.lower()
def guess_plane(s):
    if any(k in s for k in ["sag", "sg_"]): return "Sagittal"
    if any(k in s for k in ["cor", "coron"]): return "Coronal"
    if any(k in s for k in ["tra", "ax", "trans"]): return "Axial"
    return "?"
tr["plane_guess"] = sd.map(guess_plane)
print("\nParsed plane vs provided Anatomical_Plane:")
print(pd.crosstab(tr["plane_guess"], tr["Anatomical_Plane"], dropna=False).to_string())
agree = (tr["plane_guess"] == tr["Anatomical_Plane"])
known = tr["plane_guess"] != "?"
print(f"\nParseable from description: {known.mean():.1%}; "
      f"agreement where parseable: {agree[known].mean():.1%}")

fs_kw = sd.str.contains("fs|stir|fatsat|spair|spir|tirm", regex=True)
print("\nfat-sat keyword in description vs provided Fat_Suppression:")
print(pd.crosstab(fs_kw, tr["Fat_Suppression"], dropna=False).to_string())
print("-> low agreement means one source is wrong; inspect before trusting either.")

sec("SANITY")
print(f"train series rows: {len(tr)}   (train_series.csv has {len(ts_csv)})")
print(f"unmatched merge rows: {tr['Anatomical_Plane'].isna().sum()}")
print(f"read errors: {tr['read_error'].notna().sum() if 'read_error' in tr else 0}")
log("DONE")
