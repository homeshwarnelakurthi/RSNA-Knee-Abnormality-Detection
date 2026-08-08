"""
Phase 1 - build the pixel cache.

Turns 570 GB of DICOM into one uint8 array that every training run reads instead.
CPU session: this is I/O bound and must not draw against the 30 GPU-hour weekly quota.

Layout: (N_studies, 6 slots, N_SLICES, IMG, IMG) uint8, plus a per-slot presence mask.

THE SUBTLETY THAT SILENTLY WRECKS EVERYTHING
--------------------------------------------
Five of the twelve labels are side-specific (medial vs lateral meniscus, medial vs
lateral OA, MCL). A left knee is the mirror of a right knee, so every study must be
canonicalised to one side or the model sees "medial" on both sides of the image.

But mirroring is NOT "flip the image horizontally". Which image axis carries the
patient's left-right direction depends on the plane:

  coronal, axial  -> left-right runs ACROSS the image  -> mirror the columns
  sagittal        -> left-right runs THROUGH the stack -> reverse the SLICE ORDER

Flipping a sagittal image horizontally mirrors anterior/posterior instead, which leaves
medial/lateral untouched and quietly corrupts the labels it was meant to fix. So the
code below resolves each image axis to a patient axis from ImageOrientationPatient and
mirrors whichever one is patient-x.
"""
import os, sys, time, json, warnings, math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pydicom
import cv2

warnings.filterwarnings("ignore")
T0 = time.time()

# ----------------------------------------------------------------- configuration
IMG = int(os.environ.get("IMG", 224))
N_SLICES = int(os.environ.get("N_SLICES", 12))
CROP_MM = 130.0            # 99.57% of series have a field of view at least this large
BAND = (0.18, 0.82)        # fraction of the ordered stack to sample across
# VERIFY RUN: a small sample with montages, to catch a geometry bug before spending
# 20 minutes and 16 GB on the full build. Set MAX_STUDIES = 0 for the real run.
MAX_STUDIES = int(os.environ.get("MAX_STUDIES", 0))  # 0 = all
MONTAGES = int(os.environ.get("MONTAGES", 8))        # how many montages to render
WORKERS = int(os.environ.get("WORKERS", 4))

SLOTS = [("SAG_FLUID", "Sagittal", 1), ("SAG_STRUCT", "Sagittal", 0),
         ("COR_FLUID", "Coronal", 1), ("COR_STRUCT", "Coronal", 0),
         ("AX_FLUID", "Axial", 1), ("AX_STRUCT", "Axial", 0)]
N_SLOT = len(SLOTS)
OUT = "/kaggle/working"


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def find_dir(*needles):
    base = "/kaggle/input"
    for depth in range(1, 4):
        stack = [(base, 0)]
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
    raise SystemExit(f"could not locate a mount containing {needles}")


ROOT = find_dir("train_series", "train.csv")
META = find_dir("series_meta.csv")
log(f"competition: {ROOT}")
log(f"phase0 artifacts: {META}")

# ----------------------------------------------------------------- geometry helpers
AXIS_L, AXIS_P, AXIS_S = 0, 1, 2      # DICOM LPS: +x Left, +y Posterior, +z Superior


def dominant_axis(v):
    """Which patient axis this direction vector mostly points along, and its sign."""
    a = int(np.argmax(np.abs(v)))
    return a, (1.0 if v[a] >= 0 else -1.0)


def parse_vec(s, n):
    try:
        v = [float(x) for x in str(s).split(",")]
        return np.array(v, dtype=np.float64) if len(v) == n else None
    except Exception:
        return None


def normalize_pixels(a):
    """Percentile window to [0,1]. MR intensities have no absolute meaning, and window
    values in the header are inconsistent across the 16 sites, so use the data."""
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
        if hi <= lo:
            return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def crop_resize(a, pix_mm):
    """Crop to a constant CROP_MM of anatomy, then resize. PixelSpacing spans 5.14x
    across this corpus, so a fixed-pixel resize would hand the encoder knees whose
    physical scale differs several-fold."""
    if pix_mm and np.isfinite(pix_mm) and pix_mm > 0:
        want = int(round(CROP_MM / pix_mm))
        h, w = a.shape
        if 0 < want < min(h, w):
            y0, x0 = (h - want) // 2, (w - want) // 2
            a = a[y0:y0 + want, x0:x0 + want]
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)


def build_slot(series_dir, files, side):
    """Read one slot: order slices in patient space, sample a band, canonicalise side.

    Returns (N_SLICES, IMG, IMG) float32 in [0,1], or None.
    """
    # Header pass: position and orientation per slice. specific_tags keeps this cheap.
    tags = [pydicom.tag.Tag(0x0020, 0x0032), pydicom.tag.Tag(0x0020, 0x0037),
            pydicom.tag.Tag(0x0028, 0x0030), pydicom.tag.Tag(0x0020, 0x0013)]
    recs = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, f), stop_before_pixels=True,
                                 specific_tags=tags, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            recs.append((f,
                         np.array([float(x) for x in ipp], dtype=np.float64) if ipp else None,
                         np.array([float(x) for x in iop], dtype=np.float64) if iop else None,
                         float(getattr(ds, "InstanceNumber", 0) or 0),
                         getattr(ds, "PixelSpacing", None)))
        except Exception:
            continue
    if not recs:
        return None

    iop = next((r[2] for r in recs if r[2] is not None), None)
    if iop is None or len(iop) != 6:
        return None
    col_dir, row_dir = iop[0:3], iop[3:6]          # +col across, +row down
    nrm = np.cross(col_dir, row_dir)

    # Order slices along the slice normal. Fall back to InstanceNumber.
    if all(r[1] is not None for r in recs):
        recs.sort(key=lambda r: float(np.dot(r[1], nrm)))
    else:
        recs.sort(key=lambda r: r[3])

    # Canonicalise the through-plane direction so the stack always runs the same way
    # in patient space, regardless of how the scanner ordered it.
    slice_axis, slice_sign = dominant_axis(nrm)
    if slice_sign < 0:
        recs = recs[::-1]

    n = len(recs)
    if n < 3:
        return None
    lo, hi = int(n * BAND[0]), max(int(n * BAND[1]), int(n * BAND[0]) + 1)
    idx = np.unique(np.clip(np.linspace(lo, hi - 1, N_SLICES).round().astype(int), 0, n - 1))
    while len(idx) < N_SLICES:                      # short stacks: repeat the edge
        idx = np.concatenate([idx, idx[-1:]])
    idx = idx[:N_SLICES]

    ps = None
    for r in recs:
        if r[4] is not None:
            try:
                ps = float(r[4][0]); break
            except Exception:
                pass

    planes, shp = [], None
    for i in idx:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, recs[i][0]), force=True)
            a = ds.pixel_array
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
            if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
                a = a.max() - a
            a = crop_resize(normalize_pixels(a), ps)
            shp = a.shape if shp is None else shp
            planes.append(a)
        except Exception:
            planes.append(None)

    if all(p is None for p in planes):
        return None
    # A slice that fails to decode must not define the shape; substituting a zero plane
    # at the wrong size would propagate and blank the whole slot.
    fill = np.zeros(shp, np.float32)
    vol = np.stack([p if (p is not None and p.shape == shp) else fill for p in planes])

    # ---- canonicalise in-plane axes to +patient direction
    row_axis, row_sign = dominant_axis(row_dir)
    col_axis, col_sign = dominant_axis(col_dir)
    if row_sign < 0:
        vol = vol[:, ::-1, :]
    if col_sign < 0:
        vol = vol[:, :, ::-1]

    # ---- mirror a LEFT knee onto the right, along whichever axis is patient-x
    if side == "L":
        if col_axis == AXIS_L:
            vol = vol[:, :, ::-1]        # coronal / axial: left-right is across
        elif row_axis == AXIS_L:
            vol = vol[:, ::-1, :]
        elif slice_axis == AXIS_L:
            vol = vol[::-1, :, :]        # sagittal: left-right is through the stack
    return np.ascontiguousarray(vol)


# ----------------------------------------------------------------- study assembly
def pick_series(sub):
    """One series per slot: the one with the most slices."""
    chosen = {}
    for name, plane, fluid in SLOTS:
        c = sub[(sub["Anatomical_Plane"] == plane) & (sub["Fluid_Sensitive"] == fluid)]
        if len(c):
            chosen[name] = c.sort_values("n_slices", ascending=False).iloc[0]
    return chosen


def do_study(args):
    study, rows, side = args
    sub = pd.DataFrame(rows)
    chosen = pick_series(sub)
    vol = np.zeros((N_SLOT, N_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros(N_SLOT, np.uint8)
    for si, (name, _, _) in enumerate(SLOTS):
        if name not in chosen:
            continue
        r = chosen[name]
        d = os.path.join(ROOT, "train_series", study, r["SeriesInstanceUID"])
        try:
            files = sorted(e.name for e in os.scandir(d) if e.name.endswith(".dcm"))
        except OSError:
            continue
        if not files:
            continue
        v = build_slot(d, files, side)
        if v is None:
            continue
        vol[si] = (v * 255).round().clip(0, 255).astype(np.uint8)
        mask[si] = 1
    return study, vol, mask


def main():
    meta = pd.read_csv(os.path.join(META, "series_meta.csv"), low_memory=False)
    meta = meta[meta.split == "train"]
    # series_meta.csv holds the DICOM header scan only. Plane and sequence type are
    # curated by the host and live in the competition's own train_series.csv.
    ts = pd.read_csv(os.path.join(ROOT, "train_series.csv"))
    meta = meta.merge(ts, on=["StudyInstanceUID", "SeriesInstanceUID"], how="inner")
    log(f"merged series: {len(meta)}  (train_series.csv has {len(ts)})")
    lat = pd.read_csv(os.path.join(META, "laterality_sources.csv"))
    # Shipped rule: DICOM tag when present, else geometry (docs/FINDINGS.md section 1).
    lat["side"] = lat["tag"].where(lat["tag"].isin(["L", "R"]), lat["geo"])
    side_of = dict(zip(lat["StudyInstanceUID"], lat["side"]))

    studies = sorted(meta["StudyInstanceUID"].unique())
    if MAX_STUDIES:
        rng = np.random.default_rng(0)
        studies = list(rng.choice(studies, min(MAX_STUDIES, len(studies)), replace=False))
    log(f"studies: {len(studies)}   slots {N_SLOT}   slices {N_SLICES}   img {IMG}")
    gb = len(studies) * N_SLOT * N_SLICES * IMG * IMG / 1e9
    log(f"cache size will be {gb:.2f} GB")

    cols = ["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "n_slices"]
    groups = {s: g[cols].to_dict("records") for s, g in meta.groupby("StudyInstanceUID")}
    tasks = [(s, groups[s], side_of.get(s)) for s in studies]

    arr = np.lib.format.open_memmap(f"{OUT}/cache.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(studies), N_SLOT, N_SLICES, IMG, IMG))
    masks = np.zeros((len(studies), N_SLOT), np.uint8)
    order, done = [], 0

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for study, vol, mask in ex.map(do_study, tasks, chunksize=4):
            arr[done] = vol
            masks[done] = mask
            order.append(study)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                rate = done / (time.time() - T0)
                log(f"  {done}/{len(tasks)}  {rate:.1f} studies/s  "
                    f"eta {(len(tasks)-done)/max(rate,1e-6)/60:.1f} min")
    arr.flush()

    pd.DataFrame({"StudyInstanceUID": order,
                  "row": np.arange(len(order)),
                  "side": [side_of.get(s) for s in order]}).to_csv(f"{OUT}/index.csv", index=False)
    np.save(f"{OUT}/slot_mask.npy", masks)
    json.dump({"img": IMG, "n_slices": N_SLICES, "crop_mm": CROP_MM, "band": BAND,
               "slots": [s[0] for s in SLOTS], "n_studies": len(order)},
              open(f"{OUT}/cache_config.json", "w"), indent=2)

    log("\nSLOT PRESENCE (fraction of studies with a usable slot):")
    for i, (name, _, _) in enumerate(SLOTS):
        log(f"  {name:<12} {masks[:, i].mean():6.1%}")
    log(f"studies with zero usable slots: {(masks.sum(1) == 0).sum()}")
    log(f"mean slots per study: {masks.sum(1).mean():.2f}")

    # ---- montages: the only way to catch a silent geometry bug
    if MONTAGES:
        os.makedirs(f"{OUT}/montage", exist_ok=True)
        for k in range(min(MONTAGES, len(order))):
            rows_img = []
            for si in range(N_SLOT):
                strip = [arr[k, si, j] for j in range(0, N_SLICES, max(1, N_SLICES // 6))][:6]
                while len(strip) < 6:
                    strip.append(np.zeros((IMG, IMG), np.uint8))
                rows_img.append(np.hstack(strip))
            m = np.vstack(rows_img)
            lab = f"{order[k][-8:]} side={side_of.get(order[k])} mask={''.join(map(str, masks[k]))}"
            m = cv2.copyMakeBorder(m, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
            cv2.putText(m, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA)
            cv2.imwrite(f"{OUT}/montage/m{k:02d}.png", m)
        log(f"wrote {min(MONTAGES, len(order))} montages "
            f"(rows = {[s[0] for s in SLOTS]}, cols = slices)")
    log("DONE")


if __name__ == "__main__":
    main()

