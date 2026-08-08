"""
Objective test that left/right canonicalisation actually worked.

Eyeballing a montage cannot settle this: medial and lateral femoral condyles look
broadly alike, and the failure mode is silent. So test it statistically instead.

If canonicalisation is correct, the average left knee and the average right knee should
look the SAME after processing. If it silently failed, the average left knee should look
like the MIRROR of the average right knee.

    d_same = || mean_L  -  mean_R ||
    d_flip = || mean_L  -  mirror(mean_R) ||

    d_same < d_flip  -> canonicalised
    d_flip < d_same  -> still mirrored, the bug is live

The mirror axis differs per plane, which is the whole point:
  coronal / axial -> mirror the columns   (left-right runs across the image)
  sagittal        -> reverse slice order  (left-right runs through the stack)
"""
import numpy as np, pandas as pd, cv2, pathlib

OUT = pathlib.Path(r"H:\RSNA Knee Abnormality Detection\kaggle\out_cache")
arr = np.load(OUT / "cache.npy", mmap_mode="r")
idx = pd.read_csv(OUT / "index.csv")
mask = np.load(OUT / "slot_mask.npy")
SLOTS = ["SAG_FLUID", "SAG_STRUCT", "COR_FLUID", "COR_STRUCT", "AX_FLUID", "AX_STRUCT"]
print("cache:", arr.shape, " sides:", idx["side"].value_counts().to_dict())

L = idx.index[idx["side"] == "L"].values
R = idx.index[idx["side"] == "R"].values
print(f"L={len(L)}  R={len(R)}")

print("\n" + "=" * 78)
print(f"{'slot':<12} {'nL':>4} {'nR':>4} {'d_same':>9} {'d_flip':>9} {'ratio':>7}  verdict")
print("=" * 78)
rows = []
for si, name in enumerate(SLOTS):
    li = [i for i in L if mask[i, si]]
    ri = [i for i in R if mask[i, si]]
    if len(li) < 4 or len(ri) < 4:
        print(f"{name:<12} {len(li):>4} {len(ri):>4}   too few studies to test")
        continue
    ml = arr[li, si].astype(np.float32).mean(0)   # (S, H, W)
    mr = arr[ri, si].astype(np.float32).mean(0)

    if name.startswith("SAG"):
        mr_flip = mr[::-1, :, :]                  # reverse slice order
        axis = "slice order"
    else:
        mr_flip = mr[:, :, ::-1]                  # mirror columns
        axis = "columns"

    d_same = float(np.sqrt(((ml - mr) ** 2).mean()))
    d_flip = float(np.sqrt(((ml - mr_flip) ** 2).mean()))
    ratio = d_flip / max(d_same, 1e-6)
    ok = d_same < d_flip
    print(f"{name:<12} {len(li):>4} {len(ri):>4} {d_same:>9.3f} {d_flip:>9.3f} "
          f"{ratio:>7.3f}  {'OK' if ok else '*** MIRRORED ***'}  (mirror axis: {axis})")
    rows.append({"slot": name, "d_same": d_same, "d_flip": d_flip,
                 "ratio": ratio, "ok": ok})

R2 = pd.DataFrame(rows)
print("\n" + "=" * 78)
if len(R2):
    print(f"slots passing: {int(R2['ok'].sum())}/{len(R2)}   mean ratio {R2['ratio'].mean():.3f}")
    print("ratio > 1 means the mirrored average is FURTHER away, i.e. canonicalisation held.")
    print("A ratio near 1.0 is uninformative - the test has no power on that slot.")

# ---- visual: mean left vs mean right, per slot, middle slice
tiles = []
for si, name in enumerate(SLOTS):
    li = [i for i in L if mask[i, si]]
    ri = [i for i in R if mask[i, si]]
    if len(li) < 4 or len(ri) < 4:
        continue
    mid = arr.shape[2] // 2
    a = arr[li, si, mid].astype(np.float32).mean(0)
    b = arr[ri, si, mid].astype(np.float32).mean(0)
    d = np.abs(a - b)
    strip = np.hstack([a, b, d / max(d.max(), 1e-6) * 255])
    strip = cv2.copyMakeBorder(strip.astype(np.uint8), 22, 4, 0, 0,
                               cv2.BORDER_CONSTANT, value=0)
    cv2.putText(strip, f"{name}  meanL | meanR | |diff|", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
    tiles.append(strip)
if tiles:
    cv2.imwrite(str(OUT / "canon_check.png"), np.vstack(tiles))
    print(f"\n[wrote {OUT / 'canon_check.png'}]")
    print("If canonicalisation worked, meanL and meanR look alike and |diff| is dim.")
