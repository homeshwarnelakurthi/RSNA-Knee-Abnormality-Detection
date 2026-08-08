"""
Mount check. CPU, seconds.

Confirms the 15.9 GB cache built correctly and that a kernel can mount another
kernel's output, before any of it is trusted on the GPU. The cache is far too large to
download locally, so this is how we inspect it.
"""
import os, json, glob
import numpy as np, pandas as pd

print("/kaggle/input:", sorted(os.listdir("/kaggle/input")), flush=True)


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
    return None


C = find("cache.npy", "index.csv")
if C is None:
    print("cache mount NOT FOUND. tree:")
    for r, d, f in os.walk("/kaggle/input"):
        if r.count("/") < 6:
            print(" ", r, f[:6])
    raise SystemExit(1)
print("cache mount:", C)

cfg = json.load(open(os.path.join(C, "cache_config.json")))
print("config:", json.dumps(cfg, indent=2))

arr = np.load(os.path.join(C, "cache.npy"), mmap_mode="r")
idx = pd.read_csv(os.path.join(C, "index.csv"))
mask = np.load(os.path.join(C, "slot_mask.npy"))
print(f"\ncache  {arr.shape}  {arr.dtype}  {arr.nbytes/1e9:.2f} GB")
print(f"index  {idx.shape}   mask {mask.shape}")
print(f"studies expected 4407, got {len(idx)}   unique {idx.StudyInstanceUID.nunique()}")

print("\nside distribution:")
print(idx["side"].value_counts(dropna=False).to_string())

print("\nslot presence:")
for i, s in enumerate(cfg["slots"]):
    print(f"  {s:<12} {mask[:, i].mean():6.1%}")
print(f"mean slots/study {mask.sum(1).mean():.2f}   zero-slot studies {(mask.sum(1)==0).sum()}")

# Content sanity: a slot flagged present must not be blank, and one flagged absent
# must be blank. A mismatch means the mask lies, which would poison the presence
# masking in the model.
rng = np.random.default_rng(0)
sample = rng.choice(len(idx), 400, replace=False)
blank_present = wrong_absent = 0
means = []
for i in sample:
    v = arr[i]
    for s in range(mask.shape[1]):
        m = float(v[s].mean())
        if mask[i, s] == 1:
            means.append(m)
            if m < 1.0:
                blank_present += 1
        elif m > 0.5:
            wrong_absent += 1
means = np.array(means)
print(f"\nsampled {len(sample)} studies")
print(f"  slots marked PRESENT but blank : {blank_present}   (want 0)")
print(f"  slots marked ABSENT but filled : {wrong_absent}   (want 0)")
print(f"  present-slot intensity: mean {means.mean():.1f}  p1 {np.percentile(means,1):.1f}  "
      f"p99 {np.percentile(means,99):.1f}")

# Does the weak-label table line up with the cache rows?
W = find("weak_labels_v1.csv")
if W:
    w = pd.read_csv(os.path.join(W, "weak_labels_v1.csv"))
    common = set(idx.StudyInstanceUID) & set(w.StudyInstanceUID)
    print(f"\nweak labels: {len(w)} rows, {len(common)} join the cache "
          f"({len(common)/len(idx):.1%} of cache rows covered)")
print("\nOK")
