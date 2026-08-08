# Platform decision

## Decision

**Kaggle Notebooks. All GPU work runs on Kaggle — nothing trains locally.**

This is settled: the local machine has no CUDA GPU (`nvidia-smi` absent) and 86.5 GB free against
a **569.76 GB / 819,640-file** dataset. Local is for CSV analysis, label-extractor code, and git.
Every model touches a GPU only on Kaggle.

---

## Why not the other options

**Your machine — ruled out.** `nvidia-smi` is absent, so there is no usable training GPU, and the
project drive (`H:`) has 86.5 GB free against a 570 GB dataset. Even with unlimited disk you would
spend days downloading through the Kaggle API. Local stays useful for exactly what it is doing now:
EDA on the CSVs, writing code, git.

**Colab — ruled out by decision.** It cannot see competition data directly; you would have to move
570 GB into it. Even with a compact cache it would mean maintaining two environments and syncing
weights back to Kaggle for submission. Not worth the split attention. Kaggle only.

**Cloud GPU (Vast/Lambda/RunPod/GCP) — wrong shape of problem.** You would pay for the GPU and then
spend the first several hours of every rental moving data. The compute is not the bottleneck; the
570 GB is.

**Kaggle — the data is already mounted.** `/kaggle/input/rsna-knee-abnormality-detection/` is there
at session start, zero transfer. And it is not really optional anyway: **this is a Code Competition,
so the final submission must be a Kaggle notebook regardless.** Anything you build elsewhere has to
come home to Kaggle to score.

---

## The move that unlocks everything: decouple DICOM I/O from training

Do not read DICOMs in your training loop. Read them **once**, into a compact array cache, then
train against the cache forever after.

You do not need all 819,640 files. Per study you need ~6 slots × ~16 slices ≈ 96 images.
Across 4,407 studies that is **~423,000 slice decodes**, about half the corpus, and the header pass
that determines slice order is a `stop_before_pixels=True` read costing 1–2 ms each.

Rough budget, 4 worker processes:

| Stage | Work | Est. time |
|---|---|---|
| Directory scan | 819 k entries via `os.scandir` | 3–6 min |
| Header pass (ordering, geometry, laterality) | ~790 k header-only reads | 8–15 min |
| Pixel decode + mm-crop + resize | ~423 k slices | 35–60 min |
| **Total, full train cache** | | **~1–1.5 h** |

That fits inside a single session. At inference, 1,300 test studies is ~125 k decodes ≈ 10–15 min,
comfortably inside the 9-hour cap.

### Cache sizing

`4407 studies × 6 slots × S slices × P² bytes` (uint8):

| P | S | Size | Fits 20 GB output? |
|---|---|---|---|
| 192 | 16 | 15.6 GB | ✅ |
| 224 | 12 | 15.9 GB | ✅ |
| 224 | 16 | 21.2 GB | ✗ — 2 shards |
| 336 | 16 | 47.8 GB | ✗ — 3 shards |

Kaggle persists **20 GB** from `/kaggle/working`, so anything larger ships as multiple Kaggle
Datasets. `/kaggle/temp` gives you ~60 GB of scratch within a session but does not persist.

**Recommendation:** build at **224 px / 16 slices in two shards**. You can always downsample at
training time; you can never upsample. If the meniscus labels turn out to need 336 px (§5D of
STRATEGY.md — Nyquist says they do), rebuild at 336 in three shards as a second pass, once, when
the rest of the pipeline is proven.

Store `uint8`, not `float32`. Intensity is already normalised into [0,1] before quantisation, so
8 bits cost nothing a bilinear resize has not already cost, and the file is a quarter the size.

---

## Resource facts to plan against

| | Kaggle |
|---|---|
| GPU | P100 16 GB, or 2×T4 16 GB |
| GPU quota | **~30 h / week** |
| Session cap | 12 h (**9 h for a scored submission**) |
| RAM | ~30 GB |
| CPU | 4 cores |
| Persistent output | 20 GB |
| Competition data | **mounted, free** |

*Verify the quota and disk numbers in-session before committing to a long run — Kaggle adjusts them.*

### The binding constraint is 30 GPU-hours per week

Not raw speed — quota. The public baseline burns an 8-hour budget per training run, which is
roughly **3–4 full runs per week**. With Kaggle as the only GPU, budgeting those hours *is* the
strategy:

- **Never spend GPU on I/O.** The cache build and the report-labelling pass are CPU jobs. Run them
  with the accelerator switched **off** — a CPU session does not draw against the GPU quota.
- **Prove every change on a subset first.** A 1,200-study, single-fold, 3-epoch run costs ~30 min
  and kills most bad ideas. Only promote survivors to a full run.
- **Cache aggressively.** Once the pixel cache is a Kaggle Dataset, no training run ever re-reads a
  DICOM. This is the difference between 8-hour and 2-hour epochs.
- **Use 2×T4 over P100 when the model shards cleanly**, P100 when it does not — T4 pairs give more
  aggregate throughput for small backbones at modest resolution.
- **Reserve ~6 h/week for submission runs.** Scored notebook runs draw from the same pool.

Practical weekly shape: ~4 h subset experiments, ~16 h full training runs, ~6 h submissions,
~4 h slack.

---

## Recommended split

| Where | Accelerator | What |
|---|---|---|
| **Local (this machine)** | none | CSV/report EDA, label-extractor development and scoring against the 58, all code authoring, git |
| **Kaggle — job 1** | **CPU off-quota** | Build the DICOM cache once → publish as a Kaggle Dataset |
| **Kaggle — job 2** | **CPU off-quota** | Open-weights multilingual LLM labelling pass over the reports (data never leaves Kaggle → rules-safe) → publish labels as a Dataset |
| **Kaggle — training** | GPU | All model training, against the cached dataset |
| **Kaggle — submission** | GPU | Inference notebook, internet off, ≤9 h |

The label extractor is pure text on 5.4 MB of CSV — it runs on your laptop in seconds. That is where
the Tier-1 idea lives, and it needs no GPU at all.
