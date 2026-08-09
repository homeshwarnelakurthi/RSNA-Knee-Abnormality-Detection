# Experiment log

Runtime is logged on every run: it is half the Efficiency Track metric and cannot be
reconstructed afterwards.

---

## train-v1 — 2026-08-08 · first end-to-end model

| | |
|---|---|
| Backbone | `resnet34`, slices-as-channels (12 ch), masked attention over 6 slots |
| Cache | 224 px, 12 slices, 6 slots, `CROP_MM` 130, canonicalised to one side |
| Targets | severity soft labels (labeler v3), `confidence` as per-sample loss weight |
| Folds | GroupKFold on `language\|manufacturer\|model`, fold 0 of 5 (14 groups held out) |
| Train / val | 3,479 / 870, 58 gold studies excluded from training entirely |
| Epochs | 12, OneCycle, AdamW, no horizontal flip |
| Hardware | 1× Tesla T4 |
| **Runtime** | **1,224 s (20.4 min)** |

### Result

| Metric | Value |
|---|---|
| Grouped val AUC (vs weak labels) | 0.7049 |
| Random val AUC (vs weak labels) | 0.8412 |
| **Site-reliance gap** | **+0.1363** |
| **Gold macro AUC (58 held-out, true labels)** | **0.6739** (best, ep 9) |

Per-label AUC on the 58 gold studies:

| Label | n_pos | AUC | |
|---|---|---|---|
| Effusion | 35 | **0.922** | |
| Contusion | 19 | 0.776 | |
| Baker's | 12 | 0.748 | |
| ACL | 24 | 0.719 | |
| Lateral Meniscus | 23 | 0.704 | |
| Fracture | 18 | 0.672 | |
| Synovitis | 27 | 0.662 | |
| Lateral OA | 11 | 0.658 | |
| Medial OA | 15 | 0.650 | |
| PF OA | 21 | 0.574 | |
| **Medial Meniscus** | 26 | **0.476** | below chance |
| **MCL** | 9 | **0.420** | below chance |

---

## What this run actually told us

### 1. Vision capacity is the bottleneck, not label quality

This is the diagnostic `ROADMAP.md` Phase 2 was built to answer, and it resolved earlier
than expected:

| Predictor of the 58 gold labels | Macro AUC |
|---|---|
| Report labeler v3, from **text** | **0.791** |
| train-v1, from **images** | 0.674 |

**Our text labeler beats our vision model by 0.117.** The targets carry more signal than
the model is currently extracting, so effort belongs on the model — resolution, backbone,
slice sampling — not on squeezing the labeler further.

*Caveat both ways:* the labeler's 0.791 is partly in-sample (three iterations against
these same 58), and the model is one fold of `resnet34` at 224 px for 12 epochs. Neither
number is settled. The **direction** is what matters, and it is not close.

### 2. Site memorisation is real, large, and growing

| Epoch | Grouped | Random | Gap |
|---|---|---|---|
| 1 | 0.516 | 0.562 | +0.046 |
| 4 | 0.671 | 0.741 | +0.070 |
| 8 | 0.702 | 0.819 | +0.117 |
| 12 | 0.705 | 0.841 | **+0.136** |

Random validation climbs steadily to 0.841 while grouped **plateaus at ~0.70 from epoch 6**
and gold peaks at epoch 9 then declines. Everything after epoch ~7 is the model learning
*which scanner took the picture*, not what is wrong with the knee.

At +0.136 this is **2.6× the 0.053 that metadata alone explains** (`FINDINGS.md` §3.3), so
it is not just header leakage — the model is reading scanner signature out of the pixels
themselves: noise texture, reconstruction kernel, native resolution.

A team validating on random folds would read 0.841 here, feel good, and not discover the
problem until the private leaderboard.

**Actions:** aggressive intensity/gamma/noise augmentation to break the scanner
fingerprint; random-resized-crop to break the native-resolution signature; earlier
stopping (~epoch 7–9); and consider domain-adversarial training on the site label.

### 3. The two failing labels were predicted in advance

`Medial Meniscus` 0.476 and `MCL` 0.420 are **below chance**. Both are fine-detail
findings, and `FINDINGS.md` §4 predicted exactly this: at `CROP_MM` 130 and 224 px the
pixel pitch is **0.58 mm**, and Nyquist says a 1 mm meniscal tear needs **≤0.5 mm**.
336 px gives 0.387 mm and clears it.

Two competing explanations, and they are separable:

- **Resolution** — test by rebuilding the cache at 336 px. Predicts menisci improve, and
  Effusion (a large finding, already 0.922) does not.
- **Slice sampling** — on sagittal the menisci sit near the *ends* of the stack, and the
  sampling band is (0.18, 0.82) with only 12 slices. We may simply be sampling past them.
  Test by widening the band and raising the slice count on sagittal slots.

MCL additionally has only **9 positives** and was the one label the text labeler also
failed (−0.075). Its labels are probably weak, and at n=9 nothing about it is measurable.
Do not tune on MCL.

### 4. Effusion at 0.922 is the proof of concept

A well-posed, visible finding reaches 0.922 through this pipeline in 20 minutes on a
resnet34. Nothing structural is broken — geometry, canonicalisation, slot masking and the
label join all work. The weak labels are usable.

---

## Next, in priority order

1. **336 px cache** — directly tests the resolution hypothesis on the two failing labels.
2. **Anti-site augmentation** — the +0.136 gap is the single largest measured loss.
3. **Sagittal slice sampling** — widen band, more slices; cheap to test.
4. **First submission** — the gold-58 number is *not* comparable to a leaderboard AUC
   (58 enriched studies vs ~390). Only a submission gives a number comparable to the
   public baseline's 0.891.
5. Backbone comparison — deferred until 1–3 are settled; changing it now would confound.

## Not learned yet

- Whether any of this transfers to the leaderboard. No submission has been made.
- Whether 12 epochs is right — gold peaked at 9 and the run was not early-stopped.
- Multi-fold variance. Single fold, single seed, so ±0.02 on gold is unresolvable.
