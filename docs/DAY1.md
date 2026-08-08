# Day 1 — execution plan

**Goal of the day:** a scoring submission on the board, plus a decisive read on whether the
Tier-1 thesis (severity-graded targets, `STRATEGY.md` §5A) is real. Not a medal. A *measured
position*.

Two work streams run in parallel because they need different machines:
**text work is local and CPU-only; the cache build is a Kaggle job that runs unattended.**

---

## Block 0 (15 min) — start the long-running Kaggle job first

Kick this off before anything else so it bakes while you work on text.

- New Kaggle notebook, competition data attached, **GPU off** (this stage is I/O-bound; don't
  burn quota).
- Directory scan → `series_index.parquet`: for every study/series, the file list, `n_slices`,
  and header-derived `PixelSpacing`, `ImagePositionPatient`, `ImageOrientationPatient`,
  `Laterality`, `SeriesDescription`, `Manufacturer`, `Model`, `SoftwareVersions`,
  `ImagingFrequency`, `ReceiveCoilName`, `TR`, `TE`, `SliceThickness`.
- Save and publish as a Kaggle Dataset.

This single artifact answers open questions 3–5 in `STRATEGY.md` §8 and is the input to
everything downstream. **Expected ~15–25 min.** Do not build the pixel cache yet — get the index
out first, because it is cheap and everything depends on it.

---

## Block 1 (3–4 h, local) — the severity labeler. This is the day's real work.

The whole thesis is here. Everything else on day 1 is scaffolding.

**1.1 Build a presence baseline first (45 min).** Port a straightforward multilingual
presence extractor — negation + normality + anatomy + pathology lexicons. This is the control.
Score it on the 58.

**1.2 Add a severity axis (90 min).** For each of the 12 labels, extract an **ordinal grade**
rather than a boolean. The magnitude vocabulary is small and repeats across all 9 languages:

| Level | English | Spanish | Turkish | German | Greek | Dutch |
|---|---|---|---|---|---|---|
| trace/minimal | trace, minimal | mínima | minimal | geringe, minimal | ελάχιστη | minimaal |
| mild/small | mild, small | leve, pequeño | hafif, az | leicht, gering | ήπια, μικρή | licht, gering |
| moderate | moderate | moderado | orta | mäßig | μέτρια | matig |
| severe/large | severe, large, massive | severo, grande | ileri, belirgin | ausgeprägt, groß | σοβαρή, μεγάλη | ernstig, groot |

Plus the graded scales that appear literally: `grade/grado/graad/степен 1–4`, `Outerbridge I–IV`,
`ICRS`, and meniscal signal `grade 2` vs `grade 3` (grade 3 = surfacing = the competition's
positive; grade 2 = intrasubstance = **negative**).

**1.3 Apply the official thresholds (45 min).** Map grade → soft target using the rubric in
`STRATEGY.md` §3.1, not intuition:

- **Effusion / Baker's**: positive only at **moderate or large**. `trace/small/mild → ~0.1`.
- **OA (×3)**: positive only at **>50 % thickness loss over ≥1 cm**. Grade 1–2 chondromalacia
  → low. Grade 4 focal but small → mid. Grade 4 diffuse → high.
- **ACL**: complete/high-grade only. "mucoid degeneration", "thickening", "mild signal" → low.
- **MCL**: **acute** high-grade only. "grade 1 sprain", "chronic", "remote" → low.
- **Meniscus (×2)**: surfacing tear only. "intrasubstance", "grade 2 signal",
  "degenerative signal without surfacing" → low.
- **Contusion vs Fracture**: marrow edema *without* a fracture line → Contusion; discrete
  cortical break → Fracture. An **osteochondral** fracture appears not to count as Fracture
  (host-flagged example) — encode that.
- Global: **ambiguous → negative.** Hedged language (`possible`, `suspicious`, `cannot exclude`,
  `posible`, `şüpheli`, `πιθανή`, `V.a.`) should *pull the soft target down*, not up.

**1.4 Score both on the 58 (30 min).** Report per-label AUC and agreement, presence vs severity.

> **Read the result honestly.** n = 58, and the gold set is roughly **2× enriched** in prevalence,
> so it cannot resolve differences below ~0.02–0.03 and it is not a leaderboard. Treat it as a
> **direction check**: does severity grading move most of the 12 labels the same way? If 9 of 12
> improve, that is a signal even when no single label is individually significant. If it is 6/6,
> the thesis is unproven and you fall back to presence labels and revisit.

**Deliverable:** `labels_presence.csv` and `labels_severity.csv`, both 4,407 rows × 12 soft
targets, plus a scoring table.

---

## Block 2 (1.5 h, Kaggle) — pixel cache, subset first

Once `series_index` lands:

- **Do not build all 4,407 studies yet.** Build **1,200 studies** at 224 px / 12 slices / 6 slots
  ≈ 4.3 GB, in ~20 min. Prove the geometry code — laterality canonicalisation, mm-crop, slot
  assignment, presence mask — on something you can inspect.
- **Look at the images.** Plot one montage per slot for 10 studies. Confirm left and right knees
  are canonicalised to the same orientation and the joint is centred. Geometry bugs are silent
  and they poison every downstream number.
- Then launch the full build (224 px / 16 slices, two shards) and let it run.

---

## Block 3 (2 h, Kaggle GPU) — first model, deliberately small

Not to compete. To close the loop end to end.

- DINOv2-small (or `tf_efficientnetv2_s` — faster to iterate), 224 px, backbone frozen except the
  last few blocks.
- Targets: `labels_severity.csv`, soft-target BCE.
- **Site-grouped folds** from the scanner fingerprint. Single fold on day 1.
- **No horizontal flip** — it swaps medial and lateral and corrupts 5 of the 12 labels.
- Slot presence mask; masked attention pooling over slots.
- ~2 h budget, then submit.

Expect somewhere in the **0.85–0.90** range. If it lands well below that, the bug is almost
certainly in laterality or slot assignment, not the model.

---

## Block 4 (30 min) — submit and write down what happened

- Inference notebook, internet off, `submission.csv`.
- **You get 5 submissions/day and have used 0.** Spend at most 2 today.
- Record in `docs/EXPERIMENTS.md`: config, grouped-CV score, LB score, wall-clock runtime.
  **Log runtime from day 1** — it is half of the Efficiency Track metric (`STRATEGY.md` §7) and
  you cannot reconstruct it later.

---

## What "success" looks like tonight

1. A score on the leaderboard from your own pipeline, end to end. ✅
2. A defensible answer on presence-vs-severity labelling. ✅
3. A published cache + label dataset, so tomorrow starts at the model instead of at the I/O. ✅

**Explicitly not today:** ensembling, TTA, 336 px, multi-fold, external data, the text-distillation
head. Those are days 2–10, and every one of them is worthless on top of a pipeline with a
laterality bug.

---

## Traps, in the order you will hit them

| Trap | Cost | Guard |
|---|---|---|
| Random K-fold instead of site-grouped | **+0.053 phantom AUC** — measured, §3.3 | Group on scanner fingerprint |
| Horizontal flip augmentation | Silently corrupts 5 of 12 labels | No flips. Ever. |
| Ignoring laterality | Medial/lateral inverted on ~half the studies | Read `Laterality`; canonicalise |
| Bilateral studies | Labels are for **one** knee | Host says report/metadata disambiguates |
| Fixed-pixel resize | Anatomy scale varies ~3.4× | Crop to constant mm, then resize |
| Trusting `Fluid_Sensitive` ≠ `Fat_Suppression` | They are perfectly correlated — one bit | Use the 6-slot scheme |
| Tuning on the 58 | ~2× enriched, n far too small | Direction check only |
| Report text → hosted LLM API | **Possible DQ** (Rule 4.b) | Open weights, local or on Kaggle |
| Forgetting to log runtime | Forfeits the Efficiency Track | Log it every run |
