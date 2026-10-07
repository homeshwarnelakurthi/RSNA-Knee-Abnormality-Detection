# Literature review — what transfers, 2026-10-07

Sources read this session, with what each contributes. **Ranked by whether it attacks a problem we
have actually measured**, not by citation count.

## The calibration that frames everything

| Source | Task | Reported AUC |
|---|---|---|
| Bien 2018 (MRNet) | ACL tear | 0.965 |
| Bien 2018 (MRNet) | Meniscal tear | 0.847 |
| Astuto 2021 (RSNA Rad:AI) | Cartilage / meniscal horns | 0.93 / 0.93 |
| Astuto 2021 | ACL / bone marrow edema | 0.90 / 0.83 |
| Botnari 2024 (meta, 12 studies) | Meniscal tear, pooled | 0.915 |
| **Public Kaggle ensemble** | **macro over all 12** | **0.943** |

**The public ensemble already matches or beats the published single-task literature.** No paper
here contains a trick that lifts 0.943 to 0.957. These papers are useful for diagnosing **our own
broken parts** (0.674), not for beating the field.

---

## 1. The diagnostic lead — our medial/lateral result is inverted vs the literature

| | Medial meniscus | Lateral meniscus |
|---|---|---|
| Botnari 2024 pooled (3 studies each) | **0.925** | 0.844 |
| Our train-v1 (gold-58) | **0.476** | 0.704 |

Across the literature the **medial** meniscus is the *easier* of the two — it tears more often and
more visibly. Ours is the harder one, and sits at chance. That inversion is a signature of
something specific being wrong on the medial side, not of general difficulty.

**Candidate cause, and why our August check would not have caught it.**
`eda/08_canonicalisation_check.py` compared the mean left knee against the mean right knee and
confirmed they match rather than mirror. That proves **consistency**, not **correctness** — if
every study were flipped to the wrong canonical side, medial and lateral would be systematically
swapped against the labels and the test would still pass 6 of 6.

**Cheap decisive test:** score the gold-58 with the Medial/Lateral meniscus predictions *swapped*.
If swapped-medial scores materially above 0.5 while as-is scores 0.476, the canonicalisation sign
is inverted. Requires re-running inference to dump per-study predictions, which train-v1 did not save.

*Caveat:* Medial OA 0.650 vs Lateral OA 0.658 shows no such inversion, which argues against a clean
global swap. Treat this as a lead to test, not a diagnosis.

## 2. Segment, crop, classify (three independent sources)

- **Astuto 2021:** V-Net segments 11 bone/cartilage/meniscus/ligament ROIs, builds volumetric
  bounding boxes, crops subvolumes, then runs 17 small 3D classifiers. Cartilage and meniscal horns
  both 0.93.
- **Liu 2018:** an image-partition step after a segmentation CNN extracts patches containing the
  segmented cartilage, then classifies the patches.
- **Botnari 2024:** the best meniscal model reviewed is Li's **3D-Mask-RCNN** — AUC 0.907 against
  radiologists' 0.834 (p = 0.0009). Region-based detectors dominate the table.

**Why this matters to us:** we tried to solve small-structure detection by *raising global
resolution* (the Nyquist argument: 224 px gives 0.58 mm/px, too coarse for a 1 mm tear). The
literature solves it by **localising the structure and cropping**, which buys effective resolution
without paying for it globally.

**It is also efficiency-track friendly** — small crops are cheap at inference, where raising global
resolution is not. This is the single best architectural idea found.

## 3. Per-label plane specialisation (two sources)

- **Bien 2018:** trained a *separate network per plane*, combined by logistic regression, and
  reported which plane carried each task.
- **Astuto 2021**, on their own limitation: *"Collateral ligaments are better appreciated in the
  coronal view, and the use of a single sagittal sequence prevented their inclusion."*

That second quote lands directly on **MCL = 0.420**, our worst target. Our model pools the 6 slots
with **one shared attention** for all 12 labels, so a label that needs coronal competes with eleven
that may not.

**Change:** per-label (or per-label-group) attention over slots. Cheap, no inference cost.
*This was already Tier-2 idea E in `STRATEGY.md` from August and was never implemented — two papers
now independently support it.*

## 4. Domain shift — the literature says our +0.136 is expected

Our measured grouped-vs-random gap was **+0.136** and we treated it as alarming. The literature:

- Scanner domain shift is **worst in MRI** — more severe than X-ray, far more than CT, because MRI
  acquisition is not standardised.
- Models generalise well to data acquired with similar protocols but "substantially worse in
  clinical cohorts with visibly different tissue contrasts."
- **Instance normalisation instead of batch normalisation** reduces instance-specific style
  variation and improves out-of-distribution performance.
- **Intensity normalisation is "an important but underexplored factor"**; Botnari 2024 separately
  lists histogram-based intensity standardisation among the preprocessing choices that helped.

We currently do per-image 1–99 percentile windowing. Histogram standardisation plus instance norm
is a cheap, inference-free change aimed squarely at the measured gap.

## 5. 3D beats 2D, consistently

Botnari 2024's conclusion across 12 studies: 3D architectures consistently outperformed 2D, and
future work should operate on 3D images. Our model is 2D with slices-as-channels; the public
ensemble is 2.5D; the competitor writeup below is 2.5D.

**In tension with the efficiency track.** Noted as a known trade, not an action.

## 6. OsteoHRNet and the ordinal framing (arXiv 2106.14292)

HRNet plus CBAM, OAI baseline cohort, KL grade 0–4 by **ordinal regression** rather than
cross-entropy. 71.74 % five-class accuracy, MAE 0.311. X-ray, single ordinal target — not our task.

The transferable part: **Medial OA, Lateral OA and PF OA are one ordinal construct read in three
compartments**, thresholded at "greater than 50 % cartilage thickness over at least 1 cm". Astuto
independently grades on the ordinal **WORMS** scale. Verified by grepping the public training
kernel and the full 0.942 ensemble notebook:

```
binary_cross_entropy_with_logits, focal
ordinal / CORAL / CORN / cumulative / monotonic:  0 occurrences
hrnet:                                            0 occurrences
```

Twelve independent sigmoids. The ordinal structure is **unexploited by the public ensemble**.
Those three targets are 0.25 of the metric and three of our four worst.

*Caveat:* ordinal loss demonstrably helps accuracy and MAE. Whether it helps **AUC**, where our
soft targets already encode ordinality, is untested.

---

## Competitor intelligence (not literature)

**`kaszub.ski/work/rsna-kaggle`** — a public writeup of this competition:

- 2.5D **EfficientNetV2 + DINOv2** ensemble, 12 outputs; earlier baseline was a timm slice encoder
  into a BiGRU with attention pooling
- **384 px** windowed uint8 (public practice is above the 336 px we reasoned to)
- about 500 GB of DICOM preprocessed in shards
- **"GroupKFold at patient level"** for cross-validation
- 58 gold studies as held-out test; **83.5 %** of label cells correct at threshold 0.50
- **"slips most on synovitis and effusion"**
- **3 x A100 80 GB, 300+ GPU-hours**

Two things to take from this. First, **their patient-level GroupKFold does nothing** — we verified
there are 4,407 distinct `PatientID`s for 4,407 studies, so grouping on patient is identical to
random K-fold. That is now the *second* strong competitor (with the public e11 kernel's
report-hash grouping) with no site control. Our site-grouped CV remains genuinely unusual.

Second, **300+ A100-hours** against our 30 T4-hours per week. We cannot win by matching compute.

**`github.com/SE-SaaS/RSNA-Knee-Abnormality-Detection`** — checked, it is a **stub**:
`.gitignore`, `LICENSE`, empty README. No code.

---

## What this changes

Ranked by expected value given 15 days and about 30 GPU-h per week:

1. **Test the medial/lateral inversion.** Cheapest, and if it fires it is worth more than
   everything else here. Our own verification could not have caught it.
2. **Per-label slot attention.** Two papers support it; costs nothing at inference; attacks MCL.
3. **Histogram standardisation plus instance norm.** Aimed at the measured +0.136 gap; free at
   inference.
4. **Ordinal head coupling the three OA targets.** Verified unexploited; 0.25 of the metric.
5. **ROI crop for menisci.** Best idea in the literature, but needs a segmentation stage we do not
   have. Probably out of scope at 15 days.

## Sources

- [Bien 2018, MRNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- [Astuto 2021, RSNA Radiology: AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/)
- [Botnari 2024, meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11172202/)
- [Jain et al., OsteoHRNet, arXiv 2106.14292](https://arxiv.org/abs/2106.14292)
- [Mead 2025 systematic review (abstract only, paywalled)](https://pubmed.ncbi.nlm.nih.gov/39422725/)
- [Competitor writeup](https://kaszub.ski/work/rsna-kaggle)
- Liu 2018 (PMC6166867) — CAPTCHA-blocked; gist taken from the indexed snippet only
