# RSNA 2026 Knee Abnormality Detection — Strategy

*Compiled 2026-08-08. Competition opened 2026-07-30, closes 2026-10-22 (final submission).*
*Status: analysis complete, nothing trained yet. Every claim below is sourced; hypotheses are marked as such.*

---

## 1. The competition in one paragraph

Predict **12 binary findings per knee MRI study**, scored by **macro-averaged ROC-AUC** over the 12
labels. It is a **Kaggle Code Competition**: you submit a notebook, internet off, ≤9 h runtime,
producing `submission.csv`. Test set ≈ **1,300 studies** (public LB = 30%, private = 70%).
$77,000 total, including a **separate Efficiency Track** ($18,000 over 3 places) that rewards
accuracy-per-second. Final deadline **2026-10-22**; entry/merger deadline **2026-10-15**.

The 12 labels:

| | | |
|---|---|---|
| ACL | MCL | Medial Meniscus |
| Lateral Meniscus | Medial OA | Lateral OA |
| PF OA | Effusion | Synovitis |
| Baker's | Contusion | Fracture |

---

## 2. The data, as it actually is

Measured directly from the mounted competition files, not from the description.

| Fact | Value |
|---|---|
| Total size | **569.76 GB**, 819,640 files |
| Train studies | **4,407** |
| Train series | 24,371 (median **5** per study, range 3–14) |
| Slices per series | 20–45 typical (median ~30), long tail |
| Test studies | ~1,300 |
| **Studies with all 12 labels** | **58 (1.3 %)** |
| **Studies with report only** | **4,349 (98.7 %)** |
| Reports present | 4,407 (100 %) |

### 2.1 The single most important number: 58

`train.csv` has 4,407 rows. **Only 58 carry the 12 labels.** Everything else is a radiology
report and a pile of DICOMs. There is no meaningful supervised image dataset handed to you —
**you have to manufacture one.**

That reframes the competition. It is not primarily a computer-vision problem. It is a
**weak-supervision problem**: whoever converts 4,349 multilingual free-text reports into the
best training targets wins, because that determines the ceiling for every vision model
trained downstream.

### 2.2 Reports are multilingual — measured, not assumed

Detected over all 4,407 reports (`eda/02_language_and_samples.py`):

| Language | n | % |
|---|---|---|
| English | 1,736 | 39.4 |
| Turkish | 546 | 12.4 |
| Spanish | 532 | 12.1 |
| Latin-other (Croatian/Serbian, Flemish, …) | 487 | 11.1 |
| Greek | 321 | 7.3 |
| German | 257 | 5.8 |
| Cyrillic (Bulgarian/Russian) | 220 | 5.0 |
| French | 159 | 3.6 |
| Dutch | 148 | 3.4 |

Median report ≈ 129 words. Reports are de-identified with `[DATE]` / `[TIME]` placeholders.
There are **text-substitution artifacts** in some reports (e.g. numeric fragments replaced by
the token `intact`, producing strings like `intact9xintact4cm`) — worth a cleaning pass, and a
warning against naive numeric parsing.

### 2.3 Series structure — one finding worth knowing

`Fluid_Sensitive` and `Fat_Suppression` are **perfectly correlated**. The cross-tab is exactly
diagonal: 10,361 series are (0,0) and 14,010 are (1,1); there are **zero** off-diagonal series.
They are one bit, not two. So the series space is 3 planes × 2 sequence types = 6 slots:

| Slot | series | % of studies having ≥1 |
|---|---|---|
| Axial fluid-sensitive | 4,719 | **100.0 %** |
| Sagittal non-fluid | 5,197 | 96.8 % |
| Coronal fluid-sensitive | 4,624 | 96.4 % |
| Sagittal fluid-sensitive | 4,667 | 94.2 % |
| Coronal non-fluid | 3,985 | 77.3 % |
| Axial non-fluid | 1,179 | 19.4 % |

Four slots cover >94 % of studies. The 6th (axial non-fluid) is nearly useless — present for
1 study in 5. A presence mask is needed regardless.

### 2.4 DICOM headers are richer than the CSVs — and available at test time

86 tags survive de-identification. Verified present on a real file:

- `SeriesDescription` — free text, e.g. `pd_tse_tra_d`. **Far more informative than the 3 CSV
  columns**, and available at test time.
- `Laterality` = `L`/`R` — **explicit left/right knee.**
- `PatientSex` — present in DICOM even though it is **absent from `train.csv`** despite the data
  description claiming otherwise (a known, forum-acknowledged doc bug).
- `PatientID` (pseudonymous) — lets you check for repeated patients and group folds properly.
- `Manufacturer`, `ManufacturerModelName`, `SoftwareVersions`, `MagneticFieldStrength`,
  `ReceiveCoilName`, `ImagingFrequency` — a **scanner fingerprint**, i.e. a proxy for site.
- Full MR physics: `TR`, `TE`, `TI`, `FlipAngle`, `EchoTrainLength`, `PixelBandwidth`,
  `SliceThickness`, `SpacingBetweenSlices`, `PixelSpacing`, `AcquisitionMatrix`.
- Geometry: `ImagePositionPatient`, `ImageOrientationPatient` → real 3-D slice ordering.

**Physical scale varies enormously.** The sample file is 960×960 at 0.1562 mm/px; across the
corpus `PixelSpacing` varies by ~3.4×. A fixed-pixel resize therefore feeds the network images
whose anatomy differs in scale by several-fold. **Crop to a constant millimetre extent, then
resize** — this is not a refinement, it is a correctness requirement.

---

## 3. The decisive intel: how ground truth was actually made

From the competition host (Po-Hao "Howard" Chen) in
[discussion 733826](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)
and the pinned
[Overview post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343):

> **Q: Were the labels assigned independently from the MRI images, rather than extracted from the reports?**
> **A: Yes.**
>
> **Q: If image interpretation and report text disagree, should the image-derived label be considered authoritative?**
> **A: Yes.** Note that only a small sample of provided data contains both. *It is intended to help
> participants surface this conclusion.*

Ground truth = **two subspecialty MSK radiologists reading the images, adjudicated by a third.**
The reports are original clinical reads by a single signing radiologist. These are **different
measurement instruments**, and the host says so explicitly: discrepancies are "plausible and
expected… the image-based labels use multiple readers with **stricter image-based thresholds**."

An independent audit posted by another team (Nagoya Univ. Mori Lab) on 20 of the 58 dual-labelled
studies found report-derived labels agree with ground truth only **82.5 %** of the time
(PPV 73 %, recall 80 %).

### 3.1 The official label thresholds — and why they matter more than anything else

This is the part most teams will skim. Emphasis added:

- **ACL**: *high-grade* partial or full-thickness tear — complete discontinuity, or **>50 % of
  fibres disrupted**. "Mild signal change, degeneration, or thickening without discontinuity is
  graded **negative**."
- **MCL**: high-grade partial or complete **acute** tear. "Low-grade sprains and chronic or remote
  stress changes are graded **negative**."
- **Meniscus** (each): abnormal signal that **definitely contacts the meniscal surface on ≥2
  images**, or a morphologic abnormality. "Intrasubstance degeneration that does not reach the
  surface is **negative**."
- **OA** (each of 3 compartments): a **moderate or large area (≥ ~1 cm)** of **high-grade**
  cartilage loss, defined as **>50 % of cartilage thickness**.
- **Effusion**: a **moderate or large** amount of fluid distending the joint.
- **Baker's**: a **moderate or large** fluid collection.
- **Contusion**: marrow-edema-like signal from impact, **without a discrete fracture line**.
- **Fracture**: an **acute cortical break or fracture line**.

And the governing rule:

> **"In each case, ambiguous or borderline findings ('on the fence') were graded as negative to
> favour specificity."**

**Every single label is severity-thresholded.** The ground truth is not "is the finding
mentioned" — it is "is the finding *severe and unambiguous*".

This explains the apparent label errors people are posting about. A report saying "small joint
effusion" or "mild chondropathy" or "grade 2b intrasubstance meniscal signal" or "low-grade MCL
sprain" is **correctly labelled 0**. It is not noise. It is the rubric.

### 3.2 Bilateral studies

Host: *"both knees may occasionally be scanned under one StudyInstanceUID… each bilateral study
was individually reviewed, and the released report text or DICOM metadata was adjusted as needed
to provide sufficient information for participants to disambiguate."* Labels are **for a single
knee**. So laterality resolution via the `Laterality` tag is mandatory, not optional.

### 3.3 There is no metadata shortcut

Another competitor (Oleksii Zhukov,
[discussion 733517](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517))
ran the leak probe properly so we don't have to:

- Full DICOM header metadata, no pixels → **0.6515** macro-AUC under random folds.
- Same, under **scanner-grouped** folds → **0.5981**.
- Series composition alone (the 4 CSV columns) → 0.5954.

Two conclusions. First, there is no shortcut; the leaderboard reflects genuine image reading.
Second, and more useful: **random K-fold overstates performance by ~0.053 through site
memorisation.** 265 distinct scanner fingerprints exist, top 20 covering 45 % of studies.
**Grouped CV is mandatory or your validation will lie to you.**

---

## 4. Where the field stands (as of today, day 9 of 84)

Public LB: top **0.939**, then 0.933, 0.929, and a dense cluster pinned at exactly **0.891** —
the unmistakable signature of a public baseline being forked wholesale. 676 teams.

The public baseline (`pilkwang/rsna-knee-baseline-v1`, 221 votes) is not a toy. It already does:

- a hand-built **multilingual rule extractor** covering 9+ languages with negation, normality,
  and uncertainty lexicons, clause splitting, and heading-attachment;
- **physical-scale normalisation** (`CROP_MM = 130`, then resize to 224 or 336);
- **6 plane × sequence slots** with a presence mask;
- **DINOv2-small**, last 6 blocks unfrozen, `lr_backbone = 8e-6`, `lr_head = 1e-3`, 10 epochs;
- **no horizontal flip augmentation** — correctly, because flipping swaps medial and lateral,
  which are distinct labels in 5 of the 12 targets;
- an 8-hour time budget to fit the 9-hour cap.

So the floor is high and well-defended. **0.891 is table stakes, not an achievement.**
The interesting question is what the 0.939 team is doing that the 0.891 fork is not.

---

## 5. Ideas, ranked by expected value

### Tier 1 — the thesis

**A. Severity-graded targets instead of binary presence extraction.**

This is the central bet and it follows directly from §3.1. Every competing approach —
the public rule extractor, the public LLM-labelled tables — asks *"is the finding present?"*
The rubric asks *"is the finding severe?"*

Since the metric is **AUC, which is purely rank-based**, you do not want a binary target at all.
You want a target that **preserves the severity ordering**:

```
normal cartilage  <  mild chondropathy  <  grade 3 focal  <  grade 4 over 2 cm
"no effusion"     <  "trace"  <  "small"  <  "moderate"  <  "large/massive"
intact ACL        <  "mild signal change"  <  "partial tear"  <  "complete rupture"
```

Concretely: extract an **ordinal severity score** per label from the report, then map it to a
**soft target in [0,1]** approximating *P(two MSK radiologists would call this positive)*.
Train with soft-target BCE. A model that learns the severity continuum ranks correctly under
*any* threshold, which immunises it against the exact report-vs-image threshold mismatch that
is costing everyone else ~18 % label error.

The reports are full of gradable language and it is consistent across languages: `grade 2b`,
`grado 4`, `graad 2`, `Outerbridge`, `ICRS`, `mild/moderate/severe`, `leve`, `mínima`,
`minimal`, `ελάχιστη`, `minimal mayii artışı`, `geringe`, `massive`.

*Why I believe this:* the host stated the thresholds explicitly and said the 58 dual-labelled
studies exist specifically to make participants notice. Most teams will read that as "labels are
noisy, move on". It is not noise — it is a systematic, correctable, *monotone* shift.

**B. Get laterality right, and canonicalise it.**

5 of 12 labels are side-specific (Medial/Lateral Meniscus, Medial/Lateral OA, MCL).
For a left knee the medial compartment is on the opposite image side from a right knee.
Read `Laterality` from DICOM, flip every study to a canonical side, and handle bilateral studies.
Getting this wrong scrambles half the label set; the public baseline handles it, so this is
defensive, not differentiating — but it is non-negotiable.

**C. Site-grouped cross-validation from day one.**

Group by scanner fingerprint (`Manufacturer` + `Model` + `SoftwareVersions` + `ImagingFrequency`
+ `ReceiveCoilName`). Established above: random folds inflate by 0.053. Every architectural
decision you make on random-fold CV is a coin flip.

### Tier 2 — likely gains

**D. Resolution where it matters.** A meniscal tear is a ~1 mm feature. Nyquist says you need
≤0.5 mm/px to see it. At `CROP_MM=130`, 224 px gives 0.58 mm/px (**too coarse**) and 336 px gives
0.39 mm/px (adequate). Expect real gains from 336+ on the meniscus and cartilage labels — and
expect the efficiency track to punish it. Consider **per-label resolution**: effusion and Baker's
cyst are large findings and are fine at 224; meniscus/OA are not.

**E. Label-group-specific slot weighting.** PF OA is best seen axially; cruciates sagittally;
collaterals and compartments coronally. A single shared pooling over all slots wastes capacity.
Attention pooling over slots, per label group, is cheap and principled.

**F. Multiple weak-label sources, disagreement-aware.** Build ≥2 independent labelers (rule-based
+ open-weights LLM), then treat **disagreement as an uncertainty signal** — down-weight those
studies in the loss rather than forcing a hard label. Cheap, robust, and nobody bothers.

**G. Auxiliary text-supervision at training time.** The report is available at train time and
absent at test time — a textbook **privileged-information / distillation** setup. Train the image
encoder to also predict a frozen multilingual sentence embedding of the report (auxiliary
regression head, dropped at inference). This transfers signal the 12 binary labels throw away, at
zero inference cost. Higher-risk, higher-ceiling.

### Tier 3 — worth testing, lower confidence

**H. Metadata head as an ensemble member.** 0.598 site-grouped AUC from headers alone is not
nothing. Blended at low weight it may add a few thousandths, essentially free at inference.

**I. External data.** MRNet (Stanford), fastMRI+, OAI, SKM-TEA. **Blocked pending a host ruling** —
see §6. Do not build on this until answered.

**J. Efficiency Track as the real target.** See §7 — this is the most winnable prize here.

---

## 6. Rules constraints — read before writing any code

**Do not send report text to a commercial LLM API.** Competition Rule 4.b (Data Security)
forbids making Competition Data available to any party not participating. Two strong competitors
have publicly read it that way
([discussion 733652](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652));
**the host has not yet ruled.** Until they do, the safe and equally effective path is an
**open-weights multilingual LLM run locally or inside a Kaggle notebook** (Qwen3, Gemma,
multilingual-e5). Internet-off applies only to the *submission* notebook, so offline label
generation during development is unrestricted.

⚠️ **Disclosure:** during this analysis session, roughly ten report excerpts were printed into an
Anthropic-hosted assistant context in order to understand report structure. That is a small
qualitative read, not bulk label generation — but it is the same clause, and you should know it
happened and decide how to treat it. From here on, all report processing in this project runs
**locally or on Kaggle only**, and no report text goes into any hosted context.

Also open and unruled: whether click-through-agreement datasets (MRNet, OAI, fastMRI+) qualify as
"freely and publicly available". Treat as blocked.

---

## 7. The Efficiency Track is the best risk-adjusted prize

$18,000 across three places, and **far fewer teams will contest it**.

The published metric is

```
Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400
```

minimised, where `Benchmark` = sample_submission AUC (0.5) and `maxAUC` = best private-LB score.
With `maxAUC ≈ 0.95`, the denominator is ≈ −0.45, so

```
Efficiency ≈ −2.22 × AUC + Runtime / 32400
```

**Exchange rate: 0.01 AUC ≙ 720 seconds (12 minutes) of runtime.** Worked example:

| Submission | AUC | Runtime | Efficiency | Winner |
|---|---|---|---|---|
| Fast | 0.900 | 20 min | −1.963 | ✅ |
| Heavy | 0.920 | 3 h | −1.711 | |

The fast model wins decisively. Using the full 9 hours costs 1.0 — equivalent to giving away
0.45 AUC. Strategy: a small model at modest resolution, few slices, no TTA, no ensembling,
predicting in ~15–30 minutes. **A single submission can win both tracks**, so this is not a
fork in the road — it is a second lottery ticket on the same work.

---

## 8. Open questions to resolve next

1. On the 58 gold studies, how much does a **severity-thresholded** extractor beat a
   **presence** extractor? This is the Tier-1 thesis and it is cheap to test. *Caveat:* n=58 is
   too small to resolve 0.01-level differences, and the gold set appears **~2× enriched** in
   prevalence relative to the corpus, so treat it as a directional check, not a scoreboard.
2. What is the true prevalence of each label? The 58 are enriched and every one of them has ≥1
   positive — that is not a random sample.
3. How many studies are bilateral, and does `Laterality` disambiguate all of them?
4. Do any `PatientID`s repeat across studies (fold-leakage risk)?
5. Does `SeriesDescription` free text beat the provided `Fluid_Sensitive`/`Plane` columns for
   slot assignment?

---

## 9. Sources

- [Competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
- [Data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
- [Host: Challenge Overview + label criteria](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
- [Host: report/label inconsistencies Q&A](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)
- [Metadata shortcut probe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517)
- [Rules clarification: external data + LLM APIs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652)
- [Public baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
- [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
