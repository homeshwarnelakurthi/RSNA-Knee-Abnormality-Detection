# Phase 0 findings — measured, 8 Aug 2026

Job A: header-only scan of **819,078 files / 24,371 series / 4,407 studies** on Kaggle CPU.
6.5 minutes, **zero read errors**. Job B: report statistics locally, aggregates only.

Every number here is measured. Where a measurement contradicts the competition documentation or
something we assumed earlier, that is called out.

---

## 1. Laterality — the biggest correction

**Half the dataset does not say which knee it is.**

| | |
|---|---|
| Series with no `Laterality` tag | **12,367 of 24,371 (50.7 %)** |
| Studies with the tag on ≥1 series | 2,204 (50.0 %) |
| **Studies with no laterality anywhere** | **2,203 (50.0 %)** |
| Bilateral studies (≥2 distinct sides) | **25 (0.57 %)** |

Two corrections to earlier statements:

- The in-kernel report said 20.9 % missing. That counted only `NaN` and missed **blank strings**,
  which are also missing. The true figure is 50.7 %.
- Bilateral studies are **25, not 83**. The 83 came from treating blanks as a distinct value.
  At 0.57 % this is a footnote, not a workstream — earlier flagged as a possible priority, it isn't.

Missingness is strongly site-linked: German, Dutch and French reports carry the tag ~100 % of the
time; English 38 %, Turkish 35 %, Greek 37 %.

### Geometry recovers the side — with a vendor-shaped hole

Computing the image-centre x in patient coordinates (`IPP[0] + ½·cols·ps·IOP[0] + ½·rows·ps·IOP[3]`;
DICOM LPS, so **+x = patient Left**), validated against the 12,000 series that *do* carry the tag:

| Threshold | Coverage | Sign-rule accuracy |
|---|---|---|
| \|x\| ≥ 0 mm | 100 % | 97.4 % |
| \|x\| ≥ 10 mm | 98.4 % | 98.3 % |
| **\|x\| ≥ 20 mm** | **97.3 %** | **98.5 %** |
| \|x\| ≥ 40 mm | 93.0 % | 98.7 % |

Left knees centre at **+83 mm**, right at **−79 mm**. Clean separation.

**But look at accuracy by vendor:**

| Vendor | n | Accuracy |
|---|---|---|
| SIEMENS (all spellings) | 8,562 | 98.9–99.5 % |
| Philips (all spellings) | 3,039 | 97.0–98.3 % |
| GEHC | 40 | **50.0 %** |
| FUJIFILM | 27 | 85.2 % |
| Hitachi | 8 | **25.0 %** |

**`GE MEDICAL SYSTEMS` (4,914 series) does not appear at all** — none of its series carry the tag.
So we validated the rule almost entirely on Siemens and Philips, and must *apply* it to GE, Toshiba
and Canon where we have no ground truth. The tiny GEHC and Hitachi samples hint the convention may
differ there.

**Mitigation — use a third, independent source.** Reports state the side in their first line
(`SOL DİZ`, `MR Knie Rechts`, `MRI ΑΡΙΣΤΕΡΟΥ ΓΟΝΑΤΟΣ`), available for training studies including
GE ones. Built and run (`eda/07_laterality_validation.py`).

### RESOLVED — and the GE alarm was a small-sample artifact

The report-side extractor is **98.8 % accurate** where it fires (903 studies overlapping the tag),
and correct on every vendor including GE (100 %, n=37). It covers only 37.2 % of studies, but that
is enough to audit geometry.

**Geometry vs report-derived side, per vendor:**

| Vendor | n | Agreement | Wilson 95 % CI |
|---|---|---|---|
| SIEMENS | 818 | **98.9 %** | [97.9, 99.4] |
| PHILIPS | 184 | **96.2 %** | [92.4, 98.1] |
| **GE** | **485** | **92.6 %** | **[89.9, 94.6]** |
| TOSHIBA | 126 | 87.3 % | [80.4, 92.0] |
| FUJIFILM | 16 | 81.2 % | [56.9, 93.4] |
| HITACHI | 8 | 37.5 % | [13.7, 69.4] |
| **Overall** | **1,638** | **95.4 %** | |

**The earlier "GE = 50 %" reading was noise from n=40** on the `GEHC` vendor string, which is a
different, tiny population from `GE MEDICAL SYSTEMS`. With n=485, GE geometry is **92.6 %** —
weaker than Siemens but entirely usable. Hitachi does look inverted, but n=8 with a CI spanning
13–69 % does not support acting on it, and 8 studies cannot move the metric.

**Shipped rule: DICOM tag when present, else geometry. Coverage 100 %.** Where both exist they
agree 97.1 % (n=2,203). Expected end-to-end accuracy ≈ 97–98 %.

For *training* we can do better, using tag → report → geometry in that order, and down-weighting
the studies where geometry and report disagree.

---

## 2. Patient identity — one worry removed

**4,407 distinct `PatientID`s for 4,407 studies. Zero repeats.**

Folds do not need to group on patient. *Caveat:* IDs are pseudonymous and may have been
re-randomised per study during de-identification, which would hide genuine repeats. We cannot
verify this, and grouping by site partially covers the risk anyway.

---

## 3. Cross-validation grouping — the fingerprint idea was wrong

The 5-part scanner fingerprint produces **3,262 groups, of which 2,668 are singletons.** Useless
for folds. The cause is `ImagingFrequency`, which varies per *scan* (63.685238 vs 63.685256), not
per scanner.

Candidate schemes over 4,407 studies:

| Scheme | Groups | Largest | Singletons | Groups ≥50 |
|---|---|---|---|---|
| manufacturer | 7 | 44.3 % | 0 | 4 |
| manu\|model | 46 | 16.8 % | 8 | 21 |
| language | 10 | 39.4 % | 1 | 9 |
| lang\|manu | 24 | 13.4 % | 2 | 16 |
| **lang\|manu\|model** | **75** | **5.8 %** | 15 | **27** |

**Language is an excellent site proxy** — Cyrillic is 100 % Philips, Dutch/German/Greek are 100 %
Siemens. Those are single institutions showing through.

**Decision: group on `language | manufacturer | model`.** 75 groups, no group above 5.8 %,
27 groups with ≥50 studies. It is the closest thing to an institution key we can build.

### But do not treat grouped CV as the truth either

Train and test are drawn from **the same 16 sites**. So the test set is *not* an unseen-site
holdout, and site-grouped CV is therefore **pessimistic**, while random CV is optimistic. Neither
is the target.

**Report both every run.** Grouped CV is the primary decision metric because it does not reward
site memorisation; random CV is the optimistic bound; and **the gap between them measures how much
the model is leaning on site rather than anatomy.** That gap is a diagnostic worth watching, given
the host's explicit warning that prevalence differs between train, public and private splits.

---

## 4. Physical scale — `CROP_MM = 130` independently confirmed

| Field | p1 | p25 | p50 | p75 | p99 |
|---|---|---|---|---|---|
| PixelSpacing (mm) | 0.137 | 0.250 | 0.312 | 0.391 | 0.703 |
| Rows | 256 | 384 | 512 | 640 | 1024 |
| SliceThickness (mm) | 0.6 | 3.0 | 3.0 | 3.5 | 4.5 |
| SpacingBetweenSlices (mm) | 1.0 | 3.3 | 3.5 | 4.10 | 6.20 |
| **Field of view (mm)** | **130** | 160 | 160 | 170 | 205 |

**PixelSpacing spread is 5.14×**, not the 3.4× cited by the public baseline. Physical-scale
normalisation matters even more than advertised.

Coverage by crop size: **≥130 mm covers 99.57 %** of series; ≥140 mm covers only 94.91 %.
So 130 mm sits exactly at the knee of the curve. We reached the public baseline's constant
independently, from our own measurement.

Resulting pixel pitch at a 130 mm crop:

| Target | mm/px | Resolves a 1 mm tear? |
|---|---|---|
| 224 px | 0.580 | ✗ (Nyquist needs ≤0.5) |
| **336 px** | **0.387** | ✓ |
| 448 px | 0.290 | ✓, at 1.8× the cost of 336 |

Median native spacing is 0.312 mm, so 448 px is roughly native and 336 px is a mild downsample.
**Cache at 336 px** — downsampling later is free, upsampling is impossible.

---

## 5. Decode cost — much cheaper than feared

**Every single series is `Explicit VR Little Endian` (uncompressed). 100 %.**

The data description lists four transfer syntaxes including JPEG 2000. In the training data there
is exactly one, and it is the fastest. Measured decode: **5.2 ms/slice**.

| Task | Slices | Single-thread | ~4 processes |
|---|---|---|---|
| Full train cache (6 slots × 16) | ~423,000 | ~37 min | **~10–15 min** |
| Test inference (1,300 studies) | ~125,000 | ~11 min | **~3–4 min** |

Two consequences:

1. The cache build is a **15-minute CPU job**, not the multi-hour one budgeted in `PLATFORM.md`.
2. **Decoding is nearly free at inference.** For the Efficiency Track this is decisive — the runtime
   budget can go almost entirely to the model. *Caution:* the hidden test set may contain the
   compressed syntaxes the description mentions, so the inference notebook must still have
   `pylibjpeg` available and handle them.

---

## 6. Slice counts

Median 30 slices/series, but a long tail: p99 = 160, max = 320.

| Slot | n | mean | p50 | p95 | study coverage |
|---|---|---|---|---|---|
| Axial \| FLUID | 4,719 | 39.5 | 32 | 144 | **100.0 %** |
| Sagittal \| STRUCT | 5,197 | 33.7 | 30 | 60 | 96.8 % |
| Coronal \| FLUID | 4,624 | 28.5 | 30 | 38 | 96.4 % |
| Sagittal \| FLUID | 4,667 | 35.5 | 29 | 47 | 94.2 % |
| Coronal \| STRUCT | 3,985 | 28.0 | 30 | 37 | 77.3 % |
| Axial \| STRUCT | 1,179 | 41.0 | 32 | 81 | 19.4 % |

Coronal series are tight (28 ± 6); axial and sagittal carry the tail. Sampling a fixed band of the
ordered stack handles both.

---

## 7. `SeriesDescription` — useful, but not a replacement

- **82.3 % usable**, 11.8 % is the literal placeholder `DummySeriesDesc!`, 5.9 % null.
- **558 studies (12.7 %) have a placeholder on every series.**
- Where parseable, plane agrees with `Anatomical_Plane` **99.7 %** of the time — but only
  **74.4 %** is parseable.
- Fat-sat keywords disagree with `Fat_Suppression` on ~16 % of series, which is expected: the CSV
  flag is a curated "fluid sensitive" judgement, and water-excitation sequences are fat-suppressed
  without saying so.

Placeholder rate is vendor-specific (`Siemens` lowercase variant 100 %, Philips Medical Systems
29.2 %, GE 19.4 %) — so **its presence is a site marker, not a description.**

**Decision: the CSV columns stay primary. `SeriesDescription` is an auxiliary feature only.**

---

## 8. Report language — the graded-target thesis, refined

82.1 % of reports carry severity or grade language somewhere. But per-label clause-level
co-occurrence splits the twelve labels cleanly along **the kind of threshold the rubric uses**:

| Rubric asks | Labels | Severity in same clause |
|---|---|---|
| **"how much?"** | Synovitis 56.5 %, PF OA 46.6 %, Effusion 46.4 %, Medial OA 37.9 %, Baker's 29.9 %, Lateral OA 28.2 % | **28–57 %** |
| **"what kind?"** | Contusion 26.8 %, MCL 23.3 %, Medial Meniscus 22.8 %, ACL 20.5 %, Fracture 16.0 %, Lateral Meniscus 15.4 % | 15–27 % |

This is not a refutation — it is a correction of the extraction design. Magnitude words answer a
magnitude rubric. A meniscal tear is graded by **whether signal reaches the surface**, an ACL tear
by **complete vs partial vs degeneration** — categories, not degrees.

**Two extractors, not one:** magnitude vocabulary for group 1, categorical vocabulary for group 2.

### A site-correlated hazard in the labels

| | Range |
|---|---|
| Reports with section headers | **1.6 % (German) → 99.8 % (Spanish)** |
| Reports under 50 words | 0.6 % (French/Greek) → **33.9 % (Latin-other)**, 28.8 % (Spanish) |

A 30-word report does not list negatives. Mapping "not mentioned" → 0 is right for a long
structured report and wrong for a short one — and because format tracks language, which tracks
site, this injects a **site-correlated bias into the training labels**.

**Decision: make "not mentioned" depend on report completeness.** Confident 0 in long structured
reports; uncertain and down-weighted in short ones.

---

## Decisions locked by these findings

| # | Decision |
|---|---|
| 1 | **Laterality**: geometry sign rule at \|x\| ≥ 20 mm, per-vendor calibrated using report-derived side |
| 2 | **Bilateral**: 25 studies. Simple rule, no dedicated work |
| 3 | **Folds**: group on `language\|manufacturer\|model` (75 groups); report grouped **and** random CV every run |
| 4 | **Patient grouping**: not needed |
| 5 | **CROP_MM = 130**, cache at **336 px** |
| 6 | **Slots**: the 6 plane × sequence slots, with a presence mask |
| 7 | **Slot assignment**: CSV columns primary, `SeriesDescription` auxiliary |
| 8 | **Cache build**: ~15 min on a CPU session, not hours |
| 9 | **Labels**: two extractors — magnitude for 6 labels, categorical for 6 |
| 10 | **Unmentioned findings**: soft and down-weighted in short reports |

---

## 9. B4 — the thesis tested directly

Both extractors built (`src/report_labeler.py`) and scored on the 58 gold studies.
Presence is 0.7443 throughout; only the severity extractor changed.

| Version | Severity macro AUC | Delta | Improved | Bootstrap 95 % CI |
|---|---|---|---|---|
| v1 first build | 0.7691 | +0.0247 | 8/12 | [−0.0103, +0.0594] |
| v2 negation fix | 0.7772 | +0.0328 | 9/12 | [−0.0066, +0.0718] |
| **v3 rank-faithful priors** | **0.7907** | **+0.0464** | **11/12** | **[+0.0081, +0.0818]** |

**At v3 the interval no longer straddles zero — P(delta > 0) = 99.1 %.**

| Rubric family | Presence | Severity v3 | Delta | Improved |
|---|---|---|---|---|
| Magnitude (OA ×3, Effusion, Synovitis, Baker's) | 0.7198 | 0.7845 | **+0.0647** | **6/6** |
| Categorical (ACL, MCL, menisci ×2, Contusion, Fracture) | 0.7688 | 0.7970 | +0.0282 | 5/6 |

### The two bugs, because both are instructive

**v1 → v2. `_score_categorical` tested negation last.** So `"medial meniscus: no tear"` matched
`TEAR` and scored 0.72 — the negation branch was unreachable whenever any pathology word appeared,
which is nearly always. `_score_magnitude` tested it first. **That single ordering difference is
the entire reason the magnitude family gained +0.049 and the categorical family +0.000.** The
categorical idea was never wrong; the code was. Medial Meniscus went −0.020 → +0.067.

**v2 → v3. Uncertainty was being expressed in the wrong variable.** `UNMENTIONED_PRIOR` was set to
P(positive | not mentioned) — 0.18 for Synovitis, reflecting that reports under-report it badly.
That cost **0.121 AUC** on Synovitis. AUC is computed *per label*, so the absolute value carries no
information whatsoever; only rank does. A prior of 0.18 placed every silent study **above** a
report that said "mild synovitis" at 0.15 — inverting the evidence, since a mention is evidence
*for* a finding. Severity now carries the rank estimate and `confidence` carries the uncertainty,
which is what `confidence` was for. Synovitis went 0.518 → 0.676.

### Honest caveat

Three iterations against the same 58 studies, so **+0.0464 is now partly in-sample.** Both changes
were principled — an ordering bug and a rank/uncertainty conflation, not tuned constants — but the
true out-of-sample gain is below +0.046. Treat the *direction* as established and the *magnitude*
as an upper bound.

**MCL remains −0.075 with 9 positives.** Left alone deliberately: anything that "fixes" a label
with 9 positives is fitting noise.

**Mechanism, and why this generalises.** Binary output has 2 distinct values, so AUC cannot rank
within either block — every positive ties with every other positive. The severity extractor emits
6–9 distinct values per label. Granularity *is* the gain, which is why this should transfer to the
model's own outputs rather than being an artifact of these 58 studies.

### The bigger discovery: reports systematically under-report

Comparing extractor output over all 4,407 studies against gold prevalence:

| Label | Gold prevalence | Extractor severity > 0.5 | Mentioned at all |
|---|---|---|---|
| **Synovitis** | 46.6 % | **3.3 %** | 11.9 % |
| **Fracture** | 31.0 % | 4.9 % | 19.9 % |
| **Medial OA** | 25.9 % | 5.2 % | 16.5 % |
| **Baker's** | 20.7 % | 2.7 % | 44.8 % |
| Lateral OA | 19.0 % | 2.7 % | 12.6 % |
| Effusion | 60.3 % | 11.4 % | 83.8 % |

Gold is ~2× enriched, so halve it and the gaps are still large — Synovitis 23 % vs 3.3 %.
**Clinical reports simply do not comment on findings that image readers score.** Synovitis is the
extreme: radiologists annotating images call it in nearly half of knees; reporting radiologists
mention it in one in eight.

For AUC this is survivable — the metric only needs ranking — but it means several labels will be
learned mostly from pixels with very thin text supervision, and it predicts which labels a
text-only approach will be worst at. Worth checking against per-label LB behaviour later.

---

## Still open

- ~~Rebuild the meniscus categorical rules~~ — **done (v2)**. Medial −0.020 → +0.071,
  Lateral −0.063 → +0.022.
- ~~Is Synovitis an image-only label?~~ — **no.** It looked that way because of the prior bug.
  At v3 it scores 0.676 and gains +0.036 over presence. It remains our weakest text signal
  (named in 11.9 % of reports against 46.6 % of gold) and will lean on pixels, but the text
  is not worthless and should not be discarded.
- **MCL** (−0.075, 9 positives) — not fixable at this sample size. Revisit only if the
  leaderboard shows MCL as an outlier.
- True label prevalence — falls out of a trusted labeler in Phase 2.
- External datasets — still blocked pending a host ruling.

## Artifacts

Code on [GitHub](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection);
derived data (which embeds StudyInstanceUIDs, so it stays off GitHub) as the private Kaggle
dataset **`homeshwarrao/rsna-knee-2026-phase0-artifacts`** — `series_meta.csv`,
`weak_labels_v1.csv`, `laterality_sources.csv`, `b4_eval_v1.csv`.
