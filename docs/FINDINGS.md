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

**Mitigation — use a third, independent source.** Reports routinely state the side in their first
line (`SOL DİZ`, `MR Knie Rechts`, `MRI ΑΡΙΣΤΕΡΟΥ ΓΟΝΑΤΟΣ`). That is available for **all 4,407
training studies**, including every GE study. Plan:

1. Extract side from report text (training only).
2. Use it to validate and, if needed, **calibrate the geometry sign per vendor**.
3. Ship the calibrated geometry rule, which is what runs at test time where no report exists.

This turns a blind spot into a solved problem, and no public notebook appears to do it.

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

## Still open

- Does the geometry sign rule hold for GE/Toshiba/Canon? *Resolved by decision 1's calibration step.*
- Presence vs severity extractor scored on the 58 (**B4** — the direct test, not yet run).
- True label prevalence — falls out of a trusted labeler in Phase 2.
- External datasets — still blocked pending a host ruling.
