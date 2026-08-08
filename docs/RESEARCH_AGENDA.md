# Research agenda — what we still don't know, and why each answer changes a decision

Phase 0. No modelling. The point of this document is that **no question is listed here unless its
answer changes something we would otherwise do.** Curiosity is not a reason to run a scan.

Two jobs answer almost all of it:
- **Job A — metadata scan.** One pass over DICOM *headers only* (no pixels) on Kaggle, CPU session.
- **Job B — report statistics.** Local, CPU, aggregate counts only; no report text leaves the machine.

---

## Already answered (8 Aug)

| Question | Answer | Consequence |
|---|---|---|
| How many studies have labels? | **58 of 4,407 (1.3 %)** | The task is weak supervision, not supervised vision |
| Where does ground truth come from? | Two MSK radiologists on **images**, third adjudicates | Reports are a *proxy*, and a biased one |
| Are labels presence or severity? | **Severity-thresholded**, "on the fence → negative" | Motivates graded targets over binary |
| Report languages? | 9+; English 39 %, Turkish 12 %, Spanish 12 %, then Croatian/Greek/German/Bulgarian/French/Dutch | Labeler must be multilingual; no language routing |
| Is `Fluid_Sensitive` distinct from `Fat_Suppression`? | **No — perfectly correlated**, zero off-diagonal | 6 slots (3 planes × 2), not 12 |
| Is there a metadata shortcut? | No. 0.598 AUC site-grouped from headers alone | Don't chase it; but random folds inflate by **0.053** |
| Dataset size? | 569.76 GB / 819,640 files | Kaggle only; cache is mandatory |
| Is `PatientSex` usable? | Absent from `train.csv`, **present in DICOM** | Available at test time via headers |

---

## Job A — metadata scan (Kaggle, CPU session, ~20–30 min)

One pass, header-only reads (`stop_before_pixels=True`). Output: one parquet row per series.

### A1. How many studies are bilateral?
**Why it matters:** the host confirmed a single `StudyInstanceUID` can contain **both knees**, but
labels describe **one** knee. If this is 2 % of studies we handle it with a simple rule; if it is
15 % it needs real logic and becomes a Phase-1 priority.
**How:** count distinct `Laterality` values per study.
**Decision it changes:** whether laterality disambiguation is a footnote or a workstream.

### A2. Is `Laterality` always present?
**Why:** it is our canonicalisation key. Every left knee must be flipped to look like a right knee,
or medial and lateral invert on half the data — and 5 of the 12 labels are side-specific.
**How:** null rate; where missing, can slice geometry (`ImagePositionPatient`) recover it?
**Decision:** whether we need a geometric fallback at all.

### A3. Do `PatientID`s repeat across studies?
**Why:** the same patient in train and validation is leakage, and it inflates CV silently.
**How:** count studies per `PatientID`.
**Decision:** whether folds group on patient as well as on scanner.

### A4. How many distinct sites/scanners, and how large is each?
**Why:** grouped CV is only meaningful if the groups are big enough to hold out. 265 fingerprints
were reported by another team; if the largest site is 40 % of the data, grouped folds get lopsided.
**How:** cluster on `Manufacturer` + `ManufacturerModelName` + `SoftwareVersions` +
`ImagingFrequency` + `ReceiveCoilName`.
**Decision:** the exact fold-splitting scheme, and whether we stratify as well as group.

### A5. What is the real distribution of physical scale?
**Why:** `CROP_MM` is the single most consequential preprocessing constant. Cropping to 130 mm was
the public baseline's choice, justified by the field of view covering 99.6 % of series. We should
verify that against our own data rather than inherit it.
**How:** distribution of `PixelSpacing`, `Rows × PixelSpacing`, `SliceThickness`,
`SpacingBetweenSlices`.
**Decision:** `CROP_MM`, target resolution, and whether slice spacing needs resampling too.

### A6. How many slices per series, and how should we sample them?
**Why:** the cache stores a fixed number of slices per slot. Too few loses the finding; too many
costs quota and runtime. The baseline samples the middle 20–80 % of the stack.
**How:** distribution of slice counts per slot; how much anatomy the outer 20 % actually contains.
**Decision:** slices-per-slot, and the sampling band.

### A7. What transfer syntaxes are present, and how slow are the compressed ones?
**Why:** the data description lists four (uncompressed, JPEG Lossless, JPEG 2000, Implicit VR).
JPEG 2000 decodes far slower than raw. This sets the cache-build time budget *and* the inference
runtime — which is half the Efficiency metric.
**How:** count by syntax; time a decode of each.
**Decision:** whether inference can afford full decode, or needs a cheaper path.

### A8. Does `SeriesDescription` beat the provided CSV columns for slot assignment?
**Why:** `SeriesDescription` is free text like `pd_tse_tra_d` — richer than three columns, and
available at test time. If it disagrees with `Anatomical_Plane`/`Fluid_Sensitive`, one of them is
wrong and we need to know which.
**How:** cross-tab parsed description against the CSV columns; inspect disagreements.
**Decision:** the slot-assignment rule.

---

## Job B — report statistics (local, CPU, minutes)

Aggregate counts only. **No report text is printed or committed** — it is Competition Data.

### B1. How often does severity language actually appear?
**Why:** the entire Tier-1 thesis assumes reports are gradable. If only 30 % of reports contain a
severity qualifier for the relevant finding, the thesis is much weaker than it looks.
**How:** per label, the fraction of positive-mentioning reports that also carry a magnitude cue
(`mild/moderate/severe`, `grade N`, `Outerbridge`, and their equivalents in each language).
**Decision:** whether severity grading is the main bet or a secondary refinement. **This is the
single most important number in Phase 0.**

### B2. Are reports structured or free-form, and does it vary by language?
**Why:** structured reports ("MEDIAL COMPARTMENT: …") parse very differently from prose. If
structure correlates with site, a labeler tuned on English structured reports will silently
underperform on Greek prose — and that becomes a site-correlated bias, which grouped CV will expose
but only after we've wasted the time.
**How:** detect section headers; cross-tab against language.
**Decision:** whether the extractor needs per-format handling.

### B3. Does report length or completeness predict anything?
**Why:** a 40-word report and a 400-word report are different instruments. Short reports likely
omit negative findings entirely, which means "not mentioned" means something different per site.
**How:** length distribution by language; correlation with number of positives extracted.
**Decision:** whether "unmentioned" should map to 0 or to a soft value.

### B4. Presence vs severity extractor on the 58.
**Why:** the direct test of the thesis.
**How:** build both, score per-label AUC and agreement.
**Caveat, stated up front:** n = 58, ~2× enriched, every one has ≥1 positive. It cannot resolve
differences below ~0.02–0.03. **Read direction, not magnitude** — if 9 of 12 labels move the same
way, that is the signal; a single label improving is noise.
**Decision:** which label set Phase 1 trains on.

---

## Deliberately not researching yet

| Question | Why deferred |
|---|---|
| External datasets (MRNet, OAI, fastMRI+) | Rules unresolved — host has not ruled on click-through licences. Blocked, not deprioritised |
| Which backbone is best | Meaningless until labels and geometry are correct |
| Ensembling strategy | Phase 5. Ensembling a wrong pipeline just averages the wrongness |
| True label prevalence | Unknowable directly; falls out of a trusted labeler in Phase 2 |
