# Handoff — RSNA Knee Abnormality Detection, as of 2026-10-01

Written for a fresh session. **Everything below is split into VERIFIED (checked this session, with
how) and UNVERIFIED (inference, assumption, or stale).** Do not act on an UNVERIFIED item without
checking it first — several of my inferences this session turned out wrong, listed in §7.

---

## 1. Competition facts — VERIFIED (read from the Kaggle overview/data/evaluation pages 2026-10-01)

- **Metric:** macro-averaged ROC-AUC over 12 targets, `Final Score = (1/12) Σ AUC_i`.
  **Each target is worth exactly 1/12 = 0.0833 of the score, regardless of prevalence.**
- **Targets:** ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
  Synovitis, Baker's, Contusion, Fracture.
- **Submission:** `StudyInstanceUID` + 12 probability columns, file named `submission.csv`.
- **Reports are TRAIN-ONLY.** Quoted from the data page: *"The Report field will not be provided at
  the testing stage."* Reports exist to derive labels for unlabelled training studies.
- **Test set:** ~1,300 studies. Series 20–45 slices (median 30), long tail to a few hundred.
- `train_series.csv` / `test_series.csv` give `Fluid_Sensitive`, `Fat_Suppression`,
  `Anatomical_Plane` (Sagittal/Coronal/Axial) — **available at test time**.
- **DICOM:** mixed transfer syntaxes (uncompressed, JPEG Lossless, JPEG 2000, Implicit VR LE),
  stripped to 86 allowlisted tags. Intensities, orientations, resolutions vary.
- **Prevalence shift is pre-announced by the host:** *"the prevalence of abnormalities is not
  guaranteed to be the same across the training, public leaderboard, and final evaluation
  datasets."* This is the BioCell failure mode stated in advance — see §6.
- **Timeline:** started 2026-07-30 · entry + team-merger deadline **2026-10-15** ·
  **final submission 2026-10-22** · winners' requirements 2026-11-05. All 23:59 UTC.
- **Prizes, $77,000 total.** Main: 1st $9,000, 2nd $7,000, 3rd $6,500, 4th $6,000, 5th $5,500,
  6th–10th $5,000 each. **Efficiency track: $7,000 / $6,000 / $5,000.**
- **Code competition:** ≤9 h CPU and ≤9 h GPU runtime, **internet disabled**, freely/publicly
  available external data and pretrained models allowed.
- **Efficiency score** (minimise):
  `AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400`. Since `Benchmark − maxAUC` is negative, the
  first term rewards higher AUC; the second rewards shorter runtime.
- **Winners must** open-source code and weights, publish a forum link, make a short video, and
  share the final model publicly.
- **Scale:** 23,332 entrants · 5,394 participants · **4,759 teams** · 73,848 submissions.

### Public leaderboard — VERIFIED (Kaggle API, 2026-10-01)

| rank | public AUC |
|---|---|
| 1st | 0.961 |
| 10th | 0.957 |
| 50th | 0.953 |
| 100th | 0.949 |
| 200th | 0.945 |

**1st to 100th spans 0.012.** Extremely tight; small differences move many ranks, and a private
shakeup is very likely given the host's prevalence warning.

### Our account — VERIFIED
- **Competition rules ARE accepted** (`competition_list_files` returns 20 entries).
- **Submissions made: 0.** No leaderboard calibration exists.

---

## 2. Project state — VERIFIED (read from the repo, 2026-10-01)

Repo: `H:\RSNA Knee Abnormality Detection` · git · **last commit `db63fac`, 2026-08-08.**
Dormant ~7.5 weeks (that time went to the BioCell competition).

Docs already written: `STRATEGY.md`, `FINDINGS.md`, `EXPERIMENTS.md`, `ROADMAP.md`,
`RESEARCH_AGENDA.md`, `PLATFORM.md`, `DAY1.md`. **Read `FINDINGS.md` and the tail of
`EXPERIMENTS.md` first.**

### Kaggle kernels — VERIFIED all four exist and report COMPLETE
| kernel | local folder |
|---|---|
| `homeshwarrao/rsna-knee-phase-0-metadata-scan` | `kaggle/metadata_scan` |
| `homeshwarrao/rsna-knee-cache-build` | `kaggle/build_cache_full` |
| `homeshwarrao/rsna-knee-cache-check` | `kaggle/cache_check` |
| `homeshwarrao/rsna-knee-train-v1` | `kaggle/train_v1` |
| `homeshwarrao/rsna-knee-base-fork` | `kaggle/base-fork` (new, this session) |

Dataset: `homeshwarrao/rsna-knee-2026-phase0-artifacts`.

### Findings from August — these are the project's own measurements, not re-verified this session
- **Only 58 of 4,407 training studies carry per-condition labels.** The other 4,349 have a report.
  Weak supervision wearing a computer-vision costume.
- **Ground truth is image-derived, not report-derived** — two MSK radiologists plus an adjudicator,
  explicitly severity-thresholded, "on the fence" graded negative. **Report-derived labels agree
  only ~82%**, and the gap is systematic rather than random.
- **Grouped vs random folds: +0.136 gap on the first model**; random K-fold inflates AUC by ~0.053
  via scanner/site memorisation. Single fold, single seed — see §5 for the caveat.
- **Gold-58 scores:** text labeler macro **0.791**; vision v1 macro **0.674**.
  Per-target: Effusion **0.922** (proof of concept), Medial Meniscus **0.476**, MCL **0.420**
  (both below chance). **MCL has only 9 positives — nothing about it is measurable; do not tune on it.**
- **Resolution / Nyquist:** at `CROP_MM` 130 and 224 px the pixel pitch is 0.58 mm; a 1 mm meniscal
  tear needs ≤0.5 mm. **336 px gives 0.387 mm and clears it.**
- Pipeline shape: metadata scan (CPU, ~6.5 min, 819k headers) → pixel cache (CPU, ~1 h, 15.9 GB,
  4,407 studies) → training (T4) → submission. Each kernel mounts the previous kernel's output.

### Hard constraints recorded in August
- **Do not send report text to a hosted LLM API.** Competition Rule 4.b (Data Security) plausibly
  forbids it and the host has not ruled. Use open-weights models locally or in a Kaggle notebook.
  **Unresolved — see §5.8.**
- **Do not select the P100.** Kaggle's PyTorch ships no Pascal kernels; the session dies at the
  first convolution. Use `"machine_shape": "NvidiaTeslaT4"`.

---

## 3. Public landscape — VERIFIED (Kaggle API, 2026-10-01)

| notebook | votes |
|---|---|
| `pilkwang/rsna-knee-baseline-v1` | 657 |
| `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` | 468 |
| `mattiaangeli/bend-the-knee-to-the-dinosaurs` | 290 |
| `evgendvorkin/rsna-versia-5` | 264 |
| `prvsiyan/rsna-knee-read-the-report-then-the-knee` | 262 |
| `mattiaangeli/bend-the-knee-to-dinov3-ensembled` | 222 |
| `jiweiliu/rsna-knee-fast-2xt4-inference` | 206 |
| `prvsiyan/head-and-shoulders-knees-and-toes` | 201 |
| `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` | 197 |
| `maverickss26/rsna-knee-0942-restructured` | 179 |

All four I pulled reference **336** as an image size. They are **inference-only ensembles** that
mount 13–14 datasets of pretrained weights; the heavy training happened elsewhere.

### `maverickss26/rsna-knee-0942-restructured` composition — VERIFIED from its metadata
T4, GPU on, internet off. 13 dataset sources, 1 model source (`metaresearch/dinov2/PyTorch/small/1`),
and **2 kernel sources: `sofiaanjenje/rsna-knee-e11-train` and `sofiaanjenje/rsna-knee-e13-train`.**

Notable mounted assets:
- `dreaddevelopment/raptor-knee-maxspan`, `-native384`, `-native384dense`
- `mattiaangeli/knee-mri-fold-weights`, `-rsna-knee-coat-resgated-ep10-top3`,
  `-rsna-knee-coatnet-d4-depthzone-swa3-b2`
- `antoinegg1/rsna-knee-e11-diverse-heads-v20`, `-e9-radimagenet-heads-v15`
- `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`
- `marwanmath/resnet-50-radimagenet-marwan`
- `pilkwang/rsna-knee-llm-labels`, `pilkwang/rsna-knee-weights`

**VERIFIED by listing dataset contents:**
- `dreaddevelopment/raptor-knee-native384` = **`raptor_ft_coatnet_v8_full_swa.pt`, 292 MB — a
  trained CoatNet checkpoint, NOT a pixel cache.** (I initially inferred "cache" from the name and
  was wrong.)
- `pilkwang/rsna-knee-llm-labels` = **`api_labeler.py` (13 KB) + `report_labels_v2.csv` (1.0 MB)** —
  public LLM-derived labels for the unlabelled studies, plus the labeler itself.

**VERIFIED pullable:** `sofiaanjenje/rsna-knee-e11-train`, `sofiaanjenje/rsna-knee-e13-train`,
`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`. Pulled into `public_refs/`.

So the community has converged on a **shared ensemble whose training code and weak labels are
public**. That is the realistic base to build from.

---

## 4. In flight right now

**`homeshwarrao/rsna-knee-base-fork`** — an unmodified fork of `rsna-knee-0942-restructured`,
pushed 2026-10-01, running at handoff time. Purpose: a first submission for leaderboard calibration
and a floor. Watcher task was `bxno5oqhp` (that shell is gone in a new session; re-check with
`kaggle kernels status homeshwarrao/rsna-knee-base-fork`).

**Attribution:** this is someone else's public notebook. Fine as a baseline, but it is not our work
and must not be presented as such. Any final submission should be our own.

---

## 5. UNVERIFIED — do not build on these without checking

> **VERIFICATION PASS 2026-10-01 (later same day).** Items 4–8 below have since been checked.
> Resolutions are inline. See §9 for the full pass, including three items from §1–§3 that turned
> out to be wrong.

1. **That the fork scores ~0.942 for us.** **STILL UNVERIFIED.** Never submitted. Note the rank
   gradient measured in §9: 0.942 is rank ~1028, 0.943 is rank ~290. The difference between those
   two thousandths is 738 places, so "~0.942" is not a safe approximation of anything.
2. **That the +0.136 grouped-vs-random gap transfers to the public ensemble.** **STILL
   UNVERIFIED.** Measured only on our August resnet34, single fold, single seed.
3. **That MCL and Medial Meniscus are worth +0.067 macro.** **EFFECTIVELY FALSIFIED for the public
   ensemble, by arithmetic.** The +0.067 is exactly "lift both from our August values to 0.85", so
   the arithmetic is right *for our model*. But macro 0.943 means the twelve AUCs sum to 11.316;
   if the other eleven average 0.96 the worst possible target is 0.756. MCL at 0.420 is
   arithmetically impossible at 0.943 macro. The headroom table must be rebuilt from the base's
   own per-target AUCs, not ours.
4. **That the efficiency track is "less contested".** **FALSIFIED.** The official efficiency
   leaderboard carries **4,547 teams** — essentially everyone who submits is auto-eligible.
   Efficiency ranks 1–10 hold public scores 0.946–0.958 (median 0.954), and the three prizes sit at
   0.958 / 0.957 / 0.952. Runtime does enormous work *at fixed accuracy* — of the 397 teams at
   exactly 0.943, efficiency ranks run from 104 to 4,338 — but an efficiency **prize** still needs
   ~0.95+. It is the same accuracy bar plus a runtime constraint, not an easier path.
5. **That grouped-fold discipline is unexploited by others.** **SUPPORTED, from source.**
   `sofiaanjenje/rsna-knee-e11-train` defines `report_groups()` as a SHA-256 of the *report text*,
   guarded by `if len(np.unique(groups)) < 4000: raise`. With ~4,000 groups over 4,407 studies that
   is deduplication, i.e. effectively random K-fold. It does **not** control for site.
   `mattiaangeli/knee-mri-fold-weights` exists (`m_f0.pt` …) but its grouping scheme was **not**
   inspected — the name proves nothing.
6. **That the August cache still mounts and matches current competition data.** **VERIFIED.**
   `rsna-knee-cache-check` re-run 2026-10-01: mounts at
   `/kaggle/input/notebooks/homeshwarrao/rsna-knee-cache-build`, shape `(4407, 6, 12, 224, 224)`
   uint8, 15.92 GB, slot presence identical to August, 0 slots present-but-blank, 0 absent-but-
   filled over 400 sampled studies, weak labels join 100%. Separately, `train.csv`,
   `train_series.csv`, `test.csv`, `test_series.csv`, `sample_submission.csv` and one sampled
   DICOM are **byte-identical (md5)** to the August copies — the 2026-09-19 file dates are a
   re-index, not a revision.
7. **That all public notebooks "confirm" the 336 px Nyquist argument.** **STILL UNVERIFIED** and
   still not an edge.
8. **Whether API-based report labelling is rules-compliant.** **RESOLVED — IT IS PERMITTED.**
   The host ruled in discussion 733965: *"submitting Competition Data, including report text, to an
   external LLM or API for inference or other computational processing (for example, extracting
   labels from reports) will not, by itself, be considered prohibited PRIVATE SHARING."* He later
   confirmed directly: *"You can use LLM API, such as those from OpenAI, to read the reports to
   generate the labels."* The PRIVATE SHARING clause targets sharing with other participants or
   teams, not API inference. **The August constraint was wrong and is withdrawn.**

---

## 6. Carry-over from BioCell — the lessons that cost 395 places

Full versions in `H:\BioCell\docs\PLAYBOOK.md`, `TOP_SOLUTIONS.md`, `MISTAKES.md`.
BioCell finished **468/3950** (private 0.91887) after sitting ~73rd on public.

The three that apply directly here:

1. **Build the metric headroom table before modelling.** Macro-AUC over 12 equal targets means a
   rare finding with low AUC is worth exactly as much as a common one. In BioCell I treated a
   *weight* as a *ceiling* and ignored the term holding +0.082. **Here: measure per-target AUC on
   the base, then allocate.**
2. **A noisy instrument may rank candidates but must never eliminate mechanisms.** In BioCell we
   discarded W020 (public −0.014) which scored **best** on private. Here the host has explicitly
   warned that prevalence differs between public and private, so the public LB is *known* to be a
   biased instrument. Keep rejected-but-different candidates alive for final selection.
3. **Choose final submissions to maximise dissimilarity, not public score.** Our two BioCell finals
   shared ~98% of the pipeline and hedged nothing.

Plus: **look at the raw data with your eyes.** The BioCell winner's entire margin came from
hand-annotating cells and noticing divisions are visually obvious beforehand. We never opened an
image. Here that means actually viewing knee MRI series, not just AUC tables.

---

## 7. Corrections I made during this session

Recorded because the pattern matters more than the items:

- I said the **rules might not be accepted** (0 submissions). They are accepted; the empty list
  meant only what it said.
- I said the `raptor-knee-*` datasets were **prebuilt pixel caches** that would "skip the cache
  build". They are **trained model checkpoints**. Pure name-inference, and wrong.
- I said 0.942 was **"already ~200th"**. 200th is 0.945, so 0.942 is *below* the top 200.
- I presented the August decision to **"build our own pipeline rather than fork"** as settled. It
  was settled under a 75-day assumption that no longer holds with 21 days.
- I nearly recorded that the August cache/training kernels were **gone**, based on their absence
  from one page of `kernels_list`. Checking each directly showed all four COMPLETE.

The common thread, and the instruction that triggered this document: **read the source, list the
contents, check the status — do not infer from names, absences, or plausibility.**

---

## 8. Suggested first actions in the new session

1. `kaggle kernels status homeshwarrao/rsna-knee-base-fork` → get the real public score. Submit it
   if it completed, purely for calibration.
2. **Measure per-target AUC of the base** under a grouped split on the 58 gold studies, and against
   `pilkwang/rsna-knee-llm-labels` on a held-out group. This produces the real headroom table and
   replaces every number in §5.3.
3. Read `public_refs/rsna-knee-abnormalities-efficiency-lb` → decide whether the efficiency track is
   genuinely the better prize target.
4. Read `public_refs/rsna-knee-e11-train` and `e13-train` → understand how the public ensemble
   members are trained, and what a *diverse* additional member would look like.
5. Verify the August cache output still mounts (§5.6) before planning any training.
6. Then, and only then, pick the target label(s) and train.

**Open decision for the user:** main leaderboard, efficiency track, or both. I proposed both, with
efficiency as the realistic prize target, since a single submission can be eligible for each — but
that was argued from an unverified assumption about how contested the efficiency track is (§5.4).

---

## 9. Verification pass — 2026-10-01 (second session)

Every claim in §1–§5 was re-checked against the live source. Method is named for each.

### Confirmed exact (live Kaggle pages + API)

Metric and the 12 targets; submission format; all prizes (main 9/7/6.5/6/5.5/5×5k, efficiency
7/6/5k); timeline (entry 15 Oct, final **22 Oct**, winners 5 Nov); code-competition limits
(≤9 h, internet disabled, external data allowed); the efficiency formula
`AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400`; both host quotes verbatim — *"The Report field
will not be provided at the testing stage"* and the prevalence-shift notice.

Our account: rules accepted (`userHasEntered=True`), **0 lifetime submissions**, 5 remaining today.
All five kernels report COMPLETE; `rsna-knee-2026-phase0-artifacts` is `ready`.

Repo data: **58 of 4,407** studies carry all 12 labels; MCL has **9** positives; gold prevalence
matches §2 exactly; `run.json` holds gold macro **0.6739** and runtime **1223.5 s**.

`kaggle/base-fork` is **byte-identical in source** (sha256 over all 51 cells) to
`maverickss26/rsna-knee-0942-restructured`. It is an unmodified fork.

`sofiaanjenje/rsna-knee-e11-train` and `e13-train` are genuine training kernels — `EPOCHS = 10`,
`AdamW`, `OneCycleLR`, `.backward()`, `torch.save`, `GroupKFold(5)`. They differ from each other
only in slot ordering (12 diff lines), so they are sibling variants, not distinct approaches. They
also put the 58 gold studies **into** training at `w[gold] = 3.0`; we held ours out.

### Drift since the handoff was written (hours)

Teams 4,759 → **4,765** · entrants 23,332 → **23,341** · submissions 73,848 → **73,905**.

### The rank gradient — the most decision-relevant number found

Computed from the full 4,765-row leaderboard export:

| Public score | Best achievable rank | Teams tied there |
|---|---|---|
| 0.953 | 49 | 10 |
| 0.947 | 120 | 19 |
| 0.945 | 170 | 61 |
| 0.944 | 231 | 59 |
| **0.943** | **290** | **738** |
| **0.942** | **1028** | 175 |
| 0.891 | 3045 | 148 |

**One thousandth of AUC between 0.942 and 0.943 is 738 places.** A top-10 prize needs **0.957** —
`+0.014` over the 0.943 cluster. Treat any statement of the form "we are at ~0.94" as meaningless
without the third decimal.

### The data specification changed since August

Competition **data** is unchanged (md5-identical). The **description** was edited:

- `PatientSex` is gone from the `train.csv` schema — the August discrepancy we logged was a doc bug
  and the host fixed it.
- A new sentence appeared: *"although Fluid_Sensitive and Fat_Suppression are often correlated, as
  observed in the training set, they are not necessarily equivalent for every case."*

Re-verified: in train there are **0 off-diagonal series** (10,361 at `(0,0)`, 14,010 at `(1,1)`),
so the August measurement was right. But the note is plainly aimed at teams who collapsed the two
columns, which we did. Our cache keys on `Fluid_Sensitive` only, so it degrades gracefully, but at
test time a fat-suppressed non-fluid-sensitive series would be filed into a `STRUCT` slot.

### Also ruled by the host (discussion 733965)

External datasets under **non-commercial research licences** (OAI, MRNet, fastMRI+, SKM-TEA) are
*"not prohibited on that basis alone"* — prize money does not make the use commercial. The binding
test is accessibility: click-through registration generally fine; IRB, negotiated agreements or
institution-specific approval may not be. Dataset-specific licence compliance remains the team's
responsibility.

### Corrections to this document itself

- §2 said the last commit was `db63fac` (2026-08-08). There is a later commit, `607385a`, this doc.
- §3 said the public references were "all pulled into `public_refs/`". `public_refs/` holds
  **7 notebooks only**. `pilkwang/rsna-knee-llm-labels` and `dreaddevelopment/raptor-knee-native384`
  both exist on Kaggle — verified by listing their files — but **neither is present locally**.
- EXPERIMENTS.md reports gold macro 0.6739 while the per-target table it prints averages 0.6651.
  Both are correct: `run.json` saved the **best** epoch (9) and the per-target table was printed
  after the **final** epoch (12). Worth fixing in the harness so one run reports one number.

### Two errors made during this pass, recorded per the standing instruction

- A `dump_nb.py` call returned 69 bytes and I nearly concluded the public training notebooks were
  empty. It was a Windows console encoding crash, sent to `/dev/null` by my own redirect. The
  notebooks hold 278,775 characters of source. **Set `PYTHONIOENCODING=utf-8` and never discard
  stderr while verifying.**
- A DICOM comparison reported `*** DIFFERS ***` because I *inferred* the archive path from a local
  flattened copy instead of reading it from `competitions files`. With the real path the file is
  md5-identical.

### Still unverified after this pass

1. What `rsna-knee-base-fork` actually scores. Requires a submission.
2. Whether the +0.136 grouped-vs-random gap transfers to the public ensemble.
3. The public ensemble's per-target AUCs — bounded by arithmetic above, not measured.
4. What `mattiaangeli/knee-mri-fold-weights` groups on.
5. Whether 336 px is load-bearing in the public notebooks.
