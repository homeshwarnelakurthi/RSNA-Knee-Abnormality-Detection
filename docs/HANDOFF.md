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

1. **That the fork scores ~0.942 for us.** Unknown at handoff; the kernel was still running. Check
   the actual public score before treating 0.942 as our floor.
2. **That the +0.136 grouped-vs-random gap transfers to the public ensemble.** It was measured on
   *our* August resnet34, single fold, single seed. Whether the public ensemble's members leak site
   the same way is **unmeasured**. The whole "grouped folds are our edge" thesis rests on this.
3. **That MCL and Medial Meniscus are worth +0.067 macro.** That arithmetic uses *our August
   model's* gold-58 per-target AUCs — n=58, and MCL has 9 positives. **The public ensemble's
   per-target AUCs are completely unknown.** It may already handle these labels well, in which case
   the headroom is elsewhere or much smaller. **Measure the base's per-target AUC before targeting
   anything.**
4. **That the efficiency track is "less contested".** I never opened
   `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`. It is pulled into `public_refs/` — read it.
5. **That grouped-fold discipline is unexploited by others.** `mattiaangeli/knee-mri-fold-weights`
   implies other teams have a fold structure. Not inspected.
6. **That `rsna-knee-cache-build`'s 15.9 GB output is still mountable and still matches the current
   competition data.** The kernel reports COMPLETE, but the output was not listed or mounted this
   session, and the competition data may have been revised since August.
7. **That all public notebooks "confirm" the 336 px Nyquist argument.** A regex matched the string
   `336`; I did not confirm it means `IMG_SIZE` in each notebook. Weak evidence, and in any case
   336 px is now public practice, so it is not an edge.
8. **Whether API-based report labelling is rules-compliant.** `pilkwang/rsna-knee-llm-labels` ships
   an `api_labeler.py`, suggesting public teams used hosted APIs. The August decision was to avoid
   this under Rule 4.b. **This is a rules question for the host or the user, not for me to decide.**
   Using the *published labels* is a different question from calling an API ourselves.

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
