# Campaign roadmap — 8 Aug → 22 Oct 2026

**Decisions locked (2026-08-08):**

| Decision | Choice |
|---|---|
| Horizon | Full campaign to the 22 Oct final deadline (~75 days) |
| Target | **Both tracks** — one submission that is accurate *and* fast |
| Starting point | **Build our own pipeline**, adopting the public baseline's genuinely-correct ideas |
| Compute | Kaggle only. GPU for training/inference; CPU sessions for I/O; local CPU for text |
| Working style | Reasoning explained before each step — we must be able to defend the method |

That last one is not sentiment. **Winners must submit training code, model weights, a method
description and a video by 5 Nov.** A solution you cannot explain is a solution you cannot claim.

---

## The currency is GPU-hours, not days

~30 GPU-h/week × 10.5 weeks ≈ **315 GPU-hours for the whole campaign.** That is the budget that
actually constrains us. Rough allocation:

| Phase | GPU-h | Share |
|---|---|---|
| Phase 1 — foundations | 20 | 6 % |
| Phase 2 — label quality | 35 | 11 % |
| Phase 3 — vision model | 110 | 35 % |
| Phase 4 — specialisation + efficiency | 70 | 22 % |
| Phase 5 — ensemble + robustness | 50 | 16 % |
| Phase 6 — freeze + deliverables | 15 | 5 % |
| Reserve (failures, reruns) | 15 | 5 % |

Rules of thumb that protect the budget:
- **I/O never touches a GPU.** Cache builds and metadata scans run on CPU sessions — those don't
  draw from the quota at all.
- **Nothing is promoted to a full run until it has won on a subset.** A 1,200-study / 1-fold /
  3-epoch probe costs ~30 min and kills most ideas.
- **Every run logs wall-clock runtime.** It's half the Efficiency metric and cannot be
  reconstructed later.

---

## Phase 0 — Research and decide (8–10 Aug) · **current**

No modelling. Answer the open questions in `RESEARCH_AGENDA.md` so that every later decision rests
on a measured fact rather than an assumption. Two jobs:

1. **Metadata scan** (Kaggle, CPU) — one pass over the DICOM headers, no pixels. Answers bilateral
   studies, patient repeats, site fingerprints, physical-scale distribution, slot assignment, and
   decode cost in a single artifact.
2. **Report structure analysis** (local, CPU) — aggregate statistics only, no text leaves the machine.

**Exit criterion:** we can state, with numbers, what one training example looks like.

## Phase 1 — Foundations (11–17 Aug) · ~20 GPU-h

- Pixel cache: 224 px / 16 slices / 6 slots, two shards, published as Kaggle Datasets.
- **Visual verification** — montages per slot for ~10 studies. Left and right knees must be
  canonicalised to the same orientation, joint centred. Geometry bugs are silent and poison
  everything downstream.
- Presence labeler (the control) + severity labeler (the thesis), both scored on the 58.
- Site-grouped CV harness.
- First end-to-end model and **first submission** — to close the loop, not to compete.

**Exit criterion:** a submission from our own pipeline, and a runtime number.

## Phase 2 — Label quality (18–31 Aug) · ~35 GPU-h

This is where the campaign is won or lost — see `STRATEGY.md` §5A.

- Open-weights multilingual LLM labelling pass on Kaggle (report text never leaves Kaggle → rules-safe).
- Three label sets compared head-to-head under identical training: presence / severity / LLM.
- Disagreement between labelers used as an uncertainty weight rather than forced to a hard label.
- **The diagnostic:** compare grouped-CV score against LB score. A large gap means *labels* are the
  bottleneck; a small gap means *vision capacity* is. This single comparison directs the next 50 days.

**Exit criterion:** the label set is frozen, and we know which bottleneck we're fighting.

## Phase 3 — Vision model (1–21 Sep) · ~110 GPU-h

- Resolution study: 224 vs 336. Nyquist says meniscus and cartilage need 336; effusion and Baker's
  do not. Expect the answer to differ per label.
- Slot fusion: attention pooling over the 6 plane × sequence slots, with the presence mask.
- Per-label heads — the metric is a **macro** average, so a 15 %-prevalence label is worth exactly
  as much as a 60 % one. A single shared loss quietly under-serves the rare ones.
- Backbone comparison; multi-fold; auxiliary text-distillation head (`STRATEGY.md` §5G) if time allows.

**Exit criterion:** a single model that beats 0.891 on our own grouped CV *and* on the LB.

## Phase 4 — Specialisation and efficiency (22 Sep – 8 Oct) · ~70 GPU-h

- Split into label groups by what actually sees them: big-fluid findings (Effusion, Baker's),
  fine-detail findings (Meniscus ×2, OA ×3), marrow findings (Contusion, Fracture), ligaments (ACL, MCL).
- Build the **efficiency submission** in parallel: small backbone, fewer slices, no TTA, target
  15–30 min total runtime. Exchange rate is 0.01 AUC ≙ 12 min, so this is cheap to chase.

**Exit criterion:** two candidate submissions — one maximal, one fast.

## Phase 5 — Ensemble and robustness (9–18 Oct) · ~50 GPU-h

- Ensemble across folds/seeds/resolutions. Weight per label, not globally.
- Robustness checks: unseen scanner holdout, missing-slot studies, bilateral studies, odd slice counts.
- **Resist LB-chasing.** Public LB is ~390 studies; the private 70 % is what pays. Trust grouped CV.

## Phase 6 — Freeze and deliver (19–22 Oct) · ~15 GPU-h

- **15 Oct: entry and team-merger deadline** — nothing to do if solo, but it is a hard gate.
- Select final submissions (accuracy + efficiency).
- Prepare winners' obligations early: training code, weights, method write-up, video, and a public
  model release. Due **5 Nov**, but assembled now while the reasoning is fresh.

---

## Standing risks

| Risk | Mitigation |
|---|---|
| Report text sent to a hosted LLM API | **Possible DQ.** Open weights on Kaggle only. See `STRATEGY.md` §6 |
| Random K-fold instead of site-grouped | +0.053 phantom score, measured. Group on scanner fingerprint |
| Horizontal flip augmentation | Corrupts 5 of 12 labels. Never flip |
| Overfitting the public LB | ~390 studies. Trust grouped CV; spend submissions sparingly |
| Tuning on the 58 gold studies | n far too small, ~2× enriched. Direction check only |
| GPU quota exhausted mid-week | Probe on subsets; keep I/O on CPU sessions |
| Forgetting runtime logs | Forfeits the Efficiency Track. Log every run |
