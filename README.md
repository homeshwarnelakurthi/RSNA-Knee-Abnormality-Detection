# RSNA 2026 Knee Abnormality Detection

Kaggle: [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
· 12 binary findings per knee MRI study · macro ROC-AUC · code competition, <=9 h, internet off
· final submission **2026-10-22**

## Start here

| Doc | What it is |
|---|---|
| [docs/STRATEGY.md](docs/STRATEGY.md) | Problem, measured data facts, host intel, ranked ideas. **Read first.** |
| [docs/FINDINGS.md](docs/FINDINGS.md) | **Phase 0 results** — 819 k files scanned; the decisions they lock |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Run log — every result, runtime, and what it changed |
| [docs/ROADMAP.md](docs/ROADMAP.md) | The 75-day campaign, phase by phase, and the GPU-hour budget |
| [docs/RESEARCH_AGENDA.md](docs/RESEARCH_AGENDA.md) | What we still don't know, and which decision each answer changes |
| [docs/PLATFORM.md](docs/PLATFORM.md) | Where to run and why — 570 GB drives the decision |
| [docs/DAY1.md](docs/DAY1.md) | Research and decide. Plus the standing trap list |

**Decisions locked (8 Aug):** full campaign to 22 Oct · target **both** the main and efficiency
tracks · **build our own** pipeline rather than fork the public baseline · Kaggle-only compute.

## The four things that matter

1. **Only 58 of 4,407 training studies carry labels.** The other 4,349 have a radiology report.
   This is a weak-supervision problem wearing a computer-vision costume.
2. **Ground truth is image-derived, not report-derived** — two MSK radiologists plus an
   adjudicator, using explicitly **severity-thresholded** criteria, with "on the fence" graded
   negative. Report-derived labels agree only ~82 %. That gap is systematic, not random.
3. **Random K-fold inflates AUC by ~0.053** through scanner memorisation — and our first model
   showed a **+0.136** grouped-vs-random gap, so the pixels leak site too. Group your folds.
4. **Vision capacity is currently the bottleneck, not labels.** On the 58 gold studies the text
   labeler scores 0.791 and the first vision model 0.674.

## Pipeline

Everything runs Kaggle-to-Kaggle; nothing large passes through a local machine.

```
metadata scan (CPU)  ->  cache build (CPU)  ->  training (T4)  ->  submission
   819k headers          15.9 GB, 4407          mounts the           <=9 h,
   ~6.5 min              studies, ~1 h          build output       internet off
```

Each kernel mounts the previous kernel's output directly, so there is no dataset
upload step between them.

## Layout

```
data/          competition CSVs + sample DICOMs   (gitignored)
eda/           analysis scripts, run locally on CPU
src/           report_labeler.py — the multilingual severity extractor
kaggle/        one folder per Kaggle kernel, each with its kernel-metadata.json
docs/          strategy, findings, roadmap, experiment log
artifacts/     derived data (gitignored — embeds StudyInstanceUIDs)
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

Kaggle CLI must be authenticated (`~/.kaggle/kaggle.json`) and the competition rules accepted.

## Two constraints that bite

**Do not send report text to any hosted LLM API.** Competition Rule 4.b (Data Security) plausibly
forbids it and the host has not ruled. Use open-weights multilingual models locally or inside a
Kaggle notebook. See [docs/STRATEGY.md](docs/STRATEGY.md) section 6.

**Do not select the P100.** Kaggle's current PyTorch ships no Pascal kernels, so the session dies
at the first convolution. Set `"machine_shape": "NvidiaTeslaT4"`.
