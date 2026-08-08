# RSNA 2026 Knee Abnormality Detection

Kaggle: [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
· 12 binary findings per knee MRI study · macro ROC-AUC · code competition, ≤9 h, internet off
· final submission **2026-10-22**

## Start here

| Doc | What it is |
|---|---|
| [docs/STRATEGY.md](docs/STRATEGY.md) | Problem, measured data facts, host intel, ranked ideas. **Read first.** |
| [docs/ROADMAP.md](docs/ROADMAP.md) | The 75-day campaign, phase by phase, and the GPU-hour budget |
| [docs/RESEARCH_AGENDA.md](docs/RESEARCH_AGENDA.md) | What we still don't know, and which decision each answer changes |
| [docs/PLATFORM.md](docs/PLATFORM.md) | Where to run and why — 570 GB drives the decision |
| [docs/DAY1.md](docs/DAY1.md) | Today: research and decide. Plus the standing trap list |

**Decisions locked (8 Aug):** full campaign to 22 Oct · target **both** the main and efficiency
tracks · **build our own** pipeline rather than fork the public baseline · Kaggle-only compute.

## The three things that matter

1. **Only 58 of 4,407 training studies carry labels.** The other 4,349 have a radiology report.
   This is a weak-supervision problem wearing a computer-vision costume.
2. **Ground truth is image-derived, not report-derived** — two MSK radiologists plus an
   adjudicator, using explicitly **severity-thresholded** criteria, with "on the fence" graded
   negative. Report-derived labels agree only ~82 %. That gap is systematic, not random.
3. **Random K-fold inflates AUC by ~0.053** through scanner memorisation. Group your folds.

## Layout

```
data/          competition CSVs + sample DICOMs   (gitignored)
eda/           analysis scripts
  01_labels_reports.py       label availability, prevalence, co-occurrence, series structure
  02_language_and_samples.py language ID over reports
  03_dicom.py                DICOM tag / geometry inspection
  dump_nb.py                 extract source from .ipynb
intel/         notes on public notebooks           (kernels/ gitignored)
docs/          strategy, platform, plan
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

Kaggle CLI must be authenticated (`~/.kaggle/kaggle.json`) and the competition rules accepted.

```bash
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data
```

## Rules constraint

**Do not send report text to any hosted LLM API.** Competition Rule 4.b (Data Security) plausibly
forbids it and the host has not ruled. Use open-weights multilingual models locally or inside a
Kaggle notebook. See [docs/STRATEGY.md](docs/STRATEGY.md) §6.
