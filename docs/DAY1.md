# Day 1 — research and decide

**Superseded framing.** This was originally written as a one-day sprint ending in a submission.
The campaign now runs to 22 Oct (`ROADMAP.md`), and the decision is to **build our own pipeline**.
That changes what Day 1 is for.

**Day 1 is not for getting a score. It is for making sure the first thing we build is the right
thing.** A submission today would come from forking someone else's work, which we explicitly chose
not to do. Foundations start Phase 1 (11 Aug).

---

## Today

| Block | Where | What |
|---|---|---|
| 1 | Kaggle, **CPU session** | **Job A — metadata scan.** Header-only pass over the DICOMs → one parquet of series-level facts. Answers A1–A8 in `RESEARCH_AGENDA.md`. ~20–30 min, costs zero GPU quota |
| 2 | Local, CPU | **Job B — report statistics.** Aggregate counts only. B1 is the priority: *does severity language actually appear often enough for the thesis to hold?* |
| 3 | Local, CPU | **B4 — presence vs severity extractor**, both scored on the 58 |
| 4 | — | Write the decisions down. Numbers, not impressions |

## What we must be able to state by tonight

1. **What one training example looks like** — how many slots, how many slices, what physical crop,
   canonicalised to which side.
2. **How folds are split** — grouped on scanner, on patient, or both.
3. **Whether the severity thesis survives first contact.** If B1 shows severity language is rare, or
   B4 shows no directional improvement across the 12 labels, we say so plainly and fall back to
   presence labels. The thesis is a bet, not a belief.
4. **How much bilateral and missing-laterality handling is actually needed.**

## Explicitly not today

Cache building, model training, submissions, backbone choice, ensembling, external data.
Every one of those is cheaper and better after the four answers above.

---

## The trap list — carry this into every phase

| Trap | Cost | Guard |
|---|---|---|
| Random K-fold instead of site-grouped | **+0.053 phantom AUC** — measured | Group on scanner fingerprint |
| Horizontal flip augmentation | Silently corrupts 5 of 12 labels | No flips. Ever. |
| Ignoring laterality | Medial/lateral inverted on ~half the studies | Read `Laterality`; canonicalise |
| Bilateral studies | Labels describe **one** knee | Host says report/metadata disambiguates |
| Fixed-pixel resize | Anatomy scale varies ~3.4× | Crop to constant mm, then resize |
| Trusting `Fluid_Sensitive` ≠ `Fat_Suppression` | Perfectly correlated — one bit | Use the 6-slot scheme |
| Tuning on the 58 | ~2× enriched, n far too small | Direction check only |
| Report text → hosted LLM API | **Possible DQ** (Rule 4.b) | Open weights, local or on Kaggle |
| Forgetting to log runtime | Forfeits the Efficiency Track | Log it every run |
| Spending GPU quota on file I/O | Burns the scarcest resource | Cache builds run on CPU sessions |
