# EXERCISE NAME: 13 | Distribution shift and feature robustness

**OBJECTIVE:** Detect covariate drift and distinguish in-cohort validation from cross-cohort generalisation.

**SCENARIO:** A new developer tool changes benign process telemetry; a bank wants to know whether an older EDR model still generalises.

**DATASET:** `datasets/edr_events.csv` versus `datasets/edr_events_challenge.csv` (separately generated).

**ALGORITHM:** Feature shift diagnostics and Random Forest classification.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/drift_check.py --dataset edr
python scripts/run_classification.py --dataset edr --dataset-file datasets/edr_events_challenge.csv --algorithm random_forest --features basic --name shift_basic
```

**WHAT THE COMMAND DOES:** First reports per-feature mean shifts standardised by baseline standard deviations, without using class labels. Second trains/validates within the challenge cohort using host-exclusive splits.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: one or more feature distributions move; within-cohort model yields a confusion matrix, but this alone does not measure performance of a previously deployed model.

**WHAT TO OBSERVE:** Ask whether the changed benign developer behaviours are represented in training. A shift in an input distribution is not proof of a new attack.

**INVESTIGATION:** 1. Does within-challenge validation simulate training on old data and deploying on new? 2. Which new telemetry feature seems most useful?

**IMPROVEMENT:** Change **only features** within the challenge cohort: `python scripts/run_classification.py --dataset edr --dataset-file datasets/edr_events_challenge.csv --algorithm random_forest --features extended --name shift_extended`. For an honest old→new assessment, run the separate `scripts/evaluate_shift.py` workflow described in the appendix.

**COMPARISON:** python scripts/compare_models.py shift_basic shift_extended --name compare_shift_features

**LESSON:** Model drift is a measured performance change; data drift is an input distribution change. To attribute deployment degradation, evaluate a frozen model trained on the old cohort against a separate new cohort, not merely a newly trained model on that cohort.

**INTERVIEW CONNECTION:** “The developer environment changed and false positives spiked. How would you detect drift and retest the existing model?”
