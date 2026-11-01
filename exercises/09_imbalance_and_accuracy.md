# EXERCISE NAME: 09 | Why accuracy can mislead

**OBJECTIVE:** Explain how a high-accuracy model can entirely miss the minority malicious class.

**SCENARIO:** A SOC dashboard says a detector is >80% accurate but it never catches phishing.

**DATASET:** `datasets/phishing_emails.csv`.

**ALGORITHM:** All-negative majority baseline versus logistic regression.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/majority_baseline.py --dataset phishing
python scripts/run_classification.py --algorithm logistic --features basic --name imbalance_logistic
```

**WHAT THE COMMAND DOES:** First command deliberately predicts every email benign without learning. Second trains the baseline model using labels.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: all-benign `accuracy≈0.83`, `recall=0`, `precision=0`; logistic regression has nonzero TP and FN.

**WHAT TO OBSERVE:** An imbalanced dataset makes TN dominate accuracy. Always inspect TP, FP, FN and TN; add PR-AUC for ranking quality.

**INVESTIGATION:** 1. Is 83% accuracy acceptable if recall is zero? 2. What is the all-benign false-positive rate?

**IMPROVEMENT:** Change **only class-weight**: `python scripts/run_classification.py --algorithm logistic --features basic --class-weight balanced --name imbalance_weighted`. Optionally compare with **training-only** duplication via `--sampling oversample` (a separate experiment).

**COMPARISON:** python scripts/compare_models.py imbalance_logistic imbalance_weighted --name compare_class_imbalance

**LESSON:** Class weighting penalises missed positives more during optimisation; it can increase recall while raising false alarms. Resampling must occur inside training folds, never before splitting.
