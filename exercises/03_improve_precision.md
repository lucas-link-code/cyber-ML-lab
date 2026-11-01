# EXERCISE NAME: 03 | Excessive phishing false positives

**OBJECTIVE:** Raise precision while maintaining recall at or above 0.55 on **validation**.

**SCENARIO:** Legitimate marketing emails trigger too many phishing alerts. Operational review capacity is limited.

**DATASET:** `datasets/phishing_emails.csv`.

**ALGORITHM:** Logistic regression; hold the fitted algorithm and input feature set fixed.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/run_classification.py --threshold 0.30 --name phish_loose
python scripts/inspect_errors.py --experiment phish_loose --type fp --rows 15
```

**WHAT THE COMMAND DOES:** The first command fits logistic regression and applies a loose 0.30 threshold to validation probabilities. The second displays the actual benign emails falsely flagged.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: `FP=70`, `FN=65`, `precision≈0.65`, `recall≈0.68`. Your terminal prints the verified values for your environment.

**WHAT TO OBSERVE:** A 0.30 cutoff alerts on weak evidence. Examine high-urgency legitimate notices and marketing-style features. Lowering the number of FPs usually increases FNs.

**INVESTIGATION:** 1. Which repeated feature patterns appear among FPs? 2. If you raise the threshold, which predicted positives are most likely to disappear?

**IMPROVEMENT:** Change **only threshold**: `python scripts/run_classification.py --threshold 0.50 --name phish_tight`. Later try `--threshold 0.60` and test whether recall still meets the stated limit.

**COMPARISON:** python scripts/compare_models.py phish_loose phish_tight --name compare_precision_exercise
python scripts/inspect_errors.py --experiment phish_tight --type fn --rows 10

**LESSON:** At a higher threshold, fewer benign emails trigger alerts, so precision often rises; some low-score malicious messages are missed, reducing recall. ROC-AUC and PR-AUC are unchanged because the score ranking was not altered.
