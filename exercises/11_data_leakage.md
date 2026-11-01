# EXERCISE NAME: 11 | A suspiciously good model

**OBJECTIVE:** Recognise target leakage and distinguish valid predictive features from post-outcome fields.

**SCENARIO:** A phishing model suddenly achieves near-perfect validation metrics after a new table column was added.

**DATASET:** `datasets/phishing_emails_leakage_challenge.csv` (explicitly labelled deliberate trap).

**ALGORITHM:** Logistic regression using either legitimate or deliberately leaky predictors.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/leakage_demo.py
```

**WHAT THE COMMAND DOES:** The script trains two models on identical splits: one with valid pre-investigation features and one with `post_investigation_decision`, a field derived after the answer becomes known.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: the model using future/outcome information has much higher PR-AUC, precision and recall, but cannot be used at incoming-email inference time.

**WHAT TO OBSERVE:** High holdout results can still be invalid when validation itself contains a leaky after-the-fact feature. Read where each feature is generated and when it becomes available.

**INVESTIGATION:** 1. Which field would be unavailable at inference? 2. Why does a standard random train/test split fail to reveal this problem?

**IMPROVEMENT:** Open `scripts/leakage_demo.py`; find `safe+["post_investigation_decision"]`. As a later-stage coding exercise, remove that **one feature** from the comparison model and observe the score change. Do not add it to production pipelines.

**COMPARISON:** Compare the two printed rows `valid_features_only` and `LEAKY_POST_OUTCOME`. The honest result is the one that can be computed before analyst investigation.

**LESSON:** Very high scores may indicate leakage, duplication, schema errors or a trivially predictive signal. Data lineage and time-of-availability matter more than a leaderboard score.
