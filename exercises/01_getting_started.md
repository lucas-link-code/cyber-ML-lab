# EXERCISE NAME: 01 | First CSV and first phishing classifier

**OBJECTIVE:** Identify features and labels; execute a fitted classifier without editing Python.

**SCENARIO:** You are triaging email alerts and must establish a reproducible first baseline.

**DATASET:** `datasets/phishing_emails.csv` (6,000 fully synthetic messages).

**ALGORITHM:** Logistic regression (binary supervised classification).

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/inspect_data.py --dataset phishing --rows 10
python scripts/run_classification.py --dataset phishing --algorithm logistic --features basic --name exercise01
```

**WHAT THE COMMAND DOES:** First command prints 10 rows, types, missing counts and label prevalence; second takes the five basic input features, splits data, imputes/scales based on the training set, calls `fit()` and predicts probabilities on held-out validation rows.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative format only: `rows=6000`, `phishing_label: 0=... 1=...`; confusion matrix `TN=... FP=... FN=... TP=...`; `precision=0.xx`, `recall=0.xx`. Numbers are computed by your run, not promises.

**WHAT TO OBSERVE:** `email_id` is an identifier; `phishing_label` is the answer; neither belongs in the input features. `positive_probability` is a score, `predicted_label` is the 0.50-threshold decision. Read the actual FP/FN counts in the confusion matrix.

**INVESTIGATION:** 1. Which single column contains the ground-truth answer? 2. How many legitimate emails were flagged, and how many phishing emails were missed?

**IMPROVEMENT:** Open `outputs/exercise01_predictions.csv` and sort/filter by `error_type`. Do not change the model yet. This changes the **inspection view**, not the fit.

**COMPARISON:** Open `outputs/exercise01_metrics.json` and `charts/exercise01_confusion.png`. Check that TP, FP, FN and TN agree with printed precision and recall.

**LESSON:** CSV → chosen features → training-only preprocessing → `fit()` → probabilities → threshold → confusion matrix. Training learns coefficients; a threshold turns a score into an operational alert.

**INTERVIEW CONNECTION:** “What is the target variable? Why did you omit the email ID? Explain this confusion matrix to a SOC lead.”
