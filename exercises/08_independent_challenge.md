# EXERCISE NAME: 08 | Independent EDR challenge: no walkthrough solution

**OBJECTIVE:** Investigate 6,500 unfamiliar endpoint events. Seek validation precision ≥0.75, recall ≥0.60 and FPR ≤0.03; explain tradeoffs.

**SCENARIO:** The SOC says the existing EDR detector generates false positives on developer endpoints and misses malicious PowerShell-like events. You own the next model iteration.

**DATASET:** `datasets/edr_events_challenge.csv` (different synthetic cohort, withheld feature-pattern explanation).

**ALGORITHM:** Choose an appropriate supervised binary classifier. First run the given Random Forest baseline.

**COMMAND:** Run from the lab folder with your virtual environment activated.

```bash
python scripts/inspect_data.py --dataset edr --dataset-file datasets/edr_events_challenge.csv --rows 10
python scripts/run_classification.py --dataset edr --dataset-file datasets/edr_events_challenge.csv --algorithm random_forest --features basic --name independent_baseline
```

**WHAT THE COMMAND DOES:** Loads only the challenge CSV; trains on its train hosts, and prints validation metrics. The **test hosts remain unseen**.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: a confusion matrix with nonzero FP and FN, training F1 greater than validation F1, and one prediction CSV.

**WHAT TO OBSERVE:** Check actual FP/FN feature patterns, label prevalence, host population mix, model capacity and whether your features are informative. Do not use IDs or `malicious_label` as predictors.

**INVESTIGATION:** 1. What is your primary error source hypothesis, backed by five error rows? 2. Which one-variable experiment would test that hypothesis?

**IMPROVEMENT:** Choose one controlled `--features`, `--max-depth`, `--min-samples-leaf`, `--threshold`, `--algorithm` or `--class-weight` change. Save it as a new named experiment and keep a short log of why you chose it. **No answer is provided in this exercise.**

**COMPARISON:** python scripts/compare_models.py independent_baseline YOUR_NEW_NAME --name independent_comparison
# Replace YOUR_NEW_NAME with the name you chose for your second run.

**LESSON:** Operational model selection requires an explicit precision/recall/FPR objective, a held-out evaluation set and an explanation of likely generalisation risks. Do not open the instructor solution file until you have an approach.
