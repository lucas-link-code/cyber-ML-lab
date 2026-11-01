# EXERCISE NAME: 15 | Model selection, held-out assessment and deployment thinking

**OBJECTIVE:** Explain the handoff between model development and inference, and confirm a selected model once on final test.

**SCENARIO:** A detection engineering team has several validation experiments and needs a defensible candidate for limited pilot deployment.

**DATASET:** `datasets/phishing_emails.csv`.

**ALGORITHM:** Your selected Logistic Regression / Random Forest classifier; classification holdout protocol.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/run_classification.py --algorithm logistic --features extended --name candidate_validation
```

**WHAT THE COMMAND DOES:** Regenerates one candidate on validation; records features and operating threshold in JSON. This is the last routine validation check before candidate selection.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: valid precision/recall and a confusion matrix. **No test metrics** are printed until you explicitly add `--evaluate-test`.

**WHAT TO OBSERVE:** Use the outputs to document feature availability, class balance, false-negative severity, and the alert volume implied by the threshold. Review metric variance and group/time effects.

**INVESTIGATION:** 1. Which metric should determine launch based on SOC costs? 2. How would you monitor false-positive drift and delayed labels after deployment?

**IMPROVEMENT:** **Only after choosing a candidate**, add one flag: `python scripts/run_classification.py --algorithm logistic --features extended --evaluate-test --name candidate_final_test`. Do not select another threshold based on this result.

**COMPARISON:** The validation file and final-test file have different rows and must **not** be compared using `compare_models.py`. The final test estimates generalisation; record it as an independently selected-model result, not a tuning step.

**LESSON:** Batch inference scores a finite CSV; online inference handles incoming events in near real time. Monitoring requires timestamped inputs, schema checks, feature availability, delayed ground-truth feedback, drift alarms and approved retraining gates. A changed threat distribution may require retraining, not merely threshold adjustment.

**INTERVIEW CONNECTION:** “How do you move a phishing classifier from a notebook into an operational SOC while preventing leakage, drift and alert overload?”
