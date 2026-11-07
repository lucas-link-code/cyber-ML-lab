# EXERCISE NAME: 12 | TF-IDF features and reliable tuning

**OBJECTIVE:** Add text evidence and see how CV searches parameters using training folds rather than tuning the final test.

**SCENARIO:** Some phishing emails have benign-looking headers but suspicious generated text; a metadata-only classifier may miss them.

**DATASET:** `datasets/phishing_emails.csv`.

**ALGORITHM:** Logistic regression with OneHotEncoder, TF-IDF and GridSearchCV.

**COMMAND:** Run from the lab folder with your virtual environment activated.

```bash
python scripts/run_classification.py --algorithm logistic --features extended --name metadata_only
python scripts/run_classification.py --algorithm logistic --features text --name metadata_plus_text
```

**WHAT THE COMMAND DOES:** The second command adds subject/body text vectorised into sparse-in-concept TF-IDF term features inside a training-only ColumnTransformer pipeline. Both score identical validation rows.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: `PR-AUC` or recall might change; text can help or hurt because generated bodies are repetitive and overlap.

**WHAT TO OBSERVE:** Look at recall, false-positive count, PR-AUC, and whether training/validation gap increases. The TF-IDF vocabulary is learned only from training emails.

**INVESTIGATION:** 1. Why not use a raw string as a numeric regression feature? 2. What happens to feature dimensions when you add TF-IDF?

**IMPROVEMENT:** First compare text added, then **separately** run `python scripts/tune_classification.py --algorithm logistic --search grid --features extended --name metadata_cv`. CV searches `C` and class weights on training folds only; the result is subsequently scored on validation.

**COMPARISON:** python scripts/compare_models.py metadata_only metadata_plus_text metadata_cv --name compare_text_cv

**LESSON:** Feature engineering can change which observations a model can separate. GridSearchCV optimises on folds from training, not the final test; its chosen metric is average precision, which need not maximise F1 at 0.5.
