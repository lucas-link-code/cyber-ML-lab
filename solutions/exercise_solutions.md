# Instructor reference / SPOILERS

**Do not read until you have completed the relevant exercise.** The independent challenge is deliberately not resolved in the exercise handout; this file contains diagnostic hints and possible approaches, not a unique guaranteed optimal model.

## 01: Inspect and classify

- Input features: `sender_reputation`, `url_count`, `urgent_language_score`, `has_shortened_url`, `body_length`.
- Target: `phishing_label`. Identifier: `email_id`; never use as a model feature.
- Confusion matrix: rows actual 0/1, columns predicted 0/1. Precision is TP/(TP+FP); recall TP/(TP+FN). `fit()` learns from training rows only; `predict_proba()` reports validation class-1 score.

## 02: Algorithms

- Logistic regression fits a linear surface in transformed input space. Decision trees recursively partition by feature thresholds. Random Forest aggregates many trees. KNN uses distances and needs scaling. SVM learns a margin / nonlinear kernel boundary. Gaussian NB assumes class-conditional feature independence. Gradient boosting fits trees sequentially to reduce predictive loss.
- Same validation split ensures comparisons are like-for-like. Large train/validation gaps call for generalisation checks, not a victory declaration.

## 03: Precision

- Raising the threshold tends to remove low-confidence predicted positives. This typically reduces FP **and** TP. Precision can rise while recall drops. ROC-AUC / average precision remain constant because the same model assigns the same underlying scores.
- Record the specific disappearing FP and TP rows by comparing prediction CSVs keyed on `email_id`.

## 04: Recall

- Decreasing the cutoff generally increases TP and FP. Keep validation FPR under the stated operational ceiling. Compare the FN lists before and after by `event_id`.
- If results remain poor, inspect whether basic EDR features capture behaviours important to missed malicious processes and consider a new feature experiment, with threshold held fixed.

## 05: Regression

- Baseline excludes target drivers such as complexity and severity. Adding them to linear regression helps explain more variance; nonlinear boosting can capture squared complexity and workload interactions. RMSE weights extreme residuals more than MAE.
- Check whether the regression target would be measured only after triage. Historical training needs completed investigations; deployment must use known _input_ fields only.

## 06: Clustering

- Euclidean distances across unscaled byte counts and binary flags are dominated by large numerical ranges. Scaling changes distances and therefore assignments.
- Silhouette measures shape separation, not threat truth. A cluster can contain both benign and anomalous sessions. DBSCAN `-1` means noise, not maliciousness. Two-dimensional PCA drops information.

## 07: Anomalies

- Separate role baselines are appropriate when developer/server activity has a materially different normal range from ordinary users, provided group sample sizes are sufficient.
- `contamination` is a chosen score-quantile alert budget, not the true malicious fraction. Lower contamination usually reduces alerts while increasing FN. Check precision _and_ recall by population and across holdouts.
- Isolation Forest isolates rare observations in random trees; LOF compares local density; One-Class SVM learns a normality boundary; Z-scores require a meaningful reference distribution and are sensitive to skew.

## 08: Independent EDR challenge (read only after trying)

A defensible workflow: write down the starting confusion matrix; inspect at least 10 false positives and 10 false negatives; check whether features omitted by `basic` hold useful behavioural evidence; change **one** factor; compare the exact same challenge-validation events; quantify precision, recall, FPR and generalisation gap. Feature enrichment may help, but test first rather than trusting any fixed suggested recipe.

Remember: this challenge CSV is a separate generated cohort. Validation/test splits within it are host-exclusive. Training a new model on it does **not** prove that a model trained on `edr_events.csv` would generalise. Use `python scripts/evaluate_shift.py` to perform a frozen-old-model cross-cohort assessment.

A complete write-up identifies the business metric, data-quality risks, label availability, feature selection, host/group leakage avoidance, retraining strategy, and once-only final test. The 0.75 / 0.60 / 0.03 thresholds are training goals, not universal SOC operating targets.

## 09: Class imbalance

- Always predicting the majority class gives accuracy close to the benign base rate, but recall for positives is 0. Class weighting increases penalty for misclassifying positives; this may raise false-positive burden. Oversampling is legal only within the training fold.

## 10: Overfitting

- A model can memorise small artifacts in training data. Compare `train_f1` against validation `f1`; adjust capacity conservatively. Shallow trees may underfit. Keep the number of trees constant while testing `max_depth` to isolate cause.

## 11: Leakage

- `post_investigation_decision` contains answer information not available at detection time. It can inflate both training and holdout results. Even a perfect random split cannot prevent target leakage embedded in columns.

## 12: Text / CV

- TF-IDF transforms text into weighted term frequencies relative to corpus occurrence. Learn vocabulary and IDF on training folds only. One-hot encoding converts auth-result categories to binary indicators. GridSearchCV evaluates candidate hyperparameters across training folds; independent validation helps verify the selected candidate.

## 13: Drift

- A standardised mean shift identifies input distribution change, not necessarily model failure. Frozen old-cohort evaluation (`evaluate_shift.py`) isolates generalisation across cohorts better than retraining directly on the new cohort. Precision varies with prevalence; also inspect recall and FPR.

## 14: Neural network

- Backpropagated derivatives come from the chain rule. The sigmoid/BCE output gradient is `p-y`; the tanh derivative is `1-h²`. Smaller learning rates may converge slower; too-large rates can overshoot. XOR requires a nonlinear hidden representation.

## 15: Holdout and deployment

- Choose features, hyperparameters and threshold using training CV and a finite validation protocol. Evaluate the final test once, then record the signed-off configuration. Real deployments need schema validation, safe retraining, calibration monitoring, drift monitoring, labelled feedback and analyst-capacity measurement.

**The point is evidence-based diagnosis. A specific parameter change is not guaranteed to improve all validation metrics, and none of these synthetic scores is a production risk estimate.**
