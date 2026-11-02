# Cyber ML Lab — Instructor Solutions and Worked Explanations

> **SPOILERS / instructor reference.** Work through an exercise in `exercises/` before opening its corresponding section here. This guide explains *why* an outcome occurs, how to verify it in the generated artifacts, how to assess trade-offs for a SOC, and which conclusions the experiment **does not** justify.
>
> **Scope:** This companion is written for the scripts and CSV schemas in the `cyber-ML-lab` repository (15 numbered exercise handouts, plus the `run_cross_validation.py` extension). It uses the fully synthetic datasets, not real production telemetry. All commands below run from the repository root with the lab virtual environment activated.

## Contents

- [How to use this guide](#how-to-use-this-guide)
- [A. The whole ML workflow](#a-the-whole-ml-workflow)
- [B. Read the outputs and metrics](#b-read-the-outputs-and-metrics)
- [01. Inspect the CSV and train a phishing classifier](#01-inspect-the-csv-and-train-a-phishing-classifier)
- [02. Compare classification algorithms](#02-compare-classification-algorithms)
- [03. Investigate and improve precision](#03-investigate-and-improve-precision)
- [04. Improve EDR recall](#04-improve-edr-recall)
- [05. Predict alert triage time with regression](#05-predict-alert-triage-time-with-regression)
- [06. Unsupervised clustering: K-means, DBSCAN and hierarchical](#06-unsupervised-clustering-k-means-dbscan-and-hierarchical)
- [07. Anomaly detection and endpoint populations](#07-anomaly-detection-and-endpoint-populations)
- [08. Independent EDR challenge: staged hints and evaluation rubric](#08-independent-edr-challenge-staged-hints-and-evaluation-rubric)
- [09. Class imbalance and misleading accuracy](#09-class-imbalance-and-misleading-accuracy)
- [10. Overfitting, tree depth and generalisation](#10-overfitting-tree-depth-and-generalisation)
- [11. Label leakage and feature availability](#11-label-leakage-and-feature-availability)
- [12. TF-IDF, model pipelines and hyperparameter search](#12-tf-idf-model-pipelines-and-hyperparameter-search)
- [13. Data drift versus model drift](#13-data-drift-versus-model-drift)
- [14. Neural networks and backpropagation](#14-neural-networks-and-backpropagation)
- [15. Selecting a model and preparing for deployment](#15-selecting-a-model-and-preparing-for-deployment)
- [16. Five-fold cross-validation: guided extension](#16-five-fold-cross-validation-guided-extension)
- [C. Consolidated experiment results from the guided sessions](#c-consolidated-experiment-results-from-the-guided-sessions)
- [D. Troubleshooting and frequently confused concepts](#d-troubleshooting-and-frequently-confused-concepts)
- [F. Further experiments and interpretation checklist](#f-further-experiments-and-interpretation-checklist)

## How to use this guide

1. Run the command in the relevant exercise handout and save the output with a **unique** `--name`.
2. Look at the confusion matrix, errors, distributions or cluster assignments **before** reading the answer.
3. Read the matching section below and compare the explanation to your own results.
4. Change **one factor at a time**: threshold, feature set, model family, tree depth, `k`, scaling or baseline population.
5. Use `scripts/compare_models.py` for compatible saved experiments; investigate individual records when aggregate metrics hide important mistakes.
6. Treat every policy target in this lab as a **scenario-specific training constraint**, not a universal SOC rule.

### Evidence standard used in this guide

- **Observed (guided session):** a result supplied by a learner's terminal output during the worked sessions; numbers are shown as reported or recomputed directly from TP/FP/TN/FN. These are historical **validation/exploratory** results, not a promise that every installation will reproduce every decimal.
- **Verified package reference:** a result documented in the repository's `VERIFICATION.md` from the original smoke-test/reference runs.
- **Derived:** arithmetic computed from reported counts, clearly indicated where helpful.
- **Illustrative:** a hypothetical example that must not be confused with a measurement.
- **To measure:** the exercise has not supplied a measured result; run it rather than assume an improvement.

The source code and dependencies in your checkout are authoritative. For a mismatch, check `python scripts/<script_name>.py --help`, the saved experiment configuration in `outputs/<name>_metrics.json`, and whether the CSVs were regenerated or altered. A fixed random seed improves reproducibility; exact floating-point identity across different software/hardware versions is not guaranteed.

## A. The whole ML workflow

A useful mental model, shared by all four problem types:

```text
Synthetic CSV records
        |
        v
Understand the question: class, number, group, or anomaly?
        |
        v
Select valid features and (where supervised) the target
        |
        v
Prepare the data: impute missing values, encode, scale / vectorise
        |
        v
Fit the algorithm on permitted training data
        |
        v
Generate scores, predictions or cluster assignments
        |
        v
Evaluate using appropriate measures and inspect individual errors
        |
        v
Make one controlled change, compare, and explain the trade-off
        |
        v
After model selection: final untouched holdout / deployment-style test
```

### Four ML problem types in security

| Problem | What the algorithm learns/produces | Security example | Ground truth during fitting? |
| --- | --- | --- | --- |
| Classification | A decision rule and usually a score for discrete classes | Malicious vs legitimate process | **Yes** |
| Regression | A prediction of a continuous value | Alert investigation time in minutes | **Yes** |
| Clustering | Groups of observations with similar feature-space geometry | Similar network sessions | **No** |
| Anomaly detection | A measure of deviation from learned normality or density | Rare outbound connection patterns | **No labels** for the unsupervised estimator |

In this repo, classification and regression maintain disjoint training, validation and final-test partitions. EDR and network **supervised/anomaly** splits are host-exclusive, reducing the risk that nearly identical observations from one host appear on both sides. Clustering deliberately uses the entire unlabeled network set for exploration and checks `known_anomaly_label` only **after** fitting. These are different experimental protocols; do not treat clustering's post-hoc table as an independent test of an already selected detection rule.

### What happens inside `fit()`, `predict_proba()` and a threshold

For a supervised binary classifier, the underlying operations are conceptually:

```python
model.fit(X_train, y_train)                 # Learns coefficients or tree structure
scores = model.predict_proba(X_val)[:, 1]  # Produces class-1 score estimates
pred = (scores >= 0.35).astype(int)         # Converts scores into alert / no alert
```

**Important practical point:** the current `run_classification.py` performs **all three operations every time you invoke it**. It does not expose separate train-only and inference-only CLI modes. Nevertheless, running the same command with identical dataset, split, preprocessing, model settings and seed, and changing **only** `--threshold`, gives a controlled comparison: the fitting step is repeated in the same way, the learned model should be equivalent, and the threshold changes which scores generate alerts. The threshold is **not** passed to the Random Forest or Logistic Regression fitting procedure.

For Logistic Regression, fitting learns coefficients and an intercept. For a Decision Tree or Random Forest, fitting learns feature splits and leaf values rather than a simple vector of weights. KNN stores examples; an RBF SVM learns support vectors and a nonlinear boundary. **Training F1 can change with a threshold** because the lab applies the threshold when computing training predictions too. An increase in training F1 after lowering a threshold is not evidence that the fit learned better parameters.

The repository writes human-readable model details to `outputs/<name>_weights.txt` for classifiers. In Logistic Regression, weights apply to the **preprocessed** feature matrix (scaled numeric columns and one-hot-encoded categories), not directly to original raw CSV units. For Random Forest, inspect trees and impurity-based importances, not a supposed logistic weight vector.

## B. Read the outputs and metrics

### The confusion matrix and SOC interpretation

For binary detection, class 1 means the model raises an alert; class 0 means it does not. Compare model predictions with the dataset's known label:

| | Predicted benign (0) | Predicted malicious (1) |
| --- | ---: | ---: |
| **Actually benign (0)** | TN | **FP: false alert** |
| **Actually malicious (1)** | **FN: missed threat** | TP: correct alert |

- **Total model alerts** = TP + FP.
- **Precision** = TP / (TP + FP). *Of alerts raised, what fraction is genuinely malicious?* It measures **alert quality**.
- **Recall / sensitivity / TPR** = TP / (TP + FN). *Of genuinely malicious items, what fraction did we detect?* It measures **coverage**.
- **False-positive rate (FPR)** = FP / (FP + TN). *Of genuinely benign items, what fraction generated a false alert?*
- **Accuracy** = (TP + TN) / (TP + FP + TN + FN). Can be misleading when benign observations dominate.
- **F1** = 2 × precision × recall / (precision + recall), or equivalently 2TP / (2TP + FP + FN). It rewards balance between alert quality and coverage, but ignores TN and operational severity.
- **Predicted-positive rate** = (TP + FP) / total events. This estimates the alert volume implied by the threshold on that sample.

**Example (illustrative):** a classifier raises 100 alerts; 80 are true positives and 20 are false positives. Its precision is **80%**. If there were 200 real malicious cases, recall is **40%**, despite that seemingly good precision. A model that raises 100 correct alerts and zero false alarms has 100% precision, but can still have poor recall if it missed many malicious cases.

**ROC-AUC** measures ranking quality over all possible thresholds by summarising TPR versus FPR. **`pr_auc_average_precision`** in this lab comes from scikit-learn's `average_precision_score`; it is **average precision (AP)**, a particular summary of precision-recall ranking, not necessarily the trapezoidal geometric area under the curve. AP and ROC-AUC **do not change** when only the alert threshold changes and the underlying probability scores stay the same. This is an important diagnostic check in our experiments.

Neither a high ROC-AUC nor a high AP guarantees a good result at a specific threshold; the operating point and costs matter. Class-1 `predict_proba` values also need calibration checks before interpreting them as real-world probabilities.

### Where the lab writes evidence

```text
outputs/<name>_metrics.json           Configuration + metrics for one experiment
outputs/<name>_predictions.csv        Per-record predictions, scores and source fields
outputs/<name>_weights.txt            Classification model coefficients/trees/etc.
charts/<name>_*.png                   Confusion matrix, ROC/PR curves, PCA, residuals...
outputs/<comparison_name>.csv         Comparison table from compare_models.py
```

For supervised classifiers, the prediction file includes `actual_label`, `predicted_label`, `positive_probability`, and `error_type` (`FP`, `FN`, or `correct`), as well as original event fields. The original label remains in the file for analysis. To see **exact** false positives or false negatives without guessing CSV column names:

```bash
python scripts/inspect_errors.py --experiment exercise02_extended --type fp --rows 20
python scripts/inspect_errors.py --experiment edr_threshold035 --type fn --rows 20
```

This script **does not retrain** the model. It reads an existing prediction file. It also supports `--type all_errors`, `--type largest_residuals` (regression) and `--type top_alerts` (anomaly detection).

To compare experiments:

```bash
python scripts/compare_models.py exercise01 exercise01_lower_threshold --name compare_thresholds
```

The comparison script reads saved JSON records, not live fitted models. It refuses comparisons across a different problem/dataset/evaluation split. **Do not use it to draw a direct winner from a validation cohort and a different final-test cohort**, or from two different challenge CSVs. Preserve the same `--name` outputs by using distinct experiment names; repeating a name overwrites its metrics and predictions.

---

## 01. Inspect the CSV and train a phishing classifier

**Exercise:** `exercises/01_getting_started.md`  
**Question:** Can you identify the features, the target, and the model's first false alerts and misses?

```bash
python scripts/inspect_data.py --dataset phishing --rows 10
python scripts/run_classification.py --dataset phishing --algorithm logistic --features basic --name exercise01
```

`pandas` reads 6,000 synthetic email records. `email_id` is a key, not evidence of phishing; `phishing_label` is the target, not an input. The five basic features are `sender_reputation`, `url_count`, `urgent_language_score`, `has_shortened_url`, and `body_length`. The pipeline imputes missing numeric data and standardises using **training** statistics only. Logistic Regression then learns the relationship between the five features and the binary target.

### Observed worked answer (validation, 1,200 emails)

| Item | Count |
| --- | ---: |
| True negatives | 978 |
| False positives | 21 |
| False negatives | 89 |
| True positives | 112 |
| Known phishing emails | 201 |

From those counts, precision = 112 / 133 = **84.21%**; recall = 112 / 201 = **55.72%**; F1 = 224 / 334 = **67.07%**; accuracy = 1,090 / 1,200 = **90.83%**.

**Interpretation:** alerts are relatively reliable, but 89 of 201 phishing emails are missed. The apparently strong accuracy largely reflects 999 legitimate messages. We should not deploy based on accuracy alone. First establish the acceptable FN and FP costs for the SOC.

**Check the answer yourself:**

```bash
python scripts/inspect_errors.py --experiment exercise01 --type fp --rows 10
python scripts/inspect_errors.py --experiment exercise01 --type fn --rows 10
```

**Why the model makes errors:** the five features have imperfect, overlapping distributions. Legitimate marketing and administrative messages may have urgent wording, links and unusual sender reputation; some phishing may look less suspicious in those same dimensions. This is not necessarily a bug: noisy and overlapping behaviours are intentional properties of the synthetic dataset.

---

## 02. Compare classification algorithms

**Exercise:** `exercises/02_compare_classifiers.md`  
**Question:** Does a more complex algorithm necessarily improve validation detection?

```bash
python scripts/run_classification.py --algorithm logistic --features basic --name alg_logistic
python scripts/run_classification.py --algorithm tree --features basic --name alg_tree
python scripts/run_classification.py --algorithm random_forest --features basic --name alg_forest
python scripts/compare_models.py alg_logistic alg_tree alg_forest --name compare_classifier_families
```

All three are trained using the same dataset and feature whitelist. The distinction is **what function the model learns**, not a change in ground truth.

| Algorithm | Mental model | Main strengths | Main caveat |
| --- | --- | --- | --- |
| Logistic Regression | Learns a weighted linear log-odds score, converted through sigmoid into class probability | Efficient, interpretable weights; often a good baseline | Needs informative features; decision boundary linear in transformed space |
| Decision Tree | Repeated feature-threshold decisions, from root to leaf | Explains individual decision paths; handles nonlinearity | Deep trees can overfit noisy observations |
| Random Forest | Many trees trained with bootstrap samples and random feature selection; aggregate scores | Captures interactions; reduces variance versus one tree | Can overfit; less directly interpretable |
| Gaussian Naive Bayes | Fits class-specific feature distributions under a conditional-independence assumption | Fast probabilistic baseline | Correlated security features can violate assumptions |
| KNN | Uses nearby stored training observations to vote | Intuitive similarity model | Sensitive to scaling, high dimensions, and inference cost |
| RBF SVM | Finds a separating margin in a nonlinear transformed space | Useful complex decision boundary | Sensitive to scaling/hyperparameters; `probability=True` adds score fitting overhead |
| Gradient Boosting | Builds correction trees sequentially to reduce loss | Models nonlinear interactions; often strong tabular ranking | Can overfit and needs tuning; score calibration may differ |
| XGBoost (optional) | Regularised/optimised boosted trees | Additional control over complexity and training | Additional dependency; not a guaranteed improvement |

In this repository, `--class-weight balanced` is supported by Logistic Regression, Decision Tree, Random Forest and SVM. It is **not** an implemented class-weight control for Gaussian NB, KNN or Gradient Boosting; the script warns/ignores it for those algorithms. Test `python scripts/run_classification.py --help` for flags such as `--neighbors`, `--c`, `--n-estimators`, `--max-depth` and `--min-samples-leaf`.

**Optional family comparison (same features, one changed algorithm per run):**

```bash
python scripts/run_classification.py --algorithm naive_bayes --features basic --name alg_nb
python scripts/run_classification.py --algorithm knn --features basic --name alg_knn
python scripts/run_classification.py --algorithm svm --features basic --name alg_svm
python scripts/run_classification.py --algorithm gradient_boosting --features basic --name alg_gb
python scripts/compare_models.py alg_logistic alg_tree alg_forest alg_nb alg_knn alg_svm alg_gb --name compare_seven_models
```

**Expected reasoning:** select the algorithm based on *held-out* metrics, false-alert workload, inference complexity, stability and interpretability, not training F1 or algorithm popularity. A model may rank incidents well overall but be inferior at a fixed 0.30 threshold. A fair comparison holds cohort, feature set and evaluation rows fixed; threshold calibration and score distribution should later be handled separately.

### Guided-session extension: feature engineering beats blindly moving the threshold

Two Logistic Regression experiments used the same threshold (0.30), same records and different feature sets:

| Metric | Basic features | Extended features | Effect |
| --- | ---: | ---: | --- |
| TP | 137 | 162 | +25 detections |
| FP | 82 | 46 | −36 false alerts |
| FN | 64 | 39 | −25 misses |
| Precision | 62.56% | 77.88% | Improved |
| Recall | 68.16% | 80.60% | Improved |
| F1 | 65.24% | 79.22% | Improved |

The extended set adds features such as `suspicious_url_count`, SPF/DKIM/DMARC results, attachment count, subject length, and sender domain. **It adds columns describing the same emails, not more training emails.** The refitted model can learn different relationships and separate suspicious-looking legitimate messages from phishing. More features are **not automatically** better: irrelevant noise, high-cardinality identifiers, correlated artifacts or future labels can harm generalisation.

```bash
python scripts/run_classification.py --dataset phishing --algorithm logistic --features extended --threshold 0.30 --name exercise02_extended
python scripts/compare_models.py exercise01_lower_threshold exercise02_extended --name feature_comparison
```

---
## 03. Investigate and improve precision

**Exercise:** `exercises/03_improve_precision.md`  
**Question:** Too many benign marketing messages are being alerted. Can we reduce FP without destroying recall?

The crucial result is not that “raising a threshold improves the model”. Raising the threshold changes the **decision rule**, not the underlying score ranking. The lab script reruns training with the same fixed configuration, then applies the chosen threshold. A more conservative threshold usually raises fewer alerts; it can suppress both benign false alerts and lower-score true detections.

```bash
python scripts/run_classification.py --threshold 0.30 --name phish_loose
python scripts/inspect_errors.py --experiment phish_loose --type fp --rows 15
python scripts/run_classification.py --threshold 0.50 --name phish_tight
python scripts/compare_models.py phish_loose phish_tight --name compare_precision_exercise
```

The handout's main task is to maintain recall ≥ 0.55 while seeking better precision. You should **check**, not assume, whether any threshold change satisfies that floor. The first command uses basic phishing features because `basic` is the classification script's default.

### Observed example: the same basic model at two thresholds

The guided sessions compared 0.50 (the original baseline) against 0.30:

| Metric | Threshold 0.50 | Threshold 0.30 | Operational meaning |
| --- | ---: | ---: | --- |
| TP | 112 | 137 | 25 extra phishing detections |
| FP | 21 | 82 | 61 extra benign alerts |
| FN | 89 | 64 | 25 fewer missed phishing |
| TN | 978 | 917 | 61 fewer correctly ignored benign messages |
| Precision | 84.21% | 62.56% | Alert quality worsened |
| Recall | 55.72% | 68.16% | Coverage improved |
| F1 | 67.07% | 65.24% | Overall precision/recall balance fell |
| Total alerts | 133 | 219 | 86 more alerts for the same validation sample |

**Why?** At threshold 0.50 a score of 0.40 means “no alert”; at 0.30 the same score means “alert.” The model has not acquired new information. The 86 newly raised alerts contained only **25 real positives** and **61 false positives**, so the *incremental alert precision* was 25/86 = **29.1%**. That's a practical SOC workload figure, not a separate classifier metric.

**How to inspect which decisions changed:** the model's output CSV contains one row per validation email and a stable `email_id`. Merge the two prediction files on that key and find where `predicted_label` changed from 0 to 1; compare `actual_label` for the new alert group. Do this for paired experiment configurations with identical features and fitting settings.

**Better next step:** keep threshold at 0.30 and add the extended feature set, as in Exercise 02. That controlled change improved TP **and** reduced FP in the observed results, suggesting better discrimination rather than mere alert-volume expansion. The labels and number of emails remained unchanged.

**Common wrong answer:** “Precision increased because we collected more true positives.” Precision depends on **both** TP and FP. A detection system can gain TP while losing precision if FP rises faster.

---

## 04. Improve EDR recall

**Exercise:** `exercises/04_improve_recall.md`  
**Question:** Our EDR model raises trustworthy alerts but misses dangerous process executions. How do we recover missed positives?

### Exercise handout baseline (basic features)

```bash
python scripts/run_classification.py --dataset edr --algorithm logistic --features basic --name edr_default
python scripts/inspect_errors.py --experiment edr_default --type fn --rows 12
python scripts/run_classification.py --dataset edr --algorithm logistic --features basic --threshold 0.25 --name edr_low_threshold
python scripts/compare_models.py edr_default edr_low_threshold --name edr_recall_tradeoff
```

The four **basic** EDR features are `command_line_length`, `process_prevalence`, `network_connection_count`, and `unsigned_binary`. They cannot uniquely represent every benign/malicious distinction. Inspect the actual false negatives: a legitimate-looking process may have a common parent, high prevalence, or few network connections despite being labelled malicious. A lower cutoff can recover some, but may also flag legitimate developer activity.

### Observed guided extension: Random Forest with extended EDR features

During the guided session we used Random Forest with 100 trees, depth 8, extended EDR features, same host-exclusive validation cohort and two thresholds:

```bash
python scripts/run_classification.py --dataset edr --algorithm random_forest --features extended --threshold 0.50 --max-depth 8 --name edr_baseline
python scripts/run_classification.py --dataset edr --algorithm random_forest --features extended --threshold 0.35 --max-depth 8 --name edr_threshold035
python scripts/compare_models.py edr_baseline edr_threshold035 --name edr_threshold_comparison
```

| Metric | Threshold 0.50 | Threshold 0.35 | Change |
| --- | ---: | ---: | ---: |
| Validation events | 1,295 | 1,295 | 0 |
| Actual malicious events | 148 | 148 | 0 |
| TP | 100 | 124 | **+24** |
| FP | 2 | 9 | +7 |
| FN | 48 | 24 | **−24** |
| TN | 1,145 | 1,138 | −7 |
| Precision | 98.04% | 93.23% | −4.81 percentage points |
| Recall | 67.57% | 83.78% | **+16.22 percentage points** |
| F1 | 80.00% | 88.26% | **+8.26 percentage points** |
| Accuracy | 96.14% | 97.45% | +1.31 percentage points |
| Alerts | 102 | 133 | +31 |

At threshold 0.35, **24 of the 31 newly raised alerts were genuine detections**; incremental alert precision = 24/31 = **77.4%**. For this synthetic EDR validation cohort that is a much more favourable exchange than the basic phishing threshold experiment. Yet **24 malicious events remain missed**; high F1 should not conceal that operational exposure.

The 0.35 experiment reported `roc_auc=0.9898`, `pr_auc_average_precision=0.9520`, and `train_f1=0.9487`; the validation F1 was 0.8826. This is a meaningful train/validation gap (~6.6 percentage points), not automatic proof of memorisation.

### Why this is not “new training weights”

In this script, `model.fit(...)` runs on each invocation and `predict_proba(...)` produces the class-1 scores. The `--threshold` switch is applied **after** fitting. With identical training configuration and a fixed seed, the predicted scores should remain equivalent. The new cutoff raises more alerts. `train_f1` can increase even though the trained trees are unchanged, because the training predictions are also recomputed using the new cutoff.

### How to see the nine false-positive processes

```bash
python scripts/inspect_errors.py --experiment edr_threshold035 --type fp --rows 20
```

This is preferable to a fragile one-liner that guesses prediction column names. The prediction file contains original fields including `hostname`, `endpoint_type`, `process_name`, `parent_process` and behavioural features; it also has `actual_label`, `predicted_label`, `positive_probability` and `error_type`. Group the FPs by developer/server/user endpoint **before** changing parameters again.

**Population-specific question:** are legitimate Python, PowerShell or build-tool executions on developer workstations driving the new false positives? Don't presume they are; confirm using the records. If so, investigate improved features or population-specific baselines before simply lowering the threshold further. A binary model cannot identify malicious intent solely from a high connection count.

---

## 05. Predict alert triage time with regression

**Exercise:** `exercises/05_regression.md`  
**Question:** Can we predict the investigation time of a security alert, rather than classify it malicious/benign?

```bash
python scripts/inspect_data.py --dataset triage --rows 5
python scripts/run_regression.py --algorithm linear --features basic --name triage_basic
python scripts/run_regression.py --algorithm linear --features extended --name triage_extended
python scripts/run_regression.py --algorithm gradient_boosting --features extended --name triage_boosting
python scripts/compare_models.py triage_basic triage_extended triage_boosting --name compare_triage_improvements
python scripts/inspect_errors.py --experiment triage_boosting --type largest_residuals --rows 10
```

**Regression versus classification:** `estimated_triage_minutes` is a continuous target, not a 0/1 label. The model predicts a quantity such as 42.5 minutes. There is no classification threshold, TP/FP/FN or precision/recall until someone explicitly converts this numerical prediction into a new categorical decision.

The **basic** feature set includes telemetry count, related host count, and analyst workload. The **extended** set adds investigation complexity, historical alert frequency, alert type and severity. This is a controlled information-gain exercise: first change features with algorithm fixed, then change algorithm with features fixed.

### What do the regression metrics actually mean?

| Metric | Calculation / interpretation | Desired direction |
| --- | --- | --- |
| MAE | Mean absolute value of `(actual minutes − predicted minutes)` | Lower |
| MSE | Mean squared error in **minutes²** | Lower |
| RMSE | Square root of MSE, expressed in **minutes**; penalises large misses more strongly | Lower |
| R² | 1 minus model squared-error sum divided by the validation-target variance sum | Higher, but ≤1 and can be negative |
| Residual | `actual_minutes − predicted_minutes` for an alert | Close to zero |

**Illustrative error calculation:** Actual triage time = 90 minutes; predicted time = 55 minutes; residual = +35 minutes. A positive residual means we **underpredicted** workload. If predicted = 110, residual = −20 minutes, an **overprediction**. MAE averages the *absolute* size of such errors; RMSE reacts more strongly to exceptional long-tail misses.

**Verified package reference:** the original linear/basic reference run documented **MAE ≈31.29 minutes**, **RMSE ≈39.64 minutes**, **R² ≈0.228** on validation. These are not guaranteed measurements for every version and say nothing about which of the later, extended experiments will improve most.

**Why the basic model may struggle:** complexity and severity are intentionally influential target drivers, but the basic model does not receive them. Furthermore, duration may depend on nonlinear interactions (e.g., complexity multiplied by affected host count); Gradient Boosting can capture nonlinear structure a plain additive linear model may miss. The prediction target contains noise, so even a well-designed model cannot explain every minute.

**How to interpret improvement:** if MAE falls after adding extended features, check whether large residuals also fall, whether RMSE improves, and whether the additional columns would genuinely be available **before** triage starts. A highly predictive field computed after investigation is **target leakage**, not legitimate improvement. Compare training MAE with validation MAE for overfitting. A complex model improving training MAE while worsening validation MAE should raise concern.

---

## 06. Unsupervised clustering: K-means, DBSCAN and hierarchical

**Exercise:** `exercises/06_clustering.md`  
**Question:** Can the network activity features reveal groups of related behaviour even when the model is not given anomaly labels?

### 06A. K-means: understand the output before tuning

```bash
python scripts/run_clustering.py --algorithm kmeans --features behavioral --scale yes --k 4 --name clustering_baseline
```

This runs on **5,000 network sessions** and uses six raw telemetry columns as input: `connection_frequency`, `destination_count`, `bytes_sent`, `connection_duration`, `unusual_hour_activity`, `failed_connection_count`. Before fitting, the pipeline applies `log1p` to byte fields, imputes missing values and standardises numeric dimensions. **All six transformed dimensions are used together** when calculating Euclidean distances.

K-means chooses four initial centroids (with deterministic initialisation settings in this lab), assigns each session to its nearest centre, updates centres as the mean of assigned observations, and iterates. `cluster_id` is just a group identifier. There is no class probability, true/false alert, or label during this fit.

**Observed K-means group sizes:**

| Cluster | Total sessions | Normal label 0 | Known anomaly label 1 | Anomaly share within group |
| --- | ---: | ---: | ---: | ---: |
| 0 | 673 | 649 | 24 | 3.57% |
| 1 | 1,346 | 1,306 | 40 | 2.97% |
| 2 | 2,386 | 2,380 | 6 | 0.25% |
| **3** | **595** | **324** | **271** | **45.55%** |
| **Total** | **5,000** | **4,659** | **341** | **6.82%** |

**What this means operationally:** Cluster 3 occupies **11.9%** of sessions but holds **271/341 = 79.5%** of the known anomalies. That's a useful triage concentration. However, **324/595 = 54.5%** of Cluster 3's sessions are known normal. If an analyst implemented “Cluster 3 = alert”, that **new rule** would have TP=271, FP=324, FN=70, precision 45.5%, recall 79.5%. **Those are derived metrics for a hypothetical rule after clustering, not metrics K-means optimises or outputs natively.**

### 06B. Why does Cluster 3 look different?

From the guided post-hoc mean-feature analysis:

| Mean feature | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
| --- | ---: | ---: | ---: | ---: |
| Connection frequency | 24.09 | 30.47 | 10.44 | **50.59** |
| Destination count | 5.47 | 6.65 | 3.10 | **16.54** |
| Bytes sent (raw units) | 146,459 | 215,954 | 65,651 | **708,318** |
| Duration | 40.55 | 43.21 | 41.94 | **88.91** |
| Unusual-hour flag mean | 1.00 | 0.00 | 0.00 | **0.52** |
| Failed connection count | 0.93 | 1.13 | 0.52 | **6.65** |

The flag mean 0.52 in Cluster 3 means about **52%** of its sessions have `unusual_hour_activity=1`; it does *not* mean a time of 0.52 hours. Cluster 3 shows multiple distinctive signals (frequency, destinations, bytes, failures, duration), not just one. These are **descriptive averages**. They are not feature-attribution scores or proof of causality; distance to the centroid is calculated after transforming/scaling the individual rows.

To reproduce the descriptive analysis without retraining:

```bash
python -c "import pandas as pd; d=pd.read_csv('outputs/clustering_baseline_predictions.csv'); cols=['connection_frequency','destination_count','bytes_sent','connection_duration','unusual_hour_activity','failed_connection_count','known_anomaly_label']; print(d.groupby('cluster_id')[cols].mean().round(2).to_string()); print(pd.crosstab(d['cluster_id'],d['host_role']))"
```

**The developer-workstation trap:** Cluster 3 contains **397 developer**, **57 server**, and **141 ordinary user** sessions. Of the 397 developer sessions, **320 are normal** and **77 are known anomalous**. Therefore **320/324 = 98.8% of all the normal observations in Cluster 3 are developer sessions**. Grouping them together does not imply the developers are compromised; legitimate developer builds, package retrieval and automation can look high-volume. `host_role` was **not** one of the six K-means fitting features; it is a post-hoc descriptor in this experiment.

```bash
python -c "import pandas as pd; d=pd.read_csv('outputs/clustering_baseline_predictions.csv'); print(pd.crosstab([d['cluster_id'],d['host_role']],d['known_anomaly_label']).to_string())"
```

The best interpretation: Cluster 3 is a group of **high-activity network behaviour with substantial anomaly enrichment and substantial legitimate developer overlap**, not “the malware cluster”.

### 06C. Silhouette, inertia, elbow and PCA

The guided K-means baseline reported **silhouette sample ≈0.3152** and **inertia ≈11,894.31**.

- **Silhouette** ranges approximately from −1 to +1. High positive values suggest observations are closer to their own group than other groups; ~0.315 indicates modest separation/overlap. It is a *geometric* measure, not a maliciousness score.
- **Inertia** is the sum of squared distances from points to their assigned K-means centres in the transformed input space. It generally falls as K increases, even if the extra groups have no useful meaning. Never compare inertia between different feature scales as though it were a common calibrated metric.
- **Elbow method** plots K against inertia and searches for a point of diminishing returns. It is a heuristic, not a guaranteed optimal-K test. Increasing K from 4 to 5 can either improve **or worsen** silhouette.
- **PCA** in this lab projects the six-dimensional data down to two dimensions **for plotting**. It does not supply the features used to fit K-means or DBSCAN; overlap in 2D may conceal separation in six dimensions.

Controlled changes:

```bash
python scripts/run_clustering.py --features raw --scale no --k 4 --elbow --name km_unscaled
python scripts/run_clustering.py --features raw --scale yes --k 4 --elbow --name km_scaled
python scripts/run_clustering.py --features behavioral --scale yes --k 4 --elbow --name km_behavioral
python scripts/compare_models.py km_unscaled km_scaled km_behavioral --name compare_clustering
```

The first comparison changes scaling only (raw unscaled to raw scaled). The second changes the feature subset while keeping scaling and K unchanged. Do **not** compare the clustering silhouette from one configuration to an FP/FN metric from supervised learning as if they measured the same objective.

### 06D. DBSCAN: clusters versus noise

```bash
python scripts/run_clustering.py --algorithm dbscan --features behavioral --scale yes --eps 1.6 --min-samples 12 --name clustering_dbscan
```

DBSCAN discovers **dense regions** using a neighbourhood radius (`eps`) and minimum count (`min_samples`) rather than asking for a fixed number of centroids. It may designate sparse points as `cluster_id = -1`. The **−1 rows are noise points, not a third numbered cluster** and not automatically malicious.

**Observed DBSCAN output:**

| DBSCAN assignment | Total sessions | Known normal | Known anomaly | Anomaly share in assignment |
| --- | ---: | ---: | ---: | ---: |
| **Noise (−1)** | **35** | **9** | **26** | **74.29%** |
| Cluster 0 | 4,001 | 3,855 | 146 | 3.65% |
| Cluster 1 | 964 | 795 | 169 | 17.53% |
| **Total** | 5,000 | 4,659 | 341 | 6.82% |

DBSCAN found **two clusters plus 35 noise points**, noise fraction **35/5,000 = 0.7%**, silhouette (excluding noise) **~0.3701**.

**The critical distinction:** 26/35 noise observations are anomalous, so using noise as an alert rule would give **74.3% precision**. But only **26 of all 341 anomalies** fell in noise, giving **7.6% recall**. The remaining **315 known anomalies are inside ordinary clusters**. This would be useful as a high-enrichment hunting queue, not a comprehensive detector. If noise were the only alert condition, TP=26, FP=9, FN=315, TN=4,650, F1≈0.138.

The higher silhouette (~0.370 vs 0.315 for K-means) does **not** prove DBSCAN is a better security detector. DBSCAN's silhouette excludes its noise observations, and the two solutions have different group counts and shapes. Compare how well the groups aid *actual analyst decisions*, not silhouette alone.

### 06E. How DBSCAN parameters change the result

- Lower `eps`: tighter neighbourhood; often more points classed as noise or clusters fragmented.
- Higher `eps`: wider neighbourhood; often more merging and less noise.
- Higher `min_samples`: stronger local density requirement; often more noise.
- Lower `min_samples`: smaller dense groups permitted; often less noise but more small clusters.

These are tendencies, **not guarantees** for every dataset. Changing `eps` is not a probability-threshold adjustment: DBSCAN has no classifier `predict_proba()` in this lab. If your goal is to change the **noise queue**, measure noise count, its anomaly enrichment and its coverage separately.

```bash
python scripts/run_clustering.py --algorithm dbscan --features behavioral --scale yes --eps 1.2 --min-samples 12 --name dbscan_eps12
python scripts/compare_models.py clustering_dbscan dbscan_eps12 --name compare_dbscan_eps
```

**Hierarchical extension:**

```bash
python scripts/run_clustering.py --algorithm hierarchical --features behavioral --scale yes --k 4 --name network_hierarchical
```

Agglomerative clustering starts with individual observations and merges nearby groups into a hierarchy. The supplied script cuts this hierarchy at the requested K; it doesn't label maliciousness. Compared with K-means, hierarchical grouping is not constrained to centroid-based assignments in the same way. On 5,000 samples, compute/memory costs can be higher.

---
## 07. Anomaly detection and endpoint populations

**Exercise:** `exercises/07_anomaly_detection.md`  
**Question:** Can we identify rare security activity without telling the model which sessions are malicious, and can we reduce false positives from legitimate developer traffic?

### A different use of unsupervised learning

Clustering assigns observations to groups. **Anomaly detection scores how unusual an observation is relative to learned reference behaviour.** Being rare does **not** prove maliciousness; unusual but legitimate administrators, developer builds, backup jobs and new software versions can all score as anomalies.

The lab supports four approaches:

| Algorithm | What it learns or computes | Common failure mode |
| --- | --- | --- |
| **Isolation Forest** | Randomly splits the feature space; rare observations are often isolated in fewer splits | Rare but legitimate workflows score as suspicious |
| **Local Outlier Factor (LOF)** | Compares local neighbourhood density of an observation against neighbours | Sparse but legitimate subpopulations look anomalous |
| **One-Class SVM** | Learns a boundary around the bulk of a reference population | Sensitive to scaling, `nu` and kernel settings |
| **Z-score** | Measures deviation from feature means in standard-deviation units | Skewed/heavy-tailed features invalidate naive normal assumptions |

Unlike the supervised phishing classifier, the known anomaly labels **do not enter `fit()`**. They are consulted only when scoring validation examples after the detector is trained. They allow a *post-hoc* TP/FP/FN/TN measurement on synthetic ground truth; many real hunting settings have no complete labels, so precision and recall may be unavailable or biased.

### First baseline, then isolate population effects

```bash
python scripts/run_anomaly.py --algorithm isolation_forest --features raw --baseline global --contamination 0.08 --name anomaly_global
python scripts/inspect_errors.py --experiment anomaly_global --type fp --rows 12
python scripts/run_anomaly.py --algorithm isolation_forest --features raw --baseline per_role --contamination 0.08 --name anomaly_role
python scripts/compare_models.py anomaly_global anomaly_role --name compare_global_vs_role
```

The first model learns a **single global reference distribution** from hosts that may have very different legitimate network patterns. The second fits **separate role-specific models** (user, developer, server), with preprocessing also fitted separately for each role. This is an intentional model-design experiment, not just changing a reporting group. It may reduce false positives caused by distribution mismatch, but small population sample sizes and unrecognised subroles remain risks.

The original package's reference anomaly run documented approximately **precision 0.473**, **recall 0.595**, **FP 49**, **FN 30** and **FPR 0.053** for its default setup. These are reference observations, not the expected output of every subsequent role-specific experiment. The important lesson is that a moderate precision figure can coexist with high noise enrichment, but also substantial missed anomalies.

### What `contamination` means here

The lab's `--contamination 0.08` does **not** tell the algorithm that exactly 8% of the network sessions are malware. The script fits on training feature data, produces **training anomaly scores**, and chooses a cutoff using the `(1 − contamination)` quantile of those training scores. Scores above that cutoff trigger an anomaly flag on validation. The actual validation fraction alerted can differ from 8%, particularly under distribution shift or group-specific baselines.

In this implementation, **higher `anomaly_score` means more anomalous** (the script reverses scikit-learn's native `score_samples` sign when necessary). These are **not phishing/malware probabilities**. The CSV includes `anomaly_score`, `training_score_threshold` and `score_minus_threshold`; a positive score-minus-threshold flags an anomaly.

```bash
python scripts/run_anomaly.py --algorithm isolation_forest --features raw --baseline per_role --contamination 0.05 --name anomaly_role_strict
python scripts/compare_models.py anomaly_role anomaly_role_strict --name compare_role_contamination
python scripts/inspect_errors.py --experiment anomaly_role_strict --type fp --rows 12
```

Changing contamination from 0.08 to 0.05 raises the score cutoff on the training distribution, **usually reducing** the number of alerts. That may reduce FP **and** TP, increasing FN. It does not change known ground-truth prevalence. The right comparison is alert counts, FP/FPR, FN/recall, and the **per-role burden**, not an assumed perfect “5% malicious” rule.

### Practical security interpretation

Suppose developer workstations normally generate many HTTPS connections and large uploads. A global anomaly detector may consider these rare compared with ordinary user laptops. A per-role baseline may classify them as normal **relative to other developers**. That can improve false-alert precision, but can also conceal activity that is common in a compromised developer population. Consider hybrid global and role-aware signals, subgroup size and attack contamination when interpreting a better result.

**Additional experiments:** change **only** `--algorithm` to `lof`, `one_class_svm` or `zscore`, keeping role, features and contamination fixed. Then independently change `--features raw` to `behavioral`, and finally `noisy` to illustrate irrelevant-feature harm. Each run needs its own output prefix.

```bash
python scripts/run_anomaly.py --algorithm lof --features raw --baseline global --contamination 0.08 --name anomaly_lof
python scripts/run_anomaly.py --algorithm one_class_svm --features raw --baseline global --contamination 0.08 --name anomaly_ocsvm
python scripts/run_anomaly.py --algorithm zscore --features raw --baseline global --contamination 0.08 --name anomaly_zscore
python scripts/compare_models.py anomaly_global anomaly_lof anomaly_ocsvm anomaly_zscore --name anomaly_algorithm_comparison
```

---

## 08. Independent EDR challenge: staged hints and evaluation rubric

**Exercise:** `exercises/08_independent_challenge.md`  
**Goal:** Work through an unfamiliar 6,500-event synthetic EDR cohort. In this exercise, the *scenario-specific* validation targets are **precision ≥0.75**, **recall ≥0.60** and **FPR ≤0.03**.

> **Instructor note:** This section intentionally does **not** publish a single “correct” configuration or an unmeasured performance table. The challenge is about independent diagnosis, disciplined experimentation and defensible trade-offs. Multiple models may meet the exercise constraints. That is not a deficiency in the answer key.

### Start independently

```bash
python scripts/inspect_data.py --dataset edr --dataset-file datasets/edr_events_challenge.csv --rows 10
python scripts/run_classification.py --dataset edr --dataset-file datasets/edr_events_challenge.csv --algorithm random_forest --features basic --name independent_baseline
python scripts/inspect_errors.py --experiment independent_baseline --type fp --rows 15
python scripts/inspect_errors.py --experiment independent_baseline --type fn --rows 15
```

**What a good solution demonstrates:**

1. States the target and reports the **baseline** confusion matrix and prevalence.
2. Checks FP/FN records, noting whether misclassifications concentrate on developer workstations, uncommon scripting tools, parent/child relationships, missing values or unusual connections.
3. States a **testable hypothesis**, e.g. “basic features do not describe parent-child context, so enriched features may recover missed malicious-looking execution.” This is a hypothesis, not an assumed fact.
4. Changes **one independent factor**, writes a new `--name`, and compares on the **same challenge validation hosts**.
5. Explains the precision/recall/FPR trade-off and whether each of the **three** numerical constraints is actually met.
6. Avoids tuning against the final test, accidental hostname leakage and post-investigation label use.
7. Documents the chosen model, threshold, feature set and limitations in a short SOC decision memo.

### Three levels of hints (open in order)

**Hint 1 — Investigate before tuning:** Print 10–15 FP and FN rows. Do they mostly share any *available* feature patterns? If not, don't pretend one threshold is the universal cause.

**Hint 2 — Change what the model can see:** Compare basic and extended features while holding algorithm, tree settings and threshold fixed. The extended set includes `unique_destination_count`, `outbound_bytes`, `process_execution_frequency`, `unusual_parent_child`, `endpoint_type`, `process_name`, and `parent_process`.

**Hint 3 — Test generalisation rather than merely winning an in-cohort split:** Use `scripts/evaluate_shift.py` to train on the **original** EDR cohort and apply the frozen model to the separate challenge cohort. Training directly on the challenge CSV and validating within it does not demonstrate survival of a previously deployed old-cohort model.

### Grading rubric (100 points)

| Element | Points | Evidence required |
| --- | ---: | --- |
| Baseline and correct metrics | 20 | TP/FP/TN/FN plus prevalence, precision, recall, FPR |
| Error diagnosis | 20 | Specific FP/FN records, feature and endpoint-type patterns |
| Controlled experiments | 20 | New run name, exactly one change, same validation cohort |
| Operational decision | 20 | Explicit pass/fail against all targets; workload and miss trade-off |
| Generalisation and leakage safeguards | 20 | No final-test tuning; host-group awareness; feature availability and drift caveats |

**Potential strong answer (structure, not a predetermined result):** “I first measured [actual counts]. Developer host false positives were [observed examples]. My hypothesis was [reason]. With [single change] on the same challenge-validation hosts, FP moved from [x] to [y] and recall from [a] to [b]. That [meets/fails] the three constraints. Before deployment I'd verify an untouched host/time cohort and investigate any new-population differences.”

The challenge thresholds are *teaching targets*, not claims about acceptable detection performance for any enterprise.

---

## 09. Class imbalance and misleading accuracy

**Exercise:** `exercises/09_imbalance_and_accuracy.md`  
**Question:** Can a detector have high accuracy while detecting no phishing whatsoever?

```bash
python scripts/majority_baseline.py --dataset phishing
python scripts/run_classification.py --algorithm logistic --features basic --name imbalance_logistic
python scripts/run_classification.py --algorithm logistic --features basic --class-weight balanced --name imbalance_weighted
python scripts/compare_models.py imbalance_logistic imbalance_weighted --name compare_class_imbalance
```

The **all-benign** baseline predicts 0 for every email. It does not train a useful classifier. On the familiar 1,200-row phishing validation sample with 999 normal and 201 phishing emails, all-benign gives **999/1,200 = 83.25% accuracy** and **0% recall**, with 201 false negatives. Its false-positive rate is 0. Precision is mathematically undefined when no positive predictions are made; scikit-learn may report 0 under `zero_division=0`. Do not mistake this artificial precision value for “zero false alarms means high precision.”

Class weighting changes the **fitting objective**: errors on minority-class samples receive more weight. In Logistic Regression, that can move the learned boundary and alter probability scores. It is different from changing only a post-fit threshold, even if both strategies sometimes increase recall. It may increase FP and does not guarantee meeting the SOC's constraints.

**Oversampling:** `--sampling oversample` duplicates minority samples **inside the training partition only** in the current script. Duplicating samples before splitting or cross-validation would leak identical examples into the holdout and inflate results. Even training-only duplication can alter calibration and is not automatically better than class weights. For correlated or grouped security events, sampling should also respect host, user, campaign and time boundaries as appropriate.

---

## 10. Overfitting, tree depth and generalisation

**Exercise:** `exercises/10_overfitting.md`  
**Question:** Why can a model achieve a near-perfect score on examples it has already seen, while producing mediocre results on unseen examples?

### The exercise's EDR commands

```bash
python scripts/run_classification.py --dataset edr --algorithm random_forest --features basic --name rf_unrestricted
python scripts/run_classification.py --dataset edr --algorithm random_forest --features basic --max-depth 8 --name rf_depth8
python scripts/compare_models.py rf_unrestricted rf_depth8 --name compare_overfitting
```

In the Random Forest, each tree can learn complex combinations of feature splits. If allowed to grow deep, it can fit highly specific training patterns, including synthetic noise. Reducing `max_depth` limits how many successive split levels a tree can follow along a root-to-leaf path. It changes **model capacity during fitting**, not just the alert threshold. The default forest size remains 100 trees unless `--n-estimators` changes.

### Observed guided comparison on phishing (same extended features, threshold 0.30)

| Metric | Unlimited depth | Max depth 8 | Max depth 12 |
| --- | ---: | ---: | ---: |
| Training F1 | ~0.99 | ~0.85 | ~0.94 |
| Validation F1 | 0.7845 | 0.7990 | 0.7882 |
| Precision | 76.4% | **81.8%** | 78.0% |
| Recall | **80.6%** | 78.1% | 79.6% |
| TP | 162 | 157 | 160 |
| FP | 50 | **35** | 45 |
| FN | 39 | 44 | 41 |
| TN | 949 | 964 | 954 |

These counts were reconstructed from consistent guided-session confusion matrices; an initial spoken mention of “166 TP with 39 FN” was arithmetically inconsistent with 201 known positives and is **not** used. Training F1 values shown as `~` are rounded estimates from the learner's discussion, not exact metrics reconstructed from a training confusion matrix.

**Depth 8 versus unlimited:** training F1 fell by about 14 percentage points but validation F1 improved slightly. This is evidence consistent with **reduced overfitting**. However, recall fell by five true detections even as 15 false alerts were removed. Depth 8 isn't automatically the best SOC choice.

**Depth 12 versus depth 8:** training F1 rose to ~0.94, while validation F1 fell slightly; TP increased by 3 but FP increased by 10. The training-validation gap widened. That suggests increased overfitting **risk**, not proof that depth 12 will fail in production. The model is learning a more complex boundary but only modestly improves coverage on this validation cohort.

### The bias–variance lesson

- **Underfitting:** model too constrained to capture meaningful relationships; both training and validation performance may be poor.
- **Overfitting:** model is very effective on training records but less effective on unseen records; large train/validation performance gaps are warning signs.
- **Generalisation:** performance on unseen data from the relevant deployment distribution. The final untouched test and temporal/host-based evaluation are more important than training score alone.

Do not blindly seek the smallest train/validation gap. A model with training F1=0.65 and validation F1=0.64 is **not** preferable to one with 0.90 and 0.86 merely because its gap is smaller. Inspect the level of held-out performance and operational costs.

---

## 11. Label leakage and feature availability

**Exercise:** `exercises/11_data_leakage.md`  
**Question:** Why can a suspiciously good validation score still indicate a broken model?

```bash
python scripts/leakage_demo.py
```

The deliberate challenge file `datasets/phishing_emails_leakage_challenge.csv` contains `post_investigation_decision`, a field derived from information available **after** the email has been investigated. If used as a training feature, the model receives a version of the answer itself. A conventional random train/test split cannot detect that the feature would be **unavailable at inference time**: the same invalid feature can exist in both training and validation rows.

### What to observe

The script contrasts a valid feature whitelist with a deliberately leaky version. A near-perfect leaky result is *not* the goal of threat detection. It demonstrates why data lineage, data time and prediction time matter more than a leaderboard number.

Examples of potential security leakage outside this lab:

- An EDR column such as `analyst_disposition` that is populated only after triage.
- A quarantine-state or incident-ticket field created **because** an alert was already classified as malicious.
- Network session enrichment generated using knowledge from a future blocklist update and treated as historical pre-execution input.
- Events from the same file/host/campaign appearing in both training and validation when the goal is new-family or new-host generalisation.

The first three are **feature/target leakage**; the fourth is **split contamination or correlation leakage**. They are related but not identical issues. Never repair a leaky dataset by merely choosing a higher threshold; remove the invalid feature, rebuild the pipeline and repeat the evaluation.

**Instructor note:** The comparison variant exists to teach the failure mode; don't modify the main dataset by copying its leaky field into the standard feature whitelist. The independent challenge requires plausible **pre-decision** features only.

---

## 12. TF-IDF, model pipelines and hyperparameter search

**Exercise:** `exercises/12_text_and_cross_validation.md`  
**Question:** Can email text improve detection beyond metadata, and how do we tune a model without leaking validation information into training?

```bash
python scripts/run_classification.py --algorithm logistic --features extended --name metadata_only
python scripts/run_classification.py --algorithm logistic --features text --name metadata_plus_text
python scripts/compare_models.py metadata_only metadata_plus_text --name compare_metadata_text
```

### Why email text needs transformation

A classifier does not treat a paragraph as a scalar number. `TfidfVectorizer` converts text into columns representing terms or n-grams, weighted by their prevalence in individual documents relative to the corpus. This lab uses subject and body text with `max_features=180`, `min_df=3`, and 1–2-gram features per text field inside the shared preprocessing pipeline. The TF-IDF vocabulary and document-frequency statistics must be **fitted on training data only**. Do not build a vocabulary over all emails and then claim a clean holdout.

The extended metadata has categorical fields (`spf_result`, `dkim_result`, `dmarc_result`, `sender_domain`) that are one-hot encoded. Missing numeric values are median-imputed and scaled; missing categorical values are filled with the most frequent observed training category. The `ColumnTransformer` and estimator form a `Pipeline`, ensuring fitted transformations are reused during validation rather than refitted on validation data.

Adding text changes the actual feature space. It can improve ranking, add noise, or overfit repetitive synthetic wording. Record **FP, FN, F1, Average Precision and train/validation gap**. Do not promise a positive direction.

### Cross-validation hyperparameter search

```bash
python scripts/tune_classification.py --algorithm logistic --search grid --features extended --name metadata_cv
```

The repository's `tune_classification.py` searches Logistic Regression `C` and class weighting **within the training pool**, using three-fold CV and scoring with `average_precision`. For Random Forest, the search varies depth and `min_samples_leaf`; it uses `GroupKFold` for EDR to keep hosts separate within folds.

**Subtle but important:** this tuning script evaluates its chosen estimator at **threshold 0.50**, even if your earlier hand-tuned experiments used 0.30. You cannot attribute a difference between its printed validation recall and a 0.30 run solely to hyperparameter selection, because **both the fit and the operating threshold may differ**. Compare threshold, feature set, optimisation metric, split and model configuration explicitly.

`GridSearchCV` examines all listed parameter combinations; `RandomizedSearchCV` samples a defined number. In both cases, preprocessing is part of each training-fold pipeline. Training-fold CV helps avoid choosing hyperparameters based on final-test data, but repeated informal experimentation can still overfit the **validation** set used for final selection.

---
## 13. Data drift versus model drift

**Exercise:** `exercises/13_data_drift.md`  
**Question:** What happens when developer workstation behaviour changes after the detector was trained?

```bash
python scripts/drift_check.py --dataset edr
python scripts/run_classification.py --dataset edr --dataset-file datasets/edr_events_challenge.csv --algorithm random_forest --features basic --name shift_basic
python scripts/run_classification.py --dataset edr --dataset-file datasets/edr_events_challenge.csv --algorithm random_forest --features extended --name shift_extended
python scripts/compare_models.py shift_basic shift_extended --name compare_shift_features
```

**Data drift** means the distribution of observed features changes: maybe software deployment increases `network_connection_count`, developer tools become more prevalent, or process-parent patterns change. **Concept drift** means the relationship between available features and the target changes. **Measured model-performance drift** means evaluation on valid delayed labels shows degraded or changed precision, recall, errors or calibration. These terms are related but not interchangeable.

`drift_check.py` compares reference and challenge feature distributions using mean shifts relative to baseline standard deviations; it does **not** use labels and does **not** prove compromise. A shift of +1 standard deviation is a clue to investigate feature prevalence, not an automatic detection rule.

**Critical comparison trap:** `run_classification.py --dataset-file datasets/edr_events_challenge.csv` trains **a new model on the challenge cohort** and evaluates inside that cohort. This can measure whether extended features help within a *new* population. It **does not** measure how an older deployed model performs on new events.

For that, use the dedicated **frozen model** workflow:

```bash
python scripts/evaluate_shift.py --algorithm random_forest --features basic --name shift_frozen_basic
python scripts/evaluate_shift.py --algorithm random_forest --features extended --name shift_frozen_extended
```

This trains the preprocessing and estimator on **old-cohort training records**, evaluates on old validation hosts, and scores the separately generated challenge cohort **without refitting**. The outputs are `outputs/shift_frozen_basic_shift_metrics.json` etc., not the ordinary `<name>_metrics.json` generated by `run_classification.py`.

**Interpretation checklist:** compare recall, FPR, FP workload and AP; note that different class prevalence can change precision even when sensitivity and FPR remain similar. Check whether missing values, new categorical values or unseen software families cause model errors. A new cohort with different composition is not a controlled estimate of production drift; it is an intentional synthetic stress test.

---

## 14. Neural networks and backpropagation

**Exercise:** `exercises/14_neural_network.md`  
**Question:** How does a neural network learn weights differently from a tree or a linear model?

```bash
python scripts/neural_network_intro.py --epochs 3000 --rate 0.4 --name nn_default
python scripts/neural_network_intro.py --epochs 3000 --rate 0.05 --name nn_slow
```

This is a **four-row XOR toy experiment**, not a cybersecurity CSV classifier. Its purpose is to make the forward pass, loss and weight updates observable with only NumPy, without introducing PyTorch or TensorFlow at the start.

Conceptual flow:

```text
Numeric inputs X
    |
    v
First weighted layer: X @ W1 + b1
    |
    v
Nonlinear hidden activation: tanh(...)
    |
    v
Output weighted layer: hidden @ W2 + b2
    |
    v
Sigmoid -> predicted P(class 1)
    |
    v
Binary cross-entropy loss vs known label
    |
    v
Backpropagation: chain-rule derivatives
    |
    v
Gradient-descent update of W1, b1, W2, b2
```

With binary cross-entropy and a sigmoid output, the output preactivation error has the familiar form **`p − y`**. The derivative of `tanh(z)` with respect to `z` can be written **`1 − tanh(z)^2`**. Backpropagation efficiently calculates how changing each parameter would change loss. A gradient descent update subtracts the learning rate multiplied by the gradient.

**Why XOR matters:** XOR is not linearly separable. A linear classifier using only the two raw input dimensions cannot perfectly separate all four XOR labels. A hidden nonlinear layer can represent that relationship. The exercise demonstrates why nonlinear representations are powerful, *not* why a neural network should automatically replace Random Forest on security telemetry.

**Rate experiment:** 0.4 versus 0.05 changes only the learning rate, which controls update step size. A smaller rate may need more epochs; a very large rate can become unstable. Don't assert the faster rate is always better based on a single seed or toy target. Examine `outputs/nn_default_neural_metrics.json`, the equivalent `nn_slow` file, and the saved loss curves.

---

## 15. Selecting a model and preparing for deployment

**Exercise:** `exercises/15_deployment_practices.md`  
**Question:** After many validation experiments, how do we choose, verify and operate a model without fooling ourselves?

### A defensible model-selection rule

Write down the SOC policy **before** choosing a model. Example exercise policies used in our sessions:

- Policy A: recall ≥79% and **minimise FP count** on a fixed 1,200-email validation sample.
- Policy B: recall ≥80% and **strictly fewer than 50 FP** on that same validation sample.

On the observed phishing comparisons at threshold 0.30:

| Candidate | Validation recall | FP | Policy A | Policy B |
| --- | ---: | ---: | --- | --- |
| Random Forest, depth 8 | 78.1% | 35 | Fails recall | Fails recall |
| Random Forest, depth 12 | 79.6% | 45 | **Eligible, lowest FP** | Fails recall |
| Logistic Regression, extended | 80.6% | 46 | Eligible | **Eligible** |
| Random Forest, unlimited | 80.6% | 50 | Eligible | Fails strict FP `<50` |

Under **Policy A**, depth-12 RF technically wins its declared objective, **despite** not having the best recall. Under **Policy B**, Logistic Regression is the only candidate among these four that meets both conditions. Neither selection is automatically “the best model” for all SOCs. If the harm of missed phishing dwarfs one additional false alert, Logistic Regression might be preferred even under Policy A.

**Important limitation:** a fixed number such as “FP < 50” is not portable across sample sizes or populations. Production should express alert-volume budgets per relevant time window, host count or event volume, together with false-positive rates, measured investigation cost and consequences of missed threats. On a different dataset size, compare rates or normalize counts instead of applying the same absolute limit blindly.

### Separate development from final assessment

**During development:** choose features, algorithm, hyperparameters and threshold using training-fold CV and a controlled validation protocol. Record each attempt. Avoid making dozens of ad hoc choices after repeatedly examining the same validation errors, which can overfit your **decision process** to validation.

**After selection:** run **one** final untouched test evaluation, without changing the chosen configuration in response to the test result:

```bash
# Example only: use the configuration you actually selected.
python scripts/run_classification.py --algorithm logistic --features extended --threshold 0.30 --name candidate_validation
python scripts/run_classification.py --algorithm logistic --features extended --threshold 0.30 --evaluate-test --name candidate_final_test
```

The lab trains the estimator on the configured training split in both commands and switches which held-out split it scores. The final test is **not** another tuning run. If the test disappoints, document the finding and start a new, explicitly versioned development cycle with a fresh future holdout; don't quietly optimise until the test score rises.

### Training, inference and operational controls

The current lab's `run_classification.py` combines fitting and scoring in a single CLI invocation. Running it again with identical parameters and a new threshold remains a valid controlled experiment because the **fit is repeated deterministically**, but it is less efficient than loading one saved model. Production should separately version and persist the **full preprocessing-plus-estimator pipeline**, its input schema, training provenance, label definitions and chosen alert threshold. The human-readable `*_weights.txt` output is an inspection aid, **not a serialized ready-to-load estimator**.

A deployment plan should cover: feature-time availability; data/label provenance; latency and inference cost; rollback; version pinning; threshold and score calibration; population-specific FPR; delayed analyst ground truth; alert suppression/deduplication; drift monitoring; approved retraining gates; and adversarial manipulation of input telemetry. Batch inference processes a file or finite job; online inference scores live events. Online **learning** means updating weights incrementally, which these batch scripts do not automatically do.

---

## 16. Five-fold cross-validation: guided extension

**Extension:** `scripts/run_cross_validation.py` is present in the repository; a matching exercise can be documented as an optional `16_cross_validation_basics.md` handout.  
**Question:** Was 80.6% phishing recall on one validation partition a reliable estimate, or partly a favourable split?

### 16A. What does “five folds” mean?

Start with the **3,600-email training pool**. Divide it into five stratified parts of **720 emails each**. Train five separate models:

```text
Run 1: [HOLD OUT] [ TRAIN ] [ TRAIN ] [ TRAIN ] [ TRAIN ]
Run 2: [ TRAIN ] [HOLD OUT] [ TRAIN ] [ TRAIN ] [ TRAIN ]
Run 3: [ TRAIN ] [ TRAIN ] [HOLD OUT] [ TRAIN ] [ TRAIN ]
Run 4: [ TRAIN ] [ TRAIN ] [ TRAIN ] [HOLD OUT] [ TRAIN ]
Run 5: [ TRAIN ] [ TRAIN ] [ TRAIN ] [ TRAIN ] [HOLD OUT]
```

Each model trains on **2,880 rows** and evaluates on the remaining **720**, rotating until each of those training-pool emails has been held out once. This is **five independent fits**, **not** five trees, five feature sets, or five different algorithms. The fixed original validation pool (1,200) and final test pool (1,200) are not involved in this cross-validation procedure. Preprocessors are fitted **inside** each fold's training pipeline to prevent statistical leakage. For EDR, the extension uses `GroupKFold` to keep hosts separate across a fold's training and evaluation sides.

### 16B. Run and interpret the actual Logistic Regression experiments

```bash
python scripts/run_cross_validation.py --dataset phishing --algorithm logistic --features extended --threshold 0.30 --folds 5 --name cv_logistic
python scripts/run_cross_validation.py --dataset phishing --algorithm logistic --features extended --threshold 0.25 --folds 5 --name cv_logistic_025
```

Observed guided-session summaries:

| Metric, across held-out training-pool folds | Threshold 0.30 | Threshold 0.25 |
| --- | ---: | ---: |
| Mean training F1 | 0.7769 | 0.7686 |
| Mean held-out F1 | **0.7658** | 0.7542 |
| Mean precision | **0.7637** | 0.7279 |
| Mean recall | 0.7702 | **0.7851** |
| Sample std of recall | 0.0454 | 0.0425 |
| Lowest fold recall | 0.7355 | 0.7438 |
| Highest fold recall | 0.8430 | 0.8512 |
| Mean accuracy | 0.9208 | 0.9139 |
| Mean Average Precision | 0.8580 | 0.8580 |

**What changed?** Lowering the cutoff increased average recall **77.02% → 78.51%**, but decreased precision **76.37% → 72.79%** and F1 **76.58% → 75.42%**. Average Precision stayed at **0.8580** because the ranking of probability scores did not change. The mean recall still sits below the exercise's 80% policy threshold, and the **minimum** fold recall is much lower. This does not prove a production 80% floor can never be reached; it demonstrates that relying on a single observed validation rate (80.6%) was optimistic.

**Training F1 versus held-out F1:** at threshold 0.30 the mean values are ~0.7769 and ~0.7658, a gap of ~1.1 percentage points. That's relatively small compared with a heavily overfit forest, suggesting no dramatic training memorisation on this particular metric. It does not guarantee stable performance under time or population shift.

**Why mean recall can differ from 80.6% on the original validation split:** fold evaluation groups use different records, each model trains on 2,880 rather than 3,600 emails, and the held-out partitions have different difficulty. These effects can change performance even with fixed hyperparameters.

### 16C. Compare a forest using the same CV process

```bash
python scripts/run_cross_validation.py --dataset phishing --algorithm random_forest --features extended --threshold 0.30 --folds 5 --max-depth 12 --name cv_forest12
```

Compare `outputs/cv_logistic_cv_summary.json` and `outputs/cv_forest12_cv_summary.json`, plus each `*_cv_folds.csv` file. This extension saves fold training F1, held-out F1, precision, recall, TP, FP, FN, accuracy and average precision. **No measured forest CV result is asserted here**: use the actual output from your run. A higher *mean* recall with high between-fold variation is a trade-off, not automatic evidence of a more reliable model.

### Important statistical cautions

- The printed `sample_std` measures variability across five evaluated folds; it is **not** an independent deployment-population confidence interval.
- CV folds overlap heavily in their training examples, so fold scores are correlated. The best-of-many-CV-scores can still be selection biased.
- Stratified CV roughly maintains the minority-class proportion across folds; **GroupKFold** preserves host boundaries but does not necessarily preserve class proportions.
- Compare FP **rates** rather than raw FP counts when a fold has 720 emails but the earlier validation set had 1,200.
- A highly uneven threat family/time distribution may require group- or time-aware CV rather than random stratification.

---
## C. Consolidated experiment results from the guided sessions

This section is an **answer key for the runs discussed during the guided learning sessions**, not a leaderboard to blindly reproduce or a substitute for performing the exercises. Unless noted, results are from the **validation split** and depend on the lab's fixed synthetic data and recorded configuration.

### C1. Phishing — thresholds and feature engineering

| Configuration | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic, basic, cutoff 0.50 | 112 | 21 | 89 | 978 | 0.8421 | 0.5572 | 0.6707 |
| Logistic, basic, cutoff 0.30 | 137 | 82 | 64 | 917 | 0.6256 | 0.6816 | 0.6524 |
| Logistic, extended, cutoff 0.30 | 162 | 46 | 39 | 953 | 0.7788 | 0.8060 | 0.7922 |

**Lesson:** lowering a cutoff changes alert volume and moves along the same ranking. Adding informative features and refitting can improve the model's separation of cases, so TP and FP can improve **together**. If the goal is fewer missed phishing emails, compare the incremental FP/TP exchange and overall analyst capacity, rather than treating recall as free.

### C2. Phishing — algorithms with extended features, cutoff 0.30

| Algorithm | TP | FP | FN | TN | Precision | Recall | F1 | Training F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 162 | 46 | 39 | 953 | 0.7788 | 0.8060 | 0.7922 | 0.7725 |
| Random Forest, unrestricted | 162 | 50 | 39 | 949 | 0.7642 | 0.8060 | 0.7845 | ~0.99 |
| Random Forest, max depth 8 | 157 | 35 | 44 | 964 | 0.8177 | 0.7811 | 0.7990 | ~0.85 |
| Random Forest, max depth 12 | 160 | 45 | 41 | 954 | 0.7805 | 0.7960 | 0.7882 | ~0.94 |
| Gradient Boosting | 158 | 44 | 43 | 955 | 0.7822 | 0.7861 | 0.7841 | 0.8374 |

`~` denotes rounded training values verbally reported in the guided session, not checked to four decimals. Forest validation precision/F1 is calculated from the displayed counts; exact printed rounding in other installations may differ.

Gradient Boosting at **0.25**: TP=161, FP=53, FN=40, TN=946, precision=0.7523, recall=0.8010, F1=0.7759, training F1=0.8284. Its 0.30 and 0.25 runs both had ROC-AUC **0.9575** and Average Precision **0.8822**. Logistic Regression with extended features at 0.30 had ROC-AUC **0.9438** and Average Precision **0.8714**. **A higher AUC/AP did not yield a better fixed-threshold SOC trade-off** in these experiments.

### C3. EDR — same Random Forest, different thresholds

| Configuration | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF, extended, depth 8, threshold 0.50 | 100 | 2 | 48 | 1145 | 0.9804 | 0.6757 | 0.8000 |
| RF, extended, depth 8, threshold 0.35 | 124 | 9 | 24 | 1138 | 0.9323 | 0.8378 | 0.8826 |

**Lesson:** the more permissive threshold recovered 24 true detections while adding 7 false alerts. This did **not** alter the fitted forest's structure; its training-F1 change is due in part to the different threshold being applied to its training predictions.

### C4. K-means and DBSCAN — same 5,000 synthetic network sessions

| Measure | K-means baseline | DBSCAN baseline |
| --- | --- | --- |
| Fitting features | Six scaled behavioural features | Same six scaled behavioural features |
| Cluster count (excluding noise) | 4 | 2 |
| Noise count | 0 | 35 |
| Sample silhouette | 0.3152 | 0.3701 (excludes noise) |
| Strongly anomaly-enriched partition | Cluster 3, 271/595 anomaly-labelled | Noise −1, 26/35 anomaly-labelled |
| Coverage if ONLY this partition is turned into an alert rule | 271/341 = 79.5% | 26/341 = 7.6% |
| False alerts under that hypothetical rule | 324 | 9 |

**Lesson:** K-means identified a **broad anomaly-enriched group** with heavy benign developer overlap. DBSCAN found a **small high-enrichment noise queue**, missing the vast majority of anomalies if interpreted as a detector by itself. The algorithm output is **group membership**, not malware verdicts. The table's derived “coverage” and “false alerts” are explicitly hypothetical **post-hoc rule** evaluations.

### C5. What has not been measured in these guided sessions

No specific improved regression result, role-specific Isolation Forest improvement, hierarchical cluster result, challenge-winning model, TF-IDF gain, model-drift delta, or final-test deployment result was established in the dialogue recorded for this guide. The matching sections give **working commands, evaluation criteria and likely mechanisms**, not invented outputs. Run those experiments and add clearly labelled observed measurements to a later revision if desired.

---

## D. Troubleshooting and frequently confused concepts

### D1. “I changed the threshold. Did I train new weights?”

**Outcome-first answer:** In the **current lab**, both commands call `fit()` again, but with the **same training data, seed, features and estimator parameters**. The fit should be equivalent; the threshold changes **which predicted scores generate alerts**, not what features the model learned. The current CLI doesn't provide a separate load-an-existing-model inference command, but the paired experiment is still valid for learning the effect of a threshold. The threshold is not an input to `fit()` for these model families. For production, store and reload a fitted full pipeline rather than paying the retraining cost every time.

### D2. “Then why did training F1 change when I only changed the threshold?”

Because `train_f1` is computed from training-set **binary predictions**, and binary predictions are threshold-dependent. Even identical probability scores generate different 0/1 labels when you change 0.50 to 0.35. ROC-AUC and AP use the unthresholded scores, so they should stay unchanged in a properly controlled threshold-only comparison.

### D3. “Does a true positive count as an alert?”

Yes, **if** the lab treats class-1 predictions as alerts. TP and FP are both model alerts. A FN is malicious activity missed by that decision rule. Production platforms may add separate suppression, threshold tiers, deduplication or analyst triage before an actual incident ticket exists, so `TP+FP` describes the lab's raw positive predictions, not necessarily tickets closed by analysts.

### D4. “Is 99% training F1 evidence of a good detector?”

Not by itself. It only says the model predicts previously seen training examples extremely well **at the selected threshold**. Compare with held-out performance, fold stability, host/time drift, true alerts and false alarms. A large train–validation gap is an overfitting warning. Conversely, a small gap can exist even for an underfit model.

### D5. “Does adding more features always help?”

No. Informative, available-at-inference features can reduce error, as the extended phishing set demonstrated. Irrelevant or noisy fields can dilute useful distances, increase variance or allow accidental shortcuts. Some columns are **leaky** and must never be used. Test feature contributions with identical cohort, algorithm and threshold; check stability on unseen data.

### D6. “What does a five-fold result mean?”

Five folds = five subsets of **one training pool**. Each fit uses four subsets for learning, while the fifth is used to score held-out predictions. Rotate. The 3,600 training emails in this lab become five folds of 720, with 2,880 for fitting each model and 720 for evaluating it. Original 1,200 validation and 1,200 final-test emails remain separate. Fold standard deviation reflects variability across those partitions, not a guarantee of production recall.

### D7. “Why can a model have high accuracy but poor recall?”

Because benign observations may vastly outnumber malicious ones. Predicting everything benign generates many TN and few or no TP. Report prevalence, confusion counts, precision, recall, FPR and alert volume. For imbalanced threats, accuracy alone can conceal misses.

### D8. “Does K-means decide which observations are malicious?”

No. It learns groups based on distance in the chosen transformed features, without learning anomaly labels. A group containing known anomalies may be worth investigating, but group ID 3 is not a universal threat category. Cluster identifiers can permute between runs/algorithms.

### D9. “Is DBSCAN noise = a malware finding?”

No. Noise means an observation is not assigned to a sufficiently dense cluster under the selected parameters. A normal rare job can be noise and malicious sessions can sit in dense clusters. In our guided example, a noise-only alert rule caught **26 of 341** known anomalies: strong enrichment, poor coverage.

### D10. “Why do we scale features for K-means, KNN and SVM?”

These algorithms depend on distances or the geometry of the feature space. Without scaling, a column such as byte count can numerically overwhelm a binary flag. This lab also applies a `log1p` transform to byte counts before scaling to reduce extreme skew. Scaling is a *learned preprocessing operation*, not a detection label; in supervised workflows it is fit on training only.

### D11. “Why didn't higher AUC produce better recall at threshold 0.30?”

AUC summarises ranking across many possible thresholds. A specific threshold depends on how the model distributes and calibrates its scores. Different algorithms can have different score scales. To compare operating points fairly, tune thresholds using **training CV**, assess the chosen candidates on reserved validation, and test the final configuration once.

### D12. “Why did F1 increase despite precision falling?”

F1 combines precision **and** recall. A sufficiently large gain in recall can outweigh a smaller precision loss, as the EDR threshold change demonstrated. But F1 ignores true negatives, score calibration, maliciousness severity and investigation cost; a high F1 doesn't establish production readiness.

### D13. “What if `compare_models.py` refuses the comparison?”

It intentionally rejects incompatible problem, dataset name, source file or evaluation split. Compare only experiments based on the **same evaluation cohort and task**. Cross-validation summaries use a separate output format; compare their JSON/CSV summaries directly rather than feeding them into `compare_models.py` as standard experiment names.

### D14. “What does `error_type = correct` mean?”

It means the label and predicted class agree. It can be a TP **or** a TN. It is not a separate fifth confusion-matrix category. For classification, use `actual_label` and `predicted_label` to distinguish correct positives from correct negatives.

### D15. “Why is my silhouette score higher but security performance worse?”

Silhouette measures geometric group separation. It does not use threat labels. DBSCAN excludes `-1` noise from the lab's silhouette calculation, so direct comparisons to K-means silhouette have differing observation sets. Choose an unsupervised method for its operational purpose: exploration, discovering dense groups, or generating an anomaly-hunting queue; evaluate any downstream alert rule separately on independent labels.

### D16. “What did the model actually save? Can I load the trained weights?”

`outputs/<name>_weights.txt` is a readable **explanation artifact**. Logistic coefficients and tree structures can be inspected, but it is not a complete serialized inference pipeline. The lab does not currently provide a train-only/save-model command and a separate inference-only/load-model command. To build one, persist the full preprocessing-plus-estimator `Pipeline` along with versioned schema and metadata, load only trusted files, then apply a chosen threshold to `predict_proba`. Do not infer that retraining on the same source makes deployment-state management unnecessary.

---

## F. Further experiments and interpretation checklist

### A disciplined record for each new experiment

Use this template in your own notes or future pull requests:

```text
Experiment name:
Question / SOC objective:
Dataset and cohort / split:
Algorithm and training hyperparameters:
Feature whitelist and preprocessing:
Threshold or anomaly alert budget (if any):
What exactly changed from baseline (ONE factor):
Baseline TP/FP/TN/FN or MAE/RMSE or clustering metrics:
New measurements:
Largest new false-positive / false-negative pattern:
Host role / source population effects:
Training-versus-held-out performance gap:
Decision against the explicit policy:
What would falsify this explanation:
Next test (not on final holdout):
```

### Suggested follow-up experiments

1. **Threshold-only EDR check:** compare 0.50 versus 0.35 with fixed depth 8. Confirm ROC-AUC/AP remain equal and quantify newly detected malicious events per new FP.
2. **Feature-only phishing check:** hold Logistic Regression and 0.30 fixed; compare `basic` versus `extended`, then optionally `text`; inspect actual changed FP and FN records.
3. **Depth-only Random Forest check:** compare depths 8 and 12 on the same validation rows; document both F1 and SOC-constrained recall.
4. **CV stability check:** use five fixed training folds and compare mean, sample standard deviation and lowest fold recall; keep final test unseen.
5. **Role-aware anomalies:** compare `global` versus `per_role` using the same features and contamination. Inspect the number of false alarms on developer workstations, not just the aggregate precision.
6. **Clustering geometry:** compare scaled versus unscaled K-means, then K=4 versus K=5, then DBSCAN. Interpret silhouette/inertia without claiming a threat verdict.
7. **Independent shift:** fit on old endpoint telemetry and evaluate the frozen model against the new synthetic challenge cohort; document which feature distributions changed.
8. **Security controls:** test whether an added “perfect predictor” is known only after analyst investigation; reject it regardless of its validation score.

### Instructor review checklist

- [ ] The student distinguishes **classification, regression, clustering and anomaly scoring**.
- [ ] They compute TP, FP, FN, TN correctly and can explain **alerts = TP + FP**.
- [ ] They know **precision = alert quality** and **recall = detection coverage**.
- [ ] They know **FPR is measured over benign events**, not total alerts.
- [ ] They can identify which algorithm parameters are learned during `fit()` and which thresholds are applied later.
- [ ] They explain why changing a threshold does **not** change ROC-AUC/AP for fixed scores.
- [ ] They can explain why informative features improved both precision and recall, without saying more columns always help.
- [ ] They evaluate train–validation gaps without using training F1 as a deployment goal.
- [ ] They understand a five-fold CV rotation, training-only preprocessing and final-test isolation.
- [ ] They can explain K-means centroids, DBSCAN noise, silhouette, inertia and PCA without equating them with malware labels.
- [ ] They can assess anomaly alerts by **endpoint population**, including legitimate developer traffic.
- [ ] They know that `contamination` controls an anomaly score cutoff, not the true fraction of malicious records.
- [ ] They examine regression residuals, MAE, RMSE and feature-time availability.
- [ ] They reject leakage, duplicate-group contamination and final-test tuning.
- [ ] They can defend a model selection based on **measurable SOC requirements**, not just headline accuracy or F1.

### Maintaining this guide in a public repository

This guide is intended as a **companion to the existing exercise handouts**, not a replacement for them. Keep numbered sections synchronised with filenames in `exercises/`. When adding a new experiment, record its exact CLI, dataset/version/seed, observed output, what changed, the security interpretation and its limitations. Label hypothetical examples as illustrative; do not present them as executed results.

Place this file at `solutions/exercise_solutions.md`. To replace an older copy in a local Git checkout, save your original changes first, copy this new file over the old one, review the diff, and commit intentionally:

```bash
git diff -- solutions/exercise_solutions.md
git add solutions/exercise_solutions.md
git commit -m "Expand ML lab instructor solutions and worked SOC examples"
```

To make the guide discoverable **without spoiling the exercises**, add a clearly marked README link after the exercise roadmap:

```markdown
**Instructor solutions (spoilers):** [Worked explanations and investigation guide](solutions/exercise_solutions.md). Attempt each exercise first.
```

**Last principle:** the purpose of the lab is not to chase an attractive number. It is to understand which evidence the algorithm used, how it learned, how it made predictions or groups, where it failed, what changed in a controlled experiment, and whether the result would help a real security team.
