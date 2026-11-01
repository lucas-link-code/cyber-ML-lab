# Cyber ML Lab: practical machine learning with synthetic security telemetry

**Python 3.11+ | offline after installation | no credentials or external APIs | deterministic synthetic CSVs**

This is a runnable command-line laboratory for a security analyst learning ML. Start with the two commands in **Exercise 01**, then progress at your own pace. You do **not** need Jupyter, an IDE, or experience writing Python models. Run commands from the `cyber_ml_lab` directory.

## 1. Install on Windows PowerShell

Assumes the ZIP is in your Downloads folder and Python 3.11+ is installed.

```powershell
Set-Location "$HOME\Downloads"
Expand-Archive -Path .\cyber_ml_lab.zip -DestinationPath . -Force
Set-Location .\cyber_ml_lab
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/inspect_data.py --dataset phishing --rows 10
```

- `Set-Location`: open the directory with the ZIP, then the extracted project.
- `Expand-Archive`: extract the ZIP; do not accidentally open `cyber_ml_lab/cyber_ml_lab`.
- `py -3 --version`: verify installed Python is version **3.11 or newer**.
- `py -3 -m venv .venv`: create an isolated Python environment inside the lab.
- `Activate.ps1`: activate that environment so `python` and `pip` refer to its packages.
- `python -m pip install ...`: install NumPy, pandas, scikit-learn and matplotlib. Internet is needed for this first installation unless you already have cached wheels.
- Last command: print the phishing CSV's first rows, data types, missing values and label distribution. It does not train a model.

**If PowerShell blocks activation**, you do not have to change execution policy. Use the Python executable directly and omit activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/inspect_data.py --dataset phishing --rows 10
```

## 2. Install on macOS Terminal

```bash
cd ~/Downloads
unzip -o cyber_ml_lab.zip
cd cyber_ml_lab
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/inspect_data.py --dataset phishing --rows 10
```

- `cd` opens the directory containing the ZIP, then the extracted project.
- `unzip` extracts the same folder structure used by Windows.
- `python3 --version`: confirm **3.11+**. If not, install a supported Python and replace `python3` with its actual executable (e.g. `python3.12`).
- `python3 -m venv`: creates an isolated environment. `source` activates it.
- `python -m pip install`: installs dependencies **inside** that environment.
- Final command inspects real CSV rows without training.

To resume later, open the project folder and reactivate its environment (`.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on macOS). If the `python` command on macOS opens Python 2 or an unrelated installation, use `.venv/bin/python` explicitly.

## 3. Exercise 01: execute and observe

From the project root, with the environment activated:

```bash
python scripts/inspect_data.py --dataset phishing --rows 10
python scripts/run_classification.py --dataset phishing --algorithm logistic --features basic --name exercise01
```

**Command 1** reads `datasets/phishing_emails.csv` using pandas. It prints a CSV preview, column names and types, null counts, the `phishing_label` distribution and the feature set used by the first classifier. Focus on `email_id` (record key), `phishing_label` (correct outcome) and the five baseline features (`sender_reputation`, `url_count`, `urgent_language_score`, `has_shortened_url`, `body_length`). Neither `email_id` nor the label is a training input.

**Command 2** creates a numeric preprocessing pipeline, fills missing values using **training-set medians**, scales numeric features using **training-set statistics**, fits **logistic regression** on a training split and scores a separate **validation** split. It prints predictions, probability scores, the confusion matrix and accuracy, precision, recall, F1, FPR, ROC-AUC and average precision (PR-AUC). The **final test split is untouched**. Some results are deliberately wrong so that later exercises are meaningful.

Open these generated files with VS Code, Excel, an image viewer, or a text editor:

```
outputs/exercise01_metrics.json
outputs/exercise01_predictions.csv
charts/exercise01_confusion.png
charts/exercise01_precision_recall.png
charts/exercise01_roc.png
```

**Stop here** until you can explain: How many false positives? How many false negatives? Which is more expensive in your phishing detection scenario? Compare the two figures to the printed `precision` and `recall`. Share your terminal output with your instructor before continuing.

## 4. The ML workflow you will implement repeatedly

```
CSV dataset
   ↓   pandas: load and inspect data
Features + target (or unlabeled features for unsupervised learning)
   ↓   Split by record or host: training / validation / FINAL TEST
Preprocessing (fit ONLY on training data)
   ↓   missing values → encoding → scaling / TF-IDF
Algorithm.fit(training features, labels) [or just features if unsupervised]
   ↓   fitted parameters/model
Algorithm.predict / predict_proba / anomaly scoring
   ↓   validation predictions → metrics / error investigation
One controlled feature/parameter change
   ↓   compare baseline versus experiment on the SAME validation rows
Once selected → ONE final held-out test evaluation
```

For clustering, there is no prediction label during fitting. The script gives an **exploratory clustering** of the full unlabeled dataset. Ground-truth synthetic labels are accessed only after fitting for post-hoc interpretation. Do **not** select clustering parameters based on those labels and call the result an unbiased supervised test.

For anomaly detection, fitting and score-threshold calibration use **only training features/scores**; known labels are withheld until validation metrics are calculated. EDR/network splits are **host-exclusive**. For supervised classification and regression, the final test split is reserved; use cross-validation and validation for experimentation instead.

## 5. Learning roadmap

| Level | Exercises | Main activities |
| --- | --- | --- |
| 1. Execute and observe | 01, 02 | Read the CSV, fit first classifier, compare classifier families |
| 2. Understand and investigate | 03, 04, 09 | Precision / recall / threshold / class imbalance / error inspection |
| 3. Modify and improve | 05, 06, 07, 10 | Regression errors, clustering, role-specific anomaly baselines, overfitting |
| 4. Broaden technique | 11, 12, 13, 14 | Leakage, TF-IDF, cross-validation, drift, backpropagation |
| 5. Independently solve | 08 | EDR holdout challenge: investigate and justify decisions without spoilers |

**Suggested order**: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 09 → 10 → 11 → 12 → 13 → 14 → 08.

The standalone instructions are in `exercises/`. Each includes a single-variable experiment, what to observe, questions, comparisons, and an interview scenario. `solutions/exercise_solutions.md` is a spoiler file; do not open it before attempting the independent challenge.

## 6. Dataset contents

| File | Rows | Purpose | Data characteristics |
| --- | ---: | --- | --- |
| `datasets/phishing_emails.csv` | 6,000 | Classification and TF-IDF | 17% malicious; difficult legitimate marketing and stealth phishing cases |
| `datasets/edr_events.csv` | 6,500 | Classification and anomaly detection | Users, servers and developers; deliberately overlapping rare benign/malicious telemetry |
| `datasets/alert_triage.csv` | 3,500 | Regression | Alert investigation minutes; nonlinear complexity with noise |
| `datasets/network_activity.csv` | 5,000 | Clustering and anomalies | Unequal host populations, rare abnormal connections, noisy developer traffic |
| `datasets/edr_events_challenge.csv` | 6,500 | Generalisation / independent challenge | New developer-behaviour distribution; separate synthetic cohort |
| `datasets/alert_triage_challenge.csv` | 3,500 | Regression drift | Harder target noise; separate cohort |
| `datasets/network_activity_challenge.csv` | 5,000 | Anomaly drift | More legitimate anomalous-looking developer flows |
| `datasets/phishing_emails_leakage_challenge.csv` | 6,000 | Explicit leakage demonstration | Post-investigation feature correlated with answer; excluded from all standard pipelines |

Column definitions and training exclusions: `DATA_DICTIONARY.md`.

### Common command options

- `--name unique_name`: change output prefix; use a new name to preserve baseline JSON, CSV and charts.
- `--features basic|extended|text`: train on a predefined whitelist of input features; `text` is phishing-only and adds TF-IDF.
- `--threshold 0.7`: change only the decision threshold; does **not** retrain or improve probability ranking.
- `--algorithm ...`: choose a new model family (different training behaviour).
- `--class-weight balanced`: weight classification errors by class (supported by logistic, tree, Random Forest and SVM).
- `--sampling oversample`: duplicate minority rows **inside training only**; compare with weighting.
- `--scale no`: disable numeric scaling to show why KNN, SVM and regression are sensitive to magnitude.
- `--dataset-file datasets/edr_events_challenge.csv`: load a different synthetic cohort; comparisons must be made on compatible splits/data.
- `--evaluate-test`: evaluate the reserved **final test**. Do this only after selecting your approach; do not repeatedly tune against it.

Run `python scripts/run_classification.py --help`, or the corresponding script's `--help`, to discover exact choices.

## 7. Command reference

```bash
# Supervised classification, logistic regression (phishing)
python scripts/run_classification.py --name phish_base
# Supervised classification, Random Forest (EDR)
python scripts/run_classification.py --dataset edr --algorithm random_forest --features extended --name edr_rf
# Regression, linear and gradient boosting
python scripts/run_regression.py --name triage_linear
python scripts/run_regression.py --algorithm gradient_boosting --features extended --name triage_gb
# Clustering, with PCA plot and K-means elbow curve
python scripts/run_clustering.py --algorithm kmeans --k 4 --elbow --name net_kmeans
# Unsupervised anomaly detection (validation metrics use synthetic known labels)
python scripts/run_anomaly.py --algorithm isolation_forest --name net_anomaly
# Compare compatible saved results
python scripts/compare_models.py triage_linear triage_gb --name compare_triage
# Inspect the actual false positives
python scripts/inspect_errors.py --experiment phish_base --type fp
# A weak imbalanced-class baseline
python scripts/majority_baseline.py --dataset phishing
# Cross-validation, supervised hyperparameter search (training only)
python scripts/tune_classification.py --algorithm logistic --search grid --name phish_cv
# Diagnose shifts in synthetic feature distributions
python scripts/drift_check.py --dataset edr
# Explicit label leakage warning example
python scripts/leakage_demo.py
# Very small feed-forward neural network, manual backpropagation
python scripts/neural_network_intro.py
# Regenerate ALL synthetic CSVs with fixed seeds
python scripts/generate_datasets.py
# Verify split isolation, schemas and basic outcomes
python scripts/verify_lab.py
```

### What the metrics mean

- **TP**: malicious item correctly flagged; **FP**: benign item incorrectly flagged; **FN**: malicious item missed; **TN**: benign item not flagged.
- **Precision** = TP/(TP+FP): among alerts, what fraction is really malicious?
- **Recall / TPR** = TP/(TP+FN): among malicious cases, what fraction is detected?
- **FPR** = FP/(FP+TN): among benign items, what fraction triggers a false alarm?
- **Accuracy** = (TP+TN)/all: potentially misleading for rare malware.
- **F1**: harmonic mean of precision and recall; choose a metric suited to operational costs.
- **ROC-AUC**: ranking quality across TPR/FPR thresholds; **PR-AUC** in this lab is **average precision**, a useful imbalanced-class ranking summary. These metrics do not change when only the decision threshold changes.
- **MAE/RMSE**: size of continuous-minute prediction errors (RMSE penalises large misses more); **R²**: relative squared-error improvement compared with predicting the test-set mean.
- **Silhouette**: separation/compactness heuristic for unlabeled clusters; neither high silhouette nor PCA separation proves maliciousness.

## 8. Technical safeguards and limitations

1. Every row and hostname is generated from NumPy pseudorandom draws. `.example` is reserved and non-routable, and no live indicators, real mail, hostnames or executable content are used.
2. `email_id`, `event_id`, `alert_id`, `session_id`, `hostname` and `source_host` are excluded from training features. Ground-truth labels are excluded from unsupervised training features. No feature is created from the target except in the **explicitly marked leakage challenge**.
3. Numeric imputation, scaling and categorical one-hot encoding are learned on training rows only in supervised/anomaly workflows. For TF-IDF, the vocabulary is also fit on training rows only. The scripts intentionally include missing values and irrelevant fields; these are not implementation errors.
4. `outputs/` and `charts/` are generated per experiment; running the same `--name` overwrites that experiment's outputs. Charts are `.png`, metrics `.json`, predictions `.csv`.
5. Validation can also be overfit by excessive repeated tuning. Use cross-validation on training for broad search; reserve validation for choosing a candidate and the final test for once-only confirmation. Challenge variants are separate synthetic samples, not evidence of real deployment performance.
6. Classifier `predict_proba()` values are model probability estimates, not guaranteed to be calibrated. Isolated anomalies are not confirmed threats. Precision, recall and alert rates from these synthetic sets are illustrative of ML behaviour, **not expected bank/production detection performance**.
7. For clustering, the PCA chart depicts a linear 2D projection of higher-dimensional inputs; visual proximity need not correspond to all high-dimensional distances. K-means requires a chosen `k`; DBSCAN noise is not itself malicious.
8. Dataset generation regenerates the eight CSV files and **overwrites** their contents deterministically. Never regenerate after hand-editing a challenge CSV unless you have saved your changes separately.
9. XGBoost is not a core dependency: `python -m pip install xgboost` then use `--algorithm xgboost` optionally. Neural network lesson is deliberately implemented with NumPy rather than introducing another heavyweight dependency.

**Do not import these synthetic metrics into detection production.** They are designed as controlled training exercises with invented probabilities and distributions.

## 9. Cross-cohort drift and inference extension

To evaluate a **frozen model trained on the original EDR cohort** against a different synthetic deployment cohort (rather than retraining on it):

```bash
python scripts/evaluate_shift.py --algorithm random_forest --features basic --name shift_frozen_basic
python scripts/evaluate_shift.py --algorithm random_forest --features extended --name shift_frozen_extended
```

This fits feature preprocessing and the model using original **training rows only**, evaluates on old validation hosts, then applies the unchanged model and threshold to the entire separately generated challenge CSV. It writes `outputs/shift_frozen_basic_shift_metrics.json`, two prediction CSVs and a comparison chart. The new cohort is synthetic and not an unbiased estimate of any bank deployment environment. This assesses cross-cohort performance drift rather than within-cohort model selection.

For a **permutation/impurity feature-importance** starting point, train `--algorithm random_forest`; inspect `charts/<name>_feature_importance.png`. The plotted impurity importance indicates model split utility, not causal attribution. For operational interpretability, compare errors across populations, avoid treating correlated predictors as independent causes and consider a separately validated permutation-importance experiment.

For **batch vs online learning**: these scripts implement batch training and batch inference from CSVs. A streaming/online endpoint detector would need an event-time feature pipeline, stateful baseline updates, model versioning, rollback, security review and delayed-label handling. Scikit-learn `partial_fit` supports incremental training for selected estimators (e.g. `SGDClassifier`), but the exercises do not silently enable it; online updates can poison a baseline if anomalous events are incorporated uncritically.
