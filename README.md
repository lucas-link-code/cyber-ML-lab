# Cyber ML Lab: practical machine learning with synthetic security telemetry

**Python 3.11+ | offline after installation | no credentials or external APIs | deterministic synthetic CSVs**

This is a runnable command-line laboratory for a security analyst learning ML. Start with the two commands in **Exercise 01**, then progress at your own pace. You do **not** need Jupyter, an IDE, or experience writing Python models. Run commands from the lab folder. Its name depends on how you download it:

- git clone: `git clone https://github.com/lucas-link-code/cyber-ML-lab.git` creates `cyber-ML-lab`. Run `cd cyber-ML-lab`, then skip the three commands that move to Downloads and extract the ZIP, then continue from the version check in the install section for your system.
- Download ZIP: GitHub saves `cyber-ML-lab-main.zip`. The install blocks below extract it from Downloads to `cyber-ML-lab-main`.

## 1. Install on Windows PowerShell

Assumes the ZIP is in your Downloads folder and Python 3.11+ is installed.

```powershell
Set-Location "$HOME\Downloads"
Expand-Archive -Path .\cyber-ML-lab-main.zip -DestinationPath . -Force
Set-Location .\cyber-ML-lab-main
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/inspect_data.py --dataset phishing --rows 10
```

- `Set-Location`: open the directory with the ZIP, then the extracted project.
- `Expand-Archive`: extract the ZIP; do not accidentally open `cyber-ML-lab-main/cyber-ML-lab-main`.
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
unzip -o cyber-ML-lab-main.zip
cd cyber-ML-lab-main
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

**Stop here** until you can explain: How many false positives? How many false negatives? Which is more expensive in your phishing detection scenario? Compare the two figures to the printed `precision` and `recall`.

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

### What the command does to the rows

`run_classification.py` divides the CSV into three disjoint groups before it learns anything: training, validation, and a final test set. On the 6,000 row phishing file that split is 3,600, 1,200, and 1,200. EDR and network rows are grouped by host, so one hostname is not present in two groups.

`fit` runs only on the training group. That call is the learning step. For logistic regression the learned parameters are an intercept and one weight per feature after preprocessing. For a random forest they are the splits and the leaf values of the trees. When `fit` returns, those parameters stay fixed. The script then applies them to the validation rows. True positives, false positives, false negatives, and validation F1 are the comparison of those predictions with the real validation labels. The test rows stay unused unless you pass `--evaluate-test`.

A classification run also writes the learned parameters to `outputs/<name>_weights.txt`. The metrics JSON and the confusion matrix chart are the score of those fixed parameters. They are not a second round of learning.

### What fit() function is doing

`fit` means: find parameters that reduce mistakes on the training rows.

Logistic regression starts from an initial weight vector and updates it to reduce a loss. The loss is a penalty for assigning a high probability to the wrong class. Each update nudges the weights. Training stops when the loss stops improving enough, or when the iteration limit is reached. The phishing probability is the sigmoid of the intercept plus the weighted features. The default lab pipeline standardises numeric features first, so the weights multiply those scaled values.

A random forest grows many trees on the training rows. At each split a tree chooses a feature and a cutoff that separate the training labels, then repeats on the two sides. Growth stops when the tree hits its depth limit, a leaf has too few rows, or no split still helps. The forest prediction is the combined vote of those trees. `fit` is finished when the trees exist. Scoring validation only walks the rows down those trees.

Unsupervised commands use the same word and a different target. Clustering `fit` groups rows by feature similarity and does not read a label. Anomaly `fit` learns a picture of normal training rows and does not read a label. Known labels are used afterwards, to score the result.

### Three ways a later run can change the result

You can change the result in three places. Only the first two call `fit` again.

1. Change the inputs. `--features basic`, `extended`, or `text` selects a different column whitelist. `text` is phishing only and adds term weights from the subject and the body. The vocabulary is learned on the training rows.

2. Change the training settings. `--algorithm` selects the family. Depth, leaf size, `C`, class weight, the number of trees, and oversampling of the training rows change what `fit` stores. Use a new `--name` so the previous metrics, predictions, and charts remain on disk.

3. Change the decision threshold. `--threshold 0.7` keeps the same probabilities and moves the cutoff that turns a probability into class 0 or class 1. Precision, recall, and the confusion matrix move. ROC AUC and average precision stay put, because those scores describe the ranking, not one cutoff. A threshold change cannot repair a model that ranked the validation rows poorly.

### High training F1 is not evidence of model quality

Training F1 near 1.0 means the model matched the training rows very closely. That can be memorisation of those rows rather than a rule that holds on new ones.

The saved phishing forest in `outputs/exercise03_forest_metrics.json` is an unrestricted random forest on the extended features, with threshold 0.30. Training F1 is 0.997 on the 3,600 training emails. Validation F1 is 0.785 on the 1,200 validation emails. Logistic regression on the same features and the same threshold, in `outputs/exercise02_extended_metrics.json`, has validation F1 0.792. The extra trees fitted the seen mail and did not improve the unseen mail.

Tighten `--max-depth`, raise `--min-samples-leaf`, or keep the simpler model when the validation metrics are tied. Exercise 10 repeats the comparison on EDR by limiting tree depth. Moving the threshold above or below 0.5 changes the precision and recall tradeoff on the same scores. It does not close a gap between training F1 and validation F1.

## 5. Four datasets

The exercises rotate through four jobs. Section 7 lists every CSV, including the challenge cohorts. This section is the job each main file is for.

**Phishing emails.** `datasets/phishing_emails.csv` has 6,000 rows, about 17 percent labelled phishing. You train a classifier, then inspect false positives and false negatives and try to improve detection. Legitimate marketing mail is included so that a model which alerts on ordinary commercial language will show that mistake.

**Endpoint telemetry.** `datasets/edr_events.csv` has 6,500 rows. You classify suspicious process behaviour and see where legitimate developer and server activity resembles malicious execution. Hosts do not cross the training, validation, and test groups.

**Alert triage.** `datasets/alert_triage.csv` has 3,500 rows. Regression predicts investigation time in minutes. The error is a number of minutes, reported as MAE and RMSE, not as a false positive.

**Network activity.** `datasets/network_activity.csv` has 5,000 rows. Clustering looks for groups of similar connection behaviour. Anomaly detection looks for sessions that sit far from the learned normal. A cluster id or a high anomaly score is not, by itself, a malicious verdict. Labels are withheld until after fitting, and only then used to interpret the result.

### Classification

The prediction is a class. In this lab, class 0 is the legitimate or benign case and class 1 is the malicious case.

| Algorithm              | How it learns                                                               | Security example                            | In this lab                                                          |
| ---------------------- | --------------------------------------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------- |
| Logistic regression    | One weight per feature, plus an intercept, converted to a class probability | Phishing email detection                    | `logistic`                                                           |
| Decision tree          | A sequence of feature splits                                                | Malicious versus benign processes           | `tree`                                                               |
| Random forest          | A vote across many decision trees                                           | EDR alert classification                    | `random_forest`                                                      |
| Gradient boosting      | Trees grown in sequence, each correcting earlier errors                     | Malware detection                           | `gradient_boosting`                                                  |
| Naive Bayes            | A probability model of each feature given the class                         | Email spam detection                        | `naive_bayes`, Gaussian naive Bayes                                  |
| K nearest neighbours   | The labels of the nearest stored training rows                              | Samples grouped by similar features         | `knn`                                                                |
| Support vector machine | A boundary between classes                                                  | Malicious versus benign PowerShell activity | `svm`, RBF kernel                                                    |
| Neural network         | Several layers of weighted sums and nonlinear functions                     | Malware or traffic classification           | `scripts/neural_network_intro.py` only, a four row XOR demonstration |
| XGBoost                | Gradient boosting with its own tree booster                                 | The same task as gradient boosting          | optional, install `xgboost`, then `--algorithm xgboost`              |

`run_classification.py` trains every row in that table except the neural network. The neural network exercise is there to show forward propagation and backpropagation on a toy problem, not to classify the CSVs.

### Regression

The prediction is a continuous value, such as 45.6 minutes of investigation time. `run_regression.py` reads `datasets/alert_triage.csv`.

| Algorithm                    | How it learns                                       | Security example                               | In this lab         |
| ---------------------------- | --------------------------------------------------- | ---------------------------------------------- | ------------------- |
| Linear regression            | A linear combination of the features                | Predict investigation time                     | `linear`            |
| Ridge regression             | Linear regression with a penalty on large weights   | Predict analyst workload                       | `ridge`             |
| Lasso regression             | Linear regression that can set some weights to zero | Keep a subset of features for a duration model | not included        |
| Decision tree regression     | Split rules that output a number                    | Estimate triage duration                       | `tree`              |
| Random forest regression     | The average of many regression trees                | Predict investigation workload                 | `random_forest`     |
| Gradient boosting regression | Sequential correction of numeric errors             | Estimate response duration                     | `gradient_boosting` |
| Support vector regression    | A fitted function with a tolerated error band       | Predict traffic volume                         | not included        |

Random forest is two different estimators. Classification uses `RandomForestClassifier`. Regression uses `RandomForestRegressor`. The algorithm family is shared. The thing being predicted is not.

### Unsupervised learning, no labels during fit

| Task                     | Question                                    | In this lab                                                                           |
| ------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------- |
| Clustering               | Which rows resemble each other              | `run_clustering.py`: `kmeans`, `dbscan`, `hierarchical`                               |
| Anomaly detection        | Which rows sit far from the learned normal  | `run_anomaly.py`: `isolation_forest`, `lof`, `one_class_svm`, `zscore`                |
| Dimensionality reduction | How to draw many features in two dimensions | PCA chart written by the clustering script, not a model you select with `--algorithm` |

### Additional tasks with the same data

One collection of endpoint process records can be asked four different questions.

- Classification: which processes are malicious?
- Regression: how long will each alert take to investigate?
- Clustering: which processes look alike?
- Anomaly detection: which processes are unusual relative to the rest?

Classification and regression are supervised. The training rows include a known answer, a class or a number, and `fit` uses that answer. Clustering and anomaly detection are unsupervised. `fit` sees features only. If a label exists, the script uses it after fitting, to measure the result, not to grow the model.

The lab scripts for those four questions are `run_classification.py`, `run_regression.py`, `run_clustering.py`, and `run_anomaly.py`.

## 6. Learning roadmap

| Level                         | Exercises      | Main activities                                                             |
| ----------------------------- | -------------- | --------------------------------------------------------------------------- |
| 1. Execute and observe        | 01, 02         | Read the CSV, fit first classifier, compare classifier families             |
| 2. Understand and investigate | 03, 04, 09     | Precision / recall / threshold / class imbalance / error inspection         |
| 3. Modify and improve         | 05, 06, 07, 10 | Regression errors, clustering, role-specific anomaly baselines, overfitting |
| 4. Broaden technique          | 11, 12, 13, 14 | Leakage, TF-IDF, cross-validation, drift, backpropagation                   |
| 5. Independently solve        | 08             | EDR holdout challenge: investigate and justify decisions without spoilers   |

**Suggested order**: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 09 → 10 → 11 → 12 → 13 → 14 → 08.

The standalone instructions are in `exercises/`. Each includes a single-variable experiment, what to observe, questions, comparisons, and a scenario. `solutions/exercise_solutions.md` is a short spoiler file, and `solutions/exercise_solutions_explained.md` is the longer worked explanation. Do not open either before attempting the independent challenge.

## 7. Dataset contents

Section 5 names the task. This table names the file.

| File                                             |  Rows | Purpose                                | Data characteristics                                                                    |
| ------------------------------------------------ | ----: | -------------------------------------- | --------------------------------------------------------------------------------------- |
| `datasets/phishing_emails.csv`                   | 6,000 | Classification and TF-IDF              | 17% malicious; difficult legitimate marketing and stealth phishing cases                |
| `datasets/edr_events.csv`                        | 6,500 | Classification and anomaly detection   | Users, servers and developers; deliberately overlapping rare benign/malicious telemetry |
| `datasets/alert_triage.csv`                      | 3,500 | Regression                             | Alert investigation minutes; nonlinear complexity with noise                            |
| `datasets/network_activity.csv`                  | 5,000 | Clustering and anomalies               | Unequal host populations, rare abnormal connections, noisy developer traffic            |
| `datasets/edr_events_challenge.csv`              | 6,500 | Generalisation / independent challenge | New developer-behaviour distribution; separate synthetic cohort                         |
| `datasets/alert_triage_challenge.csv`            | 3,500 | Regression drift                       | Harder target noise; separate cohort                                                    |
| `datasets/network_activity_challenge.csv`        | 5,000 | Anomaly drift                          | More legitimate anomalous-looking developer flows                                       |
| `datasets/phishing_emails_leakage_challenge.csv` | 6,000 | Explicit leakage demonstration         | Post-investigation feature correlated with answer; excluded from all standard pipelines |

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

## 8. Command reference

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

## 9. Technical safeguards and limitations

1. Every row and hostname is generated from NumPy pseudorandom draws. `.example` is reserved and non-routable, and no live indicators, real mail, hostnames or executable content are used.
2. `email_id`, `event_id`, `alert_id`, `session_id`, `hostname` and `source_host` are excluded from training features. Ground-truth labels are excluded from unsupervised training features. No feature is created from the target except in the **explicitly marked leakage challenge**.
3. Numeric imputation, scaling and categorical one-hot encoding are learned on training rows only in supervised/anomaly workflows. For TF-IDF, the vocabulary is also fit on training rows only. The scripts intentionally include missing values and irrelevant fields; these are not implementation errors.
4. `outputs/` and `charts/` are generated per experiment; running the same `--name` overwrites that experiment's outputs. Charts are `.png`, metrics `.json`, predictions `.csv`.
5. Validation can also be overfit by excessive repeated tuning. Use cross-validation on training for broad search; reserve validation for choosing a candidate and the final test for once-only confirmation. Challenge variants are separate synthetic samples, not evidence of real deployment performance.
6. Classifier `predict_proba()` values are model probability estimates, not guaranteed to be calibrated. Isolated anomalies are not confirmed threats. Precision, recall and alert rates from these synthetic sets are illustrative of ML behaviour, **not expected production detection performance**.
7. For clustering, the PCA chart depicts a linear 2D projection of higher-dimensional inputs; visual proximity need not correspond to all high-dimensional distances. K-means requires a chosen `k`; DBSCAN noise is not itself malicious.
8. Dataset generation regenerates the eight CSV files and **overwrites** their contents deterministically. Never regenerate after hand-editing a challenge CSV unless you have saved your changes separately.
9. XGBoost is not a core dependency: `python -m pip install xgboost` then use `--algorithm xgboost` optionally. Neural network lesson is deliberately implemented with NumPy rather than introducing another heavyweight dependency.

**Do not import these synthetic metrics into detection production.** They are designed as controlled training exercises with invented probabilities and distributions.

## 10. Cross-cohort drift and inference extension

To evaluate a **frozen model trained on the original EDR cohort** against a different synthetic deployment cohort (rather than retraining on it):

```bash
python scripts/evaluate_shift.py --algorithm random_forest --features basic --name shift_frozen_basic
python scripts/evaluate_shift.py --algorithm random_forest --features extended --name shift_frozen_extended
```

This fits feature preprocessing and the model using original **training rows only**, evaluates on old validation hosts, then applies the unchanged model and threshold to the entire separately generated challenge CSV. It writes `outputs/shift_frozen_basic_shift_metrics.json`, two prediction CSVs and a comparison chart. The new cohort is synthetic and not an unbiased estimate of any production deployment environment. This assesses cross-cohort performance drift rather than within-cohort model selection.

For a **permutation/impurity feature-importance** starting point, train `--algorithm random_forest`; inspect `charts/<name>_feature_importance.png`. The plotted impurity importance indicates model split utility, not causal attribution. For operational interpretability, compare errors across populations, avoid treating correlated predictors as independent causes and consider a separately validated permutation-importance experiment.

For **batch vs online learning**: these scripts implement batch training and batch inference from CSVs. A streaming/online endpoint detector would need an event-time feature pipeline, stateful baseline updates, model versioning, rollback, security review and delayed-label handling. Scikit-learn `partial_fit` supports incremental training for selected estimators (e.g. `SGDClassifier`), but the exercises do not silently enable it; online updates can poison a baseline if anomalous events are incorporated uncritically.
