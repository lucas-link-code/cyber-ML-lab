# Runtime verification and reproducibility

The package was generated and command-line tested in this environment on the generated synthetic datasets.
These are **actual observed validation / exploratory metrics**, not predicted or illustrative figures. The specific values may differ slightly between library versions.

- Python: `3.13.5`
- pandas: `2.2.3`
- NumPy: `2.3.5`
- scikit-learn: `1.8.0`
- matplotlib: `3.10.8`
- Fixed dataset generator base seed: `20261031`
- Supervised/anomaly random split seed: `42`; source/host grouping for EDR and network.

## Verified examples

| Problem | Reference experiment | Actual recorded validation/exploratory output |
| --- | --- | --- |
| Classification | `reference_phishing` | precision 0.842, recall 0.557, FP 21, FN 89, FPR 0.021 |
| Regression | `reference_triage` | MAE 31.29 min, RMSE 39.64 min, R² 0.228 |
| Clustering | `reference_clustering` | 4 clusters, silhouette 0.315 |
| Anomaly | `reference_anomaly` | precision 0.473, recall 0.595, FP 49, FN 30, FPR 0.053 |

Example predictions, metrics, and charts for these four runs are included in the `outputs/` and `charts/` folders.

## Tests performed

- Executed dataset generator (all 8 CSV files) and `scripts/verify_lab.py`.
- Re-generated all 8 datasets into a temporary directory and compared SHA-256 byte-for-byte: **PASS**.
- Executed core classification, regression, clustering and anomaly scripts: **PASS**.
- Executed supervised classifiers: Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes, KNN, SVM, Gradient Boosting: **PASS**.
- Executed regression: Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting: **PASS**.
- Executed clustering: K-means, DBSCAN, Hierarchical: **PASS**.
- Executed anomaly detection: Isolation Forest, Local Outlier Factor, One-Class SVM, Z-score; EDR and network role-grouped variants: **PASS**.
- Executed TF-IDF text feature pipeline, oversampling, balanced classification, GridSearchCV and RandomizedSearchCV: **PASS**.
- Executed calibration threshold experiments, experiment comparisons, label leakage demonstration, distribution-shift diagnostics, frozen-cohort evaluation and NumPy neural network: **PASS**.
- Optional XGBoost path is implemented but XGBoost is not required for the base lab and is not included in the core runtime verification.

## CSV SHA-256 checksums

```text
9298ccab77cd1967387b8e95c2ed059308986f6fed70b815f9eee331a455cd15  datasets/alert_triage.csv
f056e23765d5cf6611629cf5ee36c9fe54b8ad7610a8aa33fe630c8e364e0dab  datasets/alert_triage_challenge.csv
d831047dacbc0da02b685b1a9595bcecfcdab0ca7ba1e3663f47cb20727200c6  datasets/edr_events.csv
4b99ba464e016f65f365a8849bcf9f58b3bf8950b3fb6c9bb64894fa94a7db6b  datasets/edr_events_challenge.csv
067c40e6948c3d7e76ec1031d2682a784534deb46631c308a5ca61bbe0bd414a  datasets/network_activity.csv
7a41dc1bdfdc356c9812284f7b4590dc15c3f6720ed382bb8dbebd50e39ade2a  datasets/network_activity_challenge.csv
1abb4780c4819fc08ca8c6a3343d3a218e856cecc72a8e3b2375e03f09ab5b8c  datasets/phishing_emails.csv
50c1504a654a0ca9b7e4e7de864e56b54b59545b3f3eac2c46e55f42ad43f3f4  datasets/phishing_emails_leakage_challenge.csv
```

## Re-run validation

```bash
python scripts/verify_lab.py
python tests/verify_regeneration.py
python tests/run_smoke_tests.py
```

The smoke test writes files prefixed `_smoke_` into `outputs/` and `charts/`. It does not evaluate a final test split or require credentials/network services.
