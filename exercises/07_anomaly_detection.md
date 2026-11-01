# EXERCISE NAME: 07 | Developer workstation false-positive problem

**OBJECTIVE:** Improve anomaly-alert precision and assess legitimate developer activity separately.

**SCENARIO:** Network anomaly detection generates too many alerts from normal but unusual developer workstations.

**DATASET:** `datasets/network_activity.csv`; role labels used for separate baselines, truth labels for evaluation only.

**ALGORITHM:** Isolation Forest, with LOF/One-Class SVM/Z-score extensions.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/run_anomaly.py --algorithm isolation_forest --features raw --baseline global --contamination 0.08 --name anomaly_global
python scripts/inspect_errors.py --experiment anomaly_global --type fp --rows 12
```

**WHAT THE COMMAND DOES:** Trains an unsupervised detector on host-exclusive training sessions, calibrates a decision cutoff from **training scores only**, and evaluates validation sessions using withheld synthetic anomaly labels.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: `fp=...`, `precision=0.xx`, a role table of alerts and false positives; `anomaly_score` is not a probability of malware.

**WHAT TO OBSERVE:** Look at the developer-vs-user alert counts and false positives. Global baselines can penalise benign role-specific behaviour.

**INVESTIGATION:** 1. What evidence would support a population mismatch rather than maliciousness? 2. Is the contamination argument a guarantee of real attack prevalence?

**IMPROVEMENT:** Change **only population baseline**: `python scripts/run_anomaly.py --features raw --baseline per_role --contamination 0.08 --name anomaly_role`. Then change **only the threshold quantile**: `python scripts/run_anomaly.py --features raw --baseline per_role --contamination 0.05 --name anomaly_role_strict`. Further optional experiment changes only the features: `--features behavioral`.

**COMPARISON:** python scripts/compare_models.py anomaly_global anomaly_role anomaly_role_strict --name compare_anomaly_baselines
python scripts/inspect_errors.py --experiment anomaly_role_strict --type fp --rows 12

**LESSON:** Learn population-specific baselines when roles differ materially; reducing contamination generally lowers alerts but may increase missed anomalies. Compare FPR, FP, recall and alerts, not just precision.

**INTERVIEW CONNECTION:** “Why does an unsupervised network model alert on legitimate developer builds? How do you tune it without treating anomalies as confirmed threats?”
