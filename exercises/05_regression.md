# EXERCISE NAME: 05 | Predict EDR alert investigation time

**OBJECTIVE:** Lower continuous prediction MAE while diagnosing omitted variables and nonlinear relationships.

**SCENARIO:** A SOC staffing planner needs to estimate the number of minutes required to triage a new alert.

**DATASET:** `datasets/alert_triage.csv`.

**ALGORITHM:** Linear regression, then Gradient Boosting Regressor.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/inspect_data.py --dataset triage --rows 5
python scripts/run_regression.py --algorithm linear --features basic --name triage_basic
```

**WHAT THE COMMAND DOES:** Predicts minutes rather than 0/1. The baseline sees workload, host count and telemetry count, but not investigation complexity or severity.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: `MAE=30-40 minutes`, `RMSE > MAE`, `R2=0.xx`; output lists actual, predicted, and residual minutes.

**WHAT TO OBSERVE:** A positive residual means the true investigation took longer than predicted. Look for large positive/negative residuals in `charts/triage_basic_residuals.png`.

**INVESTIGATION:** 1. Which important drivers are absent from the basic model? 2. Why does a few very large errors increase RMSE more than MAE?

**IMPROVEMENT:** First change **only features**: `python scripts/run_regression.py --algorithm linear --features extended --name triage_extended`. Then change **only the algorithm** while keeping extended features: `python scripts/run_regression.py --algorithm gradient_boosting --features extended --name triage_boosting`.

**COMPARISON:** python scripts/compare_models.py triage_basic triage_extended triage_boosting --name compare_triage_improvements
python scripts/inspect_errors.py --experiment triage_boosting --type largest_residuals --rows 10

**LESSON:** Adding relevant variables reduces omitted-variable error; a nonlinear learner can capture interactions between complexity, severity and workload. Better model complexity is not automatically better on held-out data.

**INTERVIEW CONNECTION:** “Your alert-duration regression has high MAE. How would you determine whether features or algorithm choice are responsible?”
