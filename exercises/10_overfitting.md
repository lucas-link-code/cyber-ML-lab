# EXERCISE NAME: 10 | Training performance is not generalisation

**OBJECTIVE:** Detect and reduce an overfitting gap with a single Random Forest capacity parameter.

**SCENARIO:** A model appears excellent on already-seen events but performs badly on new endpoint hosts.

**DATASET:** `datasets/edr_events.csv`.

**ALGORITHM:** Random Forest binary classification.

**COMMAND:** Run from the lab folder with your virtual environment activated.

```bash
python scripts/run_classification.py --dataset edr --algorithm random_forest --features basic --name rf_unrestricted
```

**WHAT THE COMMAND DOES:** Fits 100 unrestricted trees and prints `train_f1` and validation `f1` alongside FP and FN.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: `train_f1≈1.0` and much smaller validation `f1`; training-set perfection does not prove deployment accuracy.

**WHAT TO OBSERVE:** Compare train and validation F1. Is high tree complexity memorising the synthetic noise in weak features?

**INVESTIGATION:** 1. Why is a near-perfect training F1 suspicious? 2. Can a shallower tree still underfit?

**IMPROVEMENT:** Change **only maximum depth**: `python scripts/run_classification.py --dataset edr --algorithm random_forest --features basic --max-depth 8 --name rf_depth8`. If this worsens validation, report the result rather than assuming success.

**COMPARISON:** python scripts/compare_models.py rf_unrestricted rf_depth8 --name compare_overfitting

**LESSON:** Limiting tree depth reduces effective model capacity but may help or hurt validation F1. For robust selection use cross-validation on training hosts and reserve a final holdout.
