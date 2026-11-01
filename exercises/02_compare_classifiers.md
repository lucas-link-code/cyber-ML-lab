# EXERCISE NAME: 02 | Same emails, different classification algorithms

**OBJECTIVE:** Compare inductive biases and observe that the algorithm affects score quality.

**SCENARIO:** A security team wants to know whether a simple linear decision boundary is sufficient for phishing telemetry.

**DATASET:** `datasets/phishing_emails.csv`, same validation split in all runs.

**ALGORITHM:** Logistic regression, decision tree, Random Forest, Gaussian Naive Bayes, KNN, SVM, Gradient Boosting.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/run_classification.py --algorithm logistic --features basic --name alg_logistic
python scripts/run_classification.py --algorithm tree --features basic --name alg_tree
python scripts/run_classification.py --algorithm random_forest --features basic --name alg_forest
python scripts/compare_models.py alg_logistic alg_tree alg_forest --name compare_classifier_families
```

**WHAT THE COMMAND DOES:** Fits three different algorithms on identical input features and held-out row assignments. Comparison reads saved metric JSON, not freshly trained models.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: `experiment | precision | recall | f1 | roc_auc | train_f1`, with differing values for each model.

**WHAT TO OBSERVE:** Trees make conditional splits; Random Forest averages multiple trees; logistic regression learns a weighted linear log-odds score. Compare validation F1 and ROC/PR-AUC, and examine whether the training-versus-validation F1 gap grows.

**INVESTIGATION:** 1. Which model gave the best validation PR-AUC? 2. Did the most complex model also give the best precision?

**IMPROVEMENT:** Change only algorithm: `python scripts/run_classification.py --algorithm naive_bayes --features basic --name alg_nb` (Gaussian NB models conditional distributions with a strong independence assumption). Optional: `--algorithm knn`, `--algorithm svm`, `--algorithm gradient_boosting` and `--algorithm xgboost` after installing optional `xgboost`.

**COMPARISON:** python scripts/compare_models.py alg_logistic alg_tree alg_forest alg_nb --name compare_four_algorithms

**LESSON:** Model families can learn different boundaries even when fed identical features, splits and labels. The default threshold does not define ranking quality.
