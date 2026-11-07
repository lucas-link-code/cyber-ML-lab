# EXERCISE NAME: 04 | Missed malicious endpoint events

**OBJECTIVE:** Increase EDR malicious-class recall above 0.55 while keeping validation FPR at or below 0.10.

**SCENARIO:** An EDR detector is quiet but fails to catch a substantial portion of malicious process events.

**DATASET:** `datasets/edr_events.csv`; host-exclusive train/validation/test.

**ALGORITHM:** Logistic regression with the **basic** EDR feature set.

**COMMAND:** Run from the lab folder with your virtual environment activated.

```bash
python scripts/run_classification.py --dataset edr --algorithm logistic --features basic --name edr_default
python scripts/inspect_errors.py --experiment edr_default --type fn --rows 12
```

**WHAT THE COMMAND DOES:** Fits a simple EDR classifier using four numeric/binary features; prints missed malicious events (false negatives). No host appears in multiple splits.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: `recall≈0.2` and `FPR≈0.02`; low alert volume can coexist with poor detection coverage.

**WHAT TO OBSERVE:** Inspect missed cases: process prevalence, command length, unsigned flag and connection count. Thresholds change decision volume but cannot create a missing useful feature.

**INVESTIGATION:** 1. Why can a conservative threshold hide malicious activity? 2. If you lower the threshold, what happens to FPR?

**IMPROVEMENT:** Change **only threshold**: `python scripts/run_classification.py --dataset edr --algorithm logistic --features basic --threshold 0.25 --name edr_low_threshold`.

**COMPARISON:** python scripts/compare_models.py edr_default edr_low_threshold --name edr_recall_tradeoff
python scripts/inspect_errors.py --experiment edr_low_threshold --type fp --rows 12

**LESSON:** Recall is the proportion of all truly malicious events caught. A lower cutoff typically increases TP but also FP; a good production operating point requires a quantified FPR/analyst-capacity ceiling.
