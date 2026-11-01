# EXERCISE NAME: 06 | Network activity: scaling and cluster shape

**OBJECTIVE:** Explore K-means, feature magnitudes, elbow selection and PCA without assigning threat labels during training.

**SCENARIO:** The network team asks whether connection behaviours form meaningful groups, including developer and server activity.

**DATASET:** `datasets/network_activity.csv`; `known_anomaly_label` is **post-hoc only**.

**ALGORITHM:** K-means; optionally DBSCAN and hierarchical agglomerative clustering.

**COMMAND:** Run from the `cyber_ml_lab` folder with your virtual environment activated.

```bash
python scripts/run_clustering.py --features raw --scale no --k 4 --elbow --name km_unscaled
```

**WHAT THE COMMAND DOES:** Fits four K-means clusters to numeric network telemetry, then computes silhouette on a sample, PCA 2D projection and an elbow curve. Unscaled high-magnitude features dominate Euclidean distances.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: cluster sizes `0:... 1:...`, `silhouette_sample=0.xx`; chart `charts/km_unscaled_pca_clusters.png`.

**WHAT TO OBSERVE:** Compare groups by role/activity. The numbers `0...k-1` are arbitrary identifiers and **do not mean benign or malicious**. A good-looking PCA plot is not ground truth.

**INVESTIGATION:** 1. Which numeric fields have vastly different units? 2. Why is it wrong to label cluster 3 “malware” solely because some known anomalies occur there?

**IMPROVEMENT:** Change **only scaling**: `python scripts/run_clustering.py --features raw --scale yes --k 4 --elbow --name km_scaled`. Then change **only the feature set**: `python scripts/run_clustering.py --features behavioral --scale yes --k 4 --elbow --name km_behavioral`.

**COMPARISON:** python scripts/compare_models.py km_unscaled km_scaled km_behavioral --name compare_clustering

**LESSON:** K-means uses distances; `StandardScaler` changes geometry. Dropping weak signals such as raw port numbers can change clusters. The elbow is a heuristic; silhouette need not always improve. For DBSCAN try `--algorithm dbscan --eps 1.6 --min-samples 12`, and for hierarchical use `--algorithm hierarchical --k 4`.

**INTERVIEW CONNECTION:** “Why did the K-means cluster assignments change after scaling byte counts and frequencies?”
