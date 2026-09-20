# The role of separate_clusters_3D.ipynb

Based on its contents, this notebook is the next experimental stage after
`final_eda.ipynb`. It repeats the basic preparation while extending the feature
set, preprocessing, and model families. The name `final_eda` identifies the exploratory
baseline notebook, not the chronologically latest experiment. The current
`src/pipeline.py` fits all input rows; the notebooks retain their exploratory samples. The exact file creation
history was not established in this review.

## Additions

1. A broader set of activity measures: messages, advertising, time spent in
   different sections, posts, notification responses, and audience size.
2. Clipping at the 99th percentile and `log1p` for selected counts.
3. One-hot encoding of account settings and content preferences. Numeric features
   are standardized, while binary columns remain unscaled.
4. Removal of numeric features with absolute correlations above 0.90. The saved
   run removed seven features, leaving a matrix with 35 columns.
5. PCA with three components, interactive Plotly charts, and a sweep over `k=2...10`.
6. Separate experiments using only `float64` features, all numeric features,
   numeric features with encoded categories, and finally UMAP + HDBSCAN.

These are substantive extensions: the author attempts to reduce the influence
of skewness and redundant features, include categorical data, and explore
another clustering approach.

## What the saved results show

The main 3D branch uses 10,000 rows, not the full dataset. Its three components
retain **59.73% of variance** in the expanded feature matrix. This cannot be
compared directly with the sampled `final_eda` notebook's 72.14%: both the features and their
transformations have changed.

| k | Silhouette (higher is better) | Calinski-Harabasz (higher is better) | Davies-Bouldin (lower is better) |
|---:|---:|---:|---:|
| 2 | 0.431 | 8099.8 | 0.986 |
| 3 | 0.377 | 8516.3 | 0.960 |
| 4 | 0.324 | 7332.2 | 1.107 |

Among the tested values, silhouette favors two groups, while the other two metrics
favor three. Choosing four groups again requires a substantive explanation.
Mean time spent in the four groups is approximately **72.0, 285.5, 8.5, and 154.8
minutes per day**. The primary interpretation remains related to activity intensity.

The silhouette scores of 0.324 here and 0.409 in the sampled `final_eda` notebook are computed
in different spaces. They are insufficient to establish that the new model is
better or worse. A comparison requires the same users, a consistent evaluation
criterion, and stability checks. The later UMAP/HDBSCAN branch lacks a completed
numerical comparison with the main model and an interpretation of its groups
using the original features.

## Remaining issues

- The education and privacy mappings in the all-features and UMAP branches are
  incorrect: they expect `Bachelor`, `Master`, and `Low/Medium/High`, whereas the
  CSV contains `Bachelor’s`, `Master’s`, `Private/Public/Friends only`, and other
  values. The saved output confirms that `education_level_ord` and
  `privacy_setting_level_ord` were dropped as entirely missing. This is information
  loss caused by a transformation error, not a feature-selection result.
- Income becomes an input feature in those branches. This changes the task
  relative to the behavioral pipeline, where income is used only to describe the
  resulting groups independently.
- Selecting only `float64` features depends on storage type rather than meaning.
- Some columns with the `_log` suffix are only clipped: the logarithm is applied
  only to columns listed in `log_transform_cols`.
- Preprocessing, feature selection, dimensionality, and the algorithm change
  simultaneously. The contribution of each change cannot be isolated.
- There is no stability analysis or shared criterion for selecting among branches.
  HDBSCAN results should explicitly report the number of groups, noise fraction,
  cluster sizes, and profiles.
- A sequential rerun of all cells and a written conclusion are still needed.

The current revision preserves the experimental algorithms and their results.
It corrects the sample-size description and removes the unused LOF block, which
only returned labels with a preset `contamination=0.02` and did not affect fitting.
Correcting the mappings and rerunning every experimental branch remain separate
tasks; those results should not be presented as already verified.
