# Comparing the two clustering scripts

## Verdict

`pipeline.py` is the stronger primary portfolio artifact for behavioral user
segmentation. It has an explicit feature rationale, data-quality reports,
interpretable cluster profiles, sizes, shares, income comparisons, and run
metadata. `numeric_pca3d_clustering.py` is a useful generic exploratory baseline.
This is a judgment about task fit and completeness, not proof that one partition
is statistically or commercially superior.

Both scripts fit StandardScaler, PCA, and KMeans on **every input row**.
Neither reduces the training dataset. Both default to estimating silhouette on
at most 5,000 randomly selected observations; the other two clustering metrics
use all rows. Metric sampling is separate from model fitting.

## Implementation comparison

| Aspect | pipeline.py | numeric_pca3d_clustering.py |
|---|---|---|
| Purpose | Behavioral segmentation | Exploration of all numeric columns |
| Features | 10 activity measures and 3 engineered ratios | All numeric columns; 38 in the saved dataset experiment |
| User ID | Excluded from model features | Included because it is numeric |
| Categoricals | Income retained for profiling, excluded from training | Excluded from training; no categorical profiling |
| Missing values | Reports missingness; no automatic imputation | Drops all-missing numeric columns and imputes medians |
| Scaling and PCA | StandardScaler, 2 components | StandardScaler, 3 components |
| Clustering | KMeans, k=4 by default, n_init=20, seed=42 | Same algorithm and defaults |
| Interpretation | Mean behavior, cluster size/share, income composition | PCA weights and a 3D scatter plot |
| Diagnostics | Quality reports, optional k sweep, explicit training/metric row counts | Optional k sweep, no run metadata |
| Portability | Requires the Instagram feature schema | Can work with other CSVs with enough numeric columns |

## Why the behavioral pipeline is stronger for this project

The feature set follows the stated question: differences in consumption and
content creation. IDs do not enter distance calculations. Income stays attached
to each user and is used only after clustering. Profiles translate cluster
labels into quantities that can be interpreted, rather than relying on a plot.
The metadata and metric tables explicitly distinguish full-data training from
sampled silhouette estimation.

The generic script has two practical advantages: median imputation and less
dependence on a particular column schema. However, selecting every numeric
column also mixes demographics, lifestyle, account history, activity, and an
arbitrary user identifier without explaining their relevance or relative weight.
The saved user-ID PCA weights are small, so its inclusion should not be claimed
to have dominated this particular result; it remains a feature-selection flaw.

## What the results can and cannot establish

Use the current [pipeline metrics](../outputs/final_eda/cluster_metrics.csv) and
[run metadata](../outputs/final_eda/run_metadata.json) for the full-data behavioral
run. The existing [numeric-only metrics](../outputs/cluster_metrics.csv) and
[PCA variance](../outputs/pca_explained_variance.csv) are saved results from the
separate numeric experiment; that experiment was not rerun for this comparison.

| Result | Behavioral pipeline, current full-data run | Numeric script, saved experiment |
|---|---:|---:|
| PCA variance retained | 72.42% (2 components) | 42.79% (3 components) |
| Silhouette at k=2 | 0.484 | 0.468 |
| Silhouette at k=4 | 0.415 | 0.327 |

The behavioral representation retains 72.42% of variance in two components.
The saved numeric experiment retains 42.79% in three components. These percentages
refer to different feature matrices and do not provide a shared accuracy measure.
Likewise, silhouette scores in the two PCA spaces are not directly comparable as
evidence of a better model. A 3D plot or a larger feature count is not evidence
of stronger segmentation.

Both scripts still require a justified choice of k, stability checks across
seeds and resamples, and evaluation against a meaningful downstream objective
or a common, explicitly chosen evaluation space. Both train and score the
partitions on the same input population; internal clustering metrics do not
establish future business value. The behavioral ratios also mix daily and weekly
quantities, and correlated activity measures can dominate distances.

Keep `pipeline.py` as the main project entry point and describe the numeric
script as an alternative baseline. A fair next comparison would remove the ID
from the numeric baseline, use identical evaluation users, inspect stability and
profiles, and compare against a simple grouping by daily time spent in the app.
