import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


RANDOM_STATE = 42


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "raw" / "instagram_usage_lifestyle.csv"
    parser = argparse.ArgumentParser(
        description="Fit scaling, PCA and KMeans on every input row; save validation and cluster profiles."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="Path to input CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "outputs" / "final_eda"),
        help="Directory to save plots and analytics",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of clusters (default: 4 for descriptive segmentation, not a proven optimum)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute clustering metrics over a range of k values",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum k for evaluation (used with --evaluate)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Maximum k for evaluation (used with --evaluate)",
    )
    parser.add_argument(
        "--silhouette-sample",
        type=int,
        default=5000,
        help="Maximum rows used ONLY to estimate silhouette; model fitting always uses all rows",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True)


def validate_data(df: pd.DataFrame, out_dir: Path) -> None:
    missing_pct = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .to_frame("missing_pct")
    )
    save_csv(missing_pct, out_dir / "missing_pct.csv")

    dup_count = df.duplicated().sum()
    pd.DataFrame({"duplicate_rows": [dup_count]}).to_csv(
        out_dir / "duplicate_count.csv", index=False
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        neg_counts = (df[numeric_cols] < 0).sum().sort_values(ascending=False)
        neg_counts = neg_counts.to_frame("negative_count")
        save_csv(neg_counts, out_dir / "negative_counts.csv")

        zero_counts = (df[numeric_cols] == 0).sum().sort_values(ascending=False)
        zero_counts = zero_counts.to_frame("zero_count")
        save_csv(zero_counts, out_dir / "zero_counts.csv")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_work = df.copy()

    # Relative indices compare weekly posts with daily activity.
    # These are not literal posts per session; +1 smooths the denominator.
    df_work["creator_post_ratio"] = (
        df_work["posts_created_per_week"] / (df_work["sessions_per_day"] + 1)
    )

    df_work["broadcast_ratio"] = (
        df_work["posts_created_per_week"] /
        (df_work["stories_viewed_per_day"] + df_work["reels_watched_per_day"] + 1)
    )

    df_work["audience_ratio"] = (
        df_work["followers_count"] / (df_work["following_count"] + 1)
    )

    return df_work


def get_feature_columns():
    # Ten consumption/interaction measures and three content-creation
    # and audience ratios. Income is used only to describe the groups.
    activity_cols = [
        "user_engagement_score",
        "sessions_per_day",
        "daily_active_minutes_instagram",
        "average_session_length_minutes",
        "likes_given_per_day",
        "comments_written_per_day",
        "stories_viewed_per_day",
        "reels_watched_per_day",
        "time_on_feed_per_day",
        "time_on_reels_per_day",
    ]

    creator_cols = [
        "creator_post_ratio",
        "broadcast_ratio",
        "audience_ratio",
    ]

    return activity_cols, creator_cols


def compute_pca(XY_scaled: np.ndarray, out_dir: Path):
    pca_full = PCA().fit(XY_scaled)
    var_df = pd.DataFrame({
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_var": pca_full.explained_variance_ratio_,
    })
    var_df["cumulative_var"] = var_df["explained_var"].cumsum()
    var_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)

    # Two components retain ~72% of variance in this dataset and provide
    # an interpretable plane. This compression has not been shown to be optimal.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    XY_pca = pca.fit_transform(XY_scaled)
    return pca, XY_pca


def save_pca_loadings(pca: PCA, feature_names: list[str], out_dir: Path):
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=["PCA1", "PCA2"],
    )
    loadings.to_csv(out_dir / "pca_loadings.csv", index=True)


def evaluate_k_range(
    XY_pca: np.ndarray,
    k_min: int,
    k_max: int,
    out_dir: Path,
    max_silhouette_samples: int = 5000,
) -> None:
    if max_silhouette_samples < 2:
        raise ValueError("Silhouette requires at least 2 sampled rows.")
    rows = []
    silhouette_rows = min(len(XY_pca), max_silhouette_samples)
    for k in range(k_min, k_max + 1):
        print(f"Evaluating k={k} on all {len(XY_pca)} rows...", flush=True)
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(XY_pca)
        # Only this quadratic-cost metric is estimated on a fixed random subset.
        # Scaling, PCA, KMeans, and the other two metrics use every input row.
        sil = silhouette_score(
            XY_pca,
            labels,
            sample_size=silhouette_rows if silhouette_rows < len(XY_pca) else None,
            random_state=RANDOM_STATE,
        )
        rows.append({
            "k": k,
            "training_rows": len(XY_pca),
            "silhouette_rows": silhouette_rows,
            "silhouette": sil,
            "calinski_harabasz": calinski_harabasz_score(XY_pca, labels),
            "davies_bouldin": davies_bouldin_score(XY_pca, labels),
        })

    metrics_df = pd.DataFrame(rows).set_index("k")
    metrics_df.to_csv(out_dir / "cluster_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    metrics_df["silhouette"].plot(ax=axes[0], marker="o")
    axes[0].set_title(f"Silhouette (n={silhouette_rows}; higher is better)")
    axes[0].set_xlabel("k")

    metrics_df["calinski_harabasz"].plot(ax=axes[1], marker="o")
    axes[1].set_title("Calinski-Harabasz (higher is better)")
    axes[1].set_xlabel("k")

    metrics_df["davies_bouldin"].plot(ax=axes[2], marker="o")
    axes[2].set_title("Davies-Bouldin (lower is better)")
    axes[2].set_xlabel("k")

    plt.tight_layout()
    fig.savefig(out_dir / "cluster_metrics.png", dpi=150)
    plt.close(fig)


def cluster_and_plot(XY_pca: np.ndarray, k: int, out_dir: Path):
    # k=4 is a descriptive choice; metrics across k are saved separately.
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(XY_pca)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        XY_pca[:, 0],
        XY_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.6,
    )

    centers = kmeans.cluster_centers_
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        s=200,
        marker="X",
    )

    plt.xlabel("PCA1 (activity)")
    plt.ylabel("PCA2 (creator)")
    plt.title(f"KMeans clusters (k={k}) on PCA space")
    plt.tight_layout()
    plt.savefig(out_dir / "clusters_pca.png", dpi=150)
    plt.close()

    return labels


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray, out_dir: Path) -> None:
    df_seg = df.copy()
    df_seg["cluster"] = labels

    profile_cols = [
        "posts_created_per_week",
        "creator_post_ratio",
        "broadcast_ratio",
        "sessions_per_day",
        "daily_active_minutes_instagram",
    ]

    groups = df_seg.groupby("cluster")
    profiles = groups[profile_cols].mean()
    profiles.insert(0, "cluster_size", groups.size())
    profiles.insert(1, "cluster_share", profiles["cluster_size"] / len(df_seg))
    profiles.to_csv(out_dir / "cluster_profiles.csv")

    if "income_level" in df_seg.columns:
        income_share = (
            df_seg
            .groupby("cluster")["income_level"]
            .value_counts(normalize=True)
            .unstack()
        )
        income_order = ["Low", "Lower-middle", "Middle", "Upper-middle", "High"]
        income_share = income_share.reindex(columns=income_order)
        income_share.to_csv(out_dir / "income_distribution.csv")


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid")

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    ensure_output_dir(out_dir)

    df = pd.read_csv(input_path)
    input_rows = len(df)

    validate_data(df, out_dir)

    if not 2 <= args.k < input_rows:
        raise ValueError("k must be at least 2 and smaller than the dataset size.")
    if args.evaluate and not 2 <= args.k_min <= args.k_max < input_rows:
        raise ValueError("Evaluation requires 2 <= k-min <= k-max < dataset size.")
    if args.silhouette_sample < 2:
        raise ValueError("silhouette-sample must be at least 2.")

    activity_cols, creator_cols = get_feature_columns()
    source_cols = activity_cols + ["posts_created_per_week", "followers_count", "following_count"]
    missing = [c for c in source_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Retain every row. Select only needed columns to reduce memory use, keeping
    # income aligned with its original user for subsequent profile analysis.
    profile_cols = ["income_level"] if "income_level" in df.columns else []
    df_work = engineer_features(df[source_cols + profile_cols])
    del df
    print(f"Fitting scaler and PCA on all {len(df_work)} rows...", flush=True)

    required_cols = activity_cols + creator_cols

    XY = df_work[required_cols]
    scaler = StandardScaler()
    XY_scaled = scaler.fit_transform(XY)

    pca, XY_pca = compute_pca(XY_scaled, out_dir)
    save_pca_loadings(pca, required_cols, out_dir)

    if args.evaluate:
        evaluate_k_range(XY_pca, args.k_min, args.k_max, out_dir, args.silhouette_sample)

    print(f"Fitting final KMeans (k={args.k}) on all {len(df_work)} rows...", flush=True)
    labels = cluster_and_plot(XY_pca, args.k, out_dir)
    cluster_profiles(df_work, labels, out_dir)

    metadata = {
        "input": str(input_path.resolve()),
        "input_rows": input_rows,
        "training_rows": len(df_work),
        "training_sampled": False,
        "feature_columns": required_cols,
        "pca_components": 2,
        "k": args.k,
        "random_state": RANDOM_STATE,
        "evaluation_k": list(range(args.k_min, args.k_max + 1)) if args.evaluate else [],
        "silhouette_rows": min(input_rows, args.silhouette_sample) if args.evaluate else None,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Input rows: {input_rows}; training rows: {len(df_work)}; random_state: {RANDOM_STATE}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
