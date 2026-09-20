import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "raw" / "instagram_usage_lifestyle.csv"
    parser = argparse.ArgumentParser(
        description="Universal clustering: numeric-only scaling, PCA(3D), KMeans, and diagnostics."
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
        default="outputs",
        help="Directory to save plots and analytics",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of clusters for KMeans",
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
        help="Max sample size for silhouette score (speeds up large datasets)",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.empty:
        raise ValueError("No numeric columns available for clustering.")
    return numeric_df


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    medians = df.median(numeric_only=True)
    return df.fillna(medians)


def compute_pca(X_scaled: np.ndarray, out_dir: Path, feature_names: list[str]):
    pca_full = PCA().fit(X_scaled)
    var_df = pd.DataFrame({
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_var": pca_full.explained_variance_ratio_,
    })
    var_df["cumulative_var"] = var_df["explained_var"].cumsum()
    var_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)

    if X_scaled.shape[1] < 3:
        raise ValueError("Need at least 3 numeric features for a 3D PCA plot.")

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=["PCA1", "PCA2", "PCA3"],
    )
    loadings.to_csv(out_dir / "pca_loadings.csv", index=True)

    return pca, X_pca


def evaluate_k_range(
    X_pca: np.ndarray,
    k_min: int,
    k_max: int,
    out_dir: Path,
    max_silhouette_samples: int = 5000,
) -> None:
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_pca)
        n_samples = X_pca.shape[0]
        if n_samples > max_silhouette_samples:
            sil = silhouette_score(
                X_pca,
                labels,
                sample_size=max_silhouette_samples,
                random_state=42,
            )
        else:
            sil = silhouette_score(X_pca, labels)

        rows.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": calinski_harabasz_score(X_pca, labels),
            "davies_bouldin": davies_bouldin_score(X_pca, labels),
        })

    metrics_df = pd.DataFrame(rows).set_index("k")
    metrics_df.to_csv(out_dir / "cluster_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    metrics_df["silhouette"].plot(ax=axes[0], marker="o")
    axes[0].set_title("Silhouette (higher is better)")
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


def cluster_and_plot_3d(X_pca: np.ndarray, k: int, out_dir: Path):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.65,
    )

    centers = kmeans.cluster_centers_
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c="black",
        s=120,
        marker="X",
    )

    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.set_title(f"KMeans clusters (k={k}) on 3D PCA space")
    plt.tight_layout()
    plt.savefig(out_dir / "clusters_pca_3d.png", dpi=150)
    plt.close(fig)

    return labels


def main():
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    ensure_output_dir(out_dir)

    df = pd.read_csv(input_path)

    numeric_df = select_numeric(df)
    numeric_df = impute_numeric(numeric_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    _, X_pca = compute_pca(X_scaled, out_dir, numeric_df.columns.tolist())

    if args.evaluate:
        evaluate_k_range(
            X_pca,
            args.k_min,
            args.k_max,
            out_dir,
            max_silhouette_samples=args.silhouette_sample,
        )

    cluster_and_plot_3d(X_pca, args.k, out_dir)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
