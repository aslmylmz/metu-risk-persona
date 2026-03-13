"""
METU Risk Persona — K-Means Clustering Pipeline
====================================================
Features: Demographics + DOSPERT-30 + BART
Usage:
  python clustering_pipeline.py --data participants.csv
  python clustering_pipeline.py --data synthetic_metu_60.csv --output results/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ── Configuration ────────────────────────────────────────────────────────

NOMINAL_COLS = []
BINARY_COLS = []
CONTINUOUS_COLS = [
    
    "dospert_financial",
    "bart_risk_sensitivity",'bart_patience_normalized','bart_impulsivity_index','bart_rng_normalized_pumps'    # CV of pump counts across balloons (strategy variability)
]

FEATURE_GROUPS = {
    "Demographics": ["degree", "dorm", "employed", "prior_task",
                     "age", "grad_months"],
    "DOSPERT": ["dospert_financial", "dospert_health_safety", "dospert_recreational",
                "dospert_ethical", "dospert_social"],
    "BART": ["bart_adaptive_strategy", "bart_rng_normalized_pumps", "bart_impulsivity_index",
             "bart_patience_normalized", "bart_mean_latency", "bart_between_consistency"],
}

PALETTE = ["#2E86C1", "#E74C3C", "#27AE60", "#8E44AD", "#E67E22", "#1ABC9C"]

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 10,
})


# ── Preprocessing ────────────────────────────────────────────────────────

def preprocess(filepath: str):
    df_raw = pd.read_csv(filepath)
    df_work = df_raw.drop(columns=["participant_id"])

    ohe = OneHotEncoder(sparse_output=False, drop="first")
    nominal_encoded = ohe.fit_transform(df_work[NOMINAL_COLS])
    nominal_names = ohe.get_feature_names_out(NOMINAL_COLS).tolist()

    df_numeric = df_work[BINARY_COLS + CONTINUOUS_COLS].copy()
    df_ohe = pd.DataFrame(nominal_encoded, columns=nominal_names, index=df_numeric.index)
    df_combined = pd.concat([df_numeric, df_ohe], axis=1)

    # Correlation check
    corr = df_combined.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [(c, r, corr.loc[r, c])
                 for c in upper.columns for r in upper.index
                 if upper.loc[r, c] > 0.70]
    if high_corr:
        print(f"\n  High correlations (|r| > 0.70):")
        for c1, c2, r in high_corr:
            print(f"    {c1} <-> {c2}: r={r:.3f}")

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_combined),
        columns=df_combined.columns,
        index=df_combined.index,
    )

    return df_raw, df_scaled


# ── Optimal K ────────────────────────────────────────────────────────────

def find_optimal_k(X, k_range=range(2, 4)):
    inertias, silhouettes = [], {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=2000)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes[k] = silhouette_score(X, labels)
    best_k = max(silhouettes, key=silhouettes.get)
    return best_k, inertias, silhouettes


def plot_selection(inertias, silhouettes, best_k, output_dir):
    ks = list(silhouettes.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ks, inertias, "o-", color=PALETTE[0], linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia (Within-Cluster SS)", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
    ax1.set_xticks(ks)

    scores = [silhouettes[k] for k in ks]
    colors = [PALETTE[1] if k == best_k else PALETTE[0] for k in ks]
    bars = ax2.bar(ks, scores, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Analysis", fontsize=14, fontweight="bold")
    ax2.set_xticks(ks)
    for bar, s in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{s:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "01_k_selection.png", dpi=200, bbox_inches="tight")
    plt.close()


# ── Cluster Profiling ────────────────────────────────────────────────────

def profile_clusters(df_raw, labels, output_dir):
    df = df_raw.copy()
    df["cluster"] = labels

    profile_cols = BINARY_COLS + CONTINUOUS_COLS
    available = [c for c in profile_cols if c in df.columns]

    centroids = df.groupby("cluster")[available].mean()
    z_centroids = (centroids - df[available].mean()) / df[available].std()

    fig, ax = plt.subplots(figsize=(14, max(6, len(available) * 0.4)))
    cmap = LinearSegmentedColormap.from_list("riskmap", ["#2E86C1", "#FAFAFA", "#E74C3C"])
    im = ax.imshow(z_centroids.T.values, cmap=cmap, aspect="auto", vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(z_centroids)))
    ax.set_xticklabels([f"Cluster {c}\n(n={(df['cluster']==c).sum()})"
                        for c in z_centroids.index], fontsize=11)
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(available, fontsize=9)

    for i in range(len(available)):
        for j in range(len(z_centroids)):
            val = z_centroids.T.values[i, j]
            color = "white" if abs(val) > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

   
    ax.set_title("Cluster Profiles (Z-scored deviations from overall mean)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.colorbar(im, ax=ax, label="Z-score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "02_cluster_profiles.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nCluster profiles (top features |z| > 0.3):")
    for c in z_centroids.index:
        row = z_centroids.loc[c]
        top = row[row.abs() > 0.3].sort_values(key=abs, ascending=False)
        n_c = (df["cluster"] == c).sum()
        print(f"\n  Cluster {c} (n={n_c}):")
        for feat, z in top.head(10).items():
            print(f"    {'↑' if z > 0 else '↓'} {feat}: z={z:.2f} (mean={centroids.loc[c, feat]:.2f})")

    print("\nNominal distributions:")
    for col in NOMINAL_COLS:
        if col in df.columns:
            ct = pd.crosstab(df["cluster"], df[col], normalize="index") * 100
            print(f"\n  {col}:\n  {ct.round(1).to_string()}")

    return centroids, z_centroids


# ── PCA ──────────────────────────────────────────────────────────────────

def plot_pca(df_scaled, labels, output_dir):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(df_scaled.values)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, c in enumerate(sorted(set(labels))):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=PALETTE[i % len(PALETTE)], s=60, alpha=0.7,
                   edgecolors="white", linewidth=0.5,
                   label=f"Cluster {c} (n={mask.sum()})")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("PCA Projection — K-Means Clusters", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "03_pca_projection.png", dpi=200, bbox_inches="tight")
    plt.close()

    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"],
                            index=df_scaled.columns)

    print(f"\nPCA: {pca.explained_variance_ratio_.sum()*100:.1f}% variance explained")
    for pc in ["PC1", "PC2"]:
        top = loadings[pc].abs().nlargest(5)
        print(f"\n  {pc}:")
        for feat in top.index:
            v = loadings.loc[feat, pc]
            print(f"    {'+' if v > 0 else '-'} {feat}: {v:.3f}")

    return pca, loadings


# ── Export ───────────────────────────────────────────────────────────────

def export(df_raw, labels, best_k, silhouettes, z_centroids, loadings, output_dir):
    df_export = df_raw.copy()
    df_export["cluster"] = labels
    df_export.to_csv(output_dir / "clustered_participants.csv", index=False)

    with open(output_dir / "summary.txt", "w") as f:
        f.write("METU Risk Persona — Clustering Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Algorithm: K-Means\n")
        f.write(f"Optimal k: {best_k} (by silhouette)\n")
        f.write(f"Silhouette: {silhouettes[best_k]:.4f}\n\n")

        for k, s in silhouettes.items():
            f.write(f"  k={k}: {s:.4f}{' <- best' if k == best_k else ''}\n")

        f.write(f"\nCluster sizes:\n")
        df = df_export
        for c in sorted(df["cluster"].unique()):
            n = (df["cluster"] == c).sum()
            f.write(f"  Cluster {c}: n={n} ({n/len(df)*100:.1f}%)\n")

        f.write(f"\nTop features per cluster (|z| > 0.3):\n")
        for c in z_centroids.index:
            row = z_centroids.loc[c]
            top = row[row.abs() > 0.3].sort_values(key=abs, ascending=False)
            f.write(f"\n  Cluster {c}:\n")
            for feat, z in top.head(10).items():
                f.write(f"    {'↑' if z > 0 else '↓'} {feat}: z={z:.2f}\n")

        f.write(f"\nPCA loadings:\n")
        for pc in ["PC1", "PC2"]:
            f.write(f"\n  {pc}:\n")
            top = loadings[pc].abs().nlargest(5)
            for feat in top.index:
                v = loadings.loc[feat, pc]
                f.write(f"    {'+' if v > 0 else '-'} {feat}: {v:.3f}\n")

        f.write(f"\nNominal distributions:\n")
        for col in NOMINAL_COLS:
            if col in df.columns:
                ct = pd.crosstab(df["cluster"], df[col], normalize="index") * 100
                f.write(f"\n  {col}:\n{ct.round(1).to_string()}\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="METU Risk Persona — K-Means Clustering")
    parser.add_argument("--data", type=str, default="participants.csv")
    parser.add_argument("--output", type=str, default="clustering_results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw, df_scaled = preprocess(args.data)
    print(f"Loaded {len(df_raw)} participants, {df_scaled.shape[1]} features")

    best_k, inertias, silhouettes = find_optimal_k(df_scaled.values)
    print(f"Optimal k: {best_k} (silhouette={silhouettes[best_k]:.4f})")
    plot_selection(inertias, silhouettes, best_k, output_dir)

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    labels = km.fit_predict(df_scaled.values)

    centroids, z_centroids = profile_clusters(df_raw, labels, output_dir)
    pca, loadings = plot_pca(df_scaled, labels, output_dir)

    export(df_raw, labels, best_k, silhouettes, z_centroids, loadings, output_dir)

    print(f"\nOutputs saved to {output_dir}/")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
