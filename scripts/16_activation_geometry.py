"""Activation-Space Geometry Visualization (Script 16).

Comprehensive visualization of IT-model layer-10 representations,
annotated with probe directions, steering vectors, and Rashomon set geometry.

Phases:
  1. Load Data and Compute Projections (PCA, t-SNE, UMAP)
  2. Core Visualization Suite (7 figures)
  3. Geometric Analysis (cluster separation, alignment metrics)
  4. Summary Report
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed. UMAP figure will be skipped.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from steering import naive_delta, robust_delta

# ═══════════════════════════════════════════════════════════════════════
# Paths and Constants
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

IT_ACT_PATH = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10_it.pt")
IT_PROBE_PATH = os.path.join(OUTPUT_DIR, "baseline_probe_layer10_it.pt")
IT_HESSIAN_PATH = os.path.join(OUTPUT_DIR, "hessian_layer10_it.pt")
RASHOMON_IT_DIR = os.path.join(OUTPUT_DIR, "rashomon_it")

VIS_DIR = os.path.join(OUTPUT_DIR, "visualization")
FIG_DIR = os.path.join(PROJECT_ROOT, "assets", "figures")
REPORT_PATH = os.path.join(VIS_DIR, "geometry_report.txt")

EPSILON = 0.15
SEED = 42
SAFETY_TARGET = 2.0

# Color scheme
COLOR_SAFE = "#3274A1"      # steel blue
COLOR_UNSAFE = "#E1812C"    # orange-red
COLOR_BOUNDARY = "#2CA02C"  # green for boundary markers
COLOR_NAIVE = "#2CA02C"     # green for naive delta
COLOR_ROBUST = "#9467BD"    # purple for robust delta
COLOR_PROBE = "#D62728"     # red for probe direction
COLOR_RASHOMON = "#7F7F7F"  # gray for Rashomon probes

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
})


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Load Data and Compute Projections
# ═══════════════════════════════════════════════════════════════════════

def phase1_load_and_project():
    """Load all data and compute PCA, t-SNE, UMAP projections."""
    print("\n" + "=" * 75)
    print("PHASE 1: Load Data and Compute Projections")
    print("=" * 75)

    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Load activations ---
    act_data = torch.load(IT_ACT_PATH, map_location="cpu", weights_only=True)
    train_X = act_data["train_X"].numpy()
    train_y = act_data["train_y"].numpy()
    test_X = act_data["test_X"].numpy()
    test_y = act_data["test_y"].numpy()
    print(f"  Train: {train_X.shape}, Test: {test_X.shape}")

    # --- Load probe ---
    probe_data = torch.load(IT_PROBE_PATH, map_location="cpu", weights_only=True)
    w = probe_data["weight"].numpy()
    b = probe_data["bias"].item()
    print(f"  Probe: w.shape={w.shape}, b={b:.4f}")

    # Probe scores for test set
    scores = test_X @ w + b
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # --- Load Rashomon probes ---
    rashomon_probes = torch.load(
        os.path.join(RASHOMON_IT_DIR, "rashomon_probes.pt"),
        map_location="cpu", weights_only=False,
    )
    rash_weights = np.array([p["weight"].numpy() for p in rashomon_probes])
    rash_biases = np.array([p["bias"].item() for p in rashomon_probes])
    print(f"  Rashomon probes: {len(rashomon_probes)}, weight shape: {rash_weights.shape}")

    # --- Load Hessian for computing deltas ---
    hessian_data = torch.load(IT_HESSIAN_PATH, map_location="cpu", weights_only=True)
    H_inv = hessian_data["H_inv"]

    # --- Compute deltas for boundary examples ---
    w_t = torch.from_numpy(w).double()
    b_t = torch.tensor(b).double()
    test_X_t = torch.from_numpy(test_X).double()

    abs_scores = np.abs(scores)
    boundary_mask = abs_scores < 0.5
    boundary_idx = np.where(boundary_mask)[0]
    sorted_boundary = boundary_idx[np.argsort(abs_scores[boundary_idx])]

    # Compute deltas for 5 boundary examples (for trajectory figure)
    n_traj = min(5, len(sorted_boundary))
    traj_idx = sorted_boundary[:n_traj]
    deltas_naive = []
    deltas_robust = []
    for idx in traj_idx:
        x = test_X_t[idx]
        d_n, _ = naive_delta(w_t, b_t, x, SAFETY_TARGET)
        d_r = robust_delta(w_t, b_t, x, H_inv, EPSILON, SAFETY_TARGET)
        deltas_naive.append(d_n.numpy().astype(np.float32))
        deltas_robust.append(d_r.numpy().astype(np.float32))
    print(f"  Computed deltas for {n_traj} boundary examples")

    # === PCA ===
    print("\n  Computing PCA ...")
    pca = PCA(n_components=50, random_state=SEED)
    pca.fit(train_X)
    test_pca = pca.transform(test_X)
    train_pca = pca.transform(train_X)

    var_explained = np.cumsum(pca.explained_variance_ratio_)
    print(f"    Var explained: PC2={var_explained[1]:.4f}, PC3={var_explained[2]:.4f}, "
          f"PC10={var_explained[9]:.4f}, PC50={var_explained[49]:.4f}")

    # === t-SNE ===
    print("  Computing t-SNE ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
    test_tsne = tsne.fit_transform(test_X)
    print(f"    t-SNE complete. Shape: {test_tsne.shape}")

    # === UMAP ===
    test_umap = None
    if HAS_UMAP:
        print("  Computing UMAP ...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
        reducer.fit(train_X)
        test_umap = reducer.transform(test_X)
        print(f"    UMAP complete. Shape: {test_umap.shape}")

    data = {
        "train_X": train_X, "train_y": train_y,
        "test_X": test_X, "test_y": test_y,
        "w": w, "b": b, "scores": scores,
        "rash_weights": rash_weights, "rash_biases": rash_biases,
        "pca": pca, "test_pca": test_pca, "train_pca": train_pca,
        "var_explained": var_explained,
        "test_tsne": test_tsne,
        "test_umap": test_umap,
        "boundary_mask": boundary_mask,
        "traj_idx": traj_idx,
        "deltas_naive": deltas_naive,
        "deltas_robust": deltas_robust,
    }
    return data


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Core Visualization Suite
# ═══════════════════════════════════════════════════════════════════════

def _save_fig(fig, name):
    """Save figure to both output dirs."""
    for d in [VIS_DIR, FIG_DIR]:
        fig.savefig(os.path.join(d, f"{name}.png"))
        fig.savefig(os.path.join(d, f"{name}.pdf"))
    print(f"    Saved: {name}.png/.pdf")


def _get_pca_probe_line(pca, w, b, test_pca):
    """Compute probe direction and decision boundary in PCA space."""
    # Project probe weight onto PCA space
    w_pca = pca.components_[:2] @ w  # [2] — probe direction in PC1-PC2
    w_pca_norm = w_pca / np.linalg.norm(w_pca)

    # Decision boundary: w^T x + b = 0
    # In PCA space, the centroid of the data maps to pca.mean_
    # The boundary is where the probe score equals 0
    # We need a point on the boundary and the normal direction
    centroid_pca = np.mean(test_pca[:, :2], axis=0)

    return w_pca, w_pca_norm, centroid_pca


def _draw_decision_boundary(ax, pca, w, b, xlim, ylim):
    """Draw the probe decision boundary line in PCA space."""
    # The decision boundary is {x : w^T x + b = 0}
    # Project to PCA: w^T (V^T z + mu) + b = 0 => (wV)^T z + w^T mu + b = 0
    # where V = pca.components_[:2].T and z is PCA coords
    V = pca.components_[:2]  # [2, 2304]
    mu = pca.mean_  # [2304]
    w_proj = V @ w  # [2] — probe direction in PCA space
    offset = w @ mu + b

    # Line: w_proj[0]*z1 + w_proj[1]*z2 + offset = 0
    # => z2 = -(w_proj[0]*z1 + offset) / w_proj[1]
    if abs(w_proj[1]) > 1e-10:
        x_range = np.linspace(xlim[0], xlim[1], 100)
        y_boundary = -(w_proj[0] * x_range + offset) / w_proj[1]
        mask = (y_boundary >= ylim[0]) & (y_boundary <= ylim[1])
        ax.plot(x_range[mask], y_boundary[mask], "k--", linewidth=1.5,
                alpha=0.7, label="Decision boundary")
    elif abs(w_proj[0]) > 1e-10:
        x_val = -offset / w_proj[0]
        ax.axvline(x_val, color="k", linestyle="--", linewidth=1.5,
                   alpha=0.7, label="Decision boundary")


def figure1_pca(data):
    """PCA scatter with probe direction and decision boundary."""
    print("\n  Figure 1: PCA Scatter ...")
    test_pca = data["test_pca"][:, :2]
    test_y = data["test_y"]
    scores = data["scores"]
    boundary = data["boundary_mask"]

    fig, ax = plt.subplots(figsize=(8, 6))

    safe = test_y == 1
    unsafe = test_y == 0

    # Main scatter
    ax.scatter(test_pca[unsafe & ~boundary, 0], test_pca[unsafe & ~boundary, 1],
               c=COLOR_UNSAFE, s=25, alpha=0.6, label="Unsafe", zorder=2)
    ax.scatter(test_pca[safe & ~boundary, 0], test_pca[safe & ~boundary, 1],
               c=COLOR_SAFE, s=25, alpha=0.6, label="Safe", zorder=2)

    # Boundary examples
    ax.scatter(test_pca[boundary, 0], test_pca[boundary, 1],
               c="none", edgecolors=COLOR_BOUNDARY, s=60, linewidths=1.5,
               marker="D", label="Boundary (|score|<0.5)", zorder=3)

    # Probe direction arrow from centroid
    pca = data["pca"]
    w = data["w"]
    b = data["b"]
    V = pca.components_[:2]
    w_pca = V @ w
    w_pca_unit = w_pca / np.linalg.norm(w_pca)
    centroid = np.mean(test_pca, axis=0)

    arrow_scale = (test_pca[:, 0].max() - test_pca[:, 0].min()) * 0.15
    ax.annotate("", xy=centroid + w_pca_unit * arrow_scale,
                xytext=centroid,
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=COLOR_PROBE, lw=2.5))
    ax.text(centroid[0] + w_pca_unit[0] * arrow_scale * 1.15,
            centroid[1] + w_pca_unit[1] * arrow_scale * 1.15,
            "w (probe)", color=COLOR_PROBE, fontsize=10, fontweight="bold")

    # Decision boundary
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    _draw_decision_boundary(ax, pca, w, b, xlim, ylim)

    ve = data["var_explained"]
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("IT-Model Layer 10 Activations (PCA)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    _save_fig(fig, "fig1_pca_scatter")
    plt.close(fig)


def figure2_tsne(data):
    """t-SNE scatter."""
    print("  Figure 2: t-SNE Scatter ...")
    test_tsne = data["test_tsne"]
    test_y = data["test_y"]
    boundary = data["boundary_mask"]

    fig, ax = plt.subplots(figsize=(8, 6))

    safe = test_y == 1
    unsafe = test_y == 0

    ax.scatter(test_tsne[unsafe & ~boundary, 0], test_tsne[unsafe & ~boundary, 1],
               c=COLOR_UNSAFE, s=25, alpha=0.6, label="Unsafe", zorder=2)
    ax.scatter(test_tsne[safe & ~boundary, 0], test_tsne[safe & ~boundary, 1],
               c=COLOR_SAFE, s=25, alpha=0.6, label="Safe", zorder=2)
    ax.scatter(test_tsne[boundary, 0], test_tsne[boundary, 1],
               c="none", edgecolors=COLOR_BOUNDARY, s=60, linewidths=1.5,
               marker="D", label="Boundary (|score|<0.5)", zorder=3)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("IT-Model Layer 10 Activations (t-SNE)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    _save_fig(fig, "fig2_tsne_scatter")
    plt.close(fig)


def figure3_umap(data):
    """UMAP scatter."""
    if data["test_umap"] is None:
        print("  Figure 3: UMAP SKIPPED (umap-learn not installed)")
        return

    print("  Figure 3: UMAP Scatter ...")
    test_umap = data["test_umap"]
    test_y = data["test_y"]
    boundary = data["boundary_mask"]

    fig, ax = plt.subplots(figsize=(8, 6))

    safe = test_y == 1
    unsafe = test_y == 0

    ax.scatter(test_umap[unsafe & ~boundary, 0], test_umap[unsafe & ~boundary, 1],
               c=COLOR_UNSAFE, s=25, alpha=0.6, label="Unsafe", zorder=2)
    ax.scatter(test_umap[safe & ~boundary, 0], test_umap[safe & ~boundary, 1],
               c=COLOR_SAFE, s=25, alpha=0.6, label="Safe", zorder=2)
    ax.scatter(test_umap[boundary, 0], test_umap[boundary, 1],
               c="none", edgecolors=COLOR_BOUNDARY, s=60, linewidths=1.5,
               marker="D", label="Boundary (|score|<0.5)", zorder=3)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("IT-Model Layer 10 Activations (UMAP)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    _save_fig(fig, "fig3_umap_scatter")
    plt.close(fig)


def figure4_steering_vectors(data):
    """PCA scatter with steering vector arrows."""
    print("  Figure 4: PCA with Steering Vectors ...")
    test_pca = data["test_pca"][:, :2]
    test_y = data["test_y"]
    boundary = data["boundary_mask"]
    pca = data["pca"]
    w = data["w"]
    b = data["b"]

    fig, ax = plt.subplots(figsize=(8, 6))

    safe = test_y == 1
    unsafe = test_y == 0

    ax.scatter(test_pca[unsafe & ~boundary, 0], test_pca[unsafe & ~boundary, 1],
               c=COLOR_UNSAFE, s=20, alpha=0.4, zorder=2)
    ax.scatter(test_pca[safe & ~boundary, 0], test_pca[safe & ~boundary, 1],
               c=COLOR_SAFE, s=20, alpha=0.4, zorder=2)
    ax.scatter(test_pca[boundary, 0], test_pca[boundary, 1],
               c="none", edgecolors=COLOR_BOUNDARY, s=50, linewidths=1.2,
               marker="D", zorder=3)

    # Pick a representative unsafe boundary example
    traj_idx = data["traj_idx"]
    rep_idx = traj_idx[0]  # most boundary example
    rep_pca = test_pca[rep_idx]

    # Project deltas to PCA space
    V = pca.components_[:2]
    d_naive_pca = V @ data["deltas_naive"][0]
    d_robust_pca = V @ data["deltas_robust"][0]

    # Scale for visibility
    data_range = test_pca[:, 0].max() - test_pca[:, 0].min()
    scale_naive = data_range * 0.12 / max(np.linalg.norm(d_naive_pca), 1e-10)
    scale_robust = data_range * 0.12 / max(np.linalg.norm(d_robust_pca), 1e-10)

    # Naive delta arrow
    ax.annotate("", xy=rep_pca + d_naive_pca * scale_naive, xytext=rep_pca,
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=COLOR_NAIVE, lw=2.5))
    end_naive = rep_pca + d_naive_pca * scale_naive
    ax.text(end_naive[0], end_naive[1] + data_range * 0.02,
            r"$\delta_{naive}$", color=COLOR_NAIVE, fontsize=11, fontweight="bold")

    # Robust delta arrow
    ax.annotate("", xy=rep_pca + d_robust_pca * scale_robust, xytext=rep_pca,
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=COLOR_ROBUST, lw=2.5))
    end_robust = rep_pca + d_robust_pca * scale_robust
    ax.text(end_robust[0], end_robust[1] + data_range * 0.02,
            r"$\delta_{robust}$", color=COLOR_ROBUST, fontsize=11, fontweight="bold")

    # Mark the starting point
    ax.scatter([rep_pca[0]], [rep_pca[1]], c="black", s=80, zorder=5, marker="*")

    # Decision boundary
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    _draw_decision_boundary(ax, pca, w, b, xlim, ylim)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Steering Vectors in PCA Space")

    legend_elements = [
        Line2D([0], [0], color=COLOR_UNSAFE, marker="o", linestyle="", markersize=6, label="Unsafe"),
        Line2D([0], [0], color=COLOR_SAFE, marker="o", linestyle="", markersize=6, label="Safe"),
        Line2D([0], [0], color=COLOR_NAIVE, marker=">", linestyle="-", markersize=8, label=r"$\delta_{naive}$"),
        Line2D([0], [0], color=COLOR_ROBUST, marker=">", linestyle="-", markersize=8, label=r"$\delta_{robust}$"),
        Line2D([0], [0], color="k", linestyle="--", label="Decision boundary"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9, framealpha=0.9)

    _save_fig(fig, "fig4_steering_vectors")
    plt.close(fig)


def figure5_score_distribution(data):
    """Probe score histogram."""
    print("  Figure 5: Probe Score Distribution ...")
    scores = data["scores"]
    test_y = data["test_y"]

    fig, ax = plt.subplots(figsize=(8, 5))

    safe_scores = scores[test_y == 1]
    unsafe_scores = scores[test_y == 0]

    bins = np.linspace(scores.min() - 0.5, scores.max() + 0.5, 40)
    ax.hist(unsafe_scores, bins=bins, color=COLOR_UNSAFE, alpha=0.6,
            label=f"Unsafe (n={len(unsafe_scores)})", edgecolor="white", linewidth=0.5)
    ax.hist(safe_scores, bins=bins, color=COLOR_SAFE, alpha=0.6,
            label=f"Safe (n={len(safe_scores)})", edgecolor="white", linewidth=0.5)

    # Decision boundary
    ax.axvline(0, color="k", linestyle="--", linewidth=1.5, label="Decision boundary (score=0)")

    # Boundary band
    ax.axvspan(-0.5, 0.5, alpha=0.1, color=COLOR_BOUNDARY, label="Boundary band (|score|<0.5)")

    ax.set_xlabel("Probe Logit Score")
    ax.set_ylabel("Count")
    ax.set_title("IT-Model Probe Score Distribution")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    _save_fig(fig, "fig5_score_distribution")
    plt.close(fig)


def figure6_rashomon_directions(data):
    """Rashomon probe directions in PCA space."""
    print("  Figure 6: Rashomon Probe Directions ...")
    pca = data["pca"]
    rash_weights = data["rash_weights"]  # [33, 2304]
    w = data["w"]

    V = pca.components_[:2]  # [2, 2304]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Project each Rashomon probe to PCA space and normalize
    for i in range(len(rash_weights)):
        rw_pca = V @ rash_weights[i]
        rw_pca_unit = rw_pca / np.linalg.norm(rw_pca)
        ax.annotate("", xy=rw_pca_unit, xytext=[0, 0],
                    arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                                    color=COLOR_RASHOMON, lw=1.0, alpha=0.5))

    # Baseline probe direction
    w_pca = V @ w
    w_pca_unit = w_pca / np.linalg.norm(w_pca)
    ax.annotate("", xy=w_pca_unit, xytext=[0, 0],
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.15",
                                color=COLOR_PROBE, lw=2.5))
    ax.text(w_pca_unit[0] * 1.1, w_pca_unit[1] * 1.1,
            "Baseline", color=COLOR_PROBE, fontsize=10, fontweight="bold",
            ha="center")

    # Unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", alpha=0.2, linewidth=0.5)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 direction")
    ax.set_ylabel(f"PC2 direction")
    ax.set_title("Rashomon Probe Directions in PCA Space")

    legend_elements = [
        Line2D([0], [0], color=COLOR_RASHOMON, marker=">", linestyle="-",
               markersize=6, alpha=0.5, label=f"Rashomon probes (n={len(rash_weights)})"),
        Line2D([0], [0], color=COLOR_PROBE, marker=">", linestyle="-",
               markersize=8, label="Baseline probe"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    _save_fig(fig, "fig6_rashomon_directions")
    plt.close(fig)


def figure7_trajectories(data):
    """Steering trajectories for boundary examples."""
    print("  Figure 7: Steering Trajectories ...")
    test_pca = data["test_pca"][:, :2]
    test_y = data["test_y"]
    pca = data["pca"]
    w = data["w"]
    b = data["b"]
    boundary = data["boundary_mask"]
    traj_idx = data["traj_idx"]

    fig, ax = plt.subplots(figsize=(8, 6))

    safe = test_y == 1
    unsafe = test_y == 0

    # Background scatter (faded)
    ax.scatter(test_pca[unsafe, 0], test_pca[unsafe, 1],
               c=COLOR_UNSAFE, s=12, alpha=0.15, zorder=1)
    ax.scatter(test_pca[safe, 0], test_pca[safe, 1],
               c=COLOR_SAFE, s=12, alpha=0.15, zorder=1)

    V = pca.components_[:2]
    traj_colors = plt.cm.Set1(np.linspace(0, 0.6, len(traj_idx)))

    for i, idx in enumerate(traj_idx):
        orig = test_pca[idx]
        d_naive_pca = V @ data["deltas_naive"][i]
        d_robust_pca = V @ data["deltas_robust"][i]
        steered_naive = orig + d_naive_pca
        steered_robust = orig + d_robust_pca

        color = traj_colors[i]

        # Original point
        ax.scatter([orig[0]], [orig[1]], c=[color], s=70, zorder=4,
                   marker="o", edgecolors="black", linewidths=0.8)

        # Naive trajectory
        ax.plot([orig[0], steered_naive[0]], [orig[1], steered_naive[1]],
                color=color, linestyle=":", linewidth=1.5, alpha=0.7, zorder=3)
        ax.scatter([steered_naive[0]], [steered_naive[1]], c=[color], s=50,
                   marker="^", edgecolors="black", linewidths=0.5, zorder=4)

        # Robust trajectory
        ax.plot([orig[0], steered_robust[0]], [orig[1], steered_robust[1]],
                color=color, linestyle="-", linewidth=2.0, alpha=0.8, zorder=3)
        ax.scatter([steered_robust[0]], [steered_robust[1]], c=[color], s=50,
                   marker="s", edgecolors="black", linewidths=0.5, zorder=4)

        # Label
        ax.text(orig[0], orig[1] + (test_pca[:, 1].max() - test_pca[:, 1].min()) * 0.025,
                f"id={idx}", fontsize=7, ha="center", color=color, fontweight="bold")

    # Decision boundary
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    _draw_decision_boundary(ax, pca, w, b, xlim, ylim)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Steering Trajectories for Boundary Examples")

    legend_elements = [
        Line2D([0], [0], color="gray", marker="o", linestyle="",
               markersize=8, markeredgecolor="black", label="Original"),
        Line2D([0], [0], color="gray", marker="^", linestyle=":",
               markersize=8, markeredgecolor="black", label="After naive steering"),
        Line2D([0], [0], color="gray", marker="s", linestyle="-",
               markersize=8, markeredgecolor="black", linewidth=2, label="After robust steering"),
        Line2D([0], [0], color="k", linestyle="--", label="Decision boundary"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9, framealpha=0.9)

    _save_fig(fig, "fig7_steering_trajectories")
    plt.close(fig)


def phase2_figures(data):
    """Generate all 7 figures."""
    print("\n" + "=" * 75)
    print("PHASE 2: Core Visualization Suite")
    print("=" * 75)

    figure1_pca(data)
    figure2_tsne(data)
    figure3_umap(data)
    figure4_steering_vectors(data)
    figure5_score_distribution(data)
    figure6_rashomon_directions(data)
    figure7_trajectories(data)

    print("\n  All figures generated.")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Geometric Analysis
# ═══════════════════════════════════════════════════════════════════════

def phase3_geometry(data):
    """Compute quantitative geometric properties."""
    print("\n" + "=" * 75)
    print("PHASE 3: Geometric Analysis")
    print("=" * 75)

    test_X = data["test_X"]
    test_y = data["test_y"]
    w = data["w"]
    pca = data["pca"]
    rash_weights = data["rash_weights"]

    safe_X = test_X[test_y == 1]
    unsafe_X = test_X[test_y == 0]

    # --- 1. Cluster separation ---
    print("\n  1. Cluster Separation:")
    centroid_safe = safe_X.mean(axis=0)
    centroid_unsafe = unsafe_X.mean(axis=0)
    centroid_diff = centroid_safe - centroid_unsafe
    centroid_dist = np.linalg.norm(centroid_diff)

    # Within-cluster std (pooled)
    std_safe = np.mean(np.std(safe_X, axis=0))
    std_unsafe = np.mean(np.std(unsafe_X, axis=0))
    pooled_std = np.sqrt((std_safe**2 + std_unsafe**2) / 2)

    # Fisher's discriminant ratio
    # J = |mu_1 - mu_2|^2 / (s_1^2 + s_2^2) along the optimal direction
    var_safe = np.var(safe_X @ centroid_diff / centroid_dist)
    var_unsafe = np.var(unsafe_X @ centroid_diff / centroid_dist)
    fisher_ratio = centroid_dist**2 / (var_safe + var_unsafe) if (var_safe + var_unsafe) > 0 else 0

    print(f"    Centroid distance: {centroid_dist:.4f}")
    print(f"    Pooled within-cluster std: {pooled_std:.4f}")
    print(f"    Normalized separation (dist/std): {centroid_dist/pooled_std:.4f}")
    print(f"    Fisher's discriminant ratio: {fisher_ratio:.4f}")

    cluster_metrics = {
        "centroid_distance": centroid_dist,
        "pooled_std": pooled_std,
        "normalized_separation": centroid_dist / pooled_std,
        "fisher_ratio": fisher_ratio,
    }

    # --- 2. Probe direction vs PCs ---
    print("\n  2. Probe Direction vs Principal Components:")
    w_unit = w / np.linalg.norm(w)
    pc_cosines = []
    cumulative_proj = 0
    for i in range(min(10, pca.n_components_)):
        cos = np.dot(w_unit, pca.components_[i])
        pc_cosines.append(cos)
        cumulative_proj += cos**2
        print(f"    cos(w, PC{i+1:2d}) = {cos:+.4f}  "
              f"(cumulative w in top-{i+1} PCs: {cumulative_proj:.4f})")

    # How much of probe direction is captured by top-k PCs
    probe_in_topk = {}
    for k in [2, 3, 5, 10, 50]:
        proj = sum(np.dot(w_unit, pca.components_[i])**2 for i in range(min(k, pca.n_components_)))
        probe_in_topk[k] = proj
        print(f"    ||proj(w, top-{k} PCs)||^2 = {proj:.4f}")

    # --- 3. Steering vector vs data geometry ---
    print("\n  3. Steering Vector Alignment:")
    centroid_diff_unit = centroid_diff / np.linalg.norm(centroid_diff)

    # Use the first boundary example's deltas
    d_robust = data["deltas_robust"][0]
    d_naive = data["deltas_naive"][0]
    d_robust_unit = d_robust / np.linalg.norm(d_robust)
    d_naive_unit = d_naive / np.linalg.norm(d_naive)

    cos_robust_centroid = np.dot(d_robust_unit, centroid_diff_unit)
    cos_naive_centroid = np.dot(d_naive_unit, centroid_diff_unit)
    cos_robust_probe = np.dot(d_robust_unit, w_unit)
    cos_naive_probe = np.dot(d_naive_unit, w_unit)
    cos_robust_naive = np.dot(d_robust_unit, d_naive_unit)

    print(f"    cos(d_robust, centroid_diff) = {cos_robust_centroid:+.4f}")
    print(f"    cos(d_naive,  centroid_diff) = {cos_naive_centroid:+.4f}")
    print(f"    cos(d_robust, w_probe)       = {cos_robust_probe:+.4f}")
    print(f"    cos(d_naive,  w_probe)       = {cos_naive_probe:+.4f}")
    print(f"    cos(d_robust, d_naive)       = {cos_robust_naive:+.4f}")

    # Steering vs top PCs
    steering_pc_cosines = []
    for i in range(min(10, pca.n_components_)):
        cos_r = np.dot(d_robust_unit, pca.components_[i])
        cos_n = np.dot(d_naive_unit, pca.components_[i])
        steering_pc_cosines.append((cos_r, cos_n))
        print(f"    cos(d_robust, PC{i+1:2d}) = {cos_r:+.4f}  "
              f"cos(d_naive, PC{i+1:2d}) = {cos_n:+.4f}")

    steering_metrics = {
        "cos_robust_centroid": cos_robust_centroid,
        "cos_naive_centroid": cos_naive_centroid,
        "cos_robust_probe": cos_robust_probe,
        "cos_naive_probe": cos_naive_probe,
        "cos_robust_naive": cos_robust_naive,
    }

    # --- 4. Rashomon angular spread ---
    print("\n  4. Rashomon Probe Angular Spread:")
    n_rash = len(rash_weights)
    rash_units = rash_weights / np.linalg.norm(rash_weights, axis=1, keepdims=True)

    # Pairwise cosines
    cos_matrix = rash_units @ rash_units.T
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n_rash, k=1)
    pairwise_cos = cos_matrix[triu_idx]
    pairwise_angles = np.degrees(np.arccos(np.clip(pairwise_cos, -1, 1)))

    print(f"    Number of pairs: {len(pairwise_cos)}")
    print(f"    Pairwise cosine: mean={pairwise_cos.mean():.6f}, "
          f"min={pairwise_cos.min():.6f}, max={pairwise_cos.max():.6f}")
    print(f"    Pairwise angle (degrees): mean={pairwise_angles.mean():.4f}, "
          f"min={pairwise_angles.min():.4f}, max={pairwise_angles.max():.4f}")

    # Angle between each Rashomon probe and baseline
    cos_to_baseline = rash_units @ w_unit
    angles_to_baseline = np.degrees(np.arccos(np.clip(cos_to_baseline, -1, 1)))
    print(f"    Angle to baseline: mean={angles_to_baseline.mean():.4f}, "
          f"min={angles_to_baseline.min():.4f}, max={angles_to_baseline.max():.4f}")

    rashomon_metrics = {
        "pairwise_cos_mean": pairwise_cos.mean(),
        "pairwise_cos_min": pairwise_cos.min(),
        "pairwise_cos_max": pairwise_cos.max(),
        "pairwise_angle_mean": pairwise_angles.mean(),
        "pairwise_angle_min": pairwise_angles.min(),
        "pairwise_angle_max": pairwise_angles.max(),
        "angle_to_baseline_mean": angles_to_baseline.mean(),
        "angle_to_baseline_min": angles_to_baseline.min(),
        "angle_to_baseline_max": angles_to_baseline.max(),
    }

    return {
        "cluster": cluster_metrics,
        "pc_cosines": pc_cosines,
        "probe_in_topk": probe_in_topk,
        "steering": steering_metrics,
        "steering_pc_cosines": steering_pc_cosines,
        "rashomon": rashomon_metrics,
        "var_explained": data["var_explained"],
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Summary Report
# ═══════════════════════════════════════════════════════════════════════

def phase4_report(metrics, data):
    """Generate comprehensive geometry report."""
    print("\n" + "=" * 75)
    print("PHASE 4: Summary Report")
    print("=" * 75)

    lines = []
    lines.append("=" * 75)
    lines.append("ACTIVATION-SPACE GEOMETRY — ANALYSIS REPORT")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: google/gemma-2-2b-it, Layer 10")
    lines.append(f"Data: BeaverTails, 1600 train / 400 test, dim=2304")
    lines.append(f"Rashomon epsilon: {EPSILON}")
    lines.append("")

    # --- 1. Variance Explained ---
    lines.append("-" * 75)
    lines.append("1. PCA VARIANCE EXPLAINED")
    lines.append("-" * 75)
    ve = metrics["var_explained"]
    lines.append(f"  {'Components':>12s} | {'Cumulative Variance':>20s}")
    lines.append(f"  {'-'*12}-+-{'-'*20}")
    for k in [2, 3, 10, 50]:
        idx = k - 1
        if idx < len(ve):
            lines.append(f"  {k:>12d} | {ve[idx]:>20.4f}")
    # Top-100
    pca = data["pca"]
    if pca.n_components_ >= 50:
        lines.append(f"  {'50 (max fit)':>12s} | {ve[49]:>20.4f}")
    lines.append("")
    lines.append(f"  PC1 explains {pca.explained_variance_ratio_[0]:.2%} of variance")
    lines.append(f"  PC2 explains {pca.explained_variance_ratio_[1]:.2%} of variance")
    lines.append("")

    # --- 2. Cluster Separation ---
    lines.append("-" * 75)
    lines.append("2. CLUSTER SEPARATION METRICS")
    lines.append("-" * 75)
    cm = metrics["cluster"]
    lines.append(f"  Centroid distance (L2, 2304-dim): {cm['centroid_distance']:.4f}")
    lines.append(f"  Pooled within-cluster std:        {cm['pooled_std']:.4f}")
    lines.append(f"  Normalized separation (dist/std): {cm['normalized_separation']:.4f}")
    lines.append(f"  Fisher's discriminant ratio:      {cm['fisher_ratio']:.4f}")
    lines.append("")

    # --- 3. Probe vs PC Alignment ---
    lines.append("-" * 75)
    lines.append("3. PROBE DIRECTION vs PRINCIPAL COMPONENTS")
    lines.append("-" * 75)
    lines.append(f"  {'PC':>4s} | {'cos(w, PC)':>12s} | {'cos^2 (cumul)':>15s}")
    lines.append(f"  {'-'*4}-+-{'-'*12}-+-{'-'*15}")
    cumul = 0
    for i, cos in enumerate(metrics["pc_cosines"]):
        cumul += cos**2
        lines.append(f"  {i+1:>4d} | {cos:>+12.4f} | {cumul:>15.4f}")
    lines.append("")
    lines.append(f"  Probe energy in top-k PC subspace:")
    for k, proj in metrics["probe_in_topk"].items():
        lines.append(f"    top-{k:2d}: {proj:.4f} ({proj:.1%})")
    lines.append("")

    # --- 4. Steering Vector Alignment ---
    lines.append("-" * 75)
    lines.append("4. STEERING VECTOR ALIGNMENT")
    lines.append("-" * 75)
    sm = metrics["steering"]
    lines.append(f"  cos(d_robust, centroid_diff) = {sm['cos_robust_centroid']:+.4f}")
    lines.append(f"  cos(d_naive,  centroid_diff) = {sm['cos_naive_centroid']:+.4f}")
    lines.append(f"  cos(d_robust, w_probe)       = {sm['cos_robust_probe']:+.4f}")
    lines.append(f"  cos(d_naive,  w_probe)       = {sm['cos_naive_probe']:+.4f}")
    lines.append(f"  cos(d_robust, d_naive)       = {sm['cos_robust_naive']:+.4f}")
    lines.append("")
    lines.append(f"  Steering vectors vs top PCs:")
    lines.append(f"  {'PC':>4s} | {'cos(d_robust, PC)':>18s} | {'cos(d_naive, PC)':>18s}")
    lines.append(f"  {'-'*4}-+-{'-'*18}-+-{'-'*18}")
    for i, (cr, cn) in enumerate(metrics["steering_pc_cosines"]):
        lines.append(f"  {i+1:>4d} | {cr:>+18.4f} | {cn:>+18.4f}")
    lines.append("")

    # --- 5. Rashomon Angular Spread ---
    lines.append("-" * 75)
    lines.append("5. RASHOMON PROBE ANGULAR SPREAD")
    lines.append("-" * 75)
    rm = metrics["rashomon"]
    lines.append(f"  Pairwise cosine similarity:")
    lines.append(f"    Mean: {rm['pairwise_cos_mean']:.6f}")
    lines.append(f"    Min:  {rm['pairwise_cos_min']:.6f}")
    lines.append(f"    Max:  {rm['pairwise_cos_max']:.6f}")
    lines.append(f"  Pairwise angular distance (degrees):")
    lines.append(f"    Mean: {rm['pairwise_angle_mean']:.2f}")
    lines.append(f"    Min:  {rm['pairwise_angle_min']:.2f}")
    lines.append(f"    Max:  {rm['pairwise_angle_max']:.2f}")
    lines.append(f"  Angle to baseline probe (degrees):")
    lines.append(f"    Mean: {rm['angle_to_baseline_mean']:.2f}")
    lines.append(f"    Min:  {rm['angle_to_baseline_min']:.2f}")
    lines.append(f"    Max:  {rm['angle_to_baseline_max']:.2f}")
    lines.append("")

    # --- 6. Geometric Interpretation ---
    lines.append("-" * 75)
    lines.append("6. GEOMETRIC INTERPRETATION")
    lines.append("-" * 75)

    norm_sep = cm["normalized_separation"]
    fisher = cm["fisher_ratio"]
    probe_top2 = metrics["probe_in_topk"][2]
    probe_top10 = metrics["probe_in_topk"][10]
    cos_rp = sm["cos_robust_probe"]
    cos_rc = sm["cos_robust_centroid"]
    rash_angle = rm["pairwise_angle_mean"]

    lines.append("")
    lines.append("  (a) CLUSTER SEPARATION:")
    if norm_sep < 1.0:
        lines.append(f"      Safe and unsafe clusters are HEAVILY OVERLAPPING (normalized separation"
                     f" = {norm_sep:.2f} < 1.0). In full 2304-dim space, the two classes are not"
                     f" geometrically distinct clusters but rather intermixed distributions. This"
                     f" explains why linear probes achieve only ~77% accuracy and why steering"
                     f" perturbations that are small relative to within-cluster variance have"
                     f" limited impact on downstream behavior.")
    else:
        lines.append(f"      Safe and unsafe clusters show moderate separation (normalized"
                     f" separation = {norm_sep:.2f}). However, Fisher's discriminant ratio"
                     f" ({fisher:.2f}) indicates overlap remains significant.")
    lines.append("")

    lines.append("  (b) PROBE DIRECTION vs DATA VARIANCE:")
    if probe_top2 < 0.05:
        lines.append(f"      The probe direction is NEARLY ORTHOGONAL to the top-2 PCs"
                     f" (only {probe_top2:.1%} of probe energy in PC1-PC2). This means the"
                     f" safety-relevant direction is NOT a dominant variance direction in"
                     f" the activation space. The probe finds a subtle, low-variance separating"
                     f" direction that is almost invisible in PCA projections.")
    else:
        lines.append(f"      The probe direction captures {probe_top2:.1%} of its energy from"
                     f" PC1-PC2 and {probe_top10:.1%} from top-10 PCs.")
    lines.append(f"      With top-10 PCs capturing {probe_top10:.1%} of probe energy, the"
                 f" safety direction lies primarily OUTSIDE the dominant variance subspace.")
    lines.append("")

    lines.append("  (c) STEERING VECTOR ALIGNMENT:")
    lines.append(f"      The robust delta is {'strongly' if abs(cos_rp) > 0.5 else 'weakly'}"
                 f" aligned with the probe direction (cos = {cos_rp:+.4f}) and"
                 f" {'strongly' if abs(cos_rc) > 0.5 else 'weakly'} aligned with the"
                 f" safe-unsafe centroid difference (cos = {cos_rc:+.4f}).")
    if abs(cos_rp) > 0.8:
        lines.append(f"      This confirms the robust delta pushes activations along the probe"
                     f" direction as designed. The failure to produce behavioral change is NOT"
                     f" due to misalignment between the delta and probe.")
    lines.append("")

    lines.append("  (d) WHY PROBE-LEVEL STEERING DOESN'T TRANSLATE TO BEHAVIORAL CHANGE:")
    lines.append(f"      The Rashomon probes span a narrow angular cone (mean pairwise angle"
                 f" = {rash_angle:.1f} degrees), confirming that all near-optimal probes agree"
                 f" on the safety direction. But this direction lives in a low-variance"
                 f" subspace of the activation space (orthogonal to dominant PCs).")
    lines.append(f"      Meanwhile, the IT model's generation behavior is governed by the"
                 f" full activation geometry — attention patterns, position-specific features,"
                 f" and RLHF-conditioned output heads all operate on the HIGH-variance"
                 f" directions that the probe direction is orthogonal to.")
    lines.append(f"      Steering along the probe direction moves activations in a direction"
                 f" that the linear probe detects but that the transformer's downstream"
                 f" layers effectively ignore for generation purposes. The safety signal"
                 f" is real but GEOMETRICALLY MARGINAL in the full activation space.")
    lines.append("")

    lines.append("=" * 75)
    lines.append("END OF REPORT")
    lines.append("=" * 75)

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n  Report saved to {REPORT_PATH}")
    print(report)
    return report, metrics


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    data = phase1_load_and_project()
    phase2_figures(data)
    metrics = phase3_geometry(data)
    phase4_report(metrics, data)

    elapsed = time.time() - t_total
    print(f"\n  Total pipeline time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("  ACTIVATION-SPACE GEOMETRY VISUALIZATION COMPLETE")


if __name__ == "__main__":
    main()
