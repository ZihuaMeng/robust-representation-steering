"""Generate publication-quality figures from experiment outputs.

Usage:
    python scripts/generate_figures.py

Reads CSV/txt data from outputs/ and saves 6 PNG figures to assets/figures/.
Figures 5 and 6 require .pt files that are not committed to git — they will
be skipped gracefully if those files are missing.
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(ROOT, "assets", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {
    "raw": "#2563EB",      # blue
    "sae": "#DC2626",      # red
    "naive": "#F59E0B",    # amber
    "robust": "#059669",   # emerald
    "accent": "#7C3AED",   # violet
}
DPI = 300


def load_hamming(space):
    path = os.path.join(OUTPUT_DIR, space, "hamming_matrix.csv")
    return pd.read_csv(path, index_col=0)


def load_metrics(space):
    path = os.path.join(OUTPUT_DIR, space, "per_probe_metrics.csv")
    return pd.read_csv(path)


def upper_triangle(df):
    """Extract upper-triangle values (excluding diagonal) from a square matrix."""
    mat = df.values
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Hamming heatmap — Raw Space
# ═════════════════════════════════════════════════════════════════════════════
def fig1_hamming_heatmap_raw():
    df = load_hamming("rashomon")
    mat = df.values

    global VMIN, VMAX
    sae_mat = load_hamming("rashomon_sae").values
    VMIN = 0.0
    VMAX = max(mat.max(), sae_mat.max())

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=VMIN, vmax=VMAX, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hamming Distance", fontsize=12)
    ax.set_title("Pairwise Prediction Disagreement\nRaw Residual Space", fontsize=14, fontweight="bold")
    ax.set_xlabel("Probe Index", fontsize=12)
    ax.set_ylabel("Probe Index", fontsize=12)
    tick_pos = list(range(0, len(df), 10))
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_hamming_heatmap_raw.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Hamming heatmap — SAE Space
# ═════════════════════════════════════════════════════════════════════════════
def fig2_hamming_heatmap_sae():
    df = load_hamming("rashomon_sae")
    mat = df.values

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=VMIN, vmax=VMAX, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hamming Distance", fontsize=12)
    ax.set_title("Pairwise Prediction Disagreement\nSAE Feature Space", fontsize=14, fontweight="bold")
    ax.set_xlabel("Probe Index", fontsize=12)
    ax.set_ylabel("Probe Index", fontsize=12)
    tick_pos = list(range(0, len(df), 10))
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_hamming_heatmap_sae.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Distribution comparison
# ═════════════════════════════════════════════════════════════════════════════
def fig3_hamming_distribution():
    raw_vals = upper_triangle(load_hamming("rashomon"))
    sae_vals = upper_triangle(load_hamming("rashomon_sae"))

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 50)

    ax.hist(raw_vals, bins=bins, alpha=0.6, color=PALETTE["raw"],
            label=f"Raw Space (mean={raw_vals.mean():.3f})", density=True, edgecolor="white", linewidth=0.5)
    ax.hist(sae_vals, bins=bins, alpha=0.6, color=PALETTE["sae"],
            label=f"SAE Space (mean={sae_vals.mean():.3f})", density=True, edgecolor="white", linewidth=0.5)

    ax.axvline(raw_vals.mean(), color=PALETTE["raw"], linestyle="--", linewidth=2, alpha=0.9)
    ax.axvline(sae_vals.mean(), color=PALETTE["sae"], linestyle="--", linewidth=2, alpha=0.9)

    ax.set_xlabel("Pairwise Hamming Distance", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Probe Disagreement: Raw vs SAE Space", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_hamming_distribution_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: Probe performance scatter
# ═════════════════════════════════════════════════════════════════════════════
def fig4_probe_performance():
    raw_df = load_metrics("rashomon")
    sae_df = load_metrics("rashomon_sae")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Axis limits — shared
    all_acc = pd.concat([raw_df["accuracy"], sae_df["accuracy"]])
    all_f1 = pd.concat([raw_df["f1"], sae_df["f1"]])
    x_lo, x_hi = all_acc.min() - 0.02, all_acc.max() + 0.02
    y_lo, y_hi = all_f1.min() - 0.02, all_f1.max() + 0.02

    for ax, df, title, color in [
        (ax1, raw_df, "Raw Space", PALETTE["raw"]),
        (ax2, sae_df, "SAE Space", PALETTE["sae"]),
    ]:
        awp = df[df["probe"] != "baseline"]
        base = df[df["probe"] == "baseline"]

        ax.scatter(awp["accuracy"], awp["f1"], c=color, alpha=0.7, s=50,
                   edgecolors="white", linewidth=0.5, label="AWP Probes")
        ax.scatter(base["accuracy"], base["f1"], c=color, marker="*", s=250,
                   edgecolors="black", linewidth=1.0, zorder=5, label="Baseline")

        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.legend(fontsize=10, loc="lower right")

    ax1.set_ylabel("F1 Score", fontsize=12)
    fig.suptitle("Probe Performance Diversity: Raw vs SAE Space", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_probe_performance_scatter.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 & 6: Steering coverage (require .pt files or parsed report data)
# ═════════════════════════════════════════════════════════════════════════════
def parse_steering_report():
    """Parse per-example data from the steering comparison report."""
    report_path = os.path.join(OUTPUT_DIR, "steering_comparison_report.txt")
    if not os.path.exists(report_path):
        return None

    with open(report_path) as f:
        text = f.read()

    examples = []
    # Match lines like: "  2   -0.5914       0.5911        5.3389    20/50      50/50"
    pattern = re.compile(
        r"^\s*(\d+)\s+"            # example index
        r"(-?[\d.]+)\s+"           # score
        r"([\d.]+)\s+"             # ||d_naive||
        r"([\d.]+)\s+"             # ||d_robust||
        r"(\d+)/(\d+)\s+"         # naive coverage
        r"(\d+)/(\d+)",           # robust coverage
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        examples.append({
            "idx": int(m.group(1)),
            "score": float(m.group(2)),
            "norm_naive": float(m.group(3)),
            "norm_robust": float(m.group(4)),
            "cov_naive": int(m.group(5)),
            "n_probes": int(m.group(6)),
            "cov_robust": int(m.group(7)),
        })
    return examples if examples else None


def try_load_full_steering_data():
    """Try to load full per-example steering data from .pt files."""
    try:
        import torch
        act_path = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10.pt")
        baseline_path = os.path.join(OUTPUT_DIR, "baseline_probe_layer10.pt")
        rashomon_path = os.path.join(OUTPUT_DIR, "rashomon", "rashomon_probes.pt")
        hessian_path = os.path.join(OUTPUT_DIR, "hessian_layer10.pt")

        for p in [act_path, baseline_path, rashomon_path, hessian_path]:
            if not os.path.exists(p):
                return None

        sys.path.insert(0, os.path.join(ROOT, "src"))
        from steering import naive_delta, robust_delta, rashomon_coverage

        data = torch.load(act_path, map_location="cpu", weights_only=True)
        test_X = data["test_X"].double()
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        w = baseline["weight"].double()
        b = baseline["bias"].double()
        probes = torch.load(rashomon_path, map_location="cpu", weights_only=False)
        hess = torch.load(hessian_path, map_location="cpu", weights_only=True)
        H_inv = hess["H_inv"]
        epsilon = 0.15

        logits = test_X @ w + b
        unsafe_mask = logits < 0.0
        unsafe_indices = unsafe_mask.nonzero(as_tuple=True)[0]

        results = []
        for i in range(len(unsafe_indices)):
            idx = unsafe_indices[i].item()
            x = test_X[idx]
            score = logits[idx].item()
            d_naive, _ = naive_delta(w, b, x, 0.0)
            d_robust = robust_delta(w, b, x, H_inv, epsilon, 0.0)
            n_cov_naive, n_total, _ = rashomon_coverage(d_naive, x, probes, 0.0)
            n_cov_robust, _, _ = rashomon_coverage(d_robust, x, probes, 0.0)
            results.append({
                "idx": idx,
                "score": score,
                "norm_naive": d_naive.norm().item(),
                "norm_robust": d_robust.norm().item(),
                "cov_naive": n_cov_naive,
                "cov_robust": n_cov_robust,
                "n_probes": n_total,
            })
        return results
    except Exception as e:
        print(f"  Warning: Could not load .pt files for Figs 5-6: {e}")
        return None


def get_steering_data():
    """Get steering data, preferring .pt files, falling back to report parsing."""
    data = try_load_full_steering_data()
    if data is not None:
        print(f"  Loaded full steering data from .pt files ({len(data)} examples)")
        return data
    data = parse_steering_report()
    if data is not None:
        print(f"  Parsed steering data from report ({len(data)} examples)")
        return data
    return None


def fig5_steering_coverage(data):
    n_show = min(30, len(data))
    subset = data[:n_show]

    x_pos = np.arange(n_show)
    n_probes = subset[0]["n_probes"]

    naive_cov = [d["cov_naive"] for d in subset]
    robust_cov = [d["cov_robust"] for d in subset]

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_w = 0.35

    ax.bar(x_pos - bar_w / 2, naive_cov, bar_w, color=PALETTE["naive"],
           label="Naive Steering", edgecolor="white", linewidth=0.5)
    ax.bar(x_pos + bar_w / 2, robust_cov, bar_w, color=PALETTE["robust"],
           label="Robust Steering", edgecolor="white", linewidth=0.5)

    ax.axhline(n_probes, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"100% Coverage ({n_probes}/{n_probes})")

    ax.set_xlabel("Unsafe Test Example", fontsize=12)
    ax.set_ylabel(f"Probes Steered Successfully (of {n_probes})", fontsize=12)
    ax.set_title("Rashomon Coverage: Naive vs Robust Steering", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(d["idx"]) for d in subset], fontsize=8, rotation=45)
    ax.set_ylim(0, n_probes + 5)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_steering_coverage_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig6_norm_vs_coverage(data):
    naive_norms = [d["norm_naive"] for d in data]
    robust_norms = [d["norm_robust"] for d in data]
    n_probes = data[0]["n_probes"]
    naive_frac = [d["cov_naive"] / n_probes for d in data]
    robust_frac = [d["cov_robust"] / n_probes for d in data]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(naive_norms, naive_frac, c=PALETTE["naive"], s=60, alpha=0.7,
               edgecolors="white", linewidth=0.5, label="Naive Steering", zorder=3)
    ax.scatter(robust_norms, robust_frac, c=PALETTE["robust"], s=60, alpha=0.7,
               edgecolors="white", linewidth=0.5, label="Robust Steering", zorder=3)

    mean_nn = np.mean(naive_norms)
    mean_rn = np.mean(robust_norms)
    mean_nf = np.mean(naive_frac)
    mean_rf = np.mean(robust_frac)

    ax.annotate(f"Mean: ({mean_nn:.1f}, {mean_nf:.0%})",
                xy=(mean_nn, mean_nf), xytext=(mean_nn + 0.5, mean_nf - 0.12),
                fontsize=10, color=PALETTE["naive"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["naive"], lw=1.5))
    ax.annotate(f"Mean: ({mean_rn:.1f}, {mean_rf:.0%})",
                xy=(mean_rn, mean_rf), xytext=(mean_rn - 2.5, mean_rf - 0.12),
                fontsize=10, color=PALETTE["robust"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["robust"], lw=1.5))

    ax.set_xlabel(r"Perturbation Norm $\|\|\delta\|\|$", fontsize=12)
    ax.set_ylabel("Rashomon Coverage (fraction of probes)", fontsize=12)
    ax.set_title("Perturbation Cost vs Safety Coverage", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_norm_vs_coverage.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("Generating figures ...\n")

    print("[1/6] Hamming heatmap — Raw Space")
    fig1_hamming_heatmap_raw()

    print("[2/6] Hamming heatmap — SAE Space")
    fig2_hamming_heatmap_sae()

    print("[3/6] Hamming distribution comparison")
    fig3_hamming_distribution()

    print("[4/6] Probe performance scatter")
    fig4_probe_performance()

    steering_data = get_steering_data()
    if steering_data:
        print("[5/6] Steering coverage comparison")
        fig5_steering_coverage(steering_data)

        print("[6/6] Norm vs coverage")
        fig6_norm_vs_coverage(steering_data)
    else:
        print("[5/6] SKIPPED — steering data not available")
        print("[6/6] SKIPPED — steering data not available")

    print("\nDone. Figures saved to assets/figures/")


if __name__ == "__main__":
    main()
