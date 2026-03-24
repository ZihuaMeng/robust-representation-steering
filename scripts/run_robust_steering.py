"""Robust vs naive steering comparison on unsafe test examples."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from hessian import compute_hessian
from steering import naive_delta, robust_delta, rashomon_coverage

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
ACTIVATIONS_PATH = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10.pt")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_probe_layer10.pt")
RASHOMON_PROBES_PATH = os.path.join(OUTPUT_DIR, "rashomon", "rashomon_probes.pt")
HESSIAN_PATH = os.path.join(OUTPUT_DIR, "hessian_layer10.pt")
REPORT_PATH = os.path.join(OUTPUT_DIR, "steering_comparison_report.txt")

EPSILON = 0.15
THRESHOLD = 0.0  # logit threshold (sigmoid(0) = 0.5)
MAX_EXAMPLES = 50


def main():
    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading data ...")
    data = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=False)
    train_X, train_y = data["train_X"], data["train_y"]
    test_X, test_y = data["test_X"], data["test_y"]

    baseline = torch.load(BASELINE_PATH, map_location="cpu", weights_only=False)
    w = baseline["weight"].double()
    b = baseline["bias"].double()

    rashomon_probes = torch.load(RASHOMON_PROBES_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded {len(rashomon_probes)} Rashomon probes")

    # ── Phase 1: Hessian ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 1: Hessian Computation")
    print("=" * 65)

    # First pass: compute Hessian with minimal ridge to find eigenvalue scale
    H_raw, _, eig_raw = compute_hessian(w, b, train_X, train_y, ridge=1e-6)
    # Set ridge so condition number ≈ 100 (principled: regularize poorly-constrained
    # directions proportionally to the well-constrained ones)
    target_cond = 100.0
    ridge = eig_raw["eig_max"] / target_cond
    print(f"\nUsing adaptive ridge = {ridge:.4f} (target condition ≈ {target_cond})")

    H, H_inv, eig_info = compute_hessian(w, b, train_X, train_y, ridge=ridge)
    torch.save({"H": H, "H_inv": H_inv, "eig_info": eig_info, "ridge": ridge}, HESSIAN_PATH)
    print(f"Hessian saved to {HESSIAN_PATH}")

    # ── Phase 2+3: Steering deltas on unsafe examples ────────────────────
    print("\n" + "=" * 65)
    print("PHASE 2-3: Steering Delta Computation")
    print("=" * 65)

    # Find unsafe examples (baseline logit < threshold)
    test_X64 = test_X.double()
    logits = test_X64 @ w + b
    unsafe_mask = logits < THRESHOLD
    unsafe_indices = unsafe_mask.nonzero(as_tuple=True)[0]
    n_unsafe = min(len(unsafe_indices), MAX_EXAMPLES)
    print(f"Unsafe test examples: {len(unsafe_indices)} total, processing {n_unsafe}")

    results = []
    for i in range(n_unsafe):
        idx = unsafe_indices[i].item()
        x = test_X64[idx]
        score = logits[idx].item()

        d_naive, _ = naive_delta(w, b, x, THRESHOLD)
        norm_naive = d_naive.norm().item()

        d_robust = robust_delta(w, b, x, H_inv, EPSILON, THRESHOLD)
        norm_robust = d_robust.norm().item()

        n_cov_naive, n_total, frac_naive = rashomon_coverage(
            d_naive, x, rashomon_probes, THRESHOLD,
        )
        n_cov_robust, _, frac_robust = rashomon_coverage(
            d_robust, x, rashomon_probes, THRESHOLD,
        )

        results.append({
            "idx": idx,
            "score": score,
            "norm_naive": norm_naive,
            "norm_robust": norm_robust,
            "cov_naive": n_cov_naive,
            "cov_robust": n_cov_robust,
            "n_probes": n_total,
            "frac_naive": frac_naive,
            "frac_robust": frac_robust,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i+1}/{n_unsafe}: "
                  f"||d_naive||={norm_naive:.4f}  ||d_robust||={norm_robust:.4f}  "
                  f"cov_naive={n_cov_naive}/{n_total}  cov_robust={n_cov_robust}/{n_total}")

    # ── Phase 4: Report ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 4: Summary Report")
    print("=" * 65)

    mean_norm_naive = sum(r["norm_naive"] for r in results) / len(results)
    mean_norm_robust = sum(r["norm_robust"] for r in results) / len(results)
    mean_cov_naive = sum(r["frac_naive"] for r in results) / len(results)
    mean_cov_robust = sum(r["frac_robust"] for r in results) / len(results)
    full_cov_naive = sum(1 for r in results if r["cov_naive"] == r["n_probes"]) / len(results)
    full_cov_robust = sum(1 for r in results if r["cov_robust"] == r["n_probes"]) / len(results)
    norm_ratio = mean_norm_robust / mean_norm_naive if mean_norm_naive > 0 else float("inf")

    lines = [
        "=" * 75,
        "ROBUST vs NAIVE STEERING COMPARISON REPORT",
        "=" * 75,
        "",
        f"Rashomon epsilon: {EPSILON}",
        f"Logit threshold:  {THRESHOLD} (sigmoid = 0.5)",
        f"Rashomon probes:  {results[0]['n_probes']}",
        f"Unsafe examples:  {len(results)}",
        "",
        "--- Hessian Diagnostics ---",
        f"  Eigenvalue min:     {eig_info['eig_min']:.6e}",
        f"  Eigenvalue max:     {eig_info['eig_max']:.6e}",
        f"  Condition number:   {eig_info['condition']:.2f}",
        "",
        "--- Per-Example Results (first 20) ---",
        f"{'Ex':>3s}  {'Score':>8s}  {'||d_naive||':>11s}  {'||d_robust||':>12s}  "
        f"{'Naive Cov':>10s}  {'Robust Cov':>10s}",
        "-" * 65,
    ]

    for r in results[:20]:
        lines.append(
            f"{r['idx']:>3d}  {r['score']:>8.4f}  {r['norm_naive']:>11.4f}  "
            f"{r['norm_robust']:>12.4f}  "
            f"{r['cov_naive']:>4d}/{r['n_probes']:<4d}  "
            f"{r['cov_robust']:>4d}/{r['n_probes']:<4d}"
        )

    lines += [
        "",
        "--- Aggregate Statistics ---",
        f"  Mean ||delta_naive||:        {mean_norm_naive:.4f}",
        f"  Mean ||delta_robust||:       {mean_norm_robust:.4f}",
        f"  Robust/Naive norm ratio:     {norm_ratio:.2f}x",
        "",
        f"  Mean Rashomon coverage (naive):  {mean_cov_naive:.2%}",
        f"  Mean Rashomon coverage (robust): {mean_cov_robust:.2%}",
        "",
        f"  100% coverage rate (naive):  {full_cov_naive:.1%} of examples",
        f"  100% coverage rate (robust): {full_cov_robust:.1%} of examples",
        "",
    ]

    if mean_cov_robust > mean_cov_naive:
        lines.append(
            "CONCLUSION: Robust steering provides substantially higher Rashomon "
            "coverage than naive steering,"
        )
        lines.append(
            f"  achieving {mean_cov_robust:.1%} mean coverage vs {mean_cov_naive:.1%} "
            f"at a cost of {norm_ratio:.2f}x larger perturbation norm."
        )
    else:
        lines.append(
            "NOTE: Robust and naive coverage are similar. The Rashomon ellipsoid "
            "may be tightly concentrated around the baseline."
        )

    lines.append("")
    report = "\n".join(lines)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to {REPORT_PATH}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("STEERING COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
