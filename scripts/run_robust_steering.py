"""Robust vs naive steering comparison on unsafe test examples."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from hessian import compute_hessian
from steering import naive_delta, robust_delta, robust_delta_dynamic, rashomon_coverage

import argparse

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

EPSILON = 0.15
THRESHOLD = 0.0  # logit threshold (sigmoid(0) = 0.5)
MAX_EXAMPLES = 50


def _cosine_similarity(u, v):
    """Return cosine similarity between two vectors (0 if any is zero)."""
    u_norm = torch.norm(u)
    v_norm = torch.norm(v)
    if u_norm.item() == 0.0 or v_norm.item() == 0.0:
        return 0.0
    return torch.dot(u.view(-1), v.view(-1)).item() / (u_norm.item() * v_norm.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", default="mean", choices=["mean", "last", "all"],
                        help="Pooling mode used during activation extraction.")
    parser.add_argument("--model-variant", default="pt",
                        help="Model variant tag matching activation/probe files (e.g. 'pt' or 'it').")
    parser.add_argument(
        "--target-cond", type=float, default=100.0,
        help="Desired condition number when calibrating the Hessian ridge term."
    )
    args = parser.parse_args()

    sfx = "" if args.model_variant == "pt" else f"_{args.model_variant}"
    activations_path = os.path.join(OUTPUT_DIR, f"beavertails_activations_layer10_{args.pooling}{sfx}.pt")
    baseline_path = os.path.join(OUTPUT_DIR, f"baseline_probe_layer10_{args.pooling}{sfx}.pt")
    hessian_path = os.path.join(OUTPUT_DIR, f"hessian_layer10_{args.pooling}{sfx}.pt")
    rashomon_probes_path = os.path.join(OUTPUT_DIR, f"rashomon{sfx}", "rashomon_probes.pt")
    report_path = os.path.join(OUTPUT_DIR, f"steering_comparison_report_{args.pooling}{sfx}.txt")
    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading data ...")
    data = torch.load(activations_path, map_location="cpu", weights_only=False)
    train_X, train_y = data["train_X"], data["train_y"]
    test_X, test_y = data["test_X"], data["test_y"]

    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    w = baseline["weight"].double()
    b = baseline["bias"].double()

    rashomon_probes = torch.load(rashomon_probes_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(rashomon_probes)} Rashomon probes")

    # ── Phase 1: Hessian ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 1: Hessian Computation")
    print("=" * 65)

    # First pass: compute Hessian with minimal ridge to find eigenvalue scale
    H_raw, _, eig_raw = compute_hessian(w, b, train_X, train_y, ridge=1e-6)
    # Set ridge so condition number ≈ 100 (principled: regularize poorly-constrained
    # directions proportionally to the well-constrained ones)
    target_cond = args.target_cond
    if target_cond <= 0:
        raise ValueError("--target-cond must be positive")
    ridge = eig_raw["eig_max"] / target_cond
    print(f"\nUsing adaptive ridge = {ridge:.4f} (target condition ≈ {target_cond})")

    H, H_inv, eig_info = compute_hessian(w, b, train_X, train_y, ridge=ridge)
    torch.save({"H": H, "H_inv": H_inv, "eig_info": eig_info, "ridge": ridge}, hessian_path)
    print(f"Hessian saved to {hessian_path}")

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
        d_dynamic = robust_delta_dynamic(w, b, x, H_inv, EPSILON, THRESHOLD)
        norm_dynamic = d_dynamic.norm().item()

        cos_naive_robust = _cosine_similarity(d_naive, d_robust)
        cos_naive_dynamic = _cosine_similarity(d_naive, d_dynamic)
        cos_robust_dynamic = _cosine_similarity(d_robust, d_dynamic)

        n_cov_naive, n_total, frac_naive = rashomon_coverage(
            d_naive, x, rashomon_probes, THRESHOLD,
        )
        n_cov_robust, _, frac_robust = rashomon_coverage(
            d_robust, x, rashomon_probes, THRESHOLD,
        )
        n_cov_dynamic, _, frac_dynamic = rashomon_coverage(
            d_dynamic, x, rashomon_probes, THRESHOLD,
        )

        results.append({
            "idx": idx,
            "score": score,
            "norm_naive": norm_naive,
            "norm_robust": norm_robust,
            "norm_dynamic": norm_dynamic,
            "cov_naive": n_cov_naive,
            "cov_robust": n_cov_robust,
            "cov_dynamic": n_cov_dynamic,
            "n_probes": n_total,
            "frac_naive": frac_naive,
            "frac_robust": frac_robust,
            "frac_dynamic": frac_dynamic,
            "cos_naive_robust": cos_naive_robust,
            "cos_naive_dynamic": cos_naive_dynamic,
            "cos_robust_dynamic": cos_robust_dynamic,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i+1}/{n_unsafe}: "
                  f"||d_naive||={norm_naive:.4f}  ||d_robust||={norm_robust:.4f}  "
                  f"||d_dynamic||={norm_dynamic:.4f}  "
                  f"cov_naive={n_cov_naive}/{n_total}  cov_robust={n_cov_robust}/{n_total}  "
                  f"cov_dynamic={n_cov_dynamic}/{n_total}  "
                  f"cos(n,r)={cos_naive_robust:.3f}  cos(n,d)={cos_naive_dynamic:.3f}  "
                  f"cos(r,d)={cos_robust_dynamic:.3f}")

    # ── Phase 4: Report ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 4: Summary Report")
    print("=" * 65)

    mean_norm_naive = sum(r["norm_naive"] for r in results) / len(results)
    mean_norm_robust = sum(r["norm_robust"] for r in results) / len(results)
    mean_norm_dynamic = sum(r["norm_dynamic"] for r in results) / len(results)
    mean_cov_naive = sum(r["frac_naive"] for r in results) / len(results)
    mean_cov_robust = sum(r["frac_robust"] for r in results) / len(results)
    mean_cov_dynamic = sum(r["frac_dynamic"] for r in results) / len(results)
    full_cov_naive = sum(1 for r in results if r["cov_naive"] == r["n_probes"]) / len(results)
    full_cov_robust = sum(1 for r in results if r["cov_robust"] == r["n_probes"]) / len(results)
    full_cov_dynamic = sum(1 for r in results if r["cov_dynamic"] == r["n_probes"]) / len(results)
    norm_ratio = mean_norm_robust / mean_norm_naive if mean_norm_naive > 0 else float("inf")
    norm_ratio_dynamic = mean_norm_dynamic / mean_norm_naive if mean_norm_naive > 0 else float("inf")
    mean_cos_nr = sum(r["cos_naive_robust"] for r in results) / len(results)
    mean_cos_nd = sum(r["cos_naive_dynamic"] for r in results) / len(results)
    mean_cos_rd = sum(r["cos_robust_dynamic"] for r in results) / len(results)

    lines = [
        "=" * 75,
        "NAIVE vs ROBUST vs DYNAMIC-ROBUST STEERING COMPARISON REPORT",
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
        f"{'||d_dyn||':>10s}  {'Naive Cov':>10s}  {'Robust Cov':>10s}  {'Dyn Cov':>10s}",
        "-" * 104,
    ]

    for r in results[:20]:
        lines.append(
            f"{r['idx']:>3d}  {r['score']:>8.4f}  {r['norm_naive']:>11.4f}  "
            f"{r['norm_robust']:>12.4f}  {r['norm_dynamic']:>10.4f}  "
            f"{r['cov_naive']:>4d}/{r['n_probes']:<4d}  "
            f"{r['cov_robust']:>4d}/{r['n_probes']:<4d}  "
            f"{r['cov_dynamic']:>4d}/{r['n_probes']:<4d}"
        )

    lines += [
        "",
        "--- Aggregate Statistics ---",
        f"  Mean ||delta_naive||:        {mean_norm_naive:.4f}",
        f"  Mean ||delta_robust||:       {mean_norm_robust:.4f}",
        f"  Mean ||delta_dynamic||:      {mean_norm_dynamic:.4f}",
        f"  Robust/Naive norm ratio:     {norm_ratio:.2f}x",
        f"  Dynamic/Naive norm ratio:    {norm_ratio_dynamic:.2f}x",
        f"  Mean cos(n, r):              {mean_cos_nr:.3f}",
        f"  Mean cos(n, d):              {mean_cos_nd:.3f}",
        f"  Mean cos(r, d):              {mean_cos_rd:.3f}",
        "",
        f"  Mean Rashomon coverage (naive):  {mean_cov_naive:.2%}",
        f"  Mean Rashomon coverage (robust): {mean_cov_robust:.2%}",
        f"  Mean Rashomon coverage (dynamic): {mean_cov_dynamic:.2%}",
        "",
        f"  100% coverage rate (naive):  {full_cov_naive:.1%} of examples",
        f"  100% coverage rate (robust): {full_cov_robust:.1%} of examples",
        f"  100% coverage rate (dynamic): {full_cov_dynamic:.1%} of examples",
        "",
    ]

    if mean_cov_dynamic >= mean_cov_robust and mean_cov_dynamic > mean_cov_naive:
        lines.append(
            "CONCLUSION: Dynamic robust steering gives the best Rashomon coverage "
            "among the three methods on this run."
        )
        lines.append(
            f"  dynamic={mean_cov_dynamic:.1%}, robust={mean_cov_robust:.1%}, "
            f"naive={mean_cov_naive:.1%}; dynamic norm ratio={norm_ratio_dynamic:.2f}x."
        )
    elif mean_cov_robust > mean_cov_naive:
        lines.append(
            "CONCLUSION: Robust steering provides substantially higher Rashomon "
            "coverage than naive steering,"
        )
        lines.append(
            f"  achieving robust={mean_cov_robust:.1%}, dynamic={mean_cov_dynamic:.1%}, "
            f"naive={mean_cov_naive:.1%}; norm ratios robust={norm_ratio:.2f}x, "
            f"dynamic={norm_ratio_dynamic:.2f}x."
        )
    else:
        lines.append(
            "NOTE: Coverage is similar across methods. The Rashomon ellipsoid "
            "may be tightly concentrated around the baseline in this setup."
        )

    lines.append("")
    report = "\n".join(lines)
    print(report)

    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("STEERING COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
