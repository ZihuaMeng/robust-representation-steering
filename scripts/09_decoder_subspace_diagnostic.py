"""Decoder Column-Space Diagnostic.

Tests whether col(W_dec) contains the steering directions needed for
raw-space Rashomon coverage by projecting known-good raw deltas onto
the decoder's column space.

If delta_robust_raw achieves 100% coverage, and its projection onto
col(W_dec) preserves most of that coverage, then the bottleneck is the
SAE-space optimizer, not the subspace. If coverage collapses, the
subspace lacks critical directions.

Convention note:
  W_dec in code: [d_sae, d_model] = [16384, 2304], decode = z @ W_dec
  col(W_dec) in the mathematical sense = row_space(W_dec_code)
  = column space of W_dec_code.T = image of the decode mapping
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from steering import naive_delta, robust_delta, rashomon_coverage
from sae_utils import load_gemma_scope_sae

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RAW_ACT_PATH = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10.pt")
RAW_PROBE_PATH = os.path.join(OUTPUT_DIR, "baseline_probe_layer10.pt")
RAW_HESSIAN_PATH = os.path.join(OUTPUT_DIR, "hessian_layer10.pt")
RAW_RASHOMON_PATH = os.path.join(OUTPUT_DIR, "rashomon", "rashomon_probes.pt")
REPORT_PATH = os.path.join(OUTPUT_DIR, "decoder_subspace_diagnostic.txt")

EPSILON = 0.15
THRESHOLD = 0.0
MAX_EXAMPLES = 50


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Load Artifacts
# ═══════════════════════════════════════════════════════════════════════

def phase1_load():
    """Load raw-space artifacts and SAE decoder weights."""
    print("=" * 75)
    print("PHASE 1: Load Artifacts")
    print("=" * 75)

    raw_data = torch.load(RAW_ACT_PATH, map_location="cpu", weights_only=True)
    raw_test_X = raw_data["test_X"]

    raw_probe = torch.load(RAW_PROBE_PATH, map_location="cpu", weights_only=True)
    w_raw = raw_probe["weight"].double()
    b_raw = raw_probe["bias"].double()

    raw_hessian = torch.load(RAW_HESSIAN_PATH, map_location="cpu", weights_only=True)
    H_inv_raw = raw_hessian["H_inv"]

    rashomon_probes = torch.load(RAW_RASHOMON_PATH, map_location="cpu", weights_only=False)

    print(f"  Raw activations: test={raw_test_X.shape}")
    print(f"  Raw probe: w={w_raw.shape}, b={b_raw.shape}")
    print(f"  Raw H_inv: {H_inv_raw.shape}")
    print(f"  Rashomon probes: {len(rashomon_probes)}")

    print("\n  Loading SAE decoder (Gemma Scope) ...")
    sae = load_gemma_scope_sae(layer=10, width="16k", l0=77)
    W_dec = sae.W_dec.data.double()  # [16384, 2304]
    print(f"  W_dec: {W_dec.shape} (maps SAE[{W_dec.shape[0]}] -> raw[{W_dec.shape[1]}])")

    # Consistency checks
    d_raw = raw_test_X.shape[1]
    assert W_dec.shape[1] == d_raw, f"W_dec cols {W_dec.shape[1]} != d_raw {d_raw}"
    assert w_raw.shape[0] == d_raw
    assert H_inv_raw.shape == (d_raw + 1, d_raw + 1)
    print("  All consistency checks PASSED")

    return {
        "raw_test_X": raw_test_X,
        "w_raw": w_raw, "b_raw": b_raw,
        "H_inv_raw": H_inv_raw,
        "rashomon_probes": rashomon_probes,
        "W_dec": W_dec,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: W_dec Spectral Characterization
# ═══════════════════════════════════════════════════════════════════════

def phase2_spectral(W_dec):
    """Characterize W_dec via eigendecomposition of W_dec^T @ W_dec."""
    print("\n" + "=" * 75)
    print("PHASE 2: W_dec Spectral Characterization")
    print("=" * 75)

    d_sae, d_model = W_dec.shape
    print(f"  W_dec shape: [{d_sae}, {d_model}]")
    print(f"  Maximum possible rank: min({d_sae}, {d_model}) = {d_model}")

    # Gram matrix: eigenvalues = singular_values^2, eigenvectors = right singular vectors
    print(f"\n  Computing Gram matrix W_dec^T @ W_dec [{d_model}x{d_model}] ...")
    t0 = time.time()
    G = W_dec.T @ W_dec  # [2304, 2304]
    eigvals, eigvecs = torch.linalg.eigh(G)

    # Sort descending
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    singular_values = torch.sqrt(eigvals.clamp(min=0))
    print(f"  Eigendecomposition: {time.time() - t0:.1f}s")

    # Rank analysis
    numerical_rank = (singular_values > 1e-10 * singular_values[0]).sum().item()
    print(f"\n  Numerical rank: {numerical_rank} / {d_model}")

    if numerical_rank == d_model:
        print(f"  *** W_dec is FULL RANK — col(W_dec) = R^{d_model} ***")
        print(f"  Projection onto col(W_dec) is the IDENTITY.")
        print(f"  All raw-space directions are representable as SAE-decoded vectors.")

    # Top singular values
    print(f"\n  Top 20 singular values:")
    total_energy = eigvals.sum().item()
    for i in range(min(20, len(singular_values))):
        cum_energy = eigvals[:i + 1].sum().item() / total_energy
        print(f"    sigma[{i:>3d}] = {singular_values[i].item():.6f}  "
              f"(sigma^2 = {eigvals[i].item():.4f}, cumulative: {cum_energy:.6f})")

    # Bottom 5 singular values
    print(f"\n  Bottom 5 singular values:")
    for i in range(max(0, d_model - 5), d_model):
        print(f"    sigma[{i:>4d}] = {singular_values[i].item():.6e}")

    # Effective rank at thresholds
    for thresh in [0.90, 0.95, 0.99, 0.999]:
        cumsum = eigvals.cumsum(0)
        eff = (cumsum < thresh * total_energy).sum().item() + 1
        print(f"  Effective rank ({thresh:.1%} energy): {eff}")

    # Condition number
    cond = singular_values[0].item() / singular_values[numerical_rank - 1].item()
    print(f"  Condition number (sigma_max/sigma_min): {cond:.2e}")

    return numerical_rank, eigvecs, eigvals, singular_values


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Project Raw Deltas and Evaluate Coverage
# ═══════════════════════════════════════════════════════════════════════

def phase3_project_and_evaluate(artifacts, numerical_rank, eigvecs):
    """Compute raw deltas, project onto col(W_dec), evaluate all 4 variants."""
    print("\n" + "=" * 75)
    print("PHASE 3: Compute Deltas, Project, and Evaluate Coverage")
    print("=" * 75)

    w_raw = artifacts["w_raw"]
    b_raw = artifacts["b_raw"]
    H_inv_raw = artifacts["H_inv_raw"]
    rashomon_probes = artifacts["rashomon_probes"]
    raw_test_X = artifacts["raw_test_X"].double()

    # Projection basis: first `numerical_rank` eigenvectors of G
    # These are the right singular vectors of W_dec, spanning its row space
    # (= column space of W_dec^T = image of the decode mapping)
    V_r = eigvecs[:, :numerical_rank]  # [d_model, r]

    is_full_rank = (numerical_rank == raw_test_X.shape[1])
    if is_full_rank:
        print("  W_dec is full rank — projection is identity.")
        print("  Coverage will be IDENTICAL for raw and projected variants.")
        print("  (Running anyway for confirmation.)\n")

    # Find unsafe examples using raw-space baseline probe
    logits_raw = raw_test_X @ w_raw + b_raw
    unsafe_mask = logits_raw < THRESHOLD
    unsafe_indices = unsafe_mask.nonzero(as_tuple=True)[0]
    n_unsafe = min(len(unsafe_indices), MAX_EXAMPLES)
    print(f"  Unsafe test examples: {len(unsafe_indices)}, processing {n_unsafe}")

    results = []
    t_loop = time.time()

    for i in range(n_unsafe):
        idx = unsafe_indices[i].item()
        x_raw = raw_test_X[idx]

        # Compute raw-space deltas
        d_naive, _ = naive_delta(w_raw, b_raw, x_raw, THRESHOLD)
        d_robust = robust_delta(w_raw, b_raw, x_raw, H_inv_raw, EPSILON, THRESHOLD)

        # Project onto col(W_dec) = row_space(W_dec_code)
        # delta_proj = V_r @ (V_r^T @ delta)
        d_naive_proj = V_r @ (V_r.T @ d_naive)
        d_robust_proj = V_r @ (V_r.T @ d_robust)

        # Orthogonal complements
        d_naive_perp = d_naive - d_naive_proj
        d_robust_perp = d_robust - d_robust_proj

        # Norms
        norm_naive = d_naive.norm().item()
        norm_naive_proj = d_naive_proj.norm().item()
        norm_naive_perp = d_naive_perp.norm().item()
        norm_robust = d_robust.norm().item()
        norm_robust_proj = d_robust_proj.norm().item()
        norm_robust_perp = d_robust_perp.norm().item()

        # Energy fractions
        energy_naive = (norm_naive_proj ** 2 / norm_naive ** 2) if norm_naive > 0 else 0.0
        energy_robust = (norm_robust_proj ** 2 / norm_robust ** 2) if norm_robust > 0 else 0.0

        # Coverage: 4 variants against raw-space Rashomon probes
        cov_robust_raw = rashomon_coverage(d_robust, x_raw, rashomon_probes, THRESHOLD)
        cov_robust_proj = rashomon_coverage(d_robust_proj, x_raw, rashomon_probes, THRESHOLD)
        cov_naive_raw = rashomon_coverage(d_naive, x_raw, rashomon_probes, THRESHOLD)
        cov_naive_proj = rashomon_coverage(d_naive_proj, x_raw, rashomon_probes, THRESHOLD)

        results.append({
            "idx": idx,
            "norm_naive": norm_naive, "norm_naive_proj": norm_naive_proj,
            "norm_naive_perp": norm_naive_perp, "energy_naive": energy_naive,
            "norm_robust": norm_robust, "norm_robust_proj": norm_robust_proj,
            "norm_robust_perp": norm_robust_perp, "energy_robust": energy_robust,
            "cov_robust_raw": cov_robust_raw,
            "cov_robust_proj": cov_robust_proj,
            "cov_naive_raw": cov_naive_raw,
            "cov_naive_proj": cov_naive_proj,
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_loop
            print(f"  [{i+1}/{n_unsafe}] idx={idx:>3d}  "
                  f"E_naive={energy_naive:.6f}  E_robust={energy_robust:.6f}  "
                  f"cov_rob=[{cov_robust_raw[0]},{cov_robust_proj[0]}]/{cov_robust_raw[1]}  "
                  f"({elapsed:.1f}s)")

    print(f"\n  All {n_unsafe} examples processed in {time.time() - t_loop:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Diagnostic Report
# ═══════════════════════════════════════════════════════════════════════

def phase4_report(results, numerical_rank, singular_values, eigvals):
    """Generate the diagnostic report with tables and verdict."""
    print("\n" + "=" * 75)
    print("PHASE 4: Diagnostic Report")
    print("=" * 75)

    n = len(results)
    n_probes = results[0]["cov_robust_raw"][1]
    d_model = len(singular_values)

    # Aggregates — energy decomposition
    mean_norm_naive = sum(r["norm_naive"] for r in results) / n
    mean_norm_naive_proj = sum(r["norm_naive_proj"] for r in results) / n
    mean_norm_naive_perp = sum(r["norm_naive_perp"] for r in results) / n
    mean_energy_naive = sum(r["energy_naive"] for r in results) / n

    mean_norm_robust = sum(r["norm_robust"] for r in results) / n
    mean_norm_robust_proj = sum(r["norm_robust_proj"] for r in results) / n
    mean_norm_robust_perp = sum(r["norm_robust_perp"] for r in results) / n
    mean_energy_robust = sum(r["energy_robust"] for r in results) / n

    # Aggregates — coverage
    mean_cov_robust_raw = sum(r["cov_robust_raw"][2] for r in results) / n
    mean_cov_robust_proj = sum(r["cov_robust_proj"][2] for r in results) / n
    mean_cov_naive_raw = sum(r["cov_naive_raw"][2] for r in results) / n
    mean_cov_naive_proj = sum(r["cov_naive_proj"][2] for r in results) / n

    full_rate_robust_raw = sum(1 for r in results if r["cov_robust_raw"][0] == n_probes) / n
    full_rate_robust_proj = sum(1 for r in results if r["cov_robust_proj"][0] == n_probes) / n
    full_rate_naive_raw = sum(1 for r in results if r["cov_naive_raw"][0] == n_probes) / n
    full_rate_naive_proj = sum(1 for r in results if r["cov_naive_proj"][0] == n_probes) / n

    # Build report
    L = []
    L.append("=" * 95)
    L.append("DECODER COLUMN-SPACE DIAGNOSTIC REPORT")
    L.append("=" * 95)
    L.append("")
    L.append(f"Rashomon epsilon: {EPSILON}")
    L.append(f"Logit threshold:  {THRESHOLD}")
    L.append(f"Rashomon probes:  {n_probes}")
    L.append(f"Unsafe examples:  {n}")
    L.append("")

    # W_dec characterization
    L.append("--- W_dec Spectral Characterization ---")
    L.append(f"  Shape: [{16384}, {d_model}] (d_sae x d_model)")
    L.append(f"  Numerical rank: {numerical_rank} / {d_model}")
    L.append(f"  Max singular value: {singular_values[0].item():.6f}")
    L.append(f"  Min singular value: {singular_values[numerical_rank - 1].item():.6e}")
    cond = singular_values[0].item() / singular_values[numerical_rank - 1].item()
    L.append(f"  Condition number: {cond:.2e}")
    total_energy = eigvals[:numerical_rank].sum().item()
    for thresh in [0.90, 0.95, 0.99, 0.999]:
        cumsum = eigvals[:numerical_rank].cumsum(0)
        eff = (cumsum < thresh * total_energy).sum().item() + 1
        L.append(f"  Effective rank ({thresh:.1%} energy): {eff}")
    L.append("")

    if numerical_rank == d_model:
        L.append("  *** W_dec is FULL RANK ***")
        L.append(f"  col(W_dec) = R^{d_model} — the decoder can represent ANY direction.")
        L.append("  Projection onto col(W_dec) is the identity mapping.")
        L.append("")

    # Energy decomposition table
    L.append("--- Energy Decomposition ---")
    L.append("")
    hdr = (f"{'Delta type':<15s}  {'Mean ||d_raw||':>14s}  {'Mean ||d_proj||':>15s}  "
           f"{'Mean ||d_perp||':>15s}  {'Energy in col(W_dec)':>22s}")
    L.append(hdr)
    L.append("-" * len(hdr))
    L.append(f"{'Naive raw':<15s}  {mean_norm_naive:>14.4f}  {mean_norm_naive_proj:>15.4f}  "
             f"{mean_norm_naive_perp:>15.6f}  {mean_energy_naive:>22.6%}")
    L.append(f"{'Robust raw':<15s}  {mean_norm_robust:>14.4f}  {mean_norm_robust_proj:>15.4f}  "
             f"{mean_norm_robust_perp:>15.6f}  {mean_energy_robust:>22.6%}")
    L.append("")

    # Coverage comparison table
    L.append("--- Coverage Comparison ---")
    L.append("")
    hdr2 = f"{'Strategy':<25s}  {'Mean Coverage':>14s}  {'100% Coverage Rate':>18s}"
    L.append(hdr2)
    L.append("-" * len(hdr2))
    L.append(f"{'Robust raw':<25s}  {mean_cov_robust_raw:>14.2%}  {full_rate_robust_raw:>18.1%}")
    L.append(f"{'Robust projected':<25s}  {mean_cov_robust_proj:>14.2%}  {full_rate_robust_proj:>18.1%}")
    L.append(f"{'Naive raw':<25s}  {mean_cov_naive_raw:>14.2%}  {full_rate_naive_raw:>18.1%}")
    L.append(f"{'Naive projected':<25s}  {mean_cov_naive_proj:>14.2%}  {full_rate_naive_proj:>18.1%}")
    L.append("")

    # Coverage change from projection
    L.append("--- Coverage Change from Projection ---")
    robust_drop = mean_cov_robust_proj - mean_cov_robust_raw
    naive_drop = mean_cov_naive_proj - mean_cov_naive_raw
    L.append(f"  Robust: {mean_cov_robust_raw:.2%} -> {mean_cov_robust_proj:.2%} "
             f"({robust_drop:+.2%})")
    L.append(f"  Naive:  {mean_cov_naive_raw:.2%} -> {mean_cov_naive_proj:.2%} "
             f"({naive_drop:+.2%})")
    L.append("")

    # Per-example detail (first 20)
    L.append("--- Per-Example Detail (first 20) ---")
    ehdr = (f"{'Ex':>3s}  {'||d_rob||':>9s}  {'E_rob':>8s}  "
            f"{'Cov_raw':>7s}  {'Cov_proj':>8s}  "
            f"{'||d_nai||':>9s}  {'E_nai':>8s}  "
            f"{'Cov_raw':>7s}  {'Cov_proj':>8s}")
    L.append(ehdr)
    L.append("-" * len(ehdr))
    for r in results[:20]:
        L.append(
            f"{r['idx']:>3d}  {r['norm_robust']:>9.4f}  {r['energy_robust']:>8.6f}  "
            f"{r['cov_robust_raw'][0]:>3d}/{n_probes:<2d}  "
            f"{r['cov_robust_proj'][0]:>4d}/{n_probes:<2d}  "
            f"{r['norm_naive']:>9.4f}  {r['energy_naive']:>8.6f}  "
            f"{r['cov_naive_raw'][0]:>3d}/{n_probes:<2d}  "
            f"{r['cov_naive_proj'][0]:>4d}/{n_probes:<2d}"
        )
    L.append("")

    # Diagnostic verdict
    L.append("=" * 95)
    L.append("DIAGNOSTIC VERDICT")
    L.append("=" * 95)
    L.append("")

    if mean_cov_robust_proj >= 0.80:
        L.append("col(W_dec) CONTAINS effective steering directions.")
        L.append("")
        L.append("Evidence:")
        L.append(f"  - Robust projected coverage: {mean_cov_robust_proj:.2%} "
                 f"(vs {mean_cov_robust_raw:.2%} raw)")
        L.append(f"  - {mean_energy_robust:.4%} of robust delta energy lies in col(W_dec)")
        if mean_cov_robust_raw > 0:
            L.append(f"  - Projection preserves "
                     f"{mean_cov_robust_proj / mean_cov_robust_raw:.1%} of raw coverage")
        L.append(f"  - 100% coverage rate: {full_rate_robust_proj:.1%} "
                 f"(vs {full_rate_robust_raw:.1%} raw)")
        L.append("")
        L.append("The bottleneck is the SAE-space optimizer, not the subspace.")
        L.append("The decoder's column space can represent the directions needed for full")
        L.append("Rashomon coverage. The 13.56% coverage from the 4-way prototype is due")
        L.append("to the SAE-space optimization producing suboptimal coefficients, not")
        L.append("because the subspace lacks the right directions.")
        L.append("")
        L.append("Recommended next step: projection-aware optimization directly in col(W_dec).")
        L.append("This would optimize in raw space but constrain the delta to lie in the")
        L.append("decoder's column space, bypassing the SAE-space probe/Hessian entirely.")

    elif mean_cov_robust_proj < 0.30:
        out_pct = 1.0 - mean_energy_robust
        L.append("col(W_dec) LACKS the directions needed for Rashomon coverage.")
        L.append("")
        L.append("Evidence:")
        L.append(f"  - Robust projected coverage: {mean_cov_robust_proj:.2%} "
                 f"(vs {mean_cov_robust_raw:.2%} raw)")
        L.append(f"  - {out_pct:.4%} of robust delta energy lies OUTSIDE col(W_dec)")
        L.append(f"  - The out-of-subspace component is critical for coverage")
        L.append(f"  - 100% coverage rate: {full_rate_robust_proj:.1%} "
                 f"(vs {full_rate_robust_raw:.1%} raw)")
        L.append("")
        L.append("The decoder subspace fundamentally cannot represent the needed directions.")
        L.append("Recommended next step: hybrid approach combining in-subspace and")
        L.append("out-of-subspace components.")

    else:
        L.append("INTERMEDIATE RESULT — col(W_dec) partially contains needed directions.")
        L.append("")
        L.append("Evidence:")
        L.append(f"  - Robust projected coverage: {mean_cov_robust_proj:.2%} "
                 f"(vs {mean_cov_robust_raw:.2%} raw)")
        L.append(f"  - {mean_energy_robust:.4%} of robust delta energy in col(W_dec)")
        L.append(f"  - {1.0 - mean_energy_robust:.4%} outside col(W_dec)")
        L.append(f"  - 100% coverage rate: {full_rate_robust_proj:.1%} "
                 f"(vs {full_rate_robust_raw:.1%} raw)")
        L.append("")
        L.append("The subspace captures some but not all needed directions.")
        L.append("Both optimizer improvement and subspace expansion may help.")

    L.append("")

    report = "\n".join(L)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")

    return report


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    artifacts = phase1_load()
    numerical_rank, eigvecs, eigvals, singular_values = phase2_spectral(artifacts["W_dec"])
    results = phase3_project_and_evaluate(artifacts, numerical_rank, eigvecs)
    phase4_report(results, numerical_rank, singular_values, eigvals)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("DECODER COLUMN-SPACE DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
