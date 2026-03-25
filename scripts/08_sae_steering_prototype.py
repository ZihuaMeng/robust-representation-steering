"""End-to-End SAE Robust Steering Prototype — 4-Way Comparison.

2x2 factorial: {naive, robust} x {raw, SAE->raw}
All four deltas evaluated against the SAME 50 raw-space Rashomon probes.

Strategies:
  (A) naive_raw:      min-norm delta in raw space (raw baseline probe)
  (B) robust_raw:     Rashomon-robust delta in raw space (dense H_inv)
  (C) naive_SAE_raw:  min-norm delta in SAE space, decoded via W_dec
  (D) robust_SAE_raw: robust delta in SAE space (Woodbury), decoded via W_dec
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
SAE_ACT_PATH = os.path.join(OUTPUT_DIR, "beavertails_sae_activations_layer10.pt")
SAE_PROBE_PATH = os.path.join(OUTPUT_DIR, "sae_baseline_probe_layer10.pt")
REPORT_PATH = os.path.join(OUTPUT_DIR, "sae_steering_prototype_report.txt")

EPSILON = 0.15
THRESHOLD = 0.0
MAX_EXAMPLES = 50


# ═══════════════════════════════════════════════════════════════════════
# Woodbury infrastructure (from sae_hessian_feasibility.py)
# ═══════════════════════════════════════════════════════════════════════

def build_woodbury_components(train_X_sae, w_sae, b_sae, weight_decay=0.01):
    """Build Z, U, sigma_sq for Woodbury H^{-1} matvec in SAE space.

    Returns Z, U, sigma_sq, numerical_rank, lam_adaptive.
    """
    X64 = train_X_sae.double()
    w64 = w_sae.double()
    b64 = b_sae.double()

    N, d = X64.shape
    ones = torch.ones(N, 1, dtype=torch.float64)
    X_aug = torch.cat([X64, ones], dim=1)  # [N, d+1]

    # Predicted probabilities at baseline
    logits = X64 @ w64 + b64
    p = torch.sigmoid(logits)
    s = p * (1 - p)

    # Weighted data matrix Z = diag(sqrt(s/N)) @ X_aug
    sqrt_s_over_N = torch.sqrt(s / N).unsqueeze(1)
    Z = sqrt_s_over_N * X_aug  # [N, d+1]

    # Gram matrix eigendecomposition
    G = Z @ Z.T  # [N, N]
    eigvals_gram, U = torch.linalg.eigh(G)

    # Sort descending
    idx_desc = torch.argsort(eigvals_gram, descending=True)
    eigvals_gram = eigvals_gram[idx_desc]
    U = U[:, idx_desc]

    sigma_sq = eigvals_gram
    numerical_rank = (sigma_sq > 1e-10).sum().item()

    # Adaptive ridge: match raw-space approach (cond ~ 100)
    sigma_sq_max = sigma_sq[0].item()
    target_cond = 100.0
    ridge = sigma_sq_max / target_cond
    lam_adaptive = weight_decay + ridge

    print(f"  Woodbury components built: rank={numerical_rank}, "
          f"λ_adaptive={lam_adaptive:.4f}, cond≈{(sigma_sq_max + lam_adaptive)/lam_adaptive:.0f}")

    return Z, U, sigma_sq, numerical_rank, lam_adaptive


def woodbury_matvec(v, Z, U, sigma_sq, lam, numerical_rank):
    """Compute H^{-1} @ v via Woodbury identity."""
    z = Z @ v  # [N]
    U_r = U[:, :numerical_rank]
    sigma_sq_r = sigma_sq[:numerical_rank]
    u = U_r.T @ z  # [r]

    z_perp = z - U_r @ u
    inv_eigvals = 1.0 / (sigma_sq_r + lam)
    term_rank = U_r @ (inv_eigvals * u)
    term_null = (1.0 / lam) * z_perp
    inner = term_rank + term_null
    Zt_inner = Z.T @ inner

    return (1.0 / lam) * (v - Zt_inner)


def robust_delta_sae_implicit(weight, bias, x, Z, U, sigma_sq, lam, numerical_rank,
                               epsilon, threshold=0.0):
    """Robust delta in SAE space using implicit Woodbury H^{-1}."""
    d_naive, score = naive_delta(weight, bias, x, threshold)
    if score >= threshold:
        return torch.zeros_like(x)

    theta = torch.cat([weight, bias.unsqueeze(0)])
    one = torch.ones(1, dtype=x.dtype)

    def check(scale):
        delta = d_naive * scale
        x_aug = torch.cat([x + delta, one])
        linear = theta @ x_aug
        Hinv_x = woodbury_matvec(x_aug, Z, U, sigma_sq, lam, numerical_rank)
        quad = x_aug @ Hinv_x
        return (linear - torch.sqrt(2.0 * epsilon * quad.clamp(min=0.0)) - threshold).item()

    if check(1.0) >= 0:
        return d_naive

    lo, hi = 1.0, 2.0
    for _ in range(30):
        if check(hi) >= 0:
            break
        hi *= 2.0
    else:
        return d_naive * hi

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if check(mid) >= 0:
            hi = mid
        else:
            lo = mid

    return d_naive * hi


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Load All Required Artifacts
# ═══════════════════════════════════════════════════════════════════════

def phase1_load():
    """Load all artifacts and verify dimensional consistency."""
    print("=" * 75)
    print("PHASE 1: Load All Required Artifacts")
    print("=" * 75)

    # Raw-space artifacts
    raw_data = torch.load(RAW_ACT_PATH, map_location="cpu", weights_only=True)
    raw_train_X, raw_train_y = raw_data["train_X"], raw_data["train_y"]
    raw_test_X, raw_test_y = raw_data["test_X"], raw_data["test_y"]

    raw_probe = torch.load(RAW_PROBE_PATH, map_location="cpu", weights_only=True)
    w_raw = raw_probe["weight"].double()
    b_raw = raw_probe["bias"].double()

    raw_hessian = torch.load(RAW_HESSIAN_PATH, map_location="cpu", weights_only=True)
    H_inv_raw = raw_hessian["H_inv"]

    rashomon_probes = torch.load(RAW_RASHOMON_PATH, map_location="cpu", weights_only=False)

    print(f"  Raw activations: train={raw_train_X.shape}, test={raw_test_X.shape}")
    print(f"  Raw probe: w={w_raw.shape}, b={b_raw.shape}")
    print(f"  Raw H_inv: {H_inv_raw.shape}")
    print(f"  Rashomon probes: {len(rashomon_probes)}")

    # SAE-space artifacts
    sae_data = torch.load(SAE_ACT_PATH, map_location="cpu", weights_only=True)
    sae_train_X, sae_train_y = sae_data["train_X"], sae_data["train_y"]
    sae_test_X, sae_test_y = sae_data["test_X"], sae_data["test_y"]

    sae_probe = torch.load(SAE_PROBE_PATH, map_location="cpu", weights_only=True)
    w_sae = sae_probe["weight"].double()
    b_sae = sae_probe["bias"].double()

    print(f"  SAE activations: train={sae_train_X.shape}, test={sae_test_X.shape}")
    print(f"  SAE probe: w={w_sae.shape}, b={b_sae.shape}")

    # SAE decoder weights
    print("\n  Loading SAE decoder (Gemma Scope) ...")
    sae = load_gemma_scope_sae(layer=10, width="16k", l0=77)
    W_dec = sae.W_dec.data.double()  # [d_sae, d_model] = [16384, 2304]
    print(f"  SAE W_dec: {W_dec.shape} (maps {W_dec.shape[0]} -> {W_dec.shape[1]})")

    # Dimensional consistency checks
    d_raw = raw_train_X.shape[1]
    d_sae = sae_train_X.shape[1]
    assert d_raw == 2304, f"Expected d_raw=2304, got {d_raw}"
    assert d_sae == 16384, f"Expected d_sae=16384, got {d_sae}"
    assert W_dec.shape == (d_sae, d_raw), f"W_dec shape {W_dec.shape} != ({d_sae}, {d_raw})"
    assert w_raw.shape[0] == d_raw, f"Raw probe dim {w_raw.shape[0]} != {d_raw}"
    assert w_sae.shape[0] == d_sae, f"SAE probe dim {w_sae.shape[0]} != {d_sae}"
    assert H_inv_raw.shape == (d_raw + 1, d_raw + 1), f"H_inv shape {H_inv_raw.shape}"
    assert raw_train_X.shape[0] == sae_train_X.shape[0], "Train set size mismatch"
    assert raw_test_X.shape[0] == sae_test_X.shape[0], "Test set size mismatch"
    print("\n  All dimensional consistency checks PASSED")

    return {
        "raw_train_X": raw_train_X, "raw_test_X": raw_test_X,
        "raw_train_y": raw_train_y, "raw_test_y": raw_test_y,
        "w_raw": w_raw, "b_raw": b_raw, "H_inv_raw": H_inv_raw,
        "rashomon_probes": rashomon_probes,
        "sae_train_X": sae_train_X, "sae_test_X": sae_test_X,
        "w_sae": w_sae, "b_sae": b_sae,
        "W_dec": W_dec,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 2+3: Compute Deltas and Evaluate Coverage
# ═══════════════════════════════════════════════════════════════════════

def phase23_compute_and_evaluate(artifacts):
    """Compute 4 deltas per unsafe example, evaluate all against raw Rashomon probes."""
    print("\n" + "=" * 75)
    print("PHASE 2-3: Delta Computation and Rashomon Coverage Evaluation")
    print("=" * 75)

    w_raw = artifacts["w_raw"]
    b_raw = artifacts["b_raw"]
    H_inv_raw = artifacts["H_inv_raw"]
    w_sae = artifacts["w_sae"]
    b_sae = artifacts["b_sae"]
    W_dec = artifacts["W_dec"]
    rashomon_probes = artifacts["rashomon_probes"]
    raw_test_X = artifacts["raw_test_X"].double()
    sae_test_X = artifacts["sae_test_X"].double()
    sae_train_X = artifacts["sae_train_X"]

    # Build SAE-space Woodbury components
    print("\n  Building SAE-space Woodbury components ...")
    t_wb = time.time()
    Z, U, sigma_sq, numerical_rank, lam_adaptive = build_woodbury_components(
        sae_train_X, w_sae, b_sae
    )
    print(f"  Woodbury built in {time.time() - t_wb:.1f}s")

    # Find unsafe examples using RAW-space baseline probe
    logits_raw = raw_test_X @ w_raw + b_raw
    unsafe_mask = logits_raw < THRESHOLD
    unsafe_indices = unsafe_mask.nonzero(as_tuple=True)[0]
    n_unsafe = min(len(unsafe_indices), MAX_EXAMPLES)
    print(f"\n  Unsafe test examples (raw probe): {len(unsafe_indices)} total, processing {n_unsafe}")

    results = []
    t_loop = time.time()

    for i in range(n_unsafe):
        idx = unsafe_indices[i].item()
        x_raw = raw_test_X[idx]
        x_sae = sae_test_X[idx]
        score_raw = logits_raw[idx].item()

        # (A) Naive raw
        dA, _ = naive_delta(w_raw, b_raw, x_raw, THRESHOLD)
        norm_A = dA.norm().item()

        # (B) Robust raw
        dB = robust_delta(w_raw, b_raw, x_raw, H_inv_raw, EPSILON, THRESHOLD)
        norm_B = dB.norm().item()

        # (C) Naive SAE -> raw
        dC_sae, _ = naive_delta(w_sae, b_sae, x_sae, THRESHOLD)
        dC = (dC_sae @ W_dec)  # decode: [d_sae] @ [d_sae, d_model] -> [d_model]
        norm_C = dC.norm().item()

        # (D) Robust SAE -> raw
        dD_sae = robust_delta_sae_implicit(
            w_sae, b_sae, x_sae, Z, U, sigma_sq, lam_adaptive,
            numerical_rank, EPSILON, THRESHOLD
        )
        dD = (dD_sae @ W_dec)  # decode
        norm_D = dD.norm().item()

        # Evaluate all 4 against raw-space Rashomon probes
        covA = rashomon_coverage(dA, x_raw, rashomon_probes, THRESHOLD)
        covB = rashomon_coverage(dB, x_raw, rashomon_probes, THRESHOLD)
        covC = rashomon_coverage(dC, x_raw, rashomon_probes, THRESHOLD)
        covD = rashomon_coverage(dD, x_raw, rashomon_probes, THRESHOLD)

        results.append({
            "idx": idx, "score": score_raw,
            "norm_A": norm_A, "norm_B": norm_B, "norm_C": norm_C, "norm_D": norm_D,
            "cov_A": covA[0], "cov_B": covB[0], "cov_C": covC[0], "cov_D": covD[0],
            "n_probes": covA[1],
            "frac_A": covA[2], "frac_B": covB[2], "frac_C": covC[2], "frac_D": covD[2],
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_loop
            print(f"  [{i+1}/{n_unsafe}] idx={idx:>3d}  "
                  f"norms=[{norm_A:.3f}, {norm_B:.3f}, {norm_C:.3f}, {norm_D:.3f}]  "
                  f"cov=[{covA[0]}, {covB[0]}, {covC[0]}, {covD[0]}]/{covA[1]}  "
                  f"({elapsed:.1f}s)")

    print(f"\n  All {n_unsafe} examples processed in {time.time() - t_loop:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Comparison Report
# ═══════════════════════════════════════════════════════════════════════

def phase4_report(results):
    """Generate the four-way comparison report."""
    print("\n" + "=" * 75)
    print("PHASE 4: Four-Way Comparison Report")
    print("=" * 75)

    n = len(results)
    n_probes = results[0]["n_probes"]

    # Aggregates
    mean_norm = {k: sum(r[f"norm_{k}"] for r in results) / n for k in "ABCD"}
    mean_cov = {k: sum(r[f"frac_{k}"] for r in results) / n for k in "ABCD"}
    full_rate = {k: sum(1 for r in results if r[f"cov_{k}"] == n_probes) / n for k in "ABCD"}
    eff = {k: mean_cov[k] / mean_norm[k] if mean_norm[k] > 0 else 0.0 for k in "ABCD"}

    labels = {"A": "Naive Raw", "B": "Robust Raw", "C": "Naive SAE→Raw", "D": "Robust SAE→Raw"}

    # Build report
    lines = []
    lines.append("=" * 95)
    lines.append("FOUR-WAY SAE ROBUST STEERING COMPARISON REPORT")
    lines.append("=" * 95)
    lines.append("")
    lines.append(f"Rashomon epsilon: {EPSILON}")
    lines.append(f"Logit threshold:  {THRESHOLD} (sigmoid = 0.5)")
    lines.append(f"Rashomon probes:  {n_probes}")
    lines.append(f"Unsafe examples:  {n}")
    lines.append("")
    lines.append("Strategies:")
    lines.append("  (A) Naive Raw:       min-norm delta in raw space")
    lines.append("  (B) Robust Raw:      Rashomon-robust delta in raw space (dense H_inv)")
    lines.append("  (C) Naive SAE→Raw:   min-norm delta in SAE space, decoded via W_dec")
    lines.append("  (D) Robust SAE→Raw:  robust delta in SAE space (Woodbury + adaptive ridge), decoded via W_dec")
    lines.append("")

    # Per-example table
    lines.append("--- Per-Example Results (first 20) ---")
    hdr = (f"{'Ex':>3s}  {'||δ_A||':>8s}  {'||δ_B||':>8s}  {'||δ_C||':>8s}  {'||δ_D||':>8s}  "
           f"{'Cov_A':>6s}  {'Cov_B':>6s}  {'Cov_C':>6s}  {'Cov_D':>6s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in results[:20]:
        lines.append(
            f"{r['idx']:>3d}  {r['norm_A']:>8.4f}  {r['norm_B']:>8.4f}  "
            f"{r['norm_C']:>8.4f}  {r['norm_D']:>8.4f}  "
            f"{r['cov_A']:>3d}/{n_probes:<2d}  {r['cov_B']:>3d}/{n_probes:<2d}  "
            f"{r['cov_C']:>3d}/{n_probes:<2d}  {r['cov_D']:>3d}/{n_probes:<2d}"
        )

    lines.append("")

    # Aggregate table
    lines.append("--- Aggregate Statistics ---")
    lines.append("")
    agg_hdr = (f"{'Metric':<25s}  {'Naive Raw':>10s}  {'Robust Raw':>10s}  "
               f"{'Naive SAE→Raw':>14s}  {'Robust SAE→Raw':>15s}")
    lines.append(agg_hdr)
    lines.append("-" * len(agg_hdr))

    lines.append(f"{'Mean ||δ||':<25s}  {mean_norm['A']:>10.4f}  {mean_norm['B']:>10.4f}  "
                 f"{mean_norm['C']:>14.4f}  {mean_norm['D']:>15.4f}")
    lines.append(f"{'Mean Rashomon coverage':<25s}  {mean_cov['A']:>10.2%}  {mean_cov['B']:>10.2%}  "
                 f"{mean_cov['C']:>14.2%}  {mean_cov['D']:>15.2%}")
    lines.append(f"{'100% coverage rate':<25s}  {full_rate['A']:>10.1%}  {full_rate['B']:>10.1%}  "
                 f"{full_rate['C']:>14.1%}  {full_rate['D']:>15.1%}")
    lines.append(f"{'Coverage per unit norm':<25s}  {eff['A']:>10.4f}  {eff['B']:>10.4f}  "
                 f"{eff['C']:>14.4f}  {eff['D']:>15.4f}")

    lines.append("")

    # 2x2 factorial decomposition
    lines.append("--- 2x2 Factorial Decomposition ---")
    lines.append("")

    # Space effect (raw vs SAE->raw)
    space_naive = mean_cov["C"] - mean_cov["A"]
    space_robust = mean_cov["D"] - mean_cov["B"]
    lines.append("EFFECT OF SPACE (SAE→raw vs raw), holding optimization fixed:")
    lines.append(f"  Naive:  SAE→raw - raw = {mean_cov['C']:.2%} - {mean_cov['A']:.2%} = {space_naive:+.2%}")
    lines.append(f"  Robust: SAE→raw - raw = {mean_cov['D']:.2%} - {mean_cov['B']:.2%} = {space_robust:+.2%}")
    if abs(space_naive) < 0.01 and abs(space_robust) < 0.01:
        lines.append("  => Space has negligible effect on coverage.")
    elif space_naive > 0 and space_robust > 0:
        lines.append("  => SAE space IMPROVES coverage in both naive and robust settings.")
    elif space_naive < 0 and space_robust < 0:
        lines.append("  => SAE space HURTS coverage in both settings.")
    else:
        lines.append("  => Mixed effect: SAE space helps in one optimization but not the other.")
    lines.append("")

    # Optimization effect (naive vs robust)
    opt_raw = mean_cov["B"] - mean_cov["A"]
    opt_sae = mean_cov["D"] - mean_cov["C"]
    lines.append("EFFECT OF OPTIMIZATION (robust vs naive), holding space fixed:")
    lines.append(f"  Raw:     robust - naive = {mean_cov['B']:.2%} - {mean_cov['A']:.2%} = {opt_raw:+.2%}")
    lines.append(f"  SAE→raw: robust - naive = {mean_cov['D']:.2%} - {mean_cov['C']:.2%} = {opt_sae:+.2%}")
    if opt_raw > 0.01 and opt_sae > 0.01:
        lines.append("  => Robust optimization IMPROVES coverage in both spaces.")
    elif opt_raw < -0.01 and opt_sae < -0.01:
        lines.append("  => Robust optimization HURTS coverage in both spaces (unexpected).")
    else:
        lines.append("  => Mixed or negligible optimization effect.")
    lines.append("")

    # Interaction
    interaction = (mean_cov["D"] - mean_cov["C"]) - (mean_cov["B"] - mean_cov["A"])
    lines.append("INTERACTION (is robust optimization more valuable in one space?):")
    lines.append(f"  (D-C) - (B-A) = {opt_sae:+.2%} - {opt_raw:+.2%} = {interaction:+.2%}")
    if abs(interaction) < 0.05:
        lines.append("  => No meaningful interaction: robust optimization adds similar value in both spaces.")
    elif interaction > 0:
        lines.append("  => Positive interaction: robust optimization is MORE valuable in SAE→raw space.")
    else:
        lines.append("  => Negative interaction: robust optimization is MORE valuable in raw space.")
    lines.append("")

    # Norm costs
    lines.append("NORM COST ANALYSIS:")
    lines.append(f"  Raw:     robust/naive norm ratio = {mean_norm['B']/mean_norm['A']:.2f}x")
    if mean_norm["C"] > 0:
        lines.append(f"  SAE→raw: robust/naive norm ratio = {mean_norm['D']/mean_norm['C']:.2f}x")
    lines.append(f"  SAE→raw / raw norm ratio (naive):  {mean_norm['C']/mean_norm['A']:.2f}x")
    lines.append(f"  SAE→raw / raw norm ratio (robust): {mean_norm['D']/mean_norm['B']:.2f}x")
    lines.append("")

    # Scientific verdict
    lines.append("=" * 95)
    lines.append("SCIENTIFIC VERDICT")
    lines.append("=" * 95)

    # Determine verdict based on results
    best_key = max("ABCD", key=lambda k: mean_cov[k])
    best_eff_key = max("ABCD", key=lambda k: eff[k])

    lines.append("")
    lines.append(f"Best mean Rashomon coverage:     {labels[best_key]} ({mean_cov[best_key]:.2%})")
    lines.append(f"Best coverage per unit norm:     {labels[best_eff_key]} ({eff[best_eff_key]:.4f})")
    lines.append("")

    # Viability assessment
    sae_viable = mean_cov["D"] >= 0.8 * mean_cov["B"]
    if sae_viable:
        lines.append("SAE-decoded robust steering is VIABLE.")
        lines.append(f"  Robust SAE→raw achieves {mean_cov['D']:.2%} coverage "
                     f"({mean_cov['D']/mean_cov['B']:.0%} of raw-space robust).")
    else:
        lines.append("SAE-decoded robust steering shows REDUCED effectiveness.")
        lines.append(f"  Robust SAE→raw achieves {mean_cov['D']:.2%} coverage "
                     f"vs {mean_cov['B']:.2%} for raw-space robust.")

    lines.append("")
    lines.append("Factorial decomposition summary:")
    lines.append(f"  - Space effect (SAE→raw vs raw):     naive {space_naive:+.2%}, robust {space_robust:+.2%}")
    lines.append(f"  - Optimization effect (robust vs naive): raw {opt_raw:+.2%}, SAE {opt_sae:+.2%}")
    lines.append(f"  - Interaction:                        {interaction:+.2%}")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")

    return {
        "mean_norm": mean_norm, "mean_cov": mean_cov,
        "full_rate": full_rate, "eff": eff,
        "space_naive": space_naive, "space_robust": space_robust,
        "opt_raw": opt_raw, "opt_sae": opt_sae,
        "interaction": interaction,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    artifacts = phase1_load()
    results = phase23_compute_and_evaluate(artifacts)
    summary = phase4_report(results)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("4-WAY SAE STEERING PROTOTYPE COMPLETE")


if __name__ == "__main__":
    main()
