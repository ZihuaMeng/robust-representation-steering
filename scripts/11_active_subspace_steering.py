"""Active-Subspace SAE Steering Pipeline (Script 11).

Restrict the Hessian and QP solver to the ACTIVE SAE feature subspace
(features nonzero in at least K training examples). This avoids the full
16k null-space problem and may produce deltas with fundamentally different
geometry from the full-SAE approach.

Phases:
  1. Active Feature Identification
  2. Active-Subspace Probe + Hessian
  3. Active-Subspace Robust Delta via QP/Bisection
  4. Coverage Evaluation (6-Way Comparison)
  5. Minimal Fluency Evaluation (text generation)
  6. Summary Report
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from steering import naive_delta, robust_delta, rashomon_coverage
from sae_utils import load_gemma_scope_sae
from probe import train_probe, evaluate_probe
from hessian import compute_hessian

# ═══════════════════════════════════════════════════════════════════════
# Paths and Constants
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RAW_ACT_PATH = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10.pt")
RAW_PROBE_PATH = os.path.join(OUTPUT_DIR, "baseline_probe_layer10.pt")
RAW_HESSIAN_PATH = os.path.join(OUTPUT_DIR, "hessian_layer10.pt")
RAW_RASHOMON_PATH = os.path.join(OUTPUT_DIR, "rashomon", "rashomon_probes.pt")
SAE_ACT_PATH = os.path.join(OUTPUT_DIR, "beavertails_sae_activations_layer10.pt")
SAE_PROBE_PATH = os.path.join(OUTPUT_DIR, "sae_baseline_probe_layer10.pt")
REPORT_PATH = os.path.join(OUTPUT_DIR, "active_subspace_steering_report.txt")
FLUENCY_PATH = os.path.join(OUTPUT_DIR, "fluency_samples.txt")

EPSILON = 0.15
THRESHOLD = 0.0
MAX_EXAMPLES = 50
N_FLUENCY = 5


# ═══════════════════════════════════════════════════════════════════════
# Woodbury infrastructure (for strategy D: full-SAE robust)
# ═══════════════════════════════════════════════════════════════════════

def build_woodbury_components(train_X_sae, w_sae, b_sae, weight_decay=0.01):
    """Build Woodbury H^{-1} matvec components in full SAE space."""
    X64 = train_X_sae.double()
    w64 = w_sae.double()
    b64 = b_sae.double()

    N, d = X64.shape
    ones = torch.ones(N, 1, dtype=torch.float64)
    X_aug = torch.cat([X64, ones], dim=1)

    logits = X64 @ w64 + b64
    p = torch.sigmoid(logits)
    s = p * (1 - p)

    sqrt_s_over_N = torch.sqrt(s / N).unsqueeze(1)
    Z = sqrt_s_over_N * X_aug

    G = Z @ Z.T
    eigvals_gram, U = torch.linalg.eigh(G)

    idx_desc = torch.argsort(eigvals_gram, descending=True)
    eigvals_gram = eigvals_gram[idx_desc]
    U = U[:, idx_desc]

    sigma_sq = eigvals_gram
    numerical_rank = (sigma_sq > 1e-10).sum().item()

    sigma_sq_max = sigma_sq[0].item()
    target_cond = 100.0
    ridge = sigma_sq_max / target_cond
    lam_adaptive = weight_decay + ridge

    print(f"  Woodbury: rank={numerical_rank}, "
          f"lam={lam_adaptive:.4f}, cond~{(sigma_sq_max+lam_adaptive)/lam_adaptive:.0f}")

    return Z, U, sigma_sq, numerical_rank, lam_adaptive


def woodbury_matvec(v, Z, U, sigma_sq, lam, numerical_rank):
    """Compute H^{-1} @ v via Woodbury identity."""
    z = Z @ v
    U_r = U[:, :numerical_rank]
    sigma_sq_r = sigma_sq[:numerical_rank]
    u = U_r.T @ z

    z_perp = z - U_r @ u
    inv_eigvals = 1.0 / (sigma_sq_r + lam)
    term_rank = U_r @ (inv_eigvals * u)
    term_null = (1.0 / lam) * z_perp
    inner = term_rank + term_null
    Zt_inner = Z.T @ inner

    return (1.0 / lam) * (v - Zt_inner)


def robust_delta_sae_implicit(weight, bias, x, Z, U, sigma_sq, lam,
                               numerical_rank, epsilon, threshold=0.0):
    """Robust delta in full SAE space using implicit Woodbury H^{-1}."""
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
# PHASE 1: Active Feature Identification
# ═══════════════════════════════════════════════════════════════════════

def phase1_active_features(sae_train_X):
    """Identify active SAE features across training set."""
    print("\n" + "=" * 75)
    print("PHASE 1: Active Feature Identification")
    print("=" * 75)

    N, d_sae = sae_train_X.shape
    print(f"  Training activations: {sae_train_X.shape}")

    # Boolean mask of nonzero features
    active_mask = (sae_train_X > 0)  # [N, d_sae]

    # Per-feature activation frequency (how many examples activate each feature)
    feature_freq = active_mask.sum(dim=0).numpy()  # [d_sae]

    # Per-example active feature count
    active_per_example = active_mask.sum(dim=1).numpy()  # [N]

    print(f"\n  Per-example active features:")
    print(f"    Mean:   {active_per_example.mean():.1f}")
    print(f"    Std:    {active_per_example.std():.1f}")
    print(f"    Min:    {active_per_example.min()}")
    print(f"    Max:    {active_per_example.max()}")
    print(f"    Median: {np.median(active_per_example):.0f}")

    # Active subspace sizes for different K thresholds
    results = {}
    for K in [1, 2, 3, 5, 10, 20, 50]:
        mask_k = feature_freq >= K
        d_active_k = mask_k.sum()
        results[K] = {
            "d_active": int(d_active_k),
            "mask": mask_k,
        }
        pct = 100.0 * d_active_k / d_sae
        print(f"    K>={K:>2d}: d_active = {d_active_k:>5d} ({pct:.1f}% of {d_sae})")

    # Feature frequency distribution
    print(f"\n  Feature activation frequency distribution:")
    print(f"    Never active:        {(feature_freq == 0).sum():>5d}")
    print(f"    Active 1 time:       {(feature_freq == 1).sum():>5d}")
    print(f"    Active 2-4 times:    {((feature_freq >= 2) & (feature_freq < 5)).sum():>5d}")
    print(f"    Active 5-9 times:    {((feature_freq >= 5) & (feature_freq < 10)).sum():>5d}")
    print(f"    Active 10-49 times:  {((feature_freq >= 10) & (feature_freq < 50)).sum():>5d}")
    print(f"    Active 50-99 times:  {((feature_freq >= 50) & (feature_freq < 100)).sum():>5d}")
    print(f"    Active 100+ times:   {(feature_freq >= 100).sum():>5d}")

    # Top-20 most frequently active features
    top_idx = np.argsort(feature_freq)[::-1][:20]
    print(f"\n  Top-20 most frequently active features:")
    for rank, fi in enumerate(top_idx):
        print(f"    #{rank+1:>2d}: feature {fi:>5d}  freq={feature_freq[fi]:>4d} "
              f"({100*feature_freq[fi]/N:.1f}%)")

    return results, feature_freq, active_per_example


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Active-Subspace Probe and Hessian
# ═══════════════════════════════════════════════════════════════════════

def phase2_probe_and_hessian(sae_train_X, sae_train_y, sae_test_X, sae_test_y,
                              active_indices):
    """Train probe and compute Hessian in the active subspace."""
    print("\n" + "=" * 75)
    print("PHASE 2: Active-Subspace Probe and Hessian")
    print("=" * 75)

    d_active = len(active_indices)
    print(f"  Active subspace dimension: {d_active}")

    # Extract active-subspace projections
    X_train = sae_train_X[:, active_indices].float()
    y_train = sae_train_y
    X_test = sae_test_X[:, active_indices].float()
    y_test = sae_test_y

    print(f"  X_train_active: {X_train.shape}")
    print(f"  X_test_active:  {X_test.shape}")

    # Train probe
    print(f"\n  Training active-subspace probe (d={d_active}) ...")
    probe, loss_log = train_probe(X_train, y_train, input_dim=d_active,
                                   lr=1e-3, weight_decay=0.01, epochs=50)

    # Evaluate
    print(f"\n  Evaluating on test set ...")
    metrics = evaluate_probe(probe, X_test, y_test)

    # Extract probe parameters
    w_active = probe.linear.weight.data.squeeze().double()
    b_active = probe.linear.bias.data.squeeze().double()

    # Compute Hessian with ADAPTIVE RIDGE
    # d_active (3209) > N (1600) => rank-deficient Hessian => need adaptive ridge
    # matching the Woodbury convention: target condition ~ 100
    print(f"\n  Computing Hessian ({d_active+1} x {d_active+1}) with adaptive ridge ...")
    t0 = time.time()

    X64 = X_train.double()
    N_tr = X64.shape[0]
    ones = torch.ones(N_tr, 1, dtype=torch.float64)
    X_aug = torch.cat([X64, ones], dim=1)  # [N, d_active+1]

    logits = X64 @ w_active + b_active
    p = torch.sigmoid(logits)
    s = p * (1 - p)

    sqrt_s = s.sqrt().unsqueeze(1)
    X_scaled = sqrt_s * X_aug
    H_data = (X_scaled.T @ X_scaled) / N_tr  # data term only

    # Find max eigenvalue of data term
    eigvals_data = torch.linalg.eigvalsh(H_data)
    eig_max_data = eigvals_data[-1].item()
    numerical_rank = (eigvals_data > 1e-10).sum().item()

    # Adaptive ridge: match raw-space conditioning (target cond ~ 100)
    weight_decay = 0.01
    target_cond = 100.0
    adaptive_ridge = eig_max_data / target_cond
    lam_adaptive = weight_decay + adaptive_ridge

    # Final Hessian: H = H_data + lam_adaptive * I
    H = H_data + lam_adaptive * torch.eye(d_active + 1, dtype=torch.float64)

    eigvals_final = torch.linalg.eigvalsh(H)
    eig_min = eigvals_final[0].item()
    eig_max = eigvals_final[-1].item()
    cond = eig_max / eig_min if eig_min > 0 else float("inf")

    H_inv = torch.linalg.inv(H)

    eig_info = {
        "eig_min": eig_min, "eig_max": eig_max, "condition": cond,
        "eig_max_data": eig_max_data, "numerical_rank": numerical_rank,
        "adaptive_ridge": adaptive_ridge, "lam_adaptive": lam_adaptive,
    }

    print(f"  Data term: rank={numerical_rank}/{d_active+1}, eig_max={eig_max_data:.4f}")
    print(f"  Adaptive ridge: {adaptive_ridge:.4f} (lam={lam_adaptive:.4f})")
    print(f"  Final H: eig=[{eig_min:.4e}, {eig_max:.4e}], cond={cond:.1f}")
    print(f"  Hessian computed in {time.time()-t0:.1f}s")
    print(f"  H_inv shape: {H_inv.shape}")

    return {
        "w_active": w_active,
        "b_active": b_active,
        "H_active": H,
        "H_inv_active": H_inv,
        "eig_info": eig_info,
        "metrics": metrics,
        "X_train_active": X_train,
        "X_test_active": X_test,
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3+4: Compute All Deltas and Evaluate Coverage
# ═══════════════════════════════════════════════════════════════════════

def phase34_deltas_and_coverage(artifacts, active_indices):
    """Compute 6 deltas per unsafe example, evaluate against raw Rashomon probes."""
    print("\n" + "=" * 75)
    print("PHASE 3-4: Delta Computation and 6-Way Coverage Evaluation")
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

    w_active = artifacts["w_active"]
    b_active = artifacts["b_active"]
    H_inv_active = artifacts["H_inv_active"]

    # W_dec restricted to active features: [d_active, d_model]
    W_dec_active = W_dec[active_indices, :]
    print(f"  W_dec_active: {W_dec_active.shape}")

    # Build full-SAE Woodbury for strategy D
    print("\n  Building full-SAE Woodbury components ...")
    t_wb = time.time()
    Z, U, sigma_sq, num_rank, lam = build_woodbury_components(
        sae_train_X, w_sae, b_sae
    )
    print(f"  Woodbury built in {time.time()-t_wb:.1f}s")

    # Find unsafe examples using raw-space baseline probe
    logits_raw = raw_test_X @ w_raw + b_raw
    unsafe_mask = logits_raw < THRESHOLD
    unsafe_indices = unsafe_mask.nonzero(as_tuple=True)[0]
    n_unsafe = min(len(unsafe_indices), MAX_EXAMPLES)
    print(f"\n  Unsafe test examples: {len(unsafe_indices)} total, processing {n_unsafe}")

    results = []
    t_loop = time.time()

    for i in range(n_unsafe):
        idx = unsafe_indices[i].item()
        x_raw = raw_test_X[idx]
        x_sae = sae_test_X[idx]
        x_active = sae_test_X[idx, active_indices]
        score_raw = logits_raw[idx].item()

        # (A) Naive raw
        dA, _ = naive_delta(w_raw, b_raw, x_raw, THRESHOLD)

        # (B) Robust raw
        dB = robust_delta(w_raw, b_raw, x_raw, H_inv_raw, EPSILON, THRESHOLD)

        # (C) Naive full-SAE -> raw
        dC_sae, _ = naive_delta(w_sae, b_sae, x_sae, THRESHOLD)
        dC = dC_sae @ W_dec

        # (D) Robust full-SAE -> raw (Woodbury)
        dD_sae = robust_delta_sae_implicit(
            w_sae, b_sae, x_sae, Z, U, sigma_sq, lam, num_rank,
            EPSILON, THRESHOLD
        )
        dD = dD_sae @ W_dec

        # (E_naive) Naive active-subspace -> raw
        dE_naive_active, _ = naive_delta(w_active, b_active, x_active, THRESHOLD)
        # Decode: active delta -> raw via restricted decoder columns
        dE_naive_raw = dE_naive_active @ W_dec_active

        # (E_robust) Robust active-subspace -> raw
        dE_robust_active = robust_delta(
            w_active, b_active, x_active, H_inv_active, EPSILON, THRESHOLD
        )
        dE_robust_raw = dE_robust_active @ W_dec_active

        # Evaluate all 6 against raw-space Rashomon probes
        covA = rashomon_coverage(dA, x_raw, rashomon_probes, THRESHOLD)
        covB = rashomon_coverage(dB, x_raw, rashomon_probes, THRESHOLD)
        covC = rashomon_coverage(dC, x_raw, rashomon_probes, THRESHOLD)
        covD = rashomon_coverage(dD, x_raw, rashomon_probes, THRESHOLD)
        covEn = rashomon_coverage(dE_naive_raw, x_raw, rashomon_probes, THRESHOLD)
        covEr = rashomon_coverage(dE_robust_raw, x_raw, rashomon_probes, THRESHOLD)

        results.append({
            "idx": idx, "score": score_raw,
            "norm_A": dA.norm().item(),
            "norm_B": dB.norm().item(),
            "norm_C": dC.norm().item(),
            "norm_D": dD.norm().item(),
            "norm_En": dE_naive_raw.norm().item(),
            "norm_Er": dE_robust_raw.norm().item(),
            "norm_En_active": dE_naive_active.norm().item(),
            "norm_Er_active": dE_robust_active.norm().item(),
            "cov_A": covA[0], "cov_B": covB[0], "cov_C": covC[0],
            "cov_D": covD[0], "cov_En": covEn[0], "cov_Er": covEr[0],
            "n_probes": covA[1],
            "frac_A": covA[2], "frac_B": covB[2], "frac_C": covC[2],
            "frac_D": covD[2], "frac_En": covEn[2], "frac_Er": covEr[2],
            # Store raw-space deltas for fluency evaluation
            "delta_B_raw": dB,
            "delta_Er_raw": dE_robust_raw,
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_loop
            print(f"  [{i+1}/{n_unsafe}] idx={idx:>3d}  "
                  f"cov=[A:{covA[0]}, B:{covB[0]}, C:{covC[0]}, "
                  f"D:{covD[0]}, En:{covEn[0]}, Er:{covEr[0]}]/{covA[1]}  "
                  f"({elapsed:.1f}s)")

    print(f"\n  All {n_unsafe} examples processed in {time.time()-t_loop:.1f}s")
    return results


def format_coverage_table(results):
    """Format the 6-way comparison table."""
    n = len(results)
    n_probes = results[0]["n_probes"]

    keys = ["A", "B", "C", "D", "En", "Er"]
    labels = {
        "A": "Naive Raw",
        "B": "Robust Raw",
        "C": "Naive SAE->Raw (full)",
        "D": "Robust SAE->Raw (full)",
        "En": "Naive Active-Sub SAE->Raw",
        "Er": "Robust Active-Sub SAE->Raw",
    }

    mean_norm = {k: sum(r[f"norm_{k}"] for r in results) / n for k in keys}
    mean_cov = {k: sum(r[f"frac_{k}"] for r in results) / n for k in keys}
    full_rate = {k: sum(1 for r in results if r[f"cov_{k}"] == n_probes) / n for k in keys}
    eff = {k: mean_cov[k] / mean_norm[k] if mean_norm[k] > 0 else 0.0 for k in keys}

    lines = []
    lines.append("")
    lines.append("--- 6-Way Coverage Comparison ---")
    lines.append("")

    # Header
    hdr = f"{'Metric':<28s}"
    for k in keys:
        hdr += f"  {labels[k]:>22s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    # Mean norm
    row = f"{'Mean ||delta_raw||':<28s}"
    for k in keys:
        row += f"  {mean_norm[k]:>22.4f}"
    lines.append(row)

    # Mean coverage
    row = f"{'Mean Rashomon coverage':<28s}"
    for k in keys:
        row += f"  {mean_cov[k]:>21.2%} "
    lines.append(row)

    # 100% coverage rate
    row = f"{'100% coverage rate':<28s}"
    for k in keys:
        row += f"  {full_rate[k]:>21.1%} "
    lines.append(row)

    # Coverage per unit norm
    row = f"{'Coverage per unit norm':<28s}"
    for k in keys:
        row += f"  {eff[k]:>22.4f}"
    lines.append(row)

    lines.append("")

    # Also report active-subspace norms
    mean_norm_active_n = sum(r["norm_En_active"] for r in results) / n
    mean_norm_active_r = sum(r["norm_Er_active"] for r in results) / n
    lines.append(f"  Active-subspace norms (before decode):")
    lines.append(f"    Naive:  mean ||delta_active|| = {mean_norm_active_n:.4f}")
    lines.append(f"    Robust: mean ||delta_active|| = {mean_norm_active_r:.4f}")
    lines.append(f"    Robust/naive ratio: {mean_norm_active_r/mean_norm_active_n:.2f}x"
                 if mean_norm_active_n > 0 else "    Robust/naive ratio: N/A")

    return lines, {
        "mean_norm": mean_norm, "mean_cov": mean_cov,
        "full_rate": full_rate, "eff": eff,
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: Minimal Fluency Evaluation
# ═══════════════════════════════════════════════════════════════════════

def phase5_fluency(results, raw_test_y):
    """Load Gemma-2 2B and generate steered text for 5 examples."""
    print("\n" + "=" * 75)
    print("PHASE 5: Minimal Fluency Evaluation")
    print("=" * 75)

    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Re-load BeaverTails to get original prompts ──
    print("  Loading BeaverTails to recover prompts ...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")

    rng = np.random.RandomState(42)
    safe_idx = [i for i, ex in enumerate(ds) if ex["is_safe"]]
    unsafe_idx = [i for i, ex in enumerate(ds) if not ex["is_safe"]]
    safe_sample = rng.choice(safe_idx, size=min(1000, len(safe_idx)), replace=False)
    unsafe_sample = rng.choice(unsafe_idx, size=min(1000, len(unsafe_idx)), replace=False)
    all_indices = np.concatenate([safe_sample, unsafe_sample])
    all_labels = [1] * len(safe_sample) + [0] * len(unsafe_sample)

    # Reproduce exact train/test split
    idx_range = list(range(len(all_indices)))
    train_idx, test_idx = train_test_split(
        idx_range, test_size=0.2, random_state=42, stratify=all_labels
    )

    # Map test position -> dataset entry
    test_ds_indices = [int(all_indices[i]) for i in test_idx]
    print(f"  Recovered {len(test_ds_indices)} test indices")

    # ── Select 5 representative unsafe examples ──
    # Pick from the computed results, spanning the score range
    scores = [r["score"] for r in results]
    sorted_results = sorted(enumerate(results), key=lambda x: x[1]["score"])

    # Pick at 5 quantiles: 0%, 25%, 50%, 75%, 100% of the unsafe score range
    n = len(sorted_results)
    pick_positions = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    selected = []
    for pos in pick_positions:
        orig_i, r = sorted_results[pos]
        selected.append((orig_i, r))

    print(f"\n  Selected {len(selected)} examples for fluency eval:")
    for orig_i, r in selected:
        print(f"    results[{orig_i}] test_idx={r['idx']} score={r['score']:.4f}")

    # ── Load model ──
    print("\n  Loading Gemma-2 2B for generation ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  Model loaded on {device}")

    # ── Generate text with steering ──
    def make_hook(delta_raw):
        """Forward hook that adds delta to layer 10 residual stream."""
        delta_dev = delta_raw.to(device=model.device, dtype=torch.bfloat16)
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                return output + delta_dev
            # Handle tuple, ModelOutput, or named tuple
            hs = output[0] + delta_dev
            rest = tuple(output[i] for i in range(1, len(output)))
            return (hs,) + rest
        return hook_fn

    def generate_with_delta(prompt_text, delta_raw=None, max_new_tokens=128):
        """Generate text, optionally with steering delta at layer 10."""
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        handle = None
        if delta_raw is not None:
            handle = model.model.layers[10].register_forward_hook(
                make_hook(delta_raw)
            )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,
            )
        if handle is not None:
            handle.remove()
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip the input prompt from the output
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        if text.startswith(input_text):
            text = text[len(input_text):]
        return text.strip()

    samples = []
    for si, (orig_i, r) in enumerate(selected):
        test_pos = r["idx"]  # position in the test set
        ds_idx = test_ds_indices[test_pos]
        entry = ds[ds_idx]
        prompt = entry["prompt"]
        original_response = entry["response"][:200]  # truncate for display

        print(f"\n  --- Example {si+1}/{len(selected)} (test_idx={test_pos}, "
              f"score={r['score']:.4f}) ---")
        print(f"  Prompt: {prompt[:100]}...")

        # Plain text prompt for base model (no chat template)
        formatted = f"Question: {prompt}\n\nAnswer:"

        delta_B = r["delta_B_raw"].float()
        delta_Er = r["delta_Er_raw"].float()

        # Baseline (no steering)
        print("  Generating baseline ...")
        gen_baseline = generate_with_delta(formatted, delta_raw=None)

        # Robust raw steering (strategy B)
        print("  Generating with robust-raw steering ...")
        gen_robust_raw = generate_with_delta(formatted, delta_raw=delta_B)

        # Active-subspace SAE steering (strategy E_robust)
        print("  Generating with active-sub SAE steering ...")
        gen_active_sae = generate_with_delta(formatted, delta_raw=delta_Er)

        samples.append({
            "test_idx": test_pos,
            "score": r["score"],
            "prompt": prompt,
            "original_response": original_response,
            "norm_B": r["norm_B"],
            "norm_Er": r["norm_Er"],
            "cov_B": r["cov_B"],
            "cov_Er": r["cov_Er"],
            "gen_baseline": gen_baseline,
            "gen_robust_raw": gen_robust_raw,
            "gen_active_sae": gen_active_sae,
        })

    # ── Save fluency samples ──
    lines = []
    lines.append("=" * 80)
    lines.append("FLUENCY EVALUATION: Steered Text Generation Samples")
    lines.append("=" * 80)
    lines.append(f"Model: Gemma-2 2B | Layer: 10 | Temperature: 0.7 | Max tokens: 128")
    lines.append(f"Conditions: baseline (no delta), robust-raw (B), active-sub SAE (E_robust)")
    lines.append("")

    for si, s in enumerate(samples):
        lines.append("-" * 80)
        lines.append(f"EXAMPLE {si+1}: test_idx={s['test_idx']}, "
                     f"probe_score={s['score']:.4f}")
        lines.append(f"  ||delta_robust_raw|| = {s['norm_B']:.4f}, "
                     f"coverage = {s['cov_B']}/50")
        lines.append(f"  ||delta_active_SAE|| = {s['norm_Er']:.4f}, "
                     f"coverage = {s['cov_Er']}/50")
        lines.append(f"\nPROMPT: {s['prompt']}")
        lines.append(f"\nORIGINAL RESPONSE (truncated): {s['original_response']}")
        lines.append(f"\n[BASELINE - no steering]")
        lines.append(s["gen_baseline"])
        lines.append(f"\n[ROBUST RAW steering (B)]")
        lines.append(s["gen_robust_raw"])
        lines.append(f"\n[ACTIVE-SUB SAE steering (E_robust)]")
        lines.append(s["gen_active_sae"])
        lines.append("")

    fluency_text = "\n".join(lines)
    with open(FLUENCY_PATH, "w") as f:
        f.write(fluency_text)
    print(f"\n  Fluency samples saved to {FLUENCY_PATH}")

    # Cleanup model to free GPU
    del model
    torch.cuda.empty_cache()

    return samples


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: Summary Report
# ═══════════════════════════════════════════════════════════════════════

def phase6_report(phase1_results, feature_freq, active_per_example,
                  active_indices, K_used, phase2_results, coverage_results,
                  coverage_summary, fluency_samples=None):
    """Generate comprehensive summary report."""
    print("\n" + "=" * 75)
    print("PHASE 6: Summary Report")
    print("=" * 75)

    d_active = len(active_indices)
    N_train = len(active_per_example)

    lines = []
    lines.append("=" * 95)
    lines.append("ACTIVE-SUBSPACE SAE STEERING: COMPREHENSIVE REPORT")
    lines.append("=" * 95)
    lines.append(f"\nRashomon epsilon: {EPSILON}")
    lines.append(f"Logit threshold:  {THRESHOLD}")
    lines.append(f"Active subspace K threshold: {K_used}")
    lines.append(f"Active subspace dimension: {d_active}")
    lines.append(f"Training examples: {N_train}")
    lines.append(f"Unsafe test examples evaluated: {len(coverage_results)}")
    lines.append("")

    # ── Section 1: Active Subspace Characterization ──
    lines.append("=" * 95)
    lines.append("1. ACTIVE SUBSPACE CHARACTERIZATION")
    lines.append("=" * 95)
    lines.append(f"\n  d_active (K={K_used}): {d_active} out of 16384 "
                 f"({100*d_active/16384:.1f}%)")
    lines.append(f"\n  Active features per example:")
    lines.append(f"    Mean:   {active_per_example.mean():.1f}")
    lines.append(f"    Std:    {active_per_example.std():.1f}")
    lines.append(f"    Min/Max: {active_per_example.min()}/{active_per_example.max()}")

    lines.append(f"\n  Active subspace sizes for different K:")
    for K, info in sorted(phase1_results.items()):
        lines.append(f"    K>={K:>2d}: d_active = {info['d_active']:>5d} "
                     f"({100*info['d_active']/16384:.1f}%)")

    lines.append(f"\n  Feature frequency distribution:")
    lines.append(f"    Never active:      {(feature_freq == 0).sum():>5d}")
    lines.append(f"    Active 1 time:     {(feature_freq == 1).sum():>5d}")
    lines.append(f"    Active 2-4x:       {((feature_freq >= 2) & (feature_freq < 5)).sum():>5d}")
    lines.append(f"    Active 5-49x:      {((feature_freq >= 5) & (feature_freq < 50)).sum():>5d}")
    lines.append(f"    Active 50+x:       {(feature_freq >= 50).sum():>5d}")
    lines.append("")

    # ── Section 2: Active-Subspace Probe Performance ──
    lines.append("=" * 95)
    lines.append("2. ACTIVE-SUBSPACE PROBE PERFORMANCE")
    lines.append("=" * 95)
    m = phase2_results["metrics"]
    lines.append(f"\n  Accuracy:  {m['accuracy']:.4f}")
    lines.append(f"  F1:        {m['f1']:.4f}")
    lines.append(f"  AUROC:     {m['auroc']:.4f}")
    lines.append(f"  Precision: {m['precision']:.4f}")
    lines.append(f"  Recall:    {m['recall']:.4f}")

    lines.append(f"\n  Comparison with prior probes:")
    lines.append(f"    {'Probe':<25s}  {'Dim':>6s}  {'Acc':>8s}  {'F1':>8s}  {'AUROC':>8s}")
    lines.append(f"    {'-'*25}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    lines.append(f"    {'Raw-space baseline':<25s}  {'2304':>6s}  {'0.7550':>8s}  {'---':>8s}  {'---':>8s}")
    lines.append(f"    {'Full SAE-space baseline':<25s}  {'16384':>6s}  {'0.6950':>8s}  {'---':>8s}  {'---':>8s}")
    lines.append(f"    {'Active-subspace (K={K_used})':<25s}  {d_active:>6d}  {m['accuracy']:>8.4f}  {m['f1']:>8.4f}  {m['auroc']:>8.4f}")
    lines.append("")

    # ── Section 3: Hessian Comparison ──
    lines.append("=" * 95)
    lines.append("3. HESSIAN COMPARISON")
    lines.append("=" * 95)
    eig = phase2_results["eig_info"]
    lines.append(f"\n  {'Property':<30s}  {'Raw':>15s}  {'Full SAE':>15s}  {'Active-Sub':>15s}")
    lines.append(f"  {'-'*30}  {'-'*15}  {'-'*15}  {'-'*15}")
    lines.append(f"  {'Hessian dimension':<30s}  {'2,305':>15s}  {'16,385':>15s}  {d_active+1:>15,d}")
    lines.append(f"  {'Eigenvalue min':<30s}  {'~16.2':>15s}  {'~1e-10':>15s}  {eig['eig_min']:>15.4e}")
    lines.append(f"  {'Eigenvalue max':<30s}  {'~1639.7':>15s}  {'~467.9':>15s}  {eig['eig_max']:>15.4e}")
    lines.append(f"  {'Condition number':<30s}  {'~101':>15s}  {'~4.68e4':>15s}  {eig['condition']:>15.2f}")
    lines.append(f"  {'Inversion method':<30s}  {'Dense':>15s}  {'Woodbury':>15s}  {'Dense':>15s}")
    lines.append("")

    # ── Section 4: 6-Way Coverage Comparison ──
    lines.append("=" * 95)
    lines.append("4. SIX-WAY COVERAGE COMPARISON")
    lines.append("=" * 95)

    table_lines, summary = coverage_summary
    for tl in table_lines:
        lines.append(tl)

    # Per-example table (first 10)
    lines.append("\n  Per-example details (first 10):")
    hdr = (f"  {'#':>3s}  {'Score':>7s}  "
           f"{'A':>5s}  {'B':>5s}  {'C':>5s}  {'D':>5s}  {'En':>5s}  {'Er':>5s}  "
           f"{'||B||':>7s} {'||Er||':>7s}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    n_probes = coverage_results[0]["n_probes"]
    for r in coverage_results[:10]:
        lines.append(
            f"  {r['idx']:>3d}  {r['score']:>7.3f}  "
            f"{r['cov_A']:>3d}/{n_probes:<1d}  {r['cov_B']:>3d}/{n_probes:<1d}  "
            f"{r['cov_C']:>3d}/{n_probes:<1d}  {r['cov_D']:>3d}/{n_probes:<1d}  "
            f"{r['cov_En']:>3d}/{n_probes:<1d}  {r['cov_Er']:>3d}/{n_probes:<1d}  "
            f"{r['norm_B']:>7.3f} {r['norm_Er']:>7.3f}"
        )
    lines.append("")

    # ── Section 5: Fluency Samples ──
    lines.append("=" * 95)
    lines.append("5. FLUENCY EVALUATION")
    lines.append("=" * 95)
    if fluency_samples:
        lines.append(f"\n  {len(fluency_samples)} examples evaluated. "
                     f"Full samples in: outputs/fluency_samples.txt")
        for si, s in enumerate(fluency_samples):
            lines.append(f"\n  Example {si+1}: test_idx={s['test_idx']}, "
                         f"score={s['score']:.4f}")
            lines.append(f"    Prompt: {s['prompt'][:80]}...")
            baseline_preview = s['gen_baseline'][:120].replace('\n', ' ')
            robust_preview = s['gen_robust_raw'][:120].replace('\n', ' ')
            sae_preview = s['gen_active_sae'][:120].replace('\n', ' ')
            lines.append(f"    Baseline:    {baseline_preview}...")
            lines.append(f"    Robust-raw:  {robust_preview}...")
            lines.append(f"    Active-SAE:  {sae_preview}...")
    else:
        lines.append("\n  [Phase 5 skipped or failed — no fluency data available]")
    lines.append("")

    # ── Section 6: Scientific Verdict ──
    lines.append("=" * 95)
    lines.append("6. SCIENTIFIC VERDICT")
    lines.append("=" * 95)

    s = summary
    mn = s["mean_norm"]
    mc = s["mean_cov"]
    fr = s["full_rate"]

    lines.append("")
    lines.append("  QUESTION: Does active-subspace SAE steering materially improve")
    lines.append("  upon the prior full-SAE result (13.56% coverage)?")
    lines.append("")

    # Compare E_robust vs D
    improvement_over_full_sae = mc["Er"] - mc["D"]
    lines.append(f"  Active-sub robust (E) coverage:  {mc['Er']:.2%}")
    lines.append(f"  Full-SAE robust (D) coverage:    {mc['D']:.2%}")
    lines.append(f"  Improvement over full-SAE:       {improvement_over_full_sae:+.2%}")
    lines.append("")

    # Compare E_robust vs B
    gap_to_raw = mc["B"] - mc["Er"]
    lines.append(f"  Raw-space robust (B) coverage:   {mc['B']:.2%}")
    lines.append(f"  Gap to raw-space robust:         {gap_to_raw:+.2%}")
    lines.append("")

    # Norm comparison
    lines.append(f"  Norm comparison (mean ||delta_raw||):")
    lines.append(f"    Robust raw (B):         {mn['B']:.4f}")
    lines.append(f"    Active-sub robust (Er): {mn['Er']:.4f}")
    if mn["B"] > 0:
        lines.append(f"    Ratio Er/B:             {mn['Er']/mn['B']:.2f}x")
    lines.append("")

    # Verdict
    if mc["Er"] > 0.80:
        verdict = "STRONG SUCCESS"
        detail = ("Active-subspace SAE steering achieves high Rashomon coverage, "
                  "validating the hypothesis that restricting to active features "
                  "avoids the null-space problem.")
    elif mc["Er"] > mc["D"] + 0.10:
        verdict = "PARTIAL SUCCESS"
        detail = ("Active-subspace SAE steering materially improves over full-SAE "
                  f"({mc['Er']:.1%} vs {mc['D']:.1%}) but does not reach raw-space "
                  f"robust levels ({mc['B']:.1%}).")
    elif mc["Er"] > mc["D"] + 0.02:
        verdict = "MARGINAL IMPROVEMENT"
        detail = ("Active-subspace SAE steering shows modest improvement over full-SAE "
                  "but remains far from raw-space robust steering.")
    else:
        verdict = "NO MATERIAL IMPROVEMENT"
        detail = ("Active-subspace SAE steering does not materially improve over "
                  "full-SAE. The coverage collapse appears rooted in a deeper issue "
                  "than the null-space dimensionality alone.")

    lines.append(f"  VERDICT: {verdict}")
    lines.append(f"  {detail}")
    lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n  Report saved to {REPORT_PATH}")
    print(f"\n{'='*75}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*75}")

    return report


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fluency", action="store_true",
                        help="Skip Phase 5 (text generation)")
    parser.add_argument("--K", type=int, default=None,
                        help="Override K threshold for active features")
    args = parser.parse_args()

    t_start = time.time()

    # ── Load all artifacts ──
    print("=" * 75)
    print("LOADING ARTIFACTS")
    print("=" * 75)

    raw_data = torch.load(RAW_ACT_PATH, map_location="cpu", weights_only=True)
    raw_probe = torch.load(RAW_PROBE_PATH, map_location="cpu", weights_only=True)
    raw_hessian = torch.load(RAW_HESSIAN_PATH, map_location="cpu", weights_only=True)
    rashomon_probes = torch.load(RAW_RASHOMON_PATH, map_location="cpu", weights_only=False)
    sae_data = torch.load(SAE_ACT_PATH, map_location="cpu", weights_only=True)
    sae_probe_data = torch.load(SAE_PROBE_PATH, map_location="cpu", weights_only=True)

    print("  Loading SAE decoder (Gemma Scope) ...")
    sae = load_gemma_scope_sae(layer=10, width="16k", l0=77)
    W_dec = sae.W_dec.data.double()
    del sae  # free memory

    sae_train_X = sae_data["train_X"]
    sae_train_y = sae_data["train_y"]
    sae_test_X = sae_data["test_X"]
    sae_test_y = sae_data["test_y"]

    print(f"  Raw:  train={raw_data['train_X'].shape}, test={raw_data['test_X'].shape}")
    print(f"  SAE:  train={sae_train_X.shape}, test={sae_test_X.shape}")
    print(f"  W_dec: {W_dec.shape}")
    print(f"  Rashomon probes: {len(rashomon_probes)}")

    # ── Phase 1 ──
    phase1_results, feature_freq, active_per_example = phase1_active_features(sae_train_X)

    # Choose K: prefer K=1, but fall back if Hessian would be too large
    if args.K is not None:
        K_used = args.K
    else:
        d_k1 = phase1_results[1]["d_active"]
        if d_k1 <= 5000:
            K_used = 1
        elif phase1_results[5]["d_active"] <= 5000:
            K_used = 5
        else:
            K_used = 10
        print(f"\n  Auto-selected K={K_used} (d_active={phase1_results[K_used]['d_active']})")

    active_mask_bool = phase1_results[K_used]["mask"]
    active_indices = torch.from_numpy(
        np.where(active_mask_bool)[0]
    ).long()
    d_active = len(active_indices)
    print(f"  Using K={K_used}: d_active={d_active}")

    # ── Phase 2 ──
    phase2_results = phase2_probe_and_hessian(
        sae_train_X, sae_train_y, sae_test_X, sae_test_y, active_indices
    )

    # Merge into artifacts dict for Phase 3-4
    artifacts = {
        "w_raw": raw_probe["weight"].double(),
        "b_raw": raw_probe["bias"].double(),
        "H_inv_raw": raw_hessian["H_inv"],
        "rashomon_probes": rashomon_probes,
        "raw_test_X": raw_data["test_X"],
        "raw_test_y": raw_data["test_y"],
        "sae_test_X": sae_test_X,
        "sae_train_X": sae_train_X,
        "w_sae": sae_probe_data["weight"].double(),
        "b_sae": sae_probe_data["bias"].double(),
        "W_dec": W_dec,
        "w_active": phase2_results["w_active"],
        "b_active": phase2_results["b_active"],
        "H_inv_active": phase2_results["H_inv_active"],
    }

    # ── Phase 3+4 ──
    coverage_results = phase34_deltas_and_coverage(artifacts, active_indices)
    table_lines, coverage_summary_dict = format_coverage_table(coverage_results)

    # Print the coverage table immediately
    print("\n" + "=" * 75)
    print("COVERAGE RESULTS")
    print("=" * 75)
    for line in table_lines:
        print(line)

    # ── Phase 5 ──
    fluency_samples = None
    if not args.skip_fluency:
        try:
            fluency_samples = phase5_fluency(coverage_results, raw_data["test_y"])
        except Exception as e:
            print(f"\n  Phase 5 FAILED: {e}")
            print("  Continuing without fluency data ...")
            import traceback
            traceback.print_exc()
    else:
        print("\n  Phase 5 skipped (--skip-fluency)")

    # ── Phase 6 ──
    phase6_report(
        phase1_results, feature_freq, active_per_example,
        active_indices, K_used, phase2_results, coverage_results,
        (table_lines, coverage_summary_dict), fluency_samples
    )

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("ACTIVE-SUBSPACE SAE STEERING PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
