"""SAE Feature Decomposition of Raw-Space Robust Deltas.

Decomposes raw-space robust deltas into SAE feature space via:
  (1) Minimum-L2 (pseudoinverse): α_pinv = pinv(W_dec^T) @ δ
  (2) Minimum-L1 (basis pursuit via FISTA): min ||α||_1 s.t. α @ W_dec ≈ δ
  (3) Encoder-based (comparison): Δz = encode(x + δ) - encode(x)

Mathematical setup:
  W_dec (code): [d_sae, d_model] = [16384, 2304]
  decode: z @ W_dec = x_hat  (row-vector convention)
  Decomposition: find α ∈ R^16384 s.t. α @ W_dec = δ_robust ∈ R^2304
                  equivalently: W_dec^T @ α = δ   (W_dec^T is [2304, 16384])
  W_dec^T has full row rank 2304, so solution space is a
  14080-dimensional affine subspace.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from steering import naive_delta, robust_delta, rashomon_coverage
from sae_utils import load_gemma_scope_sae

# ═══════════════════════════════════════════════════════════════════════
# Paths and constants
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RAW_ACT_PATH = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10.pt")
RAW_PROBE_PATH = os.path.join(OUTPUT_DIR, "baseline_probe_layer10.pt")
RAW_HESSIAN_PATH = os.path.join(OUTPUT_DIR, "hessian_layer10.pt")
RAW_RASHOMON_PATH = os.path.join(OUTPUT_DIR, "rashomon", "rashomon_probes.pt")
SAE_ACT_PATH = os.path.join(OUTPUT_DIR, "beavertails_sae_activations_layer10.pt")
SAE_PROBE_PATH = os.path.join(OUTPUT_DIR, "sae_baseline_probe_layer10.pt")
REPORT_PATH = os.path.join(OUTPUT_DIR, "sae_feature_decomposition_report.txt")

EPSILON = 0.15
THRESHOLD = 0.0
MAX_EXAMPLES = 50
L1_MAX_EXAMPLES = 10


# ═══════════════════════════════════════════════════════════════════════
# Woodbury infrastructure (for SAE-space robust delta recomputation)
# ═══════════════════════════════════════════════════════════════════════

def build_woodbury_components(train_X_sae, w_sae, b_sae, weight_decay=0.01):
    """Build Woodbury H^{-1} components for SAE space."""
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
# FISTA for L1-minimization
# ═══════════════════════════════════════════════════════════════════════

def fista_l1(A, b, lam, L, x0=None, max_iter=2000, conv_tol=1e-7):
    """Solve min 0.5*||Ax - b||^2 + lam*||x||_1 via FISTA.

    Args:
        A: [m, n] matrix (W_dec^T, shape [2304, 16384])
        b: [m] vector (delta, shape [2304])
        lam: L1 regularization strength
        L: Lipschitz constant (spectral_norm(A)^2)
        x0: optional warm start [n]
        max_iter: maximum iterations
        conv_tol: relative change convergence tolerance

    Returns:
        x: [n] solution vector
    """
    n = A.shape[1]
    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype)
    y = x.clone()
    t = 1.0
    step = 1.0 / L
    threshold = lam * step

    for k in range(max_iter):
        # Gradient of smooth part at y
        grad = A.T @ (A @ y - b)

        # Proximal gradient step (soft thresholding)
        z = y - step * grad
        x_new = torch.sign(z) * torch.clamp(torch.abs(z) - threshold, min=0)

        # Check convergence every 100 iterations
        if k > 0 and k % 100 == 0:
            rel_change = (x_new - x).norm() / (x.norm() + 1e-15)
            if rel_change < conv_tol:
                return x_new

        # FISTA momentum
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return x


def find_sparse_decomposition(A, b, L, tol_rel=1e-3, max_bisect=20, max_fista_iter=2000):
    """Find sparsest α satisfying ||A @ α - b|| / ||b|| < tol_rel.

    Uses binary search over FISTA regularization parameter lambda.
    As lambda increases: solution becomes sparser, reconstruction error increases.
    We find the LARGEST lambda achieving error < tol_rel.

    Args:
        A: [m, n] = W_dec^T
        b: [m] = delta
        L: Lipschitz constant
        tol_rel: relative reconstruction error tolerance

    Returns:
        (alpha, rel_err, lam_used)
    """
    b_norm = b.norm().item()

    # Lambda_max: above this, optimal solution is x=0
    lam_max = (A.T @ b).abs().max().item()

    # Binary search in log space
    log_lo = np.log10(lam_max) - 8   # Very small lambda -> dense -> low error
    log_hi = np.log10(lam_max)       # lambda_max -> all zeros -> high error

    best_alpha = None
    best_err = float('inf')
    best_lam = None

    for i in range(max_bisect):
        log_mid = (log_lo + log_hi) / 2
        lam = 10 ** log_mid

        alpha = fista_l1(A, b, lam, L, x0=best_alpha, max_iter=max_fista_iter)
        rel_err = (A @ alpha - b).norm().item() / b_norm

        if rel_err < tol_rel:
            best_alpha = alpha.clone()
            best_err = rel_err
            best_lam = lam
            log_lo = log_mid   # Try larger lambda (sparser)
        else:
            log_hi = log_mid   # Need smaller lambda (denser)

    if best_alpha is None:
        # Fallback: use very small lambda
        lam = 10 ** (np.log10(lam_max) - 8)
        best_alpha = fista_l1(A, b, lam, L, max_iter=max_fista_iter * 2)
        best_err = (A @ best_alpha - b).norm().item() / b_norm
        best_lam = lam

    return best_alpha, best_err, best_lam


# ═══════════════════════════════════════════════════════════════════════
# Sparsity metrics
# ═══════════════════════════════════════════════════════════════════════

def gini_coefficient(x):
    """Gini coefficient of |x|. 0 = uniform, 1 = maximally concentrated."""
    abs_x = np.sort(np.abs(x))
    n = len(abs_x)
    total = abs_x.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * abs_x) / (n * total)) - (n + 1) / n


def count_significant(x, rel_threshold=1e-3):
    """Count components with |x_j| > rel_threshold * max|x|."""
    abs_x = np.abs(x)
    max_val = abs_x.max()
    if max_val == 0:
        return 0
    return int(np.sum(abs_x > rel_threshold * max_val))


def l1_l2_ratio(x):
    """L1/L2 ratio. Lower = sparser (minimum is 1 for a single nonzero)."""
    l1 = np.abs(x).sum()
    l2 = np.sqrt(np.sum(x ** 2))
    if l2 == 0:
        return 0.0
    return float(l1 / l2)


ENERGY_KS = [1, 5, 10, 20, 50, 100, 200, 500, 1000]


def energy_concentration(x, ks=None):
    """Fraction of ||x||^2 captured by top-k components."""
    if ks is None:
        ks = ENERGY_KS
    x_sq = x ** 2
    sorted_sq = np.sort(x_sq)[::-1]
    total = sorted_sq.sum()
    if total == 0:
        return {k: 0.0 for k in ks}
    cumsum = np.cumsum(sorted_sq)
    return {k: float(cumsum[min(k, len(x)) - 1] / total) for k in ks}


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Load artifacts
# ═══════════════════════════════════════════════════════════════════════

def phase1_load():
    """Load all required artifacts."""
    print("=" * 75)
    print("PHASE 1: Load All Artifacts")
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

    sae_data = torch.load(SAE_ACT_PATH, map_location="cpu", weights_only=True)
    sae_train_X = sae_data["train_X"]
    sae_test_X = sae_data["test_X"]

    sae_probe = torch.load(SAE_PROBE_PATH, map_location="cpu", weights_only=True)
    w_sae = sae_probe["weight"].double()
    b_sae = sae_probe["bias"].double()

    print(f"  SAE activations: train={sae_train_X.shape}, test={sae_test_X.shape}")
    print(f"  SAE probe: w={w_sae.shape}, b={b_sae.shape}")

    print("\n  Loading SAE model (Gemma Scope) ...")
    sae = load_gemma_scope_sae(layer=10, width="16k", l0=77)
    W_dec = sae.W_dec.data.double()  # [16384, 2304]
    print(f"  W_dec: {W_dec.shape}")

    return {
        "raw_test_X": raw_test_X,
        "w_raw": w_raw, "b_raw": b_raw,
        "H_inv_raw": H_inv_raw,
        "rashomon_probes": rashomon_probes,
        "sae_train_X": sae_train_X, "sae_test_X": sae_test_X,
        "w_sae": w_sae, "b_sae": b_sae,
        "sae": sae, "W_dec": W_dec,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Compute raw-space robust deltas
# ═══════════════════════════════════════════════════════════════════════

def phase2_compute_raw_deltas(artifacts):
    """Compute raw-space robust deltas for all unsafe test examples."""
    print("\n" + "=" * 75)
    print("PHASE 2: Compute Raw-Space Robust Deltas")
    print("=" * 75)

    w_raw = artifacts["w_raw"]
    b_raw = artifacts["b_raw"]
    H_inv_raw = artifacts["H_inv_raw"]
    raw_test_X = artifacts["raw_test_X"].double()

    logits_raw = raw_test_X @ w_raw + b_raw
    unsafe_mask = logits_raw < THRESHOLD
    unsafe_indices = unsafe_mask.nonzero(as_tuple=True)[0]
    n_unsafe = min(len(unsafe_indices), MAX_EXAMPLES)
    print(f"  Unsafe test examples: {len(unsafe_indices)}, processing {n_unsafe}")

    deltas = []
    x_raws = []
    example_indices = []

    t0 = time.time()
    for i in range(n_unsafe):
        idx = unsafe_indices[i].item()
        x_raw = raw_test_X[idx]
        d_robust = robust_delta(w_raw, b_raw, x_raw, H_inv_raw, EPSILON, THRESHOLD)

        deltas.append(d_robust)
        x_raws.append(x_raw)
        example_indices.append(idx)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_unsafe}] idx={idx} ||δ||={d_robust.norm().item():.4f} "
                  f"({time.time()-t0:.1f}s)")

    print(f"  All {n_unsafe} deltas computed in {time.time()-t0:.1f}s")
    return deltas, x_raws, example_indices


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Three decompositions
# ═══════════════════════════════════════════════════════════════════════

def phase3_decompositions(artifacts, deltas, x_raws):
    """Compute all three decompositions of raw-space robust deltas."""
    print("\n" + "=" * 75)
    print("PHASE 3: SAE Feature Decompositions")
    print("=" * 75)

    W_dec = artifacts["W_dec"]   # [16384, 2304]
    sae = artifacts["sae"]
    A = W_dec.T                  # [2304, 16384] — system matrix

    # ── 3a: Pseudoinverse decomposition (all examples) ────────────────
    print("\n--- 3a: Minimum-L2 (Pseudoinverse) Decomposition ---")
    t0 = time.time()

    # pinv(A) where A = W_dec^T [2304, 16384], full row rank 2304.
    # pinv(A) = A^T (A A^T)^{-1} = W_dec (W_dec^T W_dec)^{-1}
    # Shape: [16384, 2304]
    G = W_dec.T @ W_dec          # [2304, 2304]
    G_inv = torch.linalg.inv(G)
    P_pinv = W_dec @ G_inv       # [16384, 2304]
    print(f"  Pseudoinverse precomputed: P_pinv {P_pinv.shape} ({time.time()-t0:.1f}s)")

    alphas_pinv = []
    recon_errors_pinv = []

    for delta in deltas:
        alpha = P_pinv @ delta           # [16384]
        recon = alpha @ W_dec            # [2304] — should equal delta
        rel_err = (recon - delta).norm().item() / delta.norm().item()
        alphas_pinv.append(alpha)
        recon_errors_pinv.append(rel_err)

    max_err = max(recon_errors_pinv)
    mean_err = sum(recon_errors_pinv) / len(recon_errors_pinv)
    print(f"  Reconstruction error: mean={mean_err:.2e}, max={max_err:.2e}")
    assert max_err < 1e-6, f"Pseudoinverse reconstruction failed: max error {max_err:.2e}"
    print(f"  PASS: all {len(deltas)} examples verified (error < 1e-6)")

    # ── 3b: Minimum-L1 (FISTA) Decomposition ─────────────────────────
    n_l1 = min(L1_MAX_EXAMPLES, len(deltas))
    print(f"\n--- 3b: Minimum-L1 (Basis Pursuit via FISTA) [{n_l1} examples] ---")

    t0 = time.time()
    L_lip = torch.linalg.norm(A, ord=2).item() ** 2
    print(f"  Lipschitz constant L = {L_lip:.4f} (σ_max² of W_dec^T)")

    alphas_sparse = []
    recon_errors_sparse = []
    lam_values = []

    for i in range(n_l1):
        t_ex = time.time()
        alpha, rel_err, lam = find_sparse_decomposition(
            A, deltas[i], L_lip, tol_rel=1e-3, max_bisect=20, max_fista_iter=2000
        )
        elapsed = time.time() - t_ex
        n_sig = count_significant(alpha.numpy())
        alphas_sparse.append(alpha)
        recon_errors_sparse.append(rel_err)
        lam_values.append(lam)
        print(f"  [{i+1}/{n_l1}] rel_err={rel_err:.2e}, λ={lam:.2e}, "
              f"sig_features={n_sig}, ||α||_1={alpha.abs().sum().item():.2f} ({elapsed:.1f}s)")

    total_l1_time = time.time() - t0
    print(f"  L1 total: {total_l1_time:.1f}s ({total_l1_time / n_l1:.1f}s/example)")

    max_err_l1 = max(recon_errors_sparse)
    if max_err_l1 < 1e-3:
        print(f"  PASS: all {n_l1} examples verified (error < 1e-3)")
    else:
        print(f"  WARNING: max reconstruction error = {max_err_l1:.2e}")

    # ── 3c: Encoder-based Decomposition (all examples) ────────────────
    print(f"\n--- 3c: Encoder-Based Decomposition [{len(deltas)} examples] ---")
    t0 = time.time()

    delta_z_list = []
    with torch.no_grad():
        for x_raw, delta in zip(x_raws, deltas):
            x_f = x_raw.float()
            x_steered_f = (x_raw + delta).float()
            z_orig = sae.encode(x_f.unsqueeze(0)).squeeze(0)
            z_steer = sae.encode(x_steered_f.unsqueeze(0)).squeeze(0)
            delta_z_list.append((z_steer - z_orig).double())

    # Encoder is NOT an exact decomposition — check reconstruction
    recon_errors_enc = []
    for dz, delta in zip(delta_z_list, deltas):
        recon = dz @ W_dec
        rel_err = (recon - delta).norm().item() / delta.norm().item()
        recon_errors_enc.append(rel_err)

    mean_enc_err = sum(recon_errors_enc) / len(recon_errors_enc)
    max_enc_err = max(recon_errors_enc)
    print(f"  Encoder reconstruction: mean_rel_err={mean_enc_err:.4f}, max={max_enc_err:.4f}")
    print(f"  (Non-zero expected: encoder is nonlinear, not algebraic inverse)")
    print(f"  Done in {time.time()-t0:.1f}s")

    return alphas_pinv, alphas_sparse, delta_z_list, {
        "recon_pinv": recon_errors_pinv,
        "recon_sparse": recon_errors_sparse,
        "recon_enc": recon_errors_enc,
        "l1_time": total_l1_time,
        "l1_lam_values": lam_values,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Sparsity Analysis
# ═══════════════════════════════════════════════════════════════════════

def phase4_sparsity(alphas_pinv, alphas_sparse, delta_z_list):
    """Three-way sparsity comparison."""
    print("\n" + "=" * 75)
    print("PHASE 4: Sparsity Analysis")
    print("=" * 75)

    methods = {
        "pinv": ("α_pinv (min-L2)", [a.numpy() for a in alphas_pinv]),
        "sparse": ("α_sparse (min-L1)", [a.numpy() for a in alphas_sparse]),
        "encoder": ("Δz (encoder)", [a.numpy() for a in delta_z_list]),
    }

    results = {}

    for key, (name, alphas) in methods.items():
        n = len(alphas)
        sig_counts = [count_significant(a) for a in alphas]
        ratios = [l1_l2_ratio(a) for a in alphas]
        ginis = [gini_coefficient(a) for a in alphas]
        energy_curves = [energy_concentration(a) for a in alphas]

        mean_energy = {}
        for k in ENERGY_KS:
            mean_energy[k] = float(np.mean([ec[k] for ec in energy_curves]))

        results[key] = {
            "name": name,
            "sig_counts": sig_counts,
            "l1_l2_ratios": ratios,
            "ginis": ginis,
            "energy_curves": energy_curves,
            "mean_energy": mean_energy,
            "n": n,
        }

        print(f"\n  {name} ({n} examples):")
        print(f"    Significant features: mean={np.mean(sig_counts):.1f}, "
              f"median={np.median(sig_counts):.1f}, std={np.std(sig_counts):.1f}")
        print(f"    L1/L2 ratio: mean={np.mean(ratios):.2f}")
        print(f"    Gini coefficient: mean={np.mean(ginis):.4f}")
        print(f"    Energy concentration:")
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"      Top-{k:>3d}: {mean_energy[k]:.4f} ({mean_energy[k]:.2%})")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Feature Identity Analysis
# ═══════════════════════════════════════════════════════════════════════

def phase5_feature_identity(alphas_pinv, alphas_sparse, delta_z_list):
    """Identify top features and analyze consistency across examples."""
    print("\n" + "=" * 75)
    print("PHASE 5: Feature Identity Analysis")
    print("=" * 75)

    method_data = {
        "pinv": alphas_pinv,
        "sparse": alphas_sparse,
        "encoder": delta_z_list,
    }

    identity_results = {}

    for key, alphas in method_data.items():
        n = len(alphas)
        alpha_mat = np.stack([a.numpy() if isinstance(a, torch.Tensor) else a
                              for a in alphas])  # [n, d]

        # Top-20 features by mean |α_j| across examples
        mean_abs = np.mean(np.abs(alpha_mat), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:20]

        top_features = []
        for rank, j in enumerate(top_idx):
            vals = alpha_mat[:, j]
            mean_val = float(vals.mean())
            std_val = float(vals.std())
            # Sign consistency: fraction matching dominant sign
            nonzero_signs = np.sign(vals[np.abs(vals) > 1e-10])
            if len(nonzero_signs) > 0:
                pos_frac = float((nonzero_signs > 0).mean())
                sign_consistency = max(pos_frac, 1 - pos_frac)
                dominant_sign = "+" if pos_frac > 0.5 else "-"
            else:
                sign_consistency = 0.0
                dominant_sign = "0"

            top_features.append({
                "rank": rank + 1,
                "feature_idx": int(j),
                "mean_abs": float(mean_abs[j]),
                "mean_val": mean_val,
                "std": std_val,
                "sign_consistency": sign_consistency,
                "dominant_sign": dominant_sign,
            })

        # Jaccard similarity of top-20 sets across example pairs
        top_sets = []
        for i in range(n):
            abs_a = np.abs(alpha_mat[i])
            top_k = set(np.argsort(abs_a)[::-1][:20].tolist())
            top_sets.append(top_k)

        jaccards = []
        for i in range(n):
            for j_idx in range(i + 1, n):
                inter = len(top_sets[i] & top_sets[j_idx])
                union = len(top_sets[i] | top_sets[j_idx])
                jaccards.append(inter / union if union > 0 else 0.0)

        mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
        std_jaccard = float(np.std(jaccards)) if jaccards else 0.0

        identity_results[key] = {
            "top_features": top_features,
            "mean_jaccard": mean_jaccard,
            "std_jaccard": std_jaccard,
        }

        names = {"pinv": "α_pinv", "sparse": "α_sparse", "encoder": "Δz"}
        print(f"\n  {names[key]} ({n} examples):")
        print(f"    Top-5 features by mean |α_j|:")
        for f in top_features[:5]:
            print(f"      Feature {f['feature_idx']:>5d}: mean|α|={f['mean_abs']:.4f}, "
                  f"sign={f['dominant_sign']} ({f['sign_consistency']:.0%})")
        print(f"    Jaccard similarity (top-20 sets): "
              f"mean={mean_jaccard:.4f}, std={std_jaccard:.4f}")

    return identity_results


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Optimizer Divergence
# ═══════════════════════════════════════════════════════════════════════

def phase6_optimizer_divergence(artifacts, alphas_pinv, alphas_sparse, example_indices):
    """Compare SAE optimizer's δ_SAE against algebraic decompositions."""
    print("\n" + "=" * 75)
    print("PHASE 6: Optimizer Divergence")
    print("=" * 75)

    w_sae = artifacts["w_sae"]
    b_sae = artifacts["b_sae"]
    sae_test_X = artifacts["sae_test_X"].double()
    sae_train_X = artifacts["sae_train_X"]

    print("  Building Woodbury components ...")
    t0 = time.time()
    Z, U, sigma_sq, numerical_rank, lam_adaptive = build_woodbury_components(
        sae_train_X, w_sae, b_sae
    )
    print(f"  Woodbury: rank={numerical_rank}, λ={lam_adaptive:.4f} ({time.time()-t0:.1f}s)")

    n_pinv = len(alphas_pinv)
    n_sparse = len(alphas_sparse)

    cos_sim_pinv = []
    cos_sim_sparse = []
    feature_overlap_pinv = []
    feature_overlap_sparse = []
    sae_delta_norms = []
    sae_deltas = []

    t0 = time.time()
    for i in range(n_pinv):
        idx = example_indices[i]
        x_sae = sae_test_X[idx]

        d_sae = robust_delta_sae_implicit(
            w_sae, b_sae, x_sae, Z, U, sigma_sq, lam_adaptive,
            numerical_rank, EPSILON, THRESHOLD
        )
        sae_deltas.append(d_sae)
        sae_delta_norms.append(d_sae.norm().item())

        # Cosine similarity with α_pinv
        if d_sae.norm().item() > 0 and alphas_pinv[i].norm().item() > 0:
            cos_pinv = torch.nn.functional.cosine_similarity(
                d_sae.unsqueeze(0), alphas_pinv[i].unsqueeze(0)
            ).item()
        else:
            cos_pinv = 0.0
        cos_sim_pinv.append(cos_pinv)

        # Feature overlap (top-20 by magnitude)
        top_sae = set(torch.argsort(d_sae.abs(), descending=True)[:20].tolist())
        top_pinv = set(torch.argsort(alphas_pinv[i].abs(), descending=True)[:20].tolist())
        overlap_pinv = len(top_sae & top_pinv) / len(top_sae | top_pinv) if top_sae | top_pinv else 0.0
        feature_overlap_pinv.append(overlap_pinv)

        if i < n_sparse:
            if d_sae.norm().item() > 0 and alphas_sparse[i].norm().item() > 0:
                cos_sp = torch.nn.functional.cosine_similarity(
                    d_sae.unsqueeze(0), alphas_sparse[i].unsqueeze(0)
                ).item()
            else:
                cos_sp = 0.0
            cos_sim_sparse.append(cos_sp)

            top_sp = set(torch.argsort(alphas_sparse[i].abs(), descending=True)[:20].tolist())
            overlap_sp = len(top_sae & top_sp) / len(top_sae | top_sp) if top_sae | top_sp else 0.0
            feature_overlap_sparse.append(overlap_sp)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_pinv}] cos(δ_SAE, α_pinv)={cos_pinv:.4f}, "
                  f"||δ_SAE||={sae_delta_norms[-1]:.4f} ({time.time()-t0:.1f}s)")

    print(f"\n  Cosine sim (δ_SAE vs α_pinv): "
          f"mean={np.mean(cos_sim_pinv):.4f}, std={np.std(cos_sim_pinv):.4f}")
    if cos_sim_sparse:
        print(f"  Cosine sim (δ_SAE vs α_sparse): "
              f"mean={np.mean(cos_sim_sparse):.4f}, std={np.std(cos_sim_sparse):.4f}")
    print(f"  Feature overlap top-20 (δ_SAE vs α_pinv): "
          f"mean={np.mean(feature_overlap_pinv):.4f}")
    if feature_overlap_sparse:
        print(f"  Feature overlap top-20 (δ_SAE vs α_sparse): "
              f"mean={np.mean(feature_overlap_sparse):.4f}")

    return {
        "cos_sim_pinv": cos_sim_pinv,
        "cos_sim_sparse": cos_sim_sparse,
        "feature_overlap_pinv": feature_overlap_pinv,
        "feature_overlap_sparse": feature_overlap_sparse,
        "sae_delta_norms": sae_delta_norms,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Report
# ═══════════════════════════════════════════════════════════════════════

def phase7_report(sparsity_results, identity_results, divergence_results, decomp_meta):
    """Generate comprehensive report with interpretability verdict."""
    print("\n" + "=" * 75)
    print("PHASE 7: Diagnostic Report")
    print("=" * 75)

    sp = sparsity_results   # shorthand
    ir = identity_results
    dr = divergence_results
    dm = decomp_meta

    L = []
    L.append("=" * 95)
    L.append("SAE FEATURE DECOMPOSITION OF RAW-SPACE ROBUST DELTAS")
    L.append("=" * 95)
    L.append("")
    L.append(f"Rashomon epsilon: {EPSILON}")
    L.append(f"Logit threshold:  {THRESHOLD}")
    L.append(f"Max examples:     {MAX_EXAMPLES} (L1: {L1_MAX_EXAMPLES})")
    L.append("")
    L.append("Mathematical setup:")
    L.append("  W_dec: [16384, 2304], full rank 2304, cond ~ 19.4")
    L.append("  Decomposition: find alpha in R^16384 s.t. alpha @ W_dec = delta_robust in R^2304")
    L.append("  Solution space: 14080-dimensional affine subspace")
    L.append("  Three methods: min-L2 (pinv), min-L1 (FISTA basis pursuit), encoder-based")
    L.append("")

    # ── Table 1: Three-Way Sparsity Comparison ──
    L.append("=" * 95)
    L.append("TABLE 1: THREE-WAY SPARSITY COMPARISON")
    L.append("=" * 95)
    L.append("")

    keys = ["pinv", "sparse", "encoder"]
    names = [sp[k]["name"] for k in keys]

    hdr = f"{'Metric':<32s}"
    for nm in names:
        hdr += f"  {nm:>18s}"
    L.append(hdr)
    L.append("-" * len(hdr))

    def row(label, values):
        r = f"{label:<32s}"
        for v in values:
            if isinstance(v, float):
                if abs(v) < 0.01 or abs(v) > 1000:
                    r += f"  {v:>18.2e}"
                else:
                    r += f"  {v:>18.4f}"
            elif isinstance(v, int):
                r += f"  {v:>18d}"
            else:
                r += f"  {str(v):>18s}"
        return r

    L.append(row("N examples", [sp[k]["n"] for k in keys]))
    L.append(row("Mean significant features",
                 [np.mean(sp[k]["sig_counts"]) for k in keys]))
    L.append(row("Median significant features",
                 [np.median(sp[k]["sig_counts"]) for k in keys]))
    L.append(row("Std significant features",
                 [np.std(sp[k]["sig_counts"]) for k in keys]))
    L.append(row("Mean L1/L2 ratio",
                 [np.mean(sp[k]["l1_l2_ratios"]) for k in keys]))
    L.append(row("Mean Gini coefficient",
                 [np.mean(sp[k]["ginis"]) for k in keys]))

    for kk in [1, 5, 10, 20, 50, 100]:
        L.append(row(f"Top-{kk} energy fraction",
                     [sp[k]["mean_energy"][kk] for k in keys]))

    L.append(row("Mean recon error (relative)",
                 [np.mean(dm["recon_pinv"]),
                  np.mean(dm["recon_sparse"]),
                  np.mean(dm["recon_enc"])]))
    L.append("")

    # ── Table 2: Energy Concentration Curves ──
    L.append("=" * 95)
    L.append("TABLE 2: ENERGY CONCENTRATION CURVES")
    L.append("=" * 95)
    L.append("")
    L.append("Fraction of ||alpha||^2 captured by top-k features:")
    L.append("")

    hdr2 = f"{'Top-k':<10s}"
    for nm in names:
        hdr2 += f"  {nm:>18s}"
    L.append(hdr2)
    L.append("-" * len(hdr2))
    for kk in ENERGY_KS:
        vals = [sp[k]["mean_energy"].get(kk, float('nan')) for k in keys]
        L.append(row(f"k={kk}", vals))
    L.append("")

    # ── Table 3: Top-20 Features ──
    L.append("=" * 95)
    L.append("TABLE 3: TOP-20 FEATURES (FROM EACH DECOMPOSITION)")
    L.append("=" * 95)

    for key in keys:
        nm = sp[key]["name"]
        feats = ir[key]["top_features"]
        L.append("")
        L.append(f"--- {nm} ({sp[key]['n']} examples) ---")
        L.append("")
        fhdr = (f"{'Rank':>4s}  {'Feature':>7s}  {'Mean|alpha|':>12s}  "
                f"{'Mean alpha':>12s}  {'Std':>10s}  {'Sign':>4s}  {'Consistency':>11s}")
        L.append(fhdr)
        L.append("-" * len(fhdr))
        for f in feats:
            L.append(f"{f['rank']:>4d}  {f['feature_idx']:>7d}  {f['mean_abs']:>12.6f}  "
                     f"{f['mean_val']:>12.6f}  {f['std']:>10.6f}  "
                     f"{f['dominant_sign']:>4s}  {f['sign_consistency']:>11.0%}")
    L.append("")

    # ── Table 4: Feature Consistency ──
    L.append("=" * 95)
    L.append("TABLE 4: FEATURE CONSISTENCY (JACCARD SIMILARITY OF TOP-20 SETS)")
    L.append("=" * 95)
    L.append("")
    name_map = {"pinv": "α_pinv (min-L2)", "sparse": "α_sparse (min-L1)", "encoder": "Δz (encoder)"}
    for key in keys:
        ird = ir[key]
        L.append(f"  {name_map[key]:>20s}: mean Jaccard = {ird['mean_jaccard']:.4f}, "
                 f"std = {ird['std_jaccard']:.4f}")
    L.append("")
    L.append("  Interpretation: Jaccard > 0.5 = strong overlap (same features targeted)")
    L.append("                  Jaccard < 0.1 = weak overlap (example-specific features)")
    L.append("")

    # ── Table 5: Optimizer Divergence ──
    L.append("=" * 95)
    L.append("TABLE 5: OPTIMIZER DIVERGENCE")
    L.append("=" * 95)
    L.append("")
    L.append("Comparison of SAE-space optimizer's delta_SAE (from 4-way prototype)")
    L.append("against algebraic decompositions of the raw-space robust delta:")
    L.append("")
    L.append(f"  Cosine sim (delta_SAE vs alpha_pinv):  "
             f"mean={np.mean(dr['cos_sim_pinv']):.4f}, std={np.std(dr['cos_sim_pinv']):.4f}")
    if dr['cos_sim_sparse']:
        L.append(f"  Cosine sim (delta_SAE vs alpha_sparse): "
                 f"mean={np.mean(dr['cos_sim_sparse']):.4f}, std={np.std(dr['cos_sim_sparse']):.4f}")
    L.append(f"  Feature overlap top-20 (vs alpha_pinv):  "
             f"mean={np.mean(dr['feature_overlap_pinv']):.4f}")
    if dr['feature_overlap_sparse']:
        L.append(f"  Feature overlap top-20 (vs alpha_sparse): "
                 f"mean={np.mean(dr['feature_overlap_sparse']):.4f}")
    L.append(f"  Mean ||delta_SAE||: {np.mean(dr['sae_delta_norms']):.4f}")
    L.append("")
    L.append("  Interpretation: cosine ~ 0 means optimizer found a completely different")
    L.append("  direction in SAE space than the algebraically correct one.")
    L.append("  cosine ~ 1 would mean the optimizer was on the right track.")
    L.append("")

    # ── Interpretability Verdict ──
    L.append("=" * 95)
    L.append("INTERPRETABILITY VERDICT")
    L.append("=" * 95)
    L.append("")

    sparse_mean_sig = np.mean(sp["sparse"]["sig_counts"])
    pinv_mean_sig = np.mean(sp["pinv"]["sig_counts"])
    sparse_top20 = sp["sparse"]["mean_energy"].get(20, 0)
    sparse_top50 = sp["sparse"]["mean_energy"].get(50, 0)
    pinv_top20 = sp["pinv"]["mean_energy"].get(20, 0)

    # Determine case
    if sparse_mean_sig < 50 and sparse_top20 > 0.90:
        case = "a"
        top_feats = ir["sparse"]["top_features"]
        feat_list = ", ".join(str(f['feature_idx']) for f in top_feats[:10])
        L.append("Case (a): SPARSE EXACT DECOMPOSITION EXISTS")
        L.append("")
        L.append(f"An exact sparse SAE decomposition of the robust delta EXISTS.")
        L.append(f"The L1-optimal decomposition uses ~{sparse_mean_sig:.0f} significant features,")
        L.append(f"capturing {sparse_top20:.1%} of energy in top-20 features.")
        L.append(f"This provides direct interpretability: robust steering primarily")
        L.append(f"targets features [{feat_list}].")
        L.append(f"No constrained optimization is needed — post-hoc decomposition is sufficient.")

    elif sparse_mean_sig > 200:
        case = "b"
        L.append("Case (b): DECOMPOSITION IS INTRINSICALLY DISTRIBUTED")
        L.append("")
        L.append(f"Even the sparsest exact decomposition requires ~{sparse_mean_sig:.0f} significant")
        L.append(f"features. The robust delta is intrinsically distributed across many SAE features.")
        L.append(f"Top-20 captures only {sparse_top20:.1%} of energy; top-50 captures {sparse_top50:.1%}.")
        L.append(f"Interpretable steering would require trading off exact Rashomon coverage")
        L.append(f"for sparsity via L1-penalized robust optimization — a genuine algorithmic")
        L.append(f"extension, not merely a post-hoc analysis change.")

    elif pinv_mean_sig > 2 * sparse_mean_sig and sparse_mean_sig <= 200:
        case = "c"
        L.append("Case (c): PSEUDOINVERSE IS MISLEADINGLY DENSE")
        L.append("")
        L.append(f"The minimum-L2 decomposition is misleadingly dense ({pinv_mean_sig:.0f} features).")
        L.append(f"The minimum-L1 decomposition reveals that a sparser exact representation")
        L.append(f"EXISTS with ~{sparse_mean_sig:.0f} significant features.")
        L.append(f"The pseudoinverse spreads energy across the null space unnecessarily.")
        L.append(f"Top-20 energy: L1={sparse_top20:.1%} vs L2={pinv_top20:.1%}.")

    else:
        case = "intermediate"
        L.append("INTERMEDIATE RESULT")
        L.append("")
        L.append(f"The L1-optimal decomposition uses ~{sparse_mean_sig:.0f} significant features")
        L.append(f"(vs ~{pinv_mean_sig:.0f} for L2-optimal).")
        L.append(f"Top-20 energy: L1={sparse_top20:.1%}, L2={pinv_top20:.1%}.")
        if sparse_mean_sig <= 100:
            L.append(f"While not maximally sparse, the L1 decomposition is moderately concentrated.")
        else:
            L.append(f"The decomposition is moderately distributed — partial interpretability.")

    L.append("")
    L.append(f"Quantitative summary:")
    L.append(f"  L2 (pinv): {pinv_mean_sig:.0f} sig features, top-20 energy = {pinv_top20:.1%}")
    L.append(f"  L1 (FISTA): {sparse_mean_sig:.0f} sig features, top-20 energy = {sparse_top20:.1%}")
    L.append(f"  L1 timing: {dm['l1_time']:.1f}s total ({dm['l1_time'] / len(dm['recon_sparse']):.1f}s/example)")
    L.append("")

    report = "\n".join(L)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")

    return report, case


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    artifacts = phase1_load()
    deltas, x_raws, example_indices = phase2_compute_raw_deltas(artifacts)
    alphas_pinv, alphas_sparse, delta_z_list, decomp_meta = phase3_decompositions(
        artifacts, deltas, x_raws
    )
    sparsity_results = phase4_sparsity(alphas_pinv, alphas_sparse, delta_z_list)
    identity_results = phase5_feature_identity(alphas_pinv, alphas_sparse, delta_z_list)
    divergence_results = phase6_optimizer_divergence(
        artifacts, alphas_pinv, alphas_sparse, example_indices
    )
    report, case = phase7_report(
        sparsity_results, identity_results, divergence_results, decomp_meta
    )

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Verdict case: {case}")
    print("SAE FEATURE DECOMPOSITION COMPLETE")


if __name__ == "__main__":
    main()
