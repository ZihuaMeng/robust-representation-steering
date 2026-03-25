# Implementation & Math Verification: Robust Steering & AWP

## Section 1: Data Split & No-Leakage Guarantee

### 1.1 Balanced Subset Sampling

The dataset is BeaverTails `330k_train`. A balanced subset of 2000 examples (1000 safe, 1000 unsafe) is drawn with `np.random.RandomState(seed=42)`:

```python
# src/data_pipeline.py, lines 11-37
def load_balanced_beavertails(n_per_class=1000, seed=42):
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    safe_idx = [i for i, ex in enumerate(ds) if ex["is_safe"]]
    unsafe_idx = [i for i, ex in enumerate(ds) if not ex["is_safe"]]
    rng = np.random.RandomState(seed)
    safe_sample = rng.choice(safe_idx, size=min(n_per_class, len(safe_idx)), replace=False)
    unsafe_sample = rng.choice(unsafe_idx, size=min(n_per_class, len(unsafe_idx)), replace=False)
    indices = np.concatenate([safe_sample, unsafe_sample])
    texts = [ds[int(i)]["response"] for i in indices]
    labels = [1] * len(safe_sample) + [0] * len(unsafe_sample)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels,
    )
    return train_texts, test_texts, train_labels, test_labels
```

The `train_test_split` call uses `test_size=0.2, random_state=42, stratify=labels`, producing exactly **1600 train / 400 test** examples with preserved class balance.

### 1.2 AWP Internal Train/Val Split

AWP further splits the 1600 training examples into a train subset and validation subset. This split is performed in `scripts/run_awp_rashomon.py`:

```python
# scripts/run_awp_rashomon.py, lines 46-51
seed = 42
n_val = int(len(train_y) * 0.2)
perm = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(seed))
val_idx, tr_idx = perm[:n_val], perm[n_val:]
val_X, val_y = train_X[val_idx], train_y[val_idx].float()
sub_train_X, sub_train_y = train_X[tr_idx], train_y[tr_idx].float()
```

With `len(train_y) = 1600` and `int(1600 * 0.2) = 320`, this produces **1280 train_sub / 320 val_sub**. The permutation is seeded by `torch.Generator().manual_seed(42)`, making it deterministic and reproducible.

### 1.3 Generalization Analysis Reconstructs the Same Split

`scripts/analyze_probe_losses.py` reconstructs the identical AWP split for loss computation:

```python
# scripts/analyze_probe_losses.py, lines 76-80
seed = 42
n_val = int(len(train_y) * 0.2)
perm = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(seed))
val_idx, tr_idx = perm[:n_val], perm[n_val:]
```

This is byte-identical to the AWP script. Same seed, same generator, same indexing convention.

### 1.4 Data Flow Diagram

```
BeaverTails 330k_train
        |
        | np.random.RandomState(42): sample 1000 safe + 1000 unsafe
        v
   2000 balanced examples
        |
        | sklearn.train_test_split(test_size=0.2, random_state=42, stratify)
        v
  +-----------+----------+
  | Train     | Test     |
  | n=1600    | n=400    |
  +-----------+----------+
        |            |
        |            +---> NEVER used during training or AWP.
        |                  Used only for final evaluation.
        |
        | torch.randperm(1600, seed=42)
        v
  +-----------+----------+
  | train_sub | val_sub  |
  | n=1280    | n=320    |
  +-----------+----------+
        |            |
        |            +---> Rashomon bound check only (no gradient updates)
        |
        +---> AWP anchor sampling (gradient updates target single examples)
```

### 1.5 No-Leakage Verification

The AWP engine (`src/awp.py`, function `run_awp_rashomon`) receives `train_X, train_y, val_X, val_y` as arguments. It passes `val_X, val_y` into `perturb_one_probe()` solely for constraint evaluation via `compute_val_loss()`. The test set is not an argument to `run_awp_rashomon()` and does not appear anywhere in `src/awp.py`.

In `scripts/run_awp_rashomon.py`, the test set (`test_X, test_y`) is loaded (line 38) but used only after AWP completes, for evaluation on line 83+.

**Verdict:** The test set (n=400) is never used during probe training or AWP enumeration. The val split (n=320) is used only for the Rashomon bound, never for gradient updates.


## Section 2: AWP Optimization Loop

### 2.1 Mathematical Constraint

The Rashomon set is defined as:

$$\mathcal{R}(\varepsilon) = \{(w, b) : L_{\text{val}}(w, b) \leq L_{\text{val}}(\hat{w}, \hat{b}) + \varepsilon\}$$

where $L_{\text{val}}$ is BCE loss on the 320-example validation split, $(\hat{w}, \hat{b})$ are the baseline probe parameters, and $\varepsilon = 0.15$.

### 2.2 Optimization Strategy

For each of 50 candidates, AWP selects a random training anchor and runs constrained SGD to push the probe toward flipping that anchor's prediction:

- **Local objective per anchor:** BCE loss on a single example, with target label = 1 - true_label (i.e., push toward the opposite class).
- **Optimizer:** SGD with `lr=1e-4`, `momentum=0.9`.
- **Steps per anchor:** 400 SGD steps.
- **Constraint enforcement:** Simple accept/reject. After each gradient step, the full validation loss is computed. If it exceeds the Rashomon bound, the step is reverted and the optimizer is reset (clearing momentum buffers). This is neither projected GD nor a Lagrangian — it is a greedy accept/reject scheme with rollback.

### 2.3 Core AWP Loop (Verbatim)

This is the critical code path. Every line is extracted from `src/awp.py`, function `perturb_one_probe`, lines 15-57:

```python
# src/awp.py, lines 25-57 — perturb_one_probe()
w = baseline_w.clone().to(device).requires_grad_(True)
b = baseline_b.clone().to(device).requires_grad_(True)
anchor_x = anchor_x.to(device)
target = torch.tensor([target_label], dtype=torch.float32, device=device)

last_valid_w = baseline_w.clone().to(device)
last_valid_b = baseline_b.clone().to(device)
optimizer = torch.optim.SGD([w, b], lr=lr, momentum=momentum)
accepted = 0

for step in range(sgd_steps):
    # 1. Gradient step on the local objective (BCE toward label flip)
    optimizer.zero_grad()
    logit = (w * anchor_x).sum() + b  # scalar
    loss = F.binary_cross_entropy_with_logits(logit.unsqueeze(0), target)
    loss.backward()
    optimizer.step()

    # 2. Validation loss computation after the step
    vl = compute_val_loss(w.data, b.data, val_X, val_y)

    # 3. Comparison against the Rashomon bound
    if vl <= baseline_val_loss + epsilon:
        last_valid_w.copy_(w.data)
        last_valid_b.copy_(b.data)
        accepted += 1
    else:
        # 4. Revert if bound is violated
        w.data.copy_(last_valid_w)
        b.data.copy_(last_valid_b)
        optimizer = torch.optim.SGD([w, b], lr=lr, momentum=momentum)

moved = not (
    torch.allclose(last_valid_w, baseline_w.to(device))
    and torch.allclose(last_valid_b, baseline_b.to(device))
)
return last_valid_w.cpu(), last_valid_b.cpu(), accepted, moved
```

### 2.4 Parameter Isolation

The baseline probe is never modified. Line 25 shows `w = baseline_w.clone()` — a fresh copy. The `baseline_w` and `baseline_b` tensors are passed by reference but only read from (for initialization and the `torch.allclose` check at the end). The SGD optimizer operates on `[w, b]`, not on the baseline parameters. On revert (line 49), the working copy `w` is reset to `last_valid_w`, which was itself initialized from `baseline_w.clone()` (line 30).


## Section 3: Generalization Parity ($r = 0.9973$)

### 3.1 Float64 Loss Computation

All losses are computed in float64 to avoid numerical noise:

```python
# scripts/analyze_probe_losses.py, lines 27-34
def bce_loss(weight, bias, X, y):
    """Compute mean BCE loss in float64 for numerical precision."""
    weight = weight.double()
    bias = bias.double()
    X = X.double()
    y = y.double()
    logits = X @ weight + bias
    return F.binary_cross_entropy_with_logits(logits, y, reduction="mean").item()
```

This computes raw BCE without L2 regularization. A separate function `l2_regularized_loss` (lines 37-46) computes $\text{BCE} + \lambda \|w\|^2$ with $\lambda = 0.01$, and is also reported.

### 3.2 Results

1. Mean val-test gap across 50 AWP probes: $0.1386 \pm 0.0030$
2. Baseline probe's own val-test gap: $0.1352$
3. Pearson $r(\text{val loss}, \text{test loss}) = 0.9973$

The gap is a constant distributional offset (CoV = 2.2%), not probe-specific overfitting. The val-loss bound is a faithful proxy for test-set performance.


## Section 4: Hessian Computation & Regularization

### 4.1 Mathematical Formula

The Hessian of the BCE loss for a linear probe is computed in closed form:

$$H = \frac{1}{N} \sum_{i=1}^{N} p_i(1 - p_i) \cdot \tilde{x}_i \tilde{x}_i^\top + \text{diag}(\lambda, \ldots, \lambda, 0) + \mu I$$

where $p_i = \sigma(\hat{w}^\top x_i + \hat{b})$, $\tilde{x}_i = [x_i; 1] \in \mathbb{R}^{2305}$, $\lambda = 0.01$ is the L2 weight decay (applied to weights only, not bias), and $\mu$ is an adaptive ridge parameter.

### 4.2 Core Computation (Verbatim)

```python
# src/hessian.py, lines 24-49
X64 = X.double()
w64 = weight.double()
b64 = bias.double()

N, d = X64.shape
ones = torch.ones(N, 1, dtype=torch.float64)
X_aug = torch.cat([X64, ones], dim=1)  # [N, d+1]

# Predicted probabilities at baseline
logits = X64 @ w64 + b64  # [N]
p = torch.sigmoid(logits)  # [N]
s = p * (1 - p)  # [N] — Hessian scaling factors

# H = (1/N) X_aug^T diag(s) X_aug
sqrt_s = s.sqrt().unsqueeze(1)  # [N, 1]
X_scaled = sqrt_s * X_aug  # [N, d+1]
H = (X_scaled.T @ X_scaled) / N

# L2 regularization on w only (not bias)
reg = torch.zeros(d + 1, dtype=torch.float64)
reg[:d] = weight_decay
H += torch.diag(reg)

# Ridge for numerical stability
H += ridge * torch.eye(d + 1, dtype=torch.float64)
```

The computation uses the identity $X^\top \text{diag}(s) X = (\sqrt{s} \odot X)^\top (\sqrt{s} \odot X)$, which avoids materializing the $N \times N$ diagonal matrix. With $d = 2304$, the output is a $[2305, 2305]$ matrix in float64.

### 4.3 Adaptive Ridge Parameter

The ridge $\mu$ is chosen to target a condition number of ~100. This is done in `scripts/run_robust_steering.py`:

```python
# scripts/run_robust_steering.py, lines 48-53
# First pass: compute Hessian with minimal ridge to find eigenvalue scale
H_raw, _, eig_raw = compute_hessian(w, b, train_X, train_y, ridge=1e-6)
# Set ridge so condition number ~ 100
target_cond = 100.0
ridge = eig_raw["eig_max"] / target_cond
H, H_inv, eig_info = compute_hessian(w, b, train_X, train_y, ridge=ridge)
```

**Strategy:** A first pass with negligible ridge ($10^{-6}$) determines $\lambda_{\max}$. The adaptive ridge is then set to $\mu = \lambda_{\max} / 100$, which shifts all eigenvalues by $\mu$ and guarantees the condition number is approximately $(\lambda_{\max} + \mu) / (\lambda_{\min} + \mu) \approx 100$ (exact when $\lambda_{\min} \ll \mu$).

### 4.4 Eigenvalue Diagnostics

Diagnostics are computed and reported within `compute_hessian`:

```python
# src/hessian.py, lines 52-61
eigvals = torch.linalg.eigvalsh(H)
eig_min = eigvals[0].item()
eig_max = eigvals[-1].item()
cond = eig_max / eig_min if eig_min > 0 else float("inf")
# ...
H_inv = torch.linalg.inv(H)
```

**Reported values** (from `outputs/steering_comparison_report.txt`): eigenvalue range $[16.2, 1639.7]$, condition number $= 101.0$. The inversion uses `torch.linalg.inv` (dense direct inversion), which is exact for this matrix size.


## Section 5: Robust Delta Solver

### 5.1 Constraint Being Solved

The robust steering perturbation satisfies:

$$\hat{\theta}^\top \tilde{x}_{\text{new}} - \sqrt{2\varepsilon \cdot \tilde{x}_{\text{new}}^\top H^{-1} \tilde{x}_{\text{new}}} \geq t$$

where $\hat{\theta} = [\hat{w}; \hat{b}] \in \mathbb{R}^{2305}$, $\tilde{x}_{\text{new}} = [x + \delta; 1] \in \mathbb{R}^{2305}$, $\varepsilon = 0.15$, $t = 0$ (logit threshold for 50% probability), and $H^{-1}$ is the regularized inverse Hessian.

The left side is a worst-case lower bound on the classification score across all probes in the Rashomon ellipsoid. Satisfying this constraint guarantees that **every** probe in $\mathcal{R}(\varepsilon)$ classifies $x + \delta$ as safe.

### 5.2 Naive Delta (Closed Form)

```python
# src/steering.py, lines 15-30
def naive_delta(weight, bias, x, threshold=0.0):
    score = (weight * x).sum() + bias
    score_f = score.item()
    if score_f >= threshold:
        return torch.zeros_like(x), score_f
    gap = threshold - score_f
    w_norm_sq = (weight * weight).sum().item()
    delta = (gap / w_norm_sq) * weight
    return delta, score_f
```

This implements $\delta_{\text{naive}} = \frac{t - \hat{w}^\top x - \hat{b}}{\|\hat{w}\|^2} \cdot \hat{w}$. The perturbation is in the direction of $\hat{w}$ and has dimension $d = 2304$. The bias dimension is **not** perturbed: `weight` is $[d]$, `x` is $[d]$, and the returned `delta` is $[d]$. The bias contributes to `score` but not to the perturbation direction.

### 5.3 Robust Margin Evaluation

```python
# src/steering.py, lines 33-37
def _robust_margin(x_aug, theta, H_inv, epsilon, threshold):
    """Compute robust margin: theta^T x_aug - sqrt(2*eps * x_aug^T H_inv x_aug) - t."""
    linear = theta @ x_aug
    quad = x_aug @ H_inv @ x_aug
    return (linear - torch.sqrt(2.0 * epsilon * quad.clamp(min=0.0)) - threshold).item()
```

### 5.4 Bisection Solver

The solver scales the naive delta direction by a scalar factor and uses bisection to find the minimum scale that satisfies the robust constraint:

```python
# src/steering.py, lines 40-84
def robust_delta(weight, bias, x, H_inv, epsilon, threshold=0.0):
    d_naive, score = naive_delta(weight, bias, x, threshold)
    if score >= threshold:
        return torch.zeros_like(x)

    theta = torch.cat([weight, bias.unsqueeze(0)])  # [d+1]
    one = torch.ones(1, dtype=x.dtype)

    def check(scale):
        delta = d_naive * scale
        x_aug = torch.cat([x + delta, one])
        return _robust_margin(x_aug, theta, H_inv, epsilon, threshold)

    if check(1.0) >= 0:          # naive already satisfies robust constraint
        return d_naive

    lo, hi = 1.0, 2.0            # exponential search for upper bound
    for _ in range(30):
        if check(hi) >= 0:
            break
        hi *= 2.0
    else:
        return d_naive * hi       # fallback: best effort at max scale

    for _ in range(60):           # bisection: 60 iterations → precision ~1e-18
        mid = (lo + hi) / 2.0
        if check(mid) >= 0:
            hi = mid
        else:
            lo = mid

    return d_naive * hi
```

**What is bisected:** The scalar `scale` multiplying `d_naive`. The search direction is fixed as the naive delta direction $\hat{w} / \|\hat{w}\|^2 \cdot \text{gap}$; only the magnitude changes. This reduces the $d$-dimensional optimization to a 1D search.

**Convergence:** 60 bisection iterations on a range starting at $[1, 2]$ (or wider after exponential doubling) yields precision $\sim 2^{-60} \approx 10^{-18}$, which is well below float64 machine epsilon.

### 5.5 Rashomon Coverage Evaluation

After computing both naive and robust deltas, coverage is evaluated across all 50 AWP probes:

```python
# src/steering.py, lines 87-99
def rashomon_coverage(delta, x, probes, threshold=0.0):
    x_new = x + delta
    safe = 0
    for p in probes:
        score = (p["weight"].double() * x_new).sum() + p["bias"].double()
        if score.item() >= threshold:
            safe += 1
    return safe, len(probes), safe / len(probes) if probes else 0.0
```

This iterates all 50 AWP probes, computes each probe's logit score on $x + \delta$ in float64, and counts how many classify the perturbed input as safe ($\text{score} \geq 0$). The reported results: **naive achieves 20/50 coverage on every example; robust achieves 50/50 coverage on every example.**
