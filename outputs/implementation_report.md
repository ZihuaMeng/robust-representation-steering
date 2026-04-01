# Implementation Report: Active-Subspace SAE Robust Steering

## 1. Problem Setup and Notation

We study the **Rashomon effect** in linear safety probes trained on Gemma-2 2B residual-stream activations (layer 10, $d = 2304$) over a balanced subset of BeaverTails ($n = 2000$, 80/20 train/test split).

**Linear probe.** $f(x) = \sigma(w^\top x + b)$, trained with BCE + $\ell_2$ regularization ($\lambda = 0.01$), AdamW, 50 epochs. The baseline probe achieves 75.5% accuracy on the test set.

**Rashomon set.** For validation loss $L_\text{val}$ and tolerance $\varepsilon = 0.15$:

$$\mathcal{R}(\varepsilon) = \bigl\{(w, b) : L_\text{val}(w, b) \leq L_\text{val}(\hat{w}, \hat{b}) + \varepsilon\bigr\}$$

We enumerate 50 probes within $\mathcal{R}$ via adversarial weight perturbation (AWP). These probes agree in parameter cosine ($\bar{\cos} = 0.9999$) but disagree on 19.5% of predictions (mean Hamming distance).

**Robust steering objective.** Given an "unsafe" example $x$ with $\hat{w}^\top x + \hat{b} < 0$, find the minimum-norm perturbation $\delta$ such that *every* probe in $\mathcal{R}$ classifies $x + \delta$ as safe. The ellipsoidal relaxation gives:

$$\min_\delta \|\delta\|^2 \quad \text{s.t.} \quad \hat{\theta}^\top \tilde{x}_\text{new} - \sqrt{2\varepsilon \cdot \tilde{x}_\text{new}^\top H^{-1} \tilde{x}_\text{new}} \geq t$$

where $\tilde{x}_\text{new} = [x + \delta;\, 1]$, $\hat{\theta} = [\hat{w};\, \hat{b}]$, $H$ is the Hessian of the BCE loss at $(\hat{w}, \hat{b})$, and $t = 0$ (logit threshold).

**SAE decomposition.** Gemma Scope JumpReLU SAE ($d_\text{SAE} = 16384$):

$$z = \text{ReLU}(W_\text{enc}\, x + b_\text{enc}) \odot \mathbb{1}[W_\text{enc}\, x + b_\text{enc} > \tau], \qquad \hat{x} = W_\text{dec}\, z + b_\text{dec}$$

From `src/sae_utils.py`:

```python
class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        return mask * F.relu(pre_acts)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec
```

**Active subspace.** $\mathcal{S} = \{j : z_j^{(i)} > 0 \text{ for at least } K \text{ training examples}\}$, with $d_\text{active} = |\mathcal{S}|$.

---

## 2. The 5-Step Pipeline

### Step 1: SAE Encoding and Active Subspace Projection

**Math.** Let $Z \in \mathbb{R}^{N \times d_\text{SAE}}$ be the matrix of SAE-encoded training activations. Define the per-feature activation frequency $c_j = \sum_{i=1}^N \mathbb{1}[z_j^{(i)} > 0]$ and the active set $\mathcal{S}_K = \{j : c_j \geq K\}$. The masking operator $M_{\mathcal{S}}$ selects columns indexed by $\mathcal{S}$:

$$X_\text{active} = Z \, M_{\mathcal{S}}^\top \in \mathbb{R}^{N \times d_\text{active}}$$

**Implementation.** From `scripts/11_active_subspace_steering.py`:

```python
# Boolean mask of nonzero features
active_mask = (sae_train_X > 0)  # [N, d_sae]

# Per-feature activation frequency (how many examples activate each feature)
feature_freq = active_mask.sum(dim=0).numpy()  # [d_sae]

# Active subspace sizes for different K thresholds
for K in [1, 2, 3, 5, 10, 20, 50]:
    mask_k = feature_freq >= K
    d_active_k = mask_k.sum()
```

**Results.** With $N = 1600$ training examples (mean 16.7 active features/example, median 8):

| $K$ | $d_\text{active}$ | % of 16384 |
|-----|-------------------:|----------:|
| 1   | 3,209              | 19.6%     |
| 5   | 1,117              | 6.8%      |
| 10  | 382                | 2.3%      |
| 50  | 35                 | 0.2%      |

The feature frequency distribution is heavily right-skewed: 13,175 features (80.4%) are never active on this dataset, while 3 features activate on essentially 100% of examples. We proceed with $K = 1$ ($d_\text{active} = 3209$) as the canonical setting and sweep $K \in \{1, 5, 10\}$ for robustness.

### Step 2: Active-Subspace Probe Training

**Math.** Standard BCE + $\ell_2$ on the restricted representation:

$$\min_{w_\mathcal{S},\, b} \; \frac{1}{N}\sum_{i=1}^N \bigl[-y_i \log \sigma(w_\mathcal{S}^\top x_{\mathcal{S}}^{(i)} + b) - (1-y_i)\log(1-\sigma(\cdots))\bigr] + \frac{\lambda}{2}\|w_\mathcal{S}\|^2$$

**Implementation.** From `src/probe.py`:

```python
def train_probe(train_X, train_y, input_dim=2304, lr=1e-3, weight_decay=0.01,
                epochs=50, device="cpu"):
    probe = LinearProbe(input_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        logits = probe(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Called with `input_dim=d_active` on $X_\text{active}$.

**Results.**

| Probe              | Dim    | Accuracy | F1    | AUROC |
|---------------------|--------|----------|-------|-------|
| Raw-space baseline | 2,304  | 75.5%    | —     | —     |
| Full SAE-space     | 16,384 | 69.5%    | —     | —     |
| Active-subspace    | 3,209  | 69.5%    | 0.679 | 0.735 |

The active-subspace probe matches the full SAE probe. Both are 6pp below the raw-space probe, reflecting information loss in the SAE encoding.

### Step 3: Active-Subspace Hessian Computation

**Math.** The Hessian of the BCE + $\ell_2$ loss at the baseline probe parameters, restricted to the active subspace:

$$H_\mathcal{S} = \frac{1}{N}\sum_{i=1}^N p_i(1-p_i)\,\tilde{x}_{\mathcal{S}}^{(i)} {\tilde{x}_{\mathcal{S}}^{(i)}}^\top + \lambda_\text{adapt} \cdot I_{d_\text{active}+1}$$

where $\tilde{x}_\mathcal{S}^{(i)} = [x_\mathcal{S}^{(i)};\,1]$, $p_i = \sigma(\hat{w}_\mathcal{S}^\top x_\mathcal{S}^{(i)} + \hat{b})$, and $\lambda_\text{adapt}$ is the adaptive ridge.

**Critical implementation detail: adaptive ridge.** With $d_\text{active} = 3209 > N = 1600$, the data-term Hessian has numerical rank 931 and $\sim$2,279 null directions. Without proper ridge regularization, $\text{cond}(H) \approx 507{,}000$ and the robust delta norms explode to $\sim 10^9$. We match the raw-space conditioning strategy: $\lambda_\text{adapt} = \sigma_\text{max}^2 / 100 + \lambda_{\ell_2}$, yielding $\text{cond}(H) \approx 101$.

From `scripts/11_active_subspace_steering.py`:

```python
logits = X64 @ w_active + b_active
p = torch.sigmoid(logits)
s = p * (1 - p)

sqrt_s = s.sqrt().unsqueeze(1)
X_scaled = sqrt_s * X_aug
H_data = (X_scaled.T @ X_scaled) / N_tr  # data term only

# Find max eigenvalue of data term
eigvals_data = torch.linalg.eigvalsh(H_data)
eig_max_data = eigvals_data[-1].item()

# Adaptive ridge: match raw-space conditioning (target cond ~ 100)
adaptive_ridge = eig_max_data / target_cond
lam_adaptive = weight_decay + adaptive_ridge

H = H_data + lam_adaptive * torch.eye(d_active + 1, dtype=torch.float64)
H_inv = torch.linalg.inv(H)
```

**Results.**

| Property           | Raw          | Full SAE       | Active-Sub     |
|--------------------|--------------|----------------|----------------|
| Hessian dim        | 2,305        | 16,385         | 3,210          |
| Data rank          | 2,305 (full) | 931            | 931            |
| $\lambda_\text{max}$ | 1,639.7   | 467.9          | 474.1          |
| $\lambda_\text{adapt}$ | $10^{-4}$ (not needed) | 4.69 | 4.75       |
| Condition          | 101          | $\sim$101 (Woodbury) | 101      |
| Inversion          | Dense        | Woodbury       | Dense          |

The data rank is 931 in both full-SAE and active-subspace settings — it is determined by $N$ and the effective degrees of freedom in the data, not by $d$.

### Step 4: Robust Delta Computation (Bisection)

**Math.** We scale the naive (minimum-norm boundary-crossing) delta $\delta_\text{naive}$ by a factor $\alpha \geq 1$ until the robust constraint is satisfied:

$$\hat{\theta}^\top \tilde{x}_\text{new}(\alpha) - \sqrt{2\varepsilon \cdot \tilde{x}_\text{new}(\alpha)^\top H_\mathcal{S}^{-1}\, \tilde{x}_\text{new}(\alpha)} \geq 0, \qquad \tilde{x}_\text{new}(\alpha) = [x_\mathcal{S} + \alpha\,\delta_\text{naive};\, 1]$$

We find the minimum $\alpha$ via bisection (60 iterations, yielding $\sim 10^{-18}$ relative precision).

From `src/steering.py`:

```python
def robust_delta(weight, bias, x, H_inv, epsilon, threshold=0.0):
    d_naive, score = naive_delta(weight, bias, x, threshold)
    if score >= threshold:
        return torch.zeros_like(x)

    theta = torch.cat([weight, bias.unsqueeze(0)])
    one = torch.ones(1, dtype=x.dtype)

    def check(scale):
        delta = d_naive * scale
        x_aug = torch.cat([x + delta, one])
        return _robust_margin(x_aug, theta, H_inv, epsilon, threshold)

    # Find upper bound, then bisect for minimum scale
    lo, hi = 1.0, 2.0
    for _ in range(30):
        if check(hi) >= 0: break
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if check(mid) >= 0: hi = mid
        else: lo = mid
    return d_naive * hi
```

This solver is dimension-agnostic: it works identically in the raw space ($d = 2304$), full SAE space ($d = 16384$, with implicit Woodbury $H^{-1}$), and active subspace ($d = 3209$, with dense $H^{-1}$).

**Results** (active subspace, $K = 1$, 50 unsafe test examples):

| Statistic                             | Value  |
|---------------------------------------|--------|
| Mean $\|\delta_\text{naive}\|_\mathcal{S}$ | 0.31 |
| Mean $\|\delta_\text{robust}\|_\mathcal{S}$ | 1.88 |
| Robust/naive ratio                    | 6.0x   |

### Step 5: Decode to Raw Activation Space

**Math.** Construct the full $d_\text{SAE}$-dimensional vector with zeros in inactive slots and the active-subspace delta in the active slots, then apply the decoder:

$$\delta_\text{sae,full}[j] = \begin{cases} \delta_\mathcal{S}[k] & \text{if } j = \mathcal{S}_k \\ 0 & \text{otherwise} \end{cases}, \qquad \delta_\text{raw} = W_\text{dec}^\top \delta_\text{sae,full}$$

Equivalently (and more efficiently), restrict the decoder to the active rows:

$$\delta_\text{raw} = \delta_\mathcal{S} \cdot W_\text{dec}[\mathcal{S},\, :]$$

This avoids instantiating the full 16k vector entirely.

From `scripts/11_active_subspace_steering.py`:

```python
# W_dec restricted to active features: [d_active, d_model]
W_dec_active = W_dec[active_indices, :]

# (E_robust) Robust active-subspace -> raw
dE_robust_active = robust_delta(
    w_active, b_active, x_active, H_inv_active, EPSILON, THRESHOLD
)
dE_robust_raw = dE_robust_active @ W_dec_active
```

**Results.**

| Norm metric                               | Mean   |
|-------------------------------------------|--------|
| $\|\delta_\text{robust}\|_\mathcal{S}$ (active space) | 1.88 |
| $\|\delta_\text{decoded}\|$ (raw space)   | 2.47   |
| Expansion factor                          | 1.31x  |

The decoder expands norms modestly (1.3x), consistent with $W_\text{dec}$ having condition number 19.4.

---

## 3. Results — Coverage Evaluation

All six strategies are evaluated against the **same 50 raw-space Rashomon probes** on 50 unsafe test examples. The raw-space probes are the ground truth: they define the Rashomon set we aim to cover.

| Strategy              | Mean Coverage | Mean $\|\delta_\text{raw}\|$ | 100% Rate | Cov/Norm |
|-----------------------|:------------:|:----------------------------:|:---------:|:--------:|
| A: Naive raw          | 40.0%        | 2.13                         | 0%        | 0.188    |
| B: Robust raw         | **100.0%**   | 6.93                         | **100%**  | 0.144    |
| C: Naive SAE (full)   | 9.4%         | 0.42                         | 0%        | 0.224    |
| D: Robust SAE (full)  | 13.6%        | 2.54                         | 0%        | 0.054    |
| E: Naive active-sub   | 9.4%         | 0.41                         | 0%        | 0.229    |
| F: Robust active-sub  | 13.3%        | 2.47                         | 0%        | 0.054    |

**K-sweep.** Coverage is insensitive to the active subspace dimension:

| $K$ | $d_\text{active}$ | Robust Coverage | Robust $\|\delta_\text{raw}\|$ |
|-----|-------------------:|:--------------:|:------------------------------:|
| 1   | 3,209              | 12.8%          | 2.22                           |
| 5   | 1,117              | 13.4%          | 3.76                           |
| 10  | 382                | 10.4%          | 3.05                           |
| —   | 16,384 (full SAE)  | 13.6%          | 2.54                           |

The $\sim$13% ceiling persists across a 43x range in subspace dimension (382 to 16,384).

---

## 4. Diagnostic Evidence — Why SAE Steering Fails

Three independent diagnostic experiments converge on the same conclusion.

### 4.1 Decoder Column-Space Diagnostic

**Question:** Does the SAE decoder lack the raw-space directions needed for Rashomon coverage?

**Method** (`scripts/09_decoder_subspace_diagnostic.py`): Compute $W_\text{dec}^\top W_\text{dec} \in \mathbb{R}^{2304 \times 2304}$ and its eigendecomposition. Project the known-good raw-space robust delta onto $\text{col}(W_\text{dec})$ and measure coverage loss.

**Result:** $W_\text{dec}$ has **full rank** (2304/2304), condition number 19.4, so $\text{col}(W_\text{dec}) = \mathbb{R}^{2304}$. Projection is the identity. Coverage is preserved at 100%.

**Conclusion:** Decoder expressivity is **not** the bottleneck. Any raw-space direction can be expressed as a decoded SAE vector.

### 4.2 Active-Subspace Restriction

**Question:** Does the $\sim$15,000-dimensional null space of the full-SAE Hessian cause the coverage collapse?

**Method** (`scripts/11_active_subspace_steering.py`): Restrict all computation to the $d_\text{active}$-dimensional active subspace, eliminating the null directions entirely.

**Result:** Coverage remains at $\sim$13% across $K \in \{1, 5, 10\}$, indistinguishable from the full-SAE result.

**Conclusion:** Null-space dimensionality is **not** the bottleneck.

### 4.3 Feature Decomposition

**Question:** Can the raw-space robust delta be represented sparsely in SAE coordinates?

**Method** (`scripts/10_sae_feature_decomposition.py`): Decompose $\delta_\text{robust}^\text{raw}$ into SAE coefficients $\alpha$ satisfying $\alpha \cdot W_\text{dec} = \delta_\text{robust}^\text{raw}$, via min-$\ell_2$ (pseudoinverse), min-$\ell_1$ (FISTA), and encoder-based decomposition.

**Result:**

| Method            | Nonzero Features | Top-20 Energy | Recon Error   |
|-------------------|:----------------:|:-------------:|:-------------:|
| Min-$\ell_2$      | 16,292           | 2.9%          | $3.2 \times 10^{-14}$ |
| Min-$\ell_1$      | 2,433            | 13.6%         | $10^{-3}$     |
| Encoder $\Delta z$ | 14.7            | 99.6%         | 105%          |

The SAE optimizer's $\delta_\text{SAE}$ has cosine similarity $\approx 0.09$ with the correct min-$\ell_2$ decomposition and zero overlap in the top-20 features.

**Conclusion:** The raw robust delta is **intrinsically dense** in SAE coordinates. The sparsest exact decomposition requires $\sim$2,433 features, and the SAE-space optimizer finds a nearly orthogonal direction.

### Synthesis

The three diagnostics rule out three candidate explanations:

1. ~~Decoder subspace limitation~~ $\to$ Full rank, projection is identity.
2. ~~Null-space inflation~~ $\to$ Active-subspace restriction doesn't help.
3. ~~Optimizer failure~~ $\to$ The correct SAE decomposition is inherently dense.

The root cause is **structural misalignment**: the SAE-space probe and the raw-space Rashomon probes define geometrically different decision boundaries. Optimizing the robust constraint with respect to the SAE-space Hessian $H_\mathcal{S}$ produces a delta that is robust to the SAE-space Rashomon set — but that set does not align with the raw-space Rashomon set. Decoding an SAE-space-robust delta to raw space does not yield a raw-space-robust delta, regardless of subspace restriction, ridge regularization, or optimizer quality.

---

## 5. Fluency Pilot

A minimal fluency evaluation was conducted on 5 unsafe test examples spanning the full score range ($-4.83$ to $-0.25$), generating text under three conditions: no steering (baseline), robust raw (B), and robust active-sub (F). All generations used Gemma-2 2B (base model), with the steering delta injected via a forward hook at layer 10, temperature 0.7, max 128 tokens.

**Findings:**

- All three conditions produce **coherent, fluent text**. Steering at norms up to $\sim$10 does not cause degeneration.
- **No refusal behavior** was observed under any steering condition. This is expected: the base model lacks safety training, so shifting the probe's "safety score" does not activate a refusal circuit.
- Robust raw steering occasionally shifts the topic away from the unsafe query (e.g., redirecting from "ways to hurt men" toward abuse counseling), but does not produce explicit refusal.
- Meaningful refusal-rate evaluation requires an instruction-tuned model (e.g., `gemma-2-2b-it`).

Full generated samples: `outputs/fluency_samples.txt`.

---

## 6. Conclusions and Recommended Next Directions

### Established Results

1. **Raw-space robust steering works.** The Hessian-based Rashomon ellipsoid approach achieves 100% coverage across all 50 probes, on every unsafe test example, at 3.25x the naive norm ($\bar{\|\delta\|} = 6.93$). This is validated and reliable.

2. **SAE-space steering does not achieve meaningful raw-space Rashomon coverage.** Across four SAE-based strategies (naive full, robust full, naive active-sub, robust active-sub), coverage ranges from 9% to 14%. This ceiling is stable across a 43x range in subspace dimension.

3. **The failure is structural, not computational.** Three independent diagnostics (decoder rank, active-subspace restriction, feature decomposition) rule out decoder expressivity, null-space inflation, and optimizer quality as explanations. The root cause is geometric misalignment between SAE-space and raw-space probe boundaries.

4. **Steering preserves generation fluency.** Deltas at norms up to $\sim$10 do not degrade text coherence in the base model.

### Candidate Next Directions

- **Focus on raw-space robust steering.** Evaluate fluency and safety tradeoffs at scale using an instruction-tuned model, where the steering delta may amplify an existing refusal mechanism.
- **Joint characterization of SAE and raw Rashomon sets.** Study whether any geometric relationship exists between the two Rashomon ellipsoids that could be exploited.
- **Hybrid optimization.** Optimize in raw space subject to an SAE-sparsity constraint ($\ell_1$ penalty on the SAE decomposition), trading Rashomon coverage for interpretability.
- **Instruction-tuned model.** Repeat the full pipeline on `gemma-2-2b-it` to determine whether the Rashomon geometry changes with safety training.
