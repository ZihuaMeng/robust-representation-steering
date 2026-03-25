# Rashomon-Robust Representation Steering: Preliminary Results on Gemma-2 2B Safety Probes

## 1. Setup

We investigate a fundamental question in activation-based LLM safety: **is the learned probe direction unique, or do many near-optimal directions exist?** If the safety probe used for representation steering is not uniquely determined by the data, then steering along any single probe direction may fail to generalize across the space of equally-valid classifiers.

**Model and data.** We study Gemma-2 2B (2.6B parameters, 26 layers, hidden dim $d = 2304$). We extract mean-pooled residual-stream activations from **layer 10** on a balanced subset of the BeaverTails safety dataset: 1,600 training examples and 400 test examples (50% safe, 50% unsafe). A linear probe $f(x) = w^\top x + b$ is trained with BCE loss and $\ell_2$ regularization ($\lambda = 0.01$, AdamW, 50 epochs).

**Baseline probe performance.** Test accuracy = 75.5%, AUROC = 85.7%, F1 = 76.2%. This moderate accuracy is expected for a single mid-network layer on a binary safety task; the probe captures a meaningful but imperfect safety direction.

## 2. The Rashomon Effect in Probe Space

### Formal Definition

Given baseline parameters $(\hat{w}, \hat{b})$ with validation loss $L_{\text{val}}(\hat{w}, \hat{b})$, the **Rashomon set** at tolerance $\varepsilon$ is:

$$\mathcal{R}(\varepsilon) = \{(w, b) : L_{\text{val}}(w, b) \leq L_{\text{val}}(\hat{w}, \hat{b}) + \varepsilon\}$$

We enumerate members of $\mathcal{R}(\varepsilon = 0.15)$ using **Adversarial Weight Perturbation** (AWP): for each of 50 randomly-selected training examples, constrained SGD pushes the probe toward flipping that example's prediction while maintaining the validation-loss bound. Any parameter configuration that violates the bound is rejected. This produces 50 distinct probes, each certifiably within the Rashomon set.

### Core Finding

| Metric | Raw Space ($d = 2304$) | SAE Space ($d = 16384$) |
|--------|:----------------------:|:-----------------------:|
| Mean Hamming Distance | 19.5% | 44.5% |
| Max Hamming Distance | 43.8% | 90.5% |
| Cosine Similarity (mean) | 0.9999 | 0.9997 |
| Accuracy Range | 68.5 -- 78.0% | 50.8 -- 69.5% |
| F1 Range | 59.6 -- 81.0% | 6.6 -- 69.2% |

The central paradox: **cosine similarity exceeds 0.999** between all probe pairs, yet they **disagree on up to 43.8%** of test predictions (raw space) and **90.5%** (SAE space). Probes that appear nearly identical in weight space produce dramatically different classifications on examples near the decision boundary.

### Why SAE Space Amplifies the Effect

The Gemma Scope sparse autoencoder projects activations into a 16,384-dimensional space with ~99.9% sparsity (approximately 17 active features per example). This creates a vast null space: the $\varepsilon$-ball around the baseline probe encompasses a much larger volume of functionally-distinct directions when projected through such a sparse, high-dimensional representation. The bimodal Hamming distribution in SAE space (probe pairs either agree well or disagree on ~85% of examples) suggests that SAE-space Rashomon probes cluster into qualitatively different classification strategies.

## 3. Generalization of Rashomon Probes

A natural concern: probes are admitted to the Rashomon set based on a **validation** loss bound. Do they also perform comparably on held-out **test** data, or do some probes overfit the validation split?

### Methodology

We reconstruct the exact 80/20 train/validation split used during AWP (seed = 42) and compute, for each of the 51 probes (1 baseline + 50 AWP), the BCE loss on three disjoint sets: the training subset ($n = 1280$), the validation subset ($n = 320$), and the held-out test set ($n = 400$). All losses are computed in float64.

### Results

| Statistic | Value |
|-----------|------:|
| Mean Val-Test Gap (AWP probes) | $0.1386 \pm 0.0030$ |
| Max Val-Test Gap | 0.1456 |
| Baseline Val-Test Gap | 0.1352 |
| Excess gap (AWP mean $-$ baseline) | 0.0034 |
| Pearson $r$(val loss, test loss) | 0.9973 |
| AWP probes within val-loss bound | 50/50 |
| AWP probes within analogous test-loss bound | 46/50 |

### Interpretation

The val-test gap of ~0.14 is **not** evidence of overfitting. Three observations establish this:

1. **The baseline probe itself exhibits the same gap** (0.135). The offset is a property of the train/test distribution difference, not of any particular probe.

2. **The gap is nearly constant** across all 50 AWP probes (std = 0.003, coefficient of variation = 2.2%). If probes were overfitting to the validation split, we would expect heterogeneous gaps correlated with how aggressively each probe was perturbed. Instead, the gap is invariant.

3. **Val loss and test loss are near-perfectly correlated** (Pearson $r = 0.997$). Probes with lower validation loss also have lower test loss, with essentially no rank reversals. The validation-loss ordering is a faithful proxy for the test-loss ordering.

The mean excess gap attributable to AWP (0.003) is negligible. **The Rashomon bound transfers reliably to held-out data**: probes admitted by the validation criterion generalize, and the val-loss ranking among probes is preserved on test data.

## 4. Robust Steering

Given that multiple near-optimal probes exist, naive steering along the baseline direction alone is insufficient. We derive a **worst-case robust** steering perturbation.

### Formulation

Let $\hat{\theta} = [\hat{w}; \hat{b}]$ be the baseline probe parameters, $H$ the Hessian of the BCE loss at $\hat{\theta}$, and $\tilde{x} = [x_{\text{new}}; 1]$ the augmented input. The robust steering problem is:

$$\min_{\delta} \|\delta\|^2 \quad \text{s.t.} \quad \hat{\theta}^\top \tilde{x}_{\text{new}} - \sqrt{2\varepsilon \cdot \tilde{x}_{\text{new}}^\top H^{-1} \tilde{x}_{\text{new}}} \geq t$$

In words: find the minimum-norm activation perturbation $\delta$ that guarantees the safety classification threshold $t$ is met under the **worst-case probe** in the Rashomon ellipsoid $\{{\theta : (\theta - \hat{\theta})^\top H (\theta - \hat{\theta}) \leq 2\varepsilon}\}$. The correction term $\sqrt{2\varepsilon \cdot \tilde{x}^\top H^{-1} \tilde{x}}$ accounts for the maximum logit reduction any Rashomon-feasible probe could produce.

### Results

| | Naive Steering | Robust Steering |
|---|---:|---:|
| Mean $\|\delta\|$ | 2.13 | 6.93 |
| Mean Rashomon Coverage | 40% (20/50) | 100% (50/50) |
| 100% Coverage Rate | 0% of examples | 100% of examples |

Robust steering requires a **3.25$\times$ larger perturbation norm** but achieves **universal Rashomon coverage**: every unsafe example is steered past the safety threshold for all 50 probes simultaneously. Naive steering, by contrast, achieves exactly 20/50 coverage on every example --- consistent with the bimodal clustering of Rashomon probes around two opposing decision boundary orientations.

**Hessian diagnostics.** The BCE Hessian is regularized via adaptive ridge to a condition number of 101 (eigenvalue range: [16.2, 1639.7]). This ensures the quadratic approximation underlying the Rashomon ellipsoid is well-conditioned.

## 5. Limitations and Next Steps

### Current Limitations

- **Single layer, single model, single dataset.** All results are from layer 10 of Gemma-2 2B on a 2,000-example BeaverTails subset. Generalization to other layers, models, and safety benchmarks is untested.
- **Moderate probe accuracy.** The 75.5% baseline accuracy leaves room for improvement via deeper layers, nonlinear probes, or larger training sets. The Rashomon effect may change character at higher accuracy levels.
- **No generation evaluation.** We have not yet injected $\delta$ into the forward pass and measured actual text generation quality. The 3.25$\times$ norm penalty is a concern: it may degrade fluency or coherence.
- **Uniform naive coverage.** Every unsafe example shows exactly 20/50 naive coverage, suggesting the 50 Rashomon probes cluster into exactly two groups. This may be an artifact of the AWP enumeration strategy rather than the true geometry of the Rashomon set.

### Next Steps

- **End-to-end generation evaluation.** Inject robust $\delta$ into the Gemma-2 forward pass and measure safety (refusal rate) and fluency (perplexity, coherence scores) simultaneously.
- **Multi-layer analysis.** Repeat the pipeline at layers 10, 15, and 20 to study how the Rashomon effect and steering cost vary with depth.
- **Scale to full dataset.** Move from 2,000 examples to the full BeaverTails training set (~330k) to assess whether the Rashomon effect persists or diminishes with more data.
- **Baseline comparisons.** Compare robust Rashomon steering against Activation Addition (Turner et al., 2023) and RepE (Zou et al., 2023) on matched safety/fluency metrics.
- **Tighter Rashomon bounds.** Investigate whether the Hessian-based ellipsoidal approximation can be tightened (e.g., using higher-order corrections or sampling-based coverage estimates).

## 6. SAE-Space Hessian: Spectral Analysis and Woodbury Inversion

### Setup

The SAE-space Hessian operates in $d = 16{,}384$ dimensions (Gemma Scope JumpReLU, 16k features). With $N_\text{train} = 1{,}600 \ll d$, the data-dependent term $\frac{1}{N} \tilde{X}^\top \text{diag}(s) \tilde{X}$ has rank $\leq 1{,}600$. This "ridge + low-rank" structure enables exact inversion via the Woodbury identity without forming the full $16{,}385 \times 16{,}385$ matrix.

### Spectral Characterization

| Property | Raw Space ($d = 2304$) | SAE Space ($d = 16384$) |
|:---------|:----------------------:|:-----------------------:|
| Hessian dimension | 2,305 | 16,385 |
| $N_\text{train}$ | 1,600 | 1,600 |
| $N_\text{train}/d$ ratio | ~0.69 | 0.098 |
| Numerical rank | 2,305 (full) | 931 |
| Effective rank (95%) | full | 3 |
| Effective rank (99%) | full | 120 |
| Eigenvalue range (data-dep.) | [16.2, 1639.7] | [$2.4 \times 10^{-8}$, 468] |
| Condition (data-dep.) | 101 | $1.93 \times 10^{10}$ |
| Condition (full $H$, $\lambda = 0.01$) | ~101 | $4.68 \times 10^4$ |

Key findings: eigenvalue decay is extremely rapid ($\sigma_0^2 = 467.9$ captures 87.3% of spectral mass). Numerical rank 931 $\ll N_\text{train} = 1{,}600$: many training examples are linearly dependent in SAE space due to 99.9% sparsity.

### Woodbury Inversion

The Woodbury identity decomposes $H^{-1} = (\lambda I + U \Sigma U^\top)^{-1}$ using the eigendecomposition of the data-dependent term. Performance: ~22 ms/matvec, verification residual $1.06 \times 10^{-7}$, agreement with dense inverse at $10^{-12}$.

### Adaptive Ridge Requirement

With weight-decay-only regularization ($\lambda = 0.01$), the 15,454 null directions have $H^{-1}$ eigenvalue $= 1/\lambda = 100$, making the Rashomon ellipsoid infinite in practice. The bisection solver diverges (scale hits $2^{31}$). Adaptive ridge $\lambda = \sigma_\text{max}^2/100 + \lambda_\text{wd} \approx 4.69$ (condition $\approx 101$) yields finite robust deltas with mean robust/naive ratio 18.25$\times$ (vs ~1.45$\times$ in raw space). The larger ratio reflects the 15,000+ unconstrained directions in SAE space.

## 7. SAE-Decoded Steering: Four-Way Factorial

### Experimental Design

A $2 \times 2$ factorial comparison: {naive, robust} $\times$ {raw, SAE$\to$raw}. All four deltas are evaluated against the **same** 50 raw-space Rashomon probes. SAE decode: $\delta_\text{raw} = \delta_\text{SAE} W_\text{dec}$ (bias cancels for perturbations). SAE Woodbury: rank 931, $\lambda_\text{adaptive} = 4.69$.

### Results

| Metric | Naive Raw | Robust Raw | Naive SAE$\to$Raw | Robust SAE$\to$Raw |
|:-------|:---------:|:----------:|:-----------------:|:------------------:|
| Mean $\|\delta\|$ | 2.13 | 6.93 | 0.42 | 2.54 |
| Mean Rashomon Coverage | 40.0% | 100.0% | 9.4% | 13.6% |
| 100% Coverage Rate | 0% | 100% | 0% | 0% |
| Coverage / unit norm | 0.188 | 0.144 | 0.224 | 0.053 |

### Factorial Decomposition

**Space effect** (SAE$\to$raw vs raw): strongly negative ($-30.6$pp naive, $-86.4$pp robust). **Optimization effect** (robust vs naive): positive but asymmetric ($+60$pp raw, $+4.2$pp SAE). **Interaction**: $-55.8$pp. The SAE decode pathway neutralizes most of the benefit of robust optimization.

### Root Cause

The SAE decoder is not the geometric bottleneck (see Section 8). The problem is triple optimization mismatch:
1. The SAE probe has different decision boundaries than the raw probe.
2. The SAE optimizer minimizes $\|\delta_\text{SAE}\|$ rather than $\|\delta_\text{SAE} W_\text{dec}\|$.
3. The SAE Hessian captures uncertainty about SAE probes, not raw probes.

## 8. Decoder Column-Space Diagnostic

### Motivation

The 4-way factorial (Section 7) showed SAE-decoded coverage collapses. Is the decoder subspace too small, or is the optimizer pointing in the wrong direction?

### $W_\text{dec}$ Spectral Properties

| Property | Value |
|:---------|------:|
| Shape | [16384, 2304] |
| Numerical rank | 2,304 / 2,304 (**full rank**) |
| Condition number | 19.4 |
| Effective rank (90% / 95% / 99%) | 1,630 / 1,898 / 2,191 |

### Finding

$\text{col}(W_\text{dec}) = \mathbb{R}^{2304}$: the decoder spans the entire raw activation space. Projecting raw-space robust deltas onto the column space is the identity -- zero energy loss, zero coverage loss. The 13.56% coverage from the 4-way prototype is **100% attributable to the SAE-space optimizer**, not subspace limitations.

This sharpens the Section 7 root cause: the column space has every direction needed. The problem is that the SAE-space optimization pipeline (SAE probe + SAE Hessian + adaptive ridge) produces coefficients that, when decoded, point in the wrong direction.

## 9. Feature Decomposition: Intrinsic Density in SAE Coordinates

### Motivation

Since the decoder subspace is not the bottleneck, and the SAE-space optimizer produces wrong coefficients, a natural question is: does the raw-space robust delta have a **sparse representation** in SAE coordinates? If so, we might post-hoc identify which SAE features implement the steering direction.

### Three Decomposition Methods

Given $\delta_\text{robust} \in \mathbb{R}^{2304}$, find $\alpha \in \mathbb{R}^{16384}$ such that $\alpha W_\text{dec} = \delta_\text{robust}$. Since $W_\text{dec}^\top$ has full row rank 2,304, the solution space is a 14,080-dimensional affine subspace -- many exact solutions exist.

| Method | Significant Features | L1/L2 ratio | Top-20 Energy | Reconstruction Error |
|:-------|:--------------------:|:-----------:|:-------------:|:--------------------:|
| $\alpha_\text{pinv}$ (min-$\ell_2$) | 16,292 | 97.2 | 2.9% | $3.2 \times 10^{-14}$ |
| $\alpha_\text{sparse}$ (min-$\ell_1$, FISTA) | 2,433 | 35.6 | 13.6% | $1.0 \times 10^{-3}$ |
| $\Delta z$ (encoder) | 14.7 | 2.1 | 99.6% | 105% |

### Key Findings

1. **Min-$\ell_2$ is maximally dense**: the pseudoinverse spreads energy across the 14,080-dim null space. 16,292 of 16,384 features are significant.

2. **Min-$\ell_1$ is 6.7$\times$ sparser but still distributed**: 2,433 significant features, top-20 captures only 13.6% of energy. Even the mathematically sparsest exact reconstruction requires thousands of features.

3. **Encoder $\Delta z$ is sparse but not exact**: 14.7 mean features, top-5 captures 95.2% energy. But 105% reconstruction error -- the JumpReLU nonlinearity means $\Delta z \cdot W_\text{dec} \neq \delta$. This is "how the SAE perceives the perturbation," not an algebraic decomposition.

4. **Features are example-independent**: Jaccard similarity = 1.000 across all example pairs for both $\alpha_\text{pinv}$ and $\alpha_\text{sparse}$. All robust deltas point along the probe weight vector $w$; the SAE decomposition of this fixed direction yields a fixed feature pattern.

5. **SAE optimizer was nearly orthogonal to the correct answer**: $\cos(\delta_\text{SAE}, \alpha_\text{pinv}) = 0.091$, $\cos(\delta_\text{SAE}, \alpha_\text{sparse}) = 0.081$. Zero overlap in top-20 features.

### Verdict

**Case (b): the decomposition is intrinsically distributed.** The robust delta cannot be explained by a handful of SAE features. Interpretable sparse steering would require $\ell_1$-penalized robust optimization -- explicitly trading Rashomon coverage for sparsity. The Pareto frontier between coverage and sparsity is an open question.

## 10. Limitations and Next Steps

### Updated Limitations (Additions to Section 5)

- **SAE-space steering is not viable in its current form.** SAE-decoded deltas achieve only 9--14% Rashomon coverage. The optimization target mismatch (SAE probe/Hessian vs raw probe/Hessian) is the root cause, not the decoder geometry.
- **The robust delta has no sparse SAE interpretation.** Even under $\ell_1$ minimization, ~2,433 features are needed. Post-hoc decomposition cannot provide interpretability.

### Candidate Next Directions

- **$\ell_1$-penalized robust optimization**: jointly optimize for Rashomon coverage and SAE sparsity. The key open question is the Pareto frontier -- how much coverage must be sacrificed for a 20--50 feature decomposition?
- **Encoder-based analysis**: the encoder $\Delta z$ is extremely sparse (~15 features) and may offer a complementary "perceptual" notion of interpretability, even though it is not algebraically exact.
- **End-to-end generation evaluation**: inject robust $\delta$ into the Gemma-2 forward pass and measure safety (refusal rate) alongside fluency (perplexity, coherence).
- **Multi-layer analysis**: repeat at layers 10, 15, 20 to study how the Rashomon effect and steering cost vary with depth.
- **Scale to full dataset**: move from 2,000 to full BeaverTails (~330k) to assess whether the Rashomon effect persists or diminishes with more data.

---

*Appendix: Full per-probe loss metrics are available in `outputs/probe_loss_analysis.csv` (51 rows, 11 columns). Hessian eigenvalue spectrum details are in `outputs/hessian_layer10.pt`. SAE steering prototype results are in `outputs/sae_steering_prototype_report.txt`. Decoder diagnostic results are in `outputs/decoder_subspace_diagnostic.txt`. Feature decomposition results are in `outputs/sae_feature_decomposition_report.txt`.*
