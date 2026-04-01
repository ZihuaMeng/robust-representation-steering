# Rashomon-Robust Representation Steering

**Safety probes with >99.9% cosine similarity disagree on up to 90% of predictions. We derive a closed-form fix -- and systematically show that SAE features amplify probe multiplicity but cannot serve as a steering basis.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: Gemma-2 2B](https://img.shields.io/badge/Model-Gemma--2%202B-4285F4.svg)](https://huggingface.co/google/gemma-2-2b)

---

## Key Results

| | |
|:---|:---|
| **Rashomon Effect** | 19.5% mean prediction disagreement (raw), **44.5%** (SAE) among probes with equivalent loss |
| **Robust Steering** | **100% Rashomon coverage** vs 40% naive, at 3.25$\times$ norm cost |
| **Generalization** | Pearson $r = 0.997$ between val and test loss -- the Rashomon bound transfers to held-out data |
| **SAE-Decoded Steering** | **~13% coverage** across all tested variants -- full SAE, active-subspace, and K-sweep |
| **Decoder Diagnostic** | Full rank (2,304/2,304) -- expressivity is not the bottleneck |
| **Feature Decomposition** | Robust delta is intrinsically SAE-dense: **2,433 features** even under $\ell_1$ minimization |

---

## The Rashomon Effect in LLM Safety Probes

Linear probes on LLM residual activations are a standard tool for safety classification: train $f(x) = w^\top x + b$ on labeled activations and use $w$ as the steering direction. But **the probe is not unique**. Many weight configurations achieve near-identical loss yet make very different predictions.

We formalize this via the **Rashomon set** at tolerance $\varepsilon$:

$$\mathcal{R}(\varepsilon) = \{(w, b) \;:\; L_{\text{val}}(w, b) \leq L_{\text{val}}(\hat{w}, \hat{b}) + \varepsilon\}$$

Using Adversarial Weight Perturbation (AWP), we enumerate 50 probes within $\mathcal{R}(\varepsilon = 0.15)$ on Gemma-2 2B layer-10 residuals (BeaverTails, $n = 2000$, balanced). The results reveal a striking paradox:

| Metric | Raw Space ($d = 2304$) | SAE Space ($d = 16384$) |
|--------|:----------------------:|:-----------------------:|
| Mean Hamming Distance | 19.5% | 44.5% |
| Max Hamming Distance | 43.8% | 90.5% |
| Cosine Similarity (mean) | 0.9999 | 0.9997 |
| Accuracy Range | 68.5 -- 78.0% | 50.8 -- 69.5% |
| F1 Range | 59.6 -- 81.0% | 6.6 -- 69.2% |

Probes that are nearly parallel in weight space ($\cos > 0.999$) disagree on up to **43.8%** of test predictions in raw space and **90.5%** in SAE space. Tiny parameter perturbations cause large behavioral divergence on examples near the decision boundary.

![Distribution of Probe Disagreement: Raw vs SAE Space](assets/figures/fig_hamming_distribution_comparison.png)

**Why does SAE projection amplify this?** The Gemma Scope SAE maps activations into a 16,384-dimensional space with ~99.9% sparsity (~17 active features per example). This creates massive null-space freedom: the $\varepsilon$-ball around the baseline probe encompasses a far larger volume of functionally-distinct directions. The bimodal SAE-space distribution shows probe pairs either agree well or disagree on ~85% of predictions -- two coherent but opposing classification strategies coexist within the Rashomon set.

## Robust Steering

If many near-optimal probes exist, steering along the baseline direction alone is insufficient. We derive a **worst-case robust** perturbation via the loss Hessian.

**Formulation.** Let $\hat{\theta} = [\hat{w}; \hat{b}]$, let $H$ be the BCE Hessian at $\hat{\theta}$, and let $\tilde{x} = [x_{\text{new}}; 1]$. The robust steering problem is:

$$\min_{\delta} \|\delta\|^2 \quad \text{s.t.} \quad \hat{\theta}^\top \tilde{x}_{\text{new}} - \sqrt{2\varepsilon \cdot \tilde{x}_{\text{new}}^\top H^{-1} \tilde{x}_{\text{new}}} \;\geq\; t$$

The correction term $\sqrt{2\varepsilon \cdot \tilde{x}^\top H^{-1} \tilde{x}}$ accounts for the maximum logit reduction any Rashomon-feasible probe could produce -- guaranteeing safety under the **worst-case** probe in the ellipsoid.

| | Naive Steering | Robust Steering |
|:---|:---:|:---:|
| Mean $\|\delta\|$ | 2.13 | 6.93 |
| Mean Rashomon Coverage | 40% (20/50) | **100% (50/50)** |
| 100% Coverage Rate | 0% of examples | **100% of examples** |

A **3.25$\times$ norm penalty** buys universal safety coverage. Every unsafe example is steered past the threshold for all 50 probes simultaneously.

![Steering Coverage: Naive vs Robust](assets/figures/fig_steering_coverage_comparison.png)

**Hessian diagnostics.** Condition number = 101 after adaptive ridge regularization (eigenvalue range: $[16.2, 1639.7]$).

## Generalization Validation

A key concern: probes are admitted to $\mathcal{R}(\varepsilon)$ by a **validation** loss bound. Do they generalize to held-out test data?

We compute BCE loss on three disjoint splits (train subset $n = 1280$, val $n = 320$, test $n = 400$) for all 51 probes:

| Statistic | Value |
|:----------|------:|
| Mean Val-Test Gap (50 AWP probes) | $0.139 \pm 0.003$ |
| Baseline Val-Test Gap | 0.135 |
| Excess gap (AWP $-$ baseline) | 0.003 |
| Pearson $r$(val loss, test loss) | **0.997** |

The ~0.14 val-test gap is a **constant offset** from the train/test distribution split, not probe-specific overfitting: the baseline shows the same gap, the standard deviation across probes is only 0.003, and val-test losses are near-perfectly rank-correlated. The validation-loss bound transfers reliably to held-out data.

## SAE-Space Steering: A Systematic Investigation

Can sparse autoencoders provide a more interpretable basis for robust steering? We test this hypothesis through a sequence of experiments that progressively narrow the diagnosis -- from Hessian feasibility, through end-to-end evaluation, to three convergent diagnostics that identify the root cause.

### SAE-Space Hessian

The SAE-space Hessian ($d = 16{,}384$) has a fundamentally different spectral structure from raw space. With $N_\text{train} = 1{,}600 \ll d = 16{,}384$, the data-dependent term has rank $\leq 1{,}600$ (numerical rank 931, effective rank 120 at 99% spectral mass). We invert it exactly via the Woodbury identity with adaptive ridge ($\lambda = 4.69$, condition $\approx 101$).

| Property | Raw Space ($d = 2304$) | SAE Space ($d = 16384$) |
|:---------|:----------------------:|:-----------------------:|
| Hessian dimension | 2,305 | 16,385 |
| Numerical rank | 2,305 (full) | 931 |
| Effective rank (99%) | full | 120 |
| Inversion method | Dense | Woodbury (~22 ms/matvec) |

### End-to-End SAE Steering

We run a 2$\times$2 factorial -- {naive, robust} $\times$ {raw, SAE$\to$raw} -- evaluating all four strategies against the same 50 raw-space Rashomon probes.

| Metric | Naive Raw | Robust Raw | Naive SAE$\to$Raw | Robust SAE$\to$Raw |
|:-------|:---------:|:----------:|:-----------------:|:------------------:|
| Mean $\|\delta\|$ | 2.13 | 6.93 | 0.42 | 2.54 |
| Mean Rashomon Coverage | 40.0% | **100.0%** | 9.4% | 13.6% |
| 100% Coverage Rate | 0% | **100%** | 0% | 0% |

**Factorial decomposition.** The space effect (SAE$\to$raw vs raw) is strongly negative: $-30.6$pp naive, $-86.4$pp robust. The optimization effect (robust vs naive) is positive but asymmetric: $+60$pp in raw space, $+4.2$pp through SAE decode. The interaction ($-55.8$pp) shows that robust optimization is far more valuable in raw space -- the SAE decode pathway neutralizes most of the gain.

### Three Convergent Diagnostics

Three independent experiments rule out three candidate explanations for the ~13% coverage ceiling, converging on a structural root cause.

#### Decoder Column-Space Is Full-Rank

The SAE decoder $W_\text{dec} \in \mathbb{R}^{16384 \times 2304}$ has **full numerical rank** (2,304/2,304), condition number 19.4. Its column space spans the entire raw activation space: $\text{col}(W_\text{dec}) = \mathbb{R}^{2304}$. Projecting raw-space robust deltas onto the column space preserves 100% of energy and 100% of coverage.

**The bottleneck is not decoder expressivity.**

#### Active-Subspace Restriction Does Not Help

If the 16,384-dimensional SAE space contains ~13,000 always-inactive features, perhaps restricting to the active subspace would improve conditioning and coverage. We identify the $d_\text{active} = 3{,}209$ features active at least once across the dataset ($K = 1$ threshold) and repeat the full pipeline -- probe, Hessian, robust optimization -- in this restricted subspace.

| | Raw | SAE (full) | SAE (active, $K = 1$) |
|:--|:---:|:---:|:---:|
| Naive coverage | 40.0% | 9.4% | 9.4% |
| Robust coverage | **100.0%** | 13.6% | 13.3% |
| Robust $\|\delta\|$ | 6.93 | 2.54 | 2.47 |

A sweep over activation thresholds ($K = 1, 5, 10$; subspace dimensions $3{,}209 \to 382$) shows coverage pinned at 10--14% regardless of subspace size:

| $K$ | $d_\text{active}$ | Robust Coverage | Robust $\|\delta\|$ |
|:---:|:---------:|:---:|:---:|
| 1 | 3,209 | 12.8% | 2.22 |
| 5 | 1,117 | 13.4% | 3.76 |
| 10 | 382 | 10.4% | 3.05 |

**The bottleneck is not null-space dimensionality.**

#### Robust Delta Is Intrinsically SAE-Dense

If the robust delta cannot be computed in SAE space, can it at least be *explained* there? We decompose raw-space robust deltas into SAE features via three methods:

| Method | Significant Features | Top-20 Energy | Reconstruction Error |
|:-------|:--------------------:|:-------------:|:--------------------:|
| Min-$\ell_2$ (pseudoinverse) | 16,292 | 2.9% | $3.2 \times 10^{-14}$ |
| Min-$\ell_1$ (FISTA) | **2,433** | 13.6% | $1.0 \times 10^{-3}$ |
| Encoder $\Delta z$ | 14.7 | 99.6% | 105% (not exact) |

Even the sparsest exact decomposition (min-$\ell_1$) requires ~2,433 features, with top-20 capturing only 13.6% of energy. Feature identity is example-independent: both algebraic methods target identical features across all examples (Jaccard = 1.000). The SAE-space optimizer's $\delta_\text{SAE}$ was nearly orthogonal to the correct decomposition (cosine $\approx 0.09$, zero top-20 feature overlap).

**The bottleneck is not sparsity potential -- the robust delta is a fundamentally high-dimensional object in SAE coordinates.**

### Root Cause

The three diagnostics converge: the decoder can express any raw-space direction, the active subspace retains the relevant features, and even the sparsest exact decomposition spans thousands of features. The ~13% coverage ceiling originates from **structural misalignment between SAE-space and raw-space probe decision boundaries**. The SAE probe converges to a different local minimum, the SAE Hessian captures uncertainty about the wrong set of probes, and the optimizer minimizes $\|\delta_\text{SAE}\|$ rather than $\|\delta_\text{SAE} W_\text{dec}\|$.

Interpretable sparse steering would require an $\ell_1$-penalized robust optimization that explicitly trades Rashomon coverage for sparsity -- a genuinely different optimization objective, not a change in analysis.

A fluency pilot (5 examples $\times$ 3 conditions) confirms that steering at norms up to ~10 preserves text coherence in the Gemma-2 2B base model, though meaningful refusal evaluation requires an instruction-tuned model.

## Method Overview

```
BeaverTails (1000 safe + 1000 unsafe)
  -> Gemma-2 2B, Layer 10 residual stream (d=2304)
    -> Linear probe (BCE + L2, lambda=0.01)
      -> AWP: enumerate Rashomon set (50 probes, eps=0.15)
        -> Pairwise Hamming analysis
    -> Gemma Scope SAE (JumpReLU, 16k features)
      -> Same pipeline in SAE space (amplified Rashomon effect)
    -> BCE Hessian at baseline (d+1 x d+1)
      -> Robust delta via Hessian-scaled bisection
        -> Coverage comparison: naive vs robust (100% coverage)
    -> SAE-space Hessian (Woodbury, rank 931)
      -> 4-way factorial: {naive, robust} x {raw, SAE->raw}
        -> Decoder column-space diagnostic (full rank)
        -> Feature decomposition (L2 / L1 / encoder)
          -> Verdict: intrinsically SAE-dense
        -> Active-subspace steering (d=3209, K-sweep)
          -> Verdict: null-space not the bottleneck
          -> Fluency pilot: steering preserves coherence
```

## Repository Structure

```
robust-representation-steering/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── data_pipeline.py          # BeaverTails loading & activation extraction
│   ├── probe.py                  # Linear probe training & evaluation
│   ├── awp.py                    # Adversarial weight perturbation engine
│   ├── hessian.py                # BCE Hessian computation
│   ├── steering.py               # Naive & robust delta solvers
│   └── sae_utils.py              # Gemma Scope SAE utilities
├── scripts/
│   ├── 01_train_baseline_probe.py    # Data -> activations -> probe
│   ├── 02_run_awp_rashomon.py        # AWP Rashomon enumeration (raw space)
│   ├── 03_run_sae_pipeline.py        # SAE projection + AWP (SAE space)
│   ├── 04_run_robust_steering.py     # Robust vs naive steering comparison
│   ├── 05_analyze_probe_losses.py    # Train/val/test generalization analysis
│   ├── 06_generate_figures.py        # Publication figures
│   ├── 07_sae_hessian_feasibility.py # SAE-space Hessian spectral analysis
│   ├── 08_sae_steering_prototype.py  # 4-way factorial steering comparison
│   ├── 09_decoder_subspace_diagnostic.py  # Decoder column-space diagnostic
│   ├── 10_sae_feature_decomposition.py   # Feature decomposition analysis
│   ├── 11_active_subspace_steering.py    # Active-subspace SAE steering pipeline
│   ├── verify_env.py                 # Environment verification utility
│   └── verify_model_and_activations.py   # Model/activation verification
├── outputs/                          # Experiment artifacts (.pt files gitignored)
│   ├── technical_summary.md          # Full technical report
│   ├── implementation_memo.md        # Math verification & code walkthrough
│   ├── implementation_report.md      # Advisor-facing pipeline report
│   ├── probe_loss_analysis.csv       # Per-probe loss table (51 rows)
│   ├── steering_comparison_report.txt
│   ├── sae_steering_prototype_report.txt
│   ├── decoder_subspace_diagnostic.txt
│   ├── sae_feature_decomposition_report.txt
│   ├── active_subspace_steering_report.txt
│   ├── fluency_samples.txt
│   ├── rashomon/                     # Raw-space Rashomon results
│   └── rashomon_sae/                 # SAE-space Rashomon results
└── assets/figures/                   # Publication-quality figures
```

## Reproduction

### Environment

```bash
conda create -n sae_steering python=3.10 -y
conda activate sae_steering
pip install -r requirements.txt
```

### Run the Pipeline

Scripts 01--06 reproduce the core Rashomon and robust steering results. Scripts 07--11 reproduce the SAE diagnostic experiments. Steps 01--04 and 07--11 require GPU. Steps 05--06 run on CPU.

```bash
# Core pipeline
python scripts/01_train_baseline_probe.py      # Extract activations & train probe
python scripts/02_run_awp_rashomon.py           # Enumerate Rashomon set (raw)
python scripts/03_run_sae_pipeline.py           # SAE projection + Rashomon (SAE)
python scripts/04_run_robust_steering.py        # Robust vs naive steering
python scripts/05_analyze_probe_losses.py       # Generalization analysis (CPU)
python scripts/06_generate_figures.py           # Generate figures (CPU)

# SAE diagnostic experiments
python scripts/07_sae_hessian_feasibility.py    # SAE-space Hessian characterization
python scripts/08_sae_steering_prototype.py     # 4-way factorial comparison
python scripts/09_decoder_subspace_diagnostic.py  # Decoder column-space test
python scripts/10_sae_feature_decomposition.py  # Feature decomposition analysis
python scripts/11_active_subspace_steering.py   # Active-subspace SAE steering
```

## Citation

```bibtex
@software{meng2026rashomon,
  title   = {Rashomon-Robust Representation Steering for LLM Safety},
  author  = {Meng, Zihua},
  year    = {2026},
  url     = {https://github.com/ZihuaMeng/robust-representation-steering}
}

@article{team2024gemma,
  title   = {Gemma 2: Improving Open Language Models at a Practical Size},
  author  = {Gemma Team, Google DeepMind},
  year    = {2024},
  journal = {arXiv preprint arXiv:2408.00118}
}

@article{lieberum2024gemma,
  title   = {Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2},
  author  = {Lieberum, Tom and Rajamanoharan, Senthooran and Conmy, Arthur and others},
  year    = {2024},
  journal = {arXiv preprint arXiv:2408.05147}
}

@inproceedings{ji2024beavertails,
  title     = {BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset},
  author    = {Ji, Jiaming and Liu, Mickel and Dai, Josef and others},
  booktitle = {NeurIPS},
  year      = {2024}
}

@inproceedings{wu2020adversarial,
  title     = {Adversarial Weight Perturbation Helps Robust Generalization},
  author    = {Wu, Dongxian and Xia, Shu-Tao and Wang, Yisen},
  booktitle = {NeurIPS},
  year      = {2020}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
