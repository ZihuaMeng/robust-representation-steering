"""SAE integration pipeline: encode activations, train probe, run AWP, compare."""

import sys
import os
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sae_utils import load_gemma_scope_sae, verify_sae, encode_activations
from probe import train_probe, evaluate_probe
from awp import run_awp_rashomon, compute_val_loss

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
ACTIVATIONS_PATH = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10.pt")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_probe_layer10.pt")
SAE_ACTIVATIONS_PATH = os.path.join(OUTPUT_DIR, "beavertails_sae_activations_layer10.pt")
SAE_PROBE_PATH = os.path.join(OUTPUT_DIR, "sae_baseline_probe_layer10.pt")
SAE_RASHOMON_DIR = os.path.join(OUTPUT_DIR, "rashomon_sae")
RAW_RASHOMON_DIR = os.path.join(OUTPUT_DIR, "rashomon")


def predict(weight, bias, X):
    logits = X @ weight + bias
    return (logits > 0).int()


def compute_rashomon_diagnostics(probes, baseline_w, baseline_b, test_X, test_y):
    """Compute Hamming/cosine matrices and per-probe metrics. Returns summary dict."""
    y_true = test_y.numpy()
    all_weights = [baseline_w]
    all_biases = [baseline_b]
    for p in probes:
        all_weights.append(p["weight"])
        all_biases.append(p["bias"])

    n_total = len(all_weights)
    all_preds = []
    per_probe_metrics = []

    for idx in range(n_total):
        preds = predict(all_weights[idx], all_biases[idx], test_X)
        all_preds.append(preds)
        y_pred = preds.numpy()
        per_probe_metrics.append({
            "probe": "baseline" if idx == 0 else f"awp_{idx}",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    preds_matrix = torch.stack(all_preds)

    # Pairwise Hamming
    hamming = torch.zeros(n_total, n_total)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d = (preds_matrix[i] != preds_matrix[j]).float().mean().item()
            hamming[i, j] = d
            hamming[j, i] = d

    # Pairwise cosine similarity on [w; b]
    param_vecs = []
    for idx in range(n_total):
        param_vecs.append(torch.cat([all_weights[idx], all_biases[idx].unsqueeze(0)]))
    param_matrix = torch.stack(param_vecs)
    cosine = F.cosine_similarity(
        param_matrix.unsqueeze(1), param_matrix.unsqueeze(0), dim=2,
    )

    mask = ~torch.eye(n_total, dtype=torch.bool)
    h_vals = hamming[mask]
    c_vals = cosine[mask]
    accs = [m["accuracy"] for m in per_probe_metrics]
    f1s = [m["f1"] for m in per_probe_metrics]

    return {
        "hamming": hamming,
        "cosine": cosine,
        "per_probe_metrics": per_probe_metrics,
        "hamming_mean": h_vals.mean().item(),
        "hamming_std": h_vals.std().item(),
        "hamming_min": h_vals.min().item(),
        "hamming_max": h_vals.max().item(),
        "cosine_mean": c_vals.mean().item(),
        "cosine_std": c_vals.std().item(),
        "cosine_min": c_vals.min().item(),
        "cosine_max": c_vals.max().item(),
        "acc_min": min(accs),
        "acc_max": max(accs),
        "f1_min": min(f1s),
        "f1_max": max(f1s),
        "n_total": n_total,
        "labels": ["baseline"] + [f"awp_{i+1}" for i in range(n_total - 1)],
    }


def save_rashomon_artifacts(diag, probes, output_dir):
    """Save standard rashomon output files."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(probes, os.path.join(output_dir, "rashomon_probes.pt"))

    labels = diag["labels"]
    n = diag["n_total"]

    with open(os.path.join(output_dir, "hamming_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, lbl in enumerate(labels):
            w.writerow([lbl] + [f"{diag['hamming'][i,j].item():.4f}" for j in range(n)])

    with open(os.path.join(output_dir, "cosine_similarity_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, lbl in enumerate(labels):
            w.writerow([lbl] + [f"{diag['cosine'][i,j].item():.6f}" for j in range(n)])

    with open(os.path.join(output_dir, "per_probe_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["probe", "accuracy", "precision", "recall", "f1"])
        w.writeheader()
        for m in diag["per_probe_metrics"]:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m.items()})

    report = format_report_section("SAE SPACE", diag)
    with open(os.path.join(output_dir, "rashomon_report.txt"), "w") as f:
        f.write(report)


def format_report_section(label, d):
    lines = [
        f"--- {label} Rashomon Diagnostics ---",
        f"  Probes: {d['n_total']} (1 baseline + {d['n_total']-1} AWP)",
        f"  Hamming:  mean={d['hamming_mean']:.4f} std={d['hamming_std']:.4f} "
        f"min={d['hamming_min']:.4f} max={d['hamming_max']:.4f}",
        f"  Cosine:   mean={d['cosine_mean']:.6f} std={d['cosine_std']:.6f} "
        f"min={d['cosine_min']:.6f} max={d['cosine_max']:.6f}",
        f"  Acc range:  [{d['acc_min']:.4f}, {d['acc_max']:.4f}]",
        f"  F1 range:   [{d['f1_min']:.4f}, {d['f1_max']:.4f}]",
        "",
    ]
    return "\n".join(lines)


def main():
    t_start = time.time()

    # ── Phase 1: Load SAE ────────────────────────────────────────────────
    print("=" * 65)
    print("PHASE 1: Load Gemma Scope SAE (Layer 10, 16k width)")
    print("=" * 65)
    sae = load_gemma_scope_sae(layer=10, width="16k", l0=77)
    d_sae = sae.W_enc.shape[1]

    # Verify d_model
    d_model = sae.W_enc.shape[0]
    assert d_model == 2304, f"d_model mismatch: expected 2304, got {d_model}"
    print(f"d_model={d_model} confirmed. SAE width (d_sae)={d_sae}")

    # ── Load cached activations ──────────────────────────────────────────
    print("\nLoading cached raw activations ...")
    data = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=False)
    train_X, train_y = data["train_X"], data["train_y"]
    test_X, test_y = data["test_X"], data["test_y"]
    print(f"Raw activations — train: {train_X.shape}, test: {test_X.shape}")

    # Reconstruction verification on 5 samples
    verify_sae(sae, train_X[:5])

    # ── Phase 2: Transform activations ───────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 2: Transform activations to SAE feature space")
    print("=" * 65)

    print("\nEncoding train activations ...")
    sae_train_X = encode_activations(sae, train_X)
    print("\nEncoding test activations ...")
    sae_test_X = encode_activations(sae, test_X)

    # Save SAE activations
    torch.save({
        "train_X": sae_train_X, "train_y": train_y,
        "test_X": sae_test_X, "test_y": test_y,
        "sae_width": d_sae,
    }, SAE_ACTIVATIONS_PATH)
    print(f"\nSAE activations saved to {SAE_ACTIVATIONS_PATH}")

    # ── Phase 3: Train SAE-space baseline probe ──────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3: Train SAE-space baseline probe")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe, loss_log = train_probe(
        sae_train_X, train_y, input_dim=d_sae, device=device,
    )
    sae_metrics = evaluate_probe(probe, sae_test_X, test_y, device=device)

    # Save SAE probe
    w = probe.linear.weight.detach().squeeze(0)
    b = probe.linear.bias.detach().squeeze(0)
    final_loss = loss_log[-1][1]
    torch.save({
        "weight": w, "bias": b,
        "val_loss": final_loss, "test_metrics": sae_metrics,
    }, SAE_PROBE_PATH)
    print(f"SAE probe saved to {SAE_PROBE_PATH}")

    # ── Phase 4: AWP Rashomon on SAE features ────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 4: AWP Rashomon enumeration (SAE space)")
    print("=" * 65)

    seed = 42
    n_val = int(len(train_y) * 0.2)
    perm = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    sae_val_X = sae_train_X[val_idx]
    sae_val_y = train_y[val_idx].float()
    sae_sub_train_X = sae_train_X[tr_idx]
    sae_sub_train_y = train_y[tr_idx].float()

    sae_probes, sae_baseline_vl = run_awp_rashomon(
        sae_sub_train_X, sae_sub_train_y, sae_val_X, sae_val_y,
        w, b,
        n_candidates=50, epsilon=0.15,
        sgd_steps=400, lr=1e-4, momentum=0.9,
        max_retries=3, seed=seed, device=device,
    )

    # Diagnostics
    sae_diag = compute_rashomon_diagnostics(sae_probes, w, b, sae_test_X, test_y)
    save_rashomon_artifacts(sae_diag, sae_probes, SAE_RASHOMON_DIR)
    print(f"\nSAE Rashomon artifacts saved to {SAE_RASHOMON_DIR}/")

    # ── Phase 5: Comparative report ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 5: Raw Space vs SAE Space Comparison")
    print("=" * 65)

    # Load raw-space results
    raw_baseline = torch.load(BASELINE_PATH, map_location="cpu", weights_only=False)
    raw_metrics = raw_baseline["test_metrics"]

    # Load raw rashomon diagnostics (recompute from saved probes)
    raw_probes_list = torch.load(
        os.path.join(RAW_RASHOMON_DIR, "rashomon_probes.pt"),
        map_location="cpu", weights_only=False,
    )
    raw_diag = compute_rashomon_diagnostics(
        raw_probes_list, raw_baseline["weight"], raw_baseline["bias"], test_X, test_y,
    )

    # Build comparison table
    table = f"""
{'='*65}
RASHOMON COMPARISON REPORT: Raw Space vs SAE Space
{'='*65}

| Diagnostic              | Raw Space       | SAE Space       |
|-------------------------|-----------------|-----------------|
| Input dimension         | 2304            | {d_sae:<15d} |
| Baseline Accuracy       | {raw_metrics['accuracy']:.2%}          | {sae_metrics['accuracy']:.2%}          |
| Baseline AUROC          | {raw_metrics['auroc']:.2%}          | {sae_metrics['auroc']:.2%}          |
| Baseline F1             | {raw_metrics['f1']:.2%}          | {sae_metrics['f1']:.2%}          |
| Mean Hamming distance   | {raw_diag['hamming_mean']:.4f}          | {sae_diag['hamming_mean']:.4f}          |
| Max Hamming distance    | {raw_diag['hamming_max']:.4f}          | {sae_diag['hamming_max']:.4f}          |
| Min Hamming distance    | {raw_diag['hamming_min']:.4f}          | {sae_diag['hamming_min']:.4f}          |
| Cosine sim (mean)       | {raw_diag['cosine_mean']:.6f}       | {sae_diag['cosine_mean']:.6f}       |
| Cosine sim (min)        | {raw_diag['cosine_min']:.6f}       | {sae_diag['cosine_min']:.6f}       |
| Accuracy range          | [{raw_diag['acc_min']:.2%}, {raw_diag['acc_max']:.2%}] | [{sae_diag['acc_min']:.2%}, {sae_diag['acc_max']:.2%}] |
| F1 range                | [{raw_diag['f1_min']:.2%}, {raw_diag['f1_max']:.2%}] | [{sae_diag['f1_min']:.2%}, {sae_diag['f1_max']:.2%}] |
| Probes generated        | {len(raw_probes_list)}/50           | {len(sae_probes)}/50           |

Raw-space Rashomon effect:  mean Hamming = {raw_diag['hamming_mean']:.2%}  {"CONFIRMED" if raw_diag['hamming_mean'] > 0.05 else "WEAK"}
SAE-space Rashomon effect:  mean Hamming = {sae_diag['hamming_mean']:.2%}  {"CONFIRMED" if sae_diag['hamming_mean'] > 0.05 else "WEAK"}
"""
    print(table)

    # Save comparison report
    with open(os.path.join(OUTPUT_DIR, "rashomon_comparison_report.txt"), "w") as f:
        f.write(table)
    print(f"Comparison report saved to {OUTPUT_DIR}/rashomon_comparison_report.txt")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("SAE PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
