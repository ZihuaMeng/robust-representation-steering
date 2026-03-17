"""AWP Rashomon set enumeration for safety probes on BeaverTails activations."""

import sys
import os
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from awp import run_awp_rashomon, compute_val_loss

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RASHOMON_DIR = os.path.join(OUTPUT_DIR, "rashomon")
ACTIVATIONS_PATH = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10.pt")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_probe_layer10.pt")


def predict(weight, bias, X):
    """Return (probs, preds) for a linear probe on X."""
    logits = X @ weight + bias
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int()
    return probs, preds


def main():
    t_start = time.time()
    os.makedirs(RASHOMON_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading cached activations and baseline probe ...")
    data = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=True)
    train_X, train_y = data["train_X"], data["train_y"]
    test_X, test_y = data["test_X"], data["test_y"]

    baseline = torch.load(BASELINE_PATH, map_location="cpu", weights_only=True)
    baseline_w, baseline_b = baseline["weight"], baseline["bias"]
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"Baseline test metrics: {baseline['test_metrics']}")

    # ── Validation split (20% of training data) ─────────────────────────
    seed = 42
    n_val = int(len(train_y) * 0.2)
    perm = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    val_X, val_y = train_X[val_idx], train_y[val_idx].float()
    sub_train_X, sub_train_y = train_X[tr_idx], train_y[tr_idx].float()
    print(f"AWP train subset: {sub_train_X.shape[0]}, Val: {val_X.shape[0]}")

    # ── Run AWP ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning AWP Rashomon enumeration on {device} ...")
    print("=" * 65)

    probes, baseline_val_loss = run_awp_rashomon(
        sub_train_X, sub_train_y, val_X, val_y,
        baseline_w, baseline_b,
        n_candidates=50, epsilon=0.15,
        sgd_steps=400, lr=1e-4, momentum=0.9,
        max_retries=3, seed=seed, device=device,
    )

    print("=" * 65)
    n_probes = len(probes)
    if n_probes == 0:
        print("No probes generated. Exiting.")
        return

    # ── Collect all probes (baseline = row 0) ────────────────────────────
    all_weights = [baseline_w]  # index 0 = baseline
    all_biases = [baseline_b]
    for p in probes:
        all_weights.append(p["weight"])
        all_biases.append(p["bias"])

    n_total = len(all_weights)  # 1 + n_probes

    # ── Per-probe test predictions and metrics ───────────────────────────
    print(f"\nEvaluating {n_total} probes on test set ({test_X.shape[0]} examples) ...")
    all_preds = []
    all_probs_list = []
    per_probe_metrics = []

    y_true = test_y.numpy()
    for idx in range(n_total):
        probs, preds = predict(all_weights[idx], all_biases[idx], test_X)
        all_preds.append(preds)
        all_probs_list.append(probs)
        y_pred = preds.numpy()
        metrics = {
            "probe": "baseline" if idx == 0 else f"awp_{idx}",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        per_probe_metrics.append(metrics)

    preds_matrix = torch.stack(all_preds)  # [n_total, n_test]

    # ── Pairwise Hamming distance matrix ─────────────────────────────────
    hamming = torch.zeros(n_total, n_total)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            dist = (preds_matrix[i] != preds_matrix[j]).float().mean().item()
            hamming[i, j] = dist
            hamming[j, i] = dist

    # ── Pairwise cosine similarity matrix ────────────────────────────────
    # Concatenate [w; b] for each probe
    param_vecs = []
    for idx in range(n_total):
        param_vecs.append(torch.cat([all_weights[idx], all_biases[idx].unsqueeze(0)]))
    param_matrix = torch.stack(param_vecs)  # [n_total, 2305]
    cosine = F.cosine_similarity(
        param_matrix.unsqueeze(1), param_matrix.unsqueeze(0), dim=2,
    )  # [n_total, n_total]

    # ── Summary statistics (exclude diagonal) ────────────────────────────
    mask = ~torch.eye(n_total, dtype=torch.bool)
    hamming_vals = hamming[mask]
    cosine_vals = cosine[mask]

    accs = [m["accuracy"] for m in per_probe_metrics]
    f1s = [m["f1"] for m in per_probe_metrics]

    report_lines = [
        "=" * 65,
        "RASHOMON SET ANALYSIS REPORT",
        "=" * 65,
        "",
        f"Baseline val loss: {baseline_val_loss:.4f}",
        f"Epsilon: 0.15",
        f"Rashomon bound: {baseline_val_loss + 0.15:.4f}",
        f"Probes generated: {n_probes}/50",
        "",
        "--- Pairwise Hamming Distance (fraction of test disagreements) ---",
        f"  Mean: {hamming_vals.mean().item():.4f}",
        f"  Std:  {hamming_vals.std().item():.4f}",
        f"  Min:  {hamming_vals.min().item():.4f}",
        f"  Max:  {hamming_vals.max().item():.4f}",
        "",
        "--- Pairwise Cosine Similarity (on [w; b] vectors) ---",
        f"  Mean: {cosine_vals.mean().item():.6f}",
        f"  Std:  {cosine_vals.std().item():.6f}",
        f"  Min:  {cosine_vals.min().item():.6f}",
        f"  Max:  {cosine_vals.max().item():.6f}",
        "",
        "--- Probe Performance Range ---",
        f"  Accuracy: [{min(accs):.4f}, {max(accs):.4f}]",
        f"  F1:       [{min(f1s):.4f}, {max(f1s):.4f}]",
        "",
    ]

    mean_hamming = hamming_vals.mean().item()
    if mean_hamming > 0.05:
        report_lines.append(
            f"RASHOMON EFFECT CONFIRMED: mean Hamming distance {mean_hamming:.2%} > 5%"
        )
        report_lines.append(
            "  => Non-trivial probe multiplicity exists for safety classification."
        )
    else:
        report_lines.append(
            f"RASHOMON EFFECT WEAK: mean Hamming distance {mean_hamming:.2%} <= 5%"
        )
        report_lines.append(
            "  => Probe multiplicity may be limited. Consider increasing epsilon or changing layer."
        )
    report_lines.append("")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # ── Save artifacts ───────────────────────────────────────────────────
    # 1. Probe weights
    torch.save(probes, os.path.join(RASHOMON_DIR, "rashomon_probes.pt"))

    # 2. Report
    with open(os.path.join(RASHOMON_DIR, "rashomon_report.txt"), "w") as f:
        f.write(report_text)

    # 3. Hamming matrix CSV
    labels = ["baseline"] + [f"awp_{i+1}" for i in range(n_probes)]
    with open(os.path.join(RASHOMON_DIR, "hamming_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{hamming[i, j].item():.4f}" for j in range(n_total)])

    # 4. Cosine similarity matrix CSV
    with open(os.path.join(RASHOMON_DIR, "cosine_similarity_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{cosine[i, j].item():.6f}" for j in range(n_total)])

    # 5. Per-probe metrics CSV
    with open(os.path.join(RASHOMON_DIR, "per_probe_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["probe", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        for m in per_probe_metrics:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m.items()})

    print(f"Artifacts saved to {RASHOMON_DIR}/")
    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("RASHOMON PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
