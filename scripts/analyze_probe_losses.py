"""Analyze train/val/test losses for all Rashomon probes.

Resolves the PI's core question: do probes that satisfy the val-loss
Rashomon bound also generalize to the held-out test set?

CPU-only. No GPU required.
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
ACTIVATIONS_PATH = os.path.join(OUTPUT_DIR, "beavertails_activations_layer10.pt")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_probe_layer10.pt")
RASHOMON_PATH = os.path.join(OUTPUT_DIR, "rashomon", "rashomon_probes.pt")
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "probe_loss_analysis.csv")


def bce_loss(weight, bias, X, y):
    """Compute mean BCE loss in float64 for numerical precision."""
    weight = weight.double()
    bias = bias.double()
    X = X.double()
    y = y.double()
    logits = X @ weight + bias
    return F.binary_cross_entropy_with_logits(logits, y, reduction="mean").item()


def l2_regularized_loss(weight, bias, X, y, lam=0.01):
    """BCE + lambda * ||w||^2 (matching AdamW weight_decay=0.01)."""
    weight = weight.double()
    bias = bias.double()
    X = X.double()
    y = y.double()
    logits = X @ weight + bias
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    l2 = lam * (weight ** 2).sum()
    return (bce + l2).item()


def predict(weight, bias, X):
    """Return predictions (0/1) for a linear probe."""
    with torch.no_grad():
        logits = X @ weight + bias
        return (torch.sigmoid(logits) >= 0.5).int()


def main():
    print("=" * 70)
    print("PROBE LOSS ANALYSIS: Train / Val / Test Generalization Check")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLoading activations and probes ...")
    data = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=True)
    train_X, train_y = data["train_X"], data["train_y"]
    test_X, test_y = data["test_X"], data["test_y"]

    baseline = torch.load(BASELINE_PATH, map_location="cpu", weights_only=True)
    baseline_w, baseline_b = baseline["weight"], baseline["bias"]

    probes = torch.load(RASHOMON_PATH, map_location="cpu", weights_only=True)
    print(f"  Train set: {train_X.shape[0]} examples, dim={train_X.shape[1]}")
    print(f"  Test set:  {test_X.shape[0]} examples")
    print(f"  AWP probes loaded: {len(probes)}")

    # ── Reconstruct exact train/val split from AWP ────────────────────
    # Replicating scripts/run_awp_rashomon.py lines 46-51
    seed = 42
    n_val = int(len(train_y) * 0.2)
    perm = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    sub_train_X = train_X[tr_idx]
    sub_train_y = train_y[tr_idx].float()
    val_X = train_X[val_idx]
    val_y = train_y[val_idx].float()
    test_y_f = test_y.float()

    print(f"  AWP train subset: {sub_train_X.shape[0]}, Val subset: {val_X.shape[0]}")
    print(f"  Split seed: {seed}, val fraction: 20%")

    # ── Collect all probes ────────────────────────────────────────────
    all_weights = [baseline_w]
    all_biases = [baseline_b]
    all_labels = ["baseline"]
    for i, p in enumerate(probes):
        all_weights.append(p["weight"])
        all_biases.append(p["bias"])
        all_labels.append(f"awp_{i+1}")

    n_total = len(all_weights)
    print(f"\n  Total probes to analyze: {n_total} (1 baseline + {n_total - 1} AWP)")

    # ── Compute losses and metrics for each probe ─────────────────────
    print("\nComputing losses (float64 precision) ...\n")

    rows = []
    header = [
        "Probe", "Train Loss", "Val Loss", "Test Loss",
        "Train Loss (reg)", "Val Loss (reg)", "Test Loss (reg)",
        "Test Acc", "Test F1", "Val-Test Gap", "Val-Test Gap (reg)",
    ]
    print(f"{'Probe':<12s} {'Train':>10s} {'Val':>10s} {'Test':>10s} "
          f"{'Test Acc':>10s} {'Test F1':>10s} {'V-T Gap':>10s}")
    print("-" * 74)

    for idx in range(n_total):
        w, b = all_weights[idx], all_biases[idx]
        label = all_labels[idx]

        # Raw BCE losses
        train_loss = bce_loss(w, b, sub_train_X, sub_train_y)
        val_loss = bce_loss(w, b, val_X, val_y)
        test_loss = bce_loss(w, b, test_X, test_y_f)

        # Regularized losses
        train_loss_reg = l2_regularized_loss(w, b, sub_train_X, sub_train_y)
        val_loss_reg = l2_regularized_loss(w, b, val_X, val_y)
        test_loss_reg = l2_regularized_loss(w, b, test_X, test_y_f)

        # Test metrics
        preds = predict(w, b, test_X)
        y_true = test_y.numpy()
        y_pred = preds.numpy()
        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, zero_division=0)

        vt_gap = abs(val_loss - test_loss)
        vt_gap_reg = abs(val_loss_reg - test_loss_reg)

        rows.append({
            "Probe": label,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Test Loss": test_loss,
            "Train Loss (reg)": train_loss_reg,
            "Val Loss (reg)": val_loss_reg,
            "Test Loss (reg)": test_loss_reg,
            "Test Acc": test_acc,
            "Test F1": test_f1,
            "Val-Test Gap": vt_gap,
            "Val-Test Gap (reg)": vt_gap_reg,
        })

        print(f"{label:<12s} {train_loss:>10.6f} {val_loss:>10.6f} {test_loss:>10.6f} "
              f"{test_acc:>10.4f} {test_f1:>10.4f} {vt_gap:>10.6f}")

    # ── Aggregate statistics ──────────────────────────────────────────
    awp_rows = [r for r in rows if r["Probe"] != "baseline"]
    baseline_row = rows[0]

    awp_val_losses = np.array([r["Val Loss"] for r in awp_rows])
    awp_test_losses = np.array([r["Test Loss"] for r in awp_rows])
    awp_vt_gaps = np.array([r["Val-Test Gap"] for r in awp_rows])
    awp_vt_gaps_reg = np.array([r["Val-Test Gap (reg)"] for r in awp_rows])

    # Pearson correlation between val and test loss
    pearson_r = np.corrcoef(awp_val_losses, awp_test_losses)[0, 1]

    # Also compute for regularized losses
    awp_val_reg = np.array([r["Val Loss (reg)"] for r in awp_rows])
    awp_test_reg = np.array([r["Test Loss (reg)"] for r in awp_rows])
    pearson_r_reg = np.corrcoef(awp_val_reg, awp_test_reg)[0, 1]

    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS (50 AWP probes)")
    print("=" * 70)

    print(f"\n--- Raw BCE Loss ---")
    print(f"  Val Loss:   mean={awp_val_losses.mean():.6f}, std={awp_val_losses.std():.6f}")
    print(f"  Test Loss:  mean={awp_test_losses.mean():.6f}, std={awp_test_losses.std():.6f}")
    print(f"  Val-Test Gap:")
    print(f"    Mean: {awp_vt_gaps.mean():.6f}")
    print(f"    Std:  {awp_vt_gaps.std():.6f}")
    print(f"    Max:  {awp_vt_gaps.max():.6f}")
    print(f"    Min:  {awp_vt_gaps.min():.6f}")
    print(f"  Pearson r(Val Loss, Test Loss): {pearson_r:.6f}")

    print(f"\n--- Regularized Loss (BCE + 0.01*L2) ---")
    print(f"  Val-Test Gap (reg):")
    print(f"    Mean: {awp_vt_gaps_reg.mean():.6f}")
    print(f"    Std:  {awp_vt_gaps_reg.std():.6f}")
    print(f"    Max:  {awp_vt_gaps_reg.max():.6f}")
    print(f"  Pearson r(Val Loss reg, Test Loss reg): {pearson_r_reg:.6f}")

    print(f"\n--- Baseline Reference ---")
    print(f"  Baseline Val Loss:  {baseline_row['Val Loss']:.6f}")
    print(f"  Baseline Test Loss: {baseline_row['Test Loss']:.6f}")
    print(f"  Baseline Val-Test Gap: {baseline_row['Val-Test Gap']:.6f}")
    print(f"  Baseline Test Acc:  {baseline_row['Test Acc']:.4f}")
    print(f"  Baseline Test F1:   {baseline_row['Test F1']:.4f}")

    # ── Rashomon bound verification ───────────────────────────────────
    epsilon = 0.15
    baseline_val = baseline_row["Val Loss"]
    rashomon_bound = baseline_val + epsilon
    probes_within_bound = sum(1 for r in awp_rows if r["Val Loss"] <= rashomon_bound)
    probes_test_within = sum(1 for r in awp_rows
                            if r["Test Loss"] <= baseline_row["Test Loss"] + epsilon)

    print(f"\n--- Rashomon Bound Verification ---")
    print(f"  Epsilon: {epsilon}")
    print(f"  Rashomon val-loss bound: {rashomon_bound:.6f}")
    print(f"  AWP probes within val bound:  {probes_within_bound}/{len(awp_rows)}")
    print(f"  AWP probes within test bound: {probes_test_within}/{len(awp_rows)} "
          f"(test_loss <= baseline_test + eps)")

    # ── Scientific Verdict ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SCIENTIFIC VERDICT")
    print("=" * 70)

    mean_gap = awp_vt_gaps.mean()
    max_gap = awp_vt_gaps.max()
    gap_std = awp_vt_gaps.std()
    baseline_gap = baseline_row["Val-Test Gap"]

    # The key question is whether the gap is PROBE-SPECIFIC (overfitting)
    # or SYSTEMATIC (distribution shift between val and test splits).
    # If the gap is constant across probes AND matches the baseline, it's systematic.
    excess_gap = mean_gap - baseline_gap  # AWP-specific gap beyond baseline
    gap_cv = gap_std / mean_gap if mean_gap > 0 else 0  # coefficient of variation

    if pearson_r > 0.95 and gap_cv < 0.05 and abs(excess_gap) < 0.02:
        verdict = ("STRONG EVIDENCE: Rashomon probes generalize. "
                   "Val loss is a reliable proxy for test loss.")
        detail = (f"The val-test gap ({mean_gap:.4f} +/- {gap_std:.4f}) is a constant "
                  f"offset matching the baseline ({baseline_gap:.4f}), NOT probe-specific "
                  f"overfitting. Pearson r = {pearson_r:.4f} confirms near-perfect "
                  f"rank preservation: probes with lower val loss also have lower test loss.")
    elif pearson_r > 0.8 and gap_cv < 0.10:
        verdict = ("MODERATE EVIDENCE: Rashomon probes show reasonable generalization "
                   "with some probe-specific variation in the val-test gap.")
        detail = (f"Mean |Val-Test Gap| = {mean_gap:.4f}, "
                  f"Pearson r = {pearson_r:.4f}, "
                  f"Excess gap vs baseline = {excess_gap:.4f}.")
    else:
        verdict = ("CONCERN: Significant probe-specific gap between val and test loss. "
                   "Some Rashomon probes may overfit the validation split.")
        detail = (f"Mean |Val-Test Gap| = {mean_gap:.4f}, "
                  f"Pearson r = {pearson_r:.4f}.")

    print(f"\n  {verdict}")
    print(f"  {detail}")
    print(f"  Max |Val-Test Gap| = {max_gap:.4f}")
    print(f"  Baseline Val-Test Gap = {baseline_gap:.4f} (reference)")
    print(f"  Excess gap (AWP mean - baseline) = {excess_gap:.4f}")
    print(f"  Gap coefficient of variation = {gap_cv:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────
    print(f"\nSaving results to {CSV_OUTPUT_PATH} ...")
    with open(CSV_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in r.items()})

    print(f"CSV saved with {len(rows)} rows (1 baseline + {len(awp_rows)} AWP).")

    # ── Print summary line for easy extraction ────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY (copy-paste for technical document):")
    print(f"  Mean Val-Test Gap: {mean_gap:.4f} +/- {awp_vt_gaps.std():.4f}")
    print(f"  Pearson r(val, test): {pearson_r:.4f}")
    print(f"  All {probes_within_bound}/50 AWP probes satisfy val-loss Rashomon bound")
    print(f"  {probes_test_within}/50 AWP probes satisfy analogous test-loss bound")
    print(f"{'='*70}")
    print("\nANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()
