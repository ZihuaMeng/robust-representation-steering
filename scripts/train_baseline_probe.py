"""End-to-end pipeline: BeaverTails → Gemma-2 activations → linear probe."""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from data_pipeline import (
    load_balanced_beavertails,
    load_model_and_tokenizer,
    extract_activations,
)
from probe import LinearProbe, train_probe, evaluate_probe

LAYER_IDX = 10
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", default="mean", choices=["mean", "last", "all"],
                        help="How to pool token activations: mean, last token, or all tokens.")
    args = parser.parse_args()

    activations_path = os.path.join(OUTPUT_DIR, f"beavertails_activations_layer{LAYER_IDX}_{args.pooling}.pt")
    probe_path = os.path.join(OUTPUT_DIR, f"baseline_probe_layer{LAYER_IDX}_{args.pooling}.pt")

    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────
    train_texts, test_texts, train_labels, test_labels = load_balanced_beavertails()

    # ── 2. Model + activation extraction ─────────────────────────────────
    model, tokenizer = load_model_and_tokenizer()

    print(f"\nExtracting layer {LAYER_IDX} activations (train, pooling={args.pooling}) ...")
    if args.pooling == "all":
        train_X, train_y = extract_activations(
            train_texts, model, tokenizer, layer_idx=LAYER_IDX,
            pooling="all", labels=train_labels,
        )
        # For inference we need one vector per test sequence — use last token
        print(f"Extracting layer {LAYER_IDX} activations (test, last-token for inference) ...")
        test_X = extract_activations(
            test_texts, model, tokenizer, layer_idx=LAYER_IDX, pooling="last",
        )
        test_y = torch.tensor(test_labels, dtype=torch.long)
    else:
        train_X = extract_activations(
            train_texts, model, tokenizer, layer_idx=LAYER_IDX, pooling=args.pooling,
        )
        print(f"Extracting layer {LAYER_IDX} activations (test, pooling={args.pooling}) ...")
        test_X = extract_activations(
            test_texts, model, tokenizer, layer_idx=LAYER_IDX, pooling=args.pooling,
        )
        train_y = torch.tensor(train_labels, dtype=torch.long)
        test_y = torch.tensor(test_labels, dtype=torch.long)

    print(f"\nActivation shapes — train: {train_X.shape}, test: {test_X.shape}")
    hidden_dim = train_X.shape[1]
    print(f"Hidden dim: {hidden_dim} (expected 2304)")

    # Save activations
    torch.save(
        {"train_X": train_X, "train_y": train_y, "test_X": test_X, "test_y": test_y},
        activations_path,
    )
    print(f"Activations saved to {activations_path}")

    # Free GPU memory — model no longer needed
    del model
    torch.cuda.empty_cache()

    # ── 3. Train linear probe ────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining linear probe on {device} ...")
    probe, loss_log = train_probe(
        train_X, train_y, input_dim=hidden_dim, device=device,
    )

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    metrics = evaluate_probe(probe, test_X, test_y, device=device)

    # ── 5. Save probe ────────────────────────────────────────────────────
    w = probe.linear.weight.detach().squeeze(0)  # [hidden_dim]
    b = probe.linear.bias.detach().squeeze(0)     # scalar
    final_loss = loss_log[-1][1] if loss_log else float("nan")
    torch.save(
        {"weight": w, "bias": b, "val_loss": final_loss, "test_metrics": metrics},
        probe_path,
    )
    print(f"\nProbe saved to {probe_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
