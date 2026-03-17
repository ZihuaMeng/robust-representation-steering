"""Adversarial Weight Perturbation (AWP) engine for Rashomon set enumeration."""

import torch
import torch.nn.functional as F


def compute_val_loss(weight, bias, val_X, val_y):
    """BCE loss of a linear probe on validation data. No autograd graph built."""
    with torch.no_grad():
        logits = val_X @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, val_y, reduction="mean")
    return loss.item()


def perturb_one_probe(
    baseline_w, baseline_b, baseline_val_loss,
    anchor_x, target_label,
    val_X, val_y,
    epsilon, sgd_steps, lr, momentum, device,
):
    """Run constrained SGD toward flipping the anchor prediction.

    Returns (final_w, final_b, steps_accepted, moved).
    """
    w = baseline_w.clone().to(device).requires_grad_(True)
    b = baseline_b.clone().to(device).requires_grad_(True)
    anchor_x = anchor_x.to(device)
    target = torch.tensor([target_label], dtype=torch.float32, device=device)

    last_valid_w = baseline_w.clone().to(device)
    last_valid_b = baseline_b.clone().to(device)
    optimizer = torch.optim.SGD([w, b], lr=lr, momentum=momentum)
    accepted = 0

    for step in range(sgd_steps):
        optimizer.zero_grad()
        logit = (w * anchor_x).sum() + b  # scalar
        loss = F.binary_cross_entropy_with_logits(logit.unsqueeze(0), target)
        loss.backward()
        optimizer.step()

        vl = compute_val_loss(w.data, b.data, val_X, val_y)
        if vl <= baseline_val_loss + epsilon:
            last_valid_w.copy_(w.data)
            last_valid_b.copy_(b.data)
            accepted += 1
        else:
            # Revert parameters and reset optimizer (clears momentum buffer)
            w.data.copy_(last_valid_w)
            b.data.copy_(last_valid_b)
            optimizer = torch.optim.SGD([w, b], lr=lr, momentum=momentum)

    moved = not (
        torch.allclose(last_valid_w, baseline_w.to(device))
        and torch.allclose(last_valid_b, baseline_b.to(device))
    )
    return last_valid_w.cpu(), last_valid_b.cpu(), accepted, moved


def run_awp_rashomon(
    train_X, train_y, val_X, val_y,
    baseline_w, baseline_b,
    n_candidates=50, epsilon=0.15,
    sgd_steps=400, lr=1e-4, momentum=0.9,
    max_retries=3, seed=42, device="cpu",
):
    """Enumerate Rashomon set members via AWP.

    Returns list of dicts with keys: weight, bias, val_loss, anchor_idx, steps_accepted.
    """
    val_X_d = val_X.to(device)
    val_y_d = val_y.float().to(device)
    train_X_d = train_X.to(device)
    train_y_f = train_y.float()

    baseline_val_loss = compute_val_loss(
        baseline_w.to(device), baseline_b.to(device), val_X_d, val_y_d,
    )
    print(f"Baseline val loss: {baseline_val_loss:.4f}")
    print(f"Rashomon bound (baseline + eps): {baseline_val_loss + epsilon:.4f}")
    print()

    probes = []
    rng = torch.Generator().manual_seed(seed)

    for i in range(n_candidates):
        found = False
        for retry in range(max_retries):
            anchor_idx = torch.randint(len(train_y), (1,), generator=rng).item()
            true_label = train_y_f[anchor_idx].item()
            target_label = 1.0 - true_label

            w_new, b_new, accepted, moved = perturb_one_probe(
                baseline_w, baseline_b, baseline_val_loss,
                train_X_d[anchor_idx], target_label,
                val_X_d, val_y_d,
                epsilon, sgd_steps, lr, momentum, device,
            )

            if moved:
                vl = compute_val_loss(w_new.to(device), b_new.to(device), val_X_d, val_y_d)
                cos = F.cosine_similarity(
                    baseline_w.unsqueeze(0), w_new.unsqueeze(0),
                ).item()
                probes.append({
                    "weight": w_new,
                    "bias": b_new,
                    "val_loss": vl,
                    "anchor_idx": anchor_idx,
                    "steps_accepted": accepted,
                })
                print(
                    f"  Probe {len(probes):>2d}/50  "
                    f"val_loss={vl:.4f}  accepted={accepted}/{sgd_steps}  "
                    f"cos_sim={cos:.6f}  (anchor={anchor_idx}, retry={retry})"
                )
                found = True
                break
            else:
                if retry < max_retries - 1:
                    print(f"  Candidate {i+1}: retry {retry+1}/{max_retries} (0 steps accepted)")

        if not found:
            print(f"  Candidate {i+1}: FAILED after {max_retries} retries — skipped")

    print(f"\nGenerated {len(probes)}/{n_candidates} Rashomon set members.")
    return probes, baseline_val_loss
