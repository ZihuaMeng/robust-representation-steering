"""Hessian computation for BCE linear probe at baseline parameters."""

import torch


def compute_hessian(weight, bias, X, y, weight_decay=0.01, ridge=1e-4):
    """Compute Hessian of BCE + L2 loss w.r.t. [w; b] at the baseline.

    H = (1/N) sum_i p_i(1-p_i) [x_i;1][x_i;1]^T  +  diag(lambda,...,lambda,0)

    Args:
        weight: [d] baseline weight vector
        bias:   scalar baseline bias
        X:      [N, d] training activations
        y:      [N] binary labels
        weight_decay: L2 regularization coefficient
        ridge:  additional diagonal regularization for numerical stability

    Returns:
        H: [d+1, d+1] Hessian matrix (float64)
        H_inv: [d+1, d+1] inverse Hessian
        eig_info: dict with eigenvalue diagnostics
    """
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

    # H = (1/N) X_aug^T diag(s) X_aug + regularization
    # Efficient: X_aug_scaled = sqrt(s)[:,None] * X_aug, then H = X_aug_scaled^T X_aug_scaled / N
    sqrt_s = s.sqrt().unsqueeze(1)  # [N, 1]
    X_scaled = sqrt_s * X_aug  # [N, d+1]
    H = (X_scaled.T @ X_scaled) / N

    # L2 regularization on w only (not bias)
    reg = torch.zeros(d + 1, dtype=torch.float64)
    reg[:d] = weight_decay
    H += torch.diag(reg)

    # Ridge for numerical stability
    H += ridge * torch.eye(d + 1, dtype=torch.float64)

    # Eigenvalue diagnostics
    eigvals = torch.linalg.eigvalsh(H)
    eig_min = eigvals[0].item()
    eig_max = eigvals[-1].item()
    cond = eig_max / eig_min if eig_min > 0 else float("inf")

    print(f"Hessian shape: {H.shape}")
    print(f"Eigenvalue range: [{eig_min:.6e}, {eig_max:.6e}]")
    print(f"Condition number: {cond:.2f}")

    H_inv = torch.linalg.inv(H)

    return H, H_inv, {"eig_min": eig_min, "eig_max": eig_max, "condition": cond}
