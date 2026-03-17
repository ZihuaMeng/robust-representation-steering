"""Naive and robust steering delta solvers.

Naive:  min ||delta||^2  s.t.  theta^T (x_aug + [delta;0]) >= t
Robust: min ||delta||^2  s.t.  theta^T x_new_aug - sqrt(2*eps * x_new_aug^T H_inv x_new_aug) >= t

The robust solver uses bisection along the naive delta direction: scale the
naive delta until the Rashomon ellipsoid constraint is satisfied, then binary
search for the minimum scale factor. This is a 1D search and runs in
microseconds per example.
"""

import torch


def naive_delta(weight, bias, x, threshold=0.0):
    """Minimum-norm delta to cross the baseline probe's decision boundary.

    Returns:
        delta: [d] perturbation vector, or zeros if already safe
        score_before: float
    """
    score = (weight * x).sum() + bias
    score_f = score.item()
    if score_f >= threshold:
        return torch.zeros_like(x), score_f

    gap = threshold - score_f
    w_norm_sq = (weight * weight).sum().item()
    delta = (gap / w_norm_sq) * weight
    return delta, score_f


def _robust_margin(x_aug, theta, H_inv, epsilon, threshold):
    """Compute robust margin: theta^T x_aug - sqrt(2*eps * x_aug^T H_inv x_aug) - t."""
    linear = theta @ x_aug
    quad = x_aug @ H_inv @ x_aug
    return (linear - torch.sqrt(2.0 * epsilon * quad.clamp(min=0.0)) - threshold).item()


def robust_delta(weight, bias, x, H_inv, epsilon, threshold=0.0):
    """Minimum-norm delta satisfying the Rashomon ellipsoid constraint.

    Scales the naive delta direction via bisection until the robust
    constraint is met.

    Returns:
        delta: [d] perturbation vector
    """
    d_naive, score = naive_delta(weight, bias, x, threshold)
    if score >= threshold:
        return torch.zeros_like(x)

    theta = torch.cat([weight, bias.unsqueeze(0)])  # [d+1]
    one = torch.ones(1, dtype=x.dtype)

    # Check if naive delta already satisfies robust constraint
    def check(scale):
        delta = d_naive * scale
        x_aug = torch.cat([x + delta, one])
        return _robust_margin(x_aug, theta, H_inv, epsilon, threshold)

    # If naive already works (scale=1)
    if check(1.0) >= 0:
        return d_naive

    # Find upper bound where constraint is satisfied
    lo, hi = 1.0, 2.0
    for _ in range(30):
        if check(hi) >= 0:
            break
        hi *= 2.0
    else:
        # Even very large scaling doesn't work — return best effort
        return d_naive * hi

    # Bisect for minimum scale
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if check(mid) >= 0:
            hi = mid
        else:
            lo = mid

    return d_naive * hi


def rashomon_coverage(delta, x, probes, threshold=0.0):
    """Fraction of Rashomon probes that classify x+delta as safe.

    Returns:
        (n_safe, n_total, fraction)
    """
    x_new = x + delta
    safe = 0
    for p in probes:
        score = (p["weight"].double() * x_new).sum() + p["bias"].double()
        if score.item() >= threshold:
            safe += 1
    return safe, len(probes), safe / len(probes) if probes else 0.0
