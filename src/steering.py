"""Naive and robust steering delta solvers.

Naive:  min ||delta||^2  s.t.  theta^T (x_aug + [delta;0]) >= t
Robust: min ||delta||^2  s.t.  theta^T x_new_aug - sqrt(2*eps * x_new_aug^T H_inv x_new_aug) >= t

The robust solver uses bisection along the naive delta direction: scale the
naive delta until the Rashomon ellipsoid constraint is satisfied, then binary
search for the minimum scale factor. This is a 1D search and runs in
microseconds per example.

The dynamic robust solver recomputes the local worst-case probe at each iterate
and moves in that local probe direction, then performs a final 1D bisection to
tighten the norm.
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


def _worst_theta_local(x_aug, theta, H_inv, epsilon):
    """Worst-case probe in the Hessian ellipsoid at x_aug.

    Minimizes theta'^T x_aug over (theta' - theta)^T H (theta' - theta) <= 2*epsilon,
    where H_inv is the inverse Hessian.
    """
    quad = (x_aug @ H_inv @ x_aug).clamp(min=0.0)
    denom = torch.sqrt(quad)
    if denom.item() == 0.0:
        return theta
    direction = (H_inv @ x_aug) / denom
    return theta - torch.sqrt(torch.tensor(2.0 * epsilon, dtype=x_aug.dtype)) * direction


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


def robust_delta_dynamic(
    weight, bias, x, H_inv, epsilon, threshold=0.0,
    max_iters=200, lr=5e-2, robustness_weight=1.0, proximity_weight=1e-2,
    tol=1e-8,
):
    """Dynamic robust steering with Adam and local worst-case probes.

    Mirrors the continuous optimization style: at each step, compute the local
    worst probe at x+delta, optimize a robust hinge objective plus proximity
    penalty with Adam, and then tighten with 1D bisection.
    """
    d_naive, score = naive_delta(weight, bias, x, threshold)
    if score >= threshold:
        return torch.zeros_like(x)

    theta = torch.cat([weight, bias.unsqueeze(0)])  # [d+1]
    one = torch.ones(1, dtype=x.dtype)
    delta = d_naive.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_delta = None
    best_norm = float("inf")

    # Local min-max optimization with a moving worst-case probe.
    for _ in range(max_iters):
        optimizer.zero_grad()

        x_new = x + delta
        x_aug = torch.cat([x_new, one])

        # Keep the inner worst-model computation fixed per optimization step.
        with torch.no_grad():
            worst_theta = _worst_theta_local(x_aug, theta, H_inv, epsilon)

        robust_logit = (worst_theta[:-1] * x_new).sum() + worst_theta[-1]
        loss_robust = torch.relu(threshold - robust_logit)
        loss_prox = torch.norm(delta, p=2)
        loss = robustness_weight * loss_robust + proximity_weight * loss_prox

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_aug_curr = torch.cat([x + delta, one])
            margin = _robust_margin(x_aug_curr, theta, H_inv, epsilon, threshold)
            if margin >= 0:
                norm_curr = torch.norm(delta, p=2).item()
                if norm_curr < best_norm:
                    best_norm = norm_curr
                    best_delta = delta.detach().clone()
            if loss_robust.item() <= tol and margin >= 0:
                break

    # If optimization never found a feasible point, fall back to robust_delta.
    if best_delta is None:
        return robust_delta(weight, bias, x, H_inv, epsilon, threshold)

    return best_delta


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
