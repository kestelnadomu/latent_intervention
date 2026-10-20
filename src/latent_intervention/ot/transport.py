"""Small dependency-free OT and ODE solvers; failures are never silent fallbacks."""

import math
import torch


def squared_cost(x, y):
    return torch.cdist(x.double(), y.double()).square() / x.shape[-1]


@torch.no_grad()
def sinkhorn(x, y, *, epsilon=0.05, tolerance=1e-6, max_iterations=100000):
    """Balanced uniform-marginal entropic OT, in float64 log space."""
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or not len(x)
        or not len(y)
        or not torch.isfinite(x).all()
        or not torch.isfinite(y).all()
        or x.shape[1] < 1
        or x.device != y.device
        or not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or type(max_iterations) is not int
        or max_iterations < 1
    ):
        raise ValueError("invalid Sinkhorn inputs/settings")
    cost = squared_cost(x, y)
    if not torch.isfinite(cost).all():
        raise FloatingPointError("nonfinite Sinkhorn cost")
    log_a, log_b = -math.log(len(x)), -math.log(len(y))
    u, v = torch.zeros_like(cost[:, 0]), torch.zeros_like(cost[0])
    symmetric = torch.equal(x, y)
    kernel = -cost / epsilon

    def solution(iteration):
        log_plan = kernel + u[:, None] + v[None]
        plan = log_plan.exp()
        error = max(
            float((plan.sum(1) - 1 / len(x)).abs().max()),
            float((plan.sum(0) - 1 / len(y)).abs().max()),
        )
        if math.isfinite(error) and error <= tolerance:
            objective = (plan * (cost + epsilon * (log_plan - log_a - log_b))).sum()
            return (
                plan,
                dict(
                    iterations=iteration,
                    marginal_error=error,
                    objective=float(objective),
                ),
            ), error
        return None, error

    # Most high-dimensional smoothed minibatches converge without continuation.
    consumed = min(20, max_iterations)
    for _ in range(consumed):
        if symmetric:
            u = 0.5 * (u + log_a - torch.logsumexp(kernel + u[None], dim=1))
            v = u
        else:
            u = log_a - torch.logsumexp(kernel + v[None], dim=1)
            v = log_b - torch.logsumexp(kernel + u[:, None], dim=0)
    result, error = solution(consumed)
    if result is not None:
        return result
    # Epsilon continuation avoids very slow scaling near permutation couplings.
    # Warm starts only: the returned plan is always solved at the requested epsilon.
    warm_epsilon = epsilon if symmetric else max(epsilon, float(cost.max()))
    warm_steps = min(50, max_iterations // 20)
    if warm_epsilon > epsilon:
        u, v = torch.zeros_like(u), torch.zeros_like(v)
    while (
        warm_epsilon > epsilon
        and warm_steps
        and consumed + warm_steps < max_iterations // 2
    ):
        kernel = -cost / warm_epsilon
        for _ in range(warm_steps):
            u = log_a - torch.logsumexp(kernel + v[None], dim=1)
            v = log_b - torch.logsumexp(kernel + u[:, None], dim=0)
        consumed += warm_steps
        next_epsilon = max(epsilon, warm_epsilon / 2)
        u, v = u * (warm_epsilon / next_epsilon), v * (warm_epsilon / next_epsilon)
        warm_epsilon = next_epsilon
    if warm_epsilon != epsilon:
        u, v = u * (warm_epsilon / epsilon), v * (warm_epsilon / epsilon)
    kernel = -cost / epsilon
    for iteration in range(consumed + 1, max_iterations + 1):
        if symmetric:
            # Symmetric damped scaling avoids slow, asymmetric dual gauges for OT(x,x).
            u = 0.5 * (u + log_a - torch.logsumexp(kernel + u[None], dim=1))
            v = u
        else:
            u = log_a - torch.logsumexp(kernel + v[None], dim=1)
            v = log_b - torch.logsumexp(kernel + u[:, None], dim=0)
        if iteration % 10 == 0 or iteration == max_iterations:
            result, error = solution(iteration)
            if result is not None:
                return result
    raise FloatingPointError(f"Sinkhorn did not converge: marginal error {error:.3g}")


def sinkhorn_divergence(x, y, **settings):
    _, xy = sinkhorn(x, y, **settings)
    _, xx = sinkhorn(x, x, **settings)
    _, yy = sinkhorn(y, y, **settings)
    # Finite iterations can produce tiny negative numerical residuals.
    value = xy["objective"] - 0.5 * (xx["objective"] + yy["objective"])
    if value < -1e-5:
        raise FloatingPointError(
            "negative Sinkhorn divergence beyond numerical tolerance"
        )
    return max(value, 0.0)


def rk4(velocity, x, state, start, end, steps):
    step = (end - start) / steps
    for i in range(steps):
        t = start + i * step
        k1 = velocity(x, t, state)
        k2 = velocity(x + step * k1 / 2, t + step / 2, state)
        k3 = velocity(x + step * k2 / 2, t + step / 2, state)
        k4 = velocity(x + step * k3, t + step, state)
        x = x + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


@torch.no_grad()
def integrate(
    velocity, x, state, *, reverse=False, steps=64, max_steps=256, tolerance=1e-4
):
    if (
        type(steps) is not int
        or steps < 1
        or type(max_steps) is not int
        or max_steps < 2 * steps
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("invalid ODE step-doubling settings")
    start, end = (1.0, 0.0) if reverse else (0.0, 1.0)
    previous = rk4(velocity, x, state, start, end, steps)
    nfe, error = 4 * steps, float("inf")
    while 2 * steps <= max_steps:
        steps *= 2
        current = rk4(velocity, x, state, start, end, steps)
        nfe += 4 * steps
        # Per-unit coordinate RMS in standardized embedding units (absolute tolerance).
        error = float((current - previous).square().mean(-1).sqrt().max())
        if torch.isfinite(current).all() and error <= tolerance:
            return current, dict(nfe=nfe, steps=steps, step_doubling_error=error)
        previous = current
    raise FloatingPointError(
        f"OT flow solver tolerance not met: error {error:.3g}, NFE {nfe}"
    )
