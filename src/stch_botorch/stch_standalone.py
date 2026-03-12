"""
Standalone STCH scalarization — pure PyTorch, no BoTorch dependency.

Use this if you're optimizing something other than molecular properties:
quantum circuits, neural architecture search, reinforcement learning,
or any multi-objective problem where you have a differentiable objective.

The idea: convert a multi-objective problem into a sequence of single-objective
problems by scalarizing with different weight vectors. Each weight vector
emphasizes a different trade-off. Solve all of them, and the union of solutions
approximates the Pareto front.

STCH makes this work better than standard Chebyshev because:
- Chebyshev uses max(w_i * y_i) — non-differentiable at the kink
- STCH uses mu * logsumexp(w_i * y_i / mu) — smooth everywhere
- Same solution set as mu -> 0, but gradient-based optimizers actually converge
"""

import torch
from torch import Tensor
from typing import Optional, Tuple


def stch_scalarize(
    objectives: Tensor,
    weights: Tensor,
    mu: float = 0.1,
) -> Tensor:
    """Scalarize multi-objective values into a single scalar.

    Given m objective values and a weight vector, returns a single number
    that represents "how good is this solution for this particular trade-off?"

    Lower is better (minimization). Negate if your objectives are rewards.

    Args:
        objectives: (..., m) tensor of objective values.
        weights: (m,) tensor of weights on the simplex (positive, sum to 1).
        mu: Smoothing temperature. Lower = closer to hard Chebyshev max.

    Returns:
        (...,) tensor of scalarized values.

    Example:
        >>> obj = torch.tensor([0.8, 0.3, 0.5])  # 3 objectives
        >>> w = torch.tensor([0.5, 0.3, 0.2])     # weight vector
        >>> stch_scalarize(obj, w, mu=0.1)         # single scalar
    """
    return mu * torch.logsumexp(weights * objectives / mu, dim=-1)


def generate_stch_weights(
    m: int,
    K: int,
    mu: float = 0.01,
    n_restarts: int = 20,
    n_steps: int = 500,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Generate K weight vectors that cover all m objectives evenly.

    Instead of random Dirichlet draws (which cluster and miss objectives
    at high m), this optimizes K weights so the worst-covered objective
    is maximally covered.

    Think of it as: "place K spotlights on a simplex so every corner is lit."

    Args:
        m: Number of objectives.
        K: Number of weight vectors. K=m is a good default.
        mu: Temperature for the coverage criterion. Lower = more specialized.
        n_restarts: Random restarts (more = better but slower).
        n_steps: Gradient steps per restart.
        device: Torch device.

    Returns:
        (K, m) tensor of weight vectors, each row sums to 1.
    """
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}

    best_weights = None
    best_val = float("-inf")

    for _ in range(n_restarts):
        log_w = torch.randn(K, m, **tkwargs) * 0.5
        log_w.requires_grad_(True)
        opt = torch.optim.Adam([log_w], lr=0.05)

        for _ in range(n_steps):
            opt.zero_grad()
            w = torch.softmax(log_w, dim=-1)
            coverage = mu * torch.logsumexp(w / mu, dim=0)       # (m,) best weight per obj
            worst = -mu * torch.logsumexp(-coverage / mu, dim=0)  # scalar: worst obj
            (-worst).backward()
            opt.step()

        with torch.no_grad():
            w_f = torch.softmax(log_w, dim=-1)
            coverage = mu * torch.logsumexp(w_f / mu, dim=0)
            worst = -mu * torch.logsumexp(-coverage / mu, dim=0)
            if worst.item() > best_val:
                best_val = worst.item()
                best_weights = w_f.clone()

    return best_weights


def multi_objective_optimize(
    objective_fn,
    dim: int,
    m: int,
    bounds: Tuple[float, float] = (0.0, 1.0),
    K: int = None,
    mu_weights: float = 0.01,
    mu_scalarize: float = 0.1,
    n_steps: int = 200,
    lr: float = 0.01,
    n_random_starts: int = 5,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Simple multi-objective optimization using STCH scalarization.

    This is a standalone optimizer — no GP, no BO, no BoTorch needed.
    Just give it a differentiable function that maps x -> (obj_1, ..., obj_m)
    and it finds a set of Pareto-approximate solutions.

    How it works:
    1. Generate K structured weight vectors (STCH-Set)
    2. For each weight: run gradient descent on stch_scalarize(f(x), weight)
    3. Return K solutions, each targeting a different trade-off

    Args:
        objective_fn: Callable x -> (m,) tensor. Must be differentiable.
            Convention: MINIMIZE all objectives.
        dim: Input dimension.
        m: Number of objectives.
        bounds: (lower, upper) bounds for each input dimension.
        K: Number of solutions to find. Default = m.
        mu_weights: Temperature for weight generation.
        mu_scalarize: Temperature for scalarization during optimization.
        n_steps: Gradient descent steps per weight vector.
        lr: Learning rate.
        n_random_starts: Random initializations per weight (best kept).
        device: Torch device.

    Returns:
        solutions: (K, dim) tensor of input solutions.
        objectives: (K, m) tensor of objective values at each solution.

    Example:
        >>> def my_problem(x):
        ...     f1 = x.sum()           # minimize total
        ...     f2 = (x ** 2).sum()    # minimize energy
        ...     f3 = -x.min()          # maximize minimum component
        ...     return torch.stack([f1, f2, f3])
        ...
        >>> solutions, objectives = multi_objective_optimize(
        ...     my_problem, dim=5, m=3, n_steps=500
        ... )
    """
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    K = K or m
    lo, hi = bounds

    # Step 1: generate structured weights
    weights = generate_stch_weights(m, K, mu=mu_weights, device=device)

    all_solutions = []
    all_objectives = []

    # Step 2: for each weight, find the best solution
    for k in range(K):
        w = weights[k]
        best_x = None
        best_val = float("inf")

        for _ in range(n_random_starts):
            x = lo + (hi - lo) * torch.rand(dim, **tkwargs)
            x.requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr)

            for _ in range(n_steps):
                opt.zero_grad()
                obj = objective_fn(x)
                scalar = stch_scalarize(obj, w, mu=mu_scalarize)
                scalar.backward()
                opt.step()
                # Project back to bounds
                with torch.no_grad():
                    x.clamp_(lo, hi)

            with torch.no_grad():
                obj = objective_fn(x)
                val = stch_scalarize(obj, w, mu=mu_scalarize).item()
                if val < best_val:
                    best_val = val
                    best_x = x.clone()

        with torch.no_grad():
            all_solutions.append(best_x)
            all_objectives.append(objective_fn(best_x))

    return torch.stack(all_solutions), torch.stack(all_objectives)
