"""
STCH-Set weight selection for many-objective Bayesian optimization.

Generates K weight vectors on the probability simplex that maximize
coverage of all m objectives using the STCH smooth min-max criterion.

Reference:
    Lin et al., "STCH-Set: Multi-Objective Optimization via Smooth Tchebycheff
    Scalarization" (ICLR 2025, arXiv:2405.19650).
"""

import torch
from torch import Tensor
from typing import Optional


def select_stch_weights(
    m: int,
    K: int,
    mu: float = 0.01,
    n_restarts: int = 20,
    n_steps: int = 500,
    lr: float = 0.05,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.double,
) -> Tensor:
    """Select K weight vectors that maximize STCH-Set coverage of m objectives.

    Optimizes K weight vectors on the simplex so that the worst-covered
    objective is maximally covered:

        max_W  min_i { mu * logsumexp_k(w_{k,i} / mu) }

    As mu -> 0, this recovers the hard min-max: max_W min_i max_k w_{k,i}.

    Args:
        m: Number of objectives.
        K: Number of weight vectors (typically K = m).
        mu: STCH smoothing temperature. Lower = more specialized weights.
            Default 0.01. Recommended range: 0.001-0.1.
        n_restarts: Number of random restarts for optimization.
        n_steps: Gradient steps per restart.
        lr: Adam learning rate.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tensor of shape (K, m) with K weight vectors on the probability simplex.
    """
    tkwargs = {"device": device, "dtype": dtype}
    best_weights = None
    best_val = float("-inf")

    for _ in range(n_restarts):
        log_w = torch.randn(K, m, **tkwargs) * 0.5
        log_w.requires_grad_(True)
        optimizer = torch.optim.Adam([log_w], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            w = torch.softmax(log_w, dim=-1)
            # Coverage per objective: smooth max over K weights
            coverage_per_obj = mu * torch.logsumexp(w / mu, dim=0)
            # Worst-covered objective: smooth min
            worst_coverage = -mu * torch.logsumexp(-coverage_per_obj / mu, dim=0)
            (-worst_coverage).backward()
            optimizer.step()

        with torch.no_grad():
            w_final = torch.softmax(log_w, dim=-1)
            coverage_per_obj = mu * torch.logsumexp(w_final / mu, dim=0)
            worst_coverage = -mu * torch.logsumexp(-coverage_per_obj / mu, dim=0)
            if worst_coverage.item() > best_val:
                best_val = worst_coverage.item()
                best_weights = w_final.clone()

    return best_weights
