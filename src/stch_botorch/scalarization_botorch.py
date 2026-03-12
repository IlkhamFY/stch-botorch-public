"""
STCH scalarization as a drop-in replacement for BoTorch's get_chebyshev_scalarization.

This module follows BoTorch's exact pattern from
botorch.utils.multi_objective.scalarization, replacing the hard Chebyshev
max(w_i * y_i) with Lin et al.'s smooth Tchebycheff mu * logsumexp(w_i * y_i / mu).

Reference:
    Lin et al., "Smooth Tchebycheff Scalarization for Multi-Objective Optimization"
    ICML 2024, Eq. 5.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.transforms import normalize
from torch import Tensor


def get_stch_scalarization(
    weights: Tensor,
    Y: Tensor,
    mu: float = 0.1,
) -> Callable[[Tensor, Tensor | None], Tensor]:
    r"""Construct a Smooth Tchebycheff (STCH) scalarization.

    Drop-in replacement for ``get_chebyshev_scalarization``. The STCH
    scalarization replaces the non-differentiable max operator with a smooth
    LogSumExp approximation:

        g_STCH(y) = mu * LogSumExp_i( w_i * y_i / mu )

    As mu -> 0 this recovers standard Tchebycheff. For any mu > 0, the
    scalarization is fully differentiable with informative gradients everywhere,
    which is critical for gradient-based acquisition optimization (L-BFGS-B).

    Like ``get_chebyshev_scalarization``, this function:
    - Negates inputs (BoTorch maximizes; STCH minimizes internally)
    - Normalizes to [0, 1] using observed Y bounds
    - Returns a negated result (so BoTorch can maximize it)

    Unlike augmented Chebyshev, there is no alpha term. The LogSumExp itself
    provides a smooth "spread" effect that serves a similar purpose.

    Args:
        weights: A ``m``-dim tensor of weights. Positive for maximization,
            negative for minimization. Internally, absolute values are used
            after normalizing to sum to 1.
        Y: A ``n x m``-dim tensor of observed outcomes, used for computing
            normalization bounds. If ``n=0``, outcomes are left unnormalized.
        mu: Smoothing temperature. Smaller = tighter approximation to hard max.
            Default 0.1. With uniform weights, effective temperature is mu * m.

    Returns:
        Callable that maps objective samples to scalar utility values.
        Same signature as ``get_chebyshev_scalarization``'s return value:
        ``(Y: Tensor, X: Tensor | None) -> Tensor``

    Example:
        >>> weights = torch.tensor([0.5, 0.5])
        >>> transform = get_stch_scalarization(weights, Y, mu=0.1)
        >>> scalarized = transform(Y_samples)  # (..., q, m) -> (..., q)
    """
    # BoTorch convention: negate Y so all objectives are minimized internally
    Y = -Y
    if weights.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "weights must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {weights.shape} and {Y.shape}."
        )

    # Handle sign convention: negative weight = minimize that objective
    # We take absolute weights and track which to minimize
    minimize = weights < 0
    abs_weights = weights.abs()
    # Normalize to sum to 1 (same as Lin et al.)
    abs_weights = abs_weights / abs_weights.sum()

    def stch_obj(Y: Tensor, X: Tensor | None = None) -> Tensor:
        # Lin et al. ICML 2024 Eq. 5:
        # g_STCH = mu * logsumexp(w_i * y_i / mu)
        return mu * torch.logsumexp(abs_weights * Y / mu, dim=-1)

    if Y.shape[-2] == 0:
        # No observations: skip normalization
        def obj(Y: Tensor, X: Tensor | None = None) -> Tensor:
            return -stch_obj(Y=-Y)

        return obj

    # Normalization bounds from observed data (same as BoTorch's Chebyshev)
    Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])

    def obj(Y: Tensor, X: Tensor | None = None) -> Tensor:
        # Normalize -Y to [0, 1] (same as get_chebyshev_scalarization)
        Y_normalized = normalize(-Y, bounds=Y_bounds)
        # Shift minimization objectives to [-1, 0]
        Y_normalized[..., minimize] = Y_normalized[..., minimize] - 1
        # Negate so BoTorch can maximize
        return -stch_obj(Y=Y_normalized)

    return obj
