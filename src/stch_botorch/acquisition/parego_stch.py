"""
qLogSTCHParEGO: Minimal modification of BoTorch's qLogNParEGO.

The ONLY difference from qLogNParEGO: replaces get_chebyshev_scalarization
with get_stch_scalarization (smooth Tchebycheff via LogSumExp).

Everything else — model, sampler, optimizer, X_pending, EI computation —
is standard BoTorch, inherited without modification.

Reference:
    Lin et al., "Smooth Tchebycheff Scalarization for Multi-Objective Optimization"
    ICML 2024.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement, TAU_MAX, TAU_RELU
try:
    from botorch.acquisition.multi_objective.monte_carlo import MultiObjectiveMCAcquisitionFunction
except ImportError:
    from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import Model
from botorch.posteriors.fully_bayesian import MCMC_DIM
from botorch.sampling.base import MCSampler
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import is_ensemble
from torch import Tensor

from stch_botorch.scalarization_botorch import get_stch_scalarization


class qLogSTCHParEGO(
    qLogNoisyExpectedImprovement, MultiObjectiveMCAcquisitionFunction
):
    r"""q-Log Smooth Tchebycheff ParEGO.

    Identical to BoTorch's ``qLogNParEGO`` except the Chebyshev scalarization
    is replaced with STCH (Smooth Tchebycheff). The LogSumExp-based
    scalarization provides smooth, informative gradients everywhere,
    improving acquisition optimization compared to the piecewise-linear
    Chebyshev max operator.

    For batch q > 1, use with ``optimize_acqf_list`` (one instance per
    candidate with different random weights) — same as standard qNParEGO.

    Args:
        model: A fitted multi-output model.
        X_baseline: Design points already observed (``r x d``).
        mu: STCH smoothing temperature. Default 0.1.
        scalarization_weights: Optional ``m``-dim weight tensor.
            If None, sampled from the unit simplex.
        sampler: MC sampler for posterior samples.
        objective: Multi-output objective (default: identity).
        constraints: Constraint callables.
        X_pending: Pending design points.
        eta: Constraint smoothing temperature.
        fat: Log vs linear asymptotic behavior.
        prune_baseline: Prune unlikely-best baseline points.
        cache_root: Cache root decomposition for low-rank updates.
        tau_relu: ReLU approximation temperature.
        tau_max: Max approximation temperature.
        incremental: Incremental EI over pending points.
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        mu: float = 0.1,
        scalarization_weights: Tensor | None = None,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = True,
        prune_baseline: bool = False,
        cache_root: bool = True,
        tau_relu: float = TAU_RELU,
        tau_max: float = TAU_MAX,
        incremental: bool = True,
    ) -> None:
        # --- Step 1: Initialize multi-output objective ---
        MultiObjectiveMCAcquisitionFunction.__init__(
            self,
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
        )
        org_objective = self.objective

        # --- Step 2: Get baseline Y for normalization ---
        with torch.no_grad():
            Y_baseline = org_objective(model.posterior(X_baseline).mean)
        if is_ensemble(model):
            Y_baseline = torch.mean(Y_baseline, dim=MCMC_DIM)

        # --- Step 3: Sample or use provided weights ---
        scalarization_weights = (
            scalarization_weights
            if scalarization_weights is not None
            else sample_simplex(
                d=Y_baseline.shape[-1],
                device=X_baseline.device,
                dtype=X_baseline.dtype,
            ).view(-1)
        )

        # --- Step 4: Build STCH scalarization (THE ONLY DIFFERENCE) ---
        stch_scalarization = get_stch_scalarization(
            weights=scalarization_weights,
            Y=Y_baseline,
            mu=mu,
        )

        # --- Step 5: Wrap in composite objective (same as qLogNParEGO) ---
        composite_objective = GenericMCObjective(
            objective=lambda samples, X=None: stch_scalarization(
                org_objective(samples=samples, X=X), X=X
            ),
        )

        # --- Step 6: Initialize qLogNEI with composite objective ---
        qLogNoisyExpectedImprovement.__init__(
            self,
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=composite_objective,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
            tau_max=tau_max,
            tau_relu=tau_relu,
            incremental=incremental,
        )

        # --- Debugging / transparency attributes ---
        self._org_objective: MCMultiOutputObjective = org_objective
        self.stch_scalarization: Callable[[Tensor, Tensor | None], Tensor] = (
            stch_scalarization
        )
        self.scalarization_weights: Tensor = scalarization_weights
        self.Y_baseline: Tensor = Y_baseline
        self.mu: float = mu
