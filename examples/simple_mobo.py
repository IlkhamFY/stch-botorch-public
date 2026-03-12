"""
Minimal multi-objective BO example using STCH scalarization.

Optimizes DTLZ2 with m=4 objectives using qNParEGO with STCH weights.
Runs 10 BO iterations, prints R2 metric each iteration.
"""
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import draw_sobol_samples, sample_simplex
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from stch_botorch.scalarization_botorch import get_stch_scalarization
from stch_botorch.weight_selection import select_stch_weights

# --- Setup ---
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
m = 4  # objectives
dim = m + 1  # input dimensions
K = m  # candidates per iteration

problem = DTLZ2(dim=dim, num_objectives=m, negate=True).to(**tkwargs)

# Initial Sobol points
torch.manual_seed(0)
n_init = 2 * (dim + 1)
train_x = draw_sobol_samples(bounds=problem.bounds, n=n_init, q=1).squeeze(1)
train_obj = problem(train_x)

# Pre-compute STCH weights (coverage-optimal)
print(f"Computing STCH weights (m={m}, K={K})...")
stch_weights = select_stch_weights(m=m, K=K, mu=0.01)
print(f"Done. Weight matrix shape: {stch_weights.shape}")

# --- BO Loop ---
n_batch = 10
mc_samples = 128
num_restarts = 5
raw_samples = 256

for iteration in range(1, n_batch + 1):
    # Fit m independent GPs
    train_x_norm = normalize(train_x, problem.bounds)
    models = [
        SingleTaskGP(train_x_norm, train_obj[..., i:i+1], outcome_transform=Standardize(m=1))
        for i in range(m)
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Build K acquisition functions with STCH scalarization
    with torch.no_grad():
        pred = model.posterior(train_x_norm).mean

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    acq_list = []
    for k in range(K):
        # Use STCH weights instead of random simplex
        obj = GenericMCObjective(
            get_stch_scalarization(weights=stch_weights[k], Y=pred, mu=0.1)
        )
        acq = qLogNoisyExpectedImprovement(
            model=model, objective=obj, X_baseline=train_x_norm,
            sampler=sampler, prune_baseline=True
        )
        acq_list.append(acq)

    # Optimize
    bounds = torch.zeros(2, dim, **tkwargs)
    bounds[1] = 1
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_list, bounds=bounds,
        num_restarts=num_restarts, raw_samples=raw_samples
    )

    # Evaluate and add to training data
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])

    # Quick R2 metric (lower = better)
    n_pts = train_obj.shape[0]
    print(f"  Iter {iteration:>2}/{n_batch}: {n_pts} points, "
          f"best per obj: {train_obj.max(dim=0).values.tolist()}")

print(f"\nFinal: {train_obj.shape[0]} points evaluated")
print(f"Best per objective: {train_obj.max(dim=0).values.tolist()}")
