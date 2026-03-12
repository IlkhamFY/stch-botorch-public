"""
Multi-objective optimization with STCH — no BoTorch, no GP, just PyTorch.

This example optimizes a simple 3-objective problem using STCH scalarization.
Shows how the same approach works for any differentiable multi-objective problem:
quantum circuits, neural networks, control systems, etc.

The key insight: STCH converts "find the Pareto front" into
"solve K single-objective problems with structured weight vectors."
"""

import torch
from stch_botorch.stch_standalone import (
    stch_scalarize,
    generate_stch_weights,
    multi_objective_optimize,
)


# --- Define a multi-objective problem ---
# 3 objectives that trade off against each other

def three_objective_problem(x):
    """
    x: (dim,) tensor in [0, 1]

    Returns (3,) tensor of objectives to MINIMIZE:
        f1: distance to corner (1, 1, ..., 1)  — want x near top-right
        f2: sum of squares                      — want x near zero
        f3: negative min component              — want all components large

    These conflict: f1 wants x=1, f2 wants x=0, f3 wants uniform x.
    The Pareto front is the set of optimal trade-offs.
    """
    f1 = ((x - 1) ** 2).sum()    # minimize distance to (1,1,...,1)
    f2 = (x ** 2).sum()           # minimize energy (wants x=0)
    f3 = -x.min() + 0.5           # maximize worst component
    return torch.stack([f1, f2, f3])


# --- Method 1: Use the all-in-one optimizer ---
print("=" * 60)
print("Method 1: multi_objective_optimize (automatic)")
print("=" * 60)

solutions, objectives = multi_objective_optimize(
    objective_fn=three_objective_problem,
    dim=5,
    m=3,
    K=6,           # find 6 Pareto-approximate solutions
    n_steps=300,
    lr=0.02,
)

print(f"\nFound {solutions.shape[0]} solutions:\n")
print(f"  {'#':>3}  {'f1':>8}  {'f2':>8}  {'f3':>8}  {'x (first 3)':>20}")
for i in range(solutions.shape[0]):
    obj = objectives[i]
    x_str = ", ".join(f"{v:.2f}" for v in solutions[i, :3].tolist())
    print(f"  {i+1:>3}  {obj[0]:>8.4f}  {obj[1]:>8.4f}  {obj[2]:>8.4f}  [{x_str}, ...]")


# --- Method 2: Manual control (for custom optimization loops) ---
print(f"\n{'=' * 60}")
print("Method 2: manual STCH usage (for custom workflows)")
print("=" * 60)

m = 3
K = 4

# Step 1: generate structured weights
weights = generate_stch_weights(m=m, K=K, mu=0.01)
print(f"\nGenerated {K} weight vectors:")
for k in range(K):
    w_str = ", ".join(f"{v:.3f}" for v in weights[k].tolist())
    print(f"  w_{k}: [{w_str}]")

# Step 2: use each weight in your own optimization loop
print(f"\nOptimizing with each weight vector:")

for k in range(K):
    w = weights[k]
    x = torch.rand(5, dtype=torch.double, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.02)

    for step in range(300):
        opt.zero_grad()
        obj = three_objective_problem(x)
        # this is the key line — scalarize with STCH
        loss = stch_scalarize(obj, w, mu=0.1)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)

    with torch.no_grad():
        final_obj = three_objective_problem(x)
        print(f"  weight {k}: objectives = [{', '.join(f'{v:.4f}' for v in final_obj.tolist())}]")


# --- Method 3: Visualize STCH vs Chebyshev gradients ---
print(f"\n{'=' * 60}")
print("Method 3: gradient comparison (STCH vs hard Chebyshev)")
print("=" * 60)

y = torch.tensor([0.5, 0.5], dtype=torch.double, requires_grad=True)
w = torch.tensor([0.6, 0.4], dtype=torch.double)

# STCH gradient (smooth)
loss_stch = stch_scalarize(y, w, mu=0.1)
loss_stch.backward()
grad_stch = y.grad.clone()

# Chebyshev gradient (hard max — only one component gets gradient)
y2 = torch.tensor([0.5, 0.5], dtype=torch.double, requires_grad=True)
loss_cheby = (w * y2).max()
loss_cheby.backward()
grad_cheby = y2.grad.clone()

print(f"\n  objectives = [0.5, 0.5], weights = [0.6, 0.4]")
print(f"  Chebyshev gradient: [{', '.join(f'{v:.4f}' for v in grad_cheby.tolist())}]")
print(f"  STCH gradient:      [{', '.join(f'{v:.4f}' for v in grad_stch.tolist())}]")
print(f"\n  Chebyshev: only the dominant objective gets gradient signal.")
print(f"  STCH: both objectives get gradient, proportional to contribution.")
print(f"  This is why STCH converges faster — the optimizer always has direction.")
