"""
Multi-objective optimization for Variational Quantum Eigensolvers (VQE).

This example shows how STCH scalarization can optimize quantum circuit
parameters with multiple competing objectives:

    1. Energy accuracy (minimize energy error)
    2. Circuit depth (minimize number of entangling gates)
    3. Noise robustness (maximize fidelity under depolarizing noise)

The idea: instead of optimizing a single VQE loss, use STCH to explore
the Pareto front of accuracy-vs-depth-vs-noise trade-offs. Different
weight vectors give you circuits optimized for different regimes:
  - High-accuracy deep circuits (for simulators)
  - Shallow noisy circuits (for near-term quantum hardware)
  - Balanced trade-offs (for intermediate applications)

NOTE: This is a toy model using PyTorch to simulate the VQE landscape.
Replace `mock_vqe_objectives` with your actual Qiskit/PennyLane circuit
evaluation to use on real quantum problems.
"""

import torch
import math
from stch_botorch.stch_standalone import (
    stch_scalarize,
    generate_stch_weights,
    multi_objective_optimize,
)


def mock_vqe_objectives(params):
    """
    Simulated multi-objective VQE landscape.

    params: (n_params,) tensor of circuit parameters (rotation angles).

    Returns (3,) tensor of objectives to MINIMIZE:
        f1: energy error — how far from the ground state energy
        f2: effective depth — proxy for circuit complexity (entanglement cost)
        f3: noise sensitivity — how much energy changes under parameter noise

    In a real implementation, you'd replace this with:
        - f1: <psi(params)|H|psi(params)> - E_exact
        - f2: circuit.depth() or count_entangling_gates(circuit)
        - f3: E_noisy(params) - E_clean(params) (noise simulation)
    """
    n = params.shape[0]

    # f1: energy error — multi-modal landscape with global minimum
    # simulates the VQE energy surface with local minima
    energy = 0.0
    for i in range(n):
        energy = energy + torch.sin(params[i] * 2) ** 2
        if i > 0:
            energy = energy + 0.3 * torch.cos(params[i] - params[i-1])
    energy_error = energy / n + 0.1  # shift so minimum > 0

    # f2: effective depth — larger parameter magnitudes = deeper circuits
    # (proxy: circuits with small rotations can be compiled to fewer gates)
    depth_proxy = (params ** 2).mean() + 0.2 * (params[1:] - params[:-1]).abs().mean()

    # f3: noise sensitivity — how much the energy changes with noise
    # parameters near pi/2 are more sensitive to noise (steeper gradients)
    noise_sensitivity = torch.tensor(0.0, dtype=params.dtype)
    for i in range(n):
        # sensitivity ~ |d(energy)/d(param_i)| at this point
        noise_sensitivity = noise_sensitivity + (2 * torch.sin(params[i] * 2) * torch.cos(params[i] * 2) * 2).abs()
    noise_sensitivity = noise_sensitivity / n

    return torch.stack([energy_error, depth_proxy, noise_sensitivity])


# --- Run multi-objective optimization ---
print("=" * 70)
print("Multi-Objective VQE Circuit Optimization with STCH")
print("=" * 70)

n_params = 8  # circuit parameters (e.g., 8 rotation angles)
m = 3         # objectives: energy, depth, noise
K = 6         # find 6 different trade-off solutions

print(f"\nOptimizing {n_params}-parameter circuit with {m} objectives...")
print(f"Finding {K} Pareto-approximate solutions...\n")

solutions, objectives = multi_objective_optimize(
    objective_fn=mock_vqe_objectives,
    dim=n_params,
    m=m,
    bounds=(-math.pi, math.pi),  # rotation angles
    K=K,
    mu_weights=0.01,
    mu_scalarize=0.1,
    n_steps=500,
    lr=0.02,
    n_random_starts=3,
)

# --- Display results ---
print(f"{'Solution':>10}  {'Energy Err':>12}  {'Depth':>12}  {'Noise Sens':>12}  {'Profile'}")
print("-" * 70)

for i in range(K):
    obj = objectives[i]
    e, d, n = obj[0].item(), obj[1].item(), obj[2].item()

    # classify the solution
    if e < 0.15 and d > 0.3:
        profile = "high-accuracy (deep)"
    elif d < 0.15 and e > 0.2:
        profile = "shallow (hardware)"
    elif n < 0.15:
        profile = "noise-robust"
    else:
        profile = "balanced"

    print(f"  {i+1:>6}    {e:>12.4f}  {d:>12.4f}  {n:>12.4f}  {profile}")

print(f"\n{'=' * 70}")
print("How to use this with real VQE:")
print("=" * 70)
print("""
1. Replace mock_vqe_objectives with your actual circuit evaluation:

    def real_vqe_objectives(params):
        circuit = build_ansatz(params)  # your Qiskit/PennyLane circuit
        energy = estimate_energy(circuit, hamiltonian)
        depth = circuit.depth()
        noise = estimate_noise_impact(circuit, noise_model)
        return torch.tensor([energy, depth, noise])

2. If your circuit evaluation is NOT differentiable (e.g., shot-based),
   use stch_scalarize inside a black-box optimizer:

    from stch_botorch.stch_standalone import stch_scalarize, generate_stch_weights
    weights = generate_stch_weights(m=3, K=6)

    for k in range(K):
        def loss_fn(params):
            obj = evaluate_circuit(params)  # non-differentiable
            return stch_scalarize(torch.tensor(obj), weights[k], mu=0.1)

        result = scipy.optimize.minimize(loss_fn, x0, method='COBYLA')

3. For DVR-VQE specifically: the objectives could be:
   - Energy accuracy per vibrational state
   - Total measurement circuits needed (from DVR decomposition)
   - Circuit depth (entangling gate count)
   - Robustness across electronic states

   STCH lets you explore all these trade-offs simultaneously instead of
   picking one loss function and hoping for the best.
""")
