<p align="center">
  <h1 align="center">stch-botorch</h1>
  <p align="center">
    <em>smooth tchebycheff scalarization for many-objective bayesian optimization</em>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2405.19650"><img src="https://img.shields.io/badge/arXiv-2405.19650-b31b1b.svg" alt="arXiv"></a>
    <a href="https://github.com/IlkhamFY/stch-botorch-public/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
    <a href="https://botorch.org/"><img src="https://img.shields.io/badge/BoTorch-0.9+-orange.svg" alt="BoTorch"></a>
    <img src="https://img.shields.io/badge/python-3.9+-3776ab.svg" alt="Python">
  </p>
</p>

---

drop-in replacement for `get_chebyshev_scalarization` that makes multi-objective BO work at 50+ objectives.

the standard chebyshev scalarization uses a hard max — non-differentiable, bad gradients, acquisition optimization struggles. stch replaces it with a smooth LogSumExp approximation that gives fully differentiable gradients everywhere while recovering the exact chebyshev solution as temperature → 0.

based on [Lin et al., "Smooth Tchebycheff Scalarization for Multi-Objective Optimization" (ICML 2024)](https://arxiv.org/abs/2405.19650).

## why this exists

bayesian optimization hits a wall at ~12 objectives. not because the algorithms fail — because hypervolume evaluation is exponential in m. this library provides:

1. **stch scalarization** — smooth, differentiable replacement for chebyshev. better gradients → better acquisition optimization → better candidates.
2. **coverage-optimal weights** — STCH-Set weight generation that guarantees all objectives are covered, instead of hoping random dirichlet draws hit everything.
3. **qLogSTCHParEGO** — ready-to-use acquisition function that plugs into any botorch workflow.

we've run this at m=50 on DTLZ2. it works.

## install

```bash
pip install botorch  # >= 0.9
git clone https://github.com/IlkhamFY/stch-botorch-public.git
cd stch-botorch-public
pip install -e .
```

## quickstart

```python
import torch
from stch_botorch import get_stch_scalarization, select_stch_weights

# generate K coverage-optimal weights for m objectives
weights = select_stch_weights(m=8, K=8, mu=0.01)

# use exactly like get_chebyshev_scalarization
scalarization = get_stch_scalarization(
    weights=weights[0], Y=train_obj, mu=0.1
)
utility = scalarization(Y_samples)  # (..., q, m) -> (..., q)
```

### full bo loop (4 lines changed from standard qNParEGO)

```python
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from stch_botorch import get_stch_scalarization, select_stch_weights

# pre-compute structured weights (once)
stch_weights = select_stch_weights(m=m, K=m, mu=0.01)

# inside BO loop: replace get_chebyshev_scalarization
for k in range(K):
    objective = GenericMCObjective(
        get_stch_scalarization(weights=stch_weights[k], Y=pred, mu=0.1)
    )
    acq = qLogNoisyExpectedImprovement(
        model=model, objective=objective,
        X_baseline=train_x, sampler=sampler
    )
```

see [`examples/simple_mobo.py`](examples/simple_mobo.py) for a complete working example.

### standalone (no botorch needed)

if you're not doing bayesian optimization — maybe you're optimizing quantum circuits, neural architectures, or anything else with multiple objectives — use the standalone API:

```python
from stch_botorch.stch_standalone import stch_scalarize, generate_stch_weights

# works with any differentiable function
def my_objectives(x):
    return torch.stack([f1(x), f2(x), f3(x)])

weights = generate_stch_weights(m=3, K=6, mu=0.01)

# optimize each weight vector with your favorite optimizer
for k in range(6):
    loss = stch_scalarize(my_objectives(x), weights[k], mu=0.1)
    loss.backward()  # smooth gradients everywhere
```

or use the all-in-one optimizer:

```python
from stch_botorch.stch_standalone import multi_objective_optimize

solutions, objectives = multi_objective_optimize(
    objective_fn=my_objectives, dim=10, m=3, K=6
)
# solutions: 6 Pareto-approximate points, each targeting a different trade-off
```

see [`examples/standalone_optimization.py`](examples/standalone_optimization.py) and [`examples/vqe_multi_objective.py`](examples/vqe_multi_objective.py) for complete examples.

## what's inside

```
src/stch_botorch/
├── stch_standalone.py         # pure PyTorch — no BoTorch needed
├── scalarization_botorch.py   # BoTorch integration (drop-in replacement)
├── weight_selection.py        # STCH-Set coverage-optimal weights
└── acquisition/
    └── parego_stch.py         # qLogSTCHParEGO acquisition function
```

| component | what it does | needs botorch? |
|-----------|-------------|:-:|
| `stch_scalarize` | scalarize objectives with STCH | no |
| `generate_stch_weights` | coverage-optimal weight vectors | no |
| `multi_objective_optimize` | full multi-objective optimizer | no |
| `get_stch_scalarization` | BoTorch-compatible scalarization | yes |
| `select_stch_weights` | same as generate, with more options | no |
| `qLogSTCHParEGO` | acquisition function for BO | yes |

## key parameters

| parameter | default | what it controls |
|-----------|---------|-----------------|
| `mu` (scalarization) | 0.1 | smoothing temperature. lower = tighter approximation to hard chebyshev. 0.1 works well for most problems. |
| `mu` (weight selection) | 0.01 | coverage temperature. lower = more specialized weights. 0.01 gives good coverage at m=4-50. |
| `K` | m | number of weight vectors per iteration. K=m is a good default. K=2m if you have budget. |

## scaling

tested on DTLZ2 with H100 GPUs:

| objectives | wall time / iter | works? |
|-----------|-----------------|--------|
| m=4 | ~16s | ✓ (HV evaluation feasible) |
| m=8 | ~66s | ✓ |
| m=12 | ~244s | ✓ (HV evaluation starts struggling) |
| m=30 | ~32 min | ✓ (HV-free metrics only) |
| m=50 | ~108 min | ✓ (HV-free metrics only) |

at m>12, hypervolume can't be computed. use R2, IGD+, or WOC (worst-objective coverage) instead.

## citation

```bibtex
@software{yabbarov2026stchbotorch,
  author = {Yabbarov, Ilkham},
  title = {stch-botorch: Smooth Tchebycheff Scalarization for Many-Objective Bayesian Optimization},
  year = {2026},
  url = {https://github.com/IlkhamFY/stch-botorch-public}
}
```

the underlying scalarization is from:
```bibtex
@inproceedings{lin2024smooth,
  title={Smooth Tchebycheff Scalarization for Multi-Objective Optimization},
  author={Lin, Xi and Zhang, Xiaoyuan and Zhong, Zhiyuan and Li, Ke and Zhang, Qingfu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## license

MIT
