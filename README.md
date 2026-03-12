# stch-botorch

Smooth Tchebycheff (STCH) scalarization for multi-objective Bayesian optimization via BoTorch.

Drop-in replacement for `get_chebyshev_scalarization` that uses LogSumExp smoothing instead of the hard max operator, providing fully differentiable gradients for acquisition optimization.

Based on: Lin et al., "Smooth Tchebycheff Scalarization for Multi-Objective Optimization" (ICML 2024).

## Installation

```bash
pip install botorch  # requires BoTorch >= 0.9
pip install -e .
```

## Quick Start

```python
from stch_botorch.scalarization_botorch import get_stch_scalarization
from stch_botorch.weight_selection import select_stch_weights

# Generate coverage-optimal weights for m objectives
weights = select_stch_weights(m=8, K=8, mu=0.01)

# Use as drop-in replacement for get_chebyshev_scalarization
scalarization = get_stch_scalarization(weights=weights[0], Y=train_obj, mu=0.1)
scalarized_values = scalarization(Y_samples)
```

See `examples/simple_mobo.py` for a complete multi-objective BO loop.

## Structure

```
src/stch_botorch/
    scalarization_botorch.py  -- STCH scalarization (core)
    weight_selection.py       -- Weight generation strategies
    acquisition/
        parego_stch.py        -- qLogSTCHParEGO acquisition function
examples/
    simple_mobo.py            -- Minimal MOBO example (DTLZ2, m=4)
```

## Citation

Paper in preparation. If you use this code, please cite:

```bibtex
@software{yabbarov2026stch,
  author = {Yabbarov, Ilkham},
  title = {stch-botorch: Smooth Tchebycheff Scalarization for Multi-Objective Bayesian Optimization},
  year = {2026},
  url = {https://github.com/IlkhamFY/stch-botorch-public}
}
```

## License

MIT
