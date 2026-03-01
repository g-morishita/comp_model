# Tutorial: First End-to-End Fit

This tutorial walks through one complete loop:

1. simulate data from a decision problem,
2. fit a model on that data,
3. inspect fitted parameters.

By the end, you will have a minimal working workflow you can adapt.

## Prerequisites

- Python 3.11+
- `comp_model` installed in your environment

## Step 1: Simulate One Dataset

```python
from comp_model.problems import StationaryBanditProblem
from comp_model.models import AsocialStateQValueSoftmaxModel
from comp_model.runtime import SimulationConfig, run_episode

problem = StationaryBanditProblem(reward_probabilities=[0.2, 0.8])
model = AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0)

trace = run_episode(
    problem=problem,
    model=model,
    config=SimulationConfig(n_trials=120, seed=10),
)
```

`trace` is the canonical event sequence used by replay and inference.

## Step 2: Fit a Model

```python
from comp_model.inference import FitSpec, fit_model
from comp_model.models import AsocialStateQValueSoftmaxModel

fit_result = fit_model(
    trace,
    model_factory=lambda params: AsocialStateQValueSoftmaxModel(**params),
    fit_spec=FitSpec(
        estimator_type="grid_search",
        parameter_grid={
            "alpha": [0.1, 0.2, 0.3, 0.4],
            "beta": [1.0, 2.0, 3.0, 4.0],
            "initial_value": [0.0],
        },
    ),
)
```

## Step 3: Inspect Results

```python
print("best log-likelihood:", fit_result.best.log_likelihood)
print("best params:", fit_result.best.params)
```

You now have a complete simulation-to-fit pipeline.

## Next Steps

- To fit real datasets, continue with
  [Fit Existing CSV Data](../how-to/fit-existing-csv-data.md).
- To compare multiple model candidates, see
  [Compare Candidate Models](../how-to/compare-candidate-models.md).
