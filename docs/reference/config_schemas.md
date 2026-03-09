# Config Schemas

This document defines the strict config shapes accepted by the public MLE,
model-comparison, and recovery APIs. Unknown keys fail fast with a
`ValueError`.

## Common Types

```python
ComponentRef = {
    "component_id": str,
    "kwargs": dict[str, Any],  # optional, defaults to {}
}
```

## Inference Configs

### MLE (`fit_trace_from_config`, `fit_block_from_config`, ...)

```python
MLEFitConfig = {
    "model": ComponentRef,
    "estimator": MLEEstimator,
    "likelihood": LikelihoodConfig,  # optional
    "block_fit_strategy": "independent" | "joint",  # subject/study only
}

MLEEstimator = {
    "type": "mle",
    "solver": "grid_search" | "scipy_minimize" | "transformed_scipy_minimize",  # optional
}

GridSearchEstimator = {
    "type": "mle",
    "solver": "grid_search",  # optional when parameter_grid is present
    "parameter_grid": dict[str, list[float]],
}

ScipyMinimizeEstimator = {
    "type": "mle",
    "solver": "scipy_minimize",  # optional default when no grid/transforms are given
    "initial_params": dict[str, float],
    "bounds": dict[str, tuple[float | None, float | None]],  # optional
    "method": str,  # optional, default "L-BFGS-B"
    "tol": float,   # optional
    "n_starts": int,            # optional, default 5
    "random_seed": int | None,  # optional, default 0
}

TransformedScipyMinimizeEstimator = {
    "type": "mle",
    "solver": "transformed_scipy_minimize",  # optional when bounds_z/transforms are present
    "initial_params": dict[str, float],
    "bounds_z": dict[str, tuple[float | None, float | None]],  # optional
    "transforms": dict[str, "identity" | "unit_interval_logit" | "positive_log"],  # optional
    "method": str,  # optional, default "L-BFGS-B"
    "tol": float,   # optional
    "n_starts": int,            # optional, default 5
    "random_seed": int | None,  # optional, default 0
}
```

Notes:

- `fit_trace_auto_from_config(...)`, `fit_block_auto_from_config(...)`,
  `fit_subject_auto_from_config(...)`, and `fit_study_auto_from_config(...)`
  currently accept only `estimator.type == "mle"`.
- `block_fit_strategy` is only valid for subject- and study-level fitting.

### Model Selection (`compare_*_candidates_from_config`)

```python
ModelSelectionConfig = {
    "candidates": list[CandidateConfig],
    "criterion": "log_likelihood" | "aic" | "bic",  # optional
    "n_observations": int,  # dataset-level only, optional
    "likelihood": LikelihoodConfig,  # optional global default
    "block_fit_strategy": "independent" | "joint",  # subject/study only
}

CandidateConfig = {
    "name": str,
    "model": ComponentRef,
    "estimator": MLEEstimator,
    "likelihood": LikelihoodConfig,  # optional candidate override
    "n_parameters": int,             # optional
}
```

### Likelihood Config

```python
LikelihoodConfig = {
    "type": "action_replay",
} | {
    "type": "actor_subset_replay",
    "fitted_actor_id": str,                # optional, default "subject"
    "scored_actor_ids": list[str] | None,  # optional, default ["subject"]
    "auto_fill_unmodeled_actors": bool,    # optional, default True
}
```

## Recovery Configs

### Distribution Specs Used by Recovery Sampling

```python
DistributionSpec = (
    {"distribution": "normal", "mean": float, "std": float}
    | {"distribution": "uniform", "lower": float | None, "upper": float | None}
    | {"distribution": "beta", "alpha": float, "beta": float}
    | {"distribution": "log_normal", "mean_log": float, "std_log": float}
)
```

### Parameter Recovery

```python
ParameterRecoveryConfig = {
    "problem": ComponentRef,          # required when simulation omitted
    "simulation": SimulationConfig,   # optional
    "generating_model": ComponentRef,
    "fitting_model": ComponentRef,
    "estimator": MLEEstimator,
    "likelihood": LikelihoodConfig,   # optional
    "block_fit_strategy": "independent" | "joint",  # optional
    # exactly one of:
    "true_parameter_sets": list[dict[str, float]],
    "true_parameter_distributions": dict[str, DistributionSpec],
    "sampling": ParameterSamplingConfig,
    "n_parameter_sets": int,  # required with true_parameter_distributions
    "n_trials": int,
    "seed": int,              # optional
}

ParameterSamplingConfig = {
    "mode": "independent" | "hierarchical",
    "space": "param" | "z",  # optional, default "param"
    "n_parameter_sets": int,  # optional fallback to top-level n_parameter_sets
    "distributions": dict[str, DistributionSpec],  # independent mode
    "population": dict[str, DistributionSpec],     # hierarchical mode
    "individual_sd": dict[str, float],             # hierarchical mode
    "transforms": dict[str, "identity" | "unit_interval_logit" | "positive_log"],  # optional
    "bounds": dict[str, [float | None, float | None]],  # optional
    "clip_to_bounds": bool,  # optional
    "by_condition": dict[str, {
        "distributions": dict[str, DistributionSpec],  # independent mode
        "population": dict[str, DistributionSpec],     # hierarchical mode
        "individual_sd": dict[str, float],             # hierarchical mode
    }],
    "conditions": list[str],   # required when by_condition is used
    "baseline_condition": str, # required when by_condition is used
}
```

Notes:

- Parameter recovery currently supports `simulation.level == "block"` and
  `simulation.level == "subject"`.
- Subject-level recovery with one shared parameter vector across blocks requires
  `block_fit_strategy: "joint"`.

### Model Recovery

```python
ModelRecoveryConfig = {
    "problem": ComponentRef,        # required when simulation omitted
    "simulation": SimulationConfig, # optional
    "generating": list[{
        "name": str,
        "model": ComponentRef,
        "true_params": dict[str, float],  # optional
    }],
    "candidates": list[CandidateConfig],
    "likelihood": LikelihoodConfig,  # optional global default
    "block_fit_strategy": "independent" | "joint",  # optional
    "n_trials": int,
    "n_replications_per_generator": int,
    "criterion": "log_likelihood" | "aic" | "bic",  # optional
    "seed": int,  # optional
}
```

### Simulation Config

```python
SimulationConfig = {
    "type": "problem" | "generator",
    "level": "block" | "subject" | "study",  # optional, default "block"
    "problem": ComponentRef,  # optional fallback to top-level problem
    "generator": ComponentRef,
    "demonstrator_model": ComponentRef,  # required for social generators
    "block": BlockSpec,                  # mutually exclusive with "blocks"
    "blocks": list[BlockSpec],           # mutually exclusive with "block"
    "subject_id": str,                   # optional for subject-level generation
    "subject_ids": list[str],            # optional for study-level generation
    "n_subjects": int,                   # optional for study-level generation
}

BlockSpec = {
    "n_trials": int,  # optional fallback to top-level n_trials
    "block_id": str | int,            # optional
    "metadata": dict[str, Any],       # optional
    "problem_kwargs": dict[str, Any], # asocial generators
    "program_kwargs": dict[str, Any], # social generators
}
```
