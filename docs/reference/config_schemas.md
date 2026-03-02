# Config Schemas

This document defines strict config shapes accepted by the public config APIs.
Unknown keys fail fast with a `ValueError`.

## Common Types

```python
ComponentRef = {
    "component_id": str,
    "kwargs": dict[str, Any],  # optional, defaults to {}
}
```

## Inference Configs

### MLE (`fit_dataset_from_config`, `fit_block_from_config`, ...)

```python
MLEFitConfig = {
    "model": ComponentRef,
    "estimator": GridSearchEstimator | ScipyMinimizeEstimator | TransformedScipyMinimizeEstimator,
    "likelihood": LikelihoodConfig,  # optional
    "block_fit_strategy": "independent" | "joint",  # optional; subject/study APIs only
}

GridSearchEstimator = {
    "type": "grid_search",
    "parameter_grid": dict[str, list[float]],
}

ScipyMinimizeEstimator = {
    "type": "scipy_minimize",
    "initial_params": dict[str, float],
    "bounds": dict[str, tuple[float | None, float | None]],  # optional
    "method": str,  # optional
    "tol": float,   # optional
}

TransformedScipyMinimizeEstimator = {
    "type": "transformed_scipy_minimize",
    "initial_params": dict[str, float],
    "bounds_z": dict[str, tuple[float | None, float | None]],  # optional
    "transforms": dict[str, str | {"kind": str}],              # optional
    "method": str,  # optional
    "tol": float,   # optional
}
```

### MAP (`fit_map_*_from_config`)

```python
MAPFitConfig = {
    "model": ComponentRef,
    "prior": PriorConfig,
    "estimator": ScipyMapEstimator | TransformedScipyMapEstimator,
    "likelihood": LikelihoodConfig,  # optional
    "block_fit_strategy": "independent" | "joint",  # optional; subject/study APIs only
}

ScipyMapEstimator = {
    "type": "scipy_map",
    "initial_params": dict[str, float],
    "bounds": dict[str, tuple[float | None, float | None]],  # optional
    "method": str,  # optional
    "tol": float,   # optional
}

TransformedScipyMapEstimator = {
    "type": "transformed_scipy_map",
    "initial_params": dict[str, float],
    "bounds_z": dict[str, tuple[float | None, float | None]],  # optional
    "transforms": dict[str, str | {"kind": str}],              # optional
    "method": str,  # optional
    "tol": float,   # optional
}
```

### Stan Posterior Sampling (`sample_*_hierarchical_posterior_from_config`)

```python
StanPosteriorConfig = {
    "model": ComponentRef,
    "estimator": HierarchicalStanNUTSEstimator,
}

HierarchicalStanNUTSEstimator = {
    "type": "within_subject_hierarchical_stan_nuts",
    "parameter_names": list[str],
    "transforms": dict[str, str | {"kind": str}],  # optional
    "initial_group_location": dict[str, float],     # optional
    "initial_group_scale": dict[str, float],        # optional
    "initial_block_params": list[dict[str, float]], # optional (subject-level)
    "initial_block_params_by_subject": dict[str, list[dict[str, float]]],  # optional (study-level)
    "mu_prior_mean": float,         # optional
    "mu_prior_std": float,          # optional
    "log_sigma_prior_mean": float,  # optional
    "log_sigma_prior_std": float,   # optional
    "n_samples": int,
    "n_warmup": int,                # optional
    "thin": int,                    # optional
    "n_chains": int,                # optional
    "parallel_chains": int,         # optional
    "adapt_delta": float,           # optional
    "max_treedepth": int,           # optional
    "step_size": float,             # optional
    "refresh": int,                 # optional
    "random_seed": int,             # optional
}
```

### Model Selection (`compare_*_candidates_from_config`)

```python
ModelSelectionConfig = {
    "candidates": list[CandidateConfig],
    "criterion": "log_likelihood" | "aic" | "bic" | "waic" | "psis_loo",  # optional
    "n_observations": int,  # dataset-level only, optional
    "likelihood": LikelihoodConfig,  # optional global default
    "block_fit_strategy": "independent" | "joint",  # optional; subject/study APIs only
}

CandidateConfig = {
    "name": str,
    "model": ComponentRef,
    "estimator": dict[str, Any],  # estimator schema above
    "prior": PriorConfig,         # required for MAP estimators
    "likelihood": LikelihoodConfig,  # optional candidate override
    "n_parameters": int,          # optional
}
```

### Likelihood Config

```python
LikelihoodConfig = {
    "type": "action_replay"
} | {
    "type": "actor_subset_replay",
    "fitted_actor_id": str,                # optional, default "subject"
    "scored_actor_ids": list[str] | None,  # optional
    "auto_fill_unmodeled_actors": bool,    # optional, default True
}
```

### Prior Config

```python
PriorConfig = {
    "type": "independent",
    "parameters": dict[str, PriorDistribution],  # recommended form
    "require_all": bool,  # optional
}

PriorDistribution = (
    {"distribution": "normal", "mean": float, "std": float}
    | {"distribution": "uniform", "lower": float | None, "upper": float | None}
    | {"distribution": "beta", "alpha": float, "beta": float}
    | {"distribution": "log_normal", "mean_log": float, "std_log": float}
)
```

## Recovery Configs

### Parameter Recovery

```python
ParameterRecoveryConfig = {
    "problem": ComponentRef,     # required when simulation omitted
    "simulation": SimulationConfig,  # optional
    "generating_model": ComponentRef,
    "fitting_model": ComponentRef,
    "estimator": dict[str, Any],
    "prior": PriorConfig,        # optional
    "likelihood": LikelihoodConfig,  # optional
    "true_parameter_sets": list[dict[str, float]],
    "n_trials": int,
    "seed": int,                 # optional
}
```

Note: parameter recovery currently requires `simulation.level == "block"`.

### Model Recovery

```python
ModelRecoveryConfig = {
    "problem": ComponentRef,     # required when simulation omitted
    "simulation": SimulationConfig,  # optional
    "generating": list[{
        "name": str,
        "model": ComponentRef,
        "true_params": dict[str, float],  # optional
    }],
    "candidates": list[CandidateConfig],
    "likelihood": LikelihoodConfig,  # optional global default
    "block_fit_strategy": "independent" | "joint",  # optional; subject/study simulation levels
    "n_trials": int,
    "n_replications_per_generator": int,
    "criterion": "log_likelihood" | "aic" | "bic" | "waic" | "psis_loo",  # optional
    "seed": int,  # optional
}
```

### Simulation Config

```python
SimulationConfig = {
    "type": "problem" | "generator",
    "level": "block" | "subject" | "study",  # optional, default "block"
    # problem mode:
    "problem": ComponentRef,  # optional fallback to top-level "problem"
    # generator mode:
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
    "block_id": str | int,              # optional
    "metadata": dict[str, Any],         # optional
    "problem_kwargs": dict[str, Any],   # asocial generators
    "program_kwargs": dict[str, Any],   # social generators
}
```

## Example: Social Study-Level Model Recovery

```json
{
  "simulation": {
    "type": "generator",
    "level": "study",
    "n_subjects": 2,
    "generator": {
      "component_id": "event_trace_social_pre_choice_generator",
      "kwargs": {}
    },
    "demonstrator_model": {
      "component_id": "fixed_sequence_demonstrator",
      "kwargs": {
        "sequence": [1, 1, 1, 1, 1]
      }
    },
    "blocks": [
      {
        "n_trials": 40,
        "program_kwargs": {
          "reward_probabilities": [0.2, 0.8]
        }
      }
    ]
  },
  "generating": [
    {
      "name": "qrl_generator",
      "model": {
        "component_id": "asocial_state_q_value_softmax",
        "kwargs": {}
      },
      "true_params": {
        "alpha": 0.3,
        "beta": 2.0,
        "initial_value": 0.0
      }
    }
  ],
  "candidates": [
    {
      "name": "candidate_good",
      "model": {
        "component_id": "asocial_state_q_value_softmax",
        "kwargs": {}
      },
      "estimator": {
        "type": "grid_search",
        "parameter_grid": {
          "alpha": [0.3],
          "beta": [2.0],
          "initial_value": [0.0]
        }
      },
      "n_parameters": 3
    }
  ],
  "likelihood": {
    "type": "actor_subset_replay",
    "fitted_actor_id": "subject",
    "scored_actor_ids": ["subject"],
    "auto_fill_unmodeled_actors": true
  },
  "n_trials": 40,
  "n_replications_per_generator": 2,
  "criterion": "log_likelihood",
  "seed": 123
}
```
