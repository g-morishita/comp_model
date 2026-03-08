# Tutorial: End-to-End Simulation and Hierarchical Bayesian Fit

This tutorial runs a full simulation-to-inference workflow using a study-level
Stan Bayesian hierarchy. This is an advanced workflow; if you want a lighter
starting point, skip to [Parameter Recovery](parameter-recovery.md).

## Why this matters

- You still validate the full pipeline with synthetic data.
- You move from point-estimate fitting to posterior-based fitting.
- You estimate subject parameters jointly through a population-level prior.

This tutorial uses:

- task: `StationaryBanditProblem`
- model family: `AsocialStateQValueSoftmaxModel`
- estimator: `study_subject_hierarchy_stan_nuts`

In this tutorial, you will:

1. define a task and model family,
2. simulate a multi-subject, multi-block study dataset,
3. define a population -> subject Stan configuration (NUTS),
4. fit the posterior for the whole study,
5. inspect posterior output and compare to generating values.

## Prerequisites

- Python 3.11+
- Working installation of `comp_model`
- `cmdstanpy` and CmdStan installed (required for Stan estimators)

If you have not installed and verified your environment yet, complete
[Install and Verify](install-and-verify.md) first.

??? note "Hierarchy level in this tutorial"
    This tutorial uses `study_subject_hierarchy_stan_*`, which means a true
    study-level hierarchy `population -> subject`.

    Each subject gets one latent parameter vector, and that vector is shared
    across the subject's blocks.

    If you want block-specific parameters inside each subject, switch to
    `study_subject_block_hierarchy_stan_*`.

## Step 1: Define task and model family

We will use a simple two-armed stationary bandit and an asocial state-indexed Q-learning model with softmax choice.

```python
from comp_model.models import AsocialStateQValueSoftmaxModel
from comp_model.problems import StationaryBanditProblem

problem = StationaryBanditProblem(reward_probabilities=[0.2, 0.8])

# Example model instance (used for simulation).
example_model = AsocialStateQValueSoftmaxModel(
    alpha=0.25,
    beta=3.0,
    initial_value=0.0,
)
```

## Step 2: Simulate a multi-subject, multi-block study dataset

Now simulate synthetic data for multiple subjects and blocks.
This gives us a realistic `StudyData` object for study-level Bayesian fitting.

```python
from comp_model.generators import (
    AsocialBlockSpec,
    simulate_asocial_study_dataset_with_sampled_subject_params,
)
from comp_model.models import AsocialStateQValueSoftmaxModel

n_subjects = 6
n_blocks_per_subject = 2
n_trials_per_block = 80

blocks = tuple(
    AsocialBlockSpec(
        n_trials=n_trials_per_block,
        block_id=f"b{block_index + 1}",
        problem_kwargs={"reward_probabilities": [0.2, 0.8]},
    )
    for block_index in range(n_blocks_per_subject)
)

simulation = simulate_asocial_study_dataset_with_sampled_subject_params(
    n_subjects=n_subjects,
    blocks=blocks,
    model_factory=lambda params: AsocialStateQValueSoftmaxModel(
        alpha=params["alpha"],
        beta=params["beta"],
        initial_value=0.0,
    ),
    true_parameter_distributions={
        "alpha": {"distribution": "beta", "alpha": 3.5, "beta": 10.5},
        "beta": {"distribution": "log_normal", "mean_log": 1.0, "std_log": 0.25},
    },
    seed=21,
)
study = simulation.study
true_params_by_subject = simulation.true_params_by_subject

print("n_subjects:", study.n_subjects)
print("total_blocks:", sum(len(subject.blocks) for subject in study.subjects))
print("total_trials:", sum(block.n_trials for subject in study.subjects for block in subject.blocks))
```

### Quick quiz

??? question "What increases posterior precision more directly: more subjects or more trials per block?"
    Both help, but more trials per block usually improves per-subject parameter
    estimation directly, while more subjects improves population-level stability.

## Step 3: Build study-level Bayesian config (NUTS)

Define a Stan estimator config for population -> subject posterior sampling.

```python
hierarchical_config = {
    "model": {
        "component_id": "asocial_state_q_value_softmax",
        "kwargs": {
            "initial_value": 0.0,  # fixed (not sampled)
        },
    },
    "estimator": {
        "type": "study_subject_hierarchy_stan_nuts",
        "parameter_names": ["alpha", "beta"],
        "transforms": {
            "alpha": "unit_interval_logit",
            "beta": "positive_log",
        },
        "mu_prior_mean": {
            "alpha": 0.0,
            "beta": 1.0,
        },
        "mu_prior_std": {
            "alpha": 1.5,
            "beta": 1.5,
        },
        "log_sigma_prior_mean": -1.0,
        "log_sigma_prior_std": 1.0,
        "n_samples": 400,
        "n_warmup": 400,
        "thin": 1,
        "n_chains": 4,
        "parallel_chains": 4,
        "adapt_delta": 0.9,
        "max_treedepth": 12,
        "random_seed": 42,
        "refresh": 50,
    },
}
```

Key points:

- `parameter_names` are the free parameters to estimate.
- `transforms` map constrained model parameters to unconstrained Stan space.
- `model.kwargs` should include only fixed parameters.
- This estimator shares one parameter vector across each subject's blocks.

## Step 4: Fit hierarchical posterior

Now run population -> subject Bayesian fitting over the whole study.

```python
from comp_model.inference import infer_study_stan_from_config

study_result = infer_study_stan_from_config(
    study,
    config=hierarchical_config,
)

print("n_subjects:", study_result.n_subjects)
print("n_draws:", len(study_result.draws))
print("sampler_method:", study_result.diagnostics.method)
print("total_log_posterior:", study_result.total_log_posterior)
```

## Step 5: Inspect posterior output and compare to generating values

Inspect per-subject summaries from posterior draws.

```python
print("population_location_z:", study_result.map_candidate.population_location_z)

for sid, fitted_params in study_result.mean_map_params_by_subject.items():
    print("subject:", sid)
    print("  true params:", true_params_by_subject[sid])
    print("  mean MAP params:", fitted_params)
```

What to check:

1. the study result contains retained draws,
2. diagnostics look reasonable,
3. each subject's fitted values are directionally close to generating values.

## Next steps

- Continue with [Parameter Recovery](parameter-recovery.md).
- For a focused API guide, see
  [How to fit Bayesian hierarchical model](../how-to/how-to-fit-bayesian-hierarchical-model.md).

## References

- Wilson RC, Collins AGE. (2019). Ten simple rules for the computational
  modeling of behavioral data. *eLife*, 8:e49547.
  <https://doi.org/10.7554/eLife.49547>
