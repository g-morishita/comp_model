# Tutorial: End-to-End Simulation and Hierarchical Bayesian Fit

In this tutorial, you will run a full simulation-to-inference workflow using
within-subject hierarchical Bayesian estimation (Stan-backed).

Why this version matters:

- You still validate the full pipeline with synthetic data.
- You move from point-estimate fitting to posterior-based fitting.
- You can quantify uncertainty and stabilize estimates across blocks.

This tutorial uses:

- task: `StationaryBanditProblem`
- model family: `AsocialStateQValueSoftmaxModel`
- estimator: `within_subject_hierarchical_stan_nuts`

## Prerequisites

- Python 3.11+
- Working installation of `comp_model`
- `cmdstanpy` and CmdStan installed (required for Stan estimators)

If you have not installed and verified your environment yet, complete
[Install and Verify](install-and-verify.md) first.

??? note "Hierarchy level in this tutorial"
    In this library, `within_subject_hierarchical_stan_*` means hierarchical
    inference within each subject while preserving trial dynamics per block.

    For non-pooled Stan estimators, block-level latent parameters are indexed by
    block condition labels (`block.metadata["condition"]`,
    `block.metadata["block_condition"]`, or
    `block.metadata["condition_label"]`).

    If two blocks have the same condition label, they share one latent
    parameter estimate.

    If no condition label is provided, each block gets its own latent estimate.

    When you pass `StudyData`, each subject is fit independently with the same
    estimator configuration, and results are returned as a study-level
    collection.

## Step 1: Define task and model family

We will use a simple two-armed stationary bandit and an asocial state-indexed
Q-learning model with softmax choice.

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
This gives us a realistic `StudyData` object for hierarchical fitting.

```python
from __future__ import annotations

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

## Step 3: Build hierarchical Bayesian config (NUTS)

Define a Stan estimator config for hierarchical posterior sampling.

```python
hierarchical_config = {
    "model": {
        "component_id": "asocial_state_q_value_softmax",
        "kwargs": {
            "initial_value": 0.0,  # fixed (not sampled)
        },
    },
    "estimator": {
        "type": "within_subject_hierarchical_stan_nuts",
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

## Step 4: Fit hierarchical posterior

Now run hierarchical Bayesian fitting over the whole study.

```python
from comp_model.inference import sample_study_hierarchical_posterior_from_config

study_result = sample_study_hierarchical_posterior_from_config(
    study,
    config=hierarchical_config,
)

print("n_subjects:", study_result.n_subjects)
print("total_map_log_posterior:", study_result.total_map_log_posterior)
```

## Step 5: Inspect posterior output and compare to generating values

Inspect per-subject summaries from posterior draws.

```python
for subject_result in study_result.subject_results:
    sid = subject_result.subject_id
    print("subject:", sid)
    print("  draws:", len(subject_result.draws))
    print("  sampler method:", subject_result.diagnostics.method)
    print("  acceptance rate:", subject_result.diagnostics.acceptance_rate)
    print("  true params:", true_params_by_subject[sid])
    print("  mean MAP params:", subject_result.mean_map_params)
```

What to check:

1. draws are present for every subject,
2. diagnostics look reasonable (for example, acceptance rate not degenerate),
3. estimated values are directionally close to generating values.

## Step 6: Export result artifacts

Write summary and draw-level CSVs for downstream analysis.

```python
from pathlib import Path

from comp_model.inference import (
    write_hierarchical_mcmc_study_draw_records_csv,
    write_hierarchical_mcmc_study_summary_csv,
)

out_dir = Path("fit_out/hierarchical_end_to_end")
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = write_hierarchical_mcmc_study_summary_csv(
    study_result,
    out_dir / "study_summary.csv",
)
draws_path = write_hierarchical_mcmc_study_draw_records_csv(
    study_result,
    out_dir / "study_draws.csv",
)

print("summary:", summary_path)
print("draws:", draws_path)
```

## Step 7: Optional MAP variant (faster, no posterior sample)

If you want a fast optimization-based variant, switch estimator type:

```python
map_config = {
    "model": hierarchical_config["model"],
    "estimator": {
        "type": "within_subject_hierarchical_stan_map",
        "parameter_names": ["alpha", "beta"],
        "transforms": {
            "alpha": "unit_interval_logit",
            "beta": "positive_log",
        },
        "mu_prior_mean": {"alpha": 0.0, "beta": 1.0},
        "mu_prior_std": {"alpha": 1.5, "beta": 1.5},
        "log_sigma_prior_mean": -1.0,
        "log_sigma_prior_std": 1.0,
        "method": "lbfgs",
        "max_iterations": 2000,
        "random_seed": 42,
        "refresh": 50,
    },
}

map_result = sample_study_hierarchical_posterior_from_config(study, config=map_config)
print("MAP-style total log posterior:", map_result.total_map_log_posterior)
```

## Next Steps

- Continue with [Parameter Recovery](parameter-recovery.md).
- For a focused API guide, see
  [How to fit Bayesian hierarchical model](../how-to/how-to-fit-bayesian-hierarchical-model.md).

## References

- Wilson RC, Collins AGE. (2019). Ten simple rules for the computational
  modeling of behavioral data. *eLife*, 8:e49547.
  <https://doi.org/10.7554/eLife.49547>
