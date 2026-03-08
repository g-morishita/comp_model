# How-to: Fit Stan Bayesian Hierarchies (Script API)

Use this guide to fit the Stan-backed Bayesian estimators from a Python script.

The estimator matrix is:

- `subject_shared_stan_nuts` / `subject_shared_stan_map`: one subject, one parameter vector shared across blocks
- `subject_block_hierarchy_stan_nuts` / `subject_block_hierarchy_stan_map`: one subject, hierarchy `subject -> block`
- `study_subject_hierarchy_stan_nuts` / `study_subject_hierarchy_stan_map`: one study, hierarchy `population -> subject`
- `study_subject_block_hierarchy_stan_nuts` / `study_subject_block_hierarchy_stan_map`: one study, hierarchy `population -> subject -> block`

## 1. Prerequisites

- Install and import `comp_model`.
- Install `cmdstanpy` and CmdStan. Stan estimators fail fast if CmdStan is not
  available.
- Prepare a study CSV that can be loaded with
  `read_study_decisions_csv(...)`.

## 2. Write a Python Script

Create `scripts/fit_bayesian_hierarchical.py`:

```python
from __future__ import annotations

from pathlib import Path

from comp_model.inference import (
    infer_study_stan_from_config,
    write_hierarchical_mcmc_study_draw_records_csv,
    write_hierarchical_mcmc_study_summary_csv,
)
from comp_model.io import read_study_decisions_csv


def main() -> None:
    # 1) Load study-level behavioral data from CSV.
    study = read_study_decisions_csv("data/study.csv")

    # 2) Define a population -> subject Stan configuration.
    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            # Keep fixed parameters here. Do not repeat sampled params.
            "kwargs": {"initial_value": 0.0},
        },
        "estimator": {
            "type": "study_subject_hierarchy_stan_nuts",
            "parameter_names": ["alpha", "beta"],
            "transforms": {
                "alpha": "unit_interval_logit",
                "beta": "positive_log",
            },
            "mu_prior_mean": {"alpha": 0.0, "beta": 1.0},
            "mu_prior_std": {"alpha": 1.5, "beta": 1.5},
            "log_sigma_prior_mean": -1.0,
            "log_sigma_prior_std": 1.0,
            "n_samples": 500,
            "n_warmup": 500,
            "thin": 1,
            "n_chains": 4,
            "parallel_chains": 4,
            "adapt_delta": 0.9,
            "max_treedepth": 12,
            "refresh": 50,
            "random_seed": 7,
        },
    }

    # 3) Fit the population -> subject posterior for the whole study.
    result = infer_study_stan_from_config(study, config=config)

    # 4) Inspect high-level summaries.
    print("n_subjects:", result.n_subjects)
    print("total_log_posterior:", result.total_log_posterior)
    print("population_location_z:", result.map_candidate.population_location_z)
    print("mean_map_params_by_subject:", result.mean_map_params_by_subject)

    # 5) Export summary and draw-level CSV artifacts.
    out_dir = Path("fit_out/bayesian_hierarchical")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_hierarchical_mcmc_study_summary_csv(result, out_dir / "study_summary.csv")
    write_hierarchical_mcmc_study_draw_records_csv(result, out_dir / "study_draws.csv")


if __name__ == "__main__":
    main()
```

Run it as a normal Python script:

```bash
python scripts/fit_bayesian_hierarchical.py
```

## 3. Switch to MAP Instead of NUTS

For faster point-estimate fitting, change only the estimator block:

```python
config["estimator"] = {
    "type": "study_subject_hierarchy_stan_map",
    "parameter_names": ["alpha", "beta"],
    "transforms": {"alpha": "unit_interval_logit", "beta": "positive_log"},
    "method": "lbfgs",
    "max_iterations": 2000,
    "random_seed": 7,
    "refresh": 50,
}
```

The return type remains a study-level posterior result container, but it
contains one retained optimized point instead of NUTS draws.

## 4. Choose the Right Estimator

Use these estimator types depending on the hierarchy you want:

- One subject, shared across blocks:
  `subject_shared_stan_nuts` or `subject_shared_stan_map`
- One subject, block-specific parameters:
  `subject_block_hierarchy_stan_nuts` or `subject_block_hierarchy_stan_map`
- One study, subject-specific parameters shared across blocks:
  `study_subject_hierarchy_stan_nuts` or `study_subject_hierarchy_stan_map`
- One study, subject- and block-specific parameters:
  `study_subject_block_hierarchy_stan_nuts` or
  `study_subject_block_hierarchy_stan_map`

The study example above uses `study_subject_hierarchy_stan_*`, which matches
data where each subject has one parameter vector reused across blocks.

## 5. Optional: One-Call CSV + Fit Helper

If you prefer one call that loads CSV and dispatches fit:

```python
from comp_model.inference import fit_study_csv_from_config

study_result = fit_study_csv_from_config(
    "data/study.csv",
    config=config,
    level="study",
)
```

For a subject-level fit from the same CSV, switch the estimator type and call:

```python
subject_config = {
    "model": config["model"],
    "estimator": {
        "type": "subject_block_hierarchy_stan_nuts",
        "parameter_names": ["alpha", "beta"],
        "transforms": {
            "alpha": "unit_interval_logit",
            "beta": "positive_log",
        },
        "n_samples": 500,
        "n_warmup": 500,
    },
}

subject_result = fit_study_csv_from_config(
    "data/study.csv",
    config=subject_config,
    level="subject",
    subject_id="s1",
)
```

## Common Issues

- Unknown config keys raise `ValueError` (schemas are strict).
- `parameter_names` must be unique and valid for the selected model.
- Sampled parameters in `parameter_names` must not also be passed in
  `model.kwargs`.
- Supported transform kinds are `identity`, `unit_interval_logit`,
  and `positive_log`.
- Subject-level estimators require `SubjectData`; study-level estimators require
  `StudyData`.
