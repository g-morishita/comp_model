# How-to: Fit a Bayesian Hierarchical Model (Script API)

Use this guide to fit Stan-backed Bayesian hierarchical models from a Python script.

This workflow uses the within-subject hierarchical estimators:

- `within_subject_hierarchical_stan_nuts`
- `within_subject_hierarchical_stan_map`
- `within_subject_pooled_stan_nuts`
- `within_subject_pooled_stan_map`

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
    sample_study_hierarchical_posterior_from_config,
    write_hierarchical_mcmc_study_draw_records_csv,
    write_hierarchical_mcmc_study_summary_csv,
)
from comp_model.io import read_study_decisions_csv


def main() -> None:
    # 1) Load study-level behavioral data from CSV.
    study = read_study_decisions_csv("data/study.csv")

    # 2) Define hierarchical Stan configuration.
    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            # Keep fixed parameters here. Do not repeat sampled params.
            "kwargs": {"initial_value": 0.0},
        },
        "estimator": {
            "type": "within_subject_hierarchical_stan_nuts",
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

    # 3) Fit hierarchical posterior for each subject in the study.
    result = sample_study_hierarchical_posterior_from_config(study, config=config)

    # 4) Inspect high-level summaries.
    print("n_subjects:", result.n_subjects)
    print("total_map_log_posterior:", result.total_map_log_posterior)
    for subject_result in result.subject_results:
        print(
            subject_result.subject_id,
            "method=", subject_result.diagnostics.method,
            "draws=", len(subject_result.draws),
            "mean_map_params=", subject_result.mean_map_params,
        )

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
    "type": "within_subject_hierarchical_stan_map",
    "parameter_names": ["alpha", "beta"],
    "transforms": {"alpha": "unit_interval_logit", "beta": "positive_log"},
    "method": "lbfgs",
    "max_iterations": 2000,
    "random_seed": 7,
    "refresh": 50,
}
```

The return type remains hierarchical posterior result containers, but each
subject has one retained optimized draw.

## 4. Use Pooled-Within-Subject Instead of Hierarchical-Within-Subject

If you want one shared parameter set across blocks (per subject), use pooled
estimator types:

- NUTS: `within_subject_pooled_stan_nuts`
- MAP: `within_subject_pooled_stan_map`

Only change `config["estimator"]["type"]`; the rest of the script is the same.

## 5. Optional: One-Call CSV + Fit Helper

If you prefer one call that loads CSV and dispatches fit in code:

```python
from comp_model.inference import fit_study_csv_from_config

study_result = fit_study_csv_from_config(
    "data/study.csv",
    config=config,
    level="study",
)
```

This is still script-based and does not use the CLI.

## Common Issues

- Unknown config keys raise `ValueError` (schemas are strict).
- `parameter_names` must be unique and valid for the selected model.
- Sampled parameters in `parameter_names` must not also be passed in
  `model.kwargs`.
- Supported transform kinds are `identity`, `unit_interval_logit`,
  and `positive_log`.
