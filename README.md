# comp_model

`comp_model` is a computational decision-modeling library designed for
research workflows in behavioral, cognitive, and social decision tasks.

It supports the full empirical loop:

- simulate task behavior,
- fit candidate models to observed choices,
- compare models with information criteria,
- evaluate identifiability via recovery analyses.

The framework is intentionally generic: models interact with a decision
environment through observation, decision, outcome, and update steps. Bandit
tasks are included as concrete task implementations, not as the core abstraction.

## Who This Is For

- Researchers fitting reinforcement-learning and social-learning models.
- Teams running model-comparison pipelines on trial-level datasets.
- Projects that need reproducible simulation and recovery analyses.

## What You Can Do

- Fit models with MLE, MAP, and MCMC workflows.
- Compare candidate models with `log_likelihood`, `aic`, `bic`, `waic`,
  and `psis_loo` (when posterior pointwise draws are available).
- Run parameter recovery and model recovery from config files.
- Work from canonical in-memory traces or tabular CSV datasets.

## Typical Research Workflow

1. Prepare trial- or study-level decision data (CSV or in-memory structures).
2. Define candidate models and estimator settings.
3. Fit parameters per dataset, subject, or study.
4. Compare models under a chosen criterion.
5. Run recovery workflows to check robustness and identifiability.

## Fast Start: Existing CSV Data

Fit one dataset from config:

```bash
comp-model-fit \
  --config fit_config.json \
  --input-csv study.csv \
  --input-kind study \
  --level study \
  --output-dir fit_out \
  --prefix run1
```

Compare candidate models from config:

```bash
comp-model-compare \
  --config compare_config.json \
  --input-csv study.csv \
  --input-kind study \
  --level study \
  --output-dir compare_out \
  --prefix run1
```

Run recovery workflow:

```bash
comp-model-recovery \
  --config recovery_config.json \
  --mode auto \
  --output-dir recovery_out \
  --prefix run1
```

## Fast Start: Simulate Then Fit in Python

```python
from comp_model.problems import StationaryBanditProblem
from comp_model.models import AsocialStateQValueSoftmaxModel
from comp_model.runtime import SimulationConfig, run_episode
from comp_model.inference import FitSpec, fit_model

problem = StationaryBanditProblem(reward_probabilities=[0.2, 0.8])
model = AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0)
trace = run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=100, seed=7))

fit_result = fit_model(
    trace,
    model_factory=lambda params: AsocialStateQValueSoftmaxModel(**params),
    fit_spec=FitSpec(
        estimator_type="grid_search",
        parameter_grid={
            "alpha": [0.1, 0.2, 0.3],
            "beta": [2.0, 3.0, 4.0],
            "initial_value": [0.0],
        },
    ),
)
print(fit_result.best.params)
```

## Documentation

Documentation follows the Divio system:

- Published documentation: `https://g-morishita.github.io/comp_model/`
- Tutorials: `https://g-morishita.github.io/comp_model/tutorials/first-end-to-end-fit/`
- How-to guides: `https://g-morishita.github.io/comp_model/how-to/fit-existing-csv-data/`
- Reference: `https://g-morishita.github.io/comp_model/reference/api-overview/`
- Explanation: `https://g-morishita.github.io/comp_model/explanation/design-philosophy/`

For local docs preview:

```bash
mkdocs serve
```
