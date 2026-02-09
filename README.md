# comp_model

A Python toolkit for computational modeling of bandit-style decision tasks, simulation, inference, and parameter recovery. It separates core abstractions from concrete implementations so you can mix your own tasks/models with the provided tooling.

## Packages

- `comp_model_core`: interfaces, data containers, plan/spec schemas, parameter schemas/transforms, registries, and validation helpers.
- `comp_model_impl`: reference models, bandits, demonstrators, generators, estimators (MLE and Stan), and recovery utilities.

## Highlights

- Declarative study plans (JSON/YAML) with explicit per-trial interface schedules.
- Environment specs and trial specs are separate, so outcome visibility/noise and forced-choice can vary by trial.
- Lightweight data containers: `StudyData` -> `SubjectData` -> `Block` -> `Trial`.
- Parameter schemas with bounds and transforms (constrained <-> z-space).
- Built-in models: `QRL`, `VS`, `Vicarious_RL`, `Vicarious_VS`, plus within-subject shared+delta wrappers.
- Built-in tasks: `BernoulliBanditEnv` plus demonstrators for social observation.
- Generators for asocial and social simulations.
- Estimators: box-constrained MLE, transformed MLE (z-space), within-subject shared+delta MLE, Stan NUTS (subjectwise and hierarchical).
- Parameter recovery pipeline with metrics and a Streamlit GUI.
- Optional event logs for precise update timing; used by built-in generators and estimators.

## Installation

Python >= 3.14.

Editable installs (recommended for development):

```bash
python -m pip install -U pip
python -m pip install -e ./comp_model_core
python -m pip install -e ./comp_model_impl
```

`comp_model_impl` depends on `comp_model_core`, so installing `comp_model_impl` pulls in core automatically.

Optional extras:

```bash
# Stan estimators
python -m pip install cmdstanpy
python -m cmdstanpy.install_cmdstan

# Parameter recovery GUI
python -m pip install streamlit
```

## Quickstart: simulate and fit

```python
import numpy as np
from comp_model_core.plans.block import BlockPlan
from comp_model_core.data.types import StudyData
from comp_model_impl.register import make_registry
from comp_model_impl.tasks.build import build_runner_for_plan
from comp_model_impl.generators.event_log import EventLogAsocialGenerator
from comp_model_impl.models import QRL
from comp_model_impl.estimators import BoxMLESubjectwiseEstimator

rng = np.random.default_rng(0)

plan = BlockPlan(
    block_id="b1",
    n_trials=5,
    condition="c1",
    bandit_type="BernoulliBanditEnv",
    bandit_config={"probs": [0.2, 0.8]},
    trial_specs=[
        {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}
    ] * 5,
)

registry = make_registry()
builder = lambda p: build_runner_for_plan(plan=p, registries=registry)

gen = EventLogAsocialGenerator()
subject = gen.simulate_subject(
    subject_id="S001",
    block_runner_builder=builder,
    model=QRL(),
    params={"alpha": 0.2, "beta": 3.0},
    block_plans=[plan],
    rng=rng,
)

study = StudyData(subjects=[subject])
est = BoxMLESubjectwiseEstimator(model=QRL(), n_starts=5)
fit = est.fit(study=study, rng=rng)

print(fit.subject_hats["S001"])
```

## Study plans (YAML/JSON)

Plans are declarative. Trial interfaces must be explicit, but you can use a template
and optional overrides to keep files compact.

```yaml
subjects:
  S001:
    - block_id: "b1"
      condition: "A"
      n_trials: 3
      bandit_type: "BernoulliBanditEnv"
      bandit_config: {probs: [0.2, 0.8]}
      trial_spec_template:
        self_outcome: {kind: VERIDICAL}
        available_actions: [0, 1]
```

For social blocks, include `demonstrator_type` / `demonstrator_config` and provide
`demo_outcome` in each trial spec (or template).

## Parameter recovery

```python
from comp_model_impl.recovery.parameter import load_parameter_recovery_config, run_parameter_recovery
from comp_model_impl.generators.event_log import EventLogAsocialGenerator
from comp_model_impl.models import QRL
from comp_model_impl.estimators import BoxMLESubjectwiseEstimator

cfg = load_parameter_recovery_config("recovery_config.yaml")
outputs = run_parameter_recovery(
    config=cfg,
    generator=EventLogAsocialGenerator(),
    model=QRL(),
    estimator=BoxMLESubjectwiseEstimator(model=QRL(), n_starts=5),
)
```

## Real-data fitting: uncertainty + model comparison

For real-data fitting, you can request posterior uncertainty summaries from Stan
estimators and then compute model-comparison metrics from fit tables.

```python
import numpy as np
import pandas as pd

from comp_model_impl.models import Vicarious_AP_DB_STAY
from comp_model_impl.estimators import StanHierarchicalNUTSEstimator
from comp_model_impl.analysis.model_selection import add_information_criteria

# fit one model to your event-log StudyData
est = StanHierarchicalNUTSEstimator(
    model=Vicarious_AP_DB_STAY(),
    hyper_priors=hyper_priors,                    # your prior mapping
    subject_point_estimate="conditional_map",     # "mean" or "conditional_map"
    return_posterior_summary=True,                # adds uncertainty summaries
)
fit = est.fit(study=study, rng=np.random.default_rng(0))

# uncertainty summaries are in diagnostics
population_summary = fit.diagnostics.get("population_posterior_summary", {})
subject_summary = fit.diagnostics.get("subject_posterior_summary", {})

# compare multiple fitted models (example fit table schema)
fit_table = pd.DataFrame(
    [
        {"candidate_model": "m1", "ll_total": -120.1, "k_total": 10, "n_obs_total": 500},
        {"candidate_model": "m2", "ll_total": -126.4, "k_total": 9, "n_obs_total": 500},
    ]
)
fit_table = add_information_criteria(fit_table, group_cols=())
print(fit_table[["candidate_model", "aic", "bic", "bf_best_vs_model_bic"]])
```

Notes:
- `return_posterior_summary=True` is intended for real-data inference; keep it
  `False` for large recovery runs unless you need uncertainty outputs.
- `bf_best_vs_model_bic` and `bf_model_vs_best_bic` are BIC-based Bayes-factor
  approximations, not bridge-sampling Bayes factors.

## Parameter recovery GUI

```bash
streamlit run apps/parameter_recovery_gui/app.py
```

## Repository layout

```
comp_model_core/     # core interfaces + data types
comp_model_impl/     # implementations (models/generators/estimators/stan)
apps/                # Streamlit parameter recovery GUI
comp_model_core/docs/
comp_model_impl/docs/
```
