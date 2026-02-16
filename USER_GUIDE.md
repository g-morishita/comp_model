# User Guide: QRL Parameter Recovery (Minimal)

This is a lightweight, step-by-step example for users who want to run parameter
recovery with the built-in QRL model. You will:

1) Define a Bernoulli bandit task with reward probabilities.
2) Define block information and the per-trial interface schedule.
3) Create a parameter-recovery config.
4) Run the recovery and inspect outputs.

## Concrete experiment -> library mapping

Example experiment (asocial):
- 100 trials
- 2-armed Bernoulli bandit with reward probabilities [0.2, 0.8]
- Both actions are always available
- Veridical feedback (no noise)
- Single block, single condition

Below is the exact configuration to reproduce that setting in this library.

## 1) Create a study plan (YAML)

Save as `qrl_plan.yaml` (any path is fine). This uses one subject and
one block to keep the example minimal.

```yaml
subjects:
  S001:
    - block_id: "b1"
      condition: "A"
      n_trials: 100
      bandit_type: "BernoulliBanditEnv"
      bandit_config:
        probs: [0.2, 0.8]
      trial_spec_template:
        self_outcome: {kind: VERIDICAL}
        available_actions: [0, 1]
```

Notes:
- `condition` is required for every block.
- Trial interfaces are explicit. The `trial_spec_template` expands to all trials.

**Plan fields explained**
- `subjects`: mapping from subject IDs to each subject’s block list.
- `S001`: the subject ID (string); any unique label is fine.
- `block_id`: identifier for this block (used in logs and outputs).
- `condition`: block-level condition label; required for all blocks.
- `n_trials`: number of trials in the block; must match the trial schedule length.
- `bandit_type`: registered environment name; here, a Bernoulli bandit.
- `bandit_config`: configuration passed to the bandit constructor.
- `bandit_config.probs`: reward probabilities per action (arm).
- `trial_spec_template`: a single trial interface spec expanded to all trials.
- `trial_spec_template.self_outcome.kind`: how feedback is observed; `VERIDICAL` means true outcome.
- `trial_spec_template.available_actions`: allowed actions on each trial.

## 2) Create a recovery config (YAML)

Save as `qrl_recovery.yaml` (any path is fine). The `plan_path` should
point to your plan file. Here we sample `alpha` and `beta` independently.

```yaml
plan_path: qrl_plan.yaml
n_reps: 20
seed: 0

sampling:
  mode: independent
  space: param
  individual:
    alpha:
      name: beta
      args: {a: 2.0, b: 2.0}
    beta:
      name: lognorm
      args: {s: 0.4, loc: 0.0, scale: 4.0}

output:
  out_dir: recovery_out
  save_format: csv
```

Notes:
- `alpha` is in (0, 1); `beta` is positive and capped by the QRL schema.
- Outputs go to a timestamped folder inside `out_dir`.

**Config fields explained**
- `plan_path`: path to the study plan file used for simulation.
- `n_reps`: number of independent recovery replications.
- `seed`: RNG seed for reproducible sampling and simulation.
- `sampling`: controls how true parameters are sampled.
- `sampling.mode`: `independent` samples each subject separately.
- `sampling.space`: `param` draws in constrained parameter space.
- `sampling.individual`: per-parameter distribution specs.
- `sampling.individual.alpha`: distribution for the QRL learning rate.
- `sampling.individual.alpha.name`: `beta` distribution from `scipy.stats`.
- `sampling.individual.alpha.args`: `a` and `b` shape parameters.
- `sampling.individual.beta`: distribution for the QRL inverse temperature.
- `sampling.individual.beta.name`: `lognorm` distribution from `scipy.stats`.
- `sampling.individual.beta.args`: `s` (shape), `loc`, and `scale`.
- `output`: controls where and how results are saved.
- `output.out_dir`: base output directory.
- `output.save_format`: table format (`csv` or `parquet`).

## 3) Run parameter recovery (Python)

```python
from comp_model_impl.recovery.parameter import load_parameter_recovery_config, run_parameter_recovery

cfg = load_parameter_recovery_config("qrl_recovery.yaml")

outputs = run_parameter_recovery(config=cfg)

print(outputs.out_dir)
print(outputs.records.head())
print(outputs.metrics.head())
```

**Code elements explained**
- `load_parameter_recovery_config`: parses the YAML into a config object.
- `run_parameter_recovery`: resolves configured components and orchestrates sampling, simulation, fitting, and output writing.
- `outputs.out_dir`: path to the run folder.
- `outputs.records`: table of true vs. estimated parameters.
- `outputs.metrics`: recovery summary metrics.

## 4) Inspect outputs

Inside the run directory you will find:

- `parameter_recovery_records.csv` (true vs. estimated parameters)
- `parameter_recovery_metrics.csv` (correlation, RMSE, bias, etc.)
- `run_manifest.json` (provenance info)

## Model validity: what to check next

Parameter recovery is the first validity check: can the model recover known
parameters under the same task and trial interface you use in the experiment?

Suggested next steps:

1) Check recovery metrics
   - High correlation and low RMSE for `alpha` and `beta` is a good sign.
2) Scale up realism
   - Increase `n_trials` or add multiple blocks/conditions if your experiment does.
3) Fit real data with the same estimator
   - Once recovery looks reasonable, fit your observed study data using the
     same model and estimator so the likelihood definition is consistent.

If recovery is poor, typical fixes are:
- add more trials (power issue),
- tighten priors or sampling ranges,
- ensure your trial interface matches the experiment (feedback visibility,
  forced-choice actions, etc.).

## Fit real data: uncertainty and model comparison

For real datasets, parameter recovery outputs are usually not enough. You often
want:
- posterior uncertainty for population/subject parameters, and
- model-comparison metrics (AIC/BIC and Bayes-factor approximations).

### 1) Fit with uncertainty summaries

```python
import numpy as np
from comp_model_impl.estimators import StanHierarchicalNUTSEstimator
from comp_model_impl.models import Vicarious_AP_DB_STAY

est = StanHierarchicalNUTSEstimator(
    model=Vicarious_AP_DB_STAY(),
    hyper_priors=hyper_priors,
    subject_point_estimate="conditional_map",  # or "mean"
    return_posterior_summary=True,
)
fit = est.fit(study=real_study, rng=np.random.default_rng(0))

pop = fit.diagnostics["population_posterior_summary"]      # param -> {mean, sd, q025, q50, q975}
subj = fit.diagnostics["subject_posterior_summary"]        # subject -> param -> {mean, sd, ...}
```

Interpretation:
- `subject_point_estimate="mean"`: posterior mean for each subject parameter.
- `subject_point_estimate="conditional_map"`: empirical-Bayes conditional MAP
  (less shrinkage than posterior mean, still regularized by group hyperparameters).

### 2) Compare candidate models on real data

`comp_model_impl.analysis.model_selection.add_information_criteria` adds model
comparison columns to your fit table.

```python
import pandas as pd
from comp_model_impl.analysis.model_selection import add_information_criteria

fit_table = pd.DataFrame(
    [
        {"candidate_model": "m1", "ll_total": -120.1, "k_total": 10, "n_obs_total": 500},
        {"candidate_model": "m2", "ll_total": -126.4, "k_total": 9, "n_obs_total": 500},
    ]
)

# group_cols=() means "single real dataset comparison"
fit_table = add_information_criteria(fit_table, group_cols=())
print(fit_table[["candidate_model", "aic", "bic", "bf_best_vs_model_bic"]])
```

Added columns:
- `aic`, `bic`: information criteria.
- `delta_aic`, `delta_bic`: difference from the best model in each group.
- `akaike_weight`, `bic_weight`: relative weights within each group.
- `bf_best_vs_model_bic`, `bf_model_vs_best_bic`: BIC-based Bayes-factor
  approximations.
- If `waic` (or `elpd_waic`) is present in the input table:
  `waic`, `delta_waic`, `waic_weight` are also added.

Notes:
- These BF values are BIC approximations, not exact marginal-likelihood Bayes
  factors (e.g., bridge sampling).
- WAIC requires pointwise log-likelihood draws (or precomputed WAIC/ELPD-WAIC);
  it cannot be recovered from scalar `ll_total` alone.
- For parameter recovery batches, keep `return_posterior_summary=False` unless
  you explicitly need uncertainty summaries.

### 3) MLE uncertainty (approximate)

MLE estimators also support optional uncertainty diagnostics:

```python
import numpy as np
from comp_model_impl.estimators import BoxMLESubjectwiseEstimator
from comp_model_impl.models import QRL

est = BoxMLESubjectwiseEstimator(
    model=QRL(),
    return_uncertainty=True,
    uncertainty_ci=0.95,
)
fit = est.fit(study=real_study, rng=np.random.default_rng(0))
print(fit.diagnostics["subj_S001"]["uncertainty"])
```

Interpretation:
- These intervals are asymptotic local-curvature approximations, not Bayesian
  credible intervals.
- For z-space MLE, uncertainty is transformed to parameter space via a
  delta-method Jacobian.

## Common checks

- If you change `n_trials`, your `trial_specs` (or template) must still cover all trials.
- If you introduce social observation, you must add `demo_outcome` to trial specs.
- If your model or estimator changes, update the sampled parameter names.

## Implementing a computational model (custom)

This section is for users who want to add their own model. A model must implement
the `ComputationalModel` interface from `comp_model_core.interfaces.model`.

### Minimal example (asocial Q-learning style)

```python
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.params import ParameterSchema, ParamDef, Bound, Sigmoid, BoundedTanh
from comp_model_core.requirements import RequireAsocialBlock, RequireAnySelfOutcomeObservable, Requirement
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax


def _my_schema(alpha_default: float, beta_default: float, beta_max: float) -> ParameterSchema:
    return ParameterSchema(
        params=(
            ParamDef("alpha", alpha_default, Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("beta", beta_default, Bound(1e-6, beta_max), transform=BoundedTanh(1e-6, beta_max)),
        )
    )


@dataclass(slots=True)
class MyQModel(ComputationalModel):
    alpha: float = 0.2
    beta: float = 5.0
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        self._q: list[np.ndarray] = []

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        return (RequireAsocialBlock(), RequireAnySelfOutcomeObservable())

    @property
    def param_schema(self) -> ParameterSchema:
        return _my_schema(self.alpha, self.beta, self.beta_max)

    def supports(self, spec: EnvironmentSpec) -> bool:
        return (not spec.is_social) and int(spec.n_actions) >= 2

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        self._q = []

    def _ensure_state(self, s: int, n_actions: int) -> None:
        while len(self._q) <= s:
            self._q.append(np.zeros(n_actions, dtype=float))
        if self._q[s].shape[0] != n_actions:
            self._q[s] = np.zeros(n_actions, dtype=float)

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)
        return _softmax(self._q[s], self.beta)

    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if outcome is None:
            return
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)
        a = int(action)
        self._q[s][a] += float(self.alpha) * (float(outcome) - self._q[s][a])
```

**Every element explained**
- `MyQModel(ComputationalModel)`: subclassing the interface is required.
- `alpha`, `beta`: model parameters; names must match `param_schema` names.
- `beta_max`: upper bound used by the schema (not estimated directly).
- `__post_init__`: initialize latent state containers.
- `requirements`: optional plan constraints (asocial + observable outcomes).
- `param_schema`: declares parameter names, defaults, bounds, and transforms.
- `supports`: runtime check against an `EnvironmentSpec` (e.g., social vs asocial).
- `reset_block`: clears latent state at the start of each block.
- `_ensure_state`: internal helper to size latent state to the task.
- `action_probs`: returns a probability vector over actions.
- `update`: updates latent state from the **observed** outcome (can be `None`).
- `info`, `rng`: optional hooks if you need extra metadata or stochastic updates.

### Registering your model (optional but recommended)

If you want to use your model by name in YAML/JSON plans, add it to the registry:

```python
from comp_model_impl.register import make_registry
from my_models import MyQModel

reg = make_registry()
reg.models.register("MyQModel", MyQModel)
```

**Registry elements explained**
- `make_registry()`: returns the default registry with built-in models/bandits.
- `reg.models.register`: associates a string name with your class.
- `"MyQModel"`: the name you will use in plans (`model: MyQModel`).

### Using your model in recovery

```python
from comp_model_impl.recovery.parameter import run_parameter_recovery

outputs = run_parameter_recovery(config=cfg)
```

**Usage elements explained**
- `model=MyQModel()`: the simulator and fitter must use the same model class.
- `estimator=...`: choose an estimator that matches your parameter schema.
- `EventLogAsocialGenerator`: generates event logs for consistent likelihood replay.
