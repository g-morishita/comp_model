# Tutorial: First End-to-End Fit

Before fitting a computational model to real participant data, first check that
your model workflow behaves as expected and is free of obvious implementation
bugs. A standard first check is simulation: generate synthetic behavior from a
known parameter setting, then fit the same model back to that synthetic trace
(Wilson and Collins, 2019).
In this tutorial, you will learn how to perform a standard
reinforcement-learning workflow through a concrete example: simulate behavior
with `AsocialStateQValueSoftmaxModel` (a Q-learning model with a softmax choice
rule) on a `StationaryBanditProblem` (a two-armed task with fixed reward
probabilities), then fit the model back to synthetic data as a first
validation step before parameter recovery.

## Prerequisites

- Python 3.11+
- Working installation of `comp_model`

If you have not installed and verified your environment yet, complete
[Install and Verify](install-and-verify.md) first.

## Terms Used in This Tutorial

- `episode`: one full simulation run across consecutive trials.
- `event trace`: the recorded trial-by-trial log of what happened
  (using `comp_model` event labels).
- `comp_model` event labels (library-specific, not universal terms):
  `observation` (task information shown) -> `decision` (choice made) ->
  `outcome` (feedback/reward returned) -> `update` (model state update).
- `replay`: score how well a model explains observed choices by running the
  model on the same trial history and computing choice probabilities.
- `inference` (estimation): fit parameters or compare models using replay-based
  scores (for example, log-likelihood).

## Step 1: Instantiate a Problem

First decide what task you want to model. Here, we want a simple stationary two-option
bandit task where option 0 gives reward with probability 0.2 and option 1 gives
reward with probability 0.8. In `comp_model`, task environments are
implemented as Problem classes (`DecisionProblem` interface), and this specific
task is implemented by `StationaryBanditProblem`.

To express this in code, use `StationaryBanditProblem` and pass
`reward_probabilities=[0.2, 0.8]`. The list index is the option ID, and the
value is that option's Bernoulli reward probability.

```python
from comp_model.problems import StationaryBanditProblem

problem = StationaryBanditProblem(reward_probabilities=[0.2, 0.8])
```

### Quick Quiz

Hover on each question to see the answer.

- <span title="Use reward_probabilities=[0.5, 0.8].">If you want to change option 0 reward probability from 0.2 to 0.5, what should you change in code?</span>
- <span title="Add one more entry: reward_probabilities=[0.2, 0.8, 0.5].">If you add a third option with reward probability 0.5, how should reward_probabilities look?</span>

### Optional: Interact with the Problem Directly

This subsection is optional and covers low-level implementation details. You can skip to
[Step 2: Instantiate a Model](#step-2-instantiate-a-model).

Before adding a model, you can directly inspect how the problem behaves. This
helps you confirm what observation and outcome objects look like.

This low-level interaction requires a `DecisionContext`. The context carries
trial metadata (for example, trial index and available actions) so problem and
model methods share the same explicit trial information. If you use
`run_episode`, you usually do not need to build this manually.

```python
import numpy as np
from comp_model.core.contracts import DecisionContext

rng = np.random.default_rng(10)
problem.reset(rng=rng)

trial_index = 0
available_actions = tuple(problem.available_actions(trial_index=trial_index))
context = DecisionContext(trial_index=trial_index, available_actions=available_actions)

observation = problem.observe(context=context)
outcome = problem.transition(action=1, context=context, rng=rng)  # force option 1

print("available_actions:", available_actions)
print("observation:", observation)
print("outcome:", outcome)
```

## Step 2: Instantiate a Model

Next decide what kind of computational model you want to simulate. Here, we want a
standard asocial RL learner that updates action values from rewards and chooses
probabilistically based on those values. In `comp_model`, this is
`AsocialStateQValueSoftmaxModel`.

To express this in code, set model parameters explicitly:

- `alpha`: learning rate (how fast values update after outcomes),
- `beta`: inverse temperature (how deterministic choices are),
- `initial_value`: starting value for each option before learning.

```python
from comp_model.models import AsocialStateQValueSoftmaxModel

generating_model = AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0)
```

### Quick Quiz

Hover on each question to see the answer.

- <span title="AsocialStateQValueSoftmaxModel(alpha=-0.2, beta=3.0, initial_value=0.0) raises ValueError: alpha must be in [0, 1].">What exact error behavior should you expect if alpha is negative?</span>

### Guardrails

The model validates parameters at construction time:

- `alpha` must be in `[0, 1]`,
- `beta` must be `>= 0`.

So invalid settings fail immediately instead of producing silent bad behavior.

```python
from comp_model.models import AsocialStateQValueSoftmaxModel

try:
    AsocialStateQValueSoftmaxModel(alpha=-0.2, beta=3.0, initial_value=0.0)
except ValueError as exc:
    print(exc)  # alpha must be in [0, 1]
```

### Optional: Run One Trial Loop Manually

This subsection is optional and covers low-level implementation details. You can
skip to [Step 3: Run a Pilot Episode](#step-3-run-a-pilot-episode).

```python
import numpy as np
from comp_model.core.contracts import DecisionContext
from comp_model.runtime.probabilities import normalize_distribution, sample_action

rng = np.random.default_rng(10)
problem.reset(rng=rng)
generating_model.start_episode()

trial_index = 0
available_actions = tuple(problem.available_actions(trial_index=trial_index))
context = DecisionContext(trial_index=trial_index, available_actions=available_actions)

observation = problem.observe(context=context)
raw_distribution = generating_model.action_distribution(observation, context=context)
distribution = normalize_distribution(raw_distribution, available_actions)
action = sample_action(distribution, rng)
outcome = problem.transition(action, context=context, rng=rng)
generating_model.update(observation, action, outcome, context=context)

print("observation:", observation)
print("distribution:", distribution)
print("action:", action)
print("outcome:", outcome)
print("updated_q:", generating_model.q_values_snapshot())
```

This single loop is the core interaction contract:
`observe -> action_distribution -> transition -> update`.

## Step 3: Run a Pilot Episode

Now decide how much data to generate for a quick sanity check. In `comp_model`,
an episode means one full simulation run across consecutive trials: the model
is initialized at the start of the run, then repeatedly interacts with the
problem trial by trial.

For a pilot check, keep it small (for example, 3 trials) so you can quickly
inspect the event trace. In code, use `run_episode(problem, model, config)`,
where:

- `problem` is the task environment,
- `model` is the computational model you want to simulate,
- `SimulationConfig(n_trials=3, seed=10)` sets trial count and RNG seed for
  reproducibility.

```python
from comp_model.runtime import SimulationConfig, run_episode

pilot_trace = run_episode(
    problem=problem,
    model=AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0),
    config=SimulationConfig(n_trials=3, seed=10),
)

print("pilot event count:", len(pilot_trace.events))
for event in pilot_trace.by_trial(0):
    print(event.phase.value, event.payload)
```

In each printed line:

- `event.phase.value` is the event label (`observation`, `decision`,
  `outcome`, or `update`), which tells you which part of the trial this row
  represents.
- `event.payload` is a dictionary with the details for that event.
  For example, in a `decision` event it includes the chosen action and action
  probabilities; in an `outcome` event it includes the feedback/reward info.

You should see four ordered `comp_model` event labels for each trial:
`observation` (task info) -> `decision` (choice) -> `outcome` (feedback) ->
`update` (model learning step).

`run_episode` automates the trial loop and records an event trace. That trace is
then used for replay (scoring model-predicted choice probabilities on observed
choices) and inference/estimation (fitting parameters or comparing models).

### Quick Quiz

Hover on each question to see the answer.

- <span title="SimulationConfig(n_trials=10, seed=10)">If you want the pilot run to simulate 10 trials (keeping the same seed), what should `SimulationConfig` be?</span>
- <span title="Change only the seed, for example SimulationConfig(n_trials=3, seed=42).">If you want a different random sequence but keep trial count fixed, what should you change?</span>
- <span title="Use pilot_trace.by_trial(0).">If you want only the events from trial 0, what method call should you use?</span>
- <span title="The trace stores `comp_model` event labels in order: observation -> decision -> outcome -> update.">What event-label order should you expect inside one trial?</span>

## Step 4: Simulate a Full Dataset for Fitting

The pilot run is only for quick inspection. For fitting, you usually need more
trials so the parameter signal is stronger and estimation is more stable.

Here we generate a full synthetic dataset with 120 trials. In this call:

- `n_trials=120` controls dataset size for fitting,
- `seed=11` makes the simulation reproducible,
- a new model instance starts from a clean state before trial 0.

```python
trace = run_episode(
    problem=problem,
    model=AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0),
    config=SimulationConfig(n_trials=120, seed=11),
)
```

Use a fresh model instance for the full dataset so learned values from the
pilot run do not leak into the fitting dataset.

### Quick Quiz

Hover on each question to see the answer.

- <span title="trace.by_trial(0)">What call returns only trial 0 events from the full dataset trace?</span>
- <span title="len(trace.by_trial(10))">What expression gives the number of logged events in trial 10?</span>
- <span title="for event in trace.by_trial(5): print(event.phase.value, event.payload)">How can you print phase labels and payloads for trial 5?</span>
- <span title="sorted({event.trial_index for event in trace.events})">How can you inspect which trial indices are present in the trace?</span>

## Step 5: Fit the Model with SciPy MLE

Now estimate model parameters from the synthetic dataset. Here we use
maximum-likelihood estimation (MLE): find parameter values that make the
observed choices in `trace` as probable as possible under the model.

In `comp_model`, `fit_model(...)` runs this estimation. In this call:

- `trace` is the dataset to fit,
- `model_factory` creates a fresh model instance from each candidate parameter
  set,
- `initial_value` is fixed to `0.0` (common practice in this tutorial setup),
  so only `alpha` and `beta` are estimated,
- `estimator_type="scipy_minimize"` uses SciPy's numerical optimizer,
- `initial_params` are the optimizer's starting values,
- `bounds` constrain the search range,
- `method="L-BFGS-B"` chooses the optimizer algorithm.

Fixing some parameters is often useful when they are not primary targets of
inference. It simplifies optimization and makes interpretation cleaner.

```python
from comp_model.inference import FitSpec, fit_model
from comp_model.models import AsocialStateQValueSoftmaxModel

fit_result = fit_model(
    trace,
    model_factory=lambda params: AsocialStateQValueSoftmaxModel(
        alpha=params["alpha"],
        beta=params["beta"],
        initial_value=0.0,  # fixed (not estimated)
    ),
    fit_spec=FitSpec(
        estimator_type="scipy_minimize",
        initial_params={
            "alpha": 0.3,
            "beta": 2.0,
        },
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.01, 10.0),
        },
        method="L-BFGS-B",
    ),
)
```

### Quick Quiz

Hover on each question to see the answer.

- <span title="Set initial_params['beta'] to the new starting value, for example 5.0.">If you want to start optimization from a larger beta value, what should you change in code?</span>
- <span title="Use bounds['beta'] = (0.01, 20.0).">If you want to allow larger fitted beta values, what bound should you edit?</span>
- <span title="Set method='Powell' (or another SciPy method you want to try).">If you want to try a different SciPy optimizer, which field should you change?</span>
- <span title="fit_model raises ValueError because scipy_minimize requires initial_params.">What happens if you remove initial_params while keeping estimator_type='scipy_minimize'?</span>
- <span title="Add it to the model_factory parameter mapping and also add initial_params/bounds entries for initial_value.">If you later decide to estimate initial_value too, what two places must you update?</span>

## Step 6: Check That the Fit Worked

After fitting, do a quick quality check before trusting the estimates. The goal
is to confirm that optimization finished properly and parameter values look
plausible.

- `best log-likelihood`: model score on observed choices (higher is better when
  comparing fits on the same dataset),
- `best params`: fitted parameter values,
- `optimizer success/message`: whether SciPy reports successful convergence.

```python
print("best log-likelihood:", fit_result.best.log_likelihood)
print("best params:", fit_result.best.params)
if fit_result.scipy_diagnostics is not None:
    print("optimizer success:", fit_result.scipy_diagnostics.success)
    print("optimizer message:", fit_result.scipy_diagnostics.message)
```

You can also add explicit sanity checks:

```python
import math

ll = fit_result.best.log_likelihood
params = fit_result.best.params

if not math.isfinite(ll):
    raise RuntimeError("Fit failed: log-likelihood is not finite.")
if not (0.0 <= params["alpha"] <= 1.0):
    raise RuntimeError(f"Fit failed: alpha out of range: {params['alpha']}")
if not (params["beta"] > 0.0):
    raise RuntimeError(f"Fit failed: beta must be > 0: {params['beta']}")
```

Minimum checks before moving on:

1. best log-likelihood is finite,
2. optimizer reports successful termination in most runs,
3. recovered parameters are in plausible ranges (`alpha` in `[0, 1]`,
   `beta > 0`),
4. recovered values are at least directionally close to generating values when
   trial counts are moderate.

### Quick Quiz

Hover on each question to see the answer.

- <span title="fit_result.best.log_likelihood">Which field gives the fitted log-likelihood score?</span>
- <span title="fit_result.best.params">Which field stores fitted parameter values?</span>
- <span title="fit_result.scipy_diagnostics.success">Which field indicates whether the SciPy optimizer reported convergence?</span>
- <span title="Use a finite check on log-likelihood, range checks on alpha/beta, and inspect optimizer success/message.">What is the minimum set of checks to run before trusting this fit?</span>

## Common Mistakes

1. Reusing a model that has already learned from a previous run.
2. Using too few trials (recovery becomes unstable).
3. Setting bounds too tight for plausible parameter values.

## Next Steps

- Continue with [Parameter Recovery](parameter-recovery.md).
- Then run [Model Recovery](model-recovery.md).
- Finish with [Fit Your Own Dataset](fit-your-own-dataset.md).

## References

- Wilson RC, Collins AGE. (2019). Ten simple rules for the computational
  modeling of behavioral data. *eLife*, 8:e49547.
  <https://doi.org/10.7554/eLife.49547>
