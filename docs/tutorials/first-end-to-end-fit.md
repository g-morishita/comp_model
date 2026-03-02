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

Click `Show answer` to reveal the answer, then click `Hide answer` to collapse it.

- <span class="cm-quiz" data-answer="Use reward_probabilities=[0.5, 0.8].">If you want to change option 0 reward probability from 0.2 to 0.5, what should you change in code?</span>
- <span class="cm-quiz" data-answer="Add one more entry: reward_probabilities=[0.2, 0.8, 0.5].">If you add a third option with reward probability 0.5, how should reward_probabilities look?</span>

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

Click `Show answer` to reveal the answer, then click `Hide answer` to collapse it.

- <span class="cm-quiz" data-answer="AsocialStateQValueSoftmaxModel(alpha=-0.2, beta=3.0, initial_value=0.0) raises ValueError: alpha must be in [0, 1].">What exact error behavior should you expect if alpha is negative?</span>

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

Click `Show answer` to reveal the answer, then click `Hide answer` to collapse it.

- <span class="cm-quiz" data-answer="SimulationConfig(n_trials=10, seed=10)">If you want the pilot run to simulate 10 trials (keeping the same seed), what should `SimulationConfig` be?</span>
- <span class="cm-quiz" data-answer="Change only the seed, for example SimulationConfig(n_trials=3, seed=42).">If you want a different random sequence but keep trial count fixed, what should you change?</span>
- <span class="cm-quiz" data-answer="Use pilot_trace.by_trial(0).">If you want only the events from trial 0, what method call should you use?</span>
- <span class="cm-quiz" data-answer="The trace stores comp_model event labels in this order: observation, decision, outcome, update.">What event-label order should you expect inside one trial?</span>

## Step 4: Fit the Tiny Pilot Dataset

Now fit the model to the pilot dataset (`pilot_trace`) and inspect the
estimated parameters.

To do this, we use `fit_model(...)`. It takes three core inputs:

- the simulated data (`pilot_trace`) from Step 3,
- a model definition for fitting (`model_factory`),
- and a fitting specification (`fit_spec`).

Here is the fitting code:

```python
from comp_model.inference import FitSpec, fit_model
from comp_model.models import AsocialStateQValueSoftmaxModel

pilot_fit_result = fit_model(
    pilot_trace,
    model_factory=lambda params: AsocialStateQValueSoftmaxModel(
        alpha=params["alpha"],
        beta=params["beta"],
        initial_value=0.0,  # fixed (not estimated)
    ),
    fit_spec=FitSpec(
        inference="mle",
        initial_params={
            "alpha": 0.3,
            "beta": 2.0,
        },
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.01, 10.0),
        },
        method="L-BFGS-B",
        n_starts=5,
        random_seed=10,
    ),
)

print("true params:", {"alpha": 0.2, "beta": 3.0})
print("pilot best params:", pilot_fit_result.best.params)
```

### How `model_factory` Works

`model_factory` tells `fit_model` what model family you are fitting and which
parameters are free versus fixed.

In this example:

- `alpha` and `beta` are free parameters because they come from `params`.
- `initial_value=0.0` is fixed, so it is not estimated.

Conceptually, `fit_model` repeatedly proposes candidate parameter values,
constructs a fresh model using `model_factory`, evaluates how well that model
explains the observed choices in `pilot_trace`, and then searches for the
best-scoring parameter values.

### What `fit_spec` Specifies Here

`fit_spec` defines the estimation setup:

- `inference="mle"`:
  use maximum-likelihood estimation.
- `initial_params={"alpha": 0.3, "beta": 2.0}`:
  one anchor start for optimization.
- `n_starts=5`:
  run multiple starts and keep the best-likelihood solution.
- `random_seed=10`:
  makes randomized starts reproducible.
- `bounds`:
  allowed range for each free parameter during search.
  Here, `alpha` is constrained to `[0, 1]` and `beta` to `[0.01, 10.0]`.
- `method="L-BFGS-B"`:
  numerical optimization algorithm.

After running this, compare `pilot best params` with the generating values
`{"alpha": 0.2, "beta": 3.0}`.
With only a few trials, estimates are often far from true parameters. This is
expected, so the next step is to increase `n_trials` and fit again.

### Quick Quiz

Click `Show answer` to reveal the answer, then click `Hide answer` to collapse it.

- <span class="cm-quiz" data-answer='Set `solver=\"grid_search\"` and provide `parameter_grid`, for example: FitSpec(inference=\"mle\", solver=\"grid_search\", parameter_grid={\"alpha\": [0.1, 0.2, 0.3], \"beta\": [1.0, 2.0, 3.0]})'>If you want to switch this fit from SciPy optimization to grid search, what should you change in `FitSpec`?</span>
- <span class="cm-quiz" data-answer='Change `initial_value=0.0` to `initial_value=params[\"initial_value\"]` in `model_factory`, then include `initial_value` in your search-space settings (for example, add a bound such as `\"initial_value\": (-2.0, 2.0)` or a grid entry).'>If you want to estimate `initial_value` instead of fixing it, what two places must be edited?</span>
- <span class="cm-quiz" data-answer='Set a constant in `model_factory` (for example `beta=3.0` instead of `beta=params[\"beta\"]`), then remove `beta` from your search-space settings (bounds/grid).'>If you want to fix `beta` at `3.0` and estimate only `alpha`, what should you change?</span>
- <span class="cm-quiz" data-answer='Keep `n_starts` the same and change `random_seed`, for example from `10` to `42`.'>If you want different randomized starts but still reproducible results, what is the minimal edit?</span>

## Step 5: Increase Trials and Simulate a Fitting Dataset

Now increase the trial count so estimation has enough signal. Here we simulate
120 trials with a fresh model instance.

In this call:

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

Click `Show answer` to reveal the answer, then click `Hide answer` to collapse it.

- <span class="cm-quiz" data-answer="trace.by_trial(0)">What call returns only trial 0 events from the full dataset trace?</span>
- <span class="cm-quiz" data-answer="len(trace.by_trial(10))">What expression gives the number of logged events in trial 10?</span>
- <span class="cm-quiz" data-answer="for event in trace.by_trial(5): print(event.phase.value, event.payload)">How can you print phase labels and payloads for trial 5?</span>

Now estimate model parameters from the larger synthetic dataset in `trace`.

```python
fit_result = fit_model(
    trace,
    model_factory=lambda params: AsocialStateQValueSoftmaxModel(
        alpha=params["alpha"],
        beta=params["beta"],
        initial_value=0.0,  # fixed (not estimated)
    ),
    fit_spec=FitSpec(
        inference="mle",
        initial_params={
            "alpha": 0.3,
            "beta": 2.0,
        },
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.01, 10.0),
        },
        method="L-BFGS-B",
        n_starts=5,
        random_seed=11,
    ),
)
```

You can compare tiny-data and larger-data estimates directly:

```python
print("true params:", {"alpha": 0.2, "beta": 3.0})
print("pilot best params:", pilot_fit_result.best.params)
print("full-data best params:", fit_result.best.params)
```

### Quick Quiz

Click `Show answer` to reveal the answer, then click `Hide answer` to collapse it.

- <span class="cm-quiz" data-answer="Use bounds['beta'] = (0.01, 20.0).">If you want to allow larger fitted beta values, what bound should you edit?</span>
- <span class="cm-quiz" data-answer="Use code edits: `initial_value=params['initial_value']` in `model_factory`, then include `initial_value` in your search-space settings (for example, add `'initial_value': (-2.0, 2.0)` to bounds or add grid values).">If you later decide to estimate initial_value too, what two places must you update?</span>

## Step 7: Check That the Fit Worked

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

Minimum checks before moving on:

1. best log-likelihood is finite,
2. optimizer reports successful termination in most runs,
3. recovered parameters are in plausible ranges (`alpha` in `[0, 1]`,
   `beta > 0`),
4. recovered values are at least directionally close to generating values when
   trial counts are moderate.

## Common Mistakes

What if you fail:

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
