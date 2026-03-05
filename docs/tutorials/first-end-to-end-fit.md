# Tutorial: End-to-End Simulation and Fit

This tutorial walks through a complete end-to-end simulation workflow:
generate synthetic data and then fit a computational model back to that data.

## Why this matters

- Before working with real datasets, confirm that your model-fitting workflow
  behaves the way you expect and does not contain implementation bugs.
- A standard first check is to simulate synthetic behavior from known parameter
  values and then fit the same model back to those simulated data (Wilson &
  Collins, 2019).
- This end-to-end procedure helps validate the full pipeline, both
  statistically and programmatically.

This tutorial uses:

- task: `StationaryBanditProblem`
- model: `AsocialStateQValueSoftmaxModel`
- fitter: `fit_model(...)`

You will choose a set of parameters, simulate a dataset from those values, and
then fit the same model back to that dataset. If everything is working, the
fitted model should achieve a good fit and return sensible parameter estimates.
Do not worry if the estimates are not perfect yet: at this stage, the goal is
to verify that the full pipeline runs and produces reasonable results.

To keep things simple, this tutorial uses a classic reinforcement-learning
example: a two-armed stationary bandit task paired with a basic Q-learning
model with a softmax choice rule. This bandit + RL setup is widely used as a
"first stop" in computational modeling because it is intuitive, easy to
simulate, and well studied (Sutton & Barto, 2018; Daw, 2011; Wilson & Collins,
2019).

In this tutorial, you will:

1. instantiate a task,
2. instantiate a model,
3. run a short pilot episode,
4. fit the pilot dataset,
5. simulate a larger dataset and refit,
6. sanity-check that the fit worked.

## Prerequisites

- Python 3.11+
- Working installation of `comp_model`

If you have not installed and verified your environment yet, complete
[Install and Verify](install-and-verify.md) first.

## Step 1: Instantiate a task

First, create a task (environment) that your model will act in. Here, use a
simple stationary two-option bandit task where option 0 gives reward with
probability 0.2 and option 1 gives reward with probability 0.8. In `comp_model`,
task environments are implemented as Problem classes (the `DecisionProblem`
interface), and this specific task is implemented by `StationaryBanditProblem`.

To express this in code, use `StationaryBanditProblem` and pass `reward_probabilities=[0.2, 0.8]`.
The list index is the option ID, and the value is that option's Bernoulli reward probability.

```python
from comp_model.problems import StationaryBanditProblem

problem = StationaryBanditProblem(reward_probabilities=[0.2, 0.8])
```

### Quick quiz

Click to reveal the answer.

??? question "You want to increase option 0's reward probability from 0.2 to 0.5. What should you change in the code?"
    Change it to: `reward_probabilities = [0.5, 0.8]`.

??? question "You want to add a third option with reward probability 0.5. What should `reward_probabilities` look like?"
    Add a third value: `reward_probabilities = [0.2, 0.8, 0.5]`.

## Step 2: Instantiate a model

Next, choose a computational model you want to simulate. Here, use a standard
reinforcement learning (RL) model that updates action values from rewards and
chooses probabilistically based on those values.
In `comp_model`, this is `AsocialStateQValueSoftmaxModel`.

To express this in code, set model parameters explicitly:

- `alpha`: learning rate (how fast values update after outcomes),
- `beta`: inverse temperature (how deterministic choices are),
- `initial_value`: starting value for each option before learning.

```python
from comp_model.models import AsocialStateQValueSoftmaxModel

generating_model = AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0)
```

### Quick quiz

Click to reveal the answer.

??? question "What happens if a given alpha is negative?"
    `AsocialStateQValueSoftmaxModel(alpha=-0.2, beta=3.0, initial_value=0.0)` raises `ValueError`: alpha must be in [0, 1].

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

## Step 3: Run a pilot episode

Now that you have both the task and the computational model, you are ready to
generate some synthetic choice data. For this first pilot check, keep the
simulation small (for example, 20 trials).

You can simulate behavior with:

`run_episode(problem, model, config)`

where:

- `problem` is the task environment,
- `model` is the computational model you want to simulate,
- `config` sets trial count and RNG seed for reproducibility.

To create the configuration object, instantiate `SimulationConfig` and specify
the seed and number of trials.

???+ note "What does episode mean?"
    An episode is one full simulation run across consecutive trials.
    You can think of one episode as a single block of trials.
    We borrow the term episode from the reinforcement-learning literature.

```python
from comp_model.runtime import SimulationConfig, run_episode

pilot_trace = run_episode(
    problem=problem,
    model=AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0),
    config=SimulationConfig(n_trials=20, seed=10),
)
```
`pilot_trace` contains 20 trials. Each trial consists of four phases:

`observation` (environment info) -> `decision` (choice) -> `outcome` (feedback)
-> `update` (model learning step).

This is because `run_episode` generates each trial in the following order:

1. `observation`: The model receives information from the environment (e.g., a state).
2. `decision`: The model uses that information and makes a choice based on its internal variables.
3. `outcome`: The environment receives the model's choice and returns an outcome.
4. `update`: The model receives the outcome and updates its internal variables.

In this tutorial we use a stationary bandit task, so the `observation` phase does not contain any additional information.

??? note "How to closely look at trial information"
    To inspect the simulation trial by trial, use the `trial_by` method.
    For example, look at the first trial:
    ```python
    first_trial = pilot_trace.trial_by(0)  # Indexing starts at 0.
    print(first_trial)
    ```

    You will see a list of `SimulationEvent` objects.
    Each event has a `phase` field that tells you which phase it belongs to,
    and a `payload` field whose contents depend on the phase.
    For example, the `outcome` phase includes outcome information in its payload,
    and the `decision` phase includes the model's choice probabilities.

### Quick quiz

Click to reveal the answer.

??? question "If you want the pilot run to simulate 10 trials (keeping the same seed), what should `SimulationConfig` be?"
    `SimulationConfig(n_trials=10, seed=10)`

??? question "If you want a different random sequence but keep trial count fixed, what should you change?"
    Change only the seed, for example `SimulationConfig(n_trials=3, seed=42)`.

??? question "If you want only the events from the second trial, what method call should you use?"
    Use `pilot_trace.by_trial(1)`.

## Step 4: Fit the tiny pilot dataset

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

### How `model_factory` works

`model_factory` is a Python function that takes candidate parameter values and returns a model built with those values.

`fit_model` uses that returned model to compute likelihood on the observed trace.

You can think of `model_factory` as a translator:
candidate parameters -> model instance -> likelihood score.

`model_factory` must return a fresh model instance each time.
This prevents state leakage from previous candidates.

Free versus fixed parameters follow one rule:

- if a value comes from `params[...]`, that parameter is estimated,
- if a value is written as a constant (for example `initial_value=0.0`), that parameter is fixed.

In this example, `alpha` and `beta` are estimated (`params["alpha"]`, `params["beta"]`), while `initial_value` is fixed (`0.0`).

### What `fit_spec` specifies here

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

After running this, compare `pilot best params` with the generating values `{"alpha": 0.2, "beta": 3.0}`.
With only a few trials, estimates are often far from true parameters. This is expected, so the next step is to increase `n_trials` and fit again.

### Quick quiz

Click to reveal the answer.

??? question "If you want to estimate `initial_value` instead of fixing it, what two places must be edited?"
    Change `initial_value=0.0` to `initial_value=params[\"initial_value\"]` in `model_factory`, then include `initial_value` in your search-space settings (for example, add a bound such as `\"initial_value\": (0, 1.0)`).

??? question "If you want to fix `beta` at `3.0` and estimate only `alpha`, what should you change?"
    Set a constant in `model_factory` (for example `beta=3.0` instead of `beta=params[\"beta\"]`), then remove `beta` from your search-space settings (bounds/grid).

## Step 5: Increase trials and simulate a fitting dataset

Now increase the trial count so estimation has enough signal. Here, simulate
120 trials with a fresh model instance.

```python
trace = run_episode(
    problem=problem,
    model=AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0),
    config=SimulationConfig(n_trials=120, seed=11),
)
```

Use a fresh model instance for the full dataset so learned values from the
pilot run do not leak into the fitting dataset.

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

### Quick quiz

Click to reveal the answer.

??? question "If you want to allow larger fitted beta values, what bound should you edit?"
    ```python
    bounds={
        "alpha": (0.0, 1.0),
        "beta": (0.01, 20.0),
    },
    ```

## Step 6: Check that the fit worked

After fitting, do a quick quality check before trusting the estimates.
The goal is to confirm that optimization finished properly and parameter values look plausible.

- `best log-likelihood`: model score on observed choices (higher is better when comparing fits on the same dataset),
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
3. recovered values are at least directionally close to generating values when
   trial counts are moderate.

## Next steps

- Continue with next tutorial: [Parameter Recovery](parameter-recovery.md).
- If you are interested in Bayesian hierarchical model, move on to
  [How to fit Bayesian hierarchical model](../how-to/how-to-fit-bayesian-hierarchical-model.md).

## References

- Wilson RC, Collins AGE. (2019). Ten simple rules for the computational
  modeling of behavioral data. *eLife*, 8:e49547.
  <https://doi.org/10.7554/eLife.49547>
- Sutton RS, Barto AG. (2018). Reinforcement Learning: An Introduction
(2nd ed.). MIT Press.
- Daw ND. (2011). Trial-by-trial data analysis using computational models.
