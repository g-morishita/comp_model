# Tutorial: Parameter Recovery (Python Workflow)

This tutorial walks through a parameter recovery analysis: a practical check
of whether your model-fitting procedure can recover known parameter values
across many simulated datasets.

## Why this matters

- Parameter recovery helps you understand identifiability: which parameters
  (and which ranges) can be reliably estimated from the kind of data your task
  produces, and where the model starts to struggle (Wilson & Collins 2019).
- It asks: "If the true parameter value is X, and we fit the model to data generated with X, do we get back something close to X, and over what range does this hold?"

In this tutorial, you will:

1. build a reusable fitting function,
2. run recovery with true parameters sampled from distributions,
3. inspect recovery error summaries and case-level results,
4. adjust distribution ranges and sample size when recovery is weak.

## Prerequisites

- Python 3.11+
- Working installation of `comp_model`

If you have not installed and verified your environment yet, complete
[Install and Verify](install-and-verify.md) first.

## Step 1: Build a reusable fitting function

We will use the reinforcement learning model family for generation and fitting:
`AsocialStateQValueSoftmaxModel` on `StationaryBanditProblem`.

First, define small reusable functions:

- `problem_factory()` creates one fresh task,
- `model_factory(params)` creates one model from candidate/true params,
- `fit_function(trace)` fits one synthetic dataset.

```python
from comp_model.inference import MLEFitSpec, fit_trace
from comp_model.models import AsocialStateQValueSoftmaxModel
from comp_model.problems import StationaryBanditProblem


def problem_factory() -> StationaryBanditProblem:
    return StationaryBanditProblem(reward_probabilities=[0.2, 0.8])


def model_factory(params: dict[str, float]) -> AsocialStateQValueSoftmaxModel:
    return AsocialStateQValueSoftmaxModel(
        alpha=params["alpha"],
        beta=params["beta"],
        initial_value=0.0,  # fixed in this tutorial
    )


def fit_function(trace):
    return fit_trace(
        trace,
        model_factory=model_factory,
        fit_spec=MLEFitSpec(
            solver="scipy_minimize",
            initial_params={"alpha": 0.3, "beta": 2.0},
            bounds={
                "alpha": (0.0, 1.0),
                "beta": (0.01, 10.0),
            },
            method="L-BFGS-B",
            n_starts=5,
            random_seed=10,
        ),
    )
```

### Quick quiz

??? question "If you want to estimate `initial_value` too, what should you change?"
    In `model_factory`, replace `initial_value=0.0` with `initial_value=params["initial_value"]`, then add `initial_value` to `initial_params` and `bounds` inside `MLEFitSpec`.

## Step 2: Run parameter recovery from parameter distributions

Now run recovery by sampling true parameters directly from distributions.
This gives broad coverage of parameter space in one run.

```python
from comp_model.recovery import run_parameter_recovery

result = run_parameter_recovery(
    problem_factory=problem_factory,
    model_factory=model_factory,
    fit_function=fit_function,
    true_parameter_distributions={
        "alpha": {"distribution": "uniform", "lower": 0.05, "upper": 0.8},
        "beta": {"distribution": "uniform", "lower": 0.5, "upper": 8.0},
    },
    n_parameter_sets=30,
    n_trials=120,
    seed=21,
)

print("n_cases:", len(result.cases))
print("mean_absolute_error:", result.mean_absolute_error)
print("mean_signed_error:", result.mean_signed_error)
print("true_estimate_correlation:", result.true_estimate_correlation)
```

What `run_parameter_recovery(...)` does for each case:

1. sample one true parameter vector from your declared true-parameter source,
2. simulate one synthetic dataset from that vector,
3. fit the dataset with `fit_function`,
4. store true vs estimated parameters and fit score.

### What each argument means

- `problem_factory=problem_factory`:
  function that returns a fresh task environment for each synthetic dataset.
- `model_factory=model_factory`:
  function that receives one true-parameter dictionary and returns a generating
  model instance.
- `fit_function=fit_function`:
  function that fits one simulated dataset and returns fit results.
- `true_parameter_distributions={...}`:
  where true parameter values are sampled from for each recovery case.
- `n_parameter_sets=30`:
  number of recovery cases (how many true parameter vectors to sample).
- `n_trials=120`:
  number of trials per simulated dataset.
- `seed=21`:
  random seed for reproducible true-parameter sampling and simulation seeds.

### How to specify `true_parameter_distributions`

`true_parameter_distributions` is a dictionary:

- key: parameter name (must match what `model_factory(params)` expects),
- value: one distribution specification for that parameter.

Supported specifications:

- `{"distribution": "uniform", "lower": float, "upper": float}`
- `{"distribution": "normal", "mean": float, "std": float}`
- `{"distribution": "beta", "alpha": float, "beta": float}`
- `{"distribution": "log_normal", "mean_log": float, "std_log": float}`

Practical notes:

- Parameters are sampled independently.
- `n_parameter_sets` controls how many parameter vectors are drawn.
- `seed` makes the draws reproducible.
- To effectively fix a parameter at one value, use
  `{"distribution": "uniform", "lower": x, "upper": x}`.

??? note "hierarchical sampling in Python"

    If you want hierarchical sampling (population draw + individual deviations),
    use `true_parameter_sampling` instead of `true_parameter_distributions`.

    ```python
    result_hier = run_parameter_recovery(
        problem_factory=problem_factory,
        model_factory=model_factory,
        fit_function=fit_function,
        true_parameter_sampling={
            "mode": "hierarchical",
            "space": "z",
            "n_parameter_sets": 30,
            "population": {
                "alpha": {"distribution": "normal", "mean": 0.0, "std": 0.3},
                "beta": {"distribution": "normal", "mean": 1.0, "std": 0.4},
            },
            "individual_sd": {
                "alpha": 0.2,
                "beta": 0.2,
            },
            "transforms": {
                "alpha": "unit_interval_logit",
                "beta": "positive_log",
            },
        },
        n_trials=120,
        seed=21,
    )
    ```

    Here, `population` is sampled once, then each case is sampled around that
    population using `individual_sd`.


### Quick quiz

??? question "How do you sample more points in parameter space?"
    Increase `n_parameter_sets`.

??? question "How do you make sampling reproducible?"
    Set `seed` in `run_parameter_recovery(...)`.

## Step 3: Inspect case-level estimates and write outputs

Look at each recovery case directly, then save outputs for later analysis.

```python
for case in result.cases:
    print("case", case.case_index)
    print("  true:", case.true_params)
    print("  estimated:", case.estimated_params)
    print("  log_likelihood:", case.best_log_likelihood)
```

## Step 4: Adjust sampling design when recovery is weak

If recovery is weak, adjust sampling ranges and data volume, then rerun.

```python
result_wider = run_parameter_recovery(
    problem_factory=problem_factory,
    model_factory=model_factory,
    fit_function=fit_function,
    true_parameter_distributions={
        "alpha": {"distribution": "uniform", "lower": 0.01, "upper": 0.95},
        "beta": {"distribution": "uniform", "lower": 0.2, "upper": 12.0},
    },
    n_parameter_sets=60,   # more sampled cases
    n_trials=200,          # more data per case
    seed=22,
)

print("wider n_cases:", len(result_wider.cases))
print("wider MAE:", result_wider.mean_absolute_error)
print("true_estimate_correlation:", result_wider.true_estimate_correlation)
```

Typical levers:

1. increase `n_trials` (more information per synthetic dataset),
2. increase `n_parameter_sets` (better coverage of parameter space),
3. revise distribution ranges to focus on plausible regions.

### Quick quiz

??? question "If recovery fails mostly at high beta, what should you change first?"
    Increase `n_trials`, then inspect errors by `true__beta` range.

??? question "If you only care about a narrower alpha regime, where do you encode that?"
    In `true_parameter_distributions[\"alpha\"]` by tightening `lower` and `upper`.

## Next steps

Continue with [Model Recovery](model-recovery.md).

## References

- Wilson RC, Collins AGE. (2019). Ten simple rules for the computational
  modeling of behavioral data. *eLife*, 8:e49547.
  <https://doi.org/10.7554/eLife.49547>
