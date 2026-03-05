# How-to: Fit Multiple Blocks with One Shared Parameter Set

In [End-to-End Simulation and Fit](../tutorials/first-end-to-end-fit.md), we fit one block (`EpisodeTrace`).
A common next step is:

- one subject,
- multiple blocks,
- one shared parameter set across all blocks.

This tutorial shows that workflow using `block_fit_strategy="joint"`.

## When to Use This

Use joint block fitting when your scientific assumption is that a subject has one parameter set that should explain all blocks.

If you want per-block parameters instead, use `block_fit_strategy="independent"`.

## Step 1: Simulate Multiple Blocks for One Subject

```python
from comp_model.core import BlockData, SubjectData, trial_decisions_from_trace
from comp_model.models import AsocialStateQValueSoftmaxModel
from comp_model.problems import StationaryBanditProblem
from comp_model.runtime import SimulationConfig, run_episode

true_params = {"alpha": 0.2, "beta": 3.0}

blocks: list[BlockData] = []
for block_index, seed in enumerate((11, 12, 13), start=1):
    trace = run_episode(
        problem=StationaryBanditProblem(reward_probabilities=[0.2, 0.8]),
        model=AsocialStateQValueSoftmaxModel(
            alpha=true_params["alpha"],
            beta=true_params["beta"],
            initial_value=0.0,
        ),
        config=SimulationConfig(n_trials=80, seed=seed),
    )
    blocks.append(
        BlockData(
            block_id=f"b{block_index}",
            trials=trial_decisions_from_trace(trace),
        )
    )

subject = SubjectData(subject_id="s1", blocks=tuple(blocks))
print("n_blocks:", len(subject.blocks))
```

## Step 2: Fit One Shared Parameter Set Across Blocks

```python
from comp_model.inference import FitSpec, fit_subject_data

fit_spec = FitSpec(
    inference="mle",
    initial_params={"alpha": 0.3, "beta": 2.0},
    bounds={"alpha": (0.0, 1.0), "beta": (0.01, 10.0)},
    method="L-BFGS-B",
    n_starts=8,
    random_seed=21,
)

joint_result = fit_subject_data(
    subject,
    model_component_id="asocial_state_q_value_softmax",
    fit_spec=fit_spec,
    model_kwargs={"initial_value": 0.0},  # fixed parameter
    block_fit_strategy="joint",
)

print("fit_mode:", joint_result.fit_mode)                 # joint
print("input_n_blocks:", joint_result.input_n_blocks)     # 3
print("stored_block_results:", len(joint_result.block_results))  # 1 (joint summary)
print("joint best params:", joint_result.mean_best_params)
print("joint total log-likelihood:", joint_result.total_log_likelihood)
```

What `joint` means here:

- the optimizer searches for one `alpha` and one `beta`,
- likelihood is summed across all subject blocks during fitting,
- output stores one joint block summary (`block_id="__joint__"`).

## Step 3: Compare with Independent Block Fits

```python
independent_result = fit_subject_data(
    subject,
    model_component_id="asocial_state_q_value_softmax",
    fit_spec=fit_spec,
    model_kwargs={"initial_value": 0.0},
    block_fit_strategy="independent",
)

print("independent stored blocks:", len(independent_result.block_results))  # 3
print("independent mean best params:", independent_result.mean_best_params)
print("independent total log-likelihood:", independent_result.total_log_likelihood)
```

Use this comparison to verify that your modeling assumption matches your goal:

- `joint`: one parameter set per subject.
- `independent`: one parameter set per block.

## Step 4: Config-Driven Version (Still Script, Not CLI)

If you prefer declarative configs in Python:

```python
from comp_model.inference import fit_subject_from_config

config = {
    "model": {
        "component_id": "asocial_state_q_value_softmax",
        "kwargs": {"initial_value": 0.0},
    },
    "estimator": {
        "type": "mle",
        "solver": "scipy_minimize",
        "initial_params": {"alpha": 0.3, "beta": 2.0},
        "bounds": {"alpha": [0.0, 1.0], "beta": [0.01, 10.0]},
        "method": "L-BFGS-B",
        "n_starts": 8,
        "random_seed": 21,
    },
    "block_fit_strategy": "joint",
}

joint_result_cfg = fit_subject_from_config(subject, config=config)
print(joint_result_cfg.fit_mode)
print(joint_result_cfg.mean_best_params)
```

## Practical Checks

Before trusting estimates:

1. Confirm `fit_mode` is `"joint"`.
2. Confirm `input_n_blocks` matches your subject data.
3. Confirm fitted values are plausible and not stuck at bounds.

## Next Steps

- For a compact API reference, see
  [Choose a Block Fit Strategy](../how-to/choose-block-fit-strategy.md).
- For Bayesian multi-block workflows, see
  [End-to-End Simulation and Hierarchical Bayesian Fit](../tutorials/end-to-end-hierarchical-bayesian-fit.md).
