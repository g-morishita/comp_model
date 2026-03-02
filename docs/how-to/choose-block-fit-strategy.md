# Choose a Block Fit Strategy

When a subject has multiple blocks, you can choose how parameters are shared
across those blocks.

- `independent`:
  fit each block separately, then aggregate summaries.
- `joint`:
  fit one shared parameter set across all blocks.

Use `joint` when your scientific question assumes one parameter set per
subject. Use `independent` when you expect block-specific drift or want
block-level estimates.

## Python API

```python
from comp_model.inference import FitSpec, fit_subject_data

result = fit_subject_data(
    subject_data,
    model_component_id="asocial_state_q_value_softmax",
    fit_spec=FitSpec(
        inference="mle",
        initial_params={"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        bounds={"alpha": (0.0, 1.0), "beta": (0.01, 20.0), "initial_value": (None, None)},
    ),
    block_fit_strategy="joint",  # or "independent"
)

print(result.fit_mode)        # "joint"
print(result.input_n_blocks)  # original number of blocks in subject_data
```

## YAML Config

```yaml
model:
  component_id: asocial_state_q_value_softmax
  kwargs: {}

estimator:
  type: mle
  initial_params:
    alpha: 0.3
    beta: 2.0
    initial_value: 0.0
  bounds:
    alpha: [0.0, 1.0]
    beta: [0.01, 20.0]
    initial_value: [null, null]

block_fit_strategy: joint
```

This key is supported in subject- and study-level config entrypoints for:

- MLE (`fit_subject_from_config`, `fit_study_from_config`)
- hierarchical Stan MAP and posterior sampling
  (`sample_subject_hierarchical_posterior_from_config`,
  `sample_study_hierarchical_posterior_from_config`)

## Recovery Workflows

Model recovery config also accepts `block_fit_strategy` at top level. This is
applied when simulated datasets are subject- or study-level.

```yaml
block_fit_strategy: joint
```
