# Migration Guide: v1 to `comp_model`

This guide maps internal v1 workflows to the clean-slate `comp_model` API.

## Scope

- v1 behavioral model parity is tracked in `comp_model.models.V1_MODEL_PARITY`.
- Core abstractions are now generic (`DecisionProblem`, `AgentModel`), not
  bandit-first.
- Replay, fitting, recovery, and model selection share one canonical event-log
  representation.

## Package Boundary Mapping

| v1 concern | New package |
| --- | --- |
| core model/problem contracts | `comp_model.core` |
| trial execution loop | `comp_model.runtime` |
| model implementations | `comp_model.models` |
| task/problem implementations | `comp_model.problems` |
| estimators + fitting APIs | `comp_model.inference` |
| parameter/model recovery | `comp_model.recovery` |
| diagnostics + criteria | `comp_model.analysis` |
| plugin registry and manifests | `comp_model.plugins` |

## Core API Mapping

| v1 pattern | New API |
| --- | --- |
| single-dataset MLE fit | `fit_model(...)` / `fit_model_from_registry(...)` |
| config-driven MLE fit | `fit_dataset_from_config(...)` |
| MAP fit | `fit_map_model(...)` / `fit_map_dataset_from_config(...)` |
| posterior sampling | `sample_posterior_model(...)` / `sample_posterior_dataset_from_config(...)` |
| subject/study fit loops | `fit_subject_data(...)`, `fit_study_data(...)` |
| subject/study MAP loops | `fit_map_subject_data(...)`, `fit_map_study_data(...)` |
| model comparison | `compare_candidate_models(...)` and study/subject variants |
| parameter recovery | `run_parameter_recovery(...)` |
| model recovery | `run_model_recovery(...)` |

## Model Name Mapping

Canonical naming is mechanism-first and descriptive. Full mapping lives in
`comp_model.models.V1_MODEL_PARITY`.

Examples:

| v1 model | canonical component id | canonical class |
| --- | --- | --- |
| `QRL` | `asocial_state_q_value_softmax` | `AsocialStateQValueSoftmaxModel` |
| `QRL_Stay` | `asocial_state_q_value_softmax_perseveration` | `AsocialStateQValueSoftmaxPerseverationModel` |
| `UnidentifiableQRL` | `asocial_state_q_value_softmax_split_alpha` | `AsocialStateQValueSoftmaxSplitAlphaModel` |
| `VS` | `social_self_outcome_value_shaping` | `SocialSelfOutcomeValueShapingModel` |
| `Vicarious_RL` | `social_observed_outcome_q` | `SocialObservedOutcomeQModel` |
| `Vicarious_VS` | `social_observed_outcome_value_shaping` | `SocialObservedOutcomeValueShapingModel` |

## Config Migration

- New config parsers are strict: unknown keys raise `ValueError`.
- Reference `docs/config_schemas.md` for accepted config shapes.
- Likelihood configuration is explicit (`action_replay` vs
  `actor_subset_replay`) and is shared by fitting and recovery.

## Parity Benchmark Workflow

Use parity fixtures to compare v1 likelihoods against this implementation.

1. Prepare fixture JSON using `docs/parity_fixture_template.json`.
2. Run:

```bash
python scripts/run_parity_benchmark.py \
  --fixture docs/parity_fixture_template.json \
  --output-csv parity_report.csv
```

Installed package command:

```bash
comp-model-parity \
  --fixture docs/parity_fixture_template.json \
  --output-csv parity_report.csv
```

The command returns:
- `0` when all fixture cases pass tolerance checks.
- `1` when one or more cases fail.

You can also export the static v1 mapping as a machine-readable matrix:

```python
from comp_model.analysis import (
    build_model_parity_matrix,
    write_model_parity_matrix_csv,
    write_model_parity_matrix_json,
)

rows = build_model_parity_matrix()
write_model_parity_matrix_json(rows, "parity_matrix.json")
write_model_parity_matrix_csv(rows, "parity_matrix.csv")
```

## Known Non-Goals

- Legacy aliases are removed in current mainline; use canonical names only.
- Wrapper models (`ConditionedSharedDeltaModel`,
  `ConditionedSharedDeltaSocialModel`) remain constructor-driven rather than
  plugin-instantiated zero-argument components.
