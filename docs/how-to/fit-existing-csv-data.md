# How-to: Fit Existing CSV Data

Use this guide when you already have behavioral data in CSV form and want model
fit outputs quickly.

## 1. Prepare Config (JSON or YAML)

Create `fit_config.json` (or `fit_config.yaml` with the same fields):

```json
{
  "model": {
    "component_id": "asocial_state_q_value_softmax",
    "kwargs": {}
  },
  "estimator": {
    "type": "grid_search",
    "parameter_grid": {
      "alpha": [0.1, 0.2, 0.3],
      "beta": [1.0, 2.0, 3.0],
      "initial_value": [0.0]
    }
  }
}
```

## 2. Run Fit CLI

For a study-level CSV:

```bash
comp-model-fit \
  --config fit_config.json \
  --input-csv study.csv \
  --input-kind study \
  --level study \
  --output-dir fit_out \
  --prefix run1
```

For a single trial CSV:

```bash
comp-model-fit \
  --config fit_config.json \
  --input-csv trial.csv \
  --input-kind trial \
  --output-dir fit_out \
  --prefix run1
```

## 3. Inspect Outputs

The command writes:

- `*_summary.json`: compact fit summary
- CSV artifacts for detailed fit rows (depending on workflow)

## Common Issues

- Invalid config keys: config schemas are strict; unknown keys raise errors.
- Study CSV with multiple subjects at subject-level fitting:
  pass `--level subject --subject-id <id>`.
- Component ID errors: verify IDs in the plugin registry and reference schema.
