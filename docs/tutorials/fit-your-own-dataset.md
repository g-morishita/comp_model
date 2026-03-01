# Tutorial: Fit Your Own Dataset

This tutorial shows how to fit a model to your own trial/study CSV dataset and
verify outputs.

## Step 1: Prepare a Fitting Config

Create `fit_config.json`:

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

## Step 2: Run Study-Level Fitting

```bash
comp-model-fit \
  --config fit_config.json \
  --input-csv study.csv \
  --input-kind study \
  --level study \
  --output-dir fit_out \
  --prefix your_data
```

For one trial-level CSV:

```bash
comp-model-fit \
  --config fit_config.json \
  --input-csv trial.csv \
  --input-kind trial \
  --output-dir fit_out \
  --prefix your_data
```

## Step 3: Verify Outputs

Check that this file exists:

```bash
ls fit_out/your_data_summary.json
```

Inspect summary keys:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("fit_out/your_data_summary.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload.keys())
PY
```

Expected keys include `result_type` and fit metrics (for example
`best_log_likelihood`).

## Step 4: Optional Subject-Level Fit

For one subject in a multi-subject study CSV:

```bash
comp-model-fit \
  --config fit_config.json \
  --input-csv study.csv \
  --input-kind study \
  --level subject \
  --subject-id s01 \
  --output-dir fit_out \
  --prefix subject_s01
```

## Where Next

- Move to problem-focused extensions in
  [Create a New Problem](../how-to/create-a-new-problem.md).
- Review command and schema details in
  [API Overview](../reference/api-overview.md) and
  [Configuration Schemas](../reference/config_schemas.md).
