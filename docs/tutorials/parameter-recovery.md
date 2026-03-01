# Tutorial: Parameter Recovery

This tutorial runs an end-to-end parameter-recovery analysis from a config file
and verifies the resulting artifacts.

## Step 1: Create a Recovery Config

Create `parameter_recovery.json` (or `parameter_recovery.yaml` with the same fields):

```json
{
  "problem": {
    "component_id": "stationary_bandit",
    "kwargs": {
      "reward_probabilities": [0.2, 0.8]
    }
  },
  "generating_model": {
    "component_id": "asocial_state_q_value_softmax",
    "kwargs": {}
  },
  "fitting_model": {
    "component_id": "asocial_state_q_value_softmax",
    "kwargs": {}
  },
  "estimator": {
    "type": "grid_search",
    "parameter_grid": {
      "alpha": [0.3],
      "beta": [2.0],
      "initial_value": [0.0]
    }
  },
  "true_parameter_sets": [
    {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
    {"alpha": 0.2, "beta": 1.0, "initial_value": 0.0}
  ],
  "n_trials": 40,
  "seed": 11
}
```

## Step 2: Run Recovery

```bash
comp-model-recovery \
  --config parameter_recovery.json \
  --mode parameter \
  --output-dir recovery_out \
  --prefix param
```

## Step 3: Verify Artifacts

You should get:

- `recovery_out/param_parameter_cases.csv`
- `recovery_out/param_parameter_summary.json`

Quick check:

```bash
python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("recovery_out/param_parameter_summary.json").read_text(encoding="utf-8"))
print(payload["mode"], payload["n_cases"])
PY
```

Expected:

- mode is `parameter`
- `n_cases` matches the number of `true_parameter_sets`

## Step 4: Interpret

- Use case-level CSV to inspect per-parameter estimation error.
- Use summary JSON for aggregate error metrics.

## Next Tutorial

Continue with [Model Recovery](model-recovery.md).
