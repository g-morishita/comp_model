# Tutorial: Model Recovery

This tutorial runs a model-recovery workflow and verifies confusion outputs.

## Step 1: Create a Model-Recovery Config

Create `model_recovery.json`:

```json
{
  "problem": {
    "component_id": "stationary_bandit",
    "kwargs": {
      "reward_probabilities": [0.2, 0.8]
    }
  },
  "generating": [
    {
      "name": "gen_q",
      "model": {
        "component_id": "asocial_state_q_value_softmax",
        "kwargs": {}
      },
      "true_params": {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}
    }
  ],
  "candidates": [
    {
      "name": "cand_q_good",
      "model": {
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
      }
    },
    {
      "name": "cand_q_bad",
      "model": {
        "component_id": "asocial_state_q_value_softmax",
        "kwargs": {}
      },
      "estimator": {
        "type": "grid_search",
        "parameter_grid": {
          "alpha": [0.9],
          "beta": [0.1],
          "initial_value": [1.0]
        }
      }
    }
  ],
  "n_trials": 50,
  "n_replications_per_generator": 5,
  "criterion": "log_likelihood",
  "seed": 9
}
```

## Step 2: Run Recovery

```bash
comp-model-recovery \
  --config model_recovery.json \
  --mode model \
  --output-dir recovery_out \
  --prefix model
```

## Step 3: Verify Artifacts

You should get:

- `recovery_out/model_model_cases.csv`
- `recovery_out/model_model_confusion.csv`
- `recovery_out/model_model_summary.json`

Quick check:

```bash
python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("recovery_out/model_model_summary.json").read_text(encoding="utf-8"))
print(payload["mode"], payload["n_cases"], payload["criterion"])
PY
```

Expected:

- mode is `model`
- `criterion` matches config
- `n_cases` equals number of generated replications

## Step 4: Interpret

- Confusion CSV summarizes how often each generating model is selected as each
  candidate.
- Cases CSV preserves per-dataset selected model and candidate-level summaries.

## Next Tutorial

Continue with [Fit Your Own Dataset](fit-your-own-dataset.md).
