# How-to: Compare Candidate Models

Use this guide to compare multiple candidate models on the same dataset.

## 1. Create Comparison Config

Create `compare_config.json`:

```json
{
  "criterion": "aic",
  "candidates": [
    {
      "name": "q_softmax",
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
      },
      "n_parameters": 3
    },
    {
      "name": "q_softmax_stay",
      "model": {
        "component_id": "asocial_state_q_value_softmax_perseveration",
        "kwargs": {}
      },
      "estimator": {
        "type": "grid_search",
        "parameter_grid": {
          "alpha": [0.1, 0.2, 0.3],
          "beta": [1.0, 2.0, 3.0],
          "initial_value": [0.0],
          "stay_bias": [-1.0, 0.0, 1.0]
        }
      },
      "n_parameters": 4
    }
  ]
}
```

## 2. Run Comparison CLI

```bash
comp-model-compare \
  --config compare_config.json \
  --input-csv study.csv \
  --input-kind study \
  --level study \
  --output-dir compare_out \
  --prefix run1
```

## 3. Read Outputs

Outputs include:

- `*_summary.json`: selected candidate and criterion summary
- comparison CSV files with per-candidate scores

For study-level comparison, both aggregate and per-subject CSV files are
written.

## Notes

- `waic` and `psis_loo` require candidate fitters that expose posterior
  pointwise log-likelihood draws.
- If you compare subject-level results for multi-subject study CSV input, pass
  `--level subject --subject-id <id>`.
