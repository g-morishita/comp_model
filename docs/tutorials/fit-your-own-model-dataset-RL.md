# Tutorial: Fit Your Own Dataset (Simple Reinforcement Learning Task)

This tutorial shows a workflow to fit a model to your own dataset.

## Why this matters

- Many fitting issues come from data-shape assumptions (trial ordering,
  subject/block identifiers, and action coding). Converting once into the
  canonical `StudyData` format makes those assumptions explicit.
- `StudyData` validation catches common problems early (non-unique subject IDs,
  non-contiguous trial indices, unsorted trials) before you run expensive fits.
- Once your dataset is represented as `StudyData`, you can use study-level
  utilities like `fit_study_data(...)` and `fit_study_csv_from_config(...)`.

In this tutorial, you will:

1. convert your data,
2. check converted data,
3. fit data,
4. see the result.

## Prerequisites

- Python 3.11+
- Working installation of `comp_model`

If you have not installed and verified your environment yet, complete
[Install and Verify](install-and-verify.md) first.

## Raw CSV assumptions

This tutorial assumes your raw CSV has one decision row per trial with these
columns:

| column | meaning |
|---|---|
| `subject_id` | subject identifier |
| `block_id` | block identifier |
| `trial_number` | trial order in that block (can start at 1) |
| `choice` | chosen action (here: `0` or `1`) |
| `reward` | observed reward |

If you have only one subject or one block, keep one fixed value in
`subject_id`/`block_id`.

## Canonical data model (what you must build)

For this tutorial, the fitter expects a `StudyData` object. Its shape is:

1. `StudyData(subjects=...)`
2. each item is `SubjectData(subject_id=..., blocks=...)`
3. each block is `BlockData(block_id=..., trials=...)`
4. each trial row is `TrialDecision(...)`

Minimal meaning of each type:

- `TrialDecision`: one observed decision row inside one trial.
- `BlockData`: one block/session for one subject.
- `SubjectData`: one subject containing one or more blocks.
- `StudyData`: all subjects in your dataset.

Validation rules that matter during conversion:

- `SubjectData.subject_id` must be non-empty.
- `StudyData.subject_id` values must be unique across subjects.
- each `SubjectData` must include at least one block.
- each `BlockData` must include `trials` (or `event_trace`).
- trial rows in one block must be sorted by `(trial_index, decision_index)`.
- `trial_index` must be contiguous starting at `0` within each block.

Raw CSV -> canonical field mapping used in this tutorial:

| raw column | canonical target |
|---|---|
| `subject_id` | `SubjectData.subject_id` |
| `block_id` | `BlockData.block_id` |
| `trial_number` | `TrialDecision.trial_index` (converted to `0..N-1`) |
| `choice` | `TrialDecision.action` |
| `reward` | `TrialDecision.reward` and `TrialDecision.outcome={"reward": ...}` |

`TrialDecision` fields set by this tutorial script:

- `trial_index`: `0..N-1` per block (required by validation)
- `decision_index`: `0` (single decision per trial)
- `actor_id`: `"subject"`
- `available_actions`: `(0, 1)` (edit this for your task)
- `action`: value from `choice`
- `observation`: `{"raw_trial_index": original_trial_number}`
- `outcome`: `{"reward": reward}`
- `reward`: numeric reward

## How this differs from `run_episode` output

- `run_episode(...)` returns an `EpisodeTrace` for one simulated episode
  (event stream: observation/decision/outcome/update).
- `StudyData` is a dataset container for fitting many subjects and blocks.

So they are different layers:

- Runtime/simulation layer: `EpisodeTrace`
- Data/analysis layer: `TrialDecision` -> `BlockData` -> `SubjectData` -> `StudyData`

You can bridge between them:

- `trial_decisions_from_trace(trace)` converts one trace to tabular decisions.
- `trace_from_trial_decisions(decisions)` converts tabular decisions to one trace.
- `BlockData(..., event_trace=trace)` stores a trace directly in a block.

Which API to use:

- one dataset/one block: `fit_dataset(...)` accepts `EpisodeTrace`, `BlockData`, or
  `Sequence[TrialDecision]`.
- many subjects/blocks: `fit_study_data(...)` expects `StudyData`.

## Step 1: Convert your data

`comp_model` provides a mapped conversion helper for this step:
`read_mapped_study_csv(...)`.

You pass a `column_mapping` dictionary from canonical fields to your raw CSV
columns.

Create `scripts/fit_your_own_dataset.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from comp_model.inference import (
    fit_study_csv_from_config,
    write_study_fit_records_csv,
    write_study_fit_summary_csv,
)
from comp_model.io import (
    read_mapped_study_csv,
    read_study_decisions_csv,
    study_decision_rows,
    write_study_decisions_csv,
)

RAW_INPUT_CSV = Path("data/raw/my_choices.csv")  # Change to your raw dataset path
CANONICAL_STUDY_CSV = Path("data/processed/study_decisions.csv")  # Change to processed-data path
OUTPUT_DIR = Path("fit_out")

# Update when your action set differs.
AVAILABLE_ACTIONS = (0, 1)

# Map canonical fields -> your raw CSV column names.
COLUMN_MAPPING = {
    "subject_id": "subject_id",
    "block_id": "block_id",
    "trial_index": "trial_number",
    "action": "choice",
    "reward": "reward",
}

FIT_CONFIG = {
    "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
    "estimator": {
        "type": "mle",
        "solver": "grid_search",
        "parameter_grid": {
            "alpha": [0.1, 0.3, 0.5, 0.7],
            "beta": [1.0, 2.0, 4.0, 8.0],
            "initial_value": [0.0],
        },
    },
}


def convert_your_data() -> Path:
    study = read_mapped_study_csv(
        RAW_INPUT_CSV,
        column_mapping=COLUMN_MAPPING,
        available_actions=AVAILABLE_ACTIONS,
    )
    output_path = write_study_decisions_csv(study, CANONICAL_STUDY_CSV)
    print(f"Converted CSV written to: {output_path}")
    return output_path


def check_converted_data(canonical_csv_path: Path) -> None:
    study = read_study_decisions_csv(canonical_csv_path)

    n_subjects = len(study.subjects)
    n_blocks = sum(len(subject.blocks) for subject in study.subjects)
    n_trials = sum(block.n_trials for subject in study.subjects for block in subject.blocks)

    print("Converted data summary")
    print(f"  subjects: {n_subjects}")
    print(f"  blocks:   {n_blocks}")
    print(f"  trials:   {n_trials}")

    preview = study_decision_rows(study)[:3]
    print("  preview rows:")
    for row in preview:
        print(
            {
                "subject_id": row["subject_id"],
                "block_id_json": row["block_id_json"],
                "trial_index": row["trial_index"],
                "action_json": row["action_json"],
                "reward": row["reward"],
            }
        )


def fit_data(canonical_csv_path: Path) -> Path:
    result = fit_study_csv_from_config(
        str(canonical_csv_path),
        config=FIT_CONFIG,
        level="study",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = write_study_fit_records_csv(result, OUTPUT_DIR / "study_fit_rows.csv")
    summary_path = write_study_fit_summary_csv(result, OUTPUT_DIR / "study_fit_summary.csv")

    overview = {
        "n_subjects": int(result.n_subjects),
        "total_log_likelihood": float(result.total_log_likelihood),
        "subjects": [
            {
                "subject_id": subject.subject_id,
                "total_log_likelihood": float(subject.total_log_likelihood),
                "mean_best_params": {
                    key: float(value)
                    for key, value in sorted(subject.mean_best_params.items())
                },
            }
            for subject in result.subject_results
        ],
    }
    overview_path = OUTPUT_DIR / "study_fit_overview.json"
    overview_path.write_text(json.dumps(overview, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Fit rows written to: {rows_path}")
    print(f"Fit summary written to: {summary_path}")
    print(f"Fit overview written to: {overview_path}")
    return overview_path


def see_the_result(overview_path: Path) -> None:
    overview = json.loads(overview_path.read_text(encoding="utf-8"))

    print("Fit result")
    print(f"  n_subjects: {overview['n_subjects']}")
    print(f"  total_log_likelihood: {overview['total_log_likelihood']:.3f}")
    for subject in overview["subjects"]:
        print(f"  subject={subject['subject_id']}")
        print(f"    total_log_likelihood={subject['total_log_likelihood']:.3f}")
        print(f"    mean_best_params={subject['mean_best_params']}")


def main() -> None:
    canonical = convert_your_data()
    check_converted_data(canonical)
    overview = fit_data(canonical)
    see_the_result(overview)


if __name__ == "__main__":
    main()
```

## Step 2: Check converted data

In this workflow, `check_converted_data(...)` runs immediately after conversion.
It verifies the converted file by:

- loading the canonical study CSV with `read_study_decisions_csv`,
- printing `subjects`, `blocks`, and `trials`,
- printing preview rows from `study_decision_rows(...)`.

## Step 3: Fit data

`fit_data(...)` performs study-level fitting directly from the converted CSV via
`fit_study_csv_from_config(...)` and writes:

- `fit_out/study_fit_rows.csv` (candidate-level rows),
- `fit_out/study_fit_summary.csv` (subject-level summary),
- `fit_out/study_fit_overview.json` (compact summary for quick inspection).

`fit_study_csv_from_config(...)` is a normal Python API call. In this tutorial,
`FIT_CONFIG` is an in-script Python dictionary (not a required YAML file).

If you prefer a no-config API, you can call `fit_study_data(...)` with
`FitSpec(...)` directly:

```python
from comp_model.inference import FitSpec, fit_study_data
from comp_model.io import read_study_decisions_csv

study = read_study_decisions_csv(canonical_csv_path)
result = fit_study_data(
    study,
    model_component_id="asocial_state_q_value_softmax",
    fit_spec=FitSpec(
        solver="grid_search",
        parameter_grid={
            "alpha": [0.1, 0.3, 0.5, 0.7],
            "beta": [1.0, 2.0, 4.0, 8.0],
            "initial_value": [0.0],
        },
    ),
)
```

## Step 4: See the result

`see_the_result(...)` reads `study_fit_overview.json` and prints:

- total study log-likelihood,
- per-subject log-likelihood,
- per-subject mean best parameters.

Run the script:

```bash
python3 scripts/fit_your_own_dataset.py
```

## Notes

- For tasks with more than two actions, update `AVAILABLE_ACTIONS`.
- Replace grid values in `FIT_CONFIG["estimator"]["parameter_grid"]` with values
  appropriate for your model/task.
- If your model is not `asocial_state_q_value_softmax`, replace
  `FIT_CONFIG["model"]["component_id"]` accordingly.
