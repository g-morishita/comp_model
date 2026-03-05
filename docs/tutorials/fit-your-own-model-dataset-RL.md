# Tutorial: Fit Your Own Dataset (simple reinforcement learning task)

This tutorial shows a workflow to fit a model to your own dataset.

1. Convert your data
2. Check converted data
3. Fit data
4. See the result

## Prerequisites

- Python 3.11+
- Working installation of `comp_model`

If you have not installed and verified your environment yet, complete
[Install and Verify](install-and-verify.md) first.

## Raw CSV Assumption

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

## Step 1: Convert your data

Create `scripts/fit_your_own_dataset.py`:

```python
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    fit_study_csv_from_config,
    write_study_fit_records_csv,
    write_study_fit_summary_csv,
)
from comp_model.io import read_study_decisions_csv, study_decision_rows, write_study_decisions_csv

RAW_INPUT_CSV = Path("data/raw/my_choices.csv"). # Change to a path to your own dataset
CANONICAL_STUDY_CSV = Path("data/processed/study_decisions.csv"). # Change to a path to processed data
OUTPUT_DIR = Path("fit_out")

# Update when your action set differs.
AVAILABLE_ACTIONS = (0, 1)

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
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    with RAW_INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "block_id", "trial_number", "choice", "reward"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"raw CSV missing required columns: {missing}")

        for row_index, row in enumerate(reader, start=2):
            subject_id = str(row["subject_id"]).strip()
            block_id = str(row["block_id"]).strip()
            if not subject_id:
                raise ValueError(f"row {row_index}: subject_id is empty")
            if not block_id:
                raise ValueError(f"row {row_index}: block_id is empty")
            grouped[(subject_id, block_id)].append(row)

    if not grouped:
        raise ValueError("raw CSV has no rows")

    blocks_by_subject: dict[str, list[BlockData]] = defaultdict(list)
    for (subject_id, block_id), rows in grouped.items():
        ordered_rows = sorted(rows, key=lambda item: int(item["trial_number"]))

        trials: list[TrialDecision] = []
        for index, raw in enumerate(ordered_rows):
            choice = int(raw["choice"])
            reward = float(raw["reward"])
            if choice not in AVAILABLE_ACTIONS:
                raise ValueError(
                    f"choice={choice} is outside AVAILABLE_ACTIONS={AVAILABLE_ACTIONS}"
                )
            trials.append(
                TrialDecision(
                    trial_index=index,  # force contiguous 0..N-1 per block
                    decision_index=0,
                    actor_id="subject",
                    available_actions=AVAILABLE_ACTIONS,
                    action=choice,
                    observation={"trial_number": int(raw["trial_number"])},
                    outcome={"reward": reward},
                    reward=reward,
                )
            )

        blocks_by_subject[subject_id].append(
            BlockData(block_id=block_id, trials=tuple(trials))
        )

    subjects = tuple(
        SubjectData(
            subject_id=subject_id,
            blocks=tuple(sorted(blocks, key=lambda block: str(block.block_id))),
        )
        for subject_id, blocks in sorted(blocks_by_subject.items())
    )
    study = StudyData(subjects=subjects)
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
