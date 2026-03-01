"""CLI helpers for config-driven model comparison from tabular CSV input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from comp_model.core import load_config_mapping

from .model_selection import ModelComparisonResult
from .model_selection_tabular import (
    compare_study_csv_candidates_from_config,
    compare_trial_csv_candidates_from_config,
)
from .serialization import (
    write_model_comparison_csv,
    write_study_model_comparison_csv,
    write_study_model_comparison_subject_csv,
    write_subject_model_comparison_csv,
)
from .study_model_selection import StudyModelComparisonResult, SubjectModelComparisonResult


def run_model_comparison_cli(argv: Sequence[str] | None = None) -> int:
    """Run config-driven model comparison from tabular CSV input.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        CLI argument list. When ``None``, process arguments are used.

    Returns
    -------
    int
        Exit code (`0` on success).
    """

    parser = argparse.ArgumentParser(description="Run comp_model model comparison from CSV and config.")
    parser.add_argument("--config", required=True, help="Path to model-comparison JSON or YAML config.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--input-kind",
        choices=("trial", "study"),
        default="trial",
        help="Input CSV type.",
    )
    parser.add_argument(
        "--level",
        choices=("auto", "dataset", "subject", "study"),
        default="auto",
        help="Comparison aggregation level.",
    )
    parser.add_argument(
        "--subject-id",
        default=None,
        help="Subject ID used when --level subject and --input-kind study.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output CSV and summary JSON.",
    )
    parser.add_argument(
        "--prefix",
        default="compare",
        help="Output filename prefix.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = _load_config_mapping(args.config)
    input_kind = str(args.input_kind)
    level = _resolve_level(input_kind=input_kind, level=str(args.level))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)

    if input_kind == "trial":
        result = compare_trial_csv_candidates_from_config(
            str(args.input_csv),
            config=config,
        )
    else:
        result = compare_study_csv_candidates_from_config(
            str(args.input_csv),
            config=config,
            level=level,
            subject_id=args.subject_id,
        )

    output_paths = _write_output_artifacts(
        result=result,
        output_dir=output_dir,
        prefix=prefix,
    )
    summary_path = output_dir / f"{prefix}_summary.json"
    summary = {
        "input_kind": input_kind,
        "level": level,
        **_comparison_result_summary(result),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Model comparison complete: input_kind={input_kind}, level={level}")
    for label, path in output_paths.items():
        print(f"{label}: {path}")
    print(f"Summary JSON: {summary_path}")
    return 0


def _resolve_level(*, input_kind: str, level: str) -> str:
    """Resolve level defaults and validate level/input combinations."""

    if input_kind == "trial":
        if level in {"auto", "dataset"}:
            return "dataset"
        raise ValueError("--level must be 'auto' or 'dataset' for --input-kind trial")

    if level == "auto":
        return "study"
    if level not in {"study", "subject"}:
        raise ValueError("--level must be one of {'auto', 'study', 'subject'} for --input-kind study")
    return level


def _write_output_artifacts(
    *,
    result: ModelComparisonResult | SubjectModelComparisonResult | StudyModelComparisonResult,
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """Write comparison outputs as CSV files and return created paths."""

    if isinstance(result, StudyModelComparisonResult):
        aggregate_path = write_study_model_comparison_csv(
            result,
            output_dir / f"{prefix}_study_comparison.csv",
        )
        subject_path = write_study_model_comparison_subject_csv(
            result,
            output_dir / f"{prefix}_study_subject_comparison.csv",
        )
        return {
            "Study comparison CSV": aggregate_path,
            "Study per-subject CSV": subject_path,
        }

    if isinstance(result, SubjectModelComparisonResult):
        subject_path = write_subject_model_comparison_csv(
            result,
            output_dir / f"{prefix}_subject_comparison.csv",
        )
        return {"Subject comparison CSV": subject_path}

    dataset_path = write_model_comparison_csv(
        result,
        output_dir / f"{prefix}_dataset_comparison.csv",
    )
    return {"Dataset comparison CSV": dataset_path}


def _comparison_result_summary(
    result: ModelComparisonResult | SubjectModelComparisonResult | StudyModelComparisonResult,
) -> dict[str, Any]:
    """Build compact JSON-serializable summary from comparison outputs."""

    # Shared top-level fields across dataset/subject/study comparison results.
    summary: dict[str, Any] = {
        "result_type": type(result).__name__,
        "criterion": str(result.criterion),
        "n_observations": int(result.n_observations),
        "selected_candidate_name": str(result.selected_candidate_name),
        "n_candidates": int(len(result.comparisons)),
    }

    if isinstance(result, SubjectModelComparisonResult):
        summary["subject_id"] = str(result.subject_id)
    if isinstance(result, StudyModelComparisonResult):
        summary["n_subjects"] = int(result.n_subjects)

    return summary


def _load_config_mapping(path: str | Path) -> dict[str, Any]:
    """Load one config object from JSON or YAML path."""

    return load_config_mapping(path)


def main() -> None:
    """Execute model-comparison CLI and exit with returned code."""

    raise SystemExit(run_model_comparison_cli())


if __name__ == "__main__":
    main()


__all__ = ["main", "run_model_comparison_cli"]
