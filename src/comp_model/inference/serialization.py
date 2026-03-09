"""Serialization helpers for fitting outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .best_fit_summary import extract_best_fit_summary
from .mle.group import BlockFitResult, StudyFitResult, SubjectFitResult
from .model_selection import ModelComparisonResult
from .study_model_selection import StudyModelComparisonResult, SubjectModelComparisonResult


def block_fit_records(block_result: BlockFitResult, *, subject_id: str | None = None) -> list[dict[str, Any]]:
    """Convert one block-fit result to flat row records."""

    rows: list[dict[str, Any]] = []
    for candidate in block_result.fit_result.candidates:
        row: dict[str, Any] = {
            "subject_id": subject_id,
            "block_id": block_result.block_id,
            "n_trials": int(block_result.n_trials),
            "log_likelihood": float(candidate.log_likelihood),
            "is_best": candidate == block_result.fit_result.best,
        }
        for key, value in sorted(candidate.params.items()):
            row[f"param__{key}"] = float(value)
        rows.append(row)
    return rows


def subject_fit_records(subject_result: SubjectFitResult) -> list[dict[str, Any]]:
    """Convert one subject-fit result to flat block/candidate rows."""

    rows: list[dict[str, Any]] = []
    for block in subject_result.block_results:
        rows.extend(block_fit_records(block, subject_id=subject_result.subject_id))
    return rows


def study_fit_records(study_result: StudyFitResult) -> list[dict[str, Any]]:
    """Convert study fit result to flat block/candidate rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(subject_fit_records(subject))
    return rows


def subject_summary_records(subject_result: SubjectFitResult) -> list[dict[str, Any]]:
    """Convert one subject fit result to summary row."""

    fit_mode, input_n_blocks = _subject_fit_mode_and_input_n_blocks(subject_result)
    row: dict[str, Any] = {
        "subject_id": subject_result.subject_id,
        "n_blocks": len(subject_result.block_results),
        "input_n_blocks": int(input_n_blocks),
        "fit_mode": fit_mode,
        "total_log_likelihood": float(subject_result.total_log_likelihood),
    }
    if subject_result.shared_best_params is not None:
        for key, value in sorted(subject_result.shared_best_params.items()):
            row[f"shared_best_param__{key}"] = float(value)
    return [row]


def study_summary_records(study_result: StudyFitResult) -> list[dict[str, Any]]:
    """Convert study fit result to per-subject summary rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(subject_summary_records(subject))
    return rows


def write_records_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write generic rows to CSV."""

    if not rows:
        raise ValueError("rows must not be empty")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(str(key))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_study_fit_records_csv(study_result: StudyFitResult, path: str | Path) -> Path:
    """Write candidate-level study fit rows to CSV."""

    return write_records_csv(study_fit_records(study_result), path)


def write_study_fit_summary_csv(study_result: StudyFitResult, path: str | Path) -> Path:
    """Write subject-level study fit summaries to CSV."""

    return write_records_csv(study_summary_records(study_result), path)


def model_comparison_records(result: ModelComparisonResult) -> list[dict[str, Any]]:
    """Convert model-comparison output to flat row dictionaries."""

    rows: list[dict[str, Any]] = []
    for comparison in result.comparisons:
        best = extract_best_fit_summary(comparison.fit_result)
        row: dict[str, Any] = {
            "criterion": str(result.criterion),
            "n_observations": int(result.n_observations),
            "selected_candidate_name": str(result.selected_candidate_name),
            "candidate_name": str(comparison.candidate_name),
            "is_selected": bool(comparison.candidate_name == result.selected_candidate_name),
            "log_likelihood": float(comparison.log_likelihood),
            "n_parameters": int(comparison.n_parameters),
            "aic": float(comparison.aic),
            "bic": float(comparison.bic),
            "score": float(comparison.score),
        }
        for key, value in sorted(best.params.items()):
            row[f"param__{key}"] = float(value)
        rows.append(row)
    return rows


def subject_model_comparison_records(result: SubjectModelComparisonResult) -> list[dict[str, Any]]:
    """Convert subject-level model-comparison output into row dictionaries."""

    rows: list[dict[str, Any]] = []
    for comparison in result.comparisons:
        rows.append(
            {
                "subject_id": str(result.subject_id),
                "criterion": str(result.criterion),
                "n_observations": int(result.n_observations),
                "selected_candidate_name": str(result.selected_candidate_name),
                "candidate_name": str(comparison.candidate_name),
                "is_selected": bool(comparison.candidate_name == result.selected_candidate_name),
                "log_likelihood": float(comparison.log_likelihood),
                "n_parameters": int(comparison.n_parameters),
                "aic": float(comparison.aic),
                "bic": float(comparison.bic),
                "score": float(comparison.score),
            }
        )
    return rows


def study_model_comparison_records(result: StudyModelComparisonResult) -> list[dict[str, Any]]:
    """Convert study-level model-comparison output into aggregate candidate rows."""

    rows: list[dict[str, Any]] = []
    for comparison in result.comparisons:
        rows.append(
            {
                "criterion": str(result.criterion),
                "n_subjects": int(result.n_subjects),
                "n_observations": int(result.n_observations),
                "selected_candidate_name": str(result.selected_candidate_name),
                "candidate_name": str(comparison.candidate_name),
                "is_selected": bool(comparison.candidate_name == result.selected_candidate_name),
                "log_likelihood": float(comparison.log_likelihood),
                "n_parameters": int(comparison.n_parameters),
                "aic": float(comparison.aic),
                "bic": float(comparison.bic),
                "score": float(comparison.score),
            }
        )
    return rows


def study_model_comparison_subject_records(result: StudyModelComparisonResult) -> list[dict[str, Any]]:
    """Flatten per-subject candidate comparison rows from a study result."""

    rows: list[dict[str, Any]] = []
    for subject_result in result.subject_results:
        rows.extend(subject_model_comparison_records(subject_result))
    return rows


def _subject_fit_mode_and_input_n_blocks(subject_result: Any) -> tuple[str, int]:
    """Resolve fit-mode metadata for subject-level fit outputs."""

    fit_mode_raw = getattr(subject_result, "fit_mode", None)
    if fit_mode_raw is not None:
        fit_mode = str(fit_mode_raw)
    else:
        block_results = tuple(getattr(subject_result, "block_results", ()))
        is_joint = len(block_results) == 1 and getattr(block_results[0], "block_id", None) == "__joint__"
        fit_mode = "joint" if is_joint else "independent"

    input_n_blocks_raw = getattr(subject_result, "input_n_blocks", None)
    if input_n_blocks_raw is not None:
        input_n_blocks = int(input_n_blocks_raw)
    else:
        input_n_blocks = int(len(getattr(subject_result, "block_results", ())))

    return fit_mode, input_n_blocks


def write_model_comparison_csv(result: ModelComparisonResult, path: str | Path) -> Path:
    """Write model-comparison rows to CSV."""

    return write_records_csv(model_comparison_records(result), path)


def write_subject_model_comparison_csv(result: SubjectModelComparisonResult, path: str | Path) -> Path:
    """Write subject-level model-comparison rows to CSV."""

    return write_records_csv(subject_model_comparison_records(result), path)


def write_study_model_comparison_csv(result: StudyModelComparisonResult, path: str | Path) -> Path:
    """Write study-level aggregate model-comparison rows to CSV."""

    return write_records_csv(study_model_comparison_records(result), path)


def write_study_model_comparison_subject_csv(result: StudyModelComparisonResult, path: str | Path) -> Path:
    """Write study per-subject model-comparison rows to CSV."""

    return write_records_csv(study_model_comparison_subject_records(result), path)


__all__ = [
    "block_fit_records",
    "model_comparison_records",
    "study_model_comparison_records",
    "study_model_comparison_subject_records",
    "study_fit_records",
    "study_summary_records",
    "subject_fit_records",
    "subject_model_comparison_records",
    "subject_summary_records",
    "write_model_comparison_csv",
    "write_records_csv",
    "write_study_fit_records_csv",
    "write_study_fit_summary_csv",
    "write_study_model_comparison_csv",
    "write_study_model_comparison_subject_csv",
    "write_subject_model_comparison_csv",
]
