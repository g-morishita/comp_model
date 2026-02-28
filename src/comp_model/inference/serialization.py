"""Serialization helpers for fitting outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .study_fitting import BlockFitResult, StudyFitResult, SubjectFitResult


def block_fit_records(block_result: BlockFitResult, *, subject_id: str | None = None) -> list[dict[str, Any]]:
    """Convert one block-fit result to flat row records.

    Parameters
    ----------
    block_result : BlockFitResult
        Block-level fit output.
    subject_id : str | None, optional
        Optional subject label attached to each row.

    Returns
    -------
    list[dict[str, Any]]
        Candidate-level fit rows for the block.
    """

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

    row: dict[str, Any] = {
        "subject_id": subject_result.subject_id,
        "n_blocks": len(subject_result.block_results),
        "total_log_likelihood": float(subject_result.total_log_likelihood),
    }
    for key, value in sorted(subject_result.mean_best_params.items()):
        row[f"mean_best_param__{key}"] = float(value)
    return [row]


def study_summary_records(study_result: StudyFitResult) -> list[dict[str, Any]]:
    """Convert study fit result to per-subject summary rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(subject_summary_records(subject))
    return rows


def write_records_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write generic rows to CSV.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Rows to write.
    path : str | pathlib.Path
        Destination path.

    Returns
    -------
    pathlib.Path
        Output path.

    Raises
    ------
    ValueError
        If ``rows`` is empty.
    """

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


__all__ = [
    "block_fit_records",
    "study_fit_records",
    "study_summary_records",
    "subject_fit_records",
    "subject_summary_records",
    "write_records_csv",
    "write_study_fit_records_csv",
    "write_study_fit_summary_csv",
]
