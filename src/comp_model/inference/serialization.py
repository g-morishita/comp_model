"""Serialization helpers for fitting outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .hierarchical import HierarchicalStudyMapResult, HierarchicalSubjectMapResult
from .map_study_fitting import MapBlockFitResult, MapStudyFitResult, MapSubjectFitResult
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


def hierarchical_subject_block_records(subject_result: HierarchicalSubjectMapResult) -> list[dict[str, Any]]:
    """Convert one hierarchical subject result to per-block rows.

    Parameters
    ----------
    subject_result : HierarchicalSubjectMapResult
        Subject-level hierarchical MAP output.

    Returns
    -------
    list[dict[str, Any]]
        Per-block flattened rows including parameter values.
    """

    rows: list[dict[str, Any]] = []
    for block in subject_result.block_results:
        row: dict[str, Any] = {
            "subject_id": subject_result.subject_id,
            "block_id": block.block_id,
            "log_likelihood": float(block.log_likelihood),
        }
        for key, value in sorted(block.params.items()):
            row[f"param__{key}"] = float(value)
        rows.append(row)
    return rows


def hierarchical_study_block_records(study_result: HierarchicalStudyMapResult) -> list[dict[str, Any]]:
    """Convert hierarchical study result to per-block rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(hierarchical_subject_block_records(subject))
    return rows


def hierarchical_subject_summary_records(subject_result: HierarchicalSubjectMapResult) -> list[dict[str, Any]]:
    """Convert one hierarchical subject result to summary rows.

    Parameters
    ----------
    subject_result : HierarchicalSubjectMapResult
        Subject-level hierarchical MAP output.

    Returns
    -------
    list[dict[str, Any]]
        One-row subject summary.
    """

    row: dict[str, Any] = {
        "subject_id": subject_result.subject_id,
        "n_blocks": len(subject_result.block_results),
        "total_log_likelihood": float(subject_result.total_log_likelihood),
        "total_log_prior": float(subject_result.total_log_prior),
        "total_log_posterior": float(subject_result.total_log_posterior),
    }
    for key, value in sorted(subject_result.group_location_z.items()):
        row[f"group_location_z__{key}"] = float(value)
    for key, value in sorted(subject_result.group_scale_z.items()):
        row[f"group_scale__{key}"] = float(value)
    return [row]


def hierarchical_study_summary_records(study_result: HierarchicalStudyMapResult) -> list[dict[str, Any]]:
    """Convert hierarchical study result to subject-level summary rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(hierarchical_subject_summary_records(subject))
    return rows


def map_block_fit_records(block_result: MapBlockFitResult, *, subject_id: str | None = None) -> list[dict[str, Any]]:
    """Convert one MAP block-fit result to flat row records."""

    candidate = block_result.fit_result.map_candidate
    row: dict[str, Any] = {
        "subject_id": subject_id,
        "block_id": block_result.block_id,
        "n_trials": int(block_result.n_trials),
        "log_likelihood": float(candidate.log_likelihood),
        "log_prior": float(candidate.log_prior),
        "log_posterior": float(candidate.log_posterior),
    }
    for key, value in sorted(candidate.params.items()):
        row[f"param__{key}"] = float(value)
    return [row]


def map_subject_fit_records(subject_result: MapSubjectFitResult) -> list[dict[str, Any]]:
    """Convert one MAP subject-fit result to flat block rows."""

    rows: list[dict[str, Any]] = []
    for block in subject_result.block_results:
        rows.extend(map_block_fit_records(block, subject_id=subject_result.subject_id))
    return rows


def map_study_fit_records(study_result: MapStudyFitResult) -> list[dict[str, Any]]:
    """Convert MAP study fit result to flat block rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(map_subject_fit_records(subject))
    return rows


def map_subject_summary_records(subject_result: MapSubjectFitResult) -> list[dict[str, Any]]:
    """Convert one MAP subject fit result to summary row."""

    row: dict[str, Any] = {
        "subject_id": subject_result.subject_id,
        "n_blocks": len(subject_result.block_results),
        "total_log_likelihood": float(subject_result.total_log_likelihood),
        "total_log_posterior": float(subject_result.total_log_posterior),
    }
    for key, value in sorted(subject_result.mean_map_params.items()):
        row[f"mean_map_param__{key}"] = float(value)
    return [row]


def map_study_summary_records(study_result: MapStudyFitResult) -> list[dict[str, Any]]:
    """Convert MAP study fit result to subject-level summary rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(map_subject_summary_records(subject))
    return rows


def write_hierarchical_study_block_records_csv(
    study_result: HierarchicalStudyMapResult,
    path: str | Path,
) -> Path:
    """Write hierarchical study block rows to CSV."""

    return write_records_csv(hierarchical_study_block_records(study_result), path)


def write_hierarchical_study_summary_csv(
    study_result: HierarchicalStudyMapResult,
    path: str | Path,
) -> Path:
    """Write hierarchical study summary rows to CSV."""

    return write_records_csv(hierarchical_study_summary_records(study_result), path)


def write_map_study_fit_records_csv(study_result: MapStudyFitResult, path: str | Path) -> Path:
    """Write MAP study block-level fit rows to CSV."""

    return write_records_csv(map_study_fit_records(study_result), path)


def write_map_study_fit_summary_csv(study_result: MapStudyFitResult, path: str | Path) -> Path:
    """Write MAP study subject-level summaries to CSV."""

    return write_records_csv(map_study_summary_records(study_result), path)


__all__ = [
    "block_fit_records",
    "hierarchical_study_block_records",
    "hierarchical_study_summary_records",
    "hierarchical_subject_block_records",
    "hierarchical_subject_summary_records",
    "map_block_fit_records",
    "map_study_fit_records",
    "map_study_summary_records",
    "map_subject_fit_records",
    "map_subject_summary_records",
    "study_fit_records",
    "study_summary_records",
    "subject_fit_records",
    "subject_summary_records",
    "write_hierarchical_study_block_records_csv",
    "write_hierarchical_study_summary_csv",
    "write_map_study_fit_records_csv",
    "write_map_study_fit_summary_csv",
    "write_records_csv",
    "write_study_fit_records_csv",
    "write_study_fit_summary_csv",
]
