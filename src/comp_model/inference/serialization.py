"""Serialization helpers for fitting outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .fit_result import extract_best_fit_summary
from .hierarchical import HierarchicalStudyMapResult, HierarchicalSubjectMapResult
from .hierarchical_mcmc import (
    HierarchicalStudyPosteriorResult,
    HierarchicalSubjectPosteriorResult,
)
from .map_study_fitting import MapBlockFitResult, MapStudyFitResult, MapSubjectFitResult
from .model_selection import ModelComparisonResult
from .posterior import PosteriorSummary, posterior_summary_records
from .study_fitting import BlockFitResult, StudyFitResult, SubjectFitResult
from .study_model_selection import (
    StudyModelComparisonResult,
    SubjectModelComparisonResult,
)


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

    fit_mode, input_n_blocks = _subject_fit_mode_and_input_n_blocks(subject_result)
    row: dict[str, Any] = {
        "subject_id": subject_result.subject_id,
        "n_blocks": len(subject_result.block_results),
        "input_n_blocks": int(input_n_blocks),
        "fit_mode": fit_mode,
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


def hierarchical_mcmc_subject_draw_records(
    subject_result: HierarchicalSubjectPosteriorResult,
) -> list[dict[str, Any]]:
    """Convert one hierarchical-MCMC subject result to draw-level rows.

    Parameters
    ----------
    subject_result : HierarchicalSubjectPosteriorResult
        Subject-level hierarchical posterior output.

    Returns
    -------
    list[dict[str, Any]]
        One row per retained draw.
    """

    rows: list[dict[str, Any]] = []
    for draw in subject_result.draws:
        row: dict[str, Any] = {
            "subject_id": subject_result.subject_id,
            "iteration": int(draw.iteration),
            "accepted": bool(draw.accepted),
            "log_likelihood": float(draw.candidate.log_likelihood),
            "log_prior": float(draw.candidate.log_prior),
            "log_posterior": float(draw.candidate.log_posterior),
        }
        for key, value in sorted(draw.candidate.group_location_z.items()):
            row[f"group_location_z__{key}"] = float(value)
        for key, value in sorted(draw.candidate.group_scale_z.items()):
            row[f"group_scale__{key}"] = float(value)
        for block_index, block_params in enumerate(draw.candidate.block_params):
            for key, value in sorted(block_params.items()):
                row[f"block_{block_index}__param__{key}"] = float(value)
        rows.append(row)
    return rows


def hierarchical_mcmc_study_draw_records(
    study_result: HierarchicalStudyPosteriorResult,
) -> list[dict[str, Any]]:
    """Convert hierarchical-MCMC study result to draw-level rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(hierarchical_mcmc_subject_draw_records(subject))
    return rows


def hierarchical_mcmc_subject_summary_records(
    subject_result: HierarchicalSubjectPosteriorResult,
) -> list[dict[str, Any]]:
    """Convert one hierarchical-MCMC subject result to summary row."""

    map_candidate = subject_result.map_candidate
    row: dict[str, Any] = {
        "subject_id": subject_result.subject_id,
        "n_draws": len(subject_result.draws),
        "n_blocks": int(subject_result.n_blocks),
        "acceptance_rate": float(subject_result.diagnostics.acceptance_rate),
        "map_log_likelihood": float(map_candidate.log_likelihood),
        "map_log_prior": float(map_candidate.log_prior),
        "map_log_posterior": float(map_candidate.log_posterior),
    }
    for key, value in sorted(map_candidate.group_location_z.items()):
        row[f"map_group_location_z__{key}"] = float(value)
    for key, value in sorted(map_candidate.group_scale_z.items()):
        row[f"map_group_scale__{key}"] = float(value)
    return [row]


def hierarchical_mcmc_study_summary_records(
    study_result: HierarchicalStudyPosteriorResult,
) -> list[dict[str, Any]]:
    """Convert hierarchical-MCMC study result to subject summary rows."""

    rows: list[dict[str, Any]] = []
    for subject in study_result.subject_results:
        rows.extend(hierarchical_mcmc_subject_summary_records(subject))
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

    fit_mode, input_n_blocks = _subject_fit_mode_and_input_n_blocks(subject_result)
    row: dict[str, Any] = {
        "subject_id": subject_result.subject_id,
        "n_blocks": len(subject_result.block_results),
        "input_n_blocks": int(input_n_blocks),
        "fit_mode": fit_mode,
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


def model_comparison_records(result: ModelComparisonResult) -> list[dict[str, Any]]:
    """Convert model-comparison output to flat row dictionaries.

    Parameters
    ----------
    result : ModelComparisonResult
        Candidate-model comparison output.

    Returns
    -------
    list[dict[str, Any]]
        One row per candidate.
    """

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
            "log_posterior": (
                float(best.log_posterior)
                if best.log_posterior is not None
                else None
            ),
            "n_parameters": int(comparison.n_parameters),
            "aic": float(comparison.aic),
            "bic": float(comparison.bic),
            "waic": (
                float(comparison.waic)
                if comparison.waic is not None
                else None
            ),
            "psis_loo": (
                float(comparison.psis_loo)
                if comparison.psis_loo is not None
                else None
            ),
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
                "log_posterior": (
                    float(comparison.log_posterior)
                    if comparison.log_posterior is not None
                    else None
                ),
                "n_parameters": int(comparison.n_parameters),
                "aic": float(comparison.aic),
                "bic": float(comparison.bic),
                "waic": (
                    float(comparison.waic)
                    if comparison.waic is not None
                    else None
                ),
                "psis_loo": (
                    float(comparison.psis_loo)
                    if comparison.psis_loo is not None
                    else None
                ),
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
                "log_posterior": (
                    float(comparison.log_posterior)
                    if comparison.log_posterior is not None
                    else None
                ),
                "n_parameters": int(comparison.n_parameters),
                "aic": float(comparison.aic),
                "bic": float(comparison.bic),
                "waic": (
                    float(comparison.waic)
                    if comparison.waic is not None
                    else None
                ),
                "psis_loo": (
                    float(comparison.psis_loo)
                    if comparison.psis_loo is not None
                    else None
                ),
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


def write_hierarchical_mcmc_study_draw_records_csv(
    study_result: HierarchicalStudyPosteriorResult,
    path: str | Path,
) -> Path:
    """Write hierarchical-MCMC study draw rows to CSV."""

    return write_records_csv(hierarchical_mcmc_study_draw_records(study_result), path)


def write_hierarchical_mcmc_study_summary_csv(
    study_result: HierarchicalStudyPosteriorResult,
    path: str | Path,
) -> Path:
    """Write hierarchical-MCMC study summary rows to CSV."""

    return write_records_csv(hierarchical_mcmc_study_summary_records(study_result), path)


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


def write_map_study_fit_records_csv(study_result: MapStudyFitResult, path: str | Path) -> Path:
    """Write MAP study block-level fit rows to CSV."""

    return write_records_csv(map_study_fit_records(study_result), path)


def write_map_study_fit_summary_csv(study_result: MapStudyFitResult, path: str | Path) -> Path:
    """Write MAP study subject-level summaries to CSV."""

    return write_records_csv(map_study_summary_records(study_result), path)


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


def write_posterior_summary_csv(summary: PosteriorSummary, path: str | Path) -> Path:
    """Write posterior summary rows to CSV."""

    return write_records_csv(posterior_summary_records(summary), path)


__all__ = [
    "block_fit_records",
    "hierarchical_study_block_records",
    "hierarchical_study_summary_records",
    "hierarchical_mcmc_study_draw_records",
    "hierarchical_mcmc_study_summary_records",
    "hierarchical_mcmc_subject_draw_records",
    "hierarchical_mcmc_subject_summary_records",
    "hierarchical_subject_block_records",
    "hierarchical_subject_summary_records",
    "map_block_fit_records",
    "map_study_fit_records",
    "map_study_summary_records",
    "map_subject_fit_records",
    "map_subject_summary_records",
    "model_comparison_records",
    "study_model_comparison_records",
    "study_model_comparison_subject_records",
    "subject_model_comparison_records",
    "study_fit_records",
    "study_summary_records",
    "subject_fit_records",
    "subject_summary_records",
    "write_hierarchical_study_block_records_csv",
    "write_hierarchical_mcmc_study_draw_records_csv",
    "write_hierarchical_mcmc_study_summary_csv",
    "write_hierarchical_study_summary_csv",
    "write_map_study_fit_records_csv",
    "write_map_study_fit_summary_csv",
    "write_model_comparison_csv",
    "write_study_model_comparison_csv",
    "write_study_model_comparison_subject_csv",
    "write_subject_model_comparison_csv",
    "write_posterior_summary_csv",
    "write_records_csv",
    "write_study_fit_records_csv",
    "write_study_fit_summary_csv",
]
