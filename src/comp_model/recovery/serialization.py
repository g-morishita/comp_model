"""Serialization helpers for recovery outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .model import ModelRecoveryResult
from .parameter import ParameterRecoveryResult


def parameter_recovery_records(result: ParameterRecoveryResult) -> list[dict[str, Any]]:
    """Convert parameter-recovery result into flat row dictionaries.

    Parameters
    ----------
    result : ParameterRecoveryResult
        Recovery result object.

    Returns
    -------
    list[dict[str, Any]]
        Flat records containing case metadata and parameter columns.
    """

    rows: list[dict[str, Any]] = []
    for case in result.cases:
        row: dict[str, Any] = {
            "case_index": int(case.case_index),
            "simulation_seed": int(case.simulation_seed),
            "best_log_likelihood": float(case.best_log_likelihood),
            "best_log_posterior": (
                float(case.best_log_posterior)
                if case.best_log_posterior is not None
                else None
            ),
        }

        all_keys = sorted(set(case.true_params) | set(case.estimated_params))
        for key in all_keys:
            true_value = case.true_params.get(key)
            estimated_value = case.estimated_params.get(key)
            row[f"true__{key}"] = float(true_value) if true_value is not None else None
            row[f"estimated__{key}"] = float(estimated_value) if estimated_value is not None else None
            if true_value is not None and estimated_value is not None:
                row[f"error__{key}"] = float(estimated_value) - float(true_value)
            else:
                row[f"error__{key}"] = None

        rows.append(row)

    return rows


def model_recovery_case_records(result: ModelRecoveryResult) -> list[dict[str, Any]]:
    """Convert model-recovery case summaries into flat row dictionaries."""

    rows: list[dict[str, Any]] = []
    for case in result.cases:
        for summary in case.candidate_summaries:
            row: dict[str, Any] = {
                "case_index": int(case.case_index),
                "simulation_seed": int(case.simulation_seed),
                "generating_model_name": str(case.generating_model_name),
                "selected_candidate_name": str(case.selected_candidate_name),
                "candidate_name": str(summary.candidate_name),
                "log_likelihood": float(summary.log_likelihood),
                "log_posterior": (
                    float(summary.log_posterior)
                    if summary.log_posterior is not None
                    else None
                ),
                "n_parameters": int(summary.n_parameters),
                "aic": float(summary.aic),
                "bic": float(summary.bic),
                "waic": (
                    float(summary.waic)
                    if summary.waic is not None
                    else None
                ),
                "psis_loo": (
                    float(summary.psis_loo)
                    if summary.psis_loo is not None
                    else None
                ),
                "score": float(summary.score),
            }
            for key, value in sorted(summary.best_params.items()):
                row[f"param__{key}"] = float(value)
            rows.append(row)

    return rows


def model_recovery_confusion_records(result: ModelRecoveryResult) -> list[dict[str, Any]]:
    """Convert model-recovery confusion matrix into row dictionaries."""

    rows: list[dict[str, Any]] = []
    for generating_name, selected_counts in sorted(result.confusion_matrix.items()):
        for selected_name, count in sorted(selected_counts.items()):
            rows.append(
                {
                    "generating_model_name": str(generating_name),
                    "selected_candidate_name": str(selected_name),
                    "count": int(count),
                    "criterion": str(result.criterion),
                }
            )
    return rows


def write_records_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write generic row dictionaries to CSV.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Row dictionaries to write.
    path : str | pathlib.Path
        Destination CSV path.

    Returns
    -------
    pathlib.Path
        Resolved output path.

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


def write_parameter_recovery_csv(result: ParameterRecoveryResult, path: str | Path) -> Path:
    """Serialize parameter-recovery cases as CSV."""

    return write_records_csv(parameter_recovery_records(result), path)


def write_model_recovery_cases_csv(result: ModelRecoveryResult, path: str | Path) -> Path:
    """Serialize model-recovery candidate case summaries as CSV."""

    return write_records_csv(model_recovery_case_records(result), path)


def write_model_recovery_confusion_csv(result: ModelRecoveryResult, path: str | Path) -> Path:
    """Serialize model-recovery confusion matrix as CSV."""

    return write_records_csv(model_recovery_confusion_records(result), path)


__all__ = [
    "model_recovery_case_records",
    "model_recovery_confusion_records",
    "parameter_recovery_records",
    "write_model_recovery_cases_csv",
    "write_model_recovery_confusion_csv",
    "write_parameter_recovery_csv",
    "write_records_csv",
]
