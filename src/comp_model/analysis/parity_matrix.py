"""Model parity matrix reporting utilities.

This module exports the static source-to-canonical model mapping into a
machine-readable parity matrix that can be consumed by CI pipelines.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import comp_model.models as models_pkg
from comp_model.models import MODEL_PARITY
from comp_model.plugins import PluginRegistry, build_default_registry


@dataclass(frozen=True, slots=True)
class ModelParityMatrixRow:
    """Resolved parity-matrix row for one source model.

    Parameters
    ----------
    source_name : str
        Source model name retained for mapping/reference.
    canonical_component_id : str | None
        Canonical plugin component ID in this repository when applicable.
    canonical_class_name : str | None
        Canonical model class name in :mod:`comp_model.models` when applicable.
    status : str
        Declared parity status from :data:`comp_model.models.MODEL_PARITY`.
    notes : str
        Free-text notes carried from parity mapping declarations.
    class_exists : bool
        Whether ``canonical_class_name`` resolves in :mod:`comp_model.models`.
    component_registered : bool | None
        Whether ``canonical_component_id`` exists in the plugin registry.
        ``None`` indicates component registration is not expected for this row.
    mapping_valid : bool
        Whether this row passes structural parity validation:
        implemented rows must resolve class names, and when component IDs are
        provided they must be registered.
    """

    source_name: str
    canonical_component_id: str | None
    canonical_class_name: str | None
    status: str
    notes: str
    class_exists: bool
    component_registered: bool | None
    mapping_valid: bool


@dataclass(frozen=True, slots=True)
class ModelParityMatrixSummary:
    """Aggregated parity matrix counts.

    Parameters
    ----------
    n_rows : int
        Total number of matrix rows.
    n_implemented : int
        Number of rows with ``status == "implemented"``.
    n_planned : int
        Number of rows with ``status == "planned"``.
    n_invalid : int
        Number of rows with ``mapping_valid == False``.
    """

    n_rows: int
    n_implemented: int
    n_planned: int
    n_invalid: int


def build_model_parity_matrix(
    *,
    registry: PluginRegistry | None = None,
) -> tuple[ModelParityMatrixRow, ...]:
    """Resolve model parity mapping into a machine-readable matrix.

    Parameters
    ----------
    registry : PluginRegistry | None, optional
        Optional plugin registry. Defaults to :func:`build_default_registry`.

    Returns
    -------
    tuple[ModelParityMatrixRow, ...]
        Resolved parity matrix rows in declaration order.
    """

    reg = registry if registry is not None else build_default_registry()
    available_model_ids = {manifest.component_id for manifest in reg.list("model")}

    rows: list[ModelParityMatrixRow] = []
    for entry in MODEL_PARITY:
        class_exists = (
            entry.canonical_class_name is not None
            and hasattr(models_pkg, entry.canonical_class_name)
        )
        component_registered: bool | None = None
        if entry.canonical_component_id is not None:
            component_registered = entry.canonical_component_id in available_model_ids

        mapping_valid = _is_mapping_valid(
            status=entry.status,
            class_exists=class_exists,
            component_registered=component_registered,
        )

        rows.append(
            ModelParityMatrixRow(
                source_name=entry.source_name,
                canonical_component_id=entry.canonical_component_id,
                canonical_class_name=entry.canonical_class_name,
                status=entry.status,
                notes=entry.notes,
                class_exists=bool(class_exists),
                component_registered=component_registered,
                mapping_valid=bool(mapping_valid),
            )
        )

    return tuple(rows)


def summarize_model_parity_matrix(
    rows: tuple[ModelParityMatrixRow, ...] | list[ModelParityMatrixRow],
) -> ModelParityMatrixSummary:
    """Summarize parity matrix row counts.

    Parameters
    ----------
    rows : tuple[ModelParityMatrixRow, ...] | list[ModelParityMatrixRow]
        Parity matrix rows.

    Returns
    -------
    ModelParityMatrixSummary
        Aggregate count summary.
    """

    n_rows = len(rows)
    n_implemented = sum(1 for row in rows if row.status == "implemented")
    n_planned = sum(1 for row in rows if row.status == "planned")
    n_invalid = sum(1 for row in rows if not row.mapping_valid)
    return ModelParityMatrixSummary(
        n_rows=int(n_rows),
        n_implemented=int(n_implemented),
        n_planned=int(n_planned),
        n_invalid=int(n_invalid),
    )


def write_model_parity_matrix_json(
    rows: tuple[ModelParityMatrixRow, ...] | list[ModelParityMatrixRow],
    path: str | Path,
) -> Path:
    """Write parity matrix rows and summary as JSON.

    Parameters
    ----------
    rows : tuple[ModelParityMatrixRow, ...] | list[ModelParityMatrixRow]
        Parity rows to serialize.
    path : str | pathlib.Path
        Output JSON path.

    Returns
    -------
    pathlib.Path
        Output path.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_model_parity_matrix(rows)
    payload: dict[str, Any] = {
        "summary": asdict(summary),
        "rows": [asdict(item) for item in rows],
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def write_model_parity_matrix_csv(
    rows: tuple[ModelParityMatrixRow, ...] | list[ModelParityMatrixRow],
    path: str | Path,
) -> Path:
    """Write parity matrix rows as CSV.

    Parameters
    ----------
    rows : tuple[ModelParityMatrixRow, ...] | list[ModelParityMatrixRow]
        Parity rows to serialize.
    path : str | pathlib.Path
        Output CSV path.

    Returns
    -------
    pathlib.Path
        Output path.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_name",
                "canonical_component_id",
                "canonical_class_name",
                "status",
                "notes",
                "class_exists",
                "component_registered",
                "mapping_valid",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return target


def _is_mapping_valid(
    *,
    status: str,
    class_exists: bool,
    component_registered: bool | None,
) -> bool:
    """Return structural validity for one parity row."""

    if status != "implemented":
        return True
    if not class_exists:
        return False
    if component_registered is None:
        return True
    return bool(component_registered)


__all__ = [
    "ModelParityMatrixRow",
    "ModelParityMatrixSummary",
    "build_model_parity_matrix",
    "summarize_model_parity_matrix",
    "write_model_parity_matrix_csv",
    "write_model_parity_matrix_json",
]
