"""Tabular CSV I/O helpers for canonical decision datasets.

These helpers provide a lightweight boundary between external tabular files and
the in-memory core data model (:class:`TrialDecision`, :class:`BlockData`,
:class:`SubjectData`, :class:`StudyData`).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from comp_model.core.data import (
    BlockData,
    StudyData,
    SubjectData,
    TrialDecision,
    get_block_trace,
    trial_decisions_from_trace,
)

_TRIAL_COLUMNS = (
    "trial_index",
    "decision_index",
    "actor_id",
    "learner_ids_json",
    "decision_node_id",
    "available_actions_json",
    "action_json",
    "observation_json",
    "outcome_json",
    "reward",
    "metadata_json",
)

_STUDY_COLUMNS = (
    "subject_id",
    "block_id_json",
    *_TRIAL_COLUMNS,
)

_MAPPED_STUDY_FIELDS = (
    "subject_id",
    "block_id",
    "trial_index",
    "action",
    "reward",
)


def write_trial_decisions_csv(
    decisions: Sequence[TrialDecision] | Iterable[TrialDecision],
    path: str | Path,
) -> Path:
    """Write trial decisions to a CSV file.

    Parameters
    ----------
    decisions : Sequence[TrialDecision] | Iterable[TrialDecision]
        Trial decision rows to write.
    path : str | pathlib.Path
        Destination CSV path.

    Returns
    -------
    pathlib.Path
        Output CSV path.

    Raises
    ------
    ValueError
        If no decisions are provided.
    """

    rows = list(decisions)
    if not rows:
        raise ValueError("decisions must not be empty")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_TRIAL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(_trial_row_to_csv_mapping(row))
    return output_path


def read_trial_decisions_csv(path: str | Path) -> tuple[TrialDecision, ...]:
    """Read trial decisions from CSV.

    Parameters
    ----------
    path : str | pathlib.Path
        Input CSV path.

    Returns
    -------
    tuple[TrialDecision, ...]
        Parsed trial decision rows in file order.
    """

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, required=_TRIAL_COLUMNS)
        rows = tuple(
            _trial_row_from_csv_mapping(raw, row_index=index)
            for index, raw in enumerate(reader)
        )
    return rows


def write_study_decisions_csv(study: StudyData, path: str | Path) -> Path:
    """Write flattened study decisions to CSV.

    Parameters
    ----------
    study : StudyData
        Study dataset to serialize.
    path : str | pathlib.Path
        Destination CSV path.

    Returns
    -------
    pathlib.Path
        Output CSV path.
    """

    rows = study_decision_rows(study)
    if not rows:
        raise ValueError("study must include at least one trial decision row")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_STUDY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def read_study_decisions_csv(path: str | Path) -> StudyData:
    """Read a flattened study CSV into :class:`StudyData`.

    Parameters
    ----------
    path : str | pathlib.Path
        Input CSV path written by :func:`write_study_decisions_csv`.

    Returns
    -------
    StudyData
        Parsed study dataset with trial rows attached per block.
    """

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, required=_STUDY_COLUMNS)

        grouped: dict[str, dict[Any, list[TrialDecision]]] = defaultdict(lambda: defaultdict(list))
        ordered_block_ids: dict[str, list[Any]] = defaultdict(list)
        ordered_subject_ids: list[str] = []

        for index, raw in enumerate(reader):
            subject_id = _coerce_non_empty_str(raw.get("subject_id"), field_name="subject_id", row_index=index)
            if subject_id not in grouped:
                ordered_subject_ids.append(subject_id)

            block_id_raw = raw.get("block_id_json", "")
            block_id = _json_or_none(block_id_raw)
            if block_id not in grouped[subject_id]:
                ordered_block_ids[subject_id].append(block_id)

            trial_raw = {
                key: value
                for key, value in raw.items()
                if key in _TRIAL_COLUMNS
            }
            grouped[subject_id][block_id].append(
                _trial_row_from_csv_mapping(trial_raw, row_index=index)
            )

    subjects: list[SubjectData] = []
    for subject_id in ordered_subject_ids:
        blocks = tuple(
            BlockData(
                block_id=block_id,
                trials=tuple(grouped[subject_id][block_id]),
            )
            for block_id in ordered_block_ids[subject_id]
        )
        subjects.append(SubjectData(subject_id=subject_id, blocks=blocks))
    return StudyData(subjects=tuple(subjects))


def read_mapped_study_csv(
    path: str | Path,
    *,
    column_mapping: Mapping[str, str],
    available_actions: Sequence[Any],
    actor_id: str = "subject",
    learner_id: str | None = None,
) -> StudyData:
    """Read a raw study CSV with custom column names into :class:`StudyData`.

    Parameters
    ----------
    path : str | pathlib.Path
        Raw input CSV path.
    column_mapping : Mapping[str, str]
        Mapping from canonical field names to raw source columns. Required keys
        are ``subject_id``, ``block_id``, ``trial_index``, ``action``, and
        ``reward``.
    available_actions : Sequence[Any]
        Legal action set for each trial decision.
    actor_id : str, optional
        Actor ID assigned to converted decisions.
    learner_id : str | None, optional
        Single learner ID assigned to converted decisions. Defaults to
        ``actor_id``.

    Returns
    -------
    StudyData
        Parsed study dataset.
    """

    mapping = _validated_mapped_study_column_mapping(column_mapping)

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, required=tuple(mapping.values()))
        rows = tuple(dict(raw) for raw in reader)

    return _study_from_mapped_rows(
        rows,
        column_mapping=mapping,
        available_actions=available_actions,
        actor_id=actor_id,
        learner_id=learner_id,
        row_index_offset=2,
    )


def study_from_mapped_rows(
    rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    column_mapping: Mapping[str, str],
    available_actions: Sequence[Any],
    actor_id: str = "subject",
    learner_id: str | None = None,
) -> StudyData:
    """Convert row mappings with custom source columns into :class:`StudyData`.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]
        Raw row mappings.
    column_mapping : Mapping[str, str]
        Mapping from canonical field names to row keys. Required keys are
        ``subject_id``, ``block_id``, ``trial_index``, ``action``, and
        ``reward``.
    available_actions : Sequence[Any]
        Legal action set for each converted trial decision.
    actor_id : str, optional
        Actor ID assigned to converted decisions.
    learner_id : str | None, optional
        Single learner ID assigned to converted decisions. Defaults to
        ``actor_id``.

    Returns
    -------
    StudyData
        Converted study dataset.
    """

    mapping = _validated_mapped_study_column_mapping(column_mapping)
    return _study_from_mapped_rows(
        rows,
        column_mapping=mapping,
        available_actions=available_actions,
        actor_id=actor_id,
        learner_id=learner_id,
        row_index_offset=1,
    )


def _study_from_mapped_rows(
    rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    column_mapping: Mapping[str, str],
    available_actions: Sequence[Any],
    actor_id: str = "subject",
    learner_id: str | None = None,
    row_index_offset: int = 1,
) -> StudyData:
    """Internal mapped-row converter with configurable row-index offset."""

    mapping = dict(column_mapping)
    allowed_actions = tuple(available_actions)
    if not allowed_actions:
        raise ValueError("available_actions must not be empty")

    actor = str(actor_id).strip()
    if not actor:
        raise ValueError("actor_id must be a non-empty string")
    learner = learner_id if learner_id is not None else actor

    grouped: dict[str, dict[str, list[tuple[int, Any, float]]]] = defaultdict(lambda: defaultdict(list))
    ordered_subject_ids: list[str] = []
    ordered_block_ids: dict[str, list[str]] = defaultdict(list)

    raw_rows = list(rows)
    if not raw_rows:
        raise ValueError("rows must not be empty")

    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(
                f"row {index + row_index_offset}: each row must be a mapping/object"
            )
        row = dict(raw_row)
        row_index = index + row_index_offset

        subject_id = _coerce_non_empty_str(
            row.get(mapping["subject_id"]),
            field_name=mapping["subject_id"],
            row_index=row_index,
        )
        block_id = _coerce_non_empty_str(
            row.get(mapping["block_id"]),
            field_name=mapping["block_id"],
            row_index=row_index,
        )
        trial_number = _coerce_int(
            row.get(mapping["trial_index"]),
            field_name=mapping["trial_index"],
            row_index=row_index,
        )
        action = _coerce_mapped_action(
            row.get(mapping["action"]),
            field_name=mapping["action"],
            row_index=row_index,
        )
        reward = _coerce_float(
            row.get(mapping["reward"]),
            field_name=mapping["reward"],
            row_index=row_index,
        )

        if action not in allowed_actions:
            raise ValueError(
                f"row {row_index}: action={action!r} is outside available_actions={allowed_actions!r}"
            )

        if subject_id not in grouped:
            ordered_subject_ids.append(subject_id)
        if block_id not in grouped[subject_id]:
            ordered_block_ids[subject_id].append(block_id)
        grouped[subject_id][block_id].append((trial_number, action, reward))

    subjects: list[SubjectData] = []
    for subject_id in ordered_subject_ids:
        blocks: list[BlockData] = []
        for block_id in ordered_block_ids[subject_id]:
            triples = sorted(grouped[subject_id][block_id], key=lambda item: item[0])

            seen_trial_numbers: set[int] = set()
            trials: list[TrialDecision] = []
            for normalized_index, (raw_trial_index, action, reward) in enumerate(triples):
                if raw_trial_index in seen_trial_numbers:
                    raise ValueError(
                        "raw trial indices must be unique within each subject/block; "
                        f"found duplicate {raw_trial_index} for subject_id={subject_id!r}, block_id={block_id!r}"
                    )
                seen_trial_numbers.add(raw_trial_index)

                trials.append(
                    TrialDecision(
                        trial_index=normalized_index,
                        decision_index=0,
                        actor_id=actor,
                        learner_ids=(learner,),
                        available_actions=allowed_actions,
                        action=action,
                        observation={"raw_trial_index": raw_trial_index},
                        outcome={"reward": reward},
                        reward=reward,
                    )
                )

            blocks.append(BlockData(block_id=block_id, trials=tuple(trials)))

        subjects.append(SubjectData(subject_id=subject_id, blocks=tuple(blocks)))

    return StudyData(subjects=tuple(subjects))


def study_decision_rows(study: StudyData) -> list[dict[str, Any]]:
    """Flatten :class:`StudyData` into CSV-ready row mappings.

    Parameters
    ----------
    study : StudyData
        Study dataset.

    Returns
    -------
    list[dict[str, Any]]
        Flattened row mappings including subject and block identifiers.
    """

    rows: list[dict[str, Any]] = []
    for subject in study.subjects:
        for block in subject.blocks:
            decisions = block.trials
            if not decisions:
                trace = get_block_trace(block)
                decisions = trial_decisions_from_trace(trace)
            for decision in decisions:
                row = {
                    "subject_id": subject.subject_id,
                    "block_id_json": _json_or_empty(block.block_id),
                    **_trial_row_to_csv_mapping(decision),
                }
                rows.append(row)
    return rows


def _trial_row_to_csv_mapping(row: TrialDecision) -> dict[str, Any]:
    """Convert one :class:`TrialDecision` into a CSV row mapping."""

    return {
        "trial_index": int(row.trial_index),
        "decision_index": int(row.decision_index),
        "actor_id": str(row.actor_id),
        "learner_ids_json": _json_or_empty(row.learner_ids),
        "decision_node_id": (
            "" if row.decision_node_id is None else str(row.decision_node_id)
        ),
        "available_actions_json": _json_or_empty(row.available_actions),
        "action_json": _json_or_empty(row.action),
        "observation_json": _json_or_empty(row.observation),
        "outcome_json": _json_or_empty(row.outcome),
        "reward": "" if row.reward is None else float(row.reward),
        "metadata_json": _json_or_empty(dict(row.metadata)),
    }


def _trial_row_from_csv_mapping(raw: dict[str, Any], *, row_index: int) -> TrialDecision:
    """Parse one CSV row into :class:`TrialDecision`."""

    trial_index = _coerce_int(raw.get("trial_index"), field_name="trial_index", row_index=row_index)
    decision_index = _coerce_int(
        raw.get("decision_index", 0),
        field_name="decision_index",
        row_index=row_index,
    )
    actor_id = _coerce_non_empty_str(raw.get("actor_id"), field_name="actor_id", row_index=row_index)
    learner_ids_value = _json_or_none(raw.get("learner_ids_json", ""))
    learner_ids = (
        tuple(str(value) for value in learner_ids_value)
        if learner_ids_value is not None
        else None
    )
    decision_node_id_raw = raw.get("decision_node_id", "")
    decision_node_id = (
        str(decision_node_id_raw) if str(decision_node_id_raw).strip() else None
    )

    available_actions_value = _json_or_none(raw.get("available_actions_json", ""))
    available_actions = (
        tuple(available_actions_value)
        if available_actions_value is not None
        else None
    )

    action = _json_or_none(raw.get("action_json", ""))
    observation = _json_or_none(raw.get("observation_json", ""))
    outcome = _json_or_none(raw.get("outcome_json", ""))
    reward_raw = raw.get("reward", "")
    reward = float(reward_raw) if str(reward_raw).strip() else None
    metadata_raw = _json_or_none(raw.get("metadata_json", ""))
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

    return TrialDecision(
        trial_index=trial_index,
        decision_index=decision_index,
        actor_id=actor_id,
        learner_ids=learner_ids,
        decision_node_id=decision_node_id,
        available_actions=available_actions,
        action=action,
        observation=observation,
        outcome=outcome,
        reward=reward,
        metadata=metadata,
    )


def _json_or_empty(value: Any) -> str:
    """Serialize a value to JSON string, with empty string for ``None``."""

    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _json_or_none(raw: Any) -> Any:
    """Deserialize a JSON string, returning ``None`` for empty strings."""

    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return json.loads(text)


def _coerce_int(raw: Any, *, field_name: str, row_index: int) -> int:
    """Coerce one integer field with row-index context."""

    text = str(raw).strip() if raw is not None else ""
    if not text:
        raise ValueError(f"row {row_index}: {field_name} must be an integer")
    return int(text)


def _coerce_non_empty_str(raw: Any, *, field_name: str, row_index: int) -> str:
    """Coerce one non-empty string field with row-index context."""

    text = str(raw).strip() if raw is not None else ""
    if not text:
        raise ValueError(f"row {row_index}: {field_name} must be a non-empty string")
    return text


def _coerce_float(raw: Any, *, field_name: str, row_index: int) -> float:
    """Coerce one float field with row-index context."""

    text = str(raw).strip() if raw is not None else ""
    if not text:
        raise ValueError(f"row {row_index}: {field_name} must be a float")
    return float(text)


def _coerce_mapped_action(raw: Any, *, field_name: str, row_index: int) -> Any:
    """Coerce one mapped action value from raw rows."""

    text = str(raw).strip() if raw is not None else ""
    if not text:
        raise ValueError(f"row {row_index}: {field_name} must be non-empty")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    return text


def _validated_mapped_study_column_mapping(
    raw: Mapping[str, str],
) -> dict[str, str]:
    """Validate and normalize mapped-study column mapping."""

    if not isinstance(raw, Mapping):
        raise ValueError("column_mapping must be an object/mapping")

    mapping = dict(raw)
    missing = sorted(name for name in _MAPPED_STUDY_FIELDS if name not in mapping)
    if missing:
        raise ValueError(
            "column_mapping is missing required keys: "
            f"{missing}; expected {_MAPPED_STUDY_FIELDS}"
        )

    unknown = sorted(key for key in mapping if key not in _MAPPED_STUDY_FIELDS)
    if unknown:
        raise ValueError(
            f"column_mapping has unknown keys: {unknown}; expected {_MAPPED_STUDY_FIELDS}"
        )

    out: dict[str, str] = {}
    for key in _MAPPED_STUDY_FIELDS:
        column_name = str(mapping[key]).strip()
        if not column_name:
            raise ValueError(f"column_mapping[{key!r}] must be a non-empty string")
        out[key] = column_name
    return out


def _require_columns(fieldnames: Sequence[str] | None, *, required: tuple[str, ...]) -> None:
    """Require all expected columns to exist in CSV header."""

    if fieldnames is None:
        raise ValueError("CSV file must include a header row")
    missing = [name for name in required if name not in set(fieldnames)]
    if missing:
        raise ValueError(f"CSV file missing required columns: {missing}")


__all__ = [
    "read_mapped_study_csv",
    "read_study_decisions_csv",
    "read_trial_decisions_csv",
    "study_from_mapped_rows",
    "study_decision_rows",
    "write_study_decisions_csv",
    "write_trial_decisions_csv",
]
