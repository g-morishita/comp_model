"""comp_model_core.events.convert

Converters for producing :class:`~comp_model_core.events.types.EventLog` objects
from "typical" (non-event-log) datasets.

Why this exists
--------------
Many real datasets are stored as one-row-per-trial tables (CSV, parquet, pandas
DataFrames, JSONL, etc.). In this library, likelihood and exporters consume an
**event log** (a stream of ``BLOCK_START``, ``CHOICE``, ``OUTCOME``, and optional
``SOCIAL_OBSERVED`` events) because the ordering of events is the source of truth
for model updates.

This module bridges the gap by offering:

* :func:`event_log_from_trials` - build an event log from
  :class:`~comp_model_core.data.types.Trial` objects.
* :func:`trials_from_rows` - parse a row-per-trial table into Trial objects.
* :func:`event_log_from_rows` - one-shot conversion from row-per-trial data to an
  event log.
* :func:`attach_event_logs` - attach event logs to all blocks in a
  :class:`~comp_model_core.data.types.StudyData`.

The functions are dependency-minimal (no pandas required). If you *do* have
pandas, you can pass a DataFrame directly and we will use ``to_dict('records')``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

import numpy as np

from comp_model_core.data.types import Trial, Block, SubjectData, StudyData
from comp_model_core.events.types import Event, EventLog, EventType, validate_event_log


# -----------------------------------------------------------------------------
# Column mapping
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialTableColumns:
    """Column-name mapping for row-per-trial ingestion.

    Only ``subject_id``, ``block_id``, and ``t`` are strictly required to build a
    full :class:`~comp_model_core.data.types.StudyData` from rows. For the event
    log alone, you need at least ``t`` and ``choice`` (plus outcomes if available).

    You can override these to match your dataset.
    """

    subject_id: str = "subject_id"
    block_id: str = "block_id"
    condition: str = "condition"
    t: str = "t"

    # Core trial fields
    state: str = "state"
    choice: str = "choice"
    observed_outcome: str = "observed_outcome"
    outcome: str = "outcome"
    available_actions: str = "available_actions"
    info: str = "info"

    # Social fields
    others_choices: str = "others_choices"
    others_outcomes: str = "others_outcomes"
    observed_others_outcomes: str = "observed_others_outcomes"
    social_info: str = "social_info"


# -----------------------------------------------------------------------------
# Multi-form row ingestion (self+social combined, or self/demo separated)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PartnerSelfTrialTableColumns:
    """Column-name mapping for combined self + partner/demo trial tables.

    This is for *wide* datasets where each trial row contains both the participant's
    choice/outcome fields and a demonstrator/partner's choice/outcome fields.

    The converter only has explicit knowledge of a small, general set of fields.
    Any additional dataset-specific columns are automatically preserved as metadata
    in :attr:`Trial.info` (self-side) or :attr:`Trial.social_info` (partner/demo side).
    """

    # identity / grouping
    subject_id: str = "id"
    block_id: str = "block"
    t: str = "trial"
    condition: str = "condition"

    # self (participant)
    self_choice: str = "self_choice"
    self_observed_outcome: str = "self_reward"   # common alternative: "self_observed_outcome"
    self_outcome: str = "self_outcome"           # common alternative: "outcome"
    rt: str = "rt"

    # partner / demonstrator (what subject observes about the other)
    partner_choice: str = "partner_choice"
    partner_observed_outcome: str = "partner_reward"
    partner_outcome: str = "partner_outcome"

    # optional: if the table already encodes multiple demonstrators per trial
    others_choices: str = "others_choices"
    others_outcomes: str = "others_outcomes"
    observed_others_outcomes: str = "observed_others_outcomes"


def _infer_first_present(row: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    for c in candidates:
        if c in row and not _is_null(row.get(c)):
            return row.get(c)
    return None


def trials_from_partner_self_rows(
    rows: Any,
    *,
    columns: PartnerSelfTrialTableColumns | None = None,
    default_state: Any = 0,
    # outcome strategy
    prefer_self_outcome: bool = True,
    warn_on_missing_self_outcome: bool = True,
) -> list[Trial]:
    """Parse a combined self+partner trial table into :class:`~comp_model_core.data.types.Trial` objects.

    This is designed for *wide* CSVs where each trial row contains both the
    participant's choice and a demonstrator/partner's choice (+ reward).

    Parameters
    ----------
    prefer_self_outcome
        If True, we look for self outcome columns first.
    warn_on_missing_self_outcome
        If True, emit warnings when self observed outcome and/or self true outcome
        are missing.
    """
    cols = columns or PartnerSelfTrialTableColumns()
    recs = _as_rows(rows)

    trials: list[Trial] = []
    missing_self_observed_count = 0
    missing_self_true_count = 0
    for r in recs:
        t_raw = r.get(cols.t)
        if _is_null(t_raw):
            raise ValueError(f"Missing trial index column {cols.t!r} in row: {r}")
        t = int(float(t_raw))

        state = r.get("state", default_state)

        # --- self choice
        choice_raw = _infer_first_present(r, [cols.self_choice, "choice", "self_action", "action"])
        choice = None if _is_null(choice_raw) else int(float(choice_raw))

        # --- self outcome (observed + true)
        self_obs_raw = _infer_first_present(
            r,
            [
                cols.self_observed_outcome,
                "self_observed_outcome",
                "self_reward",
                "reward",
                "observed_outcome",
            ],
        )
        self_out_raw = _infer_first_present(
            r,
            [
                cols.self_outcome,
                "outcome",
                "self_true_outcome",
                "self_reward_true",
            ],
        )

        partner_obs_raw = _infer_first_present(
            r,
            [
                cols.partner_observed_outcome,
                "partner_observed_outcome",
                "partner_reward",
                "demo_reward",
                "other_reward",
            ],
        )
        partner_out_raw = _infer_first_present(r, [cols.partner_outcome, "partner_true_outcome"])

        observed_outcome = None
        outcome = None

        if _is_null(self_obs_raw):
            missing_self_observed_count += 1
        if _is_null(self_out_raw):
            missing_self_true_count += 1

        if prefer_self_outcome:
            if not _is_null(self_obs_raw):
                observed_outcome = float(self_obs_raw)
            if not _is_null(self_out_raw):
                outcome = float(self_out_raw)

        # --- social fields (single partner OR already-list columns)
        # Prefer explicit list columns if present.
        others_choices = _maybe_list(r.get(cols.others_choices, None))
        if others_choices is None:
            pc = _infer_first_present(r, [cols.partner_choice, "demo_choice", "other_choice", "partner_action"])
            others_choices = None if _is_null(pc) else [int(float(pc))]
        else:
            others_choices = [int(float(x)) for x in others_choices]

        others_outcomes = _maybe_list(r.get(cols.others_outcomes, None))
        if others_outcomes is None:
            po = partner_out_raw
            if _is_null(po):
                po = None
            others_outcomes = None if _is_null(po) else [float(po)]
        else:
            others_outcomes = [float(x) for x in others_outcomes]

        observed_others = _maybe_list(r.get(cols.observed_others_outcomes, None))
        if observed_others is None:
            pobs = partner_obs_raw
            observed_others = None if _is_null(pobs) else [float(pobs)]
        else:
            observed_others = [float(x) for x in observed_others]

        # --- pack metadata
        info: dict[str, Any] = {}
        social_info: dict[str, Any] = {}

        # A small, general set of commonly useful metadata fields.
        rt = r.get(cols.rt, None)
        if not _is_null(rt):
            info["rt"] = float(rt)

        # Preserve any unclaimed columns in info/social_info for traceability.

        claimed = {
            cols.subject_id,
            cols.block_id,
            cols.t,
            cols.condition,
            cols.self_choice,
            cols.self_observed_outcome,
            cols.self_outcome,
            cols.rt,
            cols.partner_choice,
            cols.partner_observed_outcome,
            cols.partner_outcome,
            cols.others_choices,
            cols.others_outcomes,
            cols.observed_others_outcomes,
            "state",
            "choice",
            "observed_outcome",
            "outcome",
            "reward",
        }
        for k, v in r.items():
            if k in claimed:
                continue
            # heuristic: columns starting with 'partner_' go to social_info, others to info
            if isinstance(k, str) and (k.startswith(("partner_", "demo_", "other_", "obs_")) or ("partner" in k and not k.startswith("self_"))):
                if not _is_null(v):
                    social_info[k] = _maybe_value(v)
            else:
                if not _is_null(v):
                    info[k] = _maybe_value(v)

        trials.append(
            Trial(
                t=t,
                state=state,
                choice=choice,
                observed_outcome=observed_outcome,
                outcome=outcome,
                available_actions=None,
                info=info,
                others_choices=others_choices,
                others_outcomes=others_outcomes,
                observed_others_outcomes=observed_others,
                social_info=social_info,
            )
        )

    if warn_on_missing_self_outcome and recs:
        n = len(recs)
        if missing_self_observed_count > 0:
            warnings.warn(
                f"Self observed outcome is missing for {missing_self_observed_count}/{n} trials; "
                "observed_outcome is left as None.",
                UserWarning,
                stacklevel=2,
            )
        if missing_self_true_count > 0:
            warnings.warn(
                f"Self true outcome is missing for {missing_self_true_count}/{n} trials; "
                "outcome is left as None.",
                UserWarning,
                stacklevel=2,
            )

    trials.sort(key=lambda tr: tr.t)
    return trials


def event_log_from_partner_self_rows(
    rows: Any,
    *,
    block_id: str,
    condition: str,
    timing: str = "pre_choice",
    columns: PartnerSelfTrialTableColumns | None = None,
    default_state: Any = 0,
    prefer_self_outcome: bool = True,
    warn_on_missing_self_outcome: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> EventLog:
    """One-shot conversion for combined self+partner tables."""
    trials = trials_from_partner_self_rows(
        rows,
        columns=columns,
        default_state=default_state,
        prefer_self_outcome=prefer_self_outcome,
        warn_on_missing_self_outcome=warn_on_missing_self_outcome,
    )
    return event_log_from_trials(
        block_id=block_id,
        condition=condition,
        trials=trials,
        timing=timing,
        metadata=metadata,
    )


def merge_self_and_demo_rows(
    *,
    self_rows: Any,
    demo_rows: Any,
    keys: Sequence[str] = ("id", "block", "trial"),
    # demo column names (common defaults)
    demo_choice_col: str = "partner_choice",
    demo_reward_col: str = "partner_reward",
    demo_id_col: str | None = None,
) -> list[dict[str, Any]]:
    """Merge 'self' and 'demo' tables into one row-per-trial table.

    This supports the common real-data pattern where:
      - self table: one row per trial with participant choice/outcome
      - demo table: one (or many) rows per trial with demonstrator choice/outcome

    The output rows include ``others_choices`` and ``observed_others_outcomes`` as lists
    (compatible with :func:`trials_from_rows` or :func:`trials_from_partner_self_rows`).

    If multiple demo rows map to the same self trial, they are aggregated into lists.
    """
    s = _as_rows(self_rows)
    d = _as_rows(demo_rows)

    def k_of(r: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(r.get(k) for k in keys)

    # Index demo rows by key
    demo_map: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for r in d:
        demo_map.setdefault(k_of(r), []).append(r)

    merged: list[dict[str, Any]] = []
    for r in s:
        out = dict(r)
        k = k_of(r)
        demos = demo_map.get(k, [])
        if demos:
            out["others_choices"] = [
                int(float(dr.get(demo_choice_col))) for dr in demos if not _is_null(dr.get(demo_choice_col))
            ]
            out["observed_others_outcomes"] = [
                float(dr.get(demo_reward_col)) for dr in demos if not _is_null(dr.get(demo_reward_col))
            ]
            if demo_id_col is not None:
                out.setdefault("social_info", {})
                out["social_info"] = dict(out["social_info"]) if isinstance(out["social_info"], Mapping) else {}
                out["social_info"]["demo_ids"] = [
                    dr.get(demo_id_col) for dr in demos if not _is_null(dr.get(demo_id_col))
                ]
        merged.append(out)

    return merged


def event_log_from_separate_self_demo_rows(
    *,
    self_rows: Any,
    demo_rows: Any,
    block_id: str,
    condition: str,
    timing: str = "pre_choice",
    keys: Sequence[str] = ("id", "block", "trial"),
    demo_choice_col: str = "partner_choice",
    demo_reward_col: str = "partner_reward",
    demo_id_col: str | None = None,
    # mapping for the self table if it uses non-canonical names
    self_columns: TrialTableColumns | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EventLog:
    """One-shot conversion for datasets where self and demo are stored separately."""
    merged = merge_self_and_demo_rows(
        self_rows=self_rows,
        demo_rows=demo_rows,
        keys=keys,
        demo_choice_col=demo_choice_col,
        demo_reward_col=demo_reward_col,
        demo_id_col=demo_id_col,
    )
    # If the self table uses canonical column names, trials_from_rows is enough.
    trials = trials_from_partner_self_rows(
        merged,
        columns=PartnerSelfTrialTableColumns(t="trial", self_choice="self_choice"),
    )
    return event_log_from_trials(
        block_id=block_id,
        condition=condition,
        trials=trials,
        timing=timing,
        metadata=metadata,
    )


def event_log_from_any_rows(
    data: Any,
    *,
    block_id: str,
    condition: str,
    timing: str = "pre_choice",
    default_state: Any = 0,
    metadata: Mapping[str, Any] | None = None,
) -> EventLog:
    """Best-effort event-log conversion that *infers* the dataset form.

    Supported inputs
    ----------------
    1) A single table (list-of-rows / DataFrame / dict-of-columns).
       - If it contains partner/demo columns, we treat it as combined self+social.
       - Otherwise we treat it as canonical trial table.

    2) A dict with ``{"self": ..., "demo": ...}`` (separate tables).
    """
    if isinstance(data, Mapping) and ("self" in data and "demo" in data):
        return event_log_from_separate_self_demo_rows(
            self_rows=data["self"],
            demo_rows=data["demo"],
            block_id=block_id,
            condition=condition,
            timing=timing,
            keys=("id", "block", "trial"),
            metadata=metadata,
        )

    recs = _as_rows(data)
    if not recs:
        return event_log_from_trials(block_id=block_id, condition=condition, trials=[], timing=timing, metadata=metadata)

    cols = set(map(str, recs[0].keys()))
    if {"partner_choice", "partner_reward"}.intersection(cols) or {"demo_choice", "demo_reward"}.intersection(cols):
        return event_log_from_partner_self_rows(
            data,
            block_id=block_id,
            condition=condition,
            timing=timing,
            default_state=default_state,
            metadata=metadata,
        )

    # Fall back to canonical parser.
    return event_log_from_rows(
        data,
        block_id=block_id,
        condition=condition,
        timing=timing,
        default_state=default_state,
        metadata=metadata,
    )




# -----------------------------------------------------------------------------
# Small parsing helpers
# -----------------------------------------------------------------------------


def _is_null(x: Any) -> bool:
    """Return True for common null/missing values.

    We keep this dependency-minimal (no pandas), but try to handle common
    sentinels used by numpy/pandas (e.g., NaN).
    """
    if x is None:
        return True

    # numpy / python float NaN
    if isinstance(x, (float, np.floating)):
        return bool(np.isnan(x))

    # pandas missing sentinels (without importing pandas)
    tname = type(x).__name__
    if tname in {"NAType", "NaTType"}:
        return True

    # Final fallback: try numpy's isnan if it supports the type.
    try:
        return bool(np.isnan(x))  # type: ignore[arg-type]
    except Exception:
        return False


def _maybe_json(x: Any) -> Any:
    """Parse JSON if ``x`` looks like a JSON string, otherwise return unchanged."""
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return x
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return x
    return x


def _maybe_value(x: Any) -> Any:
    """Best-effort parsing for metadata values.

    - If the value looks like a list encoding (e.g. "0-1-2", "0,1,2", "[0,1,2]"),
      return a Python list.
    - If it looks like JSON (dict/list), parse it.
    - Otherwise return as-is.
    """
    if isinstance(x, (str, list, tuple, np.ndarray)):
        lst = _maybe_list(x)
        if lst is not None:
            return lst
    return _maybe_json(x)


def _maybe_list(x: Any) -> list[Any] | None:
    """Coerce common list encodings into a Python list.

    Supports:
      * None / NaN -> None
      * list/tuple/np.ndarray -> list
      * JSON strings like "[0, 1]"
      * comma-separated strings like "0,1"
    """
    if _is_null(x):
        return None
    x = _maybe_json(x)
    if _is_null(x):
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return [x.item(i) for i in range(int(x.size))]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # JSON string already handled above.
        # Also support common hyphen-delimited encodings like "0-1-2"
        # (used for randomized image orders in some tasks).
        if "-" in s and "," not in s:
            # Only treat as delimiter if it is a pure integer list like 0-1-2.
            import re as _re
            if _re.fullmatch(r"\d+(?:-\d+)+", s):
                return [int(p) for p in s.split("-")]

        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            out: list[Any] = []
            for p in parts:
                try:
                    out.append(int(p))
                except Exception:
                    try:
                        out.append(float(p))
                    except Exception:
                        out.append(p)
            return out
        return [s]
    return [x]


def _maybe_dict(x: Any) -> dict[str, Any]:
    """Coerce JSON-like mapping encodings into a dict."""
    if _is_null(x):
        return {}
    x = _maybe_json(x)
    if _is_null(x):
        return {}
    if isinstance(x, Mapping):
        return dict(x)
    return {"value": x}



def _maybe_bool(x: Any) -> bool | None:
    """Coerce common boolean encodings to bool.

    Returns None if the value is null/missing.
    Supports bools, 0/1 ints, and strings like 'true'/'false'.
    """
    if _is_null(x):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        if int(x) in (0, 1):
            return bool(int(x))
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _as_rows(data: Any) -> list[Mapping[str, Any]]:
    """Normalize a variety of tabular inputs into a list of row mappings."""
    if data is None:
        raise TypeError("data is None")

    # Common case: list of dict rows.
    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], Mapping):
            return [dict(r) for r in data]  # type: ignore[arg-type]

    # Pandas DataFrame duck typing.
    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        try:
            recs = to_dict(orient="records")
        except TypeError:
            recs = to_dict("records")
        if isinstance(recs, list) and (len(recs) == 0 or isinstance(recs[0], Mapping)):
            return [dict(r) for r in recs]  # type: ignore[arg-type]

    # Dict of columns.
    if isinstance(data, Mapping):
        cols = {str(k): v for k, v in data.items()}
        if not cols:
            return []
        first = next(iter(cols.values()))
        try:
            n = len(first)
        except Exception as e:
            raise TypeError("Dict-of-columns input must map to sequences (lists/arrays)") from e

        # Sanity-check that all columns have the same length.
        for k, v in cols.items():
            try:
                nv = len(v)
            except Exception as e:
                raise TypeError(f"Column {k!r} is not a sequence") from e
            if nv != n:
                raise ValueError(f"Column-length mismatch for {k!r}: expected {n}, got {nv}")
        out: list[dict[str, Any]] = []
        for i in range(n):
            out.append({k: cols[k][i] for k in cols.keys()})
        return out

    # Numpy structured array.
    if isinstance(data, np.ndarray) and data.dtype.names:
        names = list(data.dtype.names)
        out = []
        for row in data:
            out.append({n: row[n].item() if hasattr(row[n], "item") else row[n] for n in names})
        return out

    raise TypeError(
        "Unsupported tabular input. Provide a list[dict], a dict-of-columns, "
        "a pandas-like DataFrame (to_dict('records')), or a numpy structured array."
    )


# -----------------------------------------------------------------------------
# Core converters
# -----------------------------------------------------------------------------


def event_log_from_trials(
    *,
    block_id: str,
    condition: str,
    trials: Sequence[Trial],
    timing: str = "asocial",
    metadata: Mapping[str, Any] | None = None,
) -> EventLog:
    """Build an :class:`~comp_model_core.events.types.EventLog` from trials.

    Parameters
    ----------
    block_id
        Block identifier.
    condition
        Block condition label. **Required** for replay likelihood.
    trials
        Ordered trials.
    timing
        Event ordering within each trial.

        * ``"asocial"``: ``CHOICE -> OUTCOME``
        * ``"pre_choice"``: ``SOCIAL_OBSERVED -> CHOICE -> OUTCOME``
        * ``"post_outcome"``: ``CHOICE -> OUTCOME -> SOCIAL_OBSERVED``
    metadata
        Optional event-log metadata.

    Returns
    -------
    EventLog
        Validated event log.

    Notes
    -----
    * If a trial has ``choice is None``, the converter omits ``CHOICE`` and
      ``OUTCOME`` for that trial.
    * For social timings, a ``SOCIAL_OBSERVED`` event is emitted for every trial
      (with empty lists if missing) to keep timing consistent.
    """

    timing = str(timing)
    if timing not in {"asocial", "pre_choice", "post_outcome"}:
        raise ValueError(f"Unknown timing={timing!r}. Expected 'asocial', 'pre_choice', or 'post_outcome'.")

    events: list[Event] = []
    events.append(
        Event(
            idx=0,
            type=EventType.BLOCK_START,
            t=None,
            state=None,
            payload={"block_id": str(block_id), "condition": str(condition)},
        )
    )
    idx = 1

    for tr in trials:
        t = int(tr.t)
        state = tr.state

        def add_social() -> None:
            nonlocal idx
            events.append(
                Event(
                    idx=idx,
                    type=EventType.SOCIAL_OBSERVED,
                    t=t,
                    state=state,
                    payload={
                        "others_choices": list(tr.others_choices or []),
                        "others_outcomes": list(tr.others_outcomes or []),
                        "observed_others_outcomes": None
                        if tr.observed_others_outcomes is None
                        else list(tr.observed_others_outcomes),
                        "social_info": dict(tr.social_info or {}),
                    },
                )
            )
            idx += 1

        def add_choice_and_outcome() -> None:
            nonlocal idx
            if tr.choice is None:
                return
            choice = int(tr.choice)
            aa = None if tr.available_actions is None else list(tr.available_actions)
            events.append(
                Event(
                    idx=idx,
                    type=EventType.CHOICE,
                    t=t,
                    state=state,
                    payload={"choice": choice, "available_actions": aa},
                )
            )
            idx += 1
            events.append(
                Event(
                    idx=idx,
                    type=EventType.OUTCOME,
                    t=t,
                    state=state,
                    payload={
                        "action": choice,
                        "observed_outcome": tr.observed_outcome,
                        "outcome": tr.outcome,
                        "info": dict(tr.info or {}),
                    },
                )
            )
            idx += 1

        if timing == "asocial":
            add_choice_and_outcome()
        elif timing == "pre_choice":
            add_social()
            add_choice_and_outcome()
        elif timing == "post_outcome":
            add_choice_and_outcome()
            add_social()

    log = EventLog(events=events, metadata=dict(metadata or {"timing": timing}))
    validate_event_log(log)
    return log


def trials_from_rows(
    rows: Any,
    *,
    columns: TrialTableColumns | None = None,
    default_state: Any = 0,
) -> list[Trial]:
    """Parse row-per-trial data into :class:`~comp_model_core.data.types.Trial` objects.

    Parameters
    ----------
    rows
        One of: list[dict], dict-of-columns, pandas DataFrame, numpy structured array.
        Each row should have at least a trial index (``t``) and may include the
        other trial fields.
    columns
        Column-name mapping.
    default_state
        Used when the state column is missing.

    Returns
    -------
    list[Trial]
        Trials sorted by ``t``.
    """
    cols = columns or TrialTableColumns()
    recs = _as_rows(rows)

    trials: list[Trial] = []
    for r in recs:
        t_raw = r.get(cols.t)
        if _is_null(t_raw):
            raise ValueError(f"Missing trial index column {cols.t!r} in row: {r}")
        t = int(t_raw)

        state = r.get(cols.state, default_state)

        choice_raw = r.get(cols.choice, None)
        choice = None if _is_null(choice_raw) else int(choice_raw)

        obs_raw = r.get(cols.observed_outcome, None)
        observed_outcome = None if _is_null(obs_raw) else float(obs_raw)

        out_raw = r.get(cols.outcome, None)
        outcome = None if _is_null(out_raw) else float(out_raw)

        aa = _maybe_list(r.get(cols.available_actions, None))
        if aa is not None:
            aa = [int(a) for a in aa]

        info = _maybe_dict(r.get(cols.info, None))

        others_choices = _maybe_list(r.get(cols.others_choices, None))
        if others_choices is not None:
            others_choices = [int(a) for a in others_choices]

        others_outcomes = _maybe_list(r.get(cols.others_outcomes, None))
        if others_outcomes is not None:
            others_outcomes = [float(x) for x in others_outcomes]

        obs_others = _maybe_list(r.get(cols.observed_others_outcomes, None))
        if obs_others is not None:
            obs_others = [float(x) for x in obs_others]

        social_info = _maybe_dict(r.get(cols.social_info, None))

        trials.append(
            Trial(
                t=t,
                state=state,
                choice=choice,
                observed_outcome=observed_outcome,
                outcome=outcome,
                available_actions=aa,
                info=info,
                others_choices=others_choices,
                others_outcomes=others_outcomes,
                observed_others_outcomes=obs_others,
                social_info=social_info,
            )
        )

    trials.sort(key=lambda tr: tr.t)
    return trials


def event_log_from_rows(
    rows: Any,
    *,
    block_id: str,
    condition: str,
    timing: str = "asocial",
    columns: TrialTableColumns | None = None,
    default_state: Any = 0,
    metadata: Mapping[str, Any] | None = None,
) -> EventLog:
    """One-shot conversion from a trial table to an event log."""
    trials = trials_from_rows(rows, columns=columns, default_state=default_state)
    return event_log_from_trials(
        block_id=block_id,
        condition=condition,
        trials=trials,
        timing=timing,
        metadata=metadata,
    )


def attach_event_logs(
    *,
    study: StudyData,
    timing: str | Callable[[SubjectData, Block], str] = "asocial",
) -> StudyData:
    """Return a new :class:`~comp_model_core.data.types.StudyData` with event logs.

    This is a convenience for "real" datasets that already use the library's
    Trial/Block containers but were created without event logs.

    Parameters
    ----------
    study
        Input study.
    timing
        Either a single timing string (applied to all blocks), or a callable
        ``(subject, block) -> timing``.
    """
    subjects_out: list[SubjectData] = []

    for subj in study.subjects:
        blocks_out: list[Block] = []
        for blk in subj.blocks:
            if blk.event_log is not None:
                blocks_out.append(blk)
                continue

            cond = blk.condition
            timing_blk = timing(subj, blk) if callable(timing) else timing
            log = event_log_from_trials(
                block_id=blk.block_id,
                condition=cond,
                trials=blk.trials,
                timing=timing_blk,
                metadata={"timing": str(timing_blk), "source": "attach_event_logs"},
            )
            blocks_out.append(
                Block(
                    block_id=blk.block_id,
                    condition=blk.condition,
                    trials=blk.trials,
                    env_spec=blk.env_spec,
                    event_log=log,
                    metadata=dict(blk.metadata or {}),
                )
            )

        subjects_out.append(
            SubjectData(subject_id=subj.subject_id, blocks=blocks_out, metadata=dict(subj.metadata or {}))
        )

    return StudyData(subjects=subjects_out, metadata=dict(study.metadata or {}))
