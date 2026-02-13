"""Export :class:`~comp_model_core.data.types.StudyData` to Stan ``data``.

The Stan estimators operate by replaying an event log produced during simulation
or ingestion. This module converts the internal event-log format into the
arrays expected by the Stan templates.

Notes
-----
The Stan templates assume action and state indices are 1-based, with 0 reserved
for "missing" action entries.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.events.accessors import get_event_log
from comp_model_core.events.types import EventType


def _ensure_int_states(subject: SubjectData) -> None:
    """Validate that all event-log states are integer-castable.

    Parameters
    ----------
    subject : comp_model_core.data.types.SubjectData
        Subject whose event logs will be exported.

    Raises
    ------
    ValueError
        If any event state cannot be cast to an integer.
    """
    for blk in subject.blocks:
        log = get_event_log(blk)
        for e in log.events:
            if e.state is None:
                continue
            int(e.state)  # will raise if not castable


def _state_key(state: Any) -> str:
    """Build a stable key for arbitrary event states."""
    if state is None:
        return "__none__"
    try:
        s = int(state)
        if s < 0:
            raise ValueError("Stan export expects integer states >= 0.")
        return f"int:{s}"
    except Exception:
        pass
    try:
        return "json:" + json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return f"repr:{repr(state)}"


def _state_indices(events: Sequence[Any]) -> tuple[np.ndarray, int]:
    """Map event states to 1-based contiguous indices for Stan."""
    non_none = [e.state for e in events if e.state is not None]
    all_int_like = True
    for s in non_none:
        try:
            if int(s) < 0:
                raise ValueError("Stan export expects integer states >= 0.")
        except Exception:
            all_int_like = False
            break

    state = np.ones(len(events), dtype=int)
    if all_int_like:
        max_state = 0
        for i, e in enumerate(events):
            if e.state is None:
                state[i] = 1
                continue
            si = int(e.state)
            if si < 0:
                raise ValueError("Stan export expects integer states >= 0.")
            state[i] = si + 1
            max_state = max(max_state, si)
        return state, (max_state + 1)

    key_to_idx: dict[str, int] = {"__none__": 1}
    next_idx = 2
    for i, e in enumerate(events):
        key = _state_key(e.state)
        if key not in key_to_idx:
            key_to_idx[key] = next_idx
            next_idx += 1
        state[i] = int(key_to_idx[key])
    return state, int(max(key_to_idx.values()))


def _moment_stats(*, outcomes: Sequence[float], probs: Sequence[float]) -> tuple[float, float, float]:
    """Compute mean, variance, and standardized skewness."""
    x = np.asarray(outcomes, dtype=float)
    p = np.asarray(probs, dtype=float)
    if x.ndim != 1 or p.ndim != 1 or x.shape[0] != p.shape[0] or x.shape[0] == 0:
        raise ValueError("Invalid lottery shapes: expected equal-length 1D outcomes/probs.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(p)):
        raise ValueError("Lottery outcomes/probabilities must be finite.")
    if np.any(p < 0.0):
        raise ValueError("Lottery probabilities must be non-negative.")
    s = float(np.sum(p))
    if s <= 0.0:
        raise ValueError("Lottery probabilities must sum to a positive value.")
    p = p / s

    mu = float(np.sum(p * x))
    dev = x - mu
    var = float(np.sum(p * (dev ** 2)))
    if var <= 1e-12:
        skew = 0.0
    else:
        skew = float(np.sum(p * (dev ** 3)) / (var ** 1.5))
    return mu, var, skew


def _coerce_action_moments(raw: Any, *, n_actions: int) -> np.ndarray:
    """Normalize action moments to ``(A, 3)`` float array."""
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (int(n_actions), 3):
        raise ValueError(
            f"Expected action moments with shape ({n_actions}, 3), got {arr.shape}."
        )
    if np.any(~np.isfinite(arr)):
        raise ValueError("Non-finite action moments encountered in event log.")
    return arr


def _moments_from_lotteries(raw: Any, *, n_actions: int) -> np.ndarray:
    """Compute moments from explicit lottery definitions."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("lotteries must be a sequence of mappings.")
    if len(raw) != int(n_actions):
        raise ValueError(f"Expected {n_actions} lotteries, got {len(raw)}.")
    rows: list[tuple[float, float, float]] = []
    for lot in raw:
        if not isinstance(lot, Mapping):
            raise ValueError("Each lottery must be a mapping with outcomes/probs.")
        if "outcomes" not in lot or "probs" not in lot:
            raise ValueError("Lottery mapping must include 'outcomes' and 'probs'.")
        mu, var, skew = _moment_stats(
            outcomes=lot["outcomes"],
            probs=lot["probs"],
        )
        rows.append((mu, var, skew))
    return _coerce_action_moments(rows, n_actions=n_actions)


def _extract_action_moments(event: Any, *, n_actions: int) -> np.ndarray | None:
    """Extract per-action moments from event state or payload."""
    st = getattr(event, "state", None)
    if isinstance(st, Mapping):
        if "action_moments" in st:
            return _coerce_action_moments(st["action_moments"], n_actions=n_actions)
        if "lotteries" in st:
            return _moments_from_lotteries(st["lotteries"], n_actions=n_actions)

    payload = getattr(event, "payload", None)
    if not isinstance(payload, Mapping):
        return None

    if "action_moments" in payload:
        return _coerce_action_moments(payload["action_moments"], n_actions=n_actions)
    if "lotteries" in payload:
        return _moments_from_lotteries(payload["lotteries"], n_actions=n_actions)

    info = payload.get("info", None)
    if isinstance(info, Mapping):
        if "action_moments" in info:
            return _coerce_action_moments(info["action_moments"], n_actions=n_actions)
        if "lotteries" in info:
            return _moments_from_lotteries(info["lotteries"], n_actions=n_actions)
    return None




def subject_to_stan_data(subject: SubjectData) -> dict[str, Any]:
    """Convert a single subject into Stan ``data`` for ``indiv`` templates.

    Parameters
    ----------
    subject : comp_model_core.data.types.SubjectData
        Subject containing blocks with event logs.

    Returns
    -------
    dict[str, Any]
        Stan data mapping expected by individual-level templates.

    Notes
    -----
    This function assumes constant ``n_actions`` across a subject's blocks and
    uses 1-based indexing for actions and states (with 0 reserved for "missing").

    Examples
    --------
    Build a minimal subject with a single block and two events:

    >>> from comp_model_core.data.types import Block, SubjectData, Trial
    >>> from comp_model_core.events.types import Event, EventLog, EventType
    >>> from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
    >>> from comp_model_impl.estimators.stan.exporter import subject_to_stan_data
    >>> spec = EnvironmentSpec(
    ...     n_actions=2,
    ...     outcome_type=OutcomeType.BINARY,
    ...     outcome_range=(0.0, 1.0),
    ...     outcome_is_bounded=True,
    ...     is_social=False,
    ...     state_kind=StateKind.DISCRETE,
    ...     n_states=1,
    ... )
    >>> log = EventLog(events=[
    ...     Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "c1"}),
    ...     Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={"choice": 1, "available_actions": [0, 1]}),
    ...     Event(idx=2, type=EventType.OUTCOME, t=0, state=0, payload={"action": 1, "observed_outcome": 1.0, "info": {}}),
    ... ])
    >>> block = Block(
    ...     block_id="b1",
    ...     condition="c1",
    ...     trials=[Trial(t=0, state=0, choice=1, observed_outcome=1.0, outcome=1.0)],
    ...     env_spec=spec,
    ...     event_log=log,
    ... )
    >>> subject = SubjectData(subject_id="s1", blocks=[block])
    >>> data = subject_to_stan_data(subject)
    >>> data["A"], data["S"], data["E"]
    (2, 1, 3)
    """
    As = [int(blk.env_spec.n_actions) for blk in subject.blocks if blk.env_spec is not None]
    if len(set(As)) != 1:
        raise ValueError("Stan export expects constant n_actions across blocks for a subject.")
    A = As[0]

    events = []
    for blk in subject.blocks:
        events.extend(get_event_log(blk).events)

    state, S = _state_indices(events)

    E = len(events)
    etype = np.zeros(E, dtype=int)
    choice = np.zeros(E, dtype=int)      # 0 unless CHOICE
    action = np.zeros(E, dtype=int)      # 0 unless OUTCOME
    outcome_obs = np.zeros(E, dtype=float)

    demo_action = np.zeros(E, dtype=int) # 0 unless SOCIAL_OBSERVED
    demo_outcome_obs = np.zeros(E, dtype=float)
    has_demo_outcome = np.zeros(E, dtype=int)
    avail_mask = np.ones((E, A), dtype=float)  # 1 if action available for this event
    action_mean = np.zeros((E, A), dtype=float)
    action_variance = np.zeros((E, A), dtype=float)
    action_skewness = np.zeros((E, A), dtype=float)

    for i, e in enumerate(events):
        etype[i] = int(e.type)
        moments = _extract_action_moments(e, n_actions=A)
        if moments is not None:
            action_mean[i, :] = moments[:, 0]
            action_variance[i, :] = moments[:, 1]
            action_skewness[i, :] = moments[:, 2]

        p = e.payload
        if e.type == EventType.CHOICE:
            c = p.get("choice", None)
            if c is not None:
                choice[i] = int(c) + 1
            aa = p.get("available_actions", None)
            if aa is not None:
                mask = np.zeros(A, dtype=float)
                for a in aa:
                    mask[int(a)] = 1.0
                avail_mask[i] = mask

        elif e.type == EventType.OUTCOME:
            a = p.get("action", None)
            if a is not None:
                action[i] = int(a) + 1
            r = p.get("observed_outcome", None)
            outcome_obs[i] = 0.0 if r is None else float(r)

        elif e.type == EventType.SOCIAL_OBSERVED:
            oc = p.get("others_choices", []) or []
            if oc:
                demo_action[i] = int(oc[0]) + 1
            oo = p.get("observed_others_outcomes", None)
            if oo is not None and len(oo) > 0 and oo[0] is not None:
                has_demo_outcome[i] = 1
                demo_outcome_obs[i] = float(oo[0])

    return {
        "A": int(A),
        "S": int(S),
        "E": int(E),
        "etype": etype.tolist(),
        "state": state.tolist(),
        "choice": choice.tolist(),
        "action": action.tolist(),
        "outcome_obs": outcome_obs.tolist(),
        "demo_action": demo_action.tolist(),
        "demo_outcome_obs": demo_outcome_obs.tolist(),
        "has_demo_outcome": has_demo_outcome.tolist(),
        "avail_mask": avail_mask.tolist(),
        "action_mean": action_mean.tolist(),
        "action_variance": action_variance.tolist(),
        "action_skewness": action_skewness.tolist(),
    }


def study_to_stan_data(study: StudyData) -> dict[str, Any]:
    """Convert a multi-subject study into Stan ``data`` for ``hier`` templates.

    Parameters
    ----------
    study : comp_model_core.data.types.StudyData
        Study containing multiple subjects with event logs.

    Returns
    -------
    dict[str, Any]
        Stan data mapping expected by hierarchical templates.

    Raises
    ------
    ValueError
        If the study is empty or subjects have different action counts.

    Examples
    --------
    >>> from comp_model_impl.estimators.stan.exporter import study_to_stan_data
    >>> # Build two tiny subjects that share the same action/state spaces.
    >>> from comp_model_core.data.types import Block, SubjectData, Trial
    >>> from comp_model_core.events.types import Event, EventLog, EventType
    >>> from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
    >>> spec = EnvironmentSpec(
    ...     n_actions=2,
    ...     outcome_type=OutcomeType.BINARY,
    ...     outcome_range=(0.0, 1.0),
    ...     outcome_is_bounded=True,
    ...     is_social=False,
    ...     state_kind=StateKind.DISCRETE,
    ...     n_states=1,
    ... )
    >>> def _subj(sid, choice):
    ...     log = EventLog(events=[
    ...         Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "c1"}),
    ...         Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={"choice": choice, "available_actions": [0, 1]}),
    ...         Event(idx=2, type=EventType.OUTCOME, t=0, state=0, payload={"action": choice, "observed_outcome": 1.0, "info": {}}),
    ...     ])
    ...     block = Block(block_id="b1", condition="c1",
    ...                  trials=[Trial(t=0, state=0, choice=choice, observed_outcome=1.0, outcome=1.0)],
    ...                  env_spec=spec, event_log=log)
    ...     return SubjectData(subject_id=sid, blocks=[block])
    >>> study = StudyData(subjects=[_subj("s1", 0), _subj("s2", 1)])
    >>> data = study_to_stan_data(study)
    >>> data["N"], data["A"], data["S"]
    (2, 2, 1)
    """
    N = len(study.subjects)
    if N == 0:
        raise ValueError("Empty study")

    subj_chunks = [subject_to_stan_data(s) for s in study.subjects]
    A_set = {d["A"] for d in subj_chunks}
    if len(A_set) != 1:
        raise ValueError("Hier Stan export expects constant n_actions across subjects.")
    A = int(next(iter(A_set)))
    S = max(int(d["S"]) for d in subj_chunks)

    etype=[]; state=[]; choice=[]; action=[]; outcome_obs=[]
    demo_action=[]; demo_outcome_obs=[]; has_demo_outcome=[]; subj=[]
    avail_mask=[]
    action_mean=[]; action_variance=[]; action_skewness=[]
    for si, d in enumerate(subj_chunks, start=1):
        for i in range(int(d["E"])):
            etype.append(int(d["etype"][i]))
            state.append(int(d["state"][i]))   # already 1-indexed
            choice.append(int(d["choice"][i]))
            action.append(int(d["action"][i]))
            outcome_obs.append(float(d["outcome_obs"][i]))
            demo_action.append(int(d["demo_action"][i]))
            demo_outcome_obs.append(float(d["demo_outcome_obs"][i]))
            has_demo_outcome.append(int(d["has_demo_outcome"][i]))
            subj.append(si)
            avail_mask.append(list(d["avail_mask"][i]))
            action_mean.append(list(d["action_mean"][i]))
            action_variance.append(list(d["action_variance"][i]))
            action_skewness.append(list(d["action_skewness"][i]))

    return {
        "N": int(N),
        "A": int(A),
        "S": int(S),
        "E": int(len(etype)),
        "subj": subj,
        "etype": etype,
        "state": state,
        "choice": choice,
        "action": action,
        "outcome_obs": outcome_obs,
        "demo_action": demo_action,
        "demo_outcome_obs": demo_outcome_obs,
        "has_demo_outcome": has_demo_outcome,
        "avail_mask": avail_mask,
        "action_mean": action_mean,
        "action_variance": action_variance,
        "action_skewness": action_skewness,
    }


# ---------------------------------------------------------------------------
# Within-subject (block-level) condition export
# ---------------------------------------------------------------------------

def _dedupe_preserve_order_str(xs: Sequence[str]) -> list[str]:
    """Deduplicate string labels while preserving first-seen order.

    Parameters
    ----------
    xs : Sequence[str]
        Input labels.

    Returns
    -------
    list[str]
        Deduplicated labels in first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        x = str(x)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _events_for_subject(subject: SubjectData):
    """Collect events across all blocks for a subject.

    Parameters
    ----------
    subject : comp_model_core.data.types.SubjectData
        Subject whose events will be concatenated.

    Returns
    -------
    list[comp_model_core.events.types.Event]
        Concatenated event list in block order.
    """
    events = []
    for blk in subject.blocks:
        events.extend(get_event_log(blk).events)
    return events


def subject_to_stan_data_within_subject(
    subject: SubjectData,
    *,
    conditions: Sequence[str],
    baseline_condition: str,
) -> dict[str, Any]:
    """Convert a subject to Stan data for within-subject templates.

    Parameters
    ----------
    subject : comp_model_core.data.types.SubjectData
        Subject data containing blocks with event logs.
    conditions : Sequence[str]
        Ordered list of allowed condition labels. IDs will be 1-based in this
        order.
    baseline_condition : str
        Label treated as the baseline. Deltas are defined for non-baseline
        conditions.

    Returns
    -------
    dict[str, Any]
        Stan data mapping including within-subject condition indices.

    Raises
    ------
    ValueError
        If conditions are missing, baseline is not in the list, or event logs
        omit BLOCK_START conditions.

    Examples
    --------
    >>> from comp_model_core.data.types import Block, SubjectData, Trial
    >>> from comp_model_core.events.types import Event, EventLog, EventType
    >>> from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
    >>> spec = EnvironmentSpec(
    ...     n_actions=2,
    ...     outcome_type=OutcomeType.BINARY,
    ...     outcome_range=(0.0, 1.0),
    ...     outcome_is_bounded=True,
    ...     is_social=False,
    ...     state_kind=StateKind.DISCRETE,
    ...     n_states=1,
    ... )
    >>> log = EventLog(events=[
    ...     Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "A"}),
    ...     Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={"choice": 0, "available_actions": [0, 1]}),
    ...     Event(idx=2, type=EventType.OUTCOME, t=0, state=0, payload={"action": 0, "observed_outcome": 1.0, "info": {}}),
    ... ])
    >>> block = Block(block_id="b1", condition="A",
    ...               trials=[Trial(t=0, state=0, choice=0, observed_outcome=1.0, outcome=1.0)],
    ...               env_spec=spec, event_log=log)
    >>> subject = SubjectData(subject_id="s1", blocks=[block])
    >>> data = subject_to_stan_data_within_subject(subject, conditions=["A", "B"], baseline_condition="A")
    >>> data["C"], data["baseline_cond"]
    (2, 1)
    """
    base = subject_to_stan_data(subject)
    cond_labels = _dedupe_preserve_order_str([str(c) for c in conditions])
    if not cond_labels:
        raise ValueError("conditions must be non-empty for within-subject Stan export.")
    baseline = str(baseline_condition)
    if baseline not in cond_labels:
        raise ValueError(f"baseline_condition {baseline!r} must be included in conditions={cond_labels!r}")

    cond_to_id = {c: i + 1 for i, c in enumerate(cond_labels)}
    baseline_id = cond_to_id[baseline]

    events = _events_for_subject(subject)
    if len(events) != int(base["E"]):
        raise AssertionError("Internal error: event count mismatch in exporter.")

    cond = np.zeros(len(events), dtype=int)

    current: int | None = None
    for i, e in enumerate(events):
        if e.type == EventType.BLOCK_START:
            c = e.payload.get("condition", None)
            if c is None:
                raise ValueError("Missing 'condition' in BLOCK_START payload (no-default philosophy).")
            c = str(c)
            if c not in cond_to_id:
                raise ValueError(f"Unknown condition {c!r} in event log. Expected one of {cond_labels!r}.")
            current = cond_to_id[c]
        if current is None:
            raise ValueError(
                "Event log contains events before the first BLOCK_START, so condition is undefined."
            )
        cond[i] = int(current)

    base |= {
        "C": int(len(cond_labels)),
        "baseline_cond": int(baseline_id),
        "cond": cond.tolist(),
    }
    return base


def study_to_stan_data_within_subject(
    study: StudyData,
    *,
    conditions: Sequence[str],
    baseline_condition: str,
) -> dict[str, Any]:
    """Convert a study into Stan data for within-subject hierarchical templates.

    Parameters
    ----------
    study : comp_model_core.data.types.StudyData
        Study containing multiple subjects with event logs.
    conditions : Sequence[str]
        Ordered list of allowed condition labels.
    baseline_condition : str
        Label treated as the baseline.

    Returns
    -------
    dict[str, Any]
        Stan data mapping for within-subject hierarchical templates.

    Raises
    ------
    ValueError
        If the study is empty or condition metadata are inconsistent across subjects.

    Examples
    --------
    >>> from comp_model_core.data.types import Block, SubjectData, Trial, StudyData
    >>> from comp_model_core.events.types import Event, EventLog, EventType
    >>> from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
    >>> spec = EnvironmentSpec(
    ...     n_actions=2,
    ...     outcome_type=OutcomeType.BINARY,
    ...     outcome_range=(0.0, 1.0),
    ...     outcome_is_bounded=True,
    ...     is_social=False,
    ...     state_kind=StateKind.DISCRETE,
    ...     n_states=1,
    ... )
    >>> def _subj(sid, cond):
    ...     log = EventLog(events=[
    ...         Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": cond}),
    ...         Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={"choice": 0, "available_actions": [0, 1]}),
    ...         Event(idx=2, type=EventType.OUTCOME, t=0, state=0, payload={"action": 0, "observed_outcome": 1.0, "info": {}}),
    ...     ])
    ...     block = Block(block_id=f"{sid}_{cond}", condition=cond,
    ...                  trials=[Trial(t=0, state=0, choice=0, observed_outcome=1.0, outcome=1.0)],
    ...                  env_spec=spec, event_log=log)
    ...     return SubjectData(subject_id=sid, blocks=[block])
    >>> study = StudyData(subjects=[_subj("s1", "A"), _subj("s2", "A")])
    >>> data = study_to_stan_data_within_subject(study, conditions=["A"], baseline_condition="A")
    >>> data["N"], data["C"]
    (2, 1)
    """
    N = len(study.subjects)
    if N == 0:
        raise ValueError("Empty study")

    cond_labels = _dedupe_preserve_order_str([str(c) for c in conditions])
    if not cond_labels:
        raise ValueError("conditions must be non-empty for within-subject Stan export.")
    baseline = str(baseline_condition)
    if baseline not in cond_labels:
        raise ValueError(f"baseline_condition {baseline!r} must be included in conditions={cond_labels!r}")

    subj_chunks = [
        subject_to_stan_data_within_subject(s, conditions=cond_labels, baseline_condition=baseline)
        for s in study.subjects
    ]

    A_set = {d["A"] for d in subj_chunks}
    if len(A_set) != 1:
        raise ValueError("Hier Stan export expects constant n_actions across subjects.")
    A = int(next(iter(A_set)))
    S = max(int(d["S"]) for d in subj_chunks)

    C_set = {int(d["C"]) for d in subj_chunks}
    if len(C_set) != 1:
        raise ValueError("Hier within-subject export expects constant C across subjects.")
    C = int(next(iter(C_set)))

    b_set = {int(d["baseline_cond"]) for d in subj_chunks}
    if len(b_set) != 1:
        raise ValueError("Hier within-subject export expects constant baseline_cond across subjects.")
    baseline_id = int(next(iter(b_set)))

    etype=[]; state=[]; choice=[]; action=[]; outcome_obs=[]
    demo_action=[]; demo_outcome_obs=[]; has_demo_outcome=[]; subj=[]; cond=[]
    avail_mask=[]
    for si, d in enumerate(subj_chunks, start=1):
        for i in range(int(d["E"])):
            etype.append(int(d["etype"][i]))
            state.append(int(d["state"][i]))
            choice.append(int(d["choice"][i]))
            action.append(int(d["action"][i]))
            outcome_obs.append(float(d["outcome_obs"][i]))
            demo_action.append(int(d["demo_action"][i]))
            demo_outcome_obs.append(float(d["demo_outcome_obs"][i]))
            has_demo_outcome.append(int(d["has_demo_outcome"][i]))
            avail_mask.append(list(d["avail_mask"][i]))
            cond.append(int(d["cond"][i]))
            subj.append(si)

    return {
        "N": int(N),
        "A": int(A),
        "S": int(S),
        "C": int(C),
        "baseline_cond": int(baseline_id),
        "E": int(len(etype)),
        "subj": subj,
        "cond": cond,
        "etype": etype,
        "state": state,
        "choice": choice,
        "action": action,
        "outcome_obs": outcome_obs,
        "demo_action": demo_action,
        "demo_outcome_obs": demo_outcome_obs,
        "has_demo_outcome": has_demo_outcome,
        "avail_mask": avail_mask,
    }
