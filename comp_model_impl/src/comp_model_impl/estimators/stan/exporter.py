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

from typing import Any, Sequence

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
    _ensure_int_states(subject)

    As = [int(blk.env_spec.n_actions) for blk in subject.blocks if blk.env_spec is not None]
    if len(set(As)) != 1:
        raise ValueError("Stan export expects constant n_actions across blocks for a subject.")
    A = As[0]

    events = []
    for blk in subject.blocks:
        events.extend(get_event_log(blk).events)

    max_state = 0
    for e in events:
        if e.state is not None:
            max_state = max(max_state, int(e.state))
    S = max_state + 1

    E = len(events)
    etype = np.zeros(E, dtype=int)
    state = np.ones(E, dtype=int)        # 1-indexed
    choice = np.zeros(E, dtype=int)      # 0 unless CHOICE
    action = np.zeros(E, dtype=int)      # 0 unless OUTCOME
    outcome_obs = np.zeros(E, dtype=float)

    demo_action = np.zeros(E, dtype=int) # 0 unless SOCIAL_OBSERVED
    demo_outcome_obs = np.zeros(E, dtype=float)
    has_demo_outcome = np.zeros(E, dtype=int)
    avail_mask = np.ones((E, A), dtype=float)  # 1 if action available for this event

    for i, e in enumerate(events):
        etype[i] = int(e.type)
        state[i] = (0 if e.state is None else int(e.state)) + 1

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
