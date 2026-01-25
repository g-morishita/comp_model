from __future__ import annotations
from typing import Any

import numpy as np

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.events.accessors import get_event_log
from comp_model_core.events.types import EventType

def _ensure_int_states(subject: SubjectData) -> None:
    for blk in subject.blocks:
        log = get_event_log(blk)
        for e in log.events:
            if e.state is None:
                continue
            int(e.state)  # will raise if not castable

def subject_to_stan_data(subject: SubjectData) -> dict[str, Any]:
    _ensure_int_states(subject)

    # assume constant A across blocks for a subject
    As = [int(blk.task_spec.max_n_actions) for blk in subject.blocks if blk.task_spec is not None]
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

    for i, e in enumerate(events):
        etype[i] = int(e.type)
        state[i] = (0 if e.state is None else int(e.state)) + 1

        p = e.payload
        if e.type == EventType.CHOICE:
            c = p.get("choice", None)
            if c is not None:
                choice[i] = int(c) + 1

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
    }

def study_to_stan_data(study: StudyData) -> dict[str, Any]:
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
    }
