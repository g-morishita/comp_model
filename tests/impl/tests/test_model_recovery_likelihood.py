"""Tests for model recovery likelihood helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from comp_model_core.data.types import Block, StudyData, SubjectData
from comp_model_core.events.types import Event, EventLog, EventType
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.params import ParamDef, ParameterSchema
from comp_model_core.spec import EnvironmentSpec

from comp_model_impl.recovery.model import likelihood as likelihood_mod


@dataclass(slots=True)
class _DummyModel(ComputationalModel):
    """Minimal model used to satisfy interface typing in tests."""

    theta: float = 0.5

    @property
    def param_schema(self) -> ParameterSchema:
        return ParameterSchema((ParamDef("theta", float(self.theta)),))

    def supports(self, spec: EnvironmentSpec) -> bool:
        return True

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        return

    def action_probs(self, *, state, spec: EnvironmentSpec) -> np.ndarray:
        return np.array([0.5, 0.5], dtype=float)

    def update(self, *, state, action: int, outcome, spec: EnvironmentSpec, info=None, rng=None) -> None:
        return


def _event_log_with_n_choices(n_choices: int) -> EventLog:
    """Create a minimal valid event log with ``n_choices`` choice events."""
    events: list[Event] = [
        Event(
            idx=0,
            type=EventType.BLOCK_START,
            t=None,
            state=None,
            payload={"block_id": "b1", "condition": "c1"},
        )
    ]
    idx = 1
    for t in range(int(n_choices)):
        events.append(
            Event(
                idx=idx,
                type=EventType.CHOICE,
                t=t,
                state=0,
                payload={"choice": 0, "available_actions": [0, 1]},
            )
        )
        idx += 1
        events.append(
            Event(
                idx=idx,
                type=EventType.OUTCOME,
                t=t,
                state=0,
                payload={"action": 0, "observed_outcome": 1.0, "outcome": 1.0, "info": {}},
            )
        )
        idx += 1
    return EventLog(events=events, metadata={"test": True})


def _study_for_subject_choice_counts(counts: dict[str, int]) -> StudyData:
    """Create a study where each subject has one block and a chosen count."""
    subjects = []
    for sid, n in counts.items():
        block = Block(
            block_id="b1",
            condition="c1",
            trials=[],
            env_spec=None,
            event_log=_event_log_with_n_choices(n),
        )
        subjects.append(SubjectData(subject_id=sid, blocks=[block]))
    return StudyData(subjects=subjects, metadata={})


def test_count_choice_events_counts_only_choice_type() -> None:
    """Choice-event counter should count only CHOICE events."""
    subj = _study_for_subject_choice_counts({"s1": 3}).subjects[0]
    assert likelihood_mod._count_choice_events(subj) == 3


def test_compute_likelihood_summary_aggregates_subject_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Likelihood summary should aggregate values and observation counts."""
    study = _study_for_subject_choice_counts({"s1": 2, "s2": 1})
    model = _DummyModel()

    ll_values = {"s1": -1.25, "s2": -0.75}

    def fake_loglike_subject(*, subject, model, params):
        return ll_values[str(subject.subject_id)]

    monkeypatch.setattr(likelihood_mod, "loglike_subject", fake_loglike_subject)
    out = likelihood_mod.compute_likelihood_summary(
        study=study,
        model=model,
        subject_params={"s1": {"theta": 0.2}, "s2": {"theta": 0.3}},
    )

    assert out.ll_by_subject == {"s1": -1.25, "s2": -0.75}
    assert out.ll_total == pytest.approx(-2.0)
    assert out.n_obs_by_subject == {"s1": 2, "s2": 1}
    assert out.n_obs_total == 3


def test_compute_likelihood_summary_missing_subject_params_sets_neg_inf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subjects missing fit params should get ``-inf`` likelihood."""
    study = _study_for_subject_choice_counts({"s1": 1, "s2": 1})
    model = _DummyModel()

    def fake_loglike_subject(*, subject, model, params):
        return -1.0

    monkeypatch.setattr(likelihood_mod, "loglike_subject", fake_loglike_subject)
    out = likelihood_mod.compute_likelihood_summary(
        study=study,
        model=model,
        subject_params={"s1": {"theta": 0.2}},
    )

    assert out.ll_by_subject["s1"] == pytest.approx(-1.0)
    assert out.ll_by_subject["s2"] == float("-inf")
    assert out.ll_total == float("-inf")


def test_compute_likelihood_summary_loglike_exception_sets_neg_inf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Likelihood errors should be converted to ``-inf`` per subject."""
    study = _study_for_subject_choice_counts({"s1": 1, "s2": 1})
    model = _DummyModel()

    def fake_loglike_subject(*, subject, model, params):
        if str(subject.subject_id) == "s2":
            raise RuntimeError("boom")
        return -0.5

    monkeypatch.setattr(likelihood_mod, "loglike_subject", fake_loglike_subject)
    out = likelihood_mod.compute_likelihood_summary(
        study=study,
        model=model,
        subject_params={"s1": {"theta": 0.2}, "s2": {"theta": 0.3}},
    )

    assert out.ll_by_subject["s1"] == pytest.approx(-0.5)
    assert out.ll_by_subject["s2"] == float("-inf")
    assert out.ll_total == float("-inf")
