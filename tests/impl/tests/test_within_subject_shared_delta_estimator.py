"""Tests for within-subject shared+delta MLE estimator.

This suite focuses on the orchestration logic in
``comp_model_impl.estimators.within_subject_shared_delta``:
- condition inference and validation
- within-subject structure enforcement
- estimator wiring and diagnostics
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from comp_model_core.data.types import Block, StudyData, SubjectData, Trial
from comp_model_core.plans.block import BlockPlan

from comp_model_impl.estimators.within_subject_shared_delta import (
    WithinSubjectSharedDeltaTransformedMLEEstimator,
    _ensure_within_subject_structure,
    _infer_conditions_from_study,
)
from comp_model_impl.generators.event_log import EventLogAsocialGenerator
from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.register import make_registry
from comp_model_impl.tasks.build import build_runner_for_plan


def _make_within_subject_study(*, n_subjects: int = 2, n_trials: int = 6, seed: int = 0) -> StudyData:
    """Create a small within-subject study with conditions A and B.

    Parameters
    ----------
    n_subjects : int, optional
        Number of subjects to simulate.
    n_trials : int, optional
        Trials per block.
    seed : int, optional
        Random seed base for subject simulation.

    Returns
    -------
    comp_model_core.data.types.StudyData
        Study with event logs per block, suitable for event-log replay MLE.

    Notes
    -----
    This helper uses :class:`EventLogAsocialGenerator` and
    :func:`comp_model_impl.tasks.build.build_runner_for_plan` to generate
    logs compatible with the estimator under test.
    """
    reg = make_registry()

    def builder(p):
        return build_runner_for_plan(plan=p, registries=reg)

    params = {"alpha": 0.3, "beta": 2.0}
    gen = EventLogAsocialGenerator()
    subjects = []
    for i in range(int(n_subjects)):
        sid = f"s{i+1}"
        plan_a = BlockPlan(
            block_id=f"{sid}_A",
            n_trials=int(n_trials),
            condition="A",
            bandit_type="BernoulliBanditEnv",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}]
            * int(n_trials),
        )
        plan_b = BlockPlan(
            block_id=f"{sid}_B",
            n_trials=int(n_trials),
            condition="B",
            bandit_type="BernoulliBanditEnv",
            bandit_config={"probs": [0.8, 0.2]},
            trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}]
            * int(n_trials),
        )
        subj = gen.simulate_subject(
            subject_id=sid,
            block_runner_builder=builder,
            model=QRL(),
            params=params,
            block_plans=[plan_a, plan_b],
            rng=np.random.default_rng(seed + i + 1),
        )
        subjects.append(subj)

    return StudyData(subjects=subjects, metadata={"seed": int(seed)})


def _make_study_missing_condition() -> StudyData:
    """Create a study with a block missing condition (for error paths)."""
    trial = Trial(t=0, state=0, choice=0, observed_outcome=1.0, outcome=1.0)
    block = Block(block_id="b1", condition=None, trials=[trial])  # type: ignore[arg-type]
    subj = SubjectData(subject_id="s1", blocks=[block])
    return StudyData(subjects=[subj], metadata={})


def test_infer_conditions_from_study_order():
    """Infer condition order based on first appearance in the study."""
    study = _make_within_subject_study()
    conds = _infer_conditions_from_study(study)
    assert conds == ["A", "B"]


def test_infer_conditions_raises_on_missing():
    """Missing block conditions should raise during inference."""
    study = _make_study_missing_condition()
    with pytest.raises(ValueError):
        _infer_conditions_from_study(study)


def test_within_subject_structure_enforced():
    """Subjects must contain the same condition set."""
    study = _make_within_subject_study()
    subj0 = study.subjects[0]
    # Drop one block to violate the within-subject condition set.
    subj0_bad = replace(subj0, blocks=list(subj0.blocks[:1]))
    bad = StudyData(subjects=[subj0_bad] + list(study.subjects[1:]), metadata=dict(study.metadata))
    with pytest.raises(ValueError):
        _ensure_within_subject_structure(bad, ["A", "B"])


def test_within_subject_estimator_fit_outputs():
    """Estimator returns diagnostics with derived per-condition parameters."""
    study = _make_within_subject_study(n_subjects=2, n_trials=4)
    base = QRL()
    est = WithinSubjectSharedDeltaTransformedMLEEstimator(
        base_model=base,
        baseline_condition="A",
        conditions=["A", "B"],
        n_starts=2,
        maxiter=100,
        z_init_scale=0.5,
    )
    res = est.fit(study=study, rng=np.random.default_rng(0))

    assert res.success is True
    assert res.subject_hats is not None
    assert isinstance(res.diagnostics, dict)

    diag = res.diagnostics
    assert diag.get("within_subject") is True
    assert diag.get("baseline_condition") == "A"
    assert diag.get("conditions") == ["A", "B"]
    derived = diag.get("derived_params_by_condition")
    assert isinstance(derived, dict)

    # Derived parameters should exist for each subject and condition.
    for sid in ["s1", "s2"]:
        assert sid in derived
        assert set(derived[sid].keys()) == {"A", "B"}
        for cond in ["A", "B"]:
            assert set(derived[sid][cond].keys()) == set(base.param_schema.names)


def test_within_subject_estimator_rejects_missing_baseline():
    """Baseline condition must be present in the provided condition list."""
    study = _make_within_subject_study(n_subjects=1, n_trials=2)
    est = WithinSubjectSharedDeltaTransformedMLEEstimator(
        base_model=QRL(),
        baseline_condition="C",
        conditions=["A", "B"],
    )
    with pytest.raises(ValueError):
        est.fit(study=study, rng=np.random.default_rng(0))
