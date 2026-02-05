import numpy as np
import pytest

from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
from comp_model_core.data.types import Block, StudyData, SubjectData, Trial
from comp_model_core.events.types import Event, EventLog, EventType

from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.models.within_subject_shared_delta import (
    wrap_model_with_shared_delta_conditions,
    constrained_params_by_condition_from_z,
    flatten_params_by_condition,
)
from comp_model_impl.estimators.within_subject_shared_delta import WithinSubjectSharedDeltaTransformedMLEEstimator



def test_shared_delta_wrapper_requires_condition_and_applies_per_condition_params():
    base = QRL()
    wrapper = wrap_model_with_shared_delta_conditions(model=base, conditions=["A", "B"], baseline_condition="A")

    spec = EnvironmentSpec(
        n_actions=2,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=False,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )

    # reset_block requires an active condition.
    with pytest.raises(ValueError):
        wrapper.reset_block(spec=spec)

    # Choose baseline params and alternate-condition params, convert to z-space.
    base_schema = base.param_schema
    params_A = {"alpha": 0.2, "beta": 4.0}
    params_B = {"alpha": 0.8, "beta": 2.0}

    z_A = base_schema.z_from_params(params_A)
    z_B = base_schema.z_from_params(params_B)
    delta_B = z_B - z_A

    wrapper_params = {
        "alpha__shared_z": float(z_A[0]),
        "beta__shared_z": float(z_A[1]),
        "alpha__delta_z__B": float(delta_B[0]),
        "beta__delta_z__B": float(delta_B[1]),
    }

    wrapper.set_params(wrapper_params)

    # Set condition A and apply.
    wrapper.set_condition("A")
    wrapper.reset_block(spec=spec)
    assert base.get_params()["alpha"] == pytest.approx(params_A["alpha"], abs=1e-10)
    assert base.get_params()["beta"] == pytest.approx(params_A["beta"], abs=1e-10)

    # Switch to condition B, base model params update.
    wrapper.set_condition("B")
    wrapper.reset_block(spec=spec)
    assert base.get_params()["alpha"] == pytest.approx(params_B["alpha"], abs=1e-10)
    assert base.get_params()["beta"] == pytest.approx(params_B["beta"], abs=1e-10)

    # Helper should derive the same constrained params without mutating.
    derived = constrained_params_by_condition_from_z(wrapper, wrapper_params)
    assert derived["A"]["alpha"] == pytest.approx(params_A["alpha"], abs=1e-10)
    assert derived["B"]["alpha"] == pytest.approx(params_B["alpha"], abs=1e-10)

    flat = flatten_params_by_condition(derived)
    assert flat["alpha__A"] == pytest.approx(params_A["alpha"], abs=1e-10)
    assert flat["alpha__B"] == pytest.approx(params_B["alpha"], abs=1e-10)



def test_within_subject_estimator_rejects_mismatched_condition_sets():
    # Create a minimal study where subject 1 has two conditions but subject 2 only has one.
    spec = EnvironmentSpec(
        n_actions=2,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=False,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )

    def block(block_id: str, condition: str) -> Block:
        # Minimal event log required by the MLE estimator stack.
        log = EventLog(
            events=[
                Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": condition}),
                Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={"choice": 0, "available_actions": [0, 1]}),
                Event(idx=2, type=EventType.OUTCOME, t=0, state=0, payload={"action": 0, "observed_outcome": 1.0, "info": {}}),
            ]
        )
        tr = Trial(t=0, state=0, choice=0, observed_outcome=1.0, outcome=1.0, available_actions=[0, 1])
        return Block(block_id=block_id, condition=condition, trials=[tr], env_spec=spec, event_log=log)

    subj1 = SubjectData(subject_id="s1", blocks=[block("bA", "A"), block("bB", "B")])
    subj2 = SubjectData(subject_id="s2", blocks=[block("bA2", "A")])

    study = StudyData(subjects=[subj1, subj2])

    est = WithinSubjectSharedDeltaTransformedMLEEstimator(
        base_model=QRL(),
        baseline_condition="A",
        # Let estimator infer conditions from data.
        n_starts=1,
        maxiter=5,
    )

    with pytest.raises(ValueError):
        est.fit(study=study, rng=np.random.default_rng(0))
