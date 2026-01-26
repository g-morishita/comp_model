from dataclasses import FrozenInstanceError

import pytest

from comp_model_core.data.types import Trial, Block, SubjectData, StudyData
from comp_model_core.spec import EnvironmentSpec, OutcomeType


def test_data_types_constructible():
    t = Trial(t=0, state=0, choice=1, observed_outcome=1.0, outcome=1.0)
    spec = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, n_states=1)
    b = Block(block_id="b1", trials=[t], env_spec=spec)
    s = SubjectData(subject_id="S1", blocks=[b])
    study = StudyData(subjects=[s])

    assert study.subjects[0].subject_id == "S1"
    assert study.subjects[0].blocks[0].block_id == "b1"
    assert int(study.subjects[0].blocks[0].env_spec.n_actions) == 2


def test_trial_is_frozen():
    t = Trial(t=0, state=0, choice=1, observed_outcome=1.0, outcome=1.0)
    with pytest.raises(FrozenInstanceError):
        t.t = 999  # type: ignore[misc]
