import numpy as np
import pytest

from comp_model_core.data.types import SubjectData, Block, Trial, StudyData
from comp_model_core.interfaces.generator import Generator
from comp_model_core.plans.block import BlockPlan


class DummyGenerator(Generator):
    def simulate_subject(
        self,
        *,
        subject_id: str,
        task_builder,
        model,
        params,
        block_plans,
        rng,
    ) -> SubjectData:
        # minimal subject with empty block(s)
        blocks = []
        for bp in block_plans:
            blocks.append(Block(block_id=bp.block_id, trials=[Trial(t=0, state=None, choice=None, outcome=None)]))
        return SubjectData(subject_id=subject_id, blocks=blocks)


def test_simulate_study_happy_path():
    g = DummyGenerator()
    rng = np.random.default_rng(0)

    # we don't need a real task_builder or model for this base-method test
    task_builder = lambda bp: None  # type: ignore[return-value]
    model = object()

    plans = {
        "S1": [BlockPlan(block_id="b1", n_trials=1, bandit_type="x", bandit_config={})],
        "S2": [BlockPlan(block_id="b2", n_trials=1, bandit_type="x", bandit_config={})],
    }
    subj_params = {
        "S1": {"a": 1.0},
        "S2": {"a": 2.0},
    }

    study: StudyData = g.simulate_study(
        task_builder=task_builder,
        model=model,  # type: ignore[arg-type]
        subj_params=subj_params,
        subject_block_plans=plans,
        rng=rng,
    )
    assert len(study.subjects) == 2


def test_simulate_study_missing_subj_params_raises():
    g = DummyGenerator()
    rng = np.random.default_rng(0)
    task_builder = lambda bp: None  # type: ignore[return-value]
    model = object()

    plans = {"S1": [BlockPlan(block_id="b1", n_trials=1, bandit_type="x", bandit_config={})]}
    subj_params = {}  # missing

    with pytest.raises(ValueError):
        g.simulate_study(
            task_builder=task_builder,
            model=model,  # type: ignore[arg-type]
            subj_params=subj_params,
            subject_block_plans=plans,
            rng=rng,
        )
