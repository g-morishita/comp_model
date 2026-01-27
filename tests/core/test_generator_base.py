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
        block_runner_builder,
        model,
        params,
        block_plans,
        rng,
    ) -> SubjectData:
        blocks = []
        for bp in block_plans:
            blocks.append(
                Block(
                    block_id=bp.block_id,
                    condition="c",
                    trials=[Trial(t=0, state=None, choice=None, observed_outcome=None, outcome=None)],
                    env_spec=None,
                )
            )
        return SubjectData(subject_id=subject_id, blocks=blocks)


def test_simulate_study_happy_path():
    g = DummyGenerator()
    rng = np.random.default_rng(0)

    block_runner_builder = lambda bp: None  # type: ignore[return-value]
    model = object()

    ts = [{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}]

    plans = {
        "S1": [BlockPlan(block_id="b1", condition="c", n_trials=1, bandit_type="x", bandit_config={}, trial_specs=ts)],
        "S2": [BlockPlan(block_id="b2", condition="c", n_trials=1, bandit_type="x", bandit_config={}, trial_specs=ts)],
    }
    subj_params = {
        "S1": {"a": 1.0},
        "S2": {"a": 2.0},
    }

    study: StudyData = g.simulate_study(
        block_runner_builder=block_runner_builder,
        model=model,  # type: ignore[arg-type]
        subj_params=subj_params,
        subject_block_plans=plans,
        rng=rng,
    )
    assert len(study.subjects) == 2


def test_simulate_study_missing_subj_params_raises():
    g = DummyGenerator()
    rng = np.random.default_rng(0)
    block_runner_builder = lambda bp: None  # type: ignore[return-value]
    model = object()

    ts = [{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}]

    plans = {"S1": [BlockPlan(block_id="b1", condition="c", n_trials=1, bandit_type="x", bandit_config={}, trial_specs=ts)]}
    subj_params = {}  # missing

    with pytest.raises(ValueError):
        g.simulate_study(
            block_runner_builder=block_runner_builder,
            model=model,  # type: ignore[arg-type]
            subj_params=subj_params,
            subject_block_plans=plans,
            rng=rng,
        )
