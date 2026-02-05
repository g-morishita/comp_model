import copy

import numpy as np
import pytest

from comp_model_core.plans.block import BlockPlan
from comp_model_core.data.types import Block, SubjectData, Trial
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
from comp_model_core.events.types import Event, EventLog, EventType

from comp_model_impl.generators.event_log import EventLogAsocialGenerator
from comp_model_impl.likelihood.event_log_replay import loglike_subject
from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.register import make_registry
from comp_model_impl.tasks.build import build_runner_for_plan


def test_event_log_generator_replay_loglike_matches_manual_replay():
    rng = np.random.default_rng(0)
    reg = make_registry()

    plan = BlockPlan(
        block_id="b1",
        n_trials=6,
        condition="c1",
        bandit_type="BernoulliBanditEnv",
        bandit_config={"probs": [0.2, 0.8]},
        trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}] * 6,
    )

    def builder(p):
        return build_runner_for_plan(plan=p, registries=reg)

    model = QRL(alpha=0.35, beta=3.0)
    params = {"alpha": 0.35, "beta": 3.0}

    subj = EventLogAsocialGenerator().simulate_subject(
        subject_id="s1",
        block_runner_builder=builder,
        model=model,
        params=params,
        block_plans=[plan],
        rng=rng,
    )

    ll_replay = loglike_subject(subject=subj, model=model, params=params)

    # Manual replay (should match exactly, because likelihood code is deterministic given the log).
    m = copy.deepcopy(model)
    m.set_params(params)

    ll_manual = 0.0
    for block in subj.blocks:
        spec = block.env_spec
        assert spec is not None
        log = block.event_log
        assert log is not None
        for e in log.events:
            if e.type is EventType.BLOCK_START:
                m.reset_block(spec=spec)
            elif e.type is EventType.CHOICE:
                choice = int(e.payload["choice"])
                probs = m.action_probs(state=e.state, spec=spec)
                p = float(probs[choice])
                ll_manual += float(np.log(p))
            elif e.type is EventType.OUTCOME:
                m.update(
                    state=e.state,
                    action=int(e.payload["action"]),
                    outcome=e.payload.get("observed_outcome", None),
                    spec=spec,
                    info=e.payload.get("info", None),
                )
            else:
                raise AssertionError(f"unexpected event {e.type}")

    assert ll_replay == pytest.approx(ll_manual, rel=0, abs=1e-12)


def test_loglike_subject_masks_available_actions_and_does_not_mutate_model():
    # Build a minimal subject/block with a constrained action set.
    spec = EnvironmentSpec(
        n_actions=3,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=False,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )

    log = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "c1"}),
            Event(
                idx=1,
                type=EventType.CHOICE,
                t=0,
                state=0,
                payload={"choice": 1, "available_actions": [1, 2]},
            ),
            Event(
                idx=2,
                type=EventType.OUTCOME,
                t=0,
                state=0,
                payload={"action": 1, "observed_outcome": 1.0, "info": {}},
            ),
        ],
        metadata={"test": True},
    )

    block = Block(block_id="b1", condition="c1", trials=[Trial(t=0, state=0, choice=1, observed_outcome=1.0, outcome=1.0)], env_spec=spec, event_log=log)
    subj = SubjectData(subject_id="s1", blocks=[block])

    model = QRL(alpha=0.5, beta=1.0)
    params = {"alpha": 0.5, "beta": 1.0}

    # Put some state in the original model to ensure it doesn't change.
    model.set_params({"alpha": 0.1, "beta": 2.0})
    model.reset_block(spec=spec)
    _ = model.action_probs(state=0, spec=spec)

    ll = loglike_subject(subject=subj, model=model, params=params)

    # With equal Q-values initially, probs are uniform = 1/3. Masking [1,2] -> prob(choice=1)=0.5
    assert ll == pytest.approx(np.log(0.5), abs=1e-12)

    # Original model parameters/state were not mutated by loglike_subject.
    assert model.alpha == pytest.approx(0.1)
    assert model.beta == pytest.approx(2.0)
