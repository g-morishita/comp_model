import numpy as np
import pytest

from comp_model_core.errors import CompatibilityError
from comp_model_core.spec import OutcomeObservationSpec, OutcomeObservationKind, TrialSpec
from comp_model_core.plans.block import BlockPlan

from comp_model_impl.bandits.bernoulli import BernoulliBanditEnv
from comp_model_impl.demonstrators.fixed_sequence import FixedSequenceDemonstrator
from comp_model_impl.tasks.block_runner_wrappers import BanditBlockRunner, SocialBanditBlockRunner
from comp_model_impl.tasks.build import build_runner_for_plan
from comp_model_impl.register import make_registry


def test_bandit_block_runner_enforces_available_actions():
    env = BernoulliBanditEnv(probs=[0.5, 0.5])
    ts = [
        TrialSpec(self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL), available_actions=[0]),
        TrialSpec(self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL), available_actions=[0]),
    ]
    runner = BanditBlockRunner(env=env, trial_specs=ts)
    rng = np.random.default_rng(0)
    runner.reset(rng=rng)

    with pytest.raises(CompatibilityError):
        runner.step(t=0, action=1, rng=rng)


def test_social_bandit_block_runner_observe_others_is_cached_and_visibility_respected():
    class CountingEnv(BernoulliBanditEnv):
        def __post_init__(self):
            super().__post_init__()
            self.step_calls = 0

        def step(self, *, action: int, rng: np.random.Generator):
            self.step_calls += 1
            return super().step(action=action, rng=rng)

    env = CountingEnv(probs=[0.0, 1.0])
    demo = FixedSequenceDemonstrator(actions=[1, 1, 1])

    ts = [
        # Demo outcome hidden
        TrialSpec(
            self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL),
            demo_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.HIDDEN),
            available_actions=[0, 1],
        ),
        # Demo outcome visible
        TrialSpec(
            self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL),
            demo_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL),
            available_actions=[0, 1],
        ),
    ]

    runner = SocialBanditBlockRunner(env=env, demonstrator=demo, trial_specs=ts)
    rng = np.random.default_rng(123)
    runner.reset(rng=rng)

    obs0_a = runner.observe_others(t=0, rng=rng)
    obs0_b = runner.observe_others(t=0, rng=rng)
    assert obs0_a is obs0_b  # cached object
    assert env.step_calls == 1  # only stepped once for observation
    assert obs0_a.others_choices == [1]
    assert obs0_a.observed_others_outcomes is None  # hidden

    obs1 = runner.observe_others(t=1, rng=rng)
    assert env.step_calls == 2
    assert obs1.observed_others_outcomes == [1.0]  # veridical, env arm 1 always returns 1


def test_build_runner_for_plan_injects_registry_model_for_rl_demonstrator():
    reg = make_registry()

    plan = BlockPlan(
        block_id="b1",
        n_trials=2,
        condition="c1",
        bandit_type="BernoulliBanditEnv",
        bandit_config={"probs": [0.2, 0.8]},
        trial_specs=[
            {
                "self_outcome": {"kind": "VERIDICAL"},
                "demo_outcome": {"kind": "VERIDICAL"},
                "available_actions": [0, 1],
            },
            {
                "self_outcome": {"kind": "VERIDICAL"},
                "demo_outcome": {"kind": "VERIDICAL"},
                "available_actions": [0, 1],
            },
        ],
        demonstrator_type="RLDemonstrator",
        demonstrator_config={
            "model": "QRL",
            "params": {"alpha": 0.2, "beta": 5.0},
        },
    )

    runner = build_runner_for_plan(plan=plan, registries=reg)
    assert isinstance(runner, SocialBanditBlockRunner)

    # The demonstrator config should have been resolved to a callable class.
    from comp_model_impl.demonstrators.rl_agent import RLDemonstrator
    from comp_model_impl.models.qrl.qrl import QRL

    assert isinstance(runner.demonstrator, RLDemonstrator)
    assert isinstance(runner.demonstrator.model, QRL)
    assert runner.demonstrator.model.alpha == pytest.approx(0.2)
    assert runner.demonstrator.model.beta == pytest.approx(5.0)


def test_build_runner_for_plan_asocial_returns_bandit_runner():
    plan = BlockPlan(
        block_id="b1",
        n_trials=1,
        condition="c1",
        bandit_type="BernoulliBanditEnv",
        bandit_config={"probs": [0.1, 0.9]},
        trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}],
    )
    runner = build_runner_for_plan(plan=plan, registries=make_registry())
    assert isinstance(runner, BanditBlockRunner)


def test_build_runner_for_plan_does_not_mutate_input_plan():
    reg = make_registry()
    plan = BlockPlan(
        block_id="b1",
        n_trials=1,
        condition="c1",
        bandit_type="BernoulliBanditEnv",
        bandit_config={"probs": [0.2, 0.8]},
        trial_specs=[
            {
                "self_outcome": {"kind": "VERIDICAL"},
                "demo_outcome": {"kind": "VERIDICAL"},
                "available_actions": [0, 1],
            },
        ],
        demonstrator_type="RLDemonstrator",
        demonstrator_config={
            "model": "QRL",
            "params": {"alpha": 0.2, "beta": 5.0},
        },
    )

    _ = build_runner_for_plan(plan=plan, registries=reg)
    assert plan.demonstrator_config["model"] == "QRL"


def test_build_runner_for_plan_missing_demonstrator_registry_raises():
    reg = make_registry()
    reg.demonstrators = None
    plan = BlockPlan(
        block_id="b1",
        n_trials=1,
        condition="c1",
        bandit_type="BernoulliBanditEnv",
        bandit_config={"probs": [0.2, 0.8]},
        trial_specs=[
            {
                "self_outcome": {"kind": "VERIDICAL"},
                "demo_outcome": {"kind": "VERIDICAL"},
                "available_actions": [0, 1],
            },
        ],
        demonstrator_type="RLDemonstrator",
        demonstrator_config={"model": "QRL", "params": {"alpha": 0.2, "beta": 5.0}},
    )

    with pytest.raises(ValueError):
        _ = build_runner_for_plan(plan=plan, registries=reg)
