import numpy as np
import pytest

from comp_model_core.errors import CompatibilityError
from comp_model_core.plans.block import BlockPlan
from comp_model_core.requirements import (
    PredicateRequirement,
    RequireAllDemoOutcomesHidden,
    RequireAllSelfOutcomesHidden,
    RequireAnyDemoOutcomeObservable,
    RequireAnySelfOutcomeObservable,
    RequireAsocialBlock,
    RequireSocialBlock,
)
from comp_model_core.spec import EnvironmentSpec, OutcomeType
from comp_model_core.validation import validate_action_sets, validate_block_plan


def _mk_plan(*, block_id: str, condition: str, n_trials: int, trial_specs: list[dict]) -> BlockPlan:
    return BlockPlan(
        block_id=block_id,
        condition=condition,
        n_trials=n_trials,
        bandit_type="x",
        bandit_config={},
        trial_specs=trial_specs,
    )


def test_validate_action_sets_in_range():
    env = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY)
    # TrialSpec dicts parsed later; validate_action_sets operates on TrialSpec objects,
    # so we call validate_block_plan which parses and validates.
    plan = _mk_plan(
        block_id="b1",
        condition="c",
        n_trials=2,
        trial_specs=[
            {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
            {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0]},
        ],
    )
    validate_block_plan(plan=plan, env_spec=env, requirements=())

    bad = _mk_plan(
        block_id="b2",
        condition="c",
        n_trials=1,
        trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [2]}],
    )
    with pytest.raises(CompatibilityError):
        validate_block_plan(plan=bad, env_spec=env, requirements=())


def test_requirements_social_asocial_and_visibility():
    env_asocial = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, is_social=False)
    env_social = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, is_social=True)

    asocial_specs = [{"self_outcome": {"kind": "VERIDICAL"}}]
    social_specs = [{"self_outcome": {"kind": "VERIDICAL"}, "demo_outcome": {"kind": "HIDDEN"}}]

    plan_a = _mk_plan(block_id="a", condition="c", n_trials=1, trial_specs=asocial_specs)
    plan_s = _mk_plan(block_id="s", condition="c", n_trials=1, trial_specs=social_specs)

    # Social/asocial requirements
    validate_block_plan(plan=plan_a, env_spec=env_asocial, requirements=(RequireAsocialBlock(),))
    with pytest.raises(CompatibilityError):
        validate_block_plan(plan=plan_a, env_spec=env_asocial, requirements=(RequireSocialBlock(),))
    validate_block_plan(plan=plan_s, env_spec=env_social, requirements=(RequireSocialBlock(),))
    with pytest.raises(CompatibilityError):
        validate_block_plan(plan=plan_s, env_spec=env_social, requirements=(RequireAsocialBlock(),))

    # Any-self-observable fails if all hidden
    plan_hidden = _mk_plan(block_id="h", condition="c", n_trials=2, trial_specs=[{"self_outcome": {"kind": "HIDDEN"}}] * 2)
    with pytest.raises(CompatibilityError):
        validate_block_plan(plan=plan_hidden, env_spec=env_asocial, requirements=(RequireAnySelfOutcomeObservable(),))

    # All-self-hidden fails if any observable
    with pytest.raises(CompatibilityError):
        validate_block_plan(plan=plan_a, env_spec=env_asocial, requirements=(RequireAllSelfOutcomesHidden(),))
    validate_block_plan(plan=plan_hidden, env_spec=env_asocial, requirements=(RequireAllSelfOutcomesHidden(),))

    # Demo visibility requirements
    # Any-demo-observable fails here because demo is hidden
    with pytest.raises(CompatibilityError):
        validate_block_plan(plan=plan_s, env_spec=env_social, requirements=(RequireAnyDemoOutcomeObservable(),))

    # All-demo-hidden passes
    validate_block_plan(plan=plan_s, env_spec=env_social, requirements=(RequireAllDemoOutcomesHidden(),))

    # If any demo observable -> all-demo-hidden fails, any-demo-observable passes
    plan_demo_vis = _mk_plan(
        block_id="dv",
        condition="c",
        n_trials=2,
        trial_specs=[
            {"self_outcome": {"kind": "VERIDICAL"}, "demo_outcome": {"kind": "HIDDEN"}},
            {"self_outcome": {"kind": "VERIDICAL"}, "demo_outcome": {"kind": "VERIDICAL"}},
        ],
    )
    validate_block_plan(plan=plan_demo_vis, env_spec=env_social, requirements=(RequireAnyDemoOutcomeObservable(),))
    with pytest.raises(CompatibilityError):
        validate_block_plan(plan=plan_demo_vis, env_spec=env_social, requirements=(RequireAllDemoOutcomesHidden(),))


def test_predicate_requirement_error_wrapped():
    env = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY)
    plan = _mk_plan(block_id="b", condition="c", n_trials=1, trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}}])

    def boom(_plan, _env, _ts):
        raise RuntimeError("boom")

    req = PredicateRequirement(name="boom", predicate=boom, message="should not reach")
    with pytest.raises(CompatibilityError) as ei:
        validate_block_plan(plan=plan, env_spec=env, requirements=(req,))
    assert "requirement 'boom'" in str(ei.value)


def test_validate_action_sets_direct_on_trialspecs_smoke():
    # Directly cover validate_action_sets() behavior on TrialSpec objects.
    from comp_model_core.spec import TrialSpec, OutcomeObservationSpec, OutcomeObservationKind

    env = EnvironmentSpec(n_actions=3, outcome_type=OutcomeType.BINARY)
    ts = [
        TrialSpec(self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL), available_actions=(0, 2)),
        TrialSpec(self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL), available_actions=None),
    ]
    validate_action_sets(trial_specs=ts, env_spec=env, block_id="b")

    ts_bad = [TrialSpec(self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL), available_actions=(3,))]
    with pytest.raises(CompatibilityError):
        validate_action_sets(trial_specs=ts_bad, env_spec=env, block_id="b")
