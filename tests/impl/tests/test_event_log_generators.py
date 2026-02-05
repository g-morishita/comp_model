import numpy as np
import pytest

from comp_model_core.errors import CompatibilityError
from comp_model_core.events.types import Event, EventType
from comp_model_core.interfaces.block_runner import BlockRunner, SocialBlockRunner, StepResult, SocialObservation
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.params.schema import ParameterSchema, ParamDef
from comp_model_core.plans.block import BlockPlan
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind, parse_trial_specs_schedule
from comp_model_core.requirements import Requirement

from comp_model_impl.generators import (
    EventLogAsocialGenerator,
    EventLogSocialPreChoiceGenerator,
    EventLogSocialPostOutcomeGenerator,
)
from comp_model_impl.generators.event_log import (
    _build_event_log,
    _ensure_model_supports,
    _mask_and_renorm,
    _maybe_set_condition,
    _requirements_for_model,
)


class DummyRequirement(Requirement):
    def __init__(self, name: str = "dummy") -> None:
        super().__init__(name=name)

    def validate(self, *, plan, env_spec, trial_specs) -> None:
        return


class DummyModel(ComputationalModel):
    def __init__(self, probs=None, supports: bool = True) -> None:
        self._schema = ParameterSchema((ParamDef("alpha", 0.0),))
        self.probs = np.array([0.2, 0.8] if probs is None else probs, dtype=float)
        self.supports_flag = supports
        self.reset_count = 0
        self.update_calls = []
        self.condition = None

    @property
    def param_schema(self) -> ParameterSchema:
        return self._schema

    def supports(self, spec: EnvironmentSpec) -> bool:
        return bool(self.supports_flag)

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        self.reset_count += 1

    def action_probs(self, *, state, spec: EnvironmentSpec) -> np.ndarray:
        return self.probs.copy()

    def update(self, *, state, action: int, outcome, spec: EnvironmentSpec, info=None, rng=None) -> None:
        self.update_calls.append((state, action, outcome, info))

    def set_condition(self, condition: str) -> None:
        self.condition = str(condition)


class DummySocialModel(SocialComputationalModel):
    def __init__(self, probs=None, supports: bool = True) -> None:
        self._schema = ParameterSchema((ParamDef("alpha", 0.0),))
        self.probs = np.array([0.2, 0.8] if probs is None else probs, dtype=float)
        self.supports_flag = supports
        self.reset_count = 0
        self.update_calls = []
        self.social_calls = []
        self.condition = None

    @property
    def param_schema(self) -> ParameterSchema:
        return self._schema

    def supports(self, spec: EnvironmentSpec) -> bool:
        return bool(self.supports_flag)

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        self.reset_count += 1

    def action_probs(self, *, state, spec: EnvironmentSpec) -> np.ndarray:
        return self.probs.copy()

    def update(self, *, state, action: int, outcome, spec: EnvironmentSpec, info=None, rng=None) -> None:
        self.update_calls.append((state, action, outcome, info))

    def social_update(self, *, state, social: SocialObservation, spec: EnvironmentSpec, info=None, rng=None) -> None:
        self.social_calls.append((state, social, info))

    def set_condition(self, condition: str) -> None:
        self.condition = str(condition)


class DummyRunner(BlockRunner):
    def __init__(self, spec: EnvironmentSpec, trial_specs, step_info=None, social_obs=None) -> None:
        self._spec = spec
        self._trial_specs = list(trial_specs)
        self._state = 0
        self.step_info = step_info

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def reset(self, *, rng: np.random.Generator):
        self._state = 0
        return self._state

    def step(self, *, t: int, action: int, rng: np.random.Generator) -> StepResult:
        outcome = float(action)
        observed = float(action)
        return StepResult(outcome=outcome, observed_outcome=observed, done=False, info=self.step_info)

    def get_state(self):
        return self._state

    def trial_spec(self, *, t: int):
        return self._trial_specs[int(t)]


class DummySocialRunner(DummyRunner, SocialBlockRunner):
    def __init__(self, spec: EnvironmentSpec, trial_specs, step_info=None, social_obs=None) -> None:
        super().__init__(spec=spec, trial_specs=trial_specs, step_info=step_info, social_obs=social_obs)
        if social_obs is None:
            social_obs = SocialObservation(others_choices=[0], others_outcomes=[1.0], observed_others_outcomes=None, info=None)
        self._social_obs = social_obs

    def observe_others(self, *, t: int, rng: np.random.Generator) -> SocialObservation:
        return self._social_obs


def make_spec(*, is_social: bool, n_actions: int = 2) -> EnvironmentSpec:
    return EnvironmentSpec(
        n_actions=n_actions,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=bool(is_social),
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )


def make_plan(*, n_trials: int, is_social: bool, available_actions=None, metadata=None) -> BlockPlan:
    trial_specs = []
    for t in range(int(n_trials)):
        spec = {"self_outcome": {"kind": "VERIDICAL"}}
        if available_actions is not None:
            if isinstance(available_actions[0], (list, tuple)):
                spec["available_actions"] = list(available_actions[t])
            else:
                spec["available_actions"] = list(available_actions)
        if is_social:
            spec["demo_outcome"] = {"kind": "VERIDICAL"}
        trial_specs.append(spec)

    return BlockPlan(
        block_id="b1",
        n_trials=int(n_trials),
        condition="c1",
        bandit_type="DummyBandit",
        bandit_config={},
        trial_specs=trial_specs,
        demonstrator_type="demo" if is_social else None,
        demonstrator_config={} if is_social else None,
        metadata={} if metadata is None else dict(metadata),
    )


def build_runner(plan: BlockPlan, spec: EnvironmentSpec, runner_cls, *, step_info=None, social_obs=None):
    trial_specs = parse_trial_specs_schedule(
        n_trials=int(plan.n_trials),
        raw_trial_specs=plan.trial_specs,
        is_social=bool(spec.is_social),
    )
    return runner_cls(spec=spec, trial_specs=trial_specs, step_info=step_info, social_obs=social_obs)


def test_mask_and_renorm_variants():
    probs = np.array([0.2, 0.3, 0.5])

    out = _mask_and_renorm(probs, None)
    assert np.allclose(out, probs)

    out = _mask_and_renorm(probs, [0, 2])
    assert np.allclose(out, np.array([0.28571429, 0.0, 0.71428571]))

    out = _mask_and_renorm(np.array([0.2, 0.8]), [])
    assert np.allclose(out, np.array([0.0, 0.0]))


def test_helper_functions():
    spec = make_spec(is_social=False)
    plan = make_plan(n_trials=1, is_social=False)
    runner = build_runner(plan, spec, DummyRunner)
    model = DummyModel()

    _ensure_model_supports(model, runner)

    model.supports_flag = False
    with pytest.raises(CompatibilityError, match="not compatible"):
        _ensure_model_supports(model, runner)

    model = DummyModel()
    _maybe_set_condition(model, "cond-a")
    assert model.condition == "cond-a"

    no_cond = DummyModel()
    no_cond.set_condition = None  # type: ignore[assignment]
    _maybe_set_condition(no_cond, "cond-b")

    req = DummyRequirement("base")

    class BaseModel(DummyModel):
        @classmethod
        def requirements(cls):
            return (req,)

    class WrapperModel(DummyModel):
        def __init__(self, base):
            super().__init__()
            self.base_model = base

    base = BaseModel()
    wrapper = WrapperModel(base)
    assert _requirements_for_model(wrapper) == (req,)
    assert _requirements_for_model(DummyModel()) == ()
    req.validate(plan=plan, env_spec=spec, trial_specs=[])

    ev = Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={})
    log = _build_event_log([ev], metadata={"k": "v"})
    assert log.events == [ev]
    assert log.metadata == {"k": "v"}


def test_asocial_generator_builds_log_and_trials():
    rng = np.random.default_rng(0)
    spec = make_spec(is_social=False)
    plan = make_plan(n_trials=2, is_social=False, available_actions=[[0], [1]], metadata={"foo": "bar"})

    def builder(p):
        return build_runner(p, spec, DummyRunner, step_info=None)

    model = DummyModel(probs=np.array([0.2, 0.8]))
    gen = EventLogAsocialGenerator()

    subj = gen.simulate_subject(
        subject_id="s1",
        block_runner_builder=builder,
        model=model,
        params={"alpha": 0.1},
        block_plans=[plan],
        rng=rng,
    )

    assert subj.subject_id == "s1"
    assert len(subj.blocks) == 1
    block = subj.blocks[0]
    assert block.event_log is not None
    assert block.event_log.metadata["timing"] == "asocial"
    assert block.metadata["plan"]["foo"] == "bar"

    events = block.event_log.events
    assert [e.type for e in events] == [
        EventType.BLOCK_START,
        EventType.CHOICE,
        EventType.OUTCOME,
        EventType.CHOICE,
        EventType.OUTCOME,
    ]
    assert events[1].payload["available_actions"] == [0]
    assert events[3].payload["available_actions"] == [1]
    assert events[2].payload["info"] == {}

    assert block.trials[0].choice == 0
    assert block.trials[0].available_actions == [0]
    assert block.trials[1].choice == 1
    assert block.trials[1].available_actions == [1]
    assert model.condition == "c1"


def test_asocial_generator_rejects_social_spec():
    rng = np.random.default_rng(1)
    spec = make_spec(is_social=True)
    plan = make_plan(n_trials=1, is_social=True)

    def builder(p):
        return build_runner(p, spec, DummyRunner)

    model = DummyModel()

    with pytest.raises(CompatibilityError, match="cannot run a social task"):
        EventLogAsocialGenerator().simulate_subject(
            subject_id="s1",
            block_runner_builder=builder,
            model=model,
            params={"alpha": 0.1},
            block_plans=[plan],
            rng=rng,
        )


def test_social_pre_choice_happy_path():
    rng = np.random.default_rng(2)
    spec = make_spec(is_social=True)
    plan = make_plan(n_trials=1, is_social=True, metadata={"tag": "x"})
    social_obs = SocialObservation(
        others_choices=[1],
        others_outcomes=[1.0],
        observed_others_outcomes=None,
        info=None,
    )

    def builder(p):
        return build_runner(p, spec, DummySocialRunner, step_info={"ok": True}, social_obs=social_obs)

    model = DummySocialModel(probs=np.array([0.0, 1.0]))

    subj = EventLogSocialPreChoiceGenerator().simulate_subject(
        subject_id="s1",
        block_runner_builder=builder,
        model=model,
        params={"alpha": 0.1},
        block_plans=[plan],
        rng=rng,
    )

    block = subj.blocks[0]
    assert block.event_log is not None
    assert block.event_log.metadata["timing"] == "pre_choice"
    assert block.metadata["plan"]["tag"] == "x"

    events = block.event_log.events
    assert [e.type for e in events] == [
        EventType.BLOCK_START,
        EventType.SOCIAL_OBSERVED,
        EventType.CHOICE,
        EventType.OUTCOME,
    ]
    assert events[1].payload["observed_others_outcomes"] is None
    assert block.trials[0].others_choices == [1]
    assert model.social_calls


def test_social_pre_choice_requires_social_spec():
    rng = np.random.default_rng(3)
    spec = make_spec(is_social=False)
    plan = make_plan(n_trials=1, is_social=False)

    def builder(p):
        return build_runner(p, spec, DummyRunner)

    model = DummySocialModel()

    with pytest.raises(CompatibilityError, match="requires spec.is_social=True"):
        EventLogSocialPreChoiceGenerator().simulate_subject(
            subject_id="s1",
            block_runner_builder=builder,
            model=model,
            params={"alpha": 0.1},
            block_plans=[plan],
            rng=rng,
        )


def test_social_pre_choice_requires_social_runner():
    rng = np.random.default_rng(4)
    spec = make_spec(is_social=True)
    plan = make_plan(n_trials=1, is_social=True)

    def builder(p):
        return build_runner(p, spec, DummyRunner)

    model = DummySocialModel()

    with pytest.raises(CompatibilityError, match="SocialBlockRunner"):
        EventLogSocialPreChoiceGenerator().simulate_subject(
            subject_id="s1",
            block_runner_builder=builder,
            model=model,
            params={"alpha": 0.1},
            block_plans=[plan],
            rng=rng,
        )


def test_social_pre_choice_requires_social_model():
    rng = np.random.default_rng(5)
    spec = make_spec(is_social=True)
    plan = make_plan(n_trials=1, is_social=True)

    def builder(p):
        return build_runner(p, spec, DummySocialRunner)

    model = DummyModel()

    with pytest.raises(CompatibilityError, match="SocialComputationalModel"):
        EventLogSocialPreChoiceGenerator().simulate_subject(
            subject_id="s1",
            block_runner_builder=builder,
            model=model,
            params={"alpha": 0.1},
            block_plans=[plan],
            rng=rng,
        )


def test_social_post_outcome_happy_path():
    rng = np.random.default_rng(6)
    spec = make_spec(is_social=True)
    plan = make_plan(n_trials=1, is_social=True)
    social_obs = SocialObservation(
        others_choices=[0],
        others_outcomes=[0.0],
        observed_others_outcomes=[0.0],
        info={"src": "demo"},
    )

    def builder(p):
        return build_runner(p, spec, DummySocialRunner, step_info={"ok": True}, social_obs=social_obs)

    model = DummySocialModel(probs=np.array([1.0, 0.0]))

    subj = EventLogSocialPostOutcomeGenerator().simulate_subject(
        subject_id="s1",
        block_runner_builder=builder,
        model=model,
        params={"alpha": 0.1},
        block_plans=[plan],
        rng=rng,
    )

    block = subj.blocks[0]
    assert block.event_log is not None
    assert block.event_log.metadata["timing"] == "post_outcome"

    events = block.event_log.events
    assert [e.type for e in events] == [
        EventType.BLOCK_START,
        EventType.CHOICE,
        EventType.OUTCOME,
        EventType.SOCIAL_OBSERVED,
    ]
    assert events[-1].payload["observed_others_outcomes"] == [0.0]
    assert model.social_calls


def test_social_post_outcome_requires_social_spec():
    rng = np.random.default_rng(7)
    spec = make_spec(is_social=False)
    plan = make_plan(n_trials=1, is_social=False)

    def builder(p):
        return build_runner(p, spec, DummyRunner)

    model = DummySocialModel()

    with pytest.raises(CompatibilityError, match="requires spec.is_social=True"):
        EventLogSocialPostOutcomeGenerator().simulate_subject(
            subject_id="s1",
            block_runner_builder=builder,
            model=model,
            params={"alpha": 0.1},
            block_plans=[plan],
            rng=rng,
        )


def test_social_post_outcome_requires_social_runner():
    rng = np.random.default_rng(8)
    spec = make_spec(is_social=True)
    plan = make_plan(n_trials=1, is_social=True)

    def builder(p):
        return build_runner(p, spec, DummyRunner)

    model = DummySocialModel()

    with pytest.raises(CompatibilityError, match="SocialBlockRunner"):
        EventLogSocialPostOutcomeGenerator().simulate_subject(
            subject_id="s1",
            block_runner_builder=builder,
            model=model,
            params={"alpha": 0.1},
            block_plans=[plan],
            rng=rng,
        )


def test_social_post_outcome_requires_social_model():
    rng = np.random.default_rng(9)
    spec = make_spec(is_social=True)
    plan = make_plan(n_trials=1, is_social=True)

    def builder(p):
        return build_runner(p, spec, DummySocialRunner)

    model = DummyModel()

    with pytest.raises(CompatibilityError, match="SocialComputationalModel"):
        EventLogSocialPostOutcomeGenerator().simulate_subject(
            subject_id="s1",
            block_runner_builder=builder,
            model=model,
            params={"alpha": 0.1},
            block_plans=[plan],
            rng=rng,
        )


def test_generators_init_exports():
    import comp_model_impl.generators as gen

    assert gen.EventLogAsocialGenerator is EventLogAsocialGenerator
    assert gen.EventLogSocialPreChoiceGenerator is EventLogSocialPreChoiceGenerator
    assert gen.EventLogSocialPostOutcomeGenerator is EventLogSocialPostOutcomeGenerator
