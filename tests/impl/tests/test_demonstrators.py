import numpy as np
import pytest

from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
from comp_model_impl.demonstrators.noisy_best import NoisyBestArmDemonstrator
from comp_model_impl.demonstrators.fixed_sequence import FixedSequenceDemonstrator
from comp_model_impl.demonstrators.rl_agent import RLDemonstrator


def _spec(n_actions: int) -> EnvironmentSpec:
    return EnvironmentSpec(
        n_actions=int(n_actions),
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=False,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )


class StubModel:
    """A minimal model stub with deterministic probabilities and an update hook."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)
        self.reset_called = 0
        self.updated = []

    @property
    def param_schema(self):  # pragma: no cover - not needed in tests
        raise NotImplementedError

    def set_params(self, params, **kwargs):  # pragma: no cover
        return

    def reset_block(self, *, spec):
        self.reset_called += 1

    def action_probs(self, *, state, spec):
        # Return a proper probability vector.
        p = self._probs.copy()
        p = p / p.sum()
        return p

    def update(self, *, state, action, outcome, spec, info=None, rng=None):
        self.updated.append((int(action), None if outcome is None else float(outcome)))


def test_noisy_best_arm_demonstrator_extremes():
    spec = _spec(3)

    # p_best=1 -> always choose best
    demo = NoisyBestArmDemonstrator(reward_probs=[0.1, 0.9, 0.2], p_best=1.0)
    rng = np.random.default_rng(0)
    demo.reset(spec=spec, rng=rng)
    for _ in range(50):
        assert demo.act(state=0, spec=spec, rng=rng) == 1

    # p_best=0 -> never choose best
    demo2 = NoisyBestArmDemonstrator(reward_probs=[0.1, 0.9, 0.2], p_best=0.0)
    rng = np.random.default_rng(1)
    demo2.reset(spec=spec, rng=rng)
    for _ in range(50):
        assert demo2.act(state=0, spec=spec, rng=rng) in {0, 2}


def test_noisy_best_arm_demonstrator_from_config():
    spec = _spec(2)
    rng = np.random.default_rng(3)
    demo = NoisyBestArmDemonstrator.from_config(
        bandit_cfg={"probs": [0.3, 0.7]},
        demo_cfg={"p_best": 1.0},
    )
    assert list(demo.reward_probs) == [0.3, 0.7]
    assert demo.p_best == 1.0

    demo.reset(spec=spec, rng=rng)
    for _ in range(10):
        assert demo.act(state=0, spec=spec, rng=rng) == 1


def test_noisy_best_arm_demonstrator_pbest_zero_avoids_best():
    spec = _spec(3)
    rng = np.random.default_rng(4)
    demo = NoisyBestArmDemonstrator(reward_probs=[0.1, 0.9, 0.2], p_best=0.0)
    demo.reset(spec=spec, rng=rng)
    for _ in range(50):
        assert demo.act(state=0, spec=spec, rng=rng) in {0, 2}


def test_fixed_sequence_demonstrator_raises_when_exhausted():
    spec = _spec(2)
    rng = np.random.default_rng(0)

    demo = FixedSequenceDemonstrator(actions=[1, 0])
    demo.reset(spec=spec, rng=rng)
    assert demo.act(state=0, spec=spec, rng=rng) == 1
    assert demo.act(state=0, spec=spec, rng=rng) == 0

    with pytest.raises(ValueError):
        demo.act(state=0, spec=spec, rng=rng)


def test_fixed_sequence_demonstrator_from_config_and_reset():
    spec = _spec(2)
    rng = np.random.default_rng(1)

    demo = FixedSequenceDemonstrator.from_config(bandit_cfg={}, demo_cfg={"actions": [1, 0, 1]})
    demo.reset(spec=spec, rng=rng)
    assert demo.act(state=0, spec=spec, rng=rng) == 1


def test_fixed_sequence_demonstrator_emits_expected_sequence():
    spec = _spec(3)
    rng = np.random.default_rng(2)
    demo = FixedSequenceDemonstrator(actions=[2, 0, 1])
    demo.reset(spec=spec, rng=rng)
    seq = [demo.act(state=0, spec=spec, rng=rng) for _ in range(3)]
    assert seq == [2, 0, 1]

    with pytest.raises(ValueError):
        demo.act(state=0, spec=spec, rng=rng)
    demo.reset(spec=spec, rng=rng)
    assert demo.act(state=0, spec=spec, rng=rng) == 2


def test_rl_demonstrator_calls_model_and_samples_action():
    spec = _spec(2)
    rng = np.random.default_rng(42)

    # Always choose action 1.
    m = StubModel(probs=[0.0, 1.0])
    demo = RLDemonstrator(model=m)

    demo.reset(spec=spec, rng=rng)
    assert m.reset_called == 1

    a = demo.act(state=0, spec=spec, rng=rng)
    assert a == 1

    demo.update(state=0, action=a, outcome=1.0, spec=spec, rng=rng)
    assert m.updated == [(1, 1.0)]


def test_rl_demonstrator_updates_model_state():
    class UpdatingModel:
        def __init__(self):
            self.total = 0.0
            self.last = None
            self.reset_called = 0

        @property
        def param_schema(self):  # pragma: no cover - not needed in tests
            raise NotImplementedError

        def set_params(self, params, **kwargs):  # pragma: no cover
            return

        def reset_block(self, *, spec):
            self.reset_called += 1
            self.total = 0.0
            self.last = None

        def action_probs(self, *, state, spec):
            return np.array([0.5, 0.5], dtype=float)

        def update(self, *, state, action, outcome, spec, info=None, rng=None):
            self.total += float(outcome)
            self.last = (int(state), int(action), float(outcome))

    spec = _spec(2)
    rng = np.random.default_rng(7)
    model = UpdatingModel()
    demo = RLDemonstrator(model=model)

    demo.reset(spec=spec, rng=rng)
    assert model.reset_called == 1
    demo.update(state=1, action=0, outcome=0.25, spec=spec, rng=rng)
    demo.update(state=1, action=1, outcome=0.75, spec=spec, rng=rng)

    assert model.total == pytest.approx(1.0)
    assert model.last == (1, 1, 0.75)


def test_rl_demonstrator_from_config_sets_params_and_model():
    class ParamModel:
        def __init__(self):
            self.params = None
            self.reset_called = 0

        @property
        def param_schema(self):  # pragma: no cover - not needed in tests
            raise NotImplementedError

        def set_params(self, params, **kwargs):
            self.params = params

        def reset_block(self, *, spec):
            self.reset_called += 1

        def action_probs(self, *, state, spec):
            return np.array([0.5, 0.5], dtype=float)

        def update(self, *, state, action, outcome, spec, info=None, rng=None):
            return

    demo = RLDemonstrator.from_config(
        bandit_cfg={},
        demo_cfg={"model": ParamModel, "params": {"alpha": 0.2}},
    )
    assert isinstance(demo.model, ParamModel)
    assert demo.model.params == {"alpha": 0.2}
