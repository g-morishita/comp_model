import numpy as np
import pytest

from comp_model_core.spec import (
    EnvironmentSpec,
    OutcomeObservationKind,
    OutcomeObservationSpec,
    OutcomeType,
    StateKind,
    TrialSpec,
    parse_outcome_observation,
    parse_trial_spec_dict,
    parse_trial_specs_schedule,
)


def test_environment_spec_state_kind_invariants():
    # DISCRETE default is valid
    EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, state_kind=StateKind.DISCRETE, n_states=1)

    # NONE requires n_states/state_shape None
    EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, state_kind=StateKind.NONE, n_states=None, state_shape=None)
    with pytest.raises(ValueError):
        EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, state_kind=StateKind.NONE, n_states=1, state_shape=None)

    # DISCRETE requires n_states > 0 and state_shape None
    with pytest.raises(ValueError):
        EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, state_kind=StateKind.DISCRETE, n_states=0)
    with pytest.raises(ValueError):
        EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, state_kind=StateKind.DISCRETE, n_states=1, state_shape=(2,))

    # CONTINUOUS requires state_shape and n_states None
    EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, state_kind=StateKind.CONTINUOUS, n_states=None, state_shape=(3,))
    with pytest.raises(ValueError):
        EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, state_kind=StateKind.CONTINUOUS, n_states=1, state_shape=(3,))
    with pytest.raises(ValueError):
        EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, state_kind=StateKind.CONTINUOUS, n_states=None, state_shape=None)


def test_parse_outcome_observation_from_dict_and_case_insensitive():
    s = parse_outcome_observation({"kind": "gaussian", "sigma": 0.0})
    assert s.kind is OutcomeObservationKind.GAUSSIAN
    assert s.sigma == 0.0

    s2 = parse_outcome_observation({"kind": OutcomeObservationKind.HIDDEN})
    assert s2.kind is OutcomeObservationKind.HIDDEN

    with pytest.raises(TypeError):
        parse_outcome_observation(123)  # type: ignore[arg-type]


def test_outcome_observation_spec_validation_and_observe():
    env_bin = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY, outcome_range=(0.0, 1.0), outcome_is_bounded=True)
    env_cont = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.CONTINUOUS)

    # GAUSSIAN requires sigma >= 0
    with pytest.raises(ValueError):
        OutcomeObservationSpec(kind=OutcomeObservationKind.GAUSSIAN, sigma=-0.1)

    # FLIP requires flip_p in [0,1]
    with pytest.raises(ValueError):
        OutcomeObservationSpec(kind=OutcomeObservationKind.FLIP, flip_p=1.5)

    # Hidden returns None
    rng = np.random.default_rng(0)
    hidden = OutcomeObservationSpec(kind=OutcomeObservationKind.HIDDEN)
    assert hidden.observe(true_outcome=1.0, env=env_bin, rng=rng) is None

    # Veridical returns the true outcome
    rng = np.random.default_rng(0)
    ver = OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL)
    assert ver.observe(true_outcome=0.25, env=env_cont, rng=rng) == 0.25

    # Gaussian sigma=0 is deterministic (and can clip)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    g = OutcomeObservationSpec(kind=OutcomeObservationKind.GAUSSIAN, sigma=0.7, clip_to_range=True)
    obs = g.observe(true_outcome=0.9, env=env_bin, rng=rng1)
    expected = 0.9 + float(rng2.normal(0.0, 0.7))
    expected = float(np.clip(expected, 0.0, 1.0))
    assert obs == expected

    # Flip requires binary env and flips deterministically when flip_p=1
    rng = np.random.default_rng(0)
    flip = OutcomeObservationSpec(kind=OutcomeObservationKind.FLIP, flip_p=1.0)
    assert flip.observe(true_outcome=1.0, env=env_bin, rng=rng) == 0.0
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        flip.observe(true_outcome=1.0, env=env_cont, rng=rng)


def test_parse_trial_spec_dict_social_and_asocial_rules():
    asocial = {
        "self_outcome": {"kind": "VERIDICAL"},
        "available_actions": [0, 1, 1],
        "metadata": {"k": "v"},
    }
    ts = parse_trial_spec_dict(asocial, is_social=False, trial_index=0)
    assert isinstance(ts, TrialSpec)
    assert ts.demo_outcome is None
    assert ts.available_actions == (0, 1)  # deduped

    with pytest.raises(ValueError):
        parse_trial_spec_dict({"available_actions": [0, 1]}, is_social=False, trial_index=0)

    with pytest.raises(ValueError):
        parse_trial_spec_dict({"self_outcome": {"kind": "VERIDICAL"}, "demo_outcome": {"kind": "HIDDEN"}}, is_social=False, trial_index=0)

    with pytest.raises(TypeError):
        parse_trial_spec_dict({"self_outcome": {"kind": "VERIDICAL"}, "metadata": 123}, is_social=False, trial_index=0)

    with pytest.raises(ValueError):
        parse_trial_spec_dict({"self_outcome": {"kind": "VERIDICAL"}, "available_actions": []}, is_social=False, trial_index=0)

    social = {
        "self_outcome": {"kind": "VERIDICAL"},
        "demo_outcome": {"kind": "HIDDEN"},
        "available_actions": [0, 1],
    }
    ts2 = parse_trial_spec_dict(social, is_social=True, trial_index=0)
    assert ts2.demo_outcome is not None
    assert ts2.demo_outcome.kind is OutcomeObservationKind.HIDDEN

    with pytest.raises(ValueError):
        parse_trial_spec_dict({"self_outcome": {"kind": "VERIDICAL"}}, is_social=True, trial_index=0)


def test_parse_trial_specs_schedule_enforces_length_and_social_constraints():
    raw = [{"self_outcome": {"kind": "VERIDICAL"}} for _ in range(3)]
    out = parse_trial_specs_schedule(n_trials=3, raw_trial_specs=raw, is_social=False)
    assert len(out) == 3

    with pytest.raises(ValueError):
        parse_trial_specs_schedule(n_trials=2, raw_trial_specs=raw, is_social=False)

    # Social schedule must include demo_outcome
    raw_soc = [{"self_outcome": {"kind": "VERIDICAL"}, "demo_outcome": {"kind": "HIDDEN"}} for _ in range(3)]
    out2 = parse_trial_specs_schedule(n_trials=3, raw_trial_specs=raw_soc, is_social=True)
    assert all(ts.demo_outcome is not None for ts in out2)
