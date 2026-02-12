"""Behavioral tests for core model implementations."""

from __future__ import annotations

import numpy as np

from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind

from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.models.vs.vs import VS
from comp_model_impl.models.vicarious_rl.vicarious_rl import Vicarious_RL
from comp_model_impl.models.vicarious_rl_stay.vicarious_rl_stay import Vicarious_RL_Stay
from comp_model_impl.models.vicarious_vs.vicarious_vs import Vicarious_VS


def _spec(*, is_social: bool) -> EnvironmentSpec:
    return EnvironmentSpec(
        n_actions=2,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=is_social,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )


def _softmax(u: np.ndarray, beta: float) -> np.ndarray:
    z = beta * np.asarray(u, dtype=float)
    z = z - float(np.max(z))
    expz = np.exp(z)
    return expz / float(np.sum(expz))


def test_qrl_updates_chosen_action_and_softmax() -> None:
    spec = _spec(is_social=False)
    model = QRL(alpha=0.5, beta=1.0)
    model.reset_block(spec=spec)

    probs0 = model.action_probs(state=0, spec=spec)
    assert np.allclose(probs0, np.array([0.5, 0.5]))

    model.update(state=0, action=1, outcome=1.0, spec=spec)
    probs1 = model.action_probs(state=0, spec=spec)
    expected = _softmax(np.array([0.0, 0.5]), beta=1.0)
    assert np.allclose(probs1, expected)

    model.update(state=0, action=0, outcome=None, spec=spec)
    probs2 = model.action_probs(state=0, spec=spec)
    assert np.allclose(probs2, expected)


def test_vs_social_and_private_updates_with_perseveration() -> None:
    spec = _spec(is_social=True)
    model = VS(alpha_p=0.5, alpha_i=0.5, beta=1.0, kappa=0.5, pseudo_reward=1.0)
    model.reset_block(spec=spec)

    model.social_update(state=0, social=SocialObservation(others_choices=[1]), spec=spec)
    probs_after_social = model.action_probs(state=0, spec=spec)
    expected_social = _softmax(np.array([0.0, 0.5]), beta=1.0)
    assert np.allclose(probs_after_social, expected_social)

    model.update(state=0, action=0, outcome=1.0, spec=spec)
    probs_after_private = model.action_probs(state=0, spec=spec)
    expected_private = _softmax(np.array([1.0, 0.5]), beta=1.0)
    assert np.allclose(probs_after_private, expected_private)


def test_vicarious_rl_social_update_and_no_private_update() -> None:
    spec = _spec(is_social=True)
    model = Vicarious_RL(alpha_o=0.5, beta=1.0)
    model.reset_block(spec=spec)

    model.social_update(
        state=0,
        social=SocialObservation(others_choices=[1], observed_others_outcomes=[1.0]),
        spec=spec,
    )
    probs_after_social = model.action_probs(state=0, spec=spec)
    expected = _softmax(np.array([0.0, 0.5]), beta=1.0)
    assert np.allclose(probs_after_social, expected)

    model.update(state=0, action=1, outcome=0.0, spec=spec)
    probs_after_private = model.action_probs(state=0, spec=spec)
    assert np.allclose(probs_after_private, expected)

    model.reset_block(spec=spec)
    probs_uniform = model.action_probs(state=0, spec=spec)
    assert np.allclose(probs_uniform, np.array([0.5, 0.5]))

    model.social_update(state=0, social=SocialObservation(others_choices=[1]), spec=spec)
    probs_no_outcome = model.action_probs(state=0, spec=spec)
    assert np.allclose(probs_no_outcome, np.array([0.5, 0.5]))


def test_vicarious_rl_stay_tracks_private_action_for_perseveration() -> None:
    spec = _spec(is_social=True)
    model = Vicarious_RL_Stay(alpha_o=0.0, beta=1.0, kappa=1.0)
    model.reset_block(spec=spec)

    probs0 = model.action_probs(state=0, spec=spec)
    assert np.allclose(probs0, np.array([0.5, 0.5]))

    model.update(state=0, action=1, outcome=None, spec=spec)
    probs1 = model.action_probs(state=0, spec=spec)
    assert probs1[1] > probs1[0]


def test_vicarious_vs_social_update_combines_pseudo_and_outcome() -> None:
    spec = _spec(is_social=True)
    model = Vicarious_VS(alpha_o=0.5, alpha_a=0.2, beta=1.0, pseudo_reward=1.0)
    model.reset_block(spec=spec)

    model.social_update(
        state=0,
        social=SocialObservation(others_choices=[1], observed_others_outcomes=[0.0]),
        spec=spec,
    )
    probs_after_social = model.action_probs(state=0, spec=spec)
    expected = _softmax(np.array([0.0, 0.1]), beta=1.0)
    assert np.allclose(probs_after_social, expected)

    model.update(state=0, action=1, outcome=1.0, spec=spec)
    probs_after_private = model.action_probs(state=0, spec=spec)
    assert np.allclose(probs_after_private, expected)


def test_model_supports_flags() -> None:
    social_spec = _spec(is_social=True)
    asocial_spec = _spec(is_social=False)

    assert QRL().supports(asocial_spec)
    assert not QRL().supports(social_spec)

    assert VS().supports(social_spec)
    assert not VS().supports(asocial_spec)

    assert Vicarious_RL().supports(social_spec)
    assert not Vicarious_RL().supports(asocial_spec)
    assert Vicarious_RL_Stay().supports(social_spec)
    assert not Vicarious_RL_Stay().supports(asocial_spec)

    assert Vicarious_VS().supports(social_spec)
    assert not Vicarious_VS().supports(asocial_spec)
