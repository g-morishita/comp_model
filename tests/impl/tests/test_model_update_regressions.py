"""Regression tests for update behavior across model implementations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind

from comp_model_impl.models import (
    AP_RL_NoStay,
    AP_RL_Stay,
    QRL,
    VS,
    UnidentifiableQRL,
    Vicarious_AP_DB_STAY,
    Vicarious_AP_VS,
    Vicarious_DB_Stay,
    Vicarious_Dir_DB_Stay,
    Vicarious_RL,
    Vicarious_RL_Stay,
    Vicarious_VS,
    Vicarious_VS_Stay,
)


ModelFactory = Callable[[], object]


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


def _assert_valid_probs(probs: np.ndarray, *, n_actions: int = 2) -> None:
    assert probs.shape == (n_actions,)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-12)


def _social_obs(*, choice: int = 1, outcome: float = 1.0) -> SocialObservation:
    return SocialObservation(
        others_choices=[choice],
        observed_others_outcomes=[outcome],
    )


def _all_model_cases() -> list[tuple[str, ModelFactory, bool]]:
    cases: list[tuple[str, ModelFactory, bool]] = [
        ("QRL", lambda: QRL(alpha=0.5, beta=1.0), False),
        ("UnidentifiableQRL", lambda: UnidentifiableQRL(alpha_1=0.2, alpha_2=0.3, beta=1.0), False),
        ("VS", lambda: VS(alpha_p=0.5, alpha_i=0.3, beta=1.0, kappa=0.5), True),
        ("Vicarious_RL", lambda: Vicarious_RL(alpha_o=0.5, beta=1.0), True),
        ("Vicarious_RL_Stay", lambda: Vicarious_RL_Stay(alpha_o=0.5, beta=1.0, kappa=0.5), True),
        ("AP_RL_NoStay", lambda: AP_RL_NoStay(alpha_a=0.5, beta=1.0), True),
        ("AP_RL_Stay", lambda: AP_RL_Stay(alpha_a=0.5, beta=1.0, kappa=0.5), True),
        ("Vicarious_VS", lambda: Vicarious_VS(alpha_o=0.5, alpha_a=0.3, beta=1.0), True),
        ("Vicarious_VS_Stay", lambda: Vicarious_VS_Stay(alpha_o=0.5, alpha_a=0.3, beta=1.0, kappa=0.5), True),
        ("Vicarious_AP_VS", lambda: Vicarious_AP_VS(alpha_o=0.5, alpha_vs_base=0.3, alpha_a=0.2, beta=1.0, kappa=0.5), True),
        ("Vicarious_DB_Stay", lambda: Vicarious_DB_Stay(alpha_o=0.5, demo_bias=0.3, beta=1.0, kappa=0.5), True),
        (
            "Vicarious_AP_DB_STAY",
            lambda: Vicarious_AP_DB_STAY(alpha_o=0.5, alpha_a=0.2, demo_bias_rel=0.3, beta=1.0, kappa=0.5),
            True,
        ),
        (
            "Vicarious_Dir_DB_Stay",
            lambda: Vicarious_Dir_DB_Stay(
                alpha_o=0.5,
                demo_bias_rel=0.3,
                beta=1.0,
                kappa=0.5,
                demo_dirichlet_prior=1.0,
            ),
            True,
        ),
    ]
    return cases


@pytest.mark.parametrize(("name", "factory", "is_social"), _all_model_cases())
def test_models_run_update_cycle_with_valid_probabilities(
    name: str,
    factory: ModelFactory,
    is_social: bool,
) -> None:
    model = factory()
    spec = _spec(is_social=is_social)
    assert model.supports(spec), f"{name} should support this test spec"

    model.reset_block(spec=spec)
    probs_0 = model.action_probs(state=0, spec=spec)
    _assert_valid_probs(np.asarray(probs_0, dtype=float))

    if is_social:
        model.social_update(state=0, social=_social_obs(choice=1, outcome=1.0), spec=spec)
        probs_social = model.action_probs(state=0, spec=spec)
        _assert_valid_probs(np.asarray(probs_social, dtype=float))

    model.update(state=0, action=0, outcome=1.0, spec=spec)
    probs_update = model.action_probs(state=0, spec=spec)
    _assert_valid_probs(np.asarray(probs_update, dtype=float))


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("QRL", lambda: QRL(alpha=0.5, beta=1.0)),
        ("UnidentifiableQRL", lambda: UnidentifiableQRL(alpha_1=0.2, alpha_2=0.3, beta=1.0)),
    ],
)
def test_asocial_private_update_increases_chosen_action_preference(
    name: str,
    factory: ModelFactory,
) -> None:
    model = factory()
    spec = _spec(is_social=False)
    assert model.supports(spec), f"{name} should support asocial spec"

    model.reset_block(spec=spec)
    probs_0 = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)
    model.update(state=0, action=1, outcome=1.0, spec=spec)
    probs_1 = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)

    assert probs_1[1] > probs_0[1]


def _social_learning_cases() -> list[tuple[str, ModelFactory]]:
    cases: list[tuple[str, ModelFactory]] = [
        ("VS", lambda: VS(alpha_p=0.0, alpha_i=0.5, beta=1.0, kappa=0.0)),
        ("Vicarious_RL", lambda: Vicarious_RL(alpha_o=0.5, beta=1.0)),
        ("Vicarious_RL_Stay", lambda: Vicarious_RL_Stay(alpha_o=0.5, beta=1.0, kappa=0.0)),
        ("AP_RL_NoStay", lambda: AP_RL_NoStay(alpha_a=0.5, beta=1.0)),
        ("AP_RL_Stay", lambda: AP_RL_Stay(alpha_a=0.5, beta=1.0, kappa=0.0)),
        ("Vicarious_VS", lambda: Vicarious_VS(alpha_o=0.5, alpha_a=0.0, beta=1.0)),
        ("Vicarious_VS_Stay", lambda: Vicarious_VS_Stay(alpha_o=0.5, alpha_a=0.0, beta=1.0, kappa=0.0)),
        ("Vicarious_AP_VS", lambda: Vicarious_AP_VS(alpha_o=0.5, alpha_vs_base=0.0, alpha_a=0.0, beta=1.0, kappa=0.0)),
        ("Vicarious_DB_Stay", lambda: Vicarious_DB_Stay(alpha_o=0.5, demo_bias=0.0, beta=1.0, kappa=0.0)),
        (
            "Vicarious_AP_DB_STAY",
            lambda: Vicarious_AP_DB_STAY(alpha_o=0.5, alpha_a=0.0, demo_bias_rel=0.0, beta=1.0, kappa=0.0),
        ),
        (
            "Vicarious_Dir_DB_Stay",
            lambda: Vicarious_Dir_DB_Stay(alpha_o=0.5, demo_bias_rel=0.0, beta=1.0, kappa=0.0, demo_dirichlet_prior=1.0),
        ),
    ]
    return cases


@pytest.mark.parametrize(("name", "factory"), _social_learning_cases())
def test_social_update_increases_preference_for_observed_demo_action(
    name: str,
    factory: ModelFactory,
) -> None:
    model = factory()
    spec = _spec(is_social=True)
    assert model.supports(spec), f"{name} should support social spec"

    model.reset_block(spec=spec)
    probs_0 = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)
    model.social_update(state=0, social=_social_obs(choice=1, outcome=1.0), spec=spec)
    probs_1 = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)

    assert probs_1[1] > probs_0[1]


def _stay_regression_cases() -> list[tuple[str, ModelFactory]]:
    cases: list[tuple[str, ModelFactory]] = [
        ("VS", lambda: VS(alpha_p=0.0, alpha_i=0.0, beta=1.0, kappa=2.0)),
        ("Vicarious_RL_Stay", lambda: Vicarious_RL_Stay(alpha_o=0.0, beta=1.0, kappa=2.0)),
        ("AP_RL_Stay", lambda: AP_RL_Stay(alpha_a=0.0, beta=1.0, kappa=2.0)),
        ("Vicarious_AP_VS", lambda: Vicarious_AP_VS(alpha_o=0.0, alpha_vs_base=0.0, alpha_a=0.0, beta=1.0, kappa=2.0)),
        ("Vicarious_VS_Stay", lambda: Vicarious_VS_Stay(alpha_o=0.0, alpha_a=0.0, beta=1.0, kappa=2.0)),
        ("Vicarious_DB_Stay", lambda: Vicarious_DB_Stay(alpha_o=0.0, demo_bias=0.0, beta=1.0, kappa=2.0)),
        (
            "Vicarious_AP_DB_STAY",
            lambda: Vicarious_AP_DB_STAY(alpha_o=0.0, alpha_a=0.0, demo_bias_rel=0.0, beta=1.0, kappa=2.0),
        ),
        (
            "Vicarious_Dir_DB_Stay",
            lambda: Vicarious_Dir_DB_Stay(alpha_o=0.0, demo_bias_rel=0.0, beta=1.0, kappa=2.0, demo_dirichlet_prior=1.0),
        ),
    ]
    return cases


@pytest.mark.parametrize(("name", "factory"), _stay_regression_cases())
def test_stay_tracker_is_driven_by_self_action_not_demo_action(
    name: str,
    factory: ModelFactory,
) -> None:
    model = factory()
    spec = _spec(is_social=True)
    assert model.supports(spec), f"{name} should support social spec"

    model.reset_block(spec=spec)
    probs_0 = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)

    model.social_update(state=0, social=_social_obs(choice=1, outcome=0.0), spec=spec)
    probs_after_social = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)

    # With social/value/demo terms neutralized, social update should not induce stay bias.
    assert np.allclose(probs_after_social, probs_0, atol=1e-12)

    # Private update should set self-stay tracker, increasing repeat-action tendency.
    model.update(state=0, action=0, outcome=0.0, spec=spec)
    probs_after_private = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)
    assert probs_after_private[0] > probs_after_private[1]
