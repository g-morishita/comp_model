from __future__ import annotations

import pytest

from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
from comp_model_core.interfaces.block_runner import SocialObservation

from comp_model_impl.models import Vicarious_Dir_DB_Stay, Vicarious_DB_Stay


def _social_spec(n_actions: int = 2) -> EnvironmentSpec:
    return EnvironmentSpec(
        n_actions=n_actions,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=True,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )


def test_vicarious_dir_db_stay_bias_direction_positive():
    spec = _social_spec(2)
    model = Vicarious_Dir_DB_Stay(alpha_o=0.0, demo_bias_rel=2.0, beta=1.0, kappa=0.0, demo_dirichlet_prior=1.0)
    model.reset_block(spec=spec)

    social = SocialObservation(others_choices=[0], observed_others_outcomes=None)
    model.social_update(state=0, social=social, spec=spec, info=None)

    probs = model.action_probs(state=0, spec=spec)
    assert probs[0] > probs[1]


def test_vicarious_dir_db_stay_bias_direction_negative():
    spec = _social_spec(2)
    model = Vicarious_Dir_DB_Stay(alpha_o=0.0, demo_bias_rel=-2.0, beta=1.0, kappa=0.0, demo_dirichlet_prior=1.0)
    model.reset_block(spec=spec)

    social = SocialObservation(others_choices=[0], observed_others_outcomes=None)
    model.social_update(state=0, social=social, spec=spec, info=None)

    probs = model.action_probs(state=0, spec=spec)
    assert probs[0] < probs[1]


def test_vicarious_db_stay_bias_direction_positive():
    spec = _social_spec(2)
    model = Vicarious_DB_Stay(alpha_o=0.0, demo_bias=1.5, beta=1.0, kappa=0.0)
    model.reset_block(spec=spec)

    social = SocialObservation(others_choices=[0], observed_others_outcomes=None)
    model.social_update(state=0, social=social, spec=spec, info=None)

    probs = model.action_probs(state=0, spec=spec)
    assert probs[0] > probs[1]


def test_vicarious_db_stay_bias_direction_negative():
    spec = _social_spec(2)
    model = Vicarious_DB_Stay(alpha_o=0.0, demo_bias=-1.5, beta=1.0, kappa=0.0)
    model.reset_block(spec=spec)

    social = SocialObservation(others_choices=[0], observed_others_outcomes=None)
    model.social_update(state=0, social=social, spec=spec, info=None)

    probs = model.action_probs(state=0, spec=spec)
    assert probs[0] < probs[1]
