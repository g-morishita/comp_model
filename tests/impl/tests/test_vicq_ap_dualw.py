"""Targeted tests for VicQ_AP_DualW."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
from comp_model_impl.models.vicQ_ap_dualw.vicQ_ap_dualw import VicQ_AP_DualW


def _social_spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        n_actions=2,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=True,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )


def test_param_schema_uses_beta_a_name() -> None:
    model = VicQ_AP_DualW()
    assert "beta_a" in model.param_schema.names
    assert "beta_o" not in model.param_schema.names
    params = model.get_params()
    assert "beta_a" in params
    assert "beta_o" not in params


def test_policy_weight_param_beta_a_is_used_by_action_probs() -> None:
    spec = _social_spec()
    social = SocialObservation(others_choices=[1], observed_others_outcomes=[0.0])

    model_low = VicQ_AP_DualW(alpha_o=0.0, alpha_a=1.0, beta_q=0.0, beta_a=0.1, kappa=0.0)
    model_low.reset_block(spec=spec)
    model_low.social_update(state=0, social=social, spec=spec)
    p_low = np.asarray(model_low.action_probs(state=0, spec=spec), dtype=float)

    model_high = VicQ_AP_DualW(alpha_o=0.0, alpha_a=1.0, beta_q=0.0, beta_a=10.0, kappa=0.0)
    model_high.reset_block(spec=spec)
    model_high.social_update(state=0, social=social, spec=spec)
    p_high = np.asarray(model_high.action_probs(state=0, spec=spec), dtype=float)

    assert p_high[1] > p_low[1]


def test_reset_block_clears_self_stay_tracker() -> None:
    spec = _social_spec()
    model = VicQ_AP_DualW(alpha_o=0.0, alpha_a=0.0, beta_q=0.0, beta_a=0.0, kappa=2.0)

    model.reset_block(spec=spec)
    model.update(state=0, action=1, outcome=None, spec=spec)
    p_after_update = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)
    assert p_after_update[1] > p_after_update[0]

    model.reset_block(spec=spec)
    p_after_reset = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)
    assert np.allclose(p_after_reset, np.array([0.5, 0.5]), atol=1e-12)


def test_social_update_raises_on_out_of_range_demo_action() -> None:
    spec = _social_spec()
    model = VicQ_AP_DualW()
    model.reset_block(spec=spec)

    with pytest.raises(ValueError):
        model.social_update(
            state=0,
            social=SocialObservation(others_choices=[2], observed_others_outcomes=[1.0]),
            spec=spec,
        )

