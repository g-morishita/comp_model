"""Targeted tests for VicQ AP dual-weight model variants."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
from comp_model_impl.models.vicQ_ap_dualw_stay.vicQ_ap_dualw_stay import VicQ_AP_DualW_Stay
from comp_model_impl.models.vicQ_ap_dualw_nostay.vicQ_ap_dualw_nostay import VicQ_AP_DualW_NoStay
from comp_model_impl.models.vicQ_ap_dualw_stay.vicQ_ap_indep_dualw import VicQ_AP_IndepDualW


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


def test_stay_param_schema_uses_beta_mix_names() -> None:
    model = VicQ_AP_DualW_Stay()
    assert "beta" in model.param_schema.names
    assert "w" in model.param_schema.names
    assert "kappa" in model.param_schema.names
    assert "beta_q" not in model.param_schema.names
    assert "beta_a" not in model.param_schema.names
    params = model.get_params()
    assert "beta" in params
    assert "w" in params
    assert "kappa" in params


def test_nostay_param_schema_excludes_kappa() -> None:
    model = VicQ_AP_DualW_NoStay()
    assert "beta" in model.param_schema.names
    assert "w" in model.param_schema.names
    assert "kappa" not in model.param_schema.names
    params = model.get_params()
    assert "beta" in params
    assert "w" in params
    assert "kappa" not in params


def test_indep_model_keeps_dual_beta_names() -> None:
    model = VicQ_AP_IndepDualW()
    assert "beta_q" in model.param_schema.names
    assert "beta_a" in model.param_schema.names
    assert "beta" not in model.param_schema.names
    assert "w" not in model.param_schema.names


def test_mixture_weight_w_is_used_by_action_probs() -> None:
    spec = _social_spec()
    model_q = VicQ_AP_DualW_Stay(beta=10.0, w=0.95, kappa=0.0)
    model_q.reset_block(spec=spec)
    model_q._q = np.array([1.0, 0.0], dtype=float)
    model_q._demo_pi = np.array([0.0, 1.0], dtype=float)
    p_q = np.asarray(model_q.action_probs(state=0, spec=spec), dtype=float)

    model_pi = VicQ_AP_DualW_Stay(beta=10.0, w=0.05, kappa=0.0)
    model_pi.reset_block(spec=spec)
    model_pi._q = np.array([1.0, 0.0], dtype=float)
    model_pi._demo_pi = np.array([0.0, 1.0], dtype=float)
    p_pi = np.asarray(model_pi.action_probs(state=0, spec=spec), dtype=float)

    assert p_q[0] > p_q[1]
    assert p_pi[1] > p_pi[0]


def test_shared_beta_controls_choice_determinism() -> None:
    spec = _social_spec()
    model_low = VicQ_AP_DualW_Stay(beta=0.1, w=1.0, kappa=0.0)
    model_low.reset_block(spec=spec)
    model_low._q = np.array([1.0, 0.0], dtype=float)
    p_low = np.asarray(model_low.action_probs(state=0, spec=spec), dtype=float)

    model_high = VicQ_AP_DualW_Stay(beta=10.0, w=1.0, kappa=0.0)
    model_high.reset_block(spec=spec)
    model_high._q = np.array([1.0, 0.0], dtype=float)
    p_high = np.asarray(model_high.action_probs(state=0, spec=spec), dtype=float)

    assert p_high[0] > p_low[0]


def test_stay_variant_reset_block_clears_self_stay_tracker() -> None:
    spec = _social_spec()
    model = VicQ_AP_DualW_Stay(alpha_o=0.0, alpha_a=0.0, beta=1e-6, w=0.5, kappa=2.0)

    model.reset_block(spec=spec)
    model.update(state=0, action=1, outcome=None, spec=spec)
    p_after_update = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)
    assert p_after_update[1] > p_after_update[0]

    model.reset_block(spec=spec)
    p_after_reset = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)
    assert np.allclose(p_after_reset, np.array([0.5, 0.5]), atol=1e-12)


def test_nostay_variant_private_update_does_not_add_stay_bias() -> None:
    spec = _social_spec()
    model = VicQ_AP_DualW_NoStay(alpha_o=0.0, alpha_a=0.0, beta=1e-6, w=0.5)

    model.reset_block(spec=spec)
    p_before = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)
    model.update(state=0, action=1, outcome=None, spec=spec)
    p_after = np.asarray(model.action_probs(state=0, spec=spec), dtype=float)

    assert np.allclose(p_before, p_after, atol=1e-12)


def test_social_update_raises_on_out_of_range_demo_action() -> None:
    spec = _social_spec()
    model = VicQ_AP_DualW_Stay()
    model.reset_block(spec=spec)

    with pytest.raises(ValueError):
        model.social_update(
            state=0,
            social=SocialObservation(others_choices=[2], observed_others_outcomes=[1.0]),
            spec=spec,
        )
