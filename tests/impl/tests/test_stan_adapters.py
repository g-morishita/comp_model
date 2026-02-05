"""Tests for Stan adapter implementations."""

from __future__ import annotations

import pytest

from comp_model_impl.estimators.stan.adapters import (
    QRLStanAdapter,
    VicariousAPVSStanAdapter,
    VicariousRLStanAdapter,
    VicariousRLWithinSubjectStanAdapter,
    VicariousVSStanAdapter,
    VicariousVSStayStanAdapter,
    VSStanAdapter,
    VSWithinSubjectStanAdapter,
)
from comp_model_impl.estimators.stan.adapters.registry import resolve_stan_adapter
from comp_model_impl.models import QRL, VS, Vicarious_AP_VS, Vicarious_RL, Vicarious_VS, Vicarious_VS_Stay
from comp_model_impl.models.within_subject_shared_delta import wrap_model_with_shared_delta_conditions


def test_vs_adapter_adds_constants_and_priors():
    """VS adapter exposes expected priors and data constants."""
    model = VS(beta_max=20.0, kappa_abs_max=1.0, pseudo_reward=1.0)
    adapter = VSStanAdapter(model=model)

    assert adapter.program("indiv").key == "vs"
    assert adapter.required_priors("indiv") == ["alpha_p", "alpha_i", "beta", "kappa"]
    assert "mu_alpha_p" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(20.0)
    assert data["kappa_abs_max"] == pytest.approx(1.0)
    assert data["pseudo_reward"] == pytest.approx(1.0)


def test_vicarious_vs_adapter_adds_constants_and_priors():
    """Vicarious-VS adapter exposes expected priors and data constants."""
    model = Vicarious_VS(beta_max=15.0, pseudo_reward=0.7)
    adapter = VicariousVSStanAdapter(model=model)

    assert adapter.program("hier").key == "vicarious_vs"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta"]
    assert "mu_alpha_o" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["pseudo_reward"] == pytest.approx(0.7)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(15.0)


def test_vicarious_vs_stay_adapter_adds_constants_and_priors():
    """Vicarious-VS-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_VS_Stay(beta_max=14.0, kappa_max=1.5, pseudo_reward=0.6)
    adapter = VicariousVSStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_vs_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta", "kappa"]
    assert "mu_alpha_o" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["pseudo_reward"] == pytest.approx(0.6)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(14.0)
    assert data["kappa_abs_max"] == pytest.approx(1.5)


def test_vicarious_ap_vs_adapter_adds_constants_and_priors():
    """Vicarious-AP-VS adapter exposes expected priors and data constants."""
    model = Vicarious_AP_VS(beta_max=17.0, kappa_abs_max=2.5, pseudo_reward=0.8)
    adapter = VicariousAPVSStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_ap_vs"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_vs_base", "alpha_a", "beta", "kappa"]
    assert "mu_alpha_vs_base" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["pseudo_reward"] == pytest.approx(0.8)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(17.0)
    assert data["kappa_abs_max"] == pytest.approx(2.5)


def test_vicarious_rl_adapter_adds_constants_and_priors():
    """Vicarious-RL adapter exposes expected priors and data constants."""
    model = Vicarious_RL(beta_max=12.0)
    adapter = VicariousRLStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_rl"
    assert adapter.required_priors("indiv") == ["alpha_o", "beta"]
    assert adapter.required_priors("hier") == ["mu_alpha_o", "sd_alpha_o", "mu_beta", "sd_beta"]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(12.0)


def test_qrl_adapter_adds_constants_and_priors():
    """QRL adapter exposes expected priors and beta bounds."""
    model = QRL(beta_max=9.0)
    adapter = QRLStanAdapter(model=model)

    assert adapter.program("indiv").key == "qrl"
    assert adapter.required_priors("indiv") == ["alpha", "beta"]
    assert adapter.required_priors("hier") == ["mu_alpha", "sd_alpha", "mu_beta", "sd_beta"]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(9.0)


def test_vs_within_subject_adapter_uses_base_model_constants():
    """Within-subject VS adapter mirrors base-model constants."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=VS(beta_max=25.0, kappa_abs_max=2.0, pseudo_reward=0.5),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VSWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vs_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(25.0)
    assert data["kappa_abs_max"] == pytest.approx(2.0)
    assert data["pseudo_reward"] == pytest.approx(0.5)


def test_vicarious_rl_within_subject_adapter_uses_base_model_constants():
    """Within-subject Vicarious-RL adapter uses base-model beta bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=Vicarious_RL(beta_max=18.0),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicariousRLWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicarious_rl_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(18.0)


def test_resolve_stan_adapter_for_base_models():
    """Registry resolves adapters for base models."""
    assert isinstance(resolve_stan_adapter(QRL()), QRLStanAdapter)
    assert isinstance(resolve_stan_adapter(VS()), VSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_RL()), VicariousRLStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_AP_VS()), VicariousAPVSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_VS()), VicariousVSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_VS_Stay()), VicariousVSStayStanAdapter)
