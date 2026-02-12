"""Tests for Stan adapter implementations."""

from __future__ import annotations

import pytest

from comp_model_impl.estimators.stan.adapters import (
    QRLStanAdapter,
    VicQAPDualWStayStanAdapter,
    VicQAPDualWStayWithinSubjectStanAdapter,
    VicQAPDualWNoStayStanAdapter,
    VicQAPDualWNoStayWithinSubjectStanAdapter,
    VicQAPIndepDualWStanAdapter,
    VicariousAPVSStanAdapter,
    VicariousAPDBStayStanAdapter,
    VicariousDBStayStanAdapter,
    VicariousDBStayWithinSubjectStanAdapter,
    VicariousDirDBStayStanAdapter,
    VicariousRLStanAdapter,
    VicariousRLStayStanAdapter,
    VicariousRLWithinSubjectStanAdapter,
    VicariousRLStayWithinSubjectStanAdapter,
    VicariousVSStanAdapter,
    VicariousVSStayStanAdapter,
    VSStanAdapter,
    VSWithinSubjectStanAdapter,
)
from comp_model_impl.estimators.stan.adapters.registry import resolve_stan_adapter
from comp_model_impl.models import (
    QRL,
    VS,
    VicQ_AP_DualW_Stay,
    VicQ_AP_DualW_NoStay,
    VicQ_AP_IndepDualW,
    Vicarious_AP_DB_STAY,
    Vicarious_AP_VS,
    Vicarious_DB_Stay,
    Vicarious_Dir_DB_Stay,
    Vicarious_RL,
    Vicarious_RL_Stay,
    Vicarious_VS,
    Vicarious_VS_Stay,
)
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


def test_vicarious_ap_db_stay_adapter_adds_constants_and_priors():
    """Vicarious-AP-DB-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_AP_DB_STAY(beta_max=13.0, kappa_abs_max=2.0, demo_bias_abs_max=4.0)
    adapter = VicariousAPDBStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_ap_db_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "demo_bias_rel", "beta", "kappa"]
    assert "mu_demo_bias_rel" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(13.0)
    assert data["kappa_abs_max"] == pytest.approx(2.0)
    assert data["demo_bias_rel_abs_max"] == pytest.approx(4.0)


def test_vicq_ap_dualw_stay_adapter_adds_constants_and_priors():
    """VicQ-AP-DualW-Stay adapter exposes expected priors and data constants."""
    model = VicQ_AP_DualW_Stay(beta_max=19.0, kappa_abs_max=1.75)
    adapter = VicQAPDualWStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicQ_ap_dualw_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta", "w", "kappa"]
    assert "mu_beta" in adapter.required_priors("hier")
    assert "mu_w" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(19.0)
    assert data["kappa_abs_max"] == pytest.approx(1.75)


def test_vicq_ap_dualw_nostay_adapter_adds_constants_and_priors():
    """VicQ-AP-DualW-NoStay adapter exposes expected priors and data constants."""
    model = VicQ_AP_DualW_NoStay(beta_max=19.0)
    adapter = VicQAPDualWNoStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicQ_ap_dualw_nostay"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta", "w"]
    assert "mu_beta" in adapter.required_priors("hier")
    assert "mu_w" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(19.0)


def test_vicq_ap_indep_dualw_adapter_adds_constants_and_priors():
    """Independent VicQ-AP-DualW adapter exposes expected priors and constants."""
    model = VicQ_AP_IndepDualW(beta_max=19.0, kappa_abs_max=1.75)
    adapter = VicQAPIndepDualWStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicQ_ap_indep_dualw"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta_q", "beta_a", "kappa"]
    assert "mu_beta_q" in adapter.required_priors("hier")
    assert "mu_beta_a" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(19.0)
    assert data["kappa_abs_max"] == pytest.approx(1.75)


def test_vicarious_dir_db_stay_adapter_adds_constants_and_priors():
    """Vicarious-Dirichlet-DB-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_Dir_DB_Stay(beta_max=11.0, kappa_abs_max=1.5, demo_bias_abs_max=3.5, demo_dirichlet_prior=1.25)
    adapter = VicariousDirDBStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_dir_db_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "demo_bias_rel", "beta", "kappa"]
    assert "mu_demo_bias_rel" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(11.0)
    assert data["kappa_abs_max"] == pytest.approx(1.5)
    assert data["demo_bias_rel_abs_max"] == pytest.approx(3.5)
    assert data["demo_dirichlet_prior"] == pytest.approx(1.25)


def test_vicarious_db_stay_adapter_adds_constants_and_priors():
    """Vicarious-DB-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_DB_Stay(beta_max=9.5, kappa_abs_max=1.2, demo_bias_abs_max=2.25)
    adapter = VicariousDBStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_db_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "demo_bias", "beta", "kappa"]
    assert "mu_demo_bias" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(9.5)
    assert data["kappa_abs_max"] == pytest.approx(1.2)
    assert data["demo_bias_abs_max"] == pytest.approx(2.25)


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


def test_vicarious_rl_stay_adapter_adds_constants_and_priors():
    """Vicarious-RL-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_RL_Stay(beta_max=12.0, kappa_abs_max=1.5)
    adapter = VicariousRLStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_rl_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "beta", "kappa"]
    assert adapter.required_priors("hier") == [
        "mu_alpha_o",
        "sd_alpha_o",
        "mu_beta",
        "sd_beta",
        "mu_kappa",
        "sd_kappa",
    ]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(12.0)
    assert data["kappa_abs_max"] == pytest.approx(1.5)


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


def test_vicarious_rl_stay_within_subject_adapter_uses_base_model_constants():
    """Within-subject Vicarious-RL-Stay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=Vicarious_RL_Stay(beta_max=18.0, kappa_abs_max=1.3),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicariousRLStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicarious_rl_stay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(18.0)
    assert data["kappa_abs_max"] == pytest.approx(1.3)


def test_vicarious_db_stay_within_subject_adapter_uses_base_model_constants():
    """Within-subject Vicarious-DB-Stay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=Vicarious_DB_Stay(beta_max=7.5, kappa_abs_max=1.1, demo_bias_abs_max=2.2),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicariousDBStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicarious_db_stay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(7.5)
    assert data["kappa_abs_max"] == pytest.approx(1.1)
    assert data["demo_bias_abs_max"] == pytest.approx(2.2)


def test_vicq_ap_dualw_stay_within_subject_adapter_uses_base_model_constants():
    """Within-subject VicQ-AP-DualW-Stay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=VicQ_AP_DualW_Stay(beta_max=17.5, kappa_abs_max=1.7),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicQAPDualWStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicQ_ap_dualw_stay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(17.5)
    assert data["kappa_abs_max"] == pytest.approx(1.7)


def test_vicq_ap_dualw_nostay_within_subject_adapter_uses_base_model_constants():
    """Within-subject VicQ-AP-DualW-NoStay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=VicQ_AP_DualW_NoStay(beta_max=17.5),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicQAPDualWNoStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicQ_ap_dualw_nostay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert data["beta_upper"] == pytest.approx(17.5)


def test_resolve_stan_adapter_for_base_models():
    """Registry resolves adapters for base models."""
    assert isinstance(resolve_stan_adapter(QRL()), QRLStanAdapter)
    assert isinstance(resolve_stan_adapter(VS()), VSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_RL()), VicariousRLStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_RL_Stay()), VicariousRLStayStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_AP_VS()), VicariousAPVSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_AP_DB_STAY()), VicariousAPDBStayStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_Dir_DB_Stay()), VicariousDirDBStayStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_DB_Stay()), VicariousDBStayStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_VS()), VicariousVSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_VS_Stay()), VicariousVSStayStanAdapter)
    assert isinstance(resolve_stan_adapter(VicQ_AP_IndepDualW()), VicQAPIndepDualWStanAdapter)
    assert isinstance(resolve_stan_adapter(VicQ_AP_DualW_Stay()), VicQAPDualWStayStanAdapter)
    assert isinstance(resolve_stan_adapter(VicQ_AP_DualW_NoStay()), VicQAPDualWNoStayStanAdapter)
