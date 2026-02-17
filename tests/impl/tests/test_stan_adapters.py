"""Tests for Stan adapter implementations."""

from __future__ import annotations

import pytest

from comp_model_impl.estimators.stan.adapters import (
    APRLNoStayStanAdapter,
    APRLNoStayWithinSubjectStanAdapter,
    APRLStayStanAdapter,
    APRLStayWithinSubjectStanAdapter,
    QRLStanAdapter,
    QRLWithinSubjectStanAdapter,
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
    AP_RL_NoStay,
    AP_RL_Stay,
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
    model = VS(kappa_abs_max=1.0, pseudo_reward=1.0)
    adapter = VSStanAdapter(model=model)

    assert adapter.program("indiv").key == "vs"
    assert adapter.required_priors("indiv") == ["alpha_p", "alpha_i", "beta", "kappa"]
    assert "mu_alpha_p" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.0)
    assert data["pseudo_reward"] == pytest.approx(1.0)


def test_vicarious_vs_adapter_adds_constants_and_priors():
    """Vicarious-VS adapter exposes expected priors and data constants."""
    model = Vicarious_VS(pseudo_reward=0.7)
    adapter = VicariousVSStanAdapter(model=model)

    assert adapter.program("hier").key == "vicarious_vs"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta"]
    assert "mu_alpha_o" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["pseudo_reward"] == pytest.approx(0.7)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data


def test_vicarious_vs_stay_adapter_adds_constants_and_priors():
    """Vicarious-VS-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_VS_Stay(kappa_max=1.5, pseudo_reward=0.6)
    adapter = VicariousVSStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_vs_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta", "kappa"]
    assert "mu_alpha_o" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["pseudo_reward"] == pytest.approx(0.6)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.5)


def test_vicarious_ap_vs_adapter_adds_constants_and_priors():
    """Vicarious-AP-VS adapter exposes expected priors and data constants."""
    model = Vicarious_AP_VS(kappa_abs_max=2.5, pseudo_reward=0.8)
    adapter = VicariousAPVSStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_ap_vs"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_vs_base", "alpha_a", "beta", "kappa"]
    assert "mu_alpha_vs_base" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["pseudo_reward"] == pytest.approx(0.8)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(2.5)


def test_vicarious_ap_db_stay_adapter_adds_constants_and_priors():
    """Vicarious-AP-DB-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_AP_DB_STAY(kappa_abs_max=2.0, demo_bias_abs_max=4.0)
    adapter = VicariousAPDBStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_ap_db_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "demo_bias_rel", "beta", "kappa"]
    assert "mu_demo_bias_rel" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(2.0)
    assert data["demo_bias_rel_abs_max"] == pytest.approx(4.0)


def test_vicq_ap_dualw_stay_adapter_adds_constants_and_priors():
    """VicQ-AP-DualW-Stay adapter exposes expected priors and data constants."""
    model = VicQ_AP_DualW_Stay(kappa_abs_max=1.75)
    adapter = VicQAPDualWStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicQ_ap_dualw_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta", "w", "kappa"]
    assert "mu_beta" in adapter.required_priors("hier")
    assert "mu_w" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.75)


def test_vicq_ap_dualw_nostay_adapter_adds_constants_and_priors():
    """VicQ-AP-DualW-NoStay adapter exposes expected priors and data constants."""
    model = VicQ_AP_DualW_NoStay()
    adapter = VicQAPDualWNoStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicQ_ap_dualw_nostay"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta", "w"]
    assert "mu_beta" in adapter.required_priors("hier")
    assert "mu_w" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data


def test_vicq_ap_indep_dualw_adapter_adds_constants_and_priors():
    """Independent VicQ-AP-DualW adapter exposes expected priors and constants."""
    model = VicQ_AP_IndepDualW(kappa_abs_max=1.75)
    adapter = VicQAPIndepDualWStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicQ_ap_indep_dualw"
    assert adapter.required_priors("indiv") == ["alpha_o", "alpha_a", "beta_q", "beta_a", "kappa"]
    assert "mu_beta_q" in adapter.required_priors("hier")
    assert "mu_beta_a" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.75)


def test_vicarious_dir_db_stay_adapter_adds_constants_and_priors():
    """Vicarious-Dirichlet-DB-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_Dir_DB_Stay(kappa_abs_max=1.5, demo_bias_abs_max=3.5, demo_dirichlet_prior=1.25)
    adapter = VicariousDirDBStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_dir_db_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "demo_bias_rel", "beta", "kappa"]
    assert "mu_demo_bias_rel" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.5)
    assert data["demo_bias_rel_abs_max"] == pytest.approx(3.5)
    assert data["demo_dirichlet_prior"] == pytest.approx(1.25)


def test_vicarious_db_stay_adapter_adds_constants_and_priors():
    """Vicarious-DB-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_DB_Stay(kappa_abs_max=1.2, demo_bias_abs_max=2.25)
    adapter = VicariousDBStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_db_stay"
    assert adapter.required_priors("indiv") == ["alpha_o", "demo_bias", "beta", "kappa"]
    assert "mu_demo_bias" in adapter.required_priors("hier")

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.2)
    assert data["demo_bias_abs_max"] == pytest.approx(2.25)


def test_vicarious_rl_adapter_adds_constants_and_priors():
    """Vicarious-RL adapter exposes expected priors and data constants."""
    model = Vicarious_RL()
    adapter = VicariousRLStanAdapter(model=model)

    assert adapter.program("indiv").key == "vicarious_rl"
    assert adapter.required_priors("indiv") == ["alpha_o", "beta"]
    assert adapter.required_priors("hier") == ["mu_alpha_o", "sd_alpha_o", "mu_beta", "sd_beta"]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data


def test_vicarious_rl_stay_adapter_adds_constants_and_priors():
    """Vicarious-RL-Stay adapter exposes expected priors and data constants."""
    model = Vicarious_RL_Stay(kappa_abs_max=1.5)
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
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.5)


def test_ap_rl_stay_adapter_adds_constants_and_priors():
    """AP-RL-Stay adapter exposes expected priors and data constants."""
    model = AP_RL_Stay(kappa_abs_max=1.5)
    adapter = APRLStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "ap_rl_stay"
    assert adapter.required_priors("indiv") == ["alpha_a", "beta", "kappa"]
    assert adapter.required_priors("hier") == [
        "mu_alpha_a",
        "sd_alpha_a",
        "mu_beta",
        "sd_beta",
        "mu_kappa",
        "sd_kappa",
    ]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.5)


def test_ap_rl_nostay_adapter_adds_constants_and_priors():
    """AP-RL-NoStay adapter exposes expected priors and data constants."""
    model = AP_RL_NoStay()
    adapter = APRLNoStayStanAdapter(model=model)

    assert adapter.program("indiv").key == "ap_rl_nostay"
    assert adapter.required_priors("indiv") == ["alpha_a", "beta"]
    assert adapter.required_priors("hier") == ["mu_alpha_a", "sd_alpha_a", "mu_beta", "sd_beta"]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data


def test_qrl_adapter_adds_constants_and_priors():
    """QRL adapter exposes expected priors and beta bounds."""
    model = QRL()
    adapter = QRLStanAdapter(model=model)

    assert adapter.program("indiv").key == "qrl"
    assert adapter.required_priors("indiv") == ["alpha", "beta"]
    assert adapter.required_priors("hier") == ["mu_alpha", "sd_alpha", "mu_beta", "sd_beta"]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data


def test_vs_within_subject_adapter_uses_base_model_constants():
    """Within-subject VS adapter mirrors base-model constants."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=VS(kappa_abs_max=2.0, pseudo_reward=0.5),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VSWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vs_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(2.0)
    assert data["pseudo_reward"] == pytest.approx(0.5)


def test_qrl_within_subject_adapter_uses_base_model_constants():
    """Within-subject QRL adapter uses base-model lower beta bound."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=QRL(),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, QRLWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "qrl_within_subject"
    assert adapter.required_priors("indiv") == [
        "alpha__shared",
        "alpha__delta",
        "beta__shared",
        "beta__delta",
    ]
    assert adapter.required_priors("hier") == [
        "mu_alpha__shared",
        "sd_alpha__shared",
        "mu_beta__shared",
        "sd_beta__shared",
        "mu_alpha__delta",
        "sd_alpha__delta",
        "mu_beta__delta",
        "sd_beta__delta",
    ]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data

    hier_data = {}
    adapter.augment_study_data(hier_data)
    assert hier_data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in hier_data


def test_vicarious_rl_within_subject_adapter_uses_base_model_constants():
    """Within-subject Vicarious-RL adapter uses base-model beta bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=Vicarious_RL(),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicariousRLWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicarious_rl_within_subject"
    assert adapter.required_priors("indiv") == [
        "alpha_o__shared",
        "alpha_o__delta",
        "beta__shared",
        "beta__delta",
    ]
    assert adapter.required_priors("hier") == [
        "mu_alpha_o__shared",
        "sd_alpha_o__shared",
        "mu_beta__shared",
        "sd_beta__shared",
        "mu_alpha_o__delta",
        "sd_alpha_o__delta",
        "mu_beta__delta",
        "sd_beta__delta",
    ]

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data

    hier_data = {}
    adapter.augment_study_data(hier_data)
    assert hier_data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in hier_data


def test_vicarious_rl_stay_within_subject_adapter_uses_base_model_constants():
    """Within-subject Vicarious-RL-Stay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=Vicarious_RL_Stay(kappa_abs_max=1.3),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicariousRLStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicarious_rl_stay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.3)


def test_ap_rl_stay_within_subject_adapter_uses_base_model_constants():
    """Within-subject AP-RL-Stay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=AP_RL_Stay(kappa_abs_max=1.3),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, APRLStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "ap_rl_stay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.3)


def test_ap_rl_nostay_within_subject_adapter_uses_base_model_constants():
    """Within-subject AP-RL-NoStay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=AP_RL_NoStay(),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, APRLNoStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "ap_rl_nostay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data


def test_vicarious_db_stay_within_subject_adapter_uses_base_model_constants():
    """Within-subject Vicarious-DB-Stay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=Vicarious_DB_Stay(kappa_abs_max=1.1, demo_bias_abs_max=2.2),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicariousDBStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicarious_db_stay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.1)
    assert data["demo_bias_abs_max"] == pytest.approx(2.2)


def test_vicq_ap_dualw_stay_within_subject_adapter_uses_base_model_constants():
    """Within-subject VicQ-AP-DualW-Stay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=VicQ_AP_DualW_Stay(kappa_abs_max=1.7),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicQAPDualWStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicQ_ap_dualw_stay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data
    assert data["kappa_abs_max"] == pytest.approx(1.7)


def test_vicq_ap_dualw_nostay_within_subject_adapter_uses_base_model_constants():
    """Within-subject VicQ-AP-DualW-NoStay adapter uses base-model bounds."""
    wrapped = wrap_model_with_shared_delta_conditions(
        model=VicQ_AP_DualW_NoStay(),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    adapter = resolve_stan_adapter(wrapped)
    assert isinstance(adapter, VicQAPDualWNoStayWithinSubjectStanAdapter)
    assert adapter.program("indiv").key == "vicQ_ap_dualw_nostay_within_subject"

    data = {}
    adapter.augment_subject_data(data)
    assert data["beta_lower"] == pytest.approx(1e-6)
    assert "beta_upper" not in data


def test_resolve_stan_adapter_for_base_models():
    """Registry resolves adapters for base models."""
    assert isinstance(resolve_stan_adapter(QRL()), QRLStanAdapter)
    assert isinstance(resolve_stan_adapter(VS()), VSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_RL()), VicariousRLStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_RL_Stay()), VicariousRLStayStanAdapter)
    assert isinstance(resolve_stan_adapter(AP_RL_Stay()), APRLStayStanAdapter)
    assert isinstance(resolve_stan_adapter(AP_RL_NoStay()), APRLNoStayStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_AP_VS()), VicariousAPVSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_AP_DB_STAY()), VicariousAPDBStayStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_Dir_DB_Stay()), VicariousDirDBStayStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_DB_Stay()), VicariousDBStayStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_VS()), VicariousVSStanAdapter)
    assert isinstance(resolve_stan_adapter(Vicarious_VS_Stay()), VicariousVSStayStanAdapter)
    assert isinstance(resolve_stan_adapter(VicQ_AP_IndepDualW()), VicQAPIndepDualWStanAdapter)
    assert isinstance(resolve_stan_adapter(VicQ_AP_DualW_Stay()), VicQAPDualWStayStanAdapter)
    assert isinstance(resolve_stan_adapter(VicQ_AP_DualW_NoStay()), VicQAPDualWNoStayStanAdapter)
