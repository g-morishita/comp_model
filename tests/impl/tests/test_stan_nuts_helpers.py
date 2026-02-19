"""Unit tests for Stan NUTS helper utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_impl.estimators.stan.adapters.base import StanProgramRef
from comp_model_impl.estimators.stan.nuts import (
    StanHierarchicalNUTSEstimator,
    _add_subject_posterior_summaries,
    _add_subject_posterior_samples,
    _delta_labels,
    _flatten_mean,
    _hyper_means_from_pop_hat,
    _is_within_subject_model,
    _load_yaml,
    _posterior_samples_from_draws,
    _posterior_summary_from_draws,
    _safe_summary_metric,
    _strip_hat,
    _waic_diagnostics_from_fit,
)
from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.models.within_subject_shared_delta import ConditionedSharedDeltaModel


@dataclass
class _DummySeries:
    """Minimal series-like object with max/min."""

    values: list[float]

    def max(self) -> float:
        """Return the max of the series."""
        return max(self.values)

    def min(self) -> float:
        """Return the min of the series."""
        return min(self.values)


@dataclass
class _DummySummary:
    """Minimal dataframe-like object for summary metrics."""

    columns: list[str]
    data: dict[str, _DummySeries]

    def __getitem__(self, key: str) -> _DummySeries:
        """Return the series for a column."""
        return self.data[key]


@dataclass
class _DummyFitWithLogLik:
    """Minimal CmdStanPy-fit-like object for log_lik extraction."""

    log_lik: np.ndarray

    def stan_variable(self, name: str):
        """Return named Stan variable or raise KeyError."""
        if str(name) != "log_lik":
            raise KeyError(name)
        return self.log_lik


def test_strip_hat():
    """_strip_hat removes trailing suffixes and leaves other names unchanged."""
    assert _strip_hat("alpha_hat") == "alpha"
    assert _strip_hat("beta") == "beta"


def test_delta_labels_excludes_baseline():
    """_delta_labels returns non-baseline labels in order."""
    labels = ["A", "B", "C"]
    assert _delta_labels(labels, baseline_idx_1based=2) == ["A", "C"]


def test_flatten_mean_scalar_vector_matrix_and_tensor():
    """_flatten_mean flattens scalar, vector, matrix, and tensor inputs."""
    assert _flatten_mean(name="alpha_hat", mean=1.2) == {"alpha": 1.2}

    out_vec = _flatten_mean(name="beta_hat", mean=[0.1, 0.2], condition_labels=["A", "B"])
    assert out_vec == {"beta__A": 0.1, "beta__B": 0.2}

    out_delta = _flatten_mean(
        name="gamma__delta_hat",
        mean=[-0.5, 0.25],
        condition_labels=["A", "B", "C"],
        baseline_idx_1based=2,
    )
    assert out_delta == {"gamma__delta__A": -0.5, "gamma__delta__C": 0.25}

    out_mat = _flatten_mean(name="mu_hat", mean=[[1.0, 2.0], [3.0, 4.0]])
    assert out_mat == {
        "mu[1,1]": 1.0,
        "mu[1,2]": 2.0,
        "mu[2,1]": 3.0,
        "mu[2,2]": 4.0,
    }

    out_tens = _flatten_mean(name="tau_hat", mean=[[[1.0, 2.0], [3.0, 4.0]]])
    assert out_tens == {
        "tau[1,1,1]": 1.0,
        "tau[1,1,2]": 2.0,
        "tau[1,2,1]": 3.0,
        "tau[1,2,2]": 4.0,
    }


def test_safe_summary_metric_handles_missing_and_agg():
    """_safe_summary_metric extracts max/min and handles missing columns."""
    summary = _DummySummary(
        columns=["R_hat", "ESS_bulk"],
        data={
            "R_hat": _DummySeries([1.1, 1.2]),
            "ESS_bulk": _DummySeries([200, 150]),
        },
    )
    assert _safe_summary_metric(summary, ["R_hat"], agg="max") == 1.2
    assert _safe_summary_metric(summary, ["ESS_bulk"], agg="min") == 150
    assert _safe_summary_metric(summary, ["missing"], agg="max") is None


def test_is_within_subject_model_flag():
    """_is_within_subject_model identifies shared+delta wrappers."""
    base = QRL()
    ws = ConditionedSharedDeltaModel(base_model=base, conditions=["A", "B"], baseline_condition="A")
    assert _is_within_subject_model(base) is False
    assert _is_within_subject_model(ws) is True


def test_posterior_summary_from_draws_scalar_and_vector():
    """Posterior summaries return mean/sd/quantiles for flattened variables."""
    scalar = _posterior_summary_from_draws(name="alpha_hat", draws=np.array([0.1, 0.2, 0.3, 0.4]))
    assert set(scalar.keys()) == {"alpha"}
    assert scalar["alpha"]["mean"] == pytest.approx(0.25)
    assert scalar["alpha"]["q50"] == pytest.approx(0.25)

    vec_draws = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
        ]
    )
    vec = _posterior_summary_from_draws(
        name="beta_hat",
        draws=vec_draws,
        condition_labels=["A", "B"],
    )
    assert set(vec.keys()) == {"beta__A", "beta__B"}
    assert vec["beta__A"]["mean"] == pytest.approx(0.25)
    assert vec["beta__B"]["mean"] == pytest.approx(0.75)


def test_add_subject_posterior_summaries_maps_draws_axis_1():
    """Subject posterior summaries should be split by subject axis."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    draws = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
        ]
    )
    _add_subject_posterior_summaries(
        out=out,
        name="theta_hat",
        draws=draws,
        subject_ids=["S1", "S2"],
    )

    assert set(out.keys()) == {"S1", "S2"}
    assert out["S1"]["theta"]["mean"] == pytest.approx(0.25)
    assert out["S2"]["theta"]["mean"] == pytest.approx(0.75)


def test_posterior_samples_from_draws_scalar_and_vector():
    """Posterior samples should flatten into per-parameter draw series."""
    scalar = _posterior_samples_from_draws(name="alpha_hat", draws=np.array([0.1, 0.2, 0.3]))
    assert set(scalar.keys()) == {"alpha"}
    assert scalar["alpha"] == pytest.approx([0.1, 0.2, 0.3])

    vec_draws = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ]
    )
    vec = _posterior_samples_from_draws(
        name="beta_hat",
        draws=vec_draws,
        condition_labels=["A", "B"],
    )
    assert set(vec.keys()) == {"beta__A", "beta__B"}
    assert vec["beta__A"] == pytest.approx([0.1, 0.2, 0.3])
    assert vec["beta__B"] == pytest.approx([0.9, 0.8, 0.7])


def test_add_subject_posterior_samples_maps_draws_axis_1():
    """Subject posterior samples should be split by subject axis."""
    out: dict[str, dict[str, list[float]]] = {}
    draws = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ]
    )
    _add_subject_posterior_samples(
        out=out,
        name="theta_hat",
        draws=draws,
        subject_ids=["S1", "S2"],
    )

    assert set(out.keys()) == {"S1", "S2"}
    assert out["S1"]["theta"] == pytest.approx([0.1, 0.2, 0.3])
    assert out["S2"]["theta"] == pytest.approx([0.9, 0.8, 0.7])


def test_hierarchical_estimator_thin_and_posterior_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hierarchical estimator should pass CmdStan thin and emit posterior samples."""
    import comp_model_impl.estimators.stan.nuts as nuts_mod

    class _DummyAdapter:
        def program(self, family: str) -> StanProgramRef:
            return StanProgramRef(family="hier", key="dummy", program_name="dummy_hier")

        def required_priors(self, family: str) -> list[str]:
            return []

        def augment_subject_data(self, data: dict[str, object]) -> None:
            return

        def augment_study_data(self, data: dict[str, object]) -> None:
            return

        def subject_param_names(self) -> list[str]:
            return ["alpha_hat"]

        def population_var_names(self) -> list[str]:
            return ["mu_alpha_hat"]

    class _DummyFit:
        def stan_variable(self, name: str):
            if name == "mu_alpha_hat":
                return np.array([0.1, 0.2, 0.3], dtype=float)
            if name == "alpha_hat":
                return np.array(
                    [
                        [0.11, 0.21],
                        [0.12, 0.22],
                        [0.13, 0.23],
                    ],
                    dtype=float,
                )
            raise KeyError(name)

        def summary(self):
            return None

    class _DummyCompiled:
        def __init__(self) -> None:
            self.last_sample_kwargs: dict[str, object] | None = None

        def sample(self, **kwargs):
            self.last_sample_kwargs = dict(kwargs)
            return _DummyFit()

    compiled = _DummyCompiled()

    monkeypatch.setattr(nuts_mod, "resolve_stan_adapter", lambda model: _DummyAdapter())
    monkeypatch.setattr(nuts_mod, "load_stan_code", lambda kind, model_name: "data{} parameters{} model{}")
    monkeypatch.setattr(nuts_mod, "compile_cmdstan", lambda stan_code, cache_tag: compiled)
    monkeypatch.setattr(nuts_mod, "study_to_stan_data", lambda study: {})
    monkeypatch.setattr(nuts_mod, "priors_to_stan_data", lambda **kwargs: {})
    monkeypatch.setattr(nuts_mod, "_is_within_subject_model", lambda model: False)
    monkeypatch.setattr(nuts_mod.StanHierarchicalNUTSEstimator, "supports", lambda self, study: True)

    est = StanHierarchicalNUTSEstimator(
        model=object(),  # type: ignore[arg-type]
        hyper_priors={},
        sample_thin=5,
        return_posterior_samples=True,
    )
    study = StudyData(
        subjects=[
            SubjectData(subject_id="s1", blocks=[]),
            SubjectData(subject_id="s2", blocks=[]),
        ]
    )
    out = est.fit(study=study, rng=np.random.default_rng(123))

    assert compiled.last_sample_kwargs is not None
    assert int(compiled.last_sample_kwargs["thin"]) == 5
    assert out.diagnostics["sample_thin"] == 5
    assert out.diagnostics["population_posterior_samples"]["mu_alpha"] == pytest.approx([0.1, 0.2, 0.3])
    assert out.diagnostics["subject_posterior_samples"]["s1"]["alpha"] == pytest.approx([0.11, 0.12, 0.13])
    assert out.diagnostics["subject_posterior_samples"]["s2"]["alpha"] == pytest.approx([0.21, 0.22, 0.23])


def test_hierarchical_estimator_rejects_non_positive_sample_thin() -> None:
    """sample_thin must be a positive integer."""
    with pytest.raises(ValueError, match="sample_thin"):
        _ = StanHierarchicalNUTSEstimator(
            model=object(),  # type: ignore[arg-type]
            hyper_priors={},
            sample_thin=0,
        )


def test_load_yaml_reads_mapping(tmp_path):
    """_load_yaml reads a YAML mapping from disk."""
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    path.write_text("a: 1\nb: 2\n", encoding="utf-8")
    cfg = _load_yaml(str(path))
    assert isinstance(cfg, dict)
    assert cfg == {"a": 1, "b": 2}


def test_waic_diagnostics_from_fit_extracts_metrics() -> None:
    """WAIC diagnostics should be computed when log_lik draws are available."""
    fit = _DummyFitWithLogLik(
        log_lik=np.array(
            [
                [-1.0, -2.0, -0.5],
                [-1.1, -1.9, -0.7],
                [-0.8, -2.2, -0.6],
            ],
            dtype=float,
        )
    )
    out = _waic_diagnostics_from_fit(fit)
    assert out is not None
    assert set(out.keys()) == {"waic", "elpd_waic", "p_waic", "waic_n_obs"}
    assert np.isfinite(float(out["waic"]))
    assert int(out["waic_n_obs"]) == 3


def test_waic_diagnostics_from_fit_returns_none_when_missing() -> None:
    """Missing log_lik should return None rather than raising."""
    fit = _DummyFitWithLogLik(log_lik=np.array([[-1.0, -2.0]], dtype=float))
    # monkey patch method to always fail lookup
    fit.stan_variable = lambda name: (_ for _ in ()).throw(KeyError(name))  # type: ignore[method-assign]
    out = _waic_diagnostics_from_fit(fit)
    assert out is None


def test_hyper_means_from_pop_hat_requires_canonical_keys() -> None:
    """Conditional-MAP hyper key lookup should require canonical key names."""
    z_names = [
        "alpha_o__shared_z",
        "beta__shared_z",
        "alpha_o__delta_z__B",
        "beta__delta_z__B",
    ]

    # Legacy/single-underscore style should be rejected.
    pop_hat_legacy = {
        "mu_alpha_o_shared": 0.1,
        "sd_alpha_o_shared": 1.0,
        "mu_beta__shared": -0.2,
        "sd_beta__shared": 0.8,
        "mu_alpha_o_delta__B": 0.3,
        "sd_alpha_o_delta__B": 1.2,
        "mu_beta__delta__B": -0.1,
        "sd_beta__delta__B": 0.7,
    }
    with pytest.raises(ValueError, match="mu_alpha_o__shared"):
        _ = _hyper_means_from_pop_hat(pop_hat=pop_hat_legacy, z_names=z_names)

    # Canonical/double-underscore style should work.
    pop_hat_canonical = {
        "mu_alpha_o__shared": 0.4,
        "sd_alpha_o__shared": 1.5,
        "mu_beta__shared": 0.0,
        "sd_beta__shared": 0.9,
        "mu_alpha_o__delta__B": -0.2,
        "sd_alpha_o__delta__B": 1.1,
        "mu_beta__delta__B": 0.6,
        "sd_beta__delta__B": 0.5,
    }
    mu2, sd2 = _hyper_means_from_pop_hat(pop_hat=pop_hat_canonical, z_names=z_names)
    assert mu2.tolist() == pytest.approx([0.4, 0.0, -0.2, 0.6])
    assert sd2.tolist() == pytest.approx([1.5, 0.9, 1.1, 0.5])
