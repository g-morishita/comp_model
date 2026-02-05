"""Tests for Stan prior parsing and data conversion helpers."""

from __future__ import annotations

import pytest

from comp_model_impl.estimators.stan.priors import (
    Prior,
    FAMILY_CODE,
    parse_prior,
    priors_to_stan_data,
    priors_to_stan_data_strict,
)


class DummyAdapter:
    """Minimal adapter stub for required priors."""

    def __init__(self, required):
        self._required = list(required)

    def required_priors(self, family):
        return list(self._required)


def test_parse_prior_family_mappings():
    """All supported families parse to expected codes."""
    cases = [
        ("beta", {"family": "beta", "a": 2, "b": 3}, (2.0, 3.0, 0.0)),
        ("normal", {"family": "normal", "mu": 0, "sigma": 1}, (0.0, 1.0, 0.0)),
        ("lognormal", {"family": "lognormal", "mu": 0, "sigma": 1}, (0.0, 1.0, 0.0)),
        ("gamma", {"family": "gamma", "shape": 2, "rate": 3}, (2.0, 3.0, 0.0)),
        ("exponential", {"family": "exponential", "rate": 2}, (2.0, 0.0, 0.0)),
        ("half-normal", {"family": "half-normal", "sigma": 1}, (1.0, 0.0, 0.0)),
        ("student-t", {"family": "student-t", "df": 4, "mu": 0, "sigma": 2}, (4.0, 0.0, 2.0)),
        ("cauchy", {"family": "cauchy", "loc": 0, "scale": 1}, (0.0, 1.0, 0.0)),
    ]
    for fam, cfg, params in cases:
        pr = parse_prior(cfg)
        assert isinstance(pr, Prior)
        assert pr.family == fam
        assert (pr.p1, pr.p2, pr.p3) == params
        assert fam in FAMILY_CODE


def test_parse_prior_rejects_unknown_family():
    """Unknown prior families should raise a ValueError."""
    with pytest.raises(ValueError):
        parse_prior({"family": "unsupported"})


def test_priors_to_stan_data_strict_missing_and_extra():
    """Strict conversion enforces required priors and rejects extras."""
    required = ["alpha", "beta"]
    cfg = {"alpha": {"family": "normal", "mu": 0, "sigma": 1}}
    with pytest.raises(ValueError):
        priors_to_stan_data_strict(priors_cfg=cfg, required=required, forbid_extra=False)

    cfg_full = {
        "alpha": {"family": "normal", "mu": 0, "sigma": 1},
        "beta": {"family": "beta", "a": 2, "b": 2},
        "extra": {"family": "normal", "mu": 0, "sigma": 1},
    }
    with pytest.raises(ValueError):
        priors_to_stan_data_strict(priors_cfg=cfg_full, required=required, forbid_extra=True)


def test_priors_to_stan_data_accepts_prior_objects():
    """Conversion accepts already-parsed Prior objects."""
    cfg = {"alpha": Prior("normal", 0.0, 1.0, 0.0)}
    data = priors_to_stan_data_strict(priors_cfg=cfg, required=["alpha"], forbid_extra=False)
    assert data["alpha_prior_family"] == FAMILY_CODE["normal"]
    assert data["alpha_prior_p1"] == 0.0
    assert data["alpha_prior_p2"] == 1.0
    assert data["alpha_prior_p3"] == 0.0


def test_priors_to_stan_data_uses_adapter_required_list():
    """Adapter required priors are used for conversion."""
    adapter = DummyAdapter(required=["alpha"])
    cfg = {"alpha": {"family": "beta", "a": 1, "b": 3}}
    data = priors_to_stan_data(priors_cfg=cfg, adapter=adapter, family="indiv")
    assert data["alpha_prior_family"] == FAMILY_CODE["beta"]
    assert data["alpha_prior_p1"] == 1.0
    assert data["alpha_prior_p2"] == 3.0
    assert data["alpha_prior_p3"] == 0.0
