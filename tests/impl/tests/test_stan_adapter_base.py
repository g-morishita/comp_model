"""Tests for Stan adapter base definitions."""

from __future__ import annotations

from comp_model_impl.estimators.stan.adapters.base import StanProgramRef


def test_stan_program_ref_fields():
    """StanProgramRef stores family, key, and program_name."""
    ref = StanProgramRef(family="indiv", key="vs", program_name="vs_indiv")
    assert ref.family == "indiv"
    assert ref.key == "vs"
    assert ref.program_name == "vs_indiv"
