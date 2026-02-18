"""Tests for CmdStan utility helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from comp_model_impl.estimators.stan import cmdstan_utils as cu


def test_read_text_reads_file(tmp_path):
    """Read UTF-8 text from a file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    path = tmp_path / "x.txt"
    path.write_text("hello", encoding="utf-8")
    assert cu._read_text(path) == "hello"


def test_load_stan_code_concatenates_common_and_body():
    """Stan code is assembled from common functions and a template body."""
    base = Path(cu.__file__).resolve().parent
    common = (base / "common" / "prior_functions.stan").read_text(encoding="utf-8")
    body = (base / "vs" / "indiv_body.stan").read_text(encoding="utf-8")
    code = cu.load_stan_code(kind="indiv", model_name="vs")
    assert code == common + "\n\n" + body


def test_compile_cmdstan_writes_file_and_uses_cache(tmp_path, monkeypatch):
    """compile_cmdstan writes the Stan file and instantiates CmdStanModel.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Patch helper for module-level dependencies.
    """

    class DummyModel:
        """Stub CmdStanModel that records the Stan file path."""

        def __init__(self, stan_file: str):
            self.stan_file = stan_file

    monkeypatch.setitem(sys.modules, "cmdstanpy", SimpleNamespace(CmdStanModel=DummyModel))
    monkeypatch.setattr(cu.tempfile, "gettempdir", lambda: str(tmp_path))

    code = "data {} parameters {} model {}"
    model = cu.compile_cmdstan(code, cache_tag="demo")
    assert isinstance(model, DummyModel)

    stan_path = Path(model.stan_file)
    assert stan_path.exists()
    assert stan_path.read_text(encoding="utf-8") == code
    assert stan_path.parent.name.startswith("comp_model_stan_demo_")


def test_write_text_if_changed(tmp_path):
    """Stan source helper should skip identical writes and update changed files."""
    p = tmp_path / "model.stan"
    assert cu._write_text_if_changed(p, "parameters {}")
    assert not cu._write_text_if_changed(p, "parameters {}")
    assert cu._write_text_if_changed(p, "parameters { real x; }")
    assert p.read_text(encoding="utf-8") == "parameters { real x; }"


def test_compile_cmdstan_uses_lock(tmp_path, monkeypatch):
    """compile_cmdstan should enter a per-cache compile lock."""

    class DummyModel:
        def __init__(self, stan_file: str):
            self.stan_file = stan_file

    entered: list[Path] = []

    @contextmanager
    def fake_lock(path: Path):
        entered.append(path)
        yield

    monkeypatch.setitem(sys.modules, "cmdstanpy", SimpleNamespace(CmdStanModel=DummyModel))
    monkeypatch.setattr(cu.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cu, "_file_lock", fake_lock)

    model = cu.compile_cmdstan("data {} parameters {} model {}", cache_tag="demo")
    assert isinstance(model, DummyModel)
    assert entered
    assert entered[0].name == ".compile.lock"
    assert entered[0].parent == Path(model.stan_file).parent


def test_all_stan_bodies_export_pointwise_log_lik():
    """Every Stan model body should expose per-event log-likelihood draws."""
    base = Path(cu.__file__).resolve().parent
    body_files = sorted(base.rglob("*_body.stan"))
    assert body_files, "Expected Stan body templates to exist."

    for path in body_files:
        text = path.read_text(encoding="utf-8")
        assert "vector[E] log_lik = rep_vector(0.0, E);" in text, f"Missing log_lik vector in {path}"
        assert "log_lik[e] = categorical_logit_lpmf(choice[e] |" in text, f"Missing pointwise assignment in {path}"
