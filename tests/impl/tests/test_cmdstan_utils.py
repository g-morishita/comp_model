"""Tests for CmdStan utility helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
