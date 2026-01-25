from __future__ import annotations
from pathlib import Path
import hashlib
import tempfile
import os

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def load_stan_code(kind: str, model_name: str) -> str:
    """
    kind: "indiv" or "hier"
    model_name: "vs" or "vicarious_rl"
    """
    base = Path(__file__).resolve().parents[0]
    common = _read_text(base / "common" / "prior_functions.stan")
    body = _read_text(base / model_name / f"{kind}_body.stan")
    return common + "\n\n" + body

def compile_cmdstan(stan_code: str, cache_tag: str):
    try:
        from cmdstanpy import CmdStanModel
    except Exception as e:
        raise RuntimeError(
            "CmdStanPy required.\n"
            "pip install cmdstanpy\n"
            "python -m cmdstanpy.install_cmdstan\n"
        ) from e

    h = hashlib.sha256(stan_code.encode("utf-8")).hexdigest()[:16]
    workdir = Path(tempfile.gettempdir()) / f"comp_model_stan_{cache_tag}_{h}"
    workdir.mkdir(parents=True, exist_ok=True)
    stan_path = workdir / "model.stan"
    stan_path.write_text(stan_code, encoding="utf-8")
    return CmdStanModel(stan_file=str(stan_path))
