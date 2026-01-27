"""CmdStanPy helpers.

This module isolates the optional dependency on :mod:`cmdstanpy` and provides a
very small API used by :mod:`comp_model_impl.estimators.stan`.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import tempfile


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def load_stan_code(kind: str, model_name: str) -> str:
    """Load the full Stan program (common functions + model body).

    Parameters
    ----------
    kind : {"indiv", "hier"}
        Which template body to load.
    model_name : {"vs", "vicarious_rl"}
        Template family directory name.

    Returns
    -------
    str
        Stan code as a single string.

    Notes
    -----
    The Stan templates live in this package under::

        estimators/stan/common/prior_functions.stan
        estimators/stan/<model_name>/<kind>_body.stan
    """
    base = Path(__file__).resolve().parent
    common = _read_text(base / "common" / "prior_functions.stan")
    body = _read_text(base / model_name / f"{kind}_body.stan")
    return common + "\n\n" + body


def compile_cmdstan(stan_code: str, cache_tag: str):
    """Compile (or load a cached) CmdStan model from Stan code.

    Parameters
    ----------
    stan_code : str
        Full Stan program text.
    cache_tag : str
        A short name used in the cache directory path.

    Returns
    -------
    cmdstanpy.CmdStanModel
        A compiled CmdStan model.

    Raises
    ------
    RuntimeError
        If :mod:`cmdstanpy` is not installed or CmdStan is not available.
    """
    try:
        from cmdstanpy import CmdStanModel
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "CmdStanPy is required to run Stan-based estimators.\n"
            "Install: pip install cmdstanpy\n"
            "Install CmdStan: python -m cmdstanpy.install_cmdstan\n"
        ) from e

    h = hashlib.sha256(stan_code.encode("utf-8")).hexdigest()[:16]
    workdir = Path(tempfile.gettempdir()) / f"comp_model_stan_{cache_tag}_{h}"
    workdir.mkdir(parents=True, exist_ok=True)

    stan_path = workdir / "model.stan"
    stan_path.write_text(stan_code, encoding="utf-8")

    return CmdStanModel(stan_file=str(stan_path))
