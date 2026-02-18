"""CmdStanPy helpers.

This module isolates the optional dependency on :mod:`cmdstanpy` and provides a
small API used by :mod:`comp_model_impl.estimators.stan`.

Notes
-----
Stan programs are assembled from shared prior functions and per-model templates.
Compiled binaries are cached in a temporary directory based on a hash of the
Stan code and a cache tag.

Examples
--------
>>> from comp_model_impl.estimators.stan.cmdstan_utils import load_stan_code
>>> # code = load_stan_code(kind="indiv", model_name="vs_within_subject")  # doctest: +SKIP
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import hashlib
import tempfile


def _read_text(p: Path) -> str:
    """Read UTF-8 text from a path.

    Parameters
    ----------
    p : pathlib.Path
        File path to read.

    Returns
    -------
    str
        File contents.
    """
    return p.read_text(encoding="utf-8")


def load_stan_code(kind: str, model_name: str) -> str:
    """Load the full Stan program (common functions + model body).

    Parameters
    ----------
    kind : {"indiv", "hier"}
        Which template body to load.
    model_name : str
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

    Examples
    --------
    >>> from comp_model_impl.estimators.stan.cmdstan_utils import load_stan_code
    >>> # code = load_stan_code(kind="hier", model_name="vs_within_subject")  # doctest: +SKIP
    """
    base = Path(__file__).resolve().parent
    common = _read_text(base / "common" / "prior_functions.stan")
    body = _read_text(base / model_name / f"{kind}_body.stan")
    return common + "\n\n" + body


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Acquire an inter-process lock using a lock file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        try:
            import fcntl
        except ImportError:
            yield
            return

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_text_if_changed(path: Path, text: str) -> bool:
    """Write text when content differs from the existing file."""
    if path.exists() and _read_text(path) == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


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

    Examples
    --------
    >>> from comp_model_impl.estimators.stan.cmdstan_utils import compile_cmdstan
    >>> # model = compile_cmdstan(stan_code="data {} parameters {} model {}", cache_tag="demo")  # doctest: +SKIP
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
    lock_path = workdir / ".compile.lock"
    with _file_lock(lock_path):
        _write_text_if_changed(stan_path, stan_code)
        return CmdStanModel(stan_file=str(stan_path))
