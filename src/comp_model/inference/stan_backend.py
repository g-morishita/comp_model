"""CmdStan backend utilities for Stan-powered inference.

This module isolates optional :mod:`cmdstanpy` usage behind a small API. It
provides code-hash-based compile caching so repeated runs avoid recompilation.
"""

from __future__ import annotations

import hashlib
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _read_text(path: Path) -> str:
    """Read one UTF-8 text file."""

    return path.read_text(encoding="utf-8")


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Acquire an inter-process file lock when available.

    Notes
    -----
    On platforms without :mod:`fcntl`, locking is skipped.
    """

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
    """Write text only when content differs from the existing file."""

    if path.exists() and _read_text(path) == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def compile_cmdstan_model(stan_code: str, *, cache_tag: str) -> Any:
    """Compile (or reuse) a CmdStan model from Stan code.

    Parameters
    ----------
    stan_code : str
        Full Stan program source code.
    cache_tag : str
        Human-readable tag for cache directory naming.

    Returns
    -------
    Any
        ``cmdstanpy.CmdStanModel`` instance.

    Raises
    ------
    RuntimeError
        If :mod:`cmdstanpy` is unavailable or CmdStan is not installed.
    """

    try:
        from cmdstanpy import CmdStanModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "CmdStanPy is required for Stan estimators.\n"
            "Install package: pip install cmdstanpy\n"
            "Install CmdStan: python -m cmdstanpy.install_cmdstan"
        ) from exc

    code_hash = hashlib.sha256(stan_code.encode("utf-8")).hexdigest()[:16]
    workdir = Path(tempfile.gettempdir()) / f"comp_model_stan_{cache_tag}_{code_hash}"
    workdir.mkdir(parents=True, exist_ok=True)

    stan_path = workdir / "model.stan"
    lock_path = workdir / ".compile.lock"
    with _file_lock(lock_path):
        _write_text_if_changed(stan_path, stan_code)
        return CmdStanModel(stan_file=str(stan_path))


__all__ = ["compile_cmdstan_model"]

