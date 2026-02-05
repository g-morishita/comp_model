"""Profile-likelihood helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from comp_model_core.data.types import StudyData
from comp_model_core.interfaces.estimator import Estimator


def _validate_param_names(estimator: Estimator, names: Sequence[str]) -> None:
    model = getattr(estimator, "model", None)
    schema = getattr(model, "param_schema", None)
    if schema is None:
        return
    known = set(schema.names)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown parameter(s): {unknown}")


def _merge_fixed_params(
    base: Mapping[str, float] | None,
    overrides: Mapping[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    if base:
        out.update({str(k): float(v) for k, v in base.items()})
    for k, v in overrides.items():
        out[str(k)] = float(v)
    return out


def profile_likelihood_1d(
    *,
    estimator: Estimator,
    study: StudyData,
    param: str,
    grid: np.ndarray,
    rng_seed: int = 0,
    fixed_params: Mapping[str, float] | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute a 1D profile likelihood by fixing a single parameter.

    Parameters
    ----------
    estimator
        Estimator instance to fit (must accept ``fixed_params`` in ``fit``).
    study
        Study data to fit.
    param
        Name of the parameter to fix on the grid.
    grid
        1D array of parameter values.
    rng_seed
        Seed offset for per-grid-point RNGs.
    fixed_params
        Additional parameters to keep fixed at all grid points.
    show_progress
        If True, print progress updates.

    Returns
    -------
    numpy.ndarray
        Log-likelihood at each grid point (same shape as ``grid``).
    """
    param = str(param)
    _validate_param_names(estimator, [param])
    if fixed_params and param in fixed_params:
        raise ValueError(f"param {param!r} is already present in fixed_params")

    grid = np.asarray(grid, dtype=float)
    ll = np.empty(grid.shape, dtype=float)

    for i, v in enumerate(grid):
        if show_progress:
            msg = f"[{param}] {i + 1}/{len(grid)} value={float(v):.4f}"
            print(msg, end="\r", flush=True)
        fixed = _merge_fixed_params(fixed_params, {param: float(v)})
        res = estimator.fit(
            study=study,
            rng=np.random.default_rng(int(rng_seed) + i),
            fixed_params=fixed,
        )
        ll[i] = float(res.value if res.value is not None else np.nan)

    if show_progress:
        print()
    return ll


def profile_likelihood_2d(
    *,
    estimator: Estimator,
    study: StudyData,
    param_x: str,
    grid_x: np.ndarray,
    param_y: str,
    grid_y: np.ndarray,
    rng_seed: int = 0,
    fixed_params: Mapping[str, float] | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute a 2D likelihood slice by fixing two parameters on a grid.

    Parameters
    ----------
    estimator
        Estimator instance to fit (must accept ``fixed_params`` in ``fit``).
    study
        Study data to fit.
    param_x, param_y
        Parameter names to fix on the grid.
    grid_x, grid_y
        1D arrays for the two axes. Output shape is ``(len(grid_y), len(grid_x))``.
    rng_seed
        Seed offset for per-grid-point RNGs.
    fixed_params
        Additional parameters to keep fixed at all grid points.
    show_progress
        If True, print progress updates.

    Returns
    -------
    numpy.ndarray
        Log-likelihood surface with shape ``(len(grid_y), len(grid_x))``.
    """
    param_x = str(param_x)
    param_y = str(param_y)
    _validate_param_names(estimator, [param_x, param_y])
    if param_x == param_y:
        raise ValueError("param_x and param_y must be different")
    if fixed_params and (param_x in fixed_params or param_y in fixed_params):
        raise ValueError("param_x/param_y already present in fixed_params")

    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    ll = np.empty((len(grid_y), len(grid_x)), dtype=float)

    total = int(len(grid_x) * len(grid_y))
    counter = 0
    for j, yv in enumerate(grid_y):
        for i, xv in enumerate(grid_x):
            counter += 1
            if show_progress:
                msg = (
                    f"[{param_y}={float(yv):.4f}, {param_x}={float(xv):.4f}] "
                    f"{counter}/{total}"
                )
                print(msg, end="\r", flush=True)
            fixed = _merge_fixed_params(fixed_params, {param_x: float(xv), param_y: float(yv)})
            res = estimator.fit(
                study=study,
                rng=np.random.default_rng(int(rng_seed) + counter),
                fixed_params=fixed,
            )
            ll[j, i] = float(res.value if res.value is not None else np.nan)

    if show_progress:
        print()
    return ll
