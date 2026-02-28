"""Profile-likelihood utilities.

The functions in this module are intentionally generic and independent from any
specific estimator implementation. They operate on user-provided log-likelihood
evaluators and return structured profile summaries.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProfilePoint1D:
    """One 1D profile-likelihood point.

    Parameters
    ----------
    value : float
        Parameter value on the profiling grid.
    log_likelihood : float
        Evaluated log-likelihood at ``value``.
    """

    value: float
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class ProfileLikelihood1DResult:
    """Result container for 1D profile-likelihood evaluation.

    Parameters
    ----------
    parameter_name : str
        Name of profiled parameter.
    points : tuple[ProfilePoint1D, ...]
        Evaluated profile points in grid order.
    best_value : float
        Grid value with maximal log-likelihood.
    best_log_likelihood : float
        Maximum log-likelihood on the grid.
    """

    parameter_name: str
    points: tuple[ProfilePoint1D, ...]
    best_value: float
    best_log_likelihood: float


@dataclass(frozen=True, slots=True)
class ProfilePoint2D:
    """One 2D profile-likelihood point.

    Parameters
    ----------
    x_value : float
        Grid value on x-parameter axis.
    y_value : float
        Grid value on y-parameter axis.
    log_likelihood : float
        Evaluated log-likelihood at this 2D point.
    """

    x_value: float
    y_value: float
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class ProfileLikelihood2DResult:
    """Result container for 2D profile-likelihood evaluation.

    Parameters
    ----------
    x_parameter_name : str
        Name of x-axis parameter.
    y_parameter_name : str
        Name of y-axis parameter.
    points : tuple[ProfilePoint2D, ...]
        Evaluated 2D points in nested-loop order over ``x_grid`` then ``y_grid``.
    best_x_value : float
        X-value with maximal log-likelihood.
    best_y_value : float
        Y-value with maximal log-likelihood.
    best_log_likelihood : float
        Maximum log-likelihood on the 2D grid.
    grid_shape : tuple[int, int]
        Evaluated grid dimensions ``(len(x_grid), len(y_grid))``.
    """

    x_parameter_name: str
    y_parameter_name: str
    points: tuple[ProfilePoint2D, ...]
    best_x_value: float
    best_y_value: float
    best_log_likelihood: float
    grid_shape: tuple[int, int]


def profile_likelihood_1d(
    *,
    parameter_name: str,
    grid: Sequence[float],
    evaluate_log_likelihood: Callable[[float], float],
) -> ProfileLikelihood1DResult:
    """Evaluate 1D profile likelihood over a fixed grid.

    Parameters
    ----------
    parameter_name : str
        Name of profiled parameter.
    grid : Sequence[float]
        Values to evaluate.
    evaluate_log_likelihood : Callable[[float], float]
        Function returning log-likelihood for one grid value.

    Returns
    -------
    ProfileLikelihood1DResult
        Structured profile summary.

    Raises
    ------
    ValueError
        If ``grid`` is empty.
    """

    if not grid:
        raise ValueError("grid must not be empty")

    points = tuple(
        ProfilePoint1D(value=float(value), log_likelihood=float(evaluate_log_likelihood(float(value))))
        for value in grid
    )

    best = max(points, key=lambda point: point.log_likelihood)
    return ProfileLikelihood1DResult(
        parameter_name=str(parameter_name),
        points=points,
        best_value=best.value,
        best_log_likelihood=best.log_likelihood,
    )


def profile_likelihood_2d(
    *,
    x_parameter_name: str,
    y_parameter_name: str,
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    evaluate_log_likelihood: Callable[[float, float], float],
) -> ProfileLikelihood2DResult:
    """Evaluate 2D profile likelihood over a Cartesian grid.

    Parameters
    ----------
    x_parameter_name : str
        Name of x-axis parameter.
    y_parameter_name : str
        Name of y-axis parameter.
    x_grid : Sequence[float]
        X-axis grid values.
    y_grid : Sequence[float]
        Y-axis grid values.
    evaluate_log_likelihood : Callable[[float, float], float]
        Function returning log-likelihood for one 2D grid point.

    Returns
    -------
    ProfileLikelihood2DResult
        Structured profile summary.

    Raises
    ------
    ValueError
        If either grid is empty.
    """

    if not x_grid:
        raise ValueError("x_grid must not be empty")
    if not y_grid:
        raise ValueError("y_grid must not be empty")

    points = tuple(
        ProfilePoint2D(
            x_value=float(x_value),
            y_value=float(y_value),
            log_likelihood=float(evaluate_log_likelihood(float(x_value), float(y_value))),
        )
        for x_value in x_grid
        for y_value in y_grid
    )

    best = max(points, key=lambda point: point.log_likelihood)
    return ProfileLikelihood2DResult(
        x_parameter_name=str(x_parameter_name),
        y_parameter_name=str(y_parameter_name),
        points=points,
        best_x_value=best.x_value,
        best_y_value=best.y_value,
        best_log_likelihood=best.log_likelihood,
        grid_shape=(len(x_grid), len(y_grid)),
    )


__all__ = [
    "ProfileLikelihood1DResult",
    "ProfileLikelihood2DResult",
    "ProfilePoint1D",
    "ProfilePoint2D",
    "profile_likelihood_1d",
    "profile_likelihood_2d",
]
