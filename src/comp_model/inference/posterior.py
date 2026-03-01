"""Posterior sample containers and summary utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PosteriorSamples:
    """Posterior draws for model parameters.

    Parameters
    ----------
    parameter_draws : Mapping[str, numpy.ndarray]
        Parameter name to 1D draw array mapping. All arrays must share equal
        length and contain at least one draw.
    """

    parameter_draws: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.parameter_draws:
            raise ValueError("parameter_draws must not be empty")

        draw_count: int | None = None
        for name, values in self.parameter_draws.items():
            array = np.asarray(values, dtype=float)
            if array.ndim != 1:
                raise ValueError(f"parameter {name!r} draws must be 1D")
            if array.size == 0:
                raise ValueError(f"parameter {name!r} draws must not be empty")
            if draw_count is None:
                draw_count = int(array.size)
            elif int(array.size) != draw_count:
                raise ValueError("all parameter draw arrays must have equal length")

    @property
    def n_draws(self) -> int:
        """Return number of posterior draws."""

        first = next(iter(self.parameter_draws.values()))
        return int(np.asarray(first).size)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return sorted parameter names."""

        return tuple(sorted(self.parameter_draws))

    def draws(self, parameter_name: str) -> np.ndarray:
        """Return draw array for one parameter."""

        if parameter_name not in self.parameter_draws:
            available = ", ".join(self.parameter_names)
            raise KeyError(f"unknown parameter {parameter_name!r}; available: {available}")
        return np.asarray(self.parameter_draws[parameter_name], dtype=float)

    def mean(self, parameter_name: str) -> float:
        """Return posterior mean for one parameter."""

        return float(np.mean(self.draws(parameter_name)))

    def std(self, parameter_name: str, *, ddof: int = 1) -> float:
        """Return posterior standard deviation for one parameter."""

        return float(np.std(self.draws(parameter_name), ddof=ddof))

    def quantile(self, parameter_name: str, q: float) -> float:
        """Return posterior quantile for one parameter."""

        return float(np.quantile(self.draws(parameter_name), q))


@dataclass(frozen=True, slots=True)
class PosteriorParameterSummary:
    """Posterior summary statistics for one parameter.

    Parameters
    ----------
    parameter_name : str
        Parameter identifier.
    mean : float
        Posterior mean.
    std : float
        Posterior standard deviation.
    quantiles : dict[float, float]
        Requested quantile values keyed by quantile probability.
    """

    parameter_name: str
    mean: float
    std: float
    quantiles: dict[float, float]


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    """Posterior summary for all parameters."""

    n_draws: int
    parameters: tuple[PosteriorParameterSummary, ...]

    def by_name(self) -> dict[str, PosteriorParameterSummary]:
        """Return parameter summaries keyed by parameter name."""

        return {item.parameter_name: item for item in self.parameters}


def summarize_posterior(
    samples: PosteriorSamples,
    *,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> PosteriorSummary:
    """Summarize posterior draws with moments and quantiles.

    Parameters
    ----------
    samples : PosteriorSamples
        Posterior draw container.
    quantiles : Sequence[float], optional
        Quantiles to report for each parameter.

    Returns
    -------
    PosteriorSummary
        Summary statistics for all parameters.
    """

    q_values = tuple(float(value) for value in quantiles)
    for value in q_values:
        if value < 0.0 or value > 1.0:
            raise ValueError("quantiles must lie in [0, 1]")

    parameters: list[PosteriorParameterSummary] = []
    for parameter_name in samples.parameter_names:
        quantile_map = {
            value: float(samples.quantile(parameter_name, value))
            for value in q_values
        }
        parameters.append(
            PosteriorParameterSummary(
                parameter_name=parameter_name,
                mean=float(samples.mean(parameter_name)),
                std=float(samples.std(parameter_name)),
                quantiles=quantile_map,
            )
        )

    return PosteriorSummary(
        n_draws=samples.n_draws,
        parameters=tuple(parameters),
    )


def posterior_summary_records(summary: PosteriorSummary) -> list[dict[str, float | str]]:
    """Convert posterior summary into row records."""

    rows: list[dict[str, float | str]] = []
    for parameter in summary.parameters:
        row: dict[str, float | str] = {
            "parameter_name": parameter.parameter_name,
            "mean": float(parameter.mean),
            "std": float(parameter.std),
            "n_draws": float(summary.n_draws),
        }
        for quantile, value in sorted(parameter.quantiles.items()):
            row[f"q{quantile:.3f}"] = float(value)
        rows.append(row)
    return rows


__all__ = [
    "PosteriorParameterSummary",
    "PosteriorSamples",
    "PosteriorSummary",
    "posterior_summary_records",
    "summarize_posterior",
]
