"""Maximum-likelihood estimation interfaces and baseline implementation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

from comp_model.core.contracts import AgentModel
from comp_model.core.events import EpisodeTrace
from comp_model.core.requirements import ComponentRequirements
from comp_model.inference.compatibility import CompatibilityReport, assert_trace_compatible, check_trace_compatibility
from comp_model.inference.likelihood import LikelihoodProgram


@dataclass(frozen=True, slots=True)
class MLECandidate:
    """One parameter candidate evaluated by an MLE estimator.

    Parameters
    ----------
    params : dict[str, float]
        Evaluated parameter set.
    log_likelihood : float
        Total action log-likelihood for ``params``.
    """

    params: dict[str, float]
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class MLEFitResult:
    """MLE fit output.

    Parameters
    ----------
    best : MLECandidate
        Candidate with the maximum log-likelihood.
    candidates : tuple[MLECandidate, ...]
        All evaluated candidates.
    compatibility : CompatibilityReport | None
        Compatibility report when requirements were checked.
    """

    best: MLECandidate
    candidates: tuple[MLECandidate, ...]
    compatibility: CompatibilityReport | None = None


class GridSearchMLEEstimator:
    """Deterministic grid-search MLE estimator.

    Parameters
    ----------
    likelihood_program : LikelihoodProgram
        Likelihood evaluator used for each candidate model.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating a fresh model instance for one candidate parameter set.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.

    Notes
    -----
    This implementation is intentionally simple and serves as a baseline API
    for future optimizer-backed estimators.
    """

    def __init__(
        self,
        likelihood_program: LikelihoodProgram,
        model_factory: Callable[[dict[str, float]], AgentModel],
        requirements: ComponentRequirements | None = None,
    ) -> None:
        self._likelihood_program = likelihood_program
        self._model_factory = model_factory
        self._requirements = requirements

    def fit(self, trace: EpisodeTrace, parameter_grid: dict[str, list[float]]) -> MLEFitResult:
        """Fit model parameters via exhaustive grid search.

        Parameters
        ----------
        trace : EpisodeTrace
            Observed event trace used for likelihood evaluation.
        parameter_grid : dict[str, list[float]]
            Parameter grid values. Every combination is evaluated.

        Returns
        -------
        MLEFitResult
            Full candidate list and best-fit parameters.

        Raises
        ------
        ValueError
            If compatibility fails or the parameter grid is empty.
        """

        compatibility: CompatibilityReport | None = None
        if self._requirements is not None:
            compatibility = check_trace_compatibility(trace, self._requirements)
            assert_trace_compatible(trace, self._requirements)

        candidates: list[MLECandidate] = []
        for params in _iter_parameter_grid(parameter_grid):
            model = self._model_factory(params)
            replay_result = self._likelihood_program.evaluate(trace, model)
            candidates.append(
                MLECandidate(
                    params=dict(params),
                    log_likelihood=float(replay_result.total_log_likelihood),
                )
            )

        if not candidates:
            raise ValueError("parameter_grid must include at least one candidate")

        best = max(candidates, key=lambda item: item.log_likelihood)
        return MLEFitResult(best=best, candidates=tuple(candidates), compatibility=compatibility)


def _iter_parameter_grid(parameter_grid: dict[str, list[float]]) -> tuple[dict[str, float], ...]:
    """Generate deterministic parameter combinations from a grid.

    Parameters
    ----------
    parameter_grid : dict[str, list[float]]
        Mapping from parameter names to candidate values.

    Returns
    -------
    tuple[dict[str, float], ...]
        Candidate parameter dictionaries in deterministic key order.
    """

    if not parameter_grid:
        return tuple()

    names = tuple(sorted(parameter_grid))
    values = []
    for name in names:
        grid_values = tuple(float(v) for v in parameter_grid[name])
        if not grid_values:
            raise ValueError(f"parameter {name!r} has no candidate values")
        values.append(grid_values)

    combinations = []
    for candidate_values in product(*values):
        combinations.append({name: value for name, value in zip(names, candidate_values, strict=True)})
    return tuple(combinations)
