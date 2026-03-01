"""Parameter-recovery workflow utilities.

This module provides a lightweight recovery pipeline that uses the canonical
runtime loop and inference outputs in ``comp_model``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionProblem
from comp_model.inference.fit_result import extract_best_fit_summary
from comp_model.runtime import SimulationConfig, run_episode

ObsT = TypeVar("ObsT")
ActionT = TypeVar("ActionT")
OutcomeT = TypeVar("OutcomeT")


@dataclass(frozen=True, slots=True)
class ParameterRecoveryCase:
    """One generate-and-fit recovery case.

    Parameters
    ----------
    case_index : int
        Zero-based case index in this run.
    simulation_seed : int
        Seed used to generate the synthetic trace.
    true_params : dict[str, float]
        Ground-truth parameters used to generate data.
    estimated_params : dict[str, float]
        Best-fit parameters returned by the fitting procedure.
    best_log_likelihood : float
        Best log-likelihood reported by the fit result.
    best_log_posterior : float | None, optional
        Best log-posterior when available (for MAP-style fits).
    """

    case_index: int
    simulation_seed: int
    true_params: dict[str, float]
    estimated_params: dict[str, float]
    best_log_likelihood: float
    best_log_posterior: float | None = None


@dataclass(frozen=True, slots=True)
class ParameterRecoveryResult:
    """Output summary for parameter-recovery runs.

    Parameters
    ----------
    cases : tuple[ParameterRecoveryCase, ...]
        Per-case recovery records.
    mean_absolute_error : dict[str, float]
        Mean absolute error across cases for each shared parameter key.
    mean_signed_error : dict[str, float]
        Mean signed error (estimate minus truth) across cases for each shared
        parameter key.
    """

    cases: tuple[ParameterRecoveryCase, ...]
    mean_absolute_error: dict[str, float]
    mean_signed_error: dict[str, float]


def run_parameter_recovery(
    *,
    problem_factory: Callable[[], DecisionProblem[ObsT, ActionT, OutcomeT]],
    model_factory: Callable[[dict[str, float]], AgentModel[ObsT, ActionT, OutcomeT]],
    fit_function: Callable[[Any], Any],
    true_parameter_sets: Sequence[Mapping[str, float]],
    n_trials: int,
    seed: int = 0,
) -> ParameterRecoveryResult:
    """Run simulation-based parameter recovery.

    Parameters
    ----------
    problem_factory : Callable[[], DecisionProblem[ObsT, ActionT, OutcomeT]]
        Factory returning a fresh problem instance for one synthetic dataset.
    model_factory : Callable[[dict[str, float]], AgentModel[ObsT, ActionT, OutcomeT]]
        Factory returning a generating model from true parameters.
    fit_function : Callable[[Any], Any]
        Function that fits one generated trace and returns a supported
        inference fit result (MLE-style or MAP-style).
    true_parameter_sets : Sequence[Mapping[str, float]]
        Collection of true parameter mappings to recover.
    n_trials : int
        Number of trials per synthetic dataset.
    seed : int, optional
        Master seed used to derive per-case simulation seeds.

    Returns
    -------
    ParameterRecoveryResult
        Recovery records and aggregate error summaries.

    Raises
    ------
    ValueError
        If no true parameter sets are provided or ``n_trials`` is non-positive.
    """

    if not true_parameter_sets:
        raise ValueError("true_parameter_sets must not be empty")
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")

    rng = np.random.default_rng(seed)
    cases: list[ParameterRecoveryCase] = []

    for case_index, params in enumerate(true_parameter_sets):
        simulation_seed = int(rng.integers(0, 2**31 - 1))

        generating_model = model_factory({name: float(value) for name, value in params.items()})
        trace = run_episode(
            problem=problem_factory(),
            model=generating_model,
            config=SimulationConfig(n_trials=n_trials, seed=simulation_seed),
        )
        fit_result = fit_function(trace)
        best = extract_best_fit_summary(fit_result)

        cases.append(
            ParameterRecoveryCase(
                case_index=case_index,
                simulation_seed=simulation_seed,
                true_params={name: float(value) for name, value in params.items()},
                estimated_params={name: float(value) for name, value in best.params.items()},
                best_log_likelihood=float(best.log_likelihood),
                best_log_posterior=(
                    float(best.log_posterior)
                    if best.log_posterior is not None
                    else None
                ),
            )
        )

    mean_absolute_error, mean_signed_error = _aggregate_parameter_errors(cases)
    return ParameterRecoveryResult(
        cases=tuple(cases),
        mean_absolute_error=mean_absolute_error,
        mean_signed_error=mean_signed_error,
    )


def _aggregate_parameter_errors(
    cases: Sequence[ParameterRecoveryCase],
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate absolute and signed recovery errors by parameter key."""

    common_keys: set[str] | None = None
    for case in cases:
        keys = set(case.true_params) & set(case.estimated_params)
        common_keys = keys if common_keys is None else (common_keys & keys)

    if not common_keys:
        return {}, {}

    mean_abs_error: dict[str, float] = {}
    mean_signed_error: dict[str, float] = {}
    for key in sorted(common_keys):
        signed = np.asarray(
            [case.estimated_params[key] - case.true_params[key] for case in cases],
            dtype=float,
        )
        mean_abs_error[key] = float(np.mean(np.abs(signed)))
        mean_signed_error[key] = float(np.mean(signed))

    return mean_abs_error, mean_signed_error


__all__ = ["ParameterRecoveryCase", "ParameterRecoveryResult", "run_parameter_recovery"]
