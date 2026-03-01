"""Parity benchmark helpers for v1 fixture comparison workflows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comp_model.core.data import TrialDecision, trace_from_trial_decisions
from comp_model.inference import ActionReplayLikelihood, LikelihoodProgram
from comp_model.plugins import PluginRegistry, build_default_registry


@dataclass(frozen=True, slots=True)
class ParityFixtureCase:
    """One parity benchmark case loaded from fixture data.

    Parameters
    ----------
    name : str
        Human-readable case label.
    model_component_id : str
        Model plugin component ID to evaluate.
    params : dict[str, float]
        Free parameter values supplied to the model constructor.
    expected_log_likelihood : float
        Reference log-likelihood from v1 fixture output.
    trial_decisions : tuple[TrialDecision, ...]
        Canonical trial-decision rows for replay.
    model_kwargs : dict[str, Any], optional
        Fixed constructor kwargs merged with ``params``.
    """

    name: str
    model_component_id: str
    params: dict[str, float]
    expected_log_likelihood: float
    trial_decisions: tuple[TrialDecision, ...]
    model_kwargs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ParityCaseResult:
    """Evaluation result for one parity fixture case."""

    name: str
    model_component_id: str
    expected_log_likelihood: float
    observed_log_likelihood: float
    absolute_error: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ParityBenchmarkResult:
    """Aggregated parity benchmark output across all evaluated cases."""

    case_results: tuple[ParityCaseResult, ...]
    atol: float
    rtol: float

    @property
    def n_cases(self) -> int:
        """Return number of evaluated cases."""

        return len(self.case_results)

    @property
    def n_passed(self) -> int:
        """Return number of cases that pass tolerance checks."""

        return sum(1 for item in self.case_results if item.passed)

    @property
    def n_failed(self) -> int:
        """Return number of failed cases."""

        return self.n_cases - self.n_passed


def run_parity_benchmark(
    cases: tuple[ParityFixtureCase, ...] | list[ParityFixtureCase],
    *,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> ParityBenchmarkResult:
    """Run likelihood parity checks on fixture cases.

    Parameters
    ----------
    cases : tuple[ParityFixtureCase, ...] | list[ParityFixtureCase]
        Fixture cases to evaluate.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional replay likelihood evaluator.
    atol : float, optional
        Absolute error tolerance.
    rtol : float, optional
        Relative error tolerance.

    Returns
    -------
    ParityBenchmarkResult
        Per-case results and aggregate pass/fail counts.
    """

    reg = registry if registry is not None else build_default_registry()
    like = likelihood_program if likelihood_program is not None else ActionReplayLikelihood()
    rows: list[ParityCaseResult] = []

    for case in cases:
        model = reg.create_model(
            case.model_component_id,
            **_merge_kwargs(case.model_kwargs, case.params),
        )
        trace = trace_from_trial_decisions(case.trial_decisions)
        replay = like.evaluate(trace, model)
        observed = float(replay.total_log_likelihood)
        expected = float(case.expected_log_likelihood)
        abs_error = abs(observed - expected)
        tolerance = float(atol) + float(rtol) * abs(expected)
        rows.append(
            ParityCaseResult(
                name=case.name,
                model_component_id=case.model_component_id,
                expected_log_likelihood=expected,
                observed_log_likelihood=observed,
                absolute_error=abs_error,
                passed=abs_error <= tolerance,
            )
        )

    return ParityBenchmarkResult(
        case_results=tuple(rows),
        atol=float(atol),
        rtol=float(rtol),
    )


def load_parity_fixture_file(path: str | Path) -> tuple[ParityFixtureCase, ...]:
    """Load JSON fixture cases for parity benchmark execution.

    Parameters
    ----------
    path : str | pathlib.Path
        Fixture JSON path.

    Returns
    -------
    tuple[ParityFixtureCase, ...]
        Parsed fixture cases.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("fixture root must be a JSON object")
    rows_raw = raw.get("cases")
    if not isinstance(rows_raw, list):
        raise ValueError("fixture.cases must be an array")

    out: list[ParityFixtureCase] = []
    for index, item in enumerate(rows_raw):
        if not isinstance(item, dict):
            raise ValueError(f"fixture.cases[{index}] must be an object")
        decisions_raw = item.get("trial_decisions")
        if not isinstance(decisions_raw, list):
            raise ValueError(f"fixture.cases[{index}].trial_decisions must be an array")
        decisions = tuple(_trial_decision_from_mapping(row, index=index) for row in decisions_raw)
        model_kwargs_raw = item.get("model_kwargs", {})
        if not isinstance(model_kwargs_raw, dict):
            raise ValueError(f"fixture.cases[{index}].model_kwargs must be an object")
        params_raw = item.get("params")
        if not isinstance(params_raw, dict):
            raise ValueError(f"fixture.cases[{index}].params must be an object")
        out.append(
            ParityFixtureCase(
                name=str(item.get("name", f"case_{index}")),
                model_component_id=str(item["model_component_id"]),
                params={str(key): float(value) for key, value in params_raw.items()},
                expected_log_likelihood=float(item["expected_log_likelihood"]),
                trial_decisions=decisions,
                model_kwargs=dict(model_kwargs_raw),
            )
        )
    return tuple(out)


def write_parity_benchmark_csv(result: ParityBenchmarkResult, path: str | Path) -> Path:
    """Write parity benchmark case results to CSV.

    Parameters
    ----------
    result : ParityBenchmarkResult
        Benchmark result object.
    path : str | pathlib.Path
        Output CSV path.

    Returns
    -------
    pathlib.Path
        Output path.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "model_component_id",
                "expected_log_likelihood",
                "observed_log_likelihood",
                "absolute_error",
                "passed",
            ],
        )
        writer.writeheader()
        for row in result.case_results:
            writer.writerow(
                {
                    "name": row.name,
                    "model_component_id": row.model_component_id,
                    "expected_log_likelihood": row.expected_log_likelihood,
                    "observed_log_likelihood": row.observed_log_likelihood,
                    "absolute_error": row.absolute_error,
                    "passed": row.passed,
                }
            )
    return target


def _trial_decision_from_mapping(raw: Any, *, index: int) -> TrialDecision:
    """Parse one trial-decision row from fixture JSON."""

    if not isinstance(raw, dict):
        raise ValueError(f"trial_decisions[{index}] must be an object")
    available_actions_raw = raw.get("available_actions")
    if not isinstance(available_actions_raw, list):
        raise ValueError(f"trial_decisions[{index}].available_actions must be an array")
    return TrialDecision(
        trial_index=int(raw["trial_index"]),
        decision_index=int(raw.get("decision_index", 0)),
        actor_id=str(raw.get("actor_id", "subject")),
        available_actions=tuple(available_actions_raw),
        action=raw["action"],
        observation=raw.get("observation"),
        outcome=raw.get("outcome"),
    )


def _merge_kwargs(base: dict[str, Any], override: dict[str, float]) -> dict[str, Any]:
    """Merge fixed constructor kwargs with free parameters."""

    merged = dict(base)
    merged.update(override)
    return merged


__all__ = [
    "ParityBenchmarkResult",
    "ParityCaseResult",
    "ParityFixtureCase",
    "load_parity_fixture_file",
    "run_parity_benchmark",
    "write_parity_benchmark_csv",
]

