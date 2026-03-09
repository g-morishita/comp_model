"""Parameter-recovery workflow utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionProblem
from comp_model.inference.best_fit_summary import extract_best_fit_summary
from comp_model.inference.transforms import (
    ParameterTransform,
    identity_transform,
    positive_log_transform,
    unit_interval_logit_transform,
)
from comp_model.runtime import SimulationConfig, run_episode

ObsT = TypeVar("ObsT")
ActionT = TypeVar("ActionT")
OutcomeT = TypeVar("OutcomeT")
DistributionSpec: TypeAlias = Mapping[str, Any]
SamplingSpec: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ParameterRecoveryCase:
    """One generate-and-fit recovery case."""

    case_index: int
    simulation_seed: int
    true_params: dict[str, float]
    estimated_params: dict[str, float]
    best_log_likelihood: float


@dataclass(frozen=True, slots=True)
class ParameterRecoveryResult:
    """Output summary for parameter-recovery runs."""

    cases: tuple[ParameterRecoveryCase, ...]
    mean_absolute_error: dict[str, float]
    mean_signed_error: dict[str, float]
    true_estimate_correlation: dict[str, float | None]


def run_parameter_recovery(
    *,
    problem_factory: Callable[[], DecisionProblem[ObsT, ActionT, OutcomeT]] | None = None,
    model_factory: Callable[[dict[str, float]], AgentModel[ObsT, ActionT, OutcomeT]],
    fit_function: Callable[[Any], Any],
    true_parameter_sets: Sequence[Mapping[str, float]] | None = None,
    true_parameter_distributions: Mapping[str, DistributionSpec] | None = None,
    true_parameter_sampling: SamplingSpec | None = None,
    n_parameter_sets: int | None = None,
    n_trials: int,
    seed: int = 0,
    trace_factory: Callable[[AgentModel[ObsT, ActionT, OutcomeT], int], Any] | None = None,
) -> ParameterRecoveryResult:
    """Run simulation-based parameter recovery."""

    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if trace_factory is None and problem_factory is None:
        raise ValueError("either problem_factory or trace_factory must be provided")
    resolved_true_parameter_sets = resolve_true_parameter_sets(
        true_parameter_sets=true_parameter_sets,
        true_parameter_distributions=true_parameter_distributions,
        true_parameter_sampling=true_parameter_sampling,
        n_parameter_sets=n_parameter_sets,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    cases: list[ParameterRecoveryCase] = []

    for case_index, params in enumerate(resolved_true_parameter_sets):
        simulation_seed = int(rng.integers(0, 2**31 - 1))
        generating_model = model_factory({name: float(value) for name, value in params.items()})
        if trace_factory is not None:
            trace = trace_factory(generating_model, simulation_seed)
        else:
            assert problem_factory is not None
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
            )
        )

    (
        mean_absolute_error,
        mean_signed_error,
        true_estimate_correlation,
    ) = _aggregate_parameter_errors(cases)
    return ParameterRecoveryResult(
        cases=tuple(cases),
        mean_absolute_error=mean_absolute_error,
        mean_signed_error=mean_signed_error,
        true_estimate_correlation=true_estimate_correlation,
    )


def resolve_true_parameter_sets(
    *,
    true_parameter_sets: Sequence[Mapping[str, float]] | None = None,
    true_parameter_distributions: Mapping[str, DistributionSpec] | None = None,
    true_parameter_sampling: SamplingSpec | None = None,
    n_parameter_sets: int | None = None,
    seed: int = 0,
) -> tuple[dict[str, float], ...]:
    """Resolve true parameter mappings from one declared source."""

    has_sets = true_parameter_sets is not None
    has_distributions = true_parameter_distributions is not None
    has_sampling = true_parameter_sampling is not None

    selected_count = int(has_sets) + int(has_distributions) + int(has_sampling)
    if selected_count > 1:
        raise ValueError(
            "provide exactly one of true_parameter_sets, "
            "true_parameter_distributions, or true_parameter_sampling"
        )
    if selected_count == 0:
        raise ValueError(
            "one of true_parameter_sets, true_parameter_distributions, or "
            "true_parameter_sampling must be provided"
        )

    if has_sets:
        assert true_parameter_sets is not None
        rows = tuple(
            {str(key): float(value) for key, value in params.items()}
            for params in true_parameter_sets
        )
        if not rows:
            raise ValueError("true_parameter_sets must not be empty")
        return rows

    if has_distributions:
        assert true_parameter_distributions is not None
        if n_parameter_sets is None:
            raise ValueError(
                "n_parameter_sets is required when true_parameter_distributions is provided"
            )
        return sample_true_parameter_sets_from_distributions(
            true_parameter_distributions=true_parameter_distributions,
            n_parameter_sets=int(n_parameter_sets),
            seed=seed,
        )

    assert true_parameter_sampling is not None
    return sample_true_parameter_sets_from_sampling(
        true_parameter_sampling=true_parameter_sampling,
        n_parameter_sets=n_parameter_sets,
        seed=seed,
    )


def sample_true_parameter_sets_from_distributions(
    true_parameter_distributions: Mapping[str, DistributionSpec],
    *,
    n_parameter_sets: int,
    seed: int = 0,
) -> tuple[dict[str, float], ...]:
    """Sample true parameter dictionaries from independent distributions."""

    if not true_parameter_distributions:
        raise ValueError("true_parameter_distributions must not be empty")
    if n_parameter_sets <= 0:
        raise ValueError("n_parameter_sets must be > 0")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(n_parameter_sets):
        params: dict[str, float] = {}
        for name, spec in true_parameter_distributions.items():
            params[str(name)] = float(_sample_from_distribution(spec, rng=rng))
        rows.append(params)
    return tuple(rows)


def sample_true_parameter_sets_from_sampling(
    true_parameter_sampling: SamplingSpec,
    *,
    n_parameter_sets: int | None = None,
    seed: int = 0,
) -> tuple[dict[str, float], ...]:
    """Sample true parameter dictionaries from advanced sampling config."""

    if not isinstance(true_parameter_sampling, Mapping):
        raise ValueError("true_parameter_sampling must be an object")

    count_raw = true_parameter_sampling.get("n_parameter_sets", n_parameter_sets)
    if count_raw is None:
        raise ValueError("n_parameter_sets is required for true_parameter_sampling")
    count = int(count_raw)
    if count <= 0:
        raise ValueError("n_parameter_sets must be > 0")

    mode = str(true_parameter_sampling.get("mode", "independent")).strip()
    if mode == "fixed":
        raise ValueError("sampling.mode='fixed' is not supported")
    if mode not in {"independent", "hierarchical"}:
        raise ValueError("sampling.mode must be one of {'independent', 'hierarchical'}")

    space = str(true_parameter_sampling.get("space", "parameter")).strip()
    if space not in {"parameter", "z"}:
        raise ValueError("sampling.space must be one of {'parameter', 'z'}")

    if mode == "independent":
        distributions = true_parameter_sampling.get("distributions")
        if not isinstance(distributions, Mapping) or not distributions:
            raise ValueError("sampling.distributions must be a non-empty object for independent mode")
        return sample_true_parameter_sets_from_distributions(
            true_parameter_distributions=distributions,
            n_parameter_sets=count,
            seed=seed,
        )

    return _sample_hierarchical_parameter_sets(
        true_parameter_sampling,
        n_parameter_sets=count,
        seed=seed,
        space=space,
    )


def _sample_hierarchical_parameter_sets(
    sampling_cfg: SamplingSpec,
    *,
    n_parameter_sets: int,
    seed: int,
    space: str,
) -> tuple[dict[str, float], ...]:
    """Sample parameter sets from hierarchical population specifications."""

    population = sampling_cfg.get("population")
    individual_sd = sampling_cfg.get("individual_sd")
    if not isinstance(population, Mapping) or not population:
        raise ValueError("sampling.population must be a non-empty object for hierarchical mode")
    if not isinstance(individual_sd, Mapping) or not individual_sd:
        raise ValueError("sampling.individual_sd must be a non-empty object for hierarchical mode")

    transforms_cfg = sampling_cfg.get("transforms", {})
    if not isinstance(transforms_cfg, Mapping):
        raise ValueError("sampling.transforms must be an object")

    bounds_cfg = sampling_cfg.get("bounds", {})
    if not isinstance(bounds_cfg, Mapping):
        raise ValueError("sampling.bounds must be an object")

    clip_to_bounds = bool(sampling_cfg.get("clip_to_bounds", False))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    for _ in range(n_parameter_sets):
        params: dict[str, float] = {}
        for name, population_spec in population.items():
            if name not in individual_sd:
                raise ValueError(f"sampling.individual_sd.{name} is required")
            sd_value = float(individual_sd[name])
            mean_value = float(_sample_from_distribution(population_spec, rng=rng))
            sample_value = float(rng.normal(mean_value, sd_value))
            params[str(name)] = _decode_sampled_parameter(
                name=str(name),
                sampled_value=sample_value,
                space=space,
                transform_name=transforms_cfg.get(name),
                bounds_value=bounds_cfg.get(name),
                clip_to_bounds=clip_to_bounds,
            )
        rows.append(params)
    return tuple(rows)


def _decode_sampled_parameter(
    *,
    name: str,
    sampled_value: float,
    space: str,
    transform_name: Any,
    bounds_value: Any,
    clip_to_bounds: bool,
) -> float:
    """Decode sampled parameter from configured space."""

    if space == "parameter":
        value = float(sampled_value)
    else:
        transform = _transform_from_name(transform_name, field_name=f"sampling.transforms.{name}")
        value = float(transform.forward(sampled_value))

    if bounds_value is None:
        return value

    bounds = _coerce_bounds(bounds_value, field_name=f"sampling.bounds.{name}")
    lower, upper = bounds
    if clip_to_bounds:
        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)
        return value

    if lower is not None and value < lower:
        raise ValueError(f"sampled value for {name!r} fell below lower bound")
    if upper is not None and value > upper:
        raise ValueError(f"sampled value for {name!r} exceeded upper bound")
    return value


def _sample_from_distribution(spec: DistributionSpec, *, rng: np.random.Generator) -> float:
    """Sample one scalar from a configured distribution spec."""

    if not isinstance(spec, Mapping):
        raise ValueError("distribution spec must be an object")
    distribution = str(spec.get("distribution", "")).strip()
    if distribution == "uniform":
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        return float(rng.uniform(lower, upper))
    if distribution == "normal":
        return float(rng.normal(float(spec["mean"]), float(spec["std"])))
    if distribution == "beta":
        return float(rng.beta(float(spec["alpha"]), float(spec["beta"])))
    if distribution == "log_normal":
        mean_log = spec.get("mean_log", spec.get("mean"))
        std_log = spec.get("std_log", spec.get("std"))
        if mean_log is None or std_log is None:
            raise ValueError("log_normal distribution requires mean_log/std_log")
        return float(rng.lognormal(float(mean_log), float(std_log)))
    raise ValueError(
        "distribution must be one of "
        "{'uniform', 'normal', 'beta', 'log_normal'}"
    )


def _transform_from_name(raw: Any, *, field_name: str) -> ParameterTransform:
    """Resolve configured transform name to transform object."""

    if raw is None:
        return identity_transform()
    value = str(raw).strip()
    if value == "identity":
        return identity_transform()
    if value == "positive_log":
        return positive_log_transform()
    if value == "unit_interval_logit":
        return unit_interval_logit_transform()
    raise ValueError(
        f"{field_name} must be one of "
        "{'identity', 'positive_log', 'unit_interval_logit'}"
    )


def _coerce_bounds(raw: Any, *, field_name: str) -> tuple[float | None, float | None]:
    """Parse bounds pair into optional float tuple."""

    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field_name} must be a [lower, upper] pair")
    lower_raw, upper_raw = raw
    lower = float(lower_raw) if lower_raw is not None else None
    upper = float(upper_raw) if upper_raw is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"{field_name} lower bound must be <= upper bound")
    return lower, upper


def _aggregate_parameter_errors(
    cases: Sequence[ParameterRecoveryCase],
) -> tuple[dict[str, float], dict[str, float], dict[str, float | None]]:
    """Aggregate per-parameter error summaries across cases."""

    shared_keys = set(cases[0].true_params) & set(cases[0].estimated_params) if cases else set()
    for case in cases[1:]:
        shared_keys &= set(case.true_params) & set(case.estimated_params)

    mean_absolute_error: dict[str, float] = {}
    mean_signed_error: dict[str, float] = {}
    true_estimate_correlation: dict[str, float | None] = {}

    for key in sorted(shared_keys):
        true_values = np.asarray([case.true_params[key] for case in cases], dtype=float)
        estimated_values = np.asarray([case.estimated_params[key] for case in cases], dtype=float)
        signed_errors = estimated_values - true_values

        mean_absolute_error[key] = float(np.mean(np.abs(signed_errors)))
        mean_signed_error[key] = float(np.mean(signed_errors))

        if len(cases) < 2 or np.allclose(np.std(true_values), 0.0) or np.allclose(np.std(estimated_values), 0.0):
            true_estimate_correlation[key] = None
        else:
            true_estimate_correlation[key] = float(np.corrcoef(true_values, estimated_values)[0, 1])

    return mean_absolute_error, mean_signed_error, true_estimate_correlation


__all__ = [
    "DistributionSpec",
    "ParameterRecoveryCase",
    "ParameterRecoveryResult",
    "SamplingSpec",
    "resolve_true_parameter_sets",
    "run_parameter_recovery",
    "sample_true_parameter_sets_from_distributions",
    "sample_true_parameter_sets_from_sampling",
]
