"""Parameter-recovery workflow utilities.

This module provides a lightweight recovery pipeline that uses the canonical
runtime loop and inference outputs in ``comp_model``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionProblem
from comp_model.inference.fit_result import extract_best_fit_summary
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
    true_estimate_correlation : dict[str, float | None]
        Pearson correlation between true and estimated values across cases for
        each shared parameter key. ``None`` when correlation is undefined
        (e.g., fewer than two cases or zero variance).
    """

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
    """Run simulation-based parameter recovery.

    Parameters
    ----------
    problem_factory : Callable[[], DecisionProblem[ObsT, ActionT, OutcomeT]] | None, optional
        Factory returning a fresh problem instance for one synthetic dataset.
        Used when ``trace_factory`` is not provided.
    model_factory : Callable[[dict[str, float]], AgentModel[ObsT, ActionT, OutcomeT]]
        Factory returning a generating model from true parameters.
    fit_function : Callable[[Any], Any]
        Function that fits one generated trace and returns a supported
        inference fit result (MLE-style or MAP-style).
    true_parameter_sets : Sequence[Mapping[str, float]] | None, optional
        Explicit collection of true parameter mappings to recover.
    true_parameter_distributions : Mapping[str, DistributionSpec] | None, optional
        Distribution specs used to sample true parameter mappings in code
        workflows. Supported distributions are ``uniform``, ``normal``,
        ``beta``, and ``log_normal``.
    true_parameter_sampling : SamplingSpec | None, optional
        Advanced sampling specification supporting both independent and
        hierarchical true-parameter generation in parameter or z space.
        This mirrors config-style ``sampling`` semantics for code workflows.
    n_parameter_sets : int | None, optional
        Number of true parameter mappings to sample when
        ``true_parameter_distributions`` or ``true_parameter_sampling`` is
        provided. For ``true_parameter_sampling``, this can also be declared
        inside the sampling mapping as ``n_parameter_sets``.
    n_trials : int
        Number of trials per synthetic dataset.
    seed : int, optional
        Master seed used to derive per-case simulation seeds.
    trace_factory : Callable[[AgentModel[ObsT, ActionT, OutcomeT], int], Any] | None, optional
        Optional custom trace simulator receiving ``(generating_model, seed)``
        and returning a fit-compatible dataset object. This enables recovery on
        trial-program and multi-actor traces. When omitted, traces are
        generated via :func:`comp_model.runtime.run_episode` using
        ``problem_factory`` and ``n_trials``.

    Returns
    -------
    ParameterRecoveryResult
        Recovery records and aggregate error summaries.

    Raises
    ------
    ValueError
        If parameter-set inputs are invalid, ``n_trials`` is non-positive, or
        both simulation sources are missing.
    """

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
            # Custom simulator path supports richer trial programs and
            # multi-actor data generation while keeping fit_function generic.
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
                best_log_posterior=(
                    float(best.log_posterior)
                    if best.log_posterior is not None
                    else None
                ),
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
    """Resolve true parameter mappings from one declared source.

    Parameters
    ----------
    true_parameter_sets : Sequence[Mapping[str, float]] | None, optional
        Explicit true parameter mappings.
    true_parameter_distributions : Mapping[str, DistributionSpec] | None, optional
        Distribution specs keyed by parameter name.
    true_parameter_sampling : SamplingSpec | None, optional
        Sampling configuration for independent/hierarchical true-parameter
        generation in code workflows.
    n_parameter_sets : int | None, optional
        Number of samples drawn from ``true_parameter_distributions`` or
        ``true_parameter_sampling``.
    seed : int, optional
        Random seed used for distribution sampling.

    Returns
    -------
    tuple[dict[str, float], ...]
        Resolved true parameter mappings.

    Raises
    ------
    ValueError
        If source declarations are ambiguous/missing or sampling input is
        invalid.
    """

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
        if len(true_parameter_sets) == 0:
            raise ValueError("true_parameter_sets must not be empty")
        parsed: list[dict[str, float]] = []
        for index, params in enumerate(true_parameter_sets):
            if not isinstance(params, Mapping):
                raise ValueError(
                    f"true_parameter_sets[{index}] must be a mapping"
                )
            parsed.append(
                {str(name): float(value) for name, value in params.items()}
            )
        return tuple(parsed)

    if has_distributions:
        assert true_parameter_distributions is not None
        if not true_parameter_distributions:
            raise ValueError("true_parameter_distributions must not be empty")
        if n_parameter_sets is None:
            raise ValueError(
                "n_parameter_sets is required when true_parameter_distributions "
                "is used"
            )
        if int(n_parameter_sets) <= 0:
            raise ValueError("n_parameter_sets must be > 0")
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
    *,
    true_parameter_distributions: Mapping[str, DistributionSpec],
    n_parameter_sets: int,
    seed: int = 0,
) -> tuple[dict[str, float], ...]:
    """Sample true parameter mappings from declared distributions.

    Parameters
    ----------
    true_parameter_distributions : Mapping[str, DistributionSpec]
        Distribution spec per parameter.
    n_parameter_sets : int
        Number of parameter mappings to draw.
    seed : int, optional
        Random seed for sampling.

    Returns
    -------
    tuple[dict[str, float], ...]
        Sampled true parameter mappings.

    Raises
    ------
    ValueError
        If distribution specs are invalid.
    """

    if n_parameter_sets <= 0:
        raise ValueError("n_parameter_sets must be > 0")
    if not true_parameter_distributions:
        raise ValueError("true_parameter_distributions must not be empty")

    rng = np.random.default_rng(seed)
    parameter_names = tuple(sorted(str(name) for name in true_parameter_distributions))
    draws_by_parameter: dict[str, np.ndarray] = {}
    for parameter_name in parameter_names:
        spec = true_parameter_distributions.get(parameter_name)
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"true_parameter_distributions.{parameter_name} must be a mapping"
            )
        draws_by_parameter[parameter_name] = _sample_parameter_distribution(
            spec=dict(spec),
            n_draws=n_parameter_sets,
            rng=rng,
            field_name=f"true_parameter_distributions.{parameter_name}",
        )

    out: list[dict[str, float]] = []
    for index in range(n_parameter_sets):
        out.append(
            {
                parameter_name: float(draws_by_parameter[parameter_name][index])
                for parameter_name in parameter_names
            }
        )
    return tuple(out)


def sample_true_parameter_sets_from_sampling(
    *,
    true_parameter_sampling: SamplingSpec,
    n_parameter_sets: int | None = None,
    seed: int = 0,
) -> tuple[dict[str, float], ...]:
    """Sample true parameter mappings from advanced sampling specification.

    Parameters
    ----------
    true_parameter_sampling : SamplingSpec
        Sampling mapping with keys:
        ``mode`` (``independent`` or ``hierarchical``),
        ``space`` (``param`` or ``z``),
        ``n_parameter_sets`` (optional),
        and mode-specific fields.
    n_parameter_sets : int | None, optional
        Number of parameter mappings to draw. When ``None``, the function
        requires ``true_parameter_sampling['n_parameter_sets']``.
    seed : int, optional
        Random seed for sampling.

    Returns
    -------
    tuple[dict[str, float], ...]
        Sampled true parameter mappings.

    Raises
    ------
    ValueError
        If sampling specification is malformed.
    """

    if not isinstance(true_parameter_sampling, Mapping):
        raise ValueError("true_parameter_sampling must be a mapping")

    sampling = {str(key): value for key, value in true_parameter_sampling.items()}
    _validate_allowed_keys(
        sampling,
        field_name="true_parameter_sampling",
        allowed_keys=(
            "mode",
            "space",
            "n_parameter_sets",
            "distributions",
            "population",
            "individual_sd",
            "transforms",
            "bounds",
            "clip_to_bounds",
        ),
    )

    mode = _coerce_non_empty_str(
        sampling.get("mode", "independent"),
        field_name="true_parameter_sampling.mode",
    )
    if mode == "fixed":
        raise ValueError("true_parameter_sampling.mode='fixed' is not supported")
    if mode not in {"independent", "hierarchical"}:
        raise ValueError(
            "true_parameter_sampling.mode must be one of {'independent', 'hierarchical'}"
        )

    space = _coerce_non_empty_str(
        sampling.get("space", "param"),
        field_name="true_parameter_sampling.space",
    )
    if space not in {"param", "z"}:
        raise ValueError("true_parameter_sampling.space must be one of {'param', 'z'}")

    n_sets_raw = sampling.get("n_parameter_sets", n_parameter_sets)
    if n_sets_raw is None:
        raise ValueError(
            "n_parameter_sets is required for true_parameter_sampling "
            "(either argument or true_parameter_sampling.n_parameter_sets)"
        )
    n_sets = int(n_sets_raw)
    if n_sets <= 0:
        raise ValueError("n_parameter_sets must be > 0")

    rng = np.random.default_rng(seed)
    clip_to_bounds = bool(sampling.get("clip_to_bounds", True))
    bounds = _coerce_bounds_mapping(
        sampling.get("bounds"),
        field_name="true_parameter_sampling.bounds",
    )

    if mode == "independent":
        distributions = _require_mapping(
            sampling.get("distributions"),
            field_name="true_parameter_sampling.distributions",
        )
        if not distributions:
            raise ValueError("true_parameter_sampling.distributions must not be empty")

        parameter_names = tuple(sorted(str(name) for name in distributions))
        transforms = _parse_sampling_transforms(
            sampling.get("transforms"),
            field_name="true_parameter_sampling.transforms",
            parameter_names=parameter_names,
        )

        draws_by_parameter: dict[str, np.ndarray] = {}
        for parameter_name in parameter_names:
            draws_by_parameter[parameter_name] = _sample_parameter_distribution(
                spec=dict(
                    _require_mapping(
                        distributions.get(parameter_name),
                        field_name=(
                            "true_parameter_sampling.distributions."
                            f"{parameter_name}"
                        ),
                    )
                ),
                n_draws=n_sets,
                rng=rng,
                field_name=f"true_parameter_sampling.distributions.{parameter_name}",
            )

        sampled_sets: list[dict[str, float]] = []
        for case_index in range(n_sets):
            sampled_case: dict[str, float] = {}
            for parameter_name in parameter_names:
                raw_value = float(draws_by_parameter[parameter_name][case_index])
                if space == "z":
                    sampled_case[parameter_name] = float(
                        transforms[parameter_name].forward(raw_value)
                    )
                else:
                    sampled_case[parameter_name] = raw_value
            if clip_to_bounds and bounds:
                sampled_case = _clip_params_to_bounds(sampled_case, bounds=bounds)
            sampled_sets.append(sampled_case)
        return tuple(sampled_sets)

    population = _require_mapping(
        sampling.get("population"),
        field_name="true_parameter_sampling.population",
    )
    individual_sd = _coerce_float_mapping(
        sampling.get("individual_sd"),
        field_name="true_parameter_sampling.individual_sd",
    )
    if not population:
        raise ValueError("true_parameter_sampling.population must not be empty")

    parameter_names = tuple(sorted(str(name) for name in population))
    missing_sd = [name for name in parameter_names if name not in individual_sd]
    if missing_sd:
        raise ValueError(
            f"true_parameter_sampling.individual_sd missing parameters: {missing_sd}"
        )

    transforms = _parse_sampling_transforms(
        sampling.get("transforms"),
        field_name="true_parameter_sampling.transforms",
        parameter_names=parameter_names,
    )

    population_draws: dict[str, float] = {}
    for parameter_name in parameter_names:
        population_draws[parameter_name] = float(
            _sample_parameter_distribution(
                spec=dict(
                    _require_mapping(
                        population.get(parameter_name),
                        field_name=f"true_parameter_sampling.population.{parameter_name}",
                    )
                ),
                n_draws=1,
                rng=rng,
                field_name=f"true_parameter_sampling.population.{parameter_name}",
            )[0]
        )

    hierarchical_sets: list[dict[str, float]] = []
    for _ in range(n_sets):
        hierarchical_case: dict[str, float] = {}
        for parameter_name in parameter_names:
            sd = float(individual_sd[parameter_name])
            if sd < 0.0:
                raise ValueError(
                    "true_parameter_sampling.individual_sd."
                    f"{parameter_name} must be >= 0"
                )
            sampled_value = float(
                rng.normal(loc=population_draws[parameter_name], scale=sd)
            )
            if space == "z":
                hierarchical_case[parameter_name] = float(
                    transforms[parameter_name].forward(sampled_value)
                )
            else:
                hierarchical_case[parameter_name] = sampled_value
        if clip_to_bounds and bounds:
            hierarchical_case = _clip_params_to_bounds(
                hierarchical_case,
                bounds=bounds,
            )
        hierarchical_sets.append(hierarchical_case)
    return tuple(hierarchical_sets)


def _sample_parameter_distribution(
    *,
    spec: dict[str, Any],
    n_draws: int,
    rng: np.random.Generator,
    field_name: str,
) -> np.ndarray:
    """Sample one parameter vector from one distribution specification."""

    if "distribution" not in spec:
        raise ValueError(f"{field_name}.distribution is required")
    distribution = str(spec["distribution"]).strip()
    if distribution == "":
        raise ValueError(f"{field_name}.distribution must be a non-empty string")

    if distribution == "uniform":
        _validate_allowed_keys(spec, field_name=field_name, allowed_keys=("distribution", "lower", "upper"))
        if "lower" not in spec or "upper" not in spec:
            raise ValueError(f"{field_name} requires 'lower' and 'upper'")
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        if lower > upper:
            raise ValueError(f"{field_name} has lower > upper")
        if lower == upper:
            return np.full(shape=n_draws, fill_value=lower, dtype=float)
        return rng.uniform(lower, upper, size=n_draws)

    if distribution == "normal":
        _validate_allowed_keys(spec, field_name=field_name, allowed_keys=("distribution", "mean", "std"))
        if "mean" not in spec or "std" not in spec:
            raise ValueError(f"{field_name} requires 'mean' and 'std'")
        mean = float(spec["mean"])
        std = float(spec["std"])
        if std <= 0.0:
            raise ValueError(f"{field_name}.std must be > 0")
        return rng.normal(loc=mean, scale=std, size=n_draws)

    if distribution == "beta":
        _validate_allowed_keys(spec, field_name=field_name, allowed_keys=("distribution", "alpha", "beta"))
        if "alpha" not in spec or "beta" not in spec:
            raise ValueError(f"{field_name} requires 'alpha' and 'beta'")
        alpha = float(spec["alpha"])
        beta = float(spec["beta"])
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError(f"{field_name}.alpha and {field_name}.beta must be > 0")
        return rng.beta(alpha, beta, size=n_draws)

    if distribution == "log_normal":
        _validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "mean_log", "std_log"),
        )
        if "mean_log" not in spec or "std_log" not in spec:
            raise ValueError(f"{field_name} requires 'mean_log' and 'std_log'")
        mean_log = float(spec["mean_log"])
        std_log = float(spec["std_log"])
        if std_log <= 0.0:
            raise ValueError(f"{field_name}.std_log must be > 0")
        return rng.lognormal(mean=mean_log, sigma=std_log, size=n_draws)

    raise ValueError(
        f"{field_name}.distribution must be one of "
        "{'uniform', 'normal', 'beta', 'log_normal'}"
    )


def _validate_allowed_keys(
    mapping: Mapping[str, Any],
    *,
    field_name: str,
    allowed_keys: tuple[str, ...],
) -> None:
    """Validate that a mapping includes only declared keys."""

    allowed = set(allowed_keys)
    unknown = [key for key in mapping if str(key) not in allowed]
    if unknown:
        raise ValueError(
            f"{field_name} has unknown keys: {sorted(str(key) for key in unknown)}; "
            f"allowed keys are {sorted(allowed)}"
        )


def _coerce_non_empty_str(raw: Any, *, field_name: str) -> str:
    """Parse one required non-empty string value."""

    if raw is None:
        raise ValueError(f"{field_name} is required")
    value = str(raw).strip()
    if value == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_mapping(raw: Any, *, field_name: str) -> Mapping[str, Any]:
    """Require mapping and normalize keys to strings."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _coerce_float_mapping(raw: Any, *, field_name: str) -> dict[str, float]:
    """Parse mapping whose values are floats."""

    mapping = _require_mapping(raw, field_name=field_name)
    return {str(key): float(value) for key, value in mapping.items()}


def _parse_sampling_transforms(
    raw: Any,
    *,
    field_name: str,
    parameter_names: tuple[str, ...],
) -> dict[str, ParameterTransform]:
    """Parse optional transform mapping for z-space sampling."""

    mapping = _require_mapping(raw if raw is not None else {}, field_name=field_name)
    unknown = sorted(set(str(name) for name in mapping).difference(parameter_names))
    if unknown:
        raise ValueError(
            f"{field_name} contains unknown parameters {unknown}; "
            f"expected subset of {list(parameter_names)!r}"
        )

    transforms = {name: identity_transform() for name in parameter_names}
    for param_name, spec in mapping.items():
        if isinstance(spec, str):
            transforms[str(param_name)] = _sampling_transform_from_name(spec)
            continue
        spec_mapping = _require_mapping(
            spec,
            field_name=f"{field_name}.{param_name}",
        )
        kind = _coerce_non_empty_str(
            spec_mapping.get("kind"),
            field_name=f"{field_name}.{param_name}.kind",
        )
        transforms[str(param_name)] = _sampling_transform_from_name(kind)
    return transforms


def _sampling_transform_from_name(name: str) -> ParameterTransform:
    """Resolve one named transform for z-space sampling."""

    normalized = str(name)
    if normalized == "identity":
        return identity_transform()
    if normalized == "unit_interval_logit":
        return unit_interval_logit_transform()
    if normalized == "positive_log":
        return positive_log_transform()
    raise ValueError(
        f"unsupported transform {name!r}; expected one of "
        "{'identity', 'unit_interval_logit', 'positive_log'}"
    )


def _coerce_bounds_mapping(
    raw: Any,
    *,
    field_name: str,
) -> dict[str, tuple[float | None, float | None]] | None:
    """Parse optional bounds mapping from sampling specifications."""

    if raw is None:
        return None

    mapping = _require_mapping(raw, field_name=field_name)
    out: dict[str, tuple[float | None, float | None]] = {}
    for key, value in mapping.items():
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{field_name}.{key} must be a sequence of length 2")
        pair = tuple(value)
        if len(pair) != 2:
            raise ValueError(f"{field_name}.{key} must have exactly two elements")
        lower_raw, upper_raw = pair
        lower = float(lower_raw) if lower_raw is not None else None
        upper = float(upper_raw) if upper_raw is not None else None
        out[str(key)] = (lower, upper)
    return out


def _clip_params_to_bounds(
    params: Mapping[str, float],
    *,
    bounds: Mapping[str, tuple[float | None, float | None]],
) -> dict[str, float]:
    """Clip parameter values to declared bounds."""

    clipped = {str(name): float(value) for name, value in params.items()}
    for parameter_name, value in list(clipped.items()):
        if parameter_name not in bounds:
            continue
        lower, upper = bounds[parameter_name]
        if lower is not None and value < lower:
            value = float(lower)
        if upper is not None and value > upper:
            value = float(upper)
        clipped[parameter_name] = float(value)
    return clipped


def _aggregate_parameter_errors(
    cases: Sequence[ParameterRecoveryCase],
) -> tuple[dict[str, float], dict[str, float], dict[str, float | None]]:
    """Aggregate recovery errors and correlations by parameter key."""

    common_keys: set[str] | None = None
    for case in cases:
        keys = set(case.true_params) & set(case.estimated_params)
        common_keys = keys if common_keys is None else (common_keys & keys)

    if not common_keys:
        return {}, {}, {}

    mean_abs_error: dict[str, float] = {}
    mean_signed_error: dict[str, float] = {}
    true_estimate_correlation: dict[str, float | None] = {}
    for key in sorted(common_keys):
        true_values = np.asarray([case.true_params[key] for case in cases], dtype=float)
        estimated_values = np.asarray(
            [case.estimated_params[key] for case in cases],
            dtype=float,
        )
        signed = np.asarray(
            estimated_values - true_values,
            dtype=float,
        )
        mean_abs_error[key] = float(np.mean(np.abs(signed)))
        mean_signed_error[key] = float(np.mean(signed))
        true_estimate_correlation[key] = _pearson_correlation_or_none(
            true_values=true_values,
            estimated_values=estimated_values,
        )

    return mean_abs_error, mean_signed_error, true_estimate_correlation


def _pearson_correlation_or_none(
    *,
    true_values: np.ndarray,
    estimated_values: np.ndarray,
) -> float | None:
    """Return Pearson correlation or ``None`` when undefined."""

    if true_values.size < 2 or estimated_values.size < 2:
        return None
    if float(np.std(true_values)) == 0.0:
        return None
    if float(np.std(estimated_values)) == 0.0:
        return None

    correlation = float(np.corrcoef(true_values, estimated_values)[0, 1])
    if not np.isfinite(correlation):
        return None
    return correlation


__all__ = [
    "DistributionSpec",
    "SamplingSpec",
    "ParameterRecoveryCase",
    "ParameterRecoveryResult",
    "resolve_true_parameter_sets",
    "run_parameter_recovery",
    "sample_true_parameter_sets_from_sampling",
    "sample_true_parameter_sets_from_distributions",
]
