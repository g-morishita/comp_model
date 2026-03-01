"""Within-subject hierarchical Bayesian fitting (MAP approximation).

This module provides a generic hierarchical MAP estimator that pools parameter
information across blocks within a subject. It uses canonical replay
likelihoods and unconstrained optimization in transformed parameter space.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import log, pi
from typing import Any

import numpy as np

from comp_model.core.contracts import AgentModel
from comp_model.core.data import StudyData, SubjectData, get_block_trace
from comp_model.core.events import EpisodeTrace
from comp_model.core.requirements import ComponentRequirements

from .compatibility import assert_trace_compatible, check_trace_compatibility
from .likelihood import ActionReplayLikelihood, LikelihoodProgram
from .mle import ScipyMinimizeDiagnostics
from .transforms import ParameterTransform, identity_transform


@dataclass(frozen=True, slots=True)
class HierarchicalBlockResult:
    """MAP summary for one subject block.

    Parameters
    ----------
    block_id : str | int | None
        Original block identifier.
    params : dict[str, float]
        Block-level MAP parameters in constrained space.
    log_likelihood : float
        Block log-likelihood evaluated at MAP parameters.
    """

    block_id: str | int | None
    params: dict[str, float]
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class HierarchicalSubjectMapResult:
    """Within-subject hierarchical MAP output.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    parameter_names : tuple[str, ...]
        Hierarchical parameter names.
    group_location_z : dict[str, float]
        Group-level Normal location parameters in unconstrained ``z`` space.
    group_scale_z : dict[str, float]
        Group-level Normal scale parameters in unconstrained ``z`` space.
    block_results : tuple[HierarchicalBlockResult, ...]
        Per-block MAP parameter results.
    total_log_likelihood : float
        Sum of block log-likelihood values.
    total_log_prior : float
        Total hierarchical prior value.
    total_log_posterior : float
        ``total_log_likelihood + total_log_prior``.
    scipy_diagnostics : ScipyMinimizeDiagnostics
        SciPy optimizer diagnostics.
    """

    subject_id: str
    parameter_names: tuple[str, ...]
    group_location_z: dict[str, float]
    group_scale_z: dict[str, float]
    block_results: tuple[HierarchicalBlockResult, ...]
    total_log_likelihood: float
    total_log_prior: float
    total_log_posterior: float
    scipy_diagnostics: ScipyMinimizeDiagnostics


@dataclass(frozen=True, slots=True)
class HierarchicalStudyMapResult:
    """Hierarchical MAP output aggregated across study subjects.

    Parameters
    ----------
    subject_results : tuple[HierarchicalSubjectMapResult, ...]
        Per-subject hierarchical MAP results.
    total_log_likelihood : float
        Sum of subject-level log-likelihood values.
    total_log_prior : float
        Sum of subject-level log-prior values.
    total_log_posterior : float
        Sum of subject-level log-posterior values.
    """

    subject_results: tuple[HierarchicalSubjectMapResult, ...]
    total_log_likelihood: float
    total_log_prior: float
    total_log_posterior: float

    @property
    def n_subjects(self) -> int:
        """Return number of fitted subjects."""

        return len(self.subject_results)


def fit_subject_hierarchical_map(
    subject: SubjectData,
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    parameter_names: Sequence[str],
    transforms: Mapping[str, ParameterTransform] | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    requirements: ComponentRequirements | None = None,
    initial_group_location: Mapping[str, float] | None = None,
    initial_group_scale: Mapping[str, float] | None = None,
    initial_block_params: Sequence[Mapping[str, float]] | None = None,
    mu_prior_mean: float = 0.0,
    mu_prior_std: float = 2.0,
    log_sigma_prior_mean: float = -1.0,
    log_sigma_prior_std: float = 1.0,
    method: str = "L-BFGS-B",
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> HierarchicalSubjectMapResult:
    """Fit a within-subject hierarchical MAP model over blocks.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset containing one or more blocks.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory that builds one model from constrained parameter values.
    parameter_names : Sequence[str]
        Parameter names to hierarchically pool.
    transforms : Mapping[str, ParameterTransform] | None, optional
        Per-parameter transform mapping from unconstrained ``z`` to constrained
        parameter value. Missing names use identity transform.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood evaluator. Defaults to :class:`ActionReplayLikelihood`.
    requirements : ComponentRequirements | None, optional
        Optional compatibility checks applied to every block trace.
    initial_group_location : Mapping[str, float] | None, optional
        Initial constrained group-location values by parameter.
    initial_group_scale : Mapping[str, float] | None, optional
        Initial positive group-scale values by parameter.
    initial_block_params : Sequence[Mapping[str, float]] | None, optional
        Optional constrained initial values per block. Length must match block
        count when provided.
    mu_prior_mean : float, optional
        Normal prior mean for group location in ``z`` space.
    mu_prior_std : float, optional
        Positive Normal prior std for group location in ``z`` space.
    log_sigma_prior_mean : float, optional
        Normal prior mean for ``log(group_scale)``.
    log_sigma_prior_std : float, optional
        Positive Normal prior std for ``log(group_scale)``.
    method : str, optional
        SciPy minimizer method.
    tol : float | None, optional
        SciPy tolerance.
    options : Mapping[str, Any] | None, optional
        Extra SciPy optimizer options.

    Returns
    -------
    HierarchicalSubjectMapResult
        Subject-level hierarchical MAP fit output.

    Raises
    ------
    ValueError
        If inputs are invalid or optimization cannot proceed.
    """

    names = tuple(str(name) for name in parameter_names)
    if len(names) == 0:
        raise ValueError("parameter_names must include at least one parameter")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique")
    if mu_prior_std <= 0.0:
        raise ValueError("mu_prior_std must be > 0")
    if log_sigma_prior_std <= 0.0:
        raise ValueError("log_sigma_prior_std must be > 0")

    like = likelihood_program if likelihood_program is not None else ActionReplayLikelihood()
    traces = tuple(get_block_trace(block) for block in subject.blocks)

    if requirements is not None:
        for trace in traces:
            check_trace_compatibility(trace, requirements)
            assert_trace_compatible(trace, requirements)

    per_param_transform = {
        name: transforms[name] if transforms is not None and name in transforms else identity_transform()
        for name in names
    }

    n_blocks = len(subject.blocks)
    group_loc_init = {
        name: float(initial_group_location[name]) if initial_group_location is not None and name in initial_group_location else 0.0
        for name in names
    }
    group_loc_init_z = {
        name: float(per_param_transform[name].inverse(group_loc_init[name]))
        for name in names
    }

    group_scale_init = {}
    for name in names:
        raw = 1.0
        if initial_group_scale is not None and name in initial_group_scale:
            raw = float(initial_group_scale[name])
        if raw <= 0.0:
            raise ValueError(f"initial_group_scale[{name!r}] must be > 0")
        group_scale_init[name] = raw
    group_log_scale_init = {name: float(np.log(group_scale_init[name])) for name in names}

    if initial_block_params is not None and len(initial_block_params) != n_blocks:
        raise ValueError("initial_block_params must match number of subject blocks")

    x0_parts: list[float] = []
    x0_parts.extend(group_loc_init_z[name] for name in names)
    x0_parts.extend(group_log_scale_init[name] for name in names)
    for block_index in range(n_blocks):
        raw_block = initial_block_params[block_index] if initial_block_params is not None else {}
        for name in names:
            if name in raw_block:
                value = float(raw_block[name])
            else:
                value = float(group_loc_init[name])
            x0_parts.append(float(per_param_transform[name].inverse(value)))
    x0 = np.asarray(x0_parts, dtype=float)

    minimize = _load_scipy_minimize()
    scipy_options = dict(options) if options is not None else None

    def objective(vector: np.ndarray) -> float:
        decoded = _decode_parameter_vector(
            vector=vector,
            names=names,
            n_blocks=n_blocks,
            transforms=per_param_transform,
        )
        total_log_likelihood = _total_log_likelihood(
            traces=traces,
            model_factory=model_factory,
            likelihood_program=like,
            block_params=decoded.block_params,
        )
        total_log_prior = _hierarchical_log_prior(
            group_location=decoded.group_location_z,
            group_log_scale=decoded.group_log_scale_z,
            block_z=decoded.block_params_z,
            mu_prior_mean=float(mu_prior_mean),
            mu_prior_std=float(mu_prior_std),
            log_sigma_prior_mean=float(log_sigma_prior_mean),
            log_sigma_prior_std=float(log_sigma_prior_std),
        )

        total_log_posterior = total_log_likelihood + total_log_prior
        if not np.isfinite(total_log_posterior):
            return 1e15
        return float(-total_log_posterior)

    result = minimize(
        objective,
        x0,
        method=str(method),
        tol=tol,
        options=scipy_options,
    )

    final_vector = np.asarray(result.x, dtype=float)
    decoded = _decode_parameter_vector(
        vector=final_vector,
        names=names,
        n_blocks=n_blocks,
        transforms=per_param_transform,
    )
    total_log_likelihood = _total_log_likelihood(
        traces=traces,
        model_factory=model_factory,
        likelihood_program=like,
        block_params=decoded.block_params,
    )
    total_log_prior = _hierarchical_log_prior(
        group_location=decoded.group_location_z,
        group_log_scale=decoded.group_log_scale_z,
        block_z=decoded.block_params_z,
        mu_prior_mean=float(mu_prior_mean),
        mu_prior_std=float(mu_prior_std),
        log_sigma_prior_mean=float(log_sigma_prior_mean),
        log_sigma_prior_std=float(log_sigma_prior_std),
    )
    total_log_posterior = float(total_log_likelihood + total_log_prior)

    block_results: list[HierarchicalBlockResult] = []
    for block, params in zip(subject.blocks, decoded.block_params, strict=True):
        log_likelihood = _block_log_likelihood(
            trace=get_block_trace(block),
            model_factory=model_factory,
            likelihood_program=like,
            params=params,
        )
        block_results.append(
            HierarchicalBlockResult(
                block_id=block.block_id,
                params=params,
                log_likelihood=log_likelihood,
            )
        )

    diagnostics = ScipyMinimizeDiagnostics(
        method=str(method),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        n_iterations=int(getattr(result, "nit", -1)),
        n_function_evaluations=int(getattr(result, "nfev", -1)),
    )

    return HierarchicalSubjectMapResult(
        subject_id=subject.subject_id,
        parameter_names=names,
        group_location_z=dict(decoded.group_location_z),
        group_scale_z={
            name: float(np.exp(decoded.group_log_scale_z[name]))
            for name in names
        },
        block_results=tuple(block_results),
        total_log_likelihood=float(total_log_likelihood),
        total_log_prior=float(total_log_prior),
        total_log_posterior=total_log_posterior,
        scipy_diagnostics=diagnostics,
    )


def fit_study_hierarchical_map(
    study: StudyData,
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    parameter_names: Sequence[str],
    transforms: Mapping[str, ParameterTransform] | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    requirements: ComponentRequirements | None = None,
    initial_group_location: Mapping[str, float] | None = None,
    initial_group_scale: Mapping[str, float] | None = None,
    initial_block_params_by_subject: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
    mu_prior_mean: float = 0.0,
    mu_prior_std: float = 2.0,
    log_sigma_prior_mean: float = -1.0,
    log_sigma_prior_std: float = 1.0,
    method: str = "L-BFGS-B",
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> HierarchicalStudyMapResult:
    """Fit within-subject hierarchical MAP independently for all subjects.

    Parameters
    ----------
    study : StudyData
        Study dataset.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Model-construction callable.
    parameter_names : Sequence[str]
        Names of pooled parameters.
    transforms : Mapping[str, ParameterTransform] | None, optional
        Per-parameter transforms.
    likelihood_program : LikelihoodProgram | None, optional
        Replay likelihood evaluator.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements.
    initial_group_location : Mapping[str, float] | None, optional
        Initial constrained group-location values.
    initial_group_scale : Mapping[str, float] | None, optional
        Initial positive group-scale values.
    initial_block_params_by_subject : Mapping[str, Sequence[Mapping[str, float]]] | None, optional
        Optional subject-specific block initial parameters.
    mu_prior_mean : float, optional
        Group-location Normal prior mean in ``z`` space.
    mu_prior_std : float, optional
        Group-location Normal prior std in ``z`` space.
    log_sigma_prior_mean : float, optional
        Prior mean for log group scale.
    log_sigma_prior_std : float, optional
        Prior std for log group scale.
    method : str, optional
        SciPy optimizer method.
    tol : float | None, optional
        SciPy optimizer tolerance.
    options : Mapping[str, Any] | None, optional
        Extra SciPy optimizer options.

    Returns
    -------
    HierarchicalStudyMapResult
        Aggregated study-level hierarchical MAP output.
    """

    subject_results: list[HierarchicalSubjectMapResult] = []
    for subject in study.subjects:
        subject_initial_block_params = None
        if initial_block_params_by_subject is not None:
            raw_subject_params = initial_block_params_by_subject.get(subject.subject_id)
            if raw_subject_params is not None:
                subject_initial_block_params = tuple(dict(item) for item in raw_subject_params)

        subject_results.append(
            fit_subject_hierarchical_map(
                subject,
                model_factory=model_factory,
                parameter_names=parameter_names,
                transforms=transforms,
                likelihood_program=likelihood_program,
                requirements=requirements,
                initial_group_location=initial_group_location,
                initial_group_scale=initial_group_scale,
                initial_block_params=subject_initial_block_params,
                mu_prior_mean=mu_prior_mean,
                mu_prior_std=mu_prior_std,
                log_sigma_prior_mean=log_sigma_prior_mean,
                log_sigma_prior_std=log_sigma_prior_std,
                method=method,
                tol=tol,
                options=options,
            )
        )

    total_log_likelihood = float(
        sum(item.total_log_likelihood for item in subject_results)
    )
    total_log_prior = float(
        sum(item.total_log_prior for item in subject_results)
    )
    total_log_posterior = float(
        sum(item.total_log_posterior for item in subject_results)
    )
    return HierarchicalStudyMapResult(
        subject_results=tuple(subject_results),
        total_log_likelihood=total_log_likelihood,
        total_log_prior=total_log_prior,
        total_log_posterior=total_log_posterior,
    )


@dataclass(frozen=True, slots=True)
class _DecodedHierarchicalVector:
    """Decoded parameter vector for hierarchical MAP objective."""

    group_location_z: dict[str, float]
    group_log_scale_z: dict[str, float]
    block_params_z: tuple[dict[str, float], ...]
    block_params: tuple[dict[str, float], ...]


def _decode_parameter_vector(
    *,
    vector: np.ndarray,
    names: tuple[str, ...],
    n_blocks: int,
    transforms: Mapping[str, ParameterTransform],
) -> _DecodedHierarchicalVector:
    """Decode flat optimizer vector into hierarchical parameter components."""

    n_params = len(names)
    expected_len = (2 * n_params) + (n_blocks * n_params)
    if len(vector) != expected_len:
        raise ValueError(f"hierarchical vector has length {len(vector)}; expected {expected_len}")

    cursor = 0
    group_location_z: dict[str, float] = {}
    for name in names:
        group_location_z[name] = float(vector[cursor])
        cursor += 1

    group_log_scale_z: dict[str, float] = {}
    for name in names:
        group_log_scale_z[name] = float(vector[cursor])
        cursor += 1

    block_params_z: list[dict[str, float]] = []
    block_params: list[dict[str, float]] = []
    for _ in range(n_blocks):
        block_z: dict[str, float] = {}
        block_theta: dict[str, float] = {}
        for name in names:
            z_value = float(vector[cursor])
            cursor += 1
            block_z[name] = z_value
            block_theta[name] = float(transforms[name].forward(z_value))
        block_params_z.append(block_z)
        block_params.append(block_theta)

    return _DecodedHierarchicalVector(
        group_location_z=group_location_z,
        group_log_scale_z=group_log_scale_z,
        block_params_z=tuple(block_params_z),
        block_params=tuple(block_params),
    )


def _total_log_likelihood(
    *,
    traces: Sequence[EpisodeTrace],
    model_factory: Callable[[dict[str, float]], AgentModel],
    likelihood_program: LikelihoodProgram,
    block_params: Sequence[Mapping[str, float]],
) -> float:
    """Compute total log-likelihood across subject blocks."""

    total = 0.0
    for trace, params in zip(traces, block_params, strict=True):
        model = model_factory(dict(params))
        replay = likelihood_program.evaluate(trace, model)
        total += float(replay.total_log_likelihood)
    return float(total)


def _block_log_likelihood(
    *,
    trace: EpisodeTrace,
    model_factory: Callable[[dict[str, float]], AgentModel],
    likelihood_program: LikelihoodProgram,
    params: Mapping[str, float],
) -> float:
    """Compute one block log-likelihood for report output."""

    model = model_factory(dict(params))
    replay = likelihood_program.evaluate(trace, model)
    return float(replay.total_log_likelihood)


def _hierarchical_log_prior(
    *,
    group_location: Mapping[str, float],
    group_log_scale: Mapping[str, float],
    block_z: Sequence[Mapping[str, float]],
    mu_prior_mean: float,
    mu_prior_std: float,
    log_sigma_prior_mean: float,
    log_sigma_prior_std: float,
) -> float:
    """Compute hierarchical Normal prior terms in ``z`` space."""

    total = 0.0
    for name, mu in group_location.items():
        total += _normal_logpdf(mu, mean=mu_prior_mean, std=mu_prior_std)

        log_sigma = float(group_log_scale[name])
        total += _normal_logpdf(
            log_sigma,
            mean=log_sigma_prior_mean,
            std=log_sigma_prior_std,
        )

        sigma = float(np.exp(log_sigma))
        for block in block_z:
            total += _normal_logpdf(float(block[name]), mean=float(mu), std=sigma)

    return float(total)


def _normal_logpdf(value: float, *, mean: float, std: float) -> float:
    """Evaluate Normal log-density at one scalar value."""

    sigma = float(std)
    if sigma <= 0.0:
        return float(-np.inf)
    z = (float(value) - float(mean)) / sigma
    return float(-0.5 * log(2.0 * pi * sigma * sigma) - 0.5 * z * z)


def _load_scipy_minimize():
    """Import and return ``scipy.optimize.minimize``."""

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - exercised only without scipy installed
        raise ImportError(
            "Hierarchical MAP fitting requires scipy. Install with `pip install scipy`."
        ) from exc
    return minimize


__all__ = [
    "HierarchicalBlockResult",
    "HierarchicalStudyMapResult",
    "HierarchicalSubjectMapResult",
    "fit_study_hierarchical_map",
    "fit_subject_hierarchical_map",
]
