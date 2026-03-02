"""Stan NUTS backend for within-subject hierarchical Bayesian estimation.

The Stan backend supports:

- asocial family:
  - ``asocial_q_value_softmax``
  - ``asocial_state_q_value_softmax``
  - ``asocial_state_q_value_softmax_perseveration``
  - ``asocial_state_q_value_softmax_split_alpha``
- all social model families registered in
  :mod:`comp_model.inference.hierarchical_stan_social`

Design notes
------------
- This backend mirrors the same within-subject hierarchy used by the existing
  Python hierarchical samplers: group-level location/scale in unconstrained
  space and block-level latent ``z`` parameters.
- Parameters are transformed from latent ``z`` into model space in Stan using
  configurable transform kinds.
- Outputs are converted into the same public result dataclasses used by the
  random-walk hierarchical sampler for API consistency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from comp_model.core.data import (
    BlockData,
    StudyData,
    SubjectData,
    TrialDecision,
    get_block_trace,
    trial_decisions_from_trace,
)
from comp_model.core.requirements import ComponentRequirements

from .compatibility import CompatibilityReport, assert_trace_compatible, check_trace_compatibility
from .hierarchical_mcmc import (
    HierarchicalMCMCDraw,
    HierarchicalPosteriorCandidate,
    HierarchicalStudyPosteriorResult,
    HierarchicalSubjectPosteriorResult,
)
from .hierarchical_stan_social import (
    build_social_subject_inputs,
    load_social_stan_code,
    social_cache_tag,
    social_supported_component_ids,
)
from .mcmc import MCMCDiagnostics
from .stan_backend import compile_cmdstan_model

_ASOCIAL_COMPONENT_ID = "asocial_state_q_value_softmax"
_ASOCIAL_PERSEVERATION_COMPONENT_ID = "asocial_state_q_value_softmax_perseveration"
_ASOCIAL_SPLIT_ALPHA_COMPONENT_ID = "asocial_state_q_value_softmax_split_alpha"
_ASOCIAL_Q_COMPONENT_ID = "asocial_q_value_softmax"
_ASOCIAL_COMPONENT_IDS = frozenset(
    {
        _ASOCIAL_COMPONENT_ID,
        _ASOCIAL_PERSEVERATION_COMPONENT_ID,
        _ASOCIAL_SPLIT_ALPHA_COMPONENT_ID,
        _ASOCIAL_Q_COMPONENT_ID,
    }
)
_SUPPORTED_SOCIAL_COMPONENT_IDS = frozenset(social_supported_component_ids())
_SUPPORTED_COMPONENT_IDS = frozenset({*_ASOCIAL_COMPONENT_IDS, *_SUPPORTED_SOCIAL_COMPONENT_IDS})

_ASOCIAL_EXTERNAL_PARAM_CODE_BY_NAME: dict[str, int] = {
    "alpha": 1,
    "alpha_1": 1,
    "alpha_2": 2,
    "beta": 3,
    "kappa": 4,
    "initial_value": 5,
}

_TRANSFORM_CODE_BY_KIND: dict[str, int] = {
    "identity": 0,
    "unit_interval_logit": 1,
    "positive_log": 2,
}

_STAN_WITHIN_SUBJECT_DIR = Path(__file__).with_name("stan") / "within_subject"


def _load_stan_source(filename: str) -> str:
    """Read one Stan program file from package-local stan directory."""

    path = _STAN_WITHIN_SUBJECT_DIR / filename
    if not path.exists():
        raise RuntimeError(f"Stan program file is missing: {path}")
    return path.read_text(encoding="utf-8")


_ASOCIAL_STAN_CODE = _load_stan_source("asocial_state_q_value_softmax.stan")


@dataclass(frozen=True, slots=True)
class _SubjectBuild:
    """Assembled Stan inputs for one subject."""

    stan_data: dict[str, Any]
    parameter_names: tuple[str, ...]
    n_blocks: int


@dataclass(frozen=True, slots=True)
class _AsocialModelSpec:
    """Configuration for one asocial model family in the Stan backend."""

    component_id: str
    default_params: dict[str, float]
    allowed_parameter_names: frozenset[str]
    allowed_model_kwargs: frozenset[str]
    stateful: bool


def _build_asocial_specs() -> dict[str, _AsocialModelSpec]:
    """Create asocial component mapping used by the Stan backend."""

    return {
        _ASOCIAL_COMPONENT_ID: _AsocialModelSpec(
            component_id=_ASOCIAL_COMPONENT_ID,
            default_params={
                "alpha": 0.2,
                "beta": 5.0,
                "initial_value": 0.0,
            },
            allowed_parameter_names=frozenset({"alpha", "beta", "initial_value"}),
            allowed_model_kwargs=frozenset({"alpha", "beta", "initial_value"}),
            stateful=True,
        ),
        _ASOCIAL_PERSEVERATION_COMPONENT_ID: _AsocialModelSpec(
            component_id=_ASOCIAL_PERSEVERATION_COMPONENT_ID,
            default_params={
                "alpha": 0.2,
                "beta": 5.0,
                "kappa": 1.0,
                "initial_value": 0.0,
            },
            allowed_parameter_names=frozenset({"alpha", "beta", "kappa", "initial_value"}),
            allowed_model_kwargs=frozenset({"alpha", "beta", "kappa", "initial_value"}),
            stateful=True,
        ),
        _ASOCIAL_SPLIT_ALPHA_COMPONENT_ID: _AsocialModelSpec(
            component_id=_ASOCIAL_SPLIT_ALPHA_COMPONENT_ID,
            default_params={
                "alpha_1": 0.2,
                "alpha_2": 0.2,
                "beta": 5.0,
                "initial_value": 0.0,
            },
            allowed_parameter_names=frozenset({"alpha_1", "alpha_2", "beta", "initial_value"}),
            allowed_model_kwargs=frozenset({"alpha_1", "alpha_2", "beta", "initial_value"}),
            stateful=True,
        ),
        _ASOCIAL_Q_COMPONENT_ID: _AsocialModelSpec(
            component_id=_ASOCIAL_Q_COMPONENT_ID,
            default_params={
                "alpha": 0.2,
                "beta": 3.0,
                "initial_value": 0.0,
            },
            allowed_parameter_names=frozenset({"alpha", "beta", "initial_value"}),
            allowed_model_kwargs=frozenset({"alpha", "beta", "initial_value"}),
            stateful=False,
        ),
    }


_ASOCIAL_SPECS = _build_asocial_specs()


def sample_subject_hierarchical_posterior_stan(
    subject: SubjectData,
    *,
    model_component_id: str,
    model_kwargs: Mapping[str, Any] | None,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None = None,
    requirements: ComponentRequirements | None = None,
    initial_group_location: Mapping[str, float] | None = None,
    initial_group_scale: Mapping[str, float] | None = None,
    initial_block_params: Sequence[Mapping[str, float]] | None = None,
    mu_prior_mean: float | Mapping[str, float] = 0.0,
    mu_prior_std: float | Mapping[str, float] = 2.0,
    log_sigma_prior_mean: float | Mapping[str, float] = -1.0,
    log_sigma_prior_std: float | Mapping[str, float] = 1.0,
    n_samples: int = 1000,
    n_warmup: int = 500,
    thin: int = 1,
    n_chains: int = 4,
    parallel_chains: int | None = None,
    adapt_delta: float = 0.9,
    max_treedepth: int = 12,
    step_size: float | None = None,
    random_seed: int | None = None,
    refresh: int = 0,
) -> HierarchicalSubjectPosteriorResult:
    """Sample within-subject hierarchical posterior using Stan NUTS.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset.
    model_component_id : str
        Model component ID supported by the Stan backend.
    model_kwargs : Mapping[str, Any] | None
        Fixed model keyword arguments for parameters not listed in
        ``parameter_names``.
    parameter_names : Sequence[str]
        Hierarchically pooled parameter names.
    transform_kinds : Mapping[str, str] | None, optional
        Per-parameter transform kind mapping. Supported values are
        ``"identity"``, ``"unit_interval_logit"``, and ``"positive_log"``.
        Missing names use safe defaults for known parameters.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked per block trace.
    initial_group_location : Mapping[str, float] | None, optional
        Optional constrained initial group-location values.
    initial_group_scale : Mapping[str, float] | None, optional
        Optional positive initial group-scale values.
    initial_block_params : Sequence[Mapping[str, float]] | None, optional
        Optional constrained per-block initial parameter values.
    mu_prior_mean : float | Mapping[str, float], optional
        Group-location latent prior mean. When a mapping is provided, values
        are applied per sampled parameter name.
    mu_prior_std : float | Mapping[str, float], optional
        Group-location latent prior standard deviation (must be > 0). Mapping
        form applies per sampled parameter name.
    log_sigma_prior_mean : float | Mapping[str, float], optional
        Group log-scale latent prior mean. Mapping form applies per sampled
        parameter name.
    log_sigma_prior_std : float | Mapping[str, float], optional
        Group log-scale latent prior standard deviation (must be > 0). Mapping
        form applies per sampled parameter name.
    n_samples : int, optional
        Number of retained post-warmup draws per chain.
    n_warmup : int, optional
        Number of warmup iterations per chain.
    thin : int, optional
        Thinning interval.
    n_chains : int, optional
        Number of MCMC chains.
    parallel_chains : int | None, optional
        Optional number of chains to run in parallel.
    adapt_delta : float, optional
        NUTS target acceptance statistic in ``(0, 1)``.
    max_treedepth : int, optional
        NUTS max tree depth.
    step_size : float | None, optional
        Optional initial step size.
    random_seed : int | None, optional
        Optional RNG seed.
    refresh : int, optional
        CmdStan progress refresh interval.

    Returns
    -------
    HierarchicalSubjectPosteriorResult
        Subject-level hierarchical posterior output.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_warmup < 0:
        raise ValueError("n_warmup must be >= 0")
    if thin <= 0:
        raise ValueError("thin must be > 0")
    if n_chains <= 0:
        raise ValueError("n_chains must be > 0")
    if parallel_chains is not None and parallel_chains <= 0:
        raise ValueError("parallel_chains must be > 0 when provided")
    if adapt_delta <= 0.0 or adapt_delta >= 1.0:
        raise ValueError("adapt_delta must be in (0, 1)")
    if max_treedepth <= 0:
        raise ValueError("max_treedepth must be > 0")
    built, stan_code, cache_tag, compatibility = _build_subject_stan_job(
        subject=subject,
        model_component_id=model_component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        requirements=requirements,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params=initial_block_params,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
    )

    fit = _run_stan_hierarchical_nuts(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        n_samples=int(n_samples),
        n_warmup=int(n_warmup),
        thin=int(thin),
        n_chains=int(n_chains),
        parallel_chains=parallel_chains,
        adapt_delta=float(adapt_delta),
        max_treedepth=int(max_treedepth),
        step_size=step_size,
        random_seed=random_seed,
        refresh=int(refresh),
    )
    result = _decode_subject_fit(
        fit=fit,
        subject_id=subject.subject_id,
        parameter_names=built.parameter_names,
        n_blocks=built.n_blocks,
        n_samples=int(n_samples),
        n_warmup=int(n_warmup),
        thin=int(thin),
        n_chains=int(n_chains),
        random_seed=random_seed,
    )
    return HierarchicalSubjectPosteriorResult(
        subject_id=result.subject_id,
        parameter_names=result.parameter_names,
        draws=result.draws,
        diagnostics=result.diagnostics,
        compatibility=compatibility,
    )


def sample_study_hierarchical_posterior_stan(
    study: StudyData,
    *,
    model_component_id: str,
    model_kwargs: Mapping[str, Any] | None,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None = None,
    requirements: ComponentRequirements | None = None,
    initial_group_location: Mapping[str, float] | None = None,
    initial_group_scale: Mapping[str, float] | None = None,
    initial_block_params_by_subject: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
    mu_prior_mean: float | Mapping[str, float] = 0.0,
    mu_prior_std: float | Mapping[str, float] = 2.0,
    log_sigma_prior_mean: float | Mapping[str, float] = -1.0,
    log_sigma_prior_std: float | Mapping[str, float] = 1.0,
    n_samples: int = 1000,
    n_warmup: int = 500,
    thin: int = 1,
    n_chains: int = 4,
    parallel_chains: int | None = None,
    adapt_delta: float = 0.9,
    max_treedepth: int = 12,
    step_size: float | None = None,
    random_seed: int | None = None,
    refresh: int = 0,
) -> HierarchicalStudyPosteriorResult:
    """Sample within-subject hierarchical posterior for all subjects with Stan."""

    subject_results: list[HierarchicalSubjectPosteriorResult] = []
    for index, subject in enumerate(study.subjects):
        subject_initial_block_params = None
        if initial_block_params_by_subject is not None:
            subject_initial_block_params = initial_block_params_by_subject.get(subject.subject_id)

        subject_results.append(
            sample_subject_hierarchical_posterior_stan(
                subject,
                model_component_id=model_component_id,
                model_kwargs=model_kwargs,
                parameter_names=parameter_names,
                transform_kinds=transform_kinds,
                requirements=requirements,
                initial_group_location=initial_group_location,
                initial_group_scale=initial_group_scale,
                initial_block_params=subject_initial_block_params,
                mu_prior_mean=mu_prior_mean,
                mu_prior_std=mu_prior_std,
                log_sigma_prior_mean=log_sigma_prior_mean,
                log_sigma_prior_std=log_sigma_prior_std,
                n_samples=n_samples,
                n_warmup=n_warmup,
                thin=thin,
                n_chains=n_chains,
                parallel_chains=parallel_chains,
                adapt_delta=adapt_delta,
                max_treedepth=max_treedepth,
                step_size=step_size,
                random_seed=None if random_seed is None else int(random_seed) + (index * 1000),
                refresh=refresh,
            )
        )

    return HierarchicalStudyPosteriorResult(subject_results=tuple(subject_results))


def optimize_subject_hierarchical_posterior_stan(
    subject: SubjectData,
    *,
    model_component_id: str,
    model_kwargs: Mapping[str, Any] | None,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None = None,
    requirements: ComponentRequirements | None = None,
    initial_group_location: Mapping[str, float] | None = None,
    initial_group_scale: Mapping[str, float] | None = None,
    initial_block_params: Sequence[Mapping[str, float]] | None = None,
    mu_prior_mean: float | Mapping[str, float] = 0.0,
    mu_prior_std: float | Mapping[str, float] = 2.0,
    log_sigma_prior_mean: float | Mapping[str, float] = -1.0,
    log_sigma_prior_std: float | Mapping[str, float] = 1.0,
    method: str = "lbfgs",
    max_iterations: int = 2000,
    jacobian: bool = False,
    init_alpha: float | None = None,
    tol_obj: float | None = None,
    tol_rel_obj: float | None = None,
    tol_grad: float | None = None,
    tol_rel_grad: float | None = None,
    tol_param: float | None = None,
    history_size: int | None = None,
    random_seed: int | None = None,
    refresh: int = 0,
) -> HierarchicalSubjectPosteriorResult:
    """Compute one within-subject hierarchical posterior mode with Stan.

    This uses CmdStan ``optimize`` and returns a
    :class:`HierarchicalSubjectPosteriorResult` with one retained draw
    representing the optimized mode.
    """

    method_name = str(method).strip().lower()
    if method_name not in {"lbfgs", "bfgs", "newton"}:
        raise ValueError("method must be one of {'lbfgs', 'bfgs', 'newton'}")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")
    if init_alpha is not None and init_alpha <= 0.0:
        raise ValueError("init_alpha must be > 0 when provided")
    if refresh < 0:
        raise ValueError("refresh must be >= 0")
    if history_size is not None and history_size <= 0:
        raise ValueError("history_size must be > 0 when provided")

    built, stan_code, cache_tag, compatibility = _build_subject_stan_job(
        subject=subject,
        model_component_id=model_component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        requirements=requirements,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params=initial_block_params,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
    )

    fit = _run_stan_hierarchical_optimize(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        method=method_name,
        max_iterations=int(max_iterations),
        jacobian=bool(jacobian),
        init_alpha=init_alpha,
        tol_obj=tol_obj,
        tol_rel_obj=tol_rel_obj,
        tol_grad=tol_grad,
        tol_rel_grad=tol_rel_grad,
        tol_param=tol_param,
        history_size=history_size,
        random_seed=random_seed,
        refresh=int(refresh),
    )
    result = _decode_subject_optimize_fit(
        fit=fit,
        subject_id=subject.subject_id,
        parameter_names=built.parameter_names,
        n_blocks=built.n_blocks,
        random_seed=random_seed,
    )
    return HierarchicalSubjectPosteriorResult(
        subject_id=result.subject_id,
        parameter_names=result.parameter_names,
        draws=result.draws,
        diagnostics=result.diagnostics,
        compatibility=compatibility,
    )


def optimize_study_hierarchical_posterior_stan(
    study: StudyData,
    *,
    model_component_id: str,
    model_kwargs: Mapping[str, Any] | None,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None = None,
    requirements: ComponentRequirements | None = None,
    initial_group_location: Mapping[str, float] | None = None,
    initial_group_scale: Mapping[str, float] | None = None,
    initial_block_params_by_subject: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
    mu_prior_mean: float | Mapping[str, float] = 0.0,
    mu_prior_std: float | Mapping[str, float] = 2.0,
    log_sigma_prior_mean: float | Mapping[str, float] = -1.0,
    log_sigma_prior_std: float | Mapping[str, float] = 1.0,
    method: str = "lbfgs",
    max_iterations: int = 2000,
    jacobian: bool = False,
    init_alpha: float | None = None,
    tol_obj: float | None = None,
    tol_rel_obj: float | None = None,
    tol_grad: float | None = None,
    tol_rel_grad: float | None = None,
    tol_param: float | None = None,
    history_size: int | None = None,
    random_seed: int | None = None,
    refresh: int = 0,
) -> HierarchicalStudyPosteriorResult:
    """Compute within-subject hierarchical posterior modes for all subjects."""

    subject_results: list[HierarchicalSubjectPosteriorResult] = []
    for index, subject in enumerate(study.subjects):
        subject_initial_block_params = None
        if initial_block_params_by_subject is not None:
            subject_initial_block_params = initial_block_params_by_subject.get(subject.subject_id)

        subject_results.append(
            optimize_subject_hierarchical_posterior_stan(
                subject,
                model_component_id=model_component_id,
                model_kwargs=model_kwargs,
                parameter_names=parameter_names,
                transform_kinds=transform_kinds,
                requirements=requirements,
                initial_group_location=initial_group_location,
                initial_group_scale=initial_group_scale,
                initial_block_params=subject_initial_block_params,
                mu_prior_mean=mu_prior_mean,
                mu_prior_std=mu_prior_std,
                log_sigma_prior_mean=log_sigma_prior_mean,
                log_sigma_prior_std=log_sigma_prior_std,
                method=method,
                max_iterations=max_iterations,
                jacobian=jacobian,
                init_alpha=init_alpha,
                tol_obj=tol_obj,
                tol_rel_obj=tol_rel_obj,
                tol_grad=tol_grad,
                tol_rel_grad=tol_rel_grad,
                tol_param=tol_param,
                history_size=history_size,
                random_seed=None if random_seed is None else int(random_seed) + (index * 1000),
                refresh=refresh,
            )
        )

    return HierarchicalStudyPosteriorResult(subject_results=tuple(subject_results))


def _build_subject_stan_job(
    *,
    subject: SubjectData,
    model_component_id: str,
    model_kwargs: Mapping[str, Any] | None,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None,
    requirements: ComponentRequirements | None,
    initial_group_location: Mapping[str, float] | None,
    initial_group_scale: Mapping[str, float] | None,
    initial_block_params: Sequence[Mapping[str, float]] | None,
    mu_prior_mean: float | Mapping[str, float],
    mu_prior_std: float | Mapping[str, float],
    log_sigma_prior_mean: float | Mapping[str, float],
    log_sigma_prior_std: float | Mapping[str, float],
) -> tuple[_SubjectBuild, str, str, CompatibilityReport | None]:
    """Assemble validated Stan inputs and source code for one subject."""

    if model_component_id not in _SUPPORTED_COMPONENT_IDS:
        raise ValueError(
            "Stan hierarchical backend does not support "
            f"{model_component_id!r}; supported component IDs are "
            f"{sorted(_SUPPORTED_COMPONENT_IDS)}"
        )

    compatibility: CompatibilityReport | None = None
    if requirements is not None:
        for block in subject.blocks:
            trace = get_block_trace(block)
            compatibility = check_trace_compatibility(trace, requirements)
            assert_trace_compatible(trace, requirements)

    if model_component_id in _ASOCIAL_COMPONENT_IDS:
        built = _build_asocial_subject_inputs(
            model_component_id=model_component_id,
            subject=subject,
            parameter_names=parameter_names,
            transform_kinds=transform_kinds,
            model_kwargs=model_kwargs,
            initial_group_location=initial_group_location,
            initial_group_scale=initial_group_scale,
            initial_block_params=initial_block_params,
            mu_prior_mean=mu_prior_mean,
            mu_prior_std=mu_prior_std,
            log_sigma_prior_mean=log_sigma_prior_mean,
            log_sigma_prior_std=log_sigma_prior_std,
        )
        return built, _ASOCIAL_STAN_CODE, f"hierarchical_{model_component_id}", compatibility

    social_stan_data, social_param_names, social_n_blocks = build_social_subject_inputs(
        subject=subject,
        component_id=model_component_id,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        model_kwargs=model_kwargs,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params=initial_block_params,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
    )
    built = _SubjectBuild(
        stan_data=social_stan_data,
        parameter_names=social_param_names,
        n_blocks=social_n_blocks,
    )
    return built, load_social_stan_code(model_component_id), social_cache_tag(model_component_id), compatibility


def _build_asocial_subject_inputs(
    *,
    model_component_id: str,
    subject: SubjectData,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None,
    model_kwargs: Mapping[str, Any] | None,
    initial_group_location: Mapping[str, float] | None,
    initial_group_scale: Mapping[str, float] | None,
    initial_block_params: Sequence[Mapping[str, float]] | None,
    mu_prior_mean: float | Mapping[str, float],
    mu_prior_std: float | Mapping[str, float],
    log_sigma_prior_mean: float | Mapping[str, float],
    log_sigma_prior_std: float | Mapping[str, float],
) -> _SubjectBuild:
    """Build Stan data dictionary and metadata for one subject."""

    if model_component_id not in _ASOCIAL_SPECS:
        raise ValueError(
            f"unsupported asocial model_component_id {model_component_id!r}; "
            f"expected one of {sorted(_ASOCIAL_SPECS)}"
        )
    spec = _ASOCIAL_SPECS[model_component_id]

    names = tuple(str(name) for name in parameter_names)
    if len(names) == 0:
        raise ValueError("parameter_names must include at least one parameter")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique")

    unknown = [name for name in names if name not in spec.allowed_parameter_names]
    if unknown:
        raise ValueError(
            "unsupported parameter_names for Stan backend: "
            f"{unknown!r}; supported {sorted(spec.allowed_parameter_names)}"
        )

    resolved_transform_kinds = _resolve_transform_kinds(
        parameter_names=names,
        transform_kinds=transform_kinds,
    )
    mu_prior_mean_vec = _resolve_prior_vector(
        mu_prior_mean,
        parameter_names=names,
        default_value=0.0,
        field_name="mu_prior_mean",
    )
    mu_prior_std_vec = _resolve_prior_vector(
        mu_prior_std,
        parameter_names=names,
        default_value=2.0,
        field_name="mu_prior_std",
        must_be_positive=True,
    )
    log_sigma_prior_mean_vec = _resolve_prior_vector(
        log_sigma_prior_mean,
        parameter_names=names,
        default_value=-1.0,
        field_name="log_sigma_prior_mean",
    )
    log_sigma_prior_std_vec = _resolve_prior_vector(
        log_sigma_prior_std,
        parameter_names=names,
        default_value=1.0,
        field_name="log_sigma_prior_std",
        must_be_positive=True,
    )
    fixed_external = dict(spec.default_params)
    provided_kwargs = dict(model_kwargs or {})
    for key in provided_kwargs:
        if key not in spec.allowed_model_kwargs:
            raise ValueError(
                f"unsupported model kwarg {key!r} for Stan backend; "
                f"supported keys are {sorted(spec.allowed_model_kwargs)}"
            )
        if key in names:
            raise ValueError(
                f"model kwarg {key!r} conflicts with sampled parameter_names; remove one source"
            )
        fixed_external[key] = float(provided_kwargs[key])

    fixed_internal = {
        "alpha_1": float(fixed_external.get("alpha_1", fixed_external.get("alpha", 0.2))),
        "alpha_2": float(fixed_external.get("alpha_2", 0.0)),
        "beta": float(fixed_external.get("beta", 5.0)),
        "kappa": float(fixed_external.get("kappa", 0.0)),
        "initial_value": float(fixed_external.get("initial_value", 0.0)),
    }

    if fixed_internal["alpha_1"] < 0.0 or fixed_internal["alpha_1"] > 1.0:
        raise ValueError("fixed alpha_1 must be in [0, 1]")
    if fixed_internal["alpha_2"] < 0.0 or fixed_internal["alpha_2"] > 1.0:
        raise ValueError("fixed alpha_2 must be in [0, 1]")
    if fixed_internal["beta"] < 0.0:
        raise ValueError("fixed beta must be >= 0")

    block_rows = tuple(_rows_for_actor(block, actor_id="subject") for block in subject.blocks)
    if initial_block_params is not None and len(initial_block_params) != len(block_rows):
        raise ValueError("initial_block_params must match number of subject blocks")

    action_to_index: dict[Any, int] = {}
    state_to_index: dict[int, int] = {}

    for rows in block_rows:
        for row in rows:
            if row.available_actions is None:
                raise ValueError("trial decision requires available_actions for Stan backend")
            if row.action is None:
                raise ValueError("trial decision requires action for Stan backend")

            state = _state_from_observation(row.observation) if spec.stateful else 0
            if state not in state_to_index:
                state_to_index[state] = len(state_to_index) + 1

            for action in row.available_actions:
                if action not in action_to_index:
                    action_to_index[action] = len(action_to_index) + 1
            if row.action not in action_to_index:
                action_to_index[row.action] = len(action_to_index) + 1

    if not action_to_index:
        raise ValueError("no available actions found for Stan backend")
    if not state_to_index:
        raise ValueError("no states found for Stan backend")

    n_blocks = len(block_rows)
    block_lengths = [len(rows) for rows in block_rows]
    if any(length <= 0 for length in block_lengths):
        raise ValueError("each block must include at least one subject decision")

    t_max = max(block_lengths)
    n_actions = len(action_to_index)
    n_states = len(state_to_index)

    state_idx = np.ones((n_blocks, t_max), dtype=int)
    action_idx = np.ones((n_blocks, t_max), dtype=int)
    reward = np.zeros((n_blocks, t_max), dtype=float)
    is_available = np.zeros((n_blocks, t_max, n_actions), dtype=int)

    for block_index, rows in enumerate(block_rows):
        for decision_index, row in enumerate(rows):
            state_value = _state_from_observation(row.observation) if spec.stateful else 0
            if row.available_actions is None or row.action is None:
                raise ValueError("trial decision is missing actions for Stan backend")

            state_idx[block_index, decision_index] = state_to_index[state_value]
            action_idx[block_index, decision_index] = action_to_index[row.action]
            reward[block_index, decision_index] = _reward_from_row(row)

            if len(row.available_actions) == 0:
                raise ValueError("available_actions must not be empty")
            for available_action in row.available_actions:
                is_available[block_index, decision_index, action_to_index[available_action] - 1] = 1

            chosen_position = action_to_index[row.action] - 1
            if is_available[block_index, decision_index, chosen_position] != 1:
                raise ValueError("observed action must be present in available_actions")

    group_loc_init = np.zeros(len(names), dtype=float)
    group_log_scale_init = np.zeros(len(names), dtype=float)
    if initial_group_location is not None:
        for param_index, name in enumerate(names):
            if name in initial_group_location:
                theta = float(initial_group_location[name])
                group_loc_init[param_index] = _inverse_transform(theta, resolved_transform_kinds[name])

    if initial_group_scale is not None:
        for param_index, name in enumerate(names):
            if name in initial_group_scale:
                sigma = float(initial_group_scale[name])
                if sigma <= 0.0:
                    raise ValueError(f"initial_group_scale[{name!r}] must be > 0")
                group_log_scale_init[param_index] = float(np.log(sigma))

    block_z_init = np.zeros((n_blocks, len(names)), dtype=float)
    if initial_block_params is not None:
        for block_index, block_params in enumerate(initial_block_params):
            for param_index, name in enumerate(names):
                if name in block_params:
                    theta = float(block_params[name])
                    block_z_init[block_index, param_index] = _inverse_transform(
                        theta,
                        resolved_transform_kinds[name],
                    )
                elif initial_group_location is not None and name in initial_group_location:
                    block_z_init[block_index, param_index] = group_loc_init[param_index]
                else:
                    block_z_init[block_index, param_index] = 0.0
    else:
        for param_index, name in enumerate(names):
            default_value = 0.0
            if initial_group_location is not None and name in initial_group_location:
                default_value = group_loc_init[param_index]
            block_z_init[:, param_index] = default_value

    stan_data: dict[str, Any] = {
        "B": n_blocks,
        "K": len(names),
        "S": n_states,
        "A": n_actions,
        "T_max": int(t_max),
        "T": [int(value) for value in block_lengths],
        "state_idx": state_idx.tolist(),
        "action_idx": action_idx.tolist(),
        "reward": reward.tolist(),
        "is_available": is_available.tolist(),
        "param_codes": [_ASOCIAL_EXTERNAL_PARAM_CODE_BY_NAME[name] for name in names],
        "transform_codes": [_TRANSFORM_CODE_BY_KIND[resolved_transform_kinds[name]] for name in names],
        "fixed_alpha_1": float(fixed_internal["alpha_1"]),
        "fixed_alpha_2": float(fixed_internal["alpha_2"]),
        "fixed_beta": float(fixed_internal["beta"]),
        "fixed_kappa": float(fixed_internal["kappa"]),
        "fixed_initial_value": float(fixed_internal["initial_value"]),
        "mu_prior_mean": mu_prior_mean_vec.tolist(),
        "mu_prior_std": mu_prior_std_vec.tolist(),
        "log_sigma_prior_mean": log_sigma_prior_mean_vec.tolist(),
        "log_sigma_prior_std": log_sigma_prior_std_vec.tolist(),
        "group_loc_init": group_loc_init.tolist(),
        "group_log_scale_init": group_log_scale_init.tolist(),
        "block_z_init": block_z_init.tolist(),
    }
    return _SubjectBuild(stan_data=stan_data, parameter_names=names, n_blocks=n_blocks)


def _rows_for_actor(block: BlockData, *, actor_id: str) -> tuple[TrialDecision, ...]:
    """Return decision rows for one actor in chronological order."""

    if block.trials:
        rows = tuple(block.trials)
    elif block.event_trace is not None:
        rows = trial_decisions_from_trace(block.event_trace)
    else:
        raise ValueError("block has neither trials nor event_trace")

    filtered = tuple(row for row in rows if row.actor_id == actor_id)
    if len(filtered) == 0:
        raise ValueError(
            f"block {block.block_id!r} has no decisions for actor {actor_id!r}; "
            "Stan backend currently fits a single actor per block"
        )
    return filtered


def _reward_from_row(row: TrialDecision) -> float:
    """Extract scalar reward from one trial decision row."""

    if row.reward is not None:
        return float(row.reward)

    if isinstance(row.outcome, Mapping) and "reward" in row.outcome:
        return float(row.outcome["reward"])

    if row.outcome is not None and hasattr(row.outcome, "reward"):
        return float(getattr(row.outcome, "reward"))

    raise ValueError(
        f"missing reward for trial_index={row.trial_index} decision_index={row.decision_index}"
    )


def _state_from_observation(observation: Any) -> int:
    """Extract integer state index from observation payload."""

    if isinstance(observation, Mapping) and "state" in observation:
        return int(observation["state"])
    return 0


def _resolve_transform_kinds(
    *,
    parameter_names: tuple[str, ...],
    transform_kinds: Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve transform kind per parameter name with validation."""

    raw_mapping = dict(transform_kinds or {})
    out: dict[str, str] = {}
    for name in parameter_names:
        raw_kind = raw_mapping.get(name, _default_transform_kind(name))
        kind = str(raw_kind).strip()
        if kind not in _TRANSFORM_CODE_BY_KIND:
            raise ValueError(
                f"unsupported transform kind {kind!r} for parameter {name!r}; "
                f"expected one of {sorted(_TRANSFORM_CODE_BY_KIND)}"
            )
        out[name] = kind
    return out


def _resolve_prior_vector(
    spec: float | Mapping[str, float],
    *,
    parameter_names: tuple[str, ...],
    default_value: float,
    field_name: str,
    must_be_positive: bool = False,
) -> np.ndarray:
    """Resolve scalar/mapping prior input into parameter-aligned vector."""

    values = np.full(len(parameter_names), float(default_value), dtype=float)
    if isinstance(spec, Mapping):
        mapping = {str(key): float(value) for key, value in spec.items()}
        unknown = sorted(set(mapping).difference(parameter_names))
        if unknown:
            raise ValueError(
                f"{field_name} has unknown parameter names {unknown!r}; "
                f"expected subset of {list(parameter_names)!r}"
            )
        for index, name in enumerate(parameter_names):
            if name in mapping:
                values[index] = float(mapping[name])
    else:
        values[:] = float(spec)

    if must_be_positive and np.any(values <= 0.0):
        raise ValueError(f"{field_name} values must be > 0")
    return values


def _default_transform_kind(parameter_name: str) -> str:
    """Return safe default transform kind for known parameters."""

    if parameter_name in {"alpha", "alpha_1", "alpha_2"}:
        return "unit_interval_logit"
    if parameter_name == "beta":
        return "positive_log"
    return "identity"


def _inverse_transform(value: float, kind: str) -> float:
    """Map constrained value to latent ``z`` according to transform kind."""

    if kind == "identity":
        return float(value)
    if kind == "positive_log":
        clipped = max(float(value), 1e-12)
        return float(np.log(clipped))
    if kind == "unit_interval_logit":
        theta = float(np.clip(float(value), 1e-9, 1.0 - 1e-9))
        return float(np.log(theta / (1.0 - theta)))
    raise ValueError(f"unsupported transform kind {kind!r}")


def _run_stan_hierarchical_nuts(
    *,
    stan_code: str,
    cache_tag: str,
    stan_data: Mapping[str, Any],
    n_samples: int,
    n_warmup: int,
    thin: int,
    n_chains: int,
    parallel_chains: int | None,
    adapt_delta: float,
    max_treedepth: int,
    step_size: float | None,
    random_seed: int | None,
    refresh: int,
) -> Any:
    """Compile and sample one Stan hierarchical model."""

    model = compile_cmdstan_model(stan_code, cache_tag=cache_tag)
    init_data = {
        "group_loc_z": list(stan_data["group_loc_init"]),
        "group_log_scale": list(stan_data["group_log_scale_init"]),
        "block_z": list(stan_data["block_z_init"]),
    }

    sample_kwargs: dict[str, Any] = {
        "data": {key: value for key, value in stan_data.items() if not key.endswith("_init")},
        "inits": init_data,
        "iter_sampling": int(n_samples),
        "iter_warmup": int(n_warmup),
        "thin": int(thin),
        "chains": int(n_chains),
        "adapt_delta": float(adapt_delta),
        "max_treedepth": int(max_treedepth),
        "refresh": int(refresh),
    }
    if parallel_chains is not None:
        sample_kwargs["parallel_chains"] = int(parallel_chains)
    if random_seed is not None:
        sample_kwargs["seed"] = int(random_seed)
    if step_size is not None:
        sample_kwargs["step_size"] = float(step_size)

    return model.sample(**sample_kwargs)


def _run_stan_hierarchical_optimize(
    *,
    stan_code: str,
    cache_tag: str,
    stan_data: Mapping[str, Any],
    method: str,
    max_iterations: int,
    jacobian: bool,
    init_alpha: float | None,
    tol_obj: float | None,
    tol_rel_obj: float | None,
    tol_grad: float | None,
    tol_rel_grad: float | None,
    tol_param: float | None,
    history_size: int | None,
    random_seed: int | None,
    refresh: int,
) -> Any:
    """Compile and optimize one Stan hierarchical model."""

    model = compile_cmdstan_model(stan_code, cache_tag=cache_tag)
    init_data = {
        "group_loc_z": list(stan_data["group_loc_init"]),
        "group_log_scale": list(stan_data["group_log_scale_init"]),
        "block_z": list(stan_data["block_z_init"]),
    }

    optimize_kwargs: dict[str, Any] = {
        "data": {key: value for key, value in stan_data.items() if not key.endswith("_init")},
        "inits": init_data,
        "algorithm": method,
        "iter": int(max_iterations),
        "jacobian": bool(jacobian),
        "refresh": int(refresh),
    }
    if random_seed is not None:
        optimize_kwargs["seed"] = int(random_seed)
    if init_alpha is not None:
        optimize_kwargs["init_alpha"] = float(init_alpha)
    if tol_obj is not None:
        optimize_kwargs["tol_obj"] = float(tol_obj)
    if tol_rel_obj is not None:
        optimize_kwargs["tol_rel_obj"] = float(tol_rel_obj)
    if tol_grad is not None:
        optimize_kwargs["tol_grad"] = float(tol_grad)
    if tol_rel_grad is not None:
        optimize_kwargs["tol_rel_grad"] = float(tol_rel_grad)
    if tol_param is not None:
        optimize_kwargs["tol_param"] = float(tol_param)
    if history_size is not None:
        optimize_kwargs["history_size"] = int(history_size)

    return model.optimize(**optimize_kwargs)


def _decode_subject_fit(
    *,
    fit: Any,
    subject_id: str,
    parameter_names: tuple[str, ...],
    n_blocks: int,
    n_samples: int,
    n_warmup: int,
    thin: int,
    n_chains: int,
    random_seed: int | None,
) -> HierarchicalSubjectPosteriorResult:
    """Decode CmdStan fit object into public hierarchical posterior result."""

    group_loc = np.asarray(fit.stan_variable("group_loc_z"), dtype=float)
    if group_loc.ndim == 1:
        group_loc = group_loc.reshape((-1, 1))
    n_draws = int(group_loc.shape[0])
    if n_draws == 0:
        raise ValueError("Stan fit returned zero posterior draws")

    k = len(parameter_names)
    group_loc = group_loc.reshape((n_draws, k))

    group_log_scale = np.asarray(fit.stan_variable("group_log_scale"), dtype=float).reshape((n_draws, k))
    block_z = np.asarray(fit.stan_variable("block_z"), dtype=float).reshape((n_draws, n_blocks, k))
    block_param = np.asarray(fit.stan_variable("block_param"), dtype=float).reshape((n_draws, n_blocks, k))

    log_likelihood = np.asarray(fit.stan_variable("log_likelihood_total"), dtype=float).reshape((n_draws,))
    log_prior = np.asarray(fit.stan_variable("log_prior_total"), dtype=float).reshape((n_draws,))
    log_posterior = np.asarray(fit.stan_variable("log_posterior_total"), dtype=float).reshape((n_draws,))

    draws: list[HierarchicalMCMCDraw] = []
    for draw_index in range(n_draws):
        group_location_z = {
            name: float(group_loc[draw_index, param_index])
            for param_index, name in enumerate(parameter_names)
        }
        group_scale_z = {
            name: float(np.exp(group_log_scale[draw_index, param_index]))
            for param_index, name in enumerate(parameter_names)
        }

        block_params_z: list[dict[str, float]] = []
        block_params: list[dict[str, float]] = []
        for block_index in range(n_blocks):
            block_params_z.append(
                {
                    name: float(block_z[draw_index, block_index, param_index])
                    for param_index, name in enumerate(parameter_names)
                }
            )
            block_params.append(
                {
                    name: float(block_param[draw_index, block_index, param_index])
                    for param_index, name in enumerate(parameter_names)
                }
            )

        candidate = HierarchicalPosteriorCandidate(
            parameter_names=parameter_names,
            group_location_z=group_location_z,
            group_scale_z=group_scale_z,
            block_params_z=tuple(block_params_z),
            block_params=tuple(block_params),
            log_likelihood=float(log_likelihood[draw_index]),
            log_prior=float(log_prior[draw_index]),
            log_posterior=float(log_posterior[draw_index]),
        )
        draws.append(
            HierarchicalMCMCDraw(
                candidate=candidate,
                accepted=True,
                iteration=draw_index,
            )
        )

    n_iterations = int((n_warmup + (n_samples * thin)) * n_chains)
    diagnostics = MCMCDiagnostics(
        method="within_subject_hierarchical_stan_nuts",
        n_iterations=n_iterations,
        n_warmup=int(n_warmup * n_chains),
        n_kept_draws=n_draws,
        thin=int(thin),
        n_accepted=n_iterations,
        acceptance_rate=1.0,
        random_seed=random_seed,
    )
    return HierarchicalSubjectPosteriorResult(
        subject_id=subject_id,
        parameter_names=parameter_names,
        draws=tuple(draws),
        diagnostics=diagnostics,
        compatibility=None,
    )


def _decode_subject_optimize_fit(
    *,
    fit: Any,
    subject_id: str,
    parameter_names: tuple[str, ...],
    n_blocks: int,
    random_seed: int | None,
) -> HierarchicalSubjectPosteriorResult:
    """Decode one CmdStan optimize result as a single retained posterior draw."""

    decoded = _decode_subject_fit(
        fit=fit,
        subject_id=subject_id,
        parameter_names=parameter_names,
        n_blocks=n_blocks,
        n_samples=1,
        n_warmup=0,
        thin=1,
        n_chains=1,
        random_seed=random_seed,
    )
    diagnostics = MCMCDiagnostics(
        method="within_subject_hierarchical_stan_map",
        n_iterations=1,
        n_warmup=0,
        n_kept_draws=len(decoded.draws),
        thin=1,
        n_accepted=1,
        acceptance_rate=1.0,
        random_seed=random_seed,
    )
    return HierarchicalSubjectPosteriorResult(
        subject_id=decoded.subject_id,
        parameter_names=decoded.parameter_names,
        draws=decoded.draws,
        diagnostics=diagnostics,
        compatibility=None,
    )


__all__ = [
    "optimize_study_hierarchical_posterior_stan",
    "optimize_subject_hierarchical_posterior_stan",
    "sample_study_hierarchical_posterior_stan",
    "sample_subject_hierarchical_posterior_stan",
]
