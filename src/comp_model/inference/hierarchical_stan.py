"""Stan backend for explicit subject and study Bayesian hierarchy estimators.

Supported public structures:

- ``subject_shared``: one parameter set shared across a subject's blocks
- ``subject_block_hierarchy``: subject -> block
- ``study_subject_hierarchy``: population -> subject
- ``study_subject_block_hierarchy``: population -> subject -> block
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
from .hierarchical_stan_social import (
    build_social_subject_inputs,
    load_social_stan_code,
    social_cache_tag,
    social_supported_component_ids,
)
from .mcmc_diagnostics import MCMCDiagnostics
from .stan_backend import compile_cmdstan_model
from .stan_posterior import (
    StanPosteriorDraw,
    StudySubjectBlockHierarchyPosteriorCandidate,
    StudySubjectBlockHierarchyPosteriorResult,
    StudySubjectHierarchyPosteriorCandidate,
    StudySubjectHierarchyPosteriorResult,
    SubjectBlockHierarchyPosteriorCandidate,
    SubjectBlockHierarchyPosteriorResult,
    SubjectSharedPosteriorCandidate,
    SubjectSharedPosteriorResult,
)

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

_STRUCTURE_SUBJECT_SHARED = "subject_shared"
_STRUCTURE_SUBJECT_BLOCK = "subject_block_hierarchy"
_STRUCTURE_STUDY_SUBJECT = "study_subject_hierarchy"
_STRUCTURE_STUDY_SUBJECT_BLOCK = "study_subject_block_hierarchy"

_STAN_DIR = Path(__file__).with_name("stan") / "within_subject"


def _load_stan_source(filename: str) -> str:
    """Read one Stan program file from package-local stan directory."""

    path = _STAN_DIR / filename
    if not path.exists():
        raise RuntimeError(f"Stan program file is missing: {path}")
    return path.read_text(encoding="utf-8")


_ASOCIAL_STAN_CODE_BY_STRUCTURE: dict[str, str] = {
    _STRUCTURE_SUBJECT_SHARED: _load_stan_source("asocial_subject_shared.stan"),
    _STRUCTURE_SUBJECT_BLOCK: _load_stan_source("asocial_subject_block_hierarchy.stan"),
    _STRUCTURE_STUDY_SUBJECT: _load_stan_source("asocial_study_subject_hierarchy.stan"),
    _STRUCTURE_STUDY_SUBJECT_BLOCK: _load_stan_source("asocial_study_subject_block_hierarchy.stan"),
}


@dataclass(frozen=True, slots=True)
class _FlatBuild:
    """Stan-ready flat block arrays before hierarchy-specific init wiring."""

    stan_data: dict[str, Any]
    parameter_names: tuple[str, ...]
    block_ids: tuple[str | int | None, ...]

    @property
    def n_blocks(self) -> int:
        """Return number of blocks represented in the flat build."""

        return len(self.block_ids)


@dataclass(frozen=True, slots=True)
class _SubjectBuild:
    """Stan inputs and metadata for one subject-level estimator."""

    stan_data: dict[str, Any]
    init_data: dict[str, Any]
    parameter_names: tuple[str, ...]
    block_ids: tuple[str | int | None, ...]

    @property
    def n_blocks(self) -> int:
        """Return number of blocks represented in the fit."""

        return len(self.block_ids)


@dataclass(frozen=True, slots=True)
class _StudyBuild:
    """Stan inputs and metadata for one study-level estimator."""

    stan_data: dict[str, Any]
    init_data: dict[str, Any]
    parameter_names: tuple[str, ...]
    subject_ids: tuple[str, ...]
    block_ids_by_subject: tuple[tuple[str | int | None, ...], ...]

    @property
    def n_blocks(self) -> int:
        """Return total number of blocks across the study."""

        return sum(len(block_ids) for block_ids in self.block_ids_by_subject)

    @property
    def block_counts(self) -> tuple[int, ...]:
        """Return per-subject block counts."""

        return tuple(len(block_ids) for block_ids in self.block_ids_by_subject)


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


def draw_subject_shared_posterior_stan(
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
) -> SubjectSharedPosteriorResult:
    """Draw subject-shared posterior samples with Stan NUTS."""

    _validate_nuts_args(
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        n_chains=n_chains,
        parallel_chains=parallel_chains,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )
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
        structure=_STRUCTURE_SUBJECT_SHARED,
    )
    fit = _run_stan_hierarchical_nuts(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        init_data=built.init_data,
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
    return _decode_subject_shared_fit(
        fit=fit,
        subject_id=subject.subject_id,
        parameter_names=built.parameter_names,
        block_ids=built.block_ids,
        diagnostics=_nuts_diagnostics(
            method="subject_shared_stan_nuts",
            n_samples=int(n_samples),
            n_warmup=int(n_warmup),
            thin=int(thin),
            n_chains=int(n_chains),
            random_seed=random_seed,
        ),
        compatibility=compatibility,
    )


def estimate_subject_shared_map_stan(
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
) -> SubjectSharedPosteriorResult:
    """Estimate the subject-shared posterior mode with Stan optimize."""

    method_name = _validate_optimize_args(
        method=method,
        max_iterations=max_iterations,
        init_alpha=init_alpha,
        refresh=refresh,
        history_size=history_size,
    )
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
        structure=_STRUCTURE_SUBJECT_SHARED,
    )
    fit = _run_stan_hierarchical_optimize(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        init_data=built.init_data,
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
    return _decode_subject_shared_fit(
        fit=fit,
        subject_id=subject.subject_id,
        parameter_names=built.parameter_names,
        block_ids=built.block_ids,
        diagnostics=_map_diagnostics(
            method="subject_shared_stan_map",
            n_kept_draws=1,
            random_seed=random_seed,
        ),
        compatibility=compatibility,
    )


def draw_subject_block_hierarchy_posterior_stan(
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
) -> SubjectBlockHierarchyPosteriorResult:
    """Draw subject -> block posterior samples with Stan NUTS."""

    _validate_nuts_args(
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        n_chains=n_chains,
        parallel_chains=parallel_chains,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )
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
        structure=_STRUCTURE_SUBJECT_BLOCK,
    )
    fit = _run_stan_hierarchical_nuts(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        init_data=built.init_data,
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
    return _decode_subject_block_hierarchy_fit(
        fit=fit,
        subject_id=subject.subject_id,
        parameter_names=built.parameter_names,
        block_ids=built.block_ids,
        diagnostics=_nuts_diagnostics(
            method="subject_block_hierarchy_stan_nuts",
            n_samples=int(n_samples),
            n_warmup=int(n_warmup),
            thin=int(thin),
            n_chains=int(n_chains),
            random_seed=random_seed,
        ),
        compatibility=compatibility,
    )


def estimate_subject_block_hierarchy_map_stan(
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
) -> SubjectBlockHierarchyPosteriorResult:
    """Estimate the subject -> block posterior mode with Stan optimize."""

    method_name = _validate_optimize_args(
        method=method,
        max_iterations=max_iterations,
        init_alpha=init_alpha,
        refresh=refresh,
        history_size=history_size,
    )
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
        structure=_STRUCTURE_SUBJECT_BLOCK,
    )
    fit = _run_stan_hierarchical_optimize(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        init_data=built.init_data,
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
    return _decode_subject_block_hierarchy_fit(
        fit=fit,
        subject_id=subject.subject_id,
        parameter_names=built.parameter_names,
        block_ids=built.block_ids,
        diagnostics=_map_diagnostics(
            method="subject_block_hierarchy_stan_map",
            n_kept_draws=1,
            random_seed=random_seed,
        ),
        compatibility=compatibility,
    )


def draw_study_subject_hierarchy_posterior_stan(
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
) -> StudySubjectHierarchyPosteriorResult:
    """Draw population -> subject posterior samples with Stan NUTS."""

    _validate_nuts_args(
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        n_chains=n_chains,
        parallel_chains=parallel_chains,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )
    built, stan_code, cache_tag, compatibility_by_subject = _build_study_stan_job(
        study=study,
        model_component_id=model_component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        requirements=requirements,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params_by_subject=initial_block_params_by_subject,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
        structure=_STRUCTURE_STUDY_SUBJECT,
    )
    fit = _run_stan_hierarchical_nuts(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        init_data=built.init_data,
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
    return _decode_study_subject_hierarchy_fit(
        fit=fit,
        parameter_names=built.parameter_names,
        subject_ids=built.subject_ids,
        block_ids_by_subject=built.block_ids_by_subject,
        diagnostics=_nuts_diagnostics(
            method="study_subject_hierarchy_stan_nuts",
            n_samples=int(n_samples),
            n_warmup=int(n_warmup),
            thin=int(thin),
            n_chains=int(n_chains),
            random_seed=random_seed,
        ),
        compatibility_by_subject=compatibility_by_subject,
    )


def estimate_study_subject_hierarchy_map_stan(
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
) -> StudySubjectHierarchyPosteriorResult:
    """Estimate the population -> subject posterior mode with Stan optimize."""

    method_name = _validate_optimize_args(
        method=method,
        max_iterations=max_iterations,
        init_alpha=init_alpha,
        refresh=refresh,
        history_size=history_size,
    )
    built, stan_code, cache_tag, compatibility_by_subject = _build_study_stan_job(
        study=study,
        model_component_id=model_component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        requirements=requirements,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params_by_subject=initial_block_params_by_subject,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
        structure=_STRUCTURE_STUDY_SUBJECT,
    )
    fit = _run_stan_hierarchical_optimize(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        init_data=built.init_data,
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
    return _decode_study_subject_hierarchy_fit(
        fit=fit,
        parameter_names=built.parameter_names,
        subject_ids=built.subject_ids,
        block_ids_by_subject=built.block_ids_by_subject,
        diagnostics=_map_diagnostics(
            method="study_subject_hierarchy_stan_map",
            n_kept_draws=1,
            random_seed=random_seed,
        ),
        compatibility_by_subject=compatibility_by_subject,
    )


def draw_study_subject_block_hierarchy_posterior_stan(
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
) -> StudySubjectBlockHierarchyPosteriorResult:
    """Draw population -> subject -> block posterior samples with Stan NUTS."""

    _validate_nuts_args(
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        n_chains=n_chains,
        parallel_chains=parallel_chains,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )
    built, stan_code, cache_tag, compatibility_by_subject = _build_study_stan_job(
        study=study,
        model_component_id=model_component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        requirements=requirements,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params_by_subject=initial_block_params_by_subject,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
        structure=_STRUCTURE_STUDY_SUBJECT_BLOCK,
    )
    fit = _run_stan_hierarchical_nuts(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        init_data=built.init_data,
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
    return _decode_study_subject_block_hierarchy_fit(
        fit=fit,
        parameter_names=built.parameter_names,
        subject_ids=built.subject_ids,
        block_ids_by_subject=built.block_ids_by_subject,
        diagnostics=_nuts_diagnostics(
            method="study_subject_block_hierarchy_stan_nuts",
            n_samples=int(n_samples),
            n_warmup=int(n_warmup),
            thin=int(thin),
            n_chains=int(n_chains),
            random_seed=random_seed,
        ),
        compatibility_by_subject=compatibility_by_subject,
    )


def estimate_study_subject_block_hierarchy_map_stan(
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
) -> StudySubjectBlockHierarchyPosteriorResult:
    """Estimate the population -> subject -> block posterior mode with Stan optimize."""

    method_name = _validate_optimize_args(
        method=method,
        max_iterations=max_iterations,
        init_alpha=init_alpha,
        refresh=refresh,
        history_size=history_size,
    )
    built, stan_code, cache_tag, compatibility_by_subject = _build_study_stan_job(
        study=study,
        model_component_id=model_component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        requirements=requirements,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params_by_subject=initial_block_params_by_subject,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
        structure=_STRUCTURE_STUDY_SUBJECT_BLOCK,
    )
    fit = _run_stan_hierarchical_optimize(
        stan_code=stan_code,
        cache_tag=cache_tag,
        stan_data=built.stan_data,
        init_data=built.init_data,
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
    return _decode_study_subject_block_hierarchy_fit(
        fit=fit,
        parameter_names=built.parameter_names,
        subject_ids=built.subject_ids,
        block_ids_by_subject=built.block_ids_by_subject,
        diagnostics=_map_diagnostics(
            method="study_subject_block_hierarchy_stan_map",
            n_kept_draws=1,
            random_seed=random_seed,
        ),
        compatibility_by_subject=compatibility_by_subject,
    )


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
    structure: str,
) -> tuple[_SubjectBuild, str, str, CompatibilityReport | None]:
    """Assemble validated Stan inputs and source code for one subject."""

    if structure not in {_STRUCTURE_SUBJECT_SHARED, _STRUCTURE_SUBJECT_BLOCK}:
        raise ValueError(
            f"subject structure must be one of "
            f"{sorted({_STRUCTURE_SUBJECT_SHARED, _STRUCTURE_SUBJECT_BLOCK})}"
        )
    compatibility = _subject_compatibility_report(subject, requirements)
    flat_build = _build_flat_subject_inputs(
        subject=subject,
        model_component_id=model_component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params=initial_block_params,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
    )
    if structure == _STRUCTURE_SUBJECT_SHARED:
        init_data = {"subject_param_z": list(flat_build.stan_data["group_loc_init"])}
    else:
        init_data = {
            "subject_loc_z": list(flat_build.stan_data["group_loc_init"]),
            "subject_log_scale": list(flat_build.stan_data["group_log_scale_init"]),
            "block_z": list(flat_build.stan_data["block_z_init"]),
        }
    built = _SubjectBuild(
        stan_data=flat_build.stan_data,
        init_data=init_data,
        parameter_names=flat_build.parameter_names,
        block_ids=flat_build.block_ids,
    )
    if model_component_id in _ASOCIAL_COMPONENT_IDS:
        return (
            built,
            _ASOCIAL_STAN_CODE_BY_STRUCTURE[structure],
            f"{structure}_{model_component_id}",
            compatibility,
        )
    return (
        built,
        load_social_stan_code(model_component_id, structure),
        social_cache_tag(model_component_id, structure),
        compatibility,
    )


def _build_study_stan_job(
    *,
    study: StudyData,
    model_component_id: str,
    model_kwargs: Mapping[str, Any] | None,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None,
    requirements: ComponentRequirements | None,
    initial_group_location: Mapping[str, float] | None,
    initial_group_scale: Mapping[str, float] | None,
    initial_block_params_by_subject: Mapping[str, Sequence[Mapping[str, float]]] | None,
    mu_prior_mean: float | Mapping[str, float],
    mu_prior_std: float | Mapping[str, float],
    log_sigma_prior_mean: float | Mapping[str, float],
    log_sigma_prior_std: float | Mapping[str, float],
    structure: str,
) -> tuple[_StudyBuild, str, str, tuple[CompatibilityReport | None, ...] | None]:
    """Assemble validated Stan inputs and source code for one study."""

    if structure not in {_STRUCTURE_STUDY_SUBJECT, _STRUCTURE_STUDY_SUBJECT_BLOCK}:
        raise ValueError(
            f"study structure must be one of "
            f"{sorted({_STRUCTURE_STUDY_SUBJECT, _STRUCTURE_STUDY_SUBJECT_BLOCK})}"
        )

    flat_blocks: list[BlockData] = []
    subject_ids: list[str] = []
    block_ids_by_subject: list[tuple[str | int | None, ...]] = []
    subject_idx_by_block: list[int] = []
    compatibility_by_subject: list[CompatibilityReport | None] | None = (
        [] if requirements is not None else None
    )
    flat_initial_block_params: list[Mapping[str, float]] | None = (
        [] if initial_block_params_by_subject is not None else None
    )

    for subject_index, subject in enumerate(study.subjects, start=1):
        subject_ids.append(subject.subject_id)
        block_ids = tuple(block.block_id for block in subject.blocks)
        block_ids_by_subject.append(block_ids)
        flat_blocks.extend(subject.blocks)
        subject_idx_by_block.extend([subject_index] * len(subject.blocks))

        if compatibility_by_subject is not None:
            compatibility_by_subject.append(_subject_compatibility_report(subject, requirements))

        if flat_initial_block_params is None:
            continue
        subject_initial = initial_block_params_by_subject.get(subject.subject_id)
        if subject_initial is None:
            flat_initial_block_params.extend({} for _ in subject.blocks)
            continue
        if len(subject_initial) != len(subject.blocks):
            raise ValueError(
                f"initial_block_params_by_subject[{subject.subject_id!r}] must match "
                f"the subject's number of blocks"
            )
        flat_initial_block_params.extend(subject_initial)

    flat_subject = SubjectData(subject_id="__study__", blocks=tuple(flat_blocks))
    flat_build = _build_flat_subject_inputs(
        subject=flat_subject,
        model_component_id=model_component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        transform_kinds=transform_kinds,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params=tuple(flat_initial_block_params) if flat_initial_block_params is not None else None,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        log_sigma_prior_mean=log_sigma_prior_mean,
        log_sigma_prior_std=log_sigma_prior_std,
    )

    stan_data = dict(flat_build.stan_data)
    stan_data["J"] = int(len(study.subjects))
    stan_data["subject_idx"] = [int(value) for value in subject_idx_by_block]
    stan_data["population_loc_init"] = list(flat_build.stan_data["group_loc_init"])
    stan_data["population_log_scale_init"] = list(flat_build.stan_data["group_log_scale_init"])

    block_z_init = np.asarray(flat_build.stan_data["block_z_init"], dtype=float)
    group_log_scale_init = np.asarray(flat_build.stan_data["group_log_scale_init"], dtype=float)
    subject_loc_init = _subject_latent_means(
        block_z_init=block_z_init,
        subject_idx_by_block=subject_idx_by_block,
        n_subjects=len(study.subjects),
    )

    if structure == _STRUCTURE_STUDY_SUBJECT:
        init_data = {
            "population_loc_z": list(stan_data["population_loc_init"]),
            "population_log_scale": list(stan_data["population_log_scale_init"]),
            "subject_z": subject_loc_init.tolist(),
        }
        stan_data["subject_z_init"] = subject_loc_init.tolist()
    else:
        subject_log_scale_init = _subject_log_scale_init(
            block_z_init=block_z_init,
            subject_idx_by_block=subject_idx_by_block,
            n_subjects=len(study.subjects),
            default_log_scale=group_log_scale_init,
        )
        init_data = {
            "population_loc_z": list(stan_data["population_loc_init"]),
            "population_log_scale": list(stan_data["population_log_scale_init"]),
            "subject_loc_z": subject_loc_init.tolist(),
            "subject_log_scale": subject_log_scale_init.tolist(),
            "block_z": list(flat_build.stan_data["block_z_init"]),
        }
        stan_data["subject_loc_init"] = subject_loc_init.tolist()
        stan_data["subject_log_scale_init"] = subject_log_scale_init.tolist()

    built = _StudyBuild(
        stan_data=stan_data,
        init_data=init_data,
        parameter_names=flat_build.parameter_names,
        subject_ids=tuple(subject_ids),
        block_ids_by_subject=tuple(block_ids_by_subject),
    )

    if model_component_id in _ASOCIAL_COMPONENT_IDS:
        return (
            built,
            _ASOCIAL_STAN_CODE_BY_STRUCTURE[structure],
            f"{structure}_{model_component_id}",
            tuple(compatibility_by_subject) if compatibility_by_subject is not None else None,
        )
    return (
        built,
        load_social_stan_code(model_component_id, structure),
        social_cache_tag(model_component_id, structure),
        tuple(compatibility_by_subject) if compatibility_by_subject is not None else None,
    )


def _build_flat_subject_inputs(
    *,
    subject: SubjectData,
    model_component_id: str,
    model_kwargs: Mapping[str, Any] | None,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None,
    initial_group_location: Mapping[str, float] | None,
    initial_group_scale: Mapping[str, float] | None,
    initial_block_params: Sequence[Mapping[str, float]] | None,
    mu_prior_mean: float | Mapping[str, float],
    mu_prior_std: float | Mapping[str, float],
    log_sigma_prior_mean: float | Mapping[str, float],
    log_sigma_prior_std: float | Mapping[str, float],
) -> _FlatBuild:
    """Build flat block arrays for either asocial or social Stan backends."""

    if model_component_id not in _SUPPORTED_COMPONENT_IDS:
        raise ValueError(
            "Stan Bayesian backend does not support "
            f"{model_component_id!r}; supported component IDs are "
            f"{sorted(_SUPPORTED_COMPONENT_IDS)}"
        )

    if model_component_id in _ASOCIAL_COMPONENT_IDS:
        return _build_asocial_subject_inputs(
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

    social_stan_data, social_param_names, _ = build_social_subject_inputs(
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
        condition_index_by_block=None,
    )
    return _FlatBuild(
        stan_data=social_stan_data,
        parameter_names=social_param_names,
        block_ids=tuple(block.block_id for block in subject.blocks),
    )


def _subject_compatibility_report(
    subject: SubjectData,
    requirements: ComponentRequirements | None,
) -> CompatibilityReport | None:
    """Validate all subject blocks against requirements and return the last report."""

    compatibility: CompatibilityReport | None = None
    if requirements is None:
        return None
    for block in subject.blocks:
        trace = get_block_trace(block)
        compatibility = check_trace_compatibility(trace, requirements)
        assert_trace_compatible(trace, requirements)
    return compatibility


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
) -> _FlatBuild:
    """Build Stan data dictionary and metadata for one asocial subject."""

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
    return _FlatBuild(
        stan_data=stan_data,
        parameter_names=names,
        block_ids=tuple(block.block_id for block in subject.blocks),
    )


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
    """Return safe default transform kind for known asocial parameters."""

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


def _subject_latent_means(
    *,
    block_z_init: np.ndarray,
    subject_idx_by_block: Sequence[int],
    n_subjects: int,
) -> np.ndarray:
    """Return mean latent block initialization per subject."""

    if block_z_init.ndim != 2:
        raise ValueError("block_z_init must be a 2D array")
    if block_z_init.shape[0] != len(subject_idx_by_block):
        raise ValueError("subject_idx_by_block must match block_z_init rows")
    out = np.zeros((n_subjects, block_z_init.shape[1]), dtype=float)
    counts = np.zeros(n_subjects, dtype=int)
    for block_index, subject_index in enumerate(subject_idx_by_block):
        zero_based = int(subject_index) - 1
        out[zero_based, :] += block_z_init[block_index, :]
        counts[zero_based] += 1
    for subject_zero_based, count in enumerate(counts):
        if count <= 0:
            raise ValueError("each subject must contribute at least one block")
        out[subject_zero_based, :] /= float(count)
    return out


def _subject_log_scale_init(
    *,
    block_z_init: np.ndarray,
    subject_idx_by_block: Sequence[int],
    n_subjects: int,
    default_log_scale: np.ndarray,
) -> np.ndarray:
    """Build subject-level log-scale initialization from block latent values."""

    if default_log_scale.ndim != 1:
        raise ValueError("default_log_scale must be a 1D vector")
    out = np.repeat(default_log_scale.reshape((1, -1)), n_subjects, axis=0)
    grouped_rows: list[list[np.ndarray]] = [[] for _ in range(n_subjects)]
    for block_index, subject_index in enumerate(subject_idx_by_block):
        grouped_rows[int(subject_index) - 1].append(block_z_init[block_index, :])
    for subject_zero_based, rows in enumerate(grouped_rows):
        if len(rows) <= 1:
            continue
        subject_matrix = np.vstack(rows)
        std = np.std(subject_matrix, axis=0)
        out[subject_zero_based, :] = np.log(np.clip(std, 1e-6, None))
    return out


def _run_stan_hierarchical_nuts(
    *,
    stan_code: str,
    cache_tag: str,
    stan_data: Mapping[str, Any],
    init_data: Mapping[str, Any] | None,
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
    """Compile and sample one Stan Bayesian model."""

    model = compile_cmdstan_model(stan_code, cache_tag=cache_tag)
    sample_kwargs: dict[str, Any] = {
        "data": {key: value for key, value in stan_data.items() if not key.endswith("_init")},
        "inits": dict(init_data or {}),
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
    init_data: Mapping[str, Any] | None,
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
    """Compile and optimize one Stan Bayesian model."""

    model = compile_cmdstan_model(stan_code, cache_tag=cache_tag)
    optimize_kwargs: dict[str, Any] = {
        "data": {key: value for key, value in stan_data.items() if not key.endswith("_init")},
        "inits": dict(init_data or {}),
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


def _decode_subject_shared_fit(
    *,
    fit: Any,
    subject_id: str,
    parameter_names: tuple[str, ...],
    block_ids: tuple[str | int | None, ...],
    diagnostics: MCMCDiagnostics,
    compatibility: CompatibilityReport | None,
) -> SubjectSharedPosteriorResult:
    """Decode CmdStan fit object into a subject-shared posterior result."""

    subject_param_z = _ensure_draw_matrix(fit.stan_variable("subject_param_z"), len(parameter_names))
    n_draws = int(subject_param_z.shape[0])
    block_z = _ensure_draw_cube(fit.stan_variable("block_z"), n_draws, len(block_ids), len(parameter_names))
    block_param = _ensure_draw_cube(
        fit.stan_variable("block_param"),
        n_draws,
        len(block_ids),
        len(parameter_names),
    )
    log_likelihood = _ensure_draw_vector(fit.stan_variable("log_likelihood_total"), n_draws)
    log_prior = _ensure_draw_vector(fit.stan_variable("log_prior_total"), n_draws)
    log_posterior = _ensure_draw_vector(fit.stan_variable("log_posterior_total"), n_draws)

    draws: list[StanPosteriorDraw] = []
    for draw_index in range(n_draws):
        subject_params = {
            name: float(block_param[draw_index, 0, param_index])
            for param_index, name in enumerate(parameter_names)
        }
        block_params_z = _param_rows_from_matrix(block_z[draw_index], parameter_names)
        block_params = _param_rows_from_matrix(block_param[draw_index], parameter_names)
        candidate = SubjectSharedPosteriorCandidate(
            parameter_names=parameter_names,
            subject_params_z={
                name: float(subject_param_z[draw_index, param_index])
                for param_index, name in enumerate(parameter_names)
            },
            subject_params=subject_params,
            block_params_z=block_params_z,
            block_params=block_params,
            log_likelihood=float(log_likelihood[draw_index]),
            log_prior=float(log_prior[draw_index]),
            log_posterior=float(log_posterior[draw_index]),
        )
        draws.append(StanPosteriorDraw(candidate=candidate, accepted=True, iteration=draw_index))

    return SubjectSharedPosteriorResult(
        subject_id=subject_id,
        block_ids=block_ids,
        parameter_names=parameter_names,
        draws=tuple(draws),
        diagnostics=diagnostics,
        compatibility=compatibility,
    )


def _decode_subject_block_hierarchy_fit(
    *,
    fit: Any,
    subject_id: str,
    parameter_names: tuple[str, ...],
    block_ids: tuple[str | int | None, ...],
    diagnostics: MCMCDiagnostics,
    compatibility: CompatibilityReport | None,
) -> SubjectBlockHierarchyPosteriorResult:
    """Decode CmdStan fit object into a subject -> block posterior result."""

    subject_loc_z = _ensure_draw_matrix(fit.stan_variable("subject_loc_z"), len(parameter_names))
    n_draws = int(subject_loc_z.shape[0])
    subject_log_scale = _ensure_draw_matrix(fit.stan_variable("subject_log_scale"), len(parameter_names))
    block_z = _ensure_draw_cube(fit.stan_variable("block_z"), n_draws, len(block_ids), len(parameter_names))
    block_param = _ensure_draw_cube(
        fit.stan_variable("block_param"),
        n_draws,
        len(block_ids),
        len(parameter_names),
    )
    log_likelihood = _ensure_draw_vector(fit.stan_variable("log_likelihood_total"), n_draws)
    log_prior = _ensure_draw_vector(fit.stan_variable("log_prior_total"), n_draws)
    log_posterior = _ensure_draw_vector(fit.stan_variable("log_posterior_total"), n_draws)

    draws: list[StanPosteriorDraw] = []
    for draw_index in range(n_draws):
        candidate = SubjectBlockHierarchyPosteriorCandidate(
            parameter_names=parameter_names,
            subject_location_z={
                name: float(subject_loc_z[draw_index, param_index])
                for param_index, name in enumerate(parameter_names)
            },
            subject_scale={
                name: float(np.exp(subject_log_scale[draw_index, param_index]))
                for param_index, name in enumerate(parameter_names)
            },
            block_params_z=_param_rows_from_matrix(block_z[draw_index], parameter_names),
            block_params=_param_rows_from_matrix(block_param[draw_index], parameter_names),
            log_likelihood=float(log_likelihood[draw_index]),
            log_prior=float(log_prior[draw_index]),
            log_posterior=float(log_posterior[draw_index]),
        )
        draws.append(StanPosteriorDraw(candidate=candidate, accepted=True, iteration=draw_index))

    return SubjectBlockHierarchyPosteriorResult(
        subject_id=subject_id,
        block_ids=block_ids,
        parameter_names=parameter_names,
        draws=tuple(draws),
        diagnostics=diagnostics,
        compatibility=compatibility,
    )


def _decode_study_subject_hierarchy_fit(
    *,
    fit: Any,
    parameter_names: tuple[str, ...],
    subject_ids: tuple[str, ...],
    block_ids_by_subject: tuple[tuple[str | int | None, ...], ...],
    diagnostics: MCMCDiagnostics,
    compatibility_by_subject: tuple[CompatibilityReport | None, ...] | None,
) -> StudySubjectHierarchyPosteriorResult:
    """Decode CmdStan fit object into a population -> subject posterior result."""

    n_subjects = len(subject_ids)
    n_blocks = sum(len(block_ids) for block_ids in block_ids_by_subject)
    population_loc_z = _ensure_draw_matrix(fit.stan_variable("population_loc_z"), len(parameter_names))
    n_draws = int(population_loc_z.shape[0])
    population_log_scale = _ensure_draw_matrix(
        fit.stan_variable("population_log_scale"),
        len(parameter_names),
    )
    subject_z = _ensure_draw_cube(fit.stan_variable("subject_z"), n_draws, n_subjects, len(parameter_names))
    subject_param = _ensure_draw_cube(
        fit.stan_variable("subject_param"),
        n_draws,
        n_subjects,
        len(parameter_names),
    )
    block_z = _ensure_draw_cube(fit.stan_variable("block_z"), n_draws, n_blocks, len(parameter_names))
    block_param = _ensure_draw_cube(fit.stan_variable("block_param"), n_draws, n_blocks, len(parameter_names))
    log_likelihood = _ensure_draw_vector(fit.stan_variable("log_likelihood_total"), n_draws)
    log_prior = _ensure_draw_vector(fit.stan_variable("log_prior_total"), n_draws)
    log_posterior = _ensure_draw_vector(fit.stan_variable("log_posterior_total"), n_draws)

    block_counts = tuple(len(block_ids) for block_ids in block_ids_by_subject)
    draws: list[StanPosteriorDraw] = []
    for draw_index in range(n_draws):
        candidate = StudySubjectHierarchyPosteriorCandidate(
            parameter_names=parameter_names,
            population_location_z={
                name: float(population_loc_z[draw_index, param_index])
                for param_index, name in enumerate(parameter_names)
            },
            population_scale={
                name: float(np.exp(population_log_scale[draw_index, param_index]))
                for param_index, name in enumerate(parameter_names)
            },
            subject_params_z=_param_rows_from_matrix(subject_z[draw_index], parameter_names),
            subject_params=_param_rows_from_matrix(subject_param[draw_index], parameter_names),
            block_params_z_by_subject=_nest_param_rows(
                block_z[draw_index],
                parameter_names=parameter_names,
                counts=block_counts,
            ),
            block_params_by_subject=_nest_param_rows(
                block_param[draw_index],
                parameter_names=parameter_names,
                counts=block_counts,
            ),
            log_likelihood=float(log_likelihood[draw_index]),
            log_prior=float(log_prior[draw_index]),
            log_posterior=float(log_posterior[draw_index]),
        )
        draws.append(StanPosteriorDraw(candidate=candidate, accepted=True, iteration=draw_index))

    return StudySubjectHierarchyPosteriorResult(
        subject_ids=subject_ids,
        block_ids_by_subject=block_ids_by_subject,
        parameter_names=parameter_names,
        draws=tuple(draws),
        diagnostics=diagnostics,
        compatibility_by_subject=compatibility_by_subject,
    )


def _decode_study_subject_block_hierarchy_fit(
    *,
    fit: Any,
    parameter_names: tuple[str, ...],
    subject_ids: tuple[str, ...],
    block_ids_by_subject: tuple[tuple[str | int | None, ...], ...],
    diagnostics: MCMCDiagnostics,
    compatibility_by_subject: tuple[CompatibilityReport | None, ...] | None,
) -> StudySubjectBlockHierarchyPosteriorResult:
    """Decode CmdStan fit object into a population -> subject -> block result."""

    n_subjects = len(subject_ids)
    n_blocks = sum(len(block_ids) for block_ids in block_ids_by_subject)
    population_loc_z = _ensure_draw_matrix(fit.stan_variable("population_loc_z"), len(parameter_names))
    n_draws = int(population_loc_z.shape[0])
    population_log_scale = _ensure_draw_matrix(
        fit.stan_variable("population_log_scale"),
        len(parameter_names),
    )
    subject_loc_z = _ensure_draw_cube(
        fit.stan_variable("subject_loc_z"),
        n_draws,
        n_subjects,
        len(parameter_names),
    )
    subject_log_scale = _ensure_draw_cube(
        fit.stan_variable("subject_log_scale"),
        n_draws,
        n_subjects,
        len(parameter_names),
    )
    subject_param = _ensure_draw_cube(
        fit.stan_variable("subject_param"),
        n_draws,
        n_subjects,
        len(parameter_names),
    )
    block_z = _ensure_draw_cube(fit.stan_variable("block_z"), n_draws, n_blocks, len(parameter_names))
    block_param = _ensure_draw_cube(fit.stan_variable("block_param"), n_draws, n_blocks, len(parameter_names))
    log_likelihood = _ensure_draw_vector(fit.stan_variable("log_likelihood_total"), n_draws)
    log_prior = _ensure_draw_vector(fit.stan_variable("log_prior_total"), n_draws)
    log_posterior = _ensure_draw_vector(fit.stan_variable("log_posterior_total"), n_draws)

    block_counts = tuple(len(block_ids) for block_ids in block_ids_by_subject)
    draws: list[StanPosteriorDraw] = []
    for draw_index in range(n_draws):
        candidate = StudySubjectBlockHierarchyPosteriorCandidate(
            parameter_names=parameter_names,
            population_location_z={
                name: float(population_loc_z[draw_index, param_index])
                for param_index, name in enumerate(parameter_names)
            },
            population_scale={
                name: float(np.exp(population_log_scale[draw_index, param_index]))
                for param_index, name in enumerate(parameter_names)
            },
            subject_location_z=_param_rows_from_matrix(subject_loc_z[draw_index], parameter_names),
            subject_scale=tuple(
                {
                    name: float(np.exp(subject_log_scale[draw_index, subject_index, param_index]))
                    for param_index, name in enumerate(parameter_names)
                }
                for subject_index in range(n_subjects)
            ),
            subject_params=_param_rows_from_matrix(subject_param[draw_index], parameter_names),
            block_params_z_by_subject=_nest_param_rows(
                block_z[draw_index],
                parameter_names=parameter_names,
                counts=block_counts,
            ),
            block_params_by_subject=_nest_param_rows(
                block_param[draw_index],
                parameter_names=parameter_names,
                counts=block_counts,
            ),
            log_likelihood=float(log_likelihood[draw_index]),
            log_prior=float(log_prior[draw_index]),
            log_posterior=float(log_posterior[draw_index]),
        )
        draws.append(StanPosteriorDraw(candidate=candidate, accepted=True, iteration=draw_index))

    return StudySubjectBlockHierarchyPosteriorResult(
        subject_ids=subject_ids,
        block_ids_by_subject=block_ids_by_subject,
        parameter_names=parameter_names,
        draws=tuple(draws),
        diagnostics=diagnostics,
        compatibility_by_subject=compatibility_by_subject,
    )


def _param_rows_from_matrix(
    matrix: np.ndarray,
    parameter_names: tuple[str, ...],
) -> tuple[dict[str, float], ...]:
    """Convert a 2D parameter matrix into tuple[dict]."""

    return tuple(
        {
            name: float(matrix[row_index, param_index])
            for param_index, name in enumerate(parameter_names)
        }
        for row_index in range(matrix.shape[0])
    )


def _nest_param_rows(
    matrix: np.ndarray,
    *,
    parameter_names: tuple[str, ...],
    counts: tuple[int, ...],
) -> tuple[tuple[dict[str, float], ...], ...]:
    """Nest flat block parameter rows according to per-subject block counts."""

    rows = _param_rows_from_matrix(matrix, parameter_names)
    out: list[tuple[dict[str, float], ...]] = []
    cursor = 0
    for count in counts:
        out.append(tuple(rows[cursor : cursor + count]))
        cursor += count
    return tuple(out)


def _ensure_draw_vector(values: Any, n_draws: int) -> np.ndarray:
    """Coerce one Stan scalar draw variable into ``(n_draws,)``."""

    out = np.asarray(values, dtype=float).reshape((n_draws,))
    if out.shape[0] == 0:
        raise ValueError("Stan fit returned zero posterior draws")
    return out


def _ensure_draw_matrix(values: Any, n_cols: int) -> np.ndarray:
    """Coerce one Stan vector draw variable into ``(n_draws, n_cols)``."""

    out = np.asarray(values, dtype=float)
    if out.ndim == 1:
        out = out.reshape((1, n_cols))
    else:
        out = out.reshape((-1, n_cols))
    if out.shape[0] == 0:
        raise ValueError("Stan fit returned zero posterior draws")
    return out


def _ensure_draw_cube(values: Any, n_draws: int, n_rows: int, n_cols: int) -> np.ndarray:
    """Coerce one Stan array draw variable into ``(n_draws, n_rows, n_cols)``."""

    out = np.asarray(values, dtype=float).reshape((n_draws, n_rows, n_cols))
    return out


def _nuts_diagnostics(
    *,
    method: str,
    n_samples: int,
    n_warmup: int,
    thin: int,
    n_chains: int,
    random_seed: int | None,
) -> MCMCDiagnostics:
    """Build diagnostics metadata for a Stan NUTS run."""

    n_kept_draws = int(n_samples * n_chains)
    n_iterations = int((n_warmup + (n_samples * thin)) * n_chains)
    return MCMCDiagnostics(
        method=method,
        n_iterations=n_iterations,
        n_warmup=int(n_warmup * n_chains),
        n_kept_draws=n_kept_draws,
        thin=int(thin),
        n_accepted=n_iterations,
        acceptance_rate=1.0,
        random_seed=random_seed,
    )


def _map_diagnostics(
    *,
    method: str,
    n_kept_draws: int,
    random_seed: int | None,
) -> MCMCDiagnostics:
    """Build diagnostics metadata for a Stan optimize run."""

    return MCMCDiagnostics(
        method=method,
        n_iterations=1,
        n_warmup=0,
        n_kept_draws=int(n_kept_draws),
        thin=1,
        n_accepted=1,
        acceptance_rate=1.0,
        random_seed=random_seed,
    )


def _validate_nuts_args(
    *,
    n_samples: int,
    n_warmup: int,
    thin: int,
    n_chains: int,
    parallel_chains: int | None,
    adapt_delta: float,
    max_treedepth: int,
) -> None:
    """Validate common Stan NUTS arguments."""

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


def _validate_optimize_args(
    *,
    method: str,
    max_iterations: int,
    init_alpha: float | None,
    refresh: int,
    history_size: int | None,
) -> str:
    """Validate common Stan optimize arguments and return normalized method."""

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
    return method_name


__all__ = [
    "draw_study_subject_block_hierarchy_posterior_stan",
    "draw_study_subject_hierarchy_posterior_stan",
    "draw_subject_block_hierarchy_posterior_stan",
    "draw_subject_shared_posterior_stan",
    "estimate_study_subject_block_hierarchy_map_stan",
    "estimate_study_subject_hierarchy_map_stan",
    "estimate_subject_block_hierarchy_map_stan",
    "estimate_subject_shared_map_stan",
]
