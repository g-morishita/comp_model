"""Within-subject hierarchical Bayesian posterior sampling (MCMC).

This module extends hierarchical inference beyond MAP by sampling posterior
draws for block-level parameters under a within-subject hierarchical prior.
It reuses the same canonical replay likelihood semantics as all other
estimators in ``comp_model``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from comp_model.core.contracts import AgentModel
from comp_model.core.data import StudyData, SubjectData, get_block_trace
from comp_model.core.requirements import ComponentRequirements

from .compatibility import CompatibilityReport, assert_trace_compatible, check_trace_compatibility
from .hierarchical import (
    _decode_parameter_vector,
    _hierarchical_log_prior,
    _total_log_likelihood,
)
from .likelihood import ActionReplayLikelihood, LikelihoodProgram
from .mcmc import MCMCDiagnostics
from .transforms import ParameterTransform, identity_transform


@dataclass(frozen=True, slots=True)
class HierarchicalPosteriorCandidate:
    """One evaluated hierarchical posterior candidate.

    Parameters
    ----------
    parameter_names : tuple[str, ...]
        Hierarchical parameter names.
    group_location_z : dict[str, float]
        Group-level location parameters in unconstrained ``z`` space.
    group_scale_z : dict[str, float]
        Positive group-level scale parameters in ``z`` space.
    block_params_z : tuple[dict[str, float], ...]
        Per-block unconstrained parameter mappings.
    block_params : tuple[dict[str, float], ...]
        Per-block constrained parameter mappings.
    log_likelihood : float
        Total block log-likelihood under this candidate.
    log_prior : float
        Hierarchical log-prior under this candidate.
    log_posterior : float
        Total log-posterior ``log_likelihood + log_prior``.
    """

    parameter_names: tuple[str, ...]
    group_location_z: dict[str, float]
    group_scale_z: dict[str, float]
    block_params_z: tuple[dict[str, float], ...]
    block_params: tuple[dict[str, float], ...]
    log_likelihood: float
    log_prior: float
    log_posterior: float


@dataclass(frozen=True, slots=True)
class HierarchicalMCMCDraw:
    """One retained hierarchical posterior draw.

    Parameters
    ----------
    candidate : HierarchicalPosteriorCandidate
        Candidate at this retained draw.
    accepted : bool
        Whether the proposal at this iteration was accepted.
    iteration : int
        Zero-based iteration index in the full chain (including warmup).
    """

    candidate: HierarchicalPosteriorCandidate
    accepted: bool
    iteration: int


@dataclass(frozen=True, slots=True)
class HierarchicalSubjectPosteriorResult:
    """Within-subject hierarchical posterior sampling output.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    parameter_names : tuple[str, ...]
        Hierarchical parameter names.
    draws : tuple[HierarchicalMCMCDraw, ...]
        Retained post-warmup draws.
    diagnostics : MCMCDiagnostics
        MCMC run diagnostics.
    compatibility : CompatibilityReport | None, optional
        Compatibility report when requirements were checked.
    """

    subject_id: str
    parameter_names: tuple[str, ...]
    draws: tuple[HierarchicalMCMCDraw, ...]
    diagnostics: MCMCDiagnostics
    compatibility: CompatibilityReport | None = None

    @property
    def map_candidate(self) -> HierarchicalPosteriorCandidate:
        """Return highest-posterior retained candidate."""

        return max(self.draws, key=lambda draw: draw.candidate.log_posterior).candidate

    @property
    def n_blocks(self) -> int:
        """Return number of blocks represented in each draw."""

        if not self.draws:
            return 0
        return len(self.draws[0].candidate.block_params)


@dataclass(frozen=True, slots=True)
class HierarchicalStudyPosteriorResult:
    """Study-level hierarchical posterior sampling output.

    Parameters
    ----------
    subject_results : tuple[HierarchicalSubjectPosteriorResult, ...]
        Per-subject posterior sampling outputs.
    """

    subject_results: tuple[HierarchicalSubjectPosteriorResult, ...]

    @property
    def n_subjects(self) -> int:
        """Return number of sampled subjects."""

        return len(self.subject_results)

    @property
    def total_map_log_likelihood(self) -> float:
        """Return sum of subject-level MAP draw log-likelihood values."""

        return float(
            sum(item.map_candidate.log_likelihood for item in self.subject_results)
        )

    @property
    def total_map_log_prior(self) -> float:
        """Return sum of subject-level MAP draw log-prior values."""

        return float(
            sum(item.map_candidate.log_prior for item in self.subject_results)
        )

    @property
    def total_map_log_posterior(self) -> float:
        """Return sum of subject-level MAP draw log-posterior values."""

        return float(
            sum(item.map_candidate.log_posterior for item in self.subject_results)
        )


@dataclass(frozen=True, slots=True)
class _EvaluatedHierarchicalState:
    """Internal evaluated state for hierarchical random-walk proposals."""

    candidate: HierarchicalPosteriorCandidate


def sample_subject_hierarchical_posterior(
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
    n_samples: int = 1000,
    n_warmup: int = 500,
    thin: int = 1,
    proposal_scale_group_location: float = 0.08,
    proposal_scale_group_log_scale: float = 0.05,
    proposal_scale_block_z: float = 0.08,
    random_seed: int | None = None,
) -> HierarchicalSubjectPosteriorResult:
    """Sample within-subject hierarchical posterior draws.

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
    n_samples : int, optional
        Number of retained draws after warmup/thinning.
    n_warmup : int, optional
        Number of warmup iterations.
    thin : int, optional
        Thinning interval for retained draws.
    proposal_scale_group_location : float, optional
        Proposal scale for group-location coordinates.
    proposal_scale_group_log_scale : float, optional
        Proposal scale for group-log-scale coordinates.
    proposal_scale_block_z : float, optional
        Proposal scale for per-block unconstrained parameters.
    random_seed : int | None, optional
        Optional RNG seed for deterministic sampling.

    Returns
    -------
    HierarchicalSubjectPosteriorResult
        Subject-level hierarchical posterior sampling output.

    Raises
    ------
    ValueError
        If inputs are invalid or the initial state has non-finite posterior.
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
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_warmup < 0:
        raise ValueError("n_warmup must be >= 0")
    if thin <= 0:
        raise ValueError("thin must be > 0")
    if proposal_scale_group_location <= 0.0:
        raise ValueError("proposal_scale_group_location must be > 0")
    if proposal_scale_group_log_scale <= 0.0:
        raise ValueError("proposal_scale_group_log_scale must be > 0")
    if proposal_scale_block_z <= 0.0:
        raise ValueError("proposal_scale_block_z must be > 0")

    like = likelihood_program if likelihood_program is not None else ActionReplayLikelihood()
    traces = tuple(get_block_trace(block) for block in subject.blocks)

    compatibility: CompatibilityReport | None = None
    if requirements is not None:
        for trace in traces:
            compatibility = check_trace_compatibility(trace, requirements)
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
    current_vector = np.asarray(x0_parts, dtype=float)
    proposal_scales = _build_proposal_scale_vector(
        names=names,
        n_blocks=n_blocks,
        proposal_scale_group_location=float(proposal_scale_group_location),
        proposal_scale_group_log_scale=float(proposal_scale_group_log_scale),
        proposal_scale_block_z=float(proposal_scale_block_z),
    )

    current = _evaluate_hierarchical_state(
        vector=current_vector,
        names=names,
        n_blocks=n_blocks,
        transforms=per_param_transform,
        traces=traces,
        model_factory=model_factory,
        likelihood_program=like,
        mu_prior_mean=float(mu_prior_mean),
        mu_prior_std=float(mu_prior_std),
        log_sigma_prior_mean=float(log_sigma_prior_mean),
        log_sigma_prior_std=float(log_sigma_prior_std),
    )
    if not np.isfinite(current.candidate.log_posterior):
        raise ValueError("initial hierarchical state has non-finite log posterior")

    rng = np.random.default_rng(random_seed)
    n_iterations = int(n_warmup + n_samples * thin)
    accepted_total = 0
    retained: list[HierarchicalMCMCDraw] = []

    for iteration in range(n_iterations):
        proposal_vector = current_vector + rng.normal(0.0, proposal_scales, size=current_vector.shape)
        proposal = _evaluate_hierarchical_state(
            vector=proposal_vector,
            names=names,
            n_blocks=n_blocks,
            transforms=per_param_transform,
            traces=traces,
            model_factory=model_factory,
            likelihood_program=like,
            mu_prior_mean=float(mu_prior_mean),
            mu_prior_std=float(mu_prior_std),
            log_sigma_prior_mean=float(log_sigma_prior_mean),
            log_sigma_prior_std=float(log_sigma_prior_std),
        )
        accepted = _metropolis_accept(
            current_log_posterior=current.candidate.log_posterior,
            proposal_log_posterior=proposal.candidate.log_posterior,
            rng=rng,
        )
        if accepted:
            current = proposal
            current_vector = proposal_vector
            accepted_total += 1

        keep_iteration = (
            iteration >= n_warmup and ((iteration - n_warmup) % thin == 0)
        )
        if keep_iteration:
            retained.append(
                HierarchicalMCMCDraw(
                    candidate=current.candidate,
                    accepted=accepted,
                    iteration=int(iteration),
                )
            )

    draws = tuple(retained)
    diagnostics = MCMCDiagnostics(
        method="within_subject_hierarchical_random_walk_metropolis",
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        n_kept_draws=len(draws),
        thin=thin,
        n_accepted=accepted_total,
        acceptance_rate=float(accepted_total / n_iterations),
        random_seed=random_seed,
    )

    return HierarchicalSubjectPosteriorResult(
        subject_id=subject.subject_id,
        parameter_names=names,
        draws=draws,
        diagnostics=diagnostics,
        compatibility=compatibility,
    )


def sample_study_hierarchical_posterior(
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
    n_samples: int = 1000,
    n_warmup: int = 500,
    thin: int = 1,
    proposal_scale_group_location: float = 0.08,
    proposal_scale_group_log_scale: float = 0.05,
    proposal_scale_block_z: float = 0.08,
    random_seed: int | None = None,
) -> HierarchicalStudyPosteriorResult:
    """Sample within-subject hierarchical posterior draws for all subjects.

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
    n_samples : int, optional
        Number of retained draws after warmup/thinning.
    n_warmup : int, optional
        Number of warmup iterations.
    thin : int, optional
        Thinning interval for retained draws.
    proposal_scale_group_location : float, optional
        Proposal scale for group-location coordinates.
    proposal_scale_group_log_scale : float, optional
        Proposal scale for group-log-scale coordinates.
    proposal_scale_block_z : float, optional
        Proposal scale for per-block unconstrained parameters.
    random_seed : int | None, optional
        Optional RNG seed. Subject chains use deterministic offsets.

    Returns
    -------
    HierarchicalStudyPosteriorResult
        Aggregated study-level posterior sampling output.
    """

    subject_results: list[HierarchicalSubjectPosteriorResult] = []
    for index, subject in enumerate(study.subjects):
        subject_initial_block_params = None
        if initial_block_params_by_subject is not None:
            raw_subject_params = initial_block_params_by_subject.get(subject.subject_id)
            if raw_subject_params is not None:
                subject_initial_block_params = tuple(dict(item) for item in raw_subject_params)

        subject_results.append(
            sample_subject_hierarchical_posterior(
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
                n_samples=n_samples,
                n_warmup=n_warmup,
                thin=thin,
                proposal_scale_group_location=proposal_scale_group_location,
                proposal_scale_group_log_scale=proposal_scale_group_log_scale,
                proposal_scale_block_z=proposal_scale_block_z,
                random_seed=(
                    None if random_seed is None else int(random_seed) + (index * 1000)
                ),
            )
        )

    return HierarchicalStudyPosteriorResult(subject_results=tuple(subject_results))


def _evaluate_hierarchical_state(
    *,
    vector: np.ndarray,
    names: tuple[str, ...],
    n_blocks: int,
    transforms: Mapping[str, ParameterTransform],
    traces: Sequence[Any],
    model_factory: Callable[[dict[str, float]], AgentModel],
    likelihood_program: LikelihoodProgram,
    mu_prior_mean: float,
    mu_prior_std: float,
    log_sigma_prior_mean: float,
    log_sigma_prior_std: float,
) -> _EvaluatedHierarchicalState:
    """Evaluate one hierarchical parameter vector."""

    decoded = _decode_parameter_vector(
        vector=vector,
        names=names,
        n_blocks=n_blocks,
        transforms=transforms,
    )
    log_likelihood = _total_log_likelihood(
        traces=traces,
        model_factory=model_factory,
        likelihood_program=likelihood_program,
        block_params=decoded.block_params,
    )
    log_prior = _hierarchical_log_prior(
        group_location=decoded.group_location_z,
        group_log_scale=decoded.group_log_scale_z,
        block_z=decoded.block_params_z,
        mu_prior_mean=float(mu_prior_mean),
        mu_prior_std=float(mu_prior_std),
        log_sigma_prior_mean=float(log_sigma_prior_mean),
        log_sigma_prior_std=float(log_sigma_prior_std),
    )
    log_posterior = float(log_likelihood + log_prior)
    candidate = HierarchicalPosteriorCandidate(
        parameter_names=names,
        group_location_z=dict(decoded.group_location_z),
        group_scale_z={
            name: float(np.exp(decoded.group_log_scale_z[name]))
            for name in names
        },
        block_params_z=tuple(dict(item) for item in decoded.block_params_z),
        block_params=tuple(dict(item) for item in decoded.block_params),
        log_likelihood=float(log_likelihood),
        log_prior=float(log_prior),
        log_posterior=float(log_posterior),
    )
    return _EvaluatedHierarchicalState(candidate=candidate)


def _build_proposal_scale_vector(
    *,
    names: tuple[str, ...],
    n_blocks: int,
    proposal_scale_group_location: float,
    proposal_scale_group_log_scale: float,
    proposal_scale_block_z: float,
) -> np.ndarray:
    """Construct proposal-scale vector aligned to hierarchical parameter vector."""

    scales: list[float] = []
    scales.extend([float(proposal_scale_group_location)] * len(names))
    scales.extend([float(proposal_scale_group_log_scale)] * len(names))
    scales.extend([float(proposal_scale_block_z)] * (n_blocks * len(names)))
    return np.asarray(scales, dtype=float)


def _metropolis_accept(
    *,
    current_log_posterior: float,
    proposal_log_posterior: float,
    rng: np.random.Generator,
) -> bool:
    """Return whether to accept a proposal under Metropolis criterion."""

    if not np.isfinite(proposal_log_posterior):
        return False
    if proposal_log_posterior >= current_log_posterior:
        return True
    log_u = float(np.log(rng.random()))
    return log_u < float(proposal_log_posterior - current_log_posterior)


__all__ = [
    "HierarchicalMCMCDraw",
    "HierarchicalPosteriorCandidate",
    "HierarchicalStudyPosteriorResult",
    "HierarchicalSubjectPosteriorResult",
    "sample_study_hierarchical_posterior",
    "sample_subject_hierarchical_posterior",
]
