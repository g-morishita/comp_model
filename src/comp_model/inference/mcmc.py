"""Posterior sampling utilities and a baseline MCMC estimator.

This module provides a generic posterior-sampling API built on canonical replay
likelihood semantics. It currently includes a random-walk Metropolis sampler as
the baseline implementation for posterior draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from comp_model.core.contracts import AgentModel
from comp_model.core.data import BlockData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import PluginRegistry, build_default_registry

from .bayes import PosteriorCandidate, PriorProgram
from .compatibility import CompatibilityReport, assert_trace_compatible, check_trace_compatibility
from .fitting import coerce_episode_trace
from .likelihood import ActionReplayLikelihood, LikelihoodProgram
from .posterior import PosteriorSamples


@dataclass(frozen=True, slots=True)
class MCMCDraw:
    """One retained MCMC draw.

    Parameters
    ----------
    candidate : PosteriorCandidate
        Posterior candidate state at this retained draw.
    accepted : bool
        Whether the proposal at this iteration was accepted.
    iteration : int
        Zero-based iteration index in the full chain (including warmup).
    """

    candidate: PosteriorCandidate
    accepted: bool
    iteration: int


@dataclass(frozen=True, slots=True)
class MCMCDiagnostics:
    """Diagnostics for one MCMC run.

    Parameters
    ----------
    method : str
        Sampler method identifier.
    n_iterations : int
        Total number of sampler iterations including warmup.
    n_warmup : int
        Number of discarded warmup iterations.
    n_kept_draws : int
        Number of retained post-warmup draws after thinning.
    thin : int
        Thinning interval.
    n_accepted : int
        Number of accepted proposals over all iterations.
    acceptance_rate : float
        Proposal acceptance rate over all iterations.
    random_seed : int | None
        Optional RNG seed used for sampling.
    """

    method: str
    n_iterations: int
    n_warmup: int
    n_kept_draws: int
    thin: int
    n_accepted: int
    acceptance_rate: float
    random_seed: int | None


@dataclass(frozen=True, slots=True)
class MCMCPosteriorResult:
    """Posterior sampling result container.

    Parameters
    ----------
    draws : tuple[MCMCDraw, ...]
        Retained post-warmup draws.
    posterior_samples : PosteriorSamples
        Parameter-wise posterior draws derived from ``draws``.
    diagnostics : MCMCDiagnostics
        Sampler diagnostics.
    pointwise_log_likelihood_draws : numpy.ndarray
        Pointwise log-likelihood draws with shape ``(n_draws, n_observations)``
        aligned with retained ``draws``.
    compatibility : CompatibilityReport | None, optional
        Compatibility report when requirements were checked.
    """

    draws: tuple[MCMCDraw, ...]
    posterior_samples: PosteriorSamples
    diagnostics: MCMCDiagnostics
    pointwise_log_likelihood_draws: np.ndarray
    compatibility: CompatibilityReport | None = None

    @property
    def map_candidate(self) -> PosteriorCandidate:
        """Return highest-posterior retained draw candidate."""

        return max(self.draws, key=lambda draw: draw.candidate.log_posterior).candidate


@dataclass(frozen=True, slots=True)
class _EvaluatedState:
    """Internal evaluated parameter state with pointwise likelihood values."""

    candidate: PosteriorCandidate
    pointwise_log_likelihood: tuple[float, ...]


def posterior_samples_from_draws(draws: Sequence[MCMCDraw]) -> PosteriorSamples:
    """Build :class:`PosteriorSamples` from retained MCMC draws.

    Parameters
    ----------
    draws : Sequence[MCMCDraw]
        Retained draws in chain order.

    Returns
    -------
    PosteriorSamples
        Parameter draw arrays by parameter name.

    Raises
    ------
    ValueError
        If ``draws`` is empty or parameter keys are inconsistent.
    """

    if not draws:
        raise ValueError("draws must not be empty")

    reference_names = tuple(sorted(draws[0].candidate.params))
    if not reference_names:
        raise ValueError("draw candidates must contain at least one parameter")

    parameter_draws: dict[str, np.ndarray] = {}
    for name in reference_names:
        values: list[float] = []
        for draw in draws:
            current_names = tuple(sorted(draw.candidate.params))
            if current_names != reference_names:
                raise ValueError("all draw candidates must share identical parameter keys")
            values.append(float(draw.candidate.params[name]))
        parameter_draws[name] = np.asarray(values, dtype=float)

    return PosteriorSamples(parameter_draws=parameter_draws)


class RandomWalkMetropolisEstimator:
    """Random-walk Metropolis posterior sampler.

    Parameters
    ----------
    likelihood_program : LikelihoodProgram
        Likelihood evaluator used for each proposed parameter state.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating a fresh model instance from parameter mappings.
    prior_program : PriorProgram
        Prior evaluator.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before sampling.
    default_proposal_scale : float, optional
        Default Gaussian proposal standard deviation for unspecified parameters.

    Notes
    -----
    Proposals are generated in constrained parameter space. Optional bounds are
    enforced as hard constraints: out-of-bounds proposals are rejected.
    """

    def __init__(
        self,
        *,
        likelihood_program: LikelihoodProgram,
        model_factory: Callable[[dict[str, float]], AgentModel],
        prior_program: PriorProgram,
        requirements: ComponentRequirements | None = None,
        default_proposal_scale: float = 0.1,
    ) -> None:
        if default_proposal_scale <= 0.0:
            raise ValueError("default_proposal_scale must be > 0")

        self._likelihood_program = likelihood_program
        self._model_factory = model_factory
        self._prior_program = prior_program
        self._requirements = requirements
        self._default_proposal_scale = float(default_proposal_scale)

    def fit(
        self,
        trace: EpisodeTrace,
        *,
        initial_params: Mapping[str, float],
        n_samples: int,
        n_warmup: int = 500,
        thin: int = 1,
        proposal_scales: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        random_seed: int | None = None,
    ) -> MCMCPosteriorResult:
        """Sample posterior draws from one canonical trace.

        Parameters
        ----------
        trace : EpisodeTrace
            Observed event trace used for posterior sampling.
        initial_params : Mapping[str, float]
            Initial constrained parameter values.
        n_samples : int
            Number of retained posterior draws after warmup/thinning.
        n_warmup : int, optional
            Number of warmup iterations.
        thin : int, optional
            Thinning interval for retained draws.
        proposal_scales : Mapping[str, float] | None, optional
            Per-parameter Gaussian proposal scales.
        bounds : Mapping[str, tuple[float | None, float | None]] | None, optional
            Optional hard bounds by parameter name.
        random_seed : int | None, optional
            Optional RNG seed for deterministic sampling.

        Returns
        -------
        MCMCPosteriorResult
            Retained draws, posterior samples, and diagnostics.

        Raises
        ------
        ValueError
            If settings are invalid, compatibility fails, or initial state has
            non-finite posterior density.
        """

        compatibility: CompatibilityReport | None = None
        if self._requirements is not None:
            compatibility = check_trace_compatibility(trace, self._requirements)
            assert_trace_compatible(trace, self._requirements)

        names = tuple(sorted(initial_params))
        if not names:
            raise ValueError("initial_params must include at least one parameter")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if n_warmup < 0:
            raise ValueError("n_warmup must be >= 0")
        if thin <= 0:
            raise ValueError("thin must be > 0")

        rng = np.random.default_rng(random_seed)
        scales = _resolve_proposal_scales(
            names=names,
            proposal_scales=proposal_scales,
            default_scale=self._default_proposal_scale,
        )
        normalized_bounds = _normalize_bounds(names, bounds)

        current = self._evaluate_state(trace, dict(initial_params))
        if not np.isfinite(current.candidate.log_posterior):
            raise ValueError("initial_params produce non-finite log posterior")

        n_iterations = int(n_warmup + n_samples * thin)
        accepted_total = 0
        retained: list[MCMCDraw] = []
        retained_pointwise: list[np.ndarray] = []

        for iteration in range(n_iterations):
            proposal_params = _propose(
                current.candidate.params,
                names=names,
                scales=scales,
                rng=rng,
            )
            proposal = (
                self._evaluate_state(trace, proposal_params)
                if _within_bounds(proposal_params, normalized_bounds)
                else _EvaluatedState(
                    candidate=PosteriorCandidate(
                        params=proposal_params,
                        log_likelihood=float("-inf"),
                        log_prior=float("-inf"),
                        log_posterior=float("-inf"),
                    ),
                    pointwise_log_likelihood=tuple(
                        float("-inf") for _ in current.pointwise_log_likelihood
                    ),
                )
            )

            accepted = _metropolis_accept(
                current_log_posterior=current.candidate.log_posterior,
                proposal_log_posterior=proposal.candidate.log_posterior,
                rng=rng,
            )
            if accepted:
                current = proposal
                accepted_total += 1

            keep_iteration = (
                iteration >= n_warmup and ((iteration - n_warmup) % thin == 0)
            )
            if keep_iteration:
                retained.append(
                    MCMCDraw(
                        candidate=current.candidate,
                        accepted=accepted,
                        iteration=iteration,
                    )
                )
                retained_pointwise.append(
                    np.asarray(current.pointwise_log_likelihood, dtype=float)
                )

        draws = tuple(retained)
        posterior_samples = posterior_samples_from_draws(draws)
        pointwise_log_likelihood_draws = np.vstack(retained_pointwise)
        diagnostics = MCMCDiagnostics(
            method="random_walk_metropolis",
            n_iterations=n_iterations,
            n_warmup=n_warmup,
            n_kept_draws=len(draws),
            thin=thin,
            n_accepted=accepted_total,
            acceptance_rate=float(accepted_total / n_iterations),
            random_seed=random_seed,
        )
        return MCMCPosteriorResult(
            draws=draws,
            posterior_samples=posterior_samples,
            diagnostics=diagnostics,
            pointwise_log_likelihood_draws=pointwise_log_likelihood_draws,
            compatibility=compatibility,
        )

    def _evaluate_state(
        self,
        trace: EpisodeTrace,
        params: dict[str, float],
    ) -> _EvaluatedState:
        """Evaluate one parameter state under likelihood and prior."""

        model = self._model_factory(params)
        replay = self._likelihood_program.evaluate(trace, model)
        log_likelihood = float(replay.total_log_likelihood)
        log_prior = float(self._prior_program.log_prior(params))
        log_posterior = float(log_likelihood + log_prior)
        return _EvaluatedState(
            candidate=PosteriorCandidate(
                params=dict(params),
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                log_posterior=log_posterior,
            ),
            pointwise_log_likelihood=tuple(
                float(step.log_probability) for step in replay.steps
            ),
        )


def sample_posterior_model(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    prior_program: PriorProgram,
    initial_params: Mapping[str, float],
    n_samples: int,
    n_warmup: int = 500,
    thin: int = 1,
    proposal_scales: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    random_seed: int | None = None,
) -> MCMCPosteriorResult:
    """Sample posterior draws for one model and one dataset.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Dataset container.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating model instances from parameter mappings.
    prior_program : PriorProgram
        Prior evaluator.
    initial_params : Mapping[str, float]
        Initial constrained parameter values.
    n_samples : int
        Number of retained posterior draws.
    n_warmup : int, optional
        Number of warmup iterations.
    thin : int, optional
        Thinning interval.
    proposal_scales : Mapping[str, float] | None, optional
        Per-parameter Gaussian proposal scales.
    bounds : Mapping[str, tuple[float | None, float | None]] | None, optional
        Optional hard bounds.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood evaluator. Defaults to :class:`ActionReplayLikelihood`.
    random_seed : int | None, optional
        Optional deterministic sampler seed.

    Returns
    -------
    MCMCPosteriorResult
        Posterior sampling output.
    """

    trace = coerce_episode_trace(data)
    like = likelihood_program if likelihood_program is not None else ActionReplayLikelihood()
    estimator = RandomWalkMetropolisEstimator(
        likelihood_program=like,
        model_factory=model_factory,
        prior_program=prior_program,
        requirements=requirements,
    )
    return estimator.fit(
        trace,
        initial_params=initial_params,
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        proposal_scales=proposal_scales,
        bounds=bounds,
        random_seed=random_seed,
    )


def sample_posterior_model_from_registry(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    model_component_id: str,
    prior_program: PriorProgram,
    initial_params: Mapping[str, float],
    n_samples: int,
    n_warmup: int = 500,
    thin: int = 1,
    proposal_scales: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    random_seed: int | None = None,
) -> MCMCPosteriorResult:
    """Sample posterior draws for one registered model component.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Dataset container.
    model_component_id : str
        Model component ID from the plugin registry.
    prior_program : PriorProgram
        Prior evaluator.
    initial_params : Mapping[str, float]
        Initial constrained parameter values.
    n_samples : int
        Number of retained posterior draws.
    n_warmup : int, optional
        Number of warmup iterations.
    thin : int, optional
        Thinning interval.
    proposal_scales : Mapping[str, float] | None, optional
        Per-parameter Gaussian proposal scales.
    bounds : Mapping[str, tuple[float | None, float | None]] | None, optional
        Optional hard bounds.
    model_kwargs : Mapping[str, Any] | None, optional
        Fixed model constructor keyword arguments.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood evaluator. Defaults to :class:`ActionReplayLikelihood`.
    random_seed : int | None, optional
        Optional deterministic sampler seed.

    Returns
    -------
    MCMCPosteriorResult
        Posterior sampling output.
    """

    reg = registry if registry is not None else build_default_registry()
    manifest = reg.get("model", model_component_id)
    fixed_kwargs = dict(model_kwargs) if model_kwargs is not None else {}

    def _factory(params: dict[str, float]) -> AgentModel:
        merged = dict(fixed_kwargs)
        merged.update(params)
        return reg.create_model(model_component_id, **merged)

    return sample_posterior_model(
        data,
        model_factory=_factory,
        prior_program=prior_program,
        initial_params=initial_params,
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        proposal_scales=proposal_scales,
        bounds=bounds,
        requirements=manifest.requirements,
        likelihood_program=likelihood_program,
        random_seed=random_seed,
    )


def _resolve_proposal_scales(
    *,
    names: tuple[str, ...],
    proposal_scales: Mapping[str, float] | None,
    default_scale: float,
) -> np.ndarray:
    """Resolve per-parameter proposal scales."""

    provided = dict(proposal_scales) if proposal_scales is not None else {}
    unknown = sorted(set(provided) - set(names))
    if unknown:
        raise ValueError(f"proposal_scales contains unknown parameters: {unknown}")

    scales = np.asarray(
        [float(provided.get(name, default_scale)) for name in names],
        dtype=float,
    )
    if np.any(scales <= 0.0):
        raise ValueError("all proposal scales must be > 0")
    return scales


def _normalize_bounds(
    names: tuple[str, ...],
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
) -> dict[str, tuple[float | None, float | None]]:
    """Normalize and validate optional hard bounds."""

    provided = dict(bounds) if bounds is not None else {}
    unknown = sorted(set(provided) - set(names))
    if unknown:
        raise ValueError(f"bounds contains unknown parameters: {unknown}")

    normalized: dict[str, tuple[float | None, float | None]] = {}
    for name in names:
        lower, upper = provided.get(name, (None, None))
        if lower is not None and upper is not None and float(lower) > float(upper):
            raise ValueError(
                f"invalid bounds for parameter {name!r}: lower={lower} > upper={upper}"
            )
        normalized[name] = (
            float(lower) if lower is not None else None,
            float(upper) if upper is not None else None,
        )
    return normalized


def _within_bounds(
    params: Mapping[str, float],
    bounds: Mapping[str, tuple[float | None, float | None]],
) -> bool:
    """Return whether parameters satisfy all hard bounds."""

    for name, value in params.items():
        lower, upper = bounds[name]
        if lower is not None and float(value) < lower:
            return False
        if upper is not None and float(value) > upper:
            return False
    return True


def _propose(
    current_params: Mapping[str, float],
    *,
    names: tuple[str, ...],
    scales: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Generate one Gaussian random-walk proposal."""

    current = np.asarray([float(current_params[name]) for name in names], dtype=float)
    proposal = current + rng.normal(loc=0.0, scale=scales, size=len(names))
    return {name: float(value) for name, value in zip(names, proposal, strict=True)}


def _metropolis_accept(
    *,
    current_log_posterior: float,
    proposal_log_posterior: float,
    rng: np.random.Generator,
) -> bool:
    """Decide Metropolis acceptance for one proposal."""

    if not np.isfinite(proposal_log_posterior):
        return False
    if proposal_log_posterior >= current_log_posterior:
        return True

    log_alpha = float(proposal_log_posterior - current_log_posterior)
    return bool(np.log(rng.uniform(0.0, 1.0)) < log_alpha)


__all__ = [
    "MCMCDiagnostics",
    "MCMCDraw",
    "MCMCPosteriorResult",
    "RandomWalkMetropolisEstimator",
    "posterior_samples_from_draws",
    "sample_posterior_model",
    "sample_posterior_model_from_registry",
]
