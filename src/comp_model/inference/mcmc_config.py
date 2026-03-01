"""Config-driven MCMC posterior sampling helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model.core.data import BlockData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.plugins import PluginRegistry, build_default_registry

from .bayes import PriorProgram
from .bayes_config import prior_program_from_config
from .config import model_component_spec_from_config
from .likelihood import LikelihoodProgram
from .mcmc import MCMCPosteriorResult, sample_posterior_model_from_registry


@dataclass(frozen=True, slots=True)
class MCMCEstimatorSpec:
    """Parsed estimator spec for random-walk Metropolis sampling.

    Parameters
    ----------
    initial_params : dict[str, float]
        Initial constrained parameter values.
    n_samples : int
        Number of retained posterior draws.
    n_warmup : int
        Number of warmup iterations.
    thin : int
        Thinning interval.
    proposal_scales : dict[str, float] | None
        Optional per-parameter proposal scales.
    bounds : dict[str, tuple[float | None, float | None]] | None
        Optional hard bounds by parameter name.
    random_seed : int | None
        Optional RNG seed.
    """

    initial_params: dict[str, float]
    n_samples: int
    n_warmup: int = 500
    thin: int = 1
    proposal_scales: dict[str, float] | None = None
    bounds: dict[str, tuple[float | None, float | None]] | None = None
    random_seed: int | None = None


def mcmc_estimator_spec_from_config(estimator_cfg: Mapping[str, Any]) -> MCMCEstimatorSpec:
    """Parse MCMC estimator config mapping into :class:`MCMCEstimatorSpec`.

    Parameters
    ----------
    estimator_cfg : Mapping[str, Any]
        Estimator config mapping.

    Returns
    -------
    MCMCEstimatorSpec
        Parsed MCMC estimator specification.

    Raises
    ------
    ValueError
        If required fields are missing or invalid.
    """

    estimator = _require_mapping(estimator_cfg, field_name="estimator")
    estimator_type = _coerce_non_empty_str(estimator.get("type"), field_name="estimator.type")
    if estimator_type != "random_walk_metropolis":
        raise ValueError("estimator.type must be 'random_walk_metropolis' for MCMC config")

    n_samples = int(estimator.get("n_samples", 0))
    if n_samples <= 0:
        raise ValueError("estimator.n_samples must be > 0")

    n_warmup = int(estimator.get("n_warmup", 500))
    if n_warmup < 0:
        raise ValueError("estimator.n_warmup must be >= 0")

    thin = int(estimator.get("thin", 1))
    if thin <= 0:
        raise ValueError("estimator.thin must be > 0")

    return MCMCEstimatorSpec(
        initial_params=_coerce_float_mapping(
            estimator.get("initial_params"),
            field_name="estimator.initial_params",
        ),
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        proposal_scales=(
            _coerce_float_mapping(
                estimator.get("proposal_scales"),
                field_name="estimator.proposal_scales",
            )
            if estimator.get("proposal_scales") is not None
            else None
        ),
        bounds=(
            _coerce_bounds_mapping(estimator.get("bounds"), field_name="estimator.bounds")
            if estimator.get("bounds") is not None
            else None
        ),
        random_seed=(
            int(estimator["random_seed"])
            if estimator.get("random_seed") is not None
            else None
        ),
    )


def sample_posterior_dataset_from_config(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MCMCPosteriorResult:
    """Sample posterior draws from config for one dataset.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Dataset container.
    config : Mapping[str, Any]
        Config with ``model``, ``prior``, and ``estimator`` sections.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.

    Returns
    -------
    MCMCPosteriorResult
        Posterior sampling output.
    """

    cfg = _require_mapping(config, field_name="config")
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    prior_program: PriorProgram = prior_program_from_config(
        _require_mapping(cfg.get("prior"), field_name="config.prior")
    )
    estimator_spec = mcmc_estimator_spec_from_config(
        _require_mapping(cfg.get("estimator"), field_name="config.estimator")
    )

    reg = registry if registry is not None else build_default_registry()
    return sample_posterior_model_from_registry(
        data,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        initial_params=estimator_spec.initial_params,
        n_samples=estimator_spec.n_samples,
        n_warmup=estimator_spec.n_warmup,
        thin=estimator_spec.thin,
        proposal_scales=estimator_spec.proposal_scales,
        bounds=estimator_spec.bounds,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=likelihood_program,
        random_seed=estimator_spec.random_seed,
    )


def _coerce_bounds_mapping(
    raw: Any,
    *,
    field_name: str,
) -> dict[str, tuple[float | None, float | None]]:
    """Parse bounds mapping from config."""

    mapping = _require_mapping(raw, field_name=field_name)
    out: dict[str, tuple[float | None, float | None]] = {}
    for key, value in mapping.items():
        pair = _require_sequence(value, field_name=f"{field_name}.{key}")
        if len(pair) != 2:
            raise ValueError(f"{field_name}.{key} must have exactly two elements")
        lower_raw, upper_raw = pair
        lower = float(lower_raw) if lower_raw is not None else None
        upper = float(upper_raw) if upper_raw is not None else None
        out[str(key)] = (lower, upper)
    return out


def _coerce_float_mapping(raw: Any, *, field_name: str) -> dict[str, float]:
    """Coerce mapping of parameter -> float."""

    mapping = _require_mapping(raw, field_name=field_name)
    return {str(key): float(value) for key, value in mapping.items()}


def _coerce_non_empty_str(raw: Any, *, field_name: str) -> str:
    """Coerce non-empty string with explicit field context."""

    if raw is None:
        raise ValueError(f"{field_name} must be a non-empty string")

    value = str(raw).strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_mapping(raw: Any, *, field_name: str) -> dict[str, Any]:
    """Require dictionary-like config value."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(raw)


def _require_sequence(raw: Any, *, field_name: str) -> list[Any]:
    """Require list-like config value."""

    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    return list(raw)


__all__ = [
    "MCMCEstimatorSpec",
    "mcmc_estimator_spec_from_config",
    "sample_posterior_dataset_from_config",
]
