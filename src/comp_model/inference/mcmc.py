"""Shared MCMC result dataclasses.

This module intentionally contains only lightweight, backend-agnostic result
containers. Posterior sampling backends are implemented elsewhere (Stan).
"""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["MCMCDiagnostics"]
