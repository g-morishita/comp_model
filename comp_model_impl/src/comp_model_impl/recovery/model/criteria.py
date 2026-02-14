"""Model selection criteria for model recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class ModelCriterion(Protocol):
    """Protocol interface for model selection criteria.

    Notes
    -----
    Implementations define both a scoring function and score direction.
    """

    name: str

    def score(self, *, ll: float, k: int, n_obs: int, waic: float | None = None) -> float:
        """Compute a scalar model-selection score.

        Parameters
        ----------
        ll : float
            Total log-likelihood.
        k : int
            Number of free parameters.
        n_obs : int
            Number of observations.
        waic : float or None, optional
            WAIC value when available. Ignored by non-WAIC criteria.

        Returns
        -------
        float
            Criterion score.
        """
        ...

    def higher_is_better(self) -> bool:
        """Return score direction.

        Returns
        -------
        bool
            ``True`` if larger scores indicate better models.
        """
        ...


@dataclass(frozen=True, slots=True)
class LogLikelihoodCriterion:
    """Log-likelihood criterion.

    Notes
    -----
    Models are ranked by maximizing total log-likelihood.
    """
    name: str = "loglike"

    def score(self, *, ll: float, k: int, n_obs: int, waic: float | None = None) -> float:
        """Return log-likelihood score.

        Parameters
        ----------
        ll : float
            Total log-likelihood.
        k : int
            Unused.
        n_obs : int
            Unused.

        Returns
        -------
        float
            Total log-likelihood.
        """
        return float(ll)

    def higher_is_better(self) -> bool:
        """Return whether larger scores are better.

        Returns
        -------
        bool
            Always ``True`` for log-likelihood.
        """
        return True


@dataclass(frozen=True, slots=True)
class AICCriterion:
    """Akaike information criterion.

    Notes
    -----
    ``AIC = 2k - 2LL`` and lower values are better.
    """
    name: str = "aic"

    def score(self, *, ll: float, k: int, n_obs: int, waic: float | None = None) -> float:
        """Compute AIC score.

        Parameters
        ----------
        ll : float
            Total log-likelihood.
        k : int
            Number of free parameters.
        n_obs : int
            Unused.

        Returns
        -------
        float
            AIC score.
        """
        return float(2.0 * k - 2.0 * ll)

    def higher_is_better(self) -> bool:
        """Return whether larger scores are better.

        Returns
        -------
        bool
            Always ``False`` for AIC.
        """
        return False


@dataclass(frozen=True, slots=True)
class BICCriterion:
    """Bayesian information criterion.

    Notes
    -----
    ``BIC = k*log(N) - 2LL`` and lower values are better.
    """
    name: str = "bic"

    def score(self, *, ll: float, k: int, n_obs: int, waic: float | None = None) -> float:
        """Compute BIC score.

        Parameters
        ----------
        ll : float
            Total log-likelihood.
        k : int
            Number of free parameters.
        n_obs : int
            Number of observations.

        Returns
        -------
        float
            BIC score.
        """
        n = max(int(n_obs), 1)
        return float(k * math.log(n) - 2.0 * ll)

    def higher_is_better(self) -> bool:
        """Return whether larger scores are better.

        Returns
        -------
        bool
            Always ``False`` for BIC.
        """
        return False


@dataclass(frozen=True, slots=True)
class WAICCriterion:
    """WAIC criterion.

    Notes
    -----
    Lower WAIC is better. If WAIC is unavailable for a fit, the criterion
    returns ``+inf`` so that fit is never preferred when finite-WAIC fits exist.
    """

    name: str = "waic"

    def score(self, *, ll: float, k: int, n_obs: int, waic: float | None = None) -> float:
        """Return WAIC score from diagnostics.

        Parameters
        ----------
        ll : float
            Unused.
        k : int
            Unused.
        n_obs : int
            Unused.
        waic : float or None, optional
            WAIC value extracted from fit diagnostics.

        Returns
        -------
        float
            WAIC score (lower is better), or ``+inf`` if unavailable.
        """
        if waic is None:
            return float("inf")
        x = float(waic)
        return x if math.isfinite(x) else float("inf")

    def higher_is_better(self) -> bool:
        """Return whether larger scores are better.

        Returns
        -------
        bool
            Always ``False`` for WAIC.
        """
        return False


def get_criterion(name: str) -> ModelCriterion:
    """Build a model selection criterion by name.

    Parameters
    ----------
    name : str
        Criterion name.

    Returns
    -------
    ModelCriterion
        Criterion instance.

    Raises
    ------
    ValueError
        If ``name`` does not match a supported criterion.
    """
    n = str(name).strip().lower()
    if n in ("ll", "loglike", "loglik", "log_likelihood", "log-likelihood"):
        return LogLikelihoodCriterion()
    if n in ("aic",):
        return AICCriterion()
    if n in ("bic",):
        return BICCriterion()
    if n in ("waic",):
        return WAICCriterion()
    raise ValueError(f"Unknown criterion: {name!r}. Expected one of: loglike, aic, bic, waic.")
