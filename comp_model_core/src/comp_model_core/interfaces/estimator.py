"""
Estimator interface.

An estimator consumes data (typically :class:`~comp_model_core.data.types.StudyData`)
and returns fitted parameters.

This core package only defines the interface. Concrete estimation algorithms (MLE,
Bayesian inference, etc.) should live in implementation packages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..data.types import StudyData
from ..interfaces.model import ComputationalModel


@dataclass(frozen=True, slots=True)
class FitResult:
    """
    Result returned by :meth:`Estimator.fit`.

    Parameters
    ----------
    params_hat : Mapping[str, float] or None, optional
        Fitted parameters for a single-subject/single-block problem.
    population_hat : Mapping[str, float] or None, optional
        Population-level parameters for hierarchical models.
    subject_hats : Mapping[str, Mapping[str, float]] or None, optional
        Subject-level parameter estimates keyed by subject id.
    value : float or None, optional
        Objective value (e.g., negative log-likelihood) at the fitted parameters.
    success : bool, optional
        Whether the fit procedure completed successfully.
    message : str, optional
        Human-readable message describing the outcome.
    diagnostics : dict[str, Any], optional
        Optional diagnostic information (optimizer traces, posterior summaries, etc.).

    Attributes
    ----------
    params_hat : Mapping[str, float] or None
    population_hat : Mapping[str, float] or None
    subject_hats : Mapping[str, Mapping[str, float]] or None
    value : float or None
    success : bool
    message : str
    diagnostics : dict[str, Any]
    """

    params_hat: Mapping[str, float] | None = None
    population_hat: Mapping[str, float] | None = None
    subject_hats: Mapping[str, Mapping[str, float]] | None = None
    value: float | None = None
    success: bool = True
    message: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Estimator(ABC):
    """
    Abstract base class for parameter estimators.

    Implementations may use deterministic optimization (MLE/MAP), MCMC, variational
    inference, or any other approach.

    Attributes
    ----------
    model : ComputationalModel
        Model instance associated with the estimator (when applicable).
    """

    model: ComputationalModel

    @abstractmethod
    def supports(self, study: StudyData) -> bool:
        """
        Check whether this estimator can fit the provided dataset.

        Parameters
        ----------
        study : StudyData
            Dataset to be fitted.

        Returns
        -------
        bool
            ``True`` if the estimator supports this dataset.
        """
        return True

    @abstractmethod
    def fit(
        self,
        *,
        study: StudyData,
        rng: np.random.Generator,
    ) -> FitResult:
        """
        Fit the estimator to a dataset.

        Parameters
        ----------
        study : StudyData
            Dataset to fit.
        rng : numpy.random.Generator
            RNG used for stochastic estimators or random initializations.

        Returns
        -------
        FitResult
            Fitted parameters and diagnostics.
        """
        ...
