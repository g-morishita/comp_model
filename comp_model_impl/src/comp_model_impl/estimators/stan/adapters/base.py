"""Stan adapter interface.

This subpackage provides a thin adapter layer between a
:class:`~comp_model_core.interfaces.model.ComputationalModel` and the Stan
program templates shipped with :mod:`comp_model_impl`.

The goal is to keep Stan templates stable (variable names, required data, etc.)
while letting models expose their configuration values to Stan in a structured
way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from comp_model_core.interfaces.model import ComputationalModel


@dataclass(frozen=True, slots=True)
class StanProgramRef:
    """Reference to a Stan program template.

    Parameters
    ----------
    family : str
        Program family, typically ``"indiv"`` (independent per-subject) or
        ``"hier"`` (hierarchical multi-subject).
    key : str
        Template key (directory name under ``estimators/stan/``), e.g. ``"vs"``
        or ``"vicarious_rl"``.
    program_name : str
        Human-readable identifier used for compilation caching/logging.
    """

    family: str
    key: str
    program_name: str


class StanAdapter(Protocol):
    """Protocol implemented by model-specific Stan adapters.

    An adapter defines:

    * which Stan template should be used for a model,
    * which priors are required in a given mode (``"indiv"``/``"hier"``),
    * how to augment exported event-log data with model configuration values,
    * which Stan variables represent subject-level and population-level results.

    Notes
    -----
    Variable naming is part of the contract with the Stan templates. If you
    change names here you must update the corresponding ``*.stan`` files.
    """

    model: ComputationalModel

    def program(self, family: str) -> StanProgramRef:
        """Return the Stan program reference for the given family."""
        ...

    def required_priors(self, family: str) -> Sequence[str]:
        """Return required prior names for the given family."""
        ...

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        """Add model-specific constants to a per-subject Stan data dict."""
        ...

    def augment_study_data(self, data: dict[str, Any]) -> None:
        """Add model-specific constants to a hierarchical Stan data dict."""
        ...

    def subject_param_names(self) -> Sequence[str]:
        """Names of subject-level Stan variables to summarize."""
        ...

    def population_var_names(self) -> Sequence[str]:
        """Names of population-level Stan variables to summarize."""
        ...
