"""
Data containers for computational modeling.

This subpackage defines small immutable (``@dataclass(frozen=True)``) containers for
representing:

- A single trial (:class:`comp_model_core.data.types.Trial`)
- A contiguous block of trials (:class:`comp_model_core.data.types.Block`)
- A subject/session consisting of blocks (:class:`comp_model_core.data.types.SubjectData`)
- A whole study/dataset (:class:`comp_model_core.data.types.StudyData`)

These classes are intentionally "dumb" containers: they store what happened in an
experiment and optional metadata, but they do not implement modeling logic.

See Also
--------
comp_model_core.data.types
    Definitions of the underlying dataclasses.
"""

from __future__ import annotations

from ..spec import EnvironmentSpec, OutcomeType, StateKind
from ..errors import CompatibilityError
from .types import Trial, Block, SubjectData, StudyData

__all__ = [
    "EnvironmentSpec",
    "OutcomeType",
    "StateKind",
    "CompatibilityError",
    "Trial",
    "Block",
    "SubjectData",
    "StudyData",
]
