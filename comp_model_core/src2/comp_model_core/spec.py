"""
Task specifications.

A :class:`~comp_model_core.spec.TaskSpec` is the minimal "contract" a task/bandit
exposes to a model. It defines the action space size and the semantics of outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class OutcomeType(Enum):
    """
    Outcome type for a task.

    Attributes
    ----------
    BINARY
        Outcomes are binary (e.g., reward vs no reward).
    CONTINUOUS
        Outcomes are continuous real values.
    """

    BINARY = auto()
    CONTINUOUS = auto()


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """
    Specification of a task/bandit's observable contract.

    Parameters
    ----------
    n_actions : int
        Number of available discrete actions.
    outcome_type : OutcomeType
        Semantic type of the outcome (binary vs continuous).
    outcome_range : tuple[float, float] or None, optional
        Optional numeric range for outcomes, e.g. ``(0, 1)`` or ``(0, 10)``.
    outcome_is_bounded : bool, optional
        Whether outcomes are known to lie within a finite range.
    is_social : bool, optional
        Whether the task includes social observations (demonstrator information).
    has_state : bool, optional
        Whether the task has multiple states/contexts.

    Attributes
    ----------
    n_actions : int
    outcome_type : OutcomeType
    outcome_range : tuple[float, float] or None
    outcome_is_bounded : bool
    is_social : bool
    has_state : bool
    """

    n_actions: int
    outcome_type: OutcomeType
    outcome_range: Optional[Tuple[float, float]] = None
    outcome_is_bounded: bool = False
    is_social: bool = False
    has_state: bool = False
