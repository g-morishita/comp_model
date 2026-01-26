"""
comp_model_core.data.types

Core dataset dataclasses.

These containers are intentionally logic-free and serializable.
They are designed to store simulated or observed data in a consistent format
that can be written to/read from JSON-like structures.

Notes
-----
- All classes are frozen dataclasses with slots enabled to encourage immutability
  and reduce memory overhead.
- The structures are intentionally minimal: validation and computation live
  elsewhere (e.g., generators, likelihood code).

See Also
--------
comp_model_core.spec.EnvironmentSpec
    Specification of the environment used to generate a block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..spec import EnvironmentSpec

Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Trial:
    """
    One trial/time-step.

    This dataclass stores both the *true* outcome produced by the environment and
    the *observed* outcome made available to the subject, which may be hidden or
    corrupted by noise.

    Parameters
    ----------
    t
        Trial index (typically 0-based within a block).
    state
        Environment state representation at the time of the trial. The concrete
        type depends on the environment implementation.
    choice
        Chosen action index, or ``None`` if no choice was made/recorded.
    observed_outcome
        Outcome value as observed by the subject, or ``None`` if hidden or not
        recorded.
    outcome
        True outcome value emitted by the environment, or ``None`` if not
        recorded.
    available_actions
        Optional forced-choice action set for this trial. If provided, it
        indicates the only actions that were legal/available on this trial.
    info
        Arbitrary JSON-serializable per-trial metadata.

    others_choices
        (Social tasks) Actions taken by other agent(s) on this trial, or ``None``
        if not applicable/not recorded.
    others_outcomes
        (Social tasks) True outcomes for other agent(s), or ``None`` if not
        applicable/not recorded.
    observed_others_outcomes
        (Social tasks) Outcomes for other agent(s) as observed by the subject, or
        ``None`` if hidden or not applicable.
    social_info
        (Social tasks) Arbitrary JSON-serializable social metadata.

    Attributes
    ----------
    t : int
    state : Any
    choice : int or None
    observed_outcome : float or None
    outcome : float or None
    available_actions : Sequence[int] or None
    info : Json
    others_choices : Sequence[int] or None
    others_outcomes : Sequence[float] or None
    observed_others_outcomes : Sequence[float] or None
    social_info : Json

    Notes
    -----
    - ``available_actions`` enables correct likelihood computation under
      trial-varying action sets (forced-choice / missing actions).
    - Both true and observed outcomes are stored. Observed outcomes may be
      ``None`` even when a true outcome exists (e.g., hidden feedback).
    - Social fields are optional and may be unused in asocial tasks.

    See Also
    --------
    comp_model_core.spec.TrialSpec
        Declarative per-trial interface constraints used during simulation.
    """

    t: int
    state: Any
    choice: int | None

    observed_outcome: float | None  # what subject sees
    outcome: float | None  # true outcome

    # NEW: if not None, these are the only actions that were legal on this trial.
    available_actions: Sequence[int] | None = None

    info: Json = field(default_factory=dict)

    # Social-task fields (optional)
    others_choices: Sequence[int] | None = None
    others_outcomes: Sequence[float] | None = None
    observed_others_outcomes: Sequence[float] | None = None
    social_info: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Block:
    """
    A contiguous run of trials.

    Parameters
    ----------
    block_id
        Identifier for this block (often corresponds to the plan's ``block_id``).
    trials
        Sequence of :class:`~comp_model_core.data.dataset.Trial` records.
    env_spec
        Optional :class:`~comp_model_core.spec.EnvironmentSpec` describing the
        environment used to generate the block.
    metadata
        Arbitrary JSON-serializable metadata for bookkeeping.

    Attributes
    ----------
    block_id : str
    trials : Sequence[Trial]
    env_spec : EnvironmentSpec or None
    metadata : Json
    """

    block_id: str
    trials: Sequence[Trial]
    env_spec: EnvironmentSpec | None = None
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubjectData:
    """
    Dataset for a single subject.

    Parameters
    ----------
    subject_id
        Subject identifier.
    blocks
        Sequence of :class:`~comp_model_core.data.dataset.Block` objects for this
        subject (in order).
    metadata
        Arbitrary JSON-serializable metadata for bookkeeping.

    Attributes
    ----------
    subject_id : str
    blocks : Sequence[Block]
    metadata : Json
    """

    subject_id: str
    blocks: Sequence[Block]
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StudyData:
    """
    Dataset for a full study (multiple subjects).

    Parameters
    ----------
    subjects
        Sequence of :class:`~comp_model_core.data.dataset.SubjectData` records.
    metadata
        Arbitrary JSON-serializable metadata for bookkeeping.

    Attributes
    ----------
    subjects : Sequence[SubjectData]
    metadata : Json
    """

    subjects: Sequence[SubjectData]
    metadata: Json = field(default_factory=dict)
