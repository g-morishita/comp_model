"""
Core dataset dataclasses.

The classes in this module are immutable containers used throughout the library:

- :class:`Trial` represents one time step.
- :class:`Block` groups trials into a contiguous segment where model latents often reset.
- :class:`SubjectData` groups blocks for a single participant/session.
- :class:`StudyData` groups subjects for fitting hierarchical models.

Notes
-----
These objects intentionally avoid modeling logic; they are meant to be easy to
construct, serialize, and inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..spec import TaskSpec

Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Trial:
    """
    Container for one trial/time step.

    Parameters
    ----------
    t : int
        Trial index (typically 0-based within the block).
    state : Any
        State identifier presented on this trial. This may be an integer, a tuple,
        or a richer object depending on the task.
    choice : int or None
        Chosen action. ``None`` indicates no choice was made (e.g., missing data).
    observed_outcome : float or None
        Outcome observed by the agent/subject. This may differ from ``outcome`` when
        outcomes are partially observed or censored.
    outcome : float or None
        True environment outcome (if defined by the environment). This is often
        present whenever a step happened, even if ``observed_outcome`` is hidden.
    info : dict[str, Any], optional
        Arbitrary task-specific information for this trial.
    others_choices : Sequence[int] or None, optional
        Demonstrator/other agent actions for social tasks.
    others_outcomes : Sequence[float] or None, optional
        True demonstrator outcomes (environment outcomes).
    observed_others_outcomes : Sequence[float] or None, optional
        Outcomes *as observed by the subject*; may be ``None`` when hidden.
    social_info : dict[str, Any], optional
        Arbitrary social-task-specific metadata.

    Attributes
    ----------
    t : int
    state : Any
    choice : int or None
    observed_outcome : float or None
    outcome : float or None
    info : dict[str, Any]
    others_choices : Sequence[int] or None
    others_outcomes : Sequence[float] or None
    observed_others_outcomes : Sequence[float] or None
    social_info : dict[str, Any]
    """

    t: int
    state: Any
    choice: int | None

    observed_outcome: float | None  # observed outcome (what subject sees)
    outcome: float | None           # environment outcome (true)

    info: Json = field(default_factory=dict)

    # Social-task fields (optional)
    others_choices: Sequence[int] | None = None
    others_outcomes: Sequence[float] | None = None           # true outcome(s)
    observed_others_outcomes: Sequence[float] | None = None  # observed outcome(s)
    social_info: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Block:
    """
    A contiguous run of trials.

    In many experimental paradigms, latent variables (e.g., Q-values) reset at the
    beginning of each block. Blocks can also carry task specifications and metadata
    such as an event log.

    Parameters
    ----------
    block_id : str
        Identifier for the block (unique within a subject).
    trials : Sequence[Trial]
        Sequence of trials in this block.
    task_spec : TaskSpec or None, optional
        Task specification describing action space and outcome semantics.
    metadata : dict[str, Any], optional
        Arbitrary metadata (e.g., generator settings, event logs).

    Attributes
    ----------
    block_id : str
    trials : Sequence[Trial]
    task_spec : TaskSpec or None
    metadata : dict[str, Any]
    """

    block_id: str
    trials: Sequence[Trial]
    task_spec: TaskSpec | None = None
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubjectData:
    """
    Dataset for a single subject/session.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    blocks : Sequence[Block]
        Blocks belonging to this subject.
    metadata : dict[str, Any], optional
        Arbitrary metadata such as demographics or preprocessing info.

    Attributes
    ----------
    subject_id : str
    blocks : Sequence[Block]
    metadata : dict[str, Any]
    """

    subject_id: str
    blocks: Sequence[Block]
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StudyData:
    """
    Whole dataset for (potentially hierarchical) inference.

    Parameters
    ----------
    subjects : Sequence[SubjectData]
        Subject-level datasets.
    metadata : dict[str, Any], optional
        Arbitrary metadata for the dataset.

    Attributes
    ----------
    subjects : Sequence[SubjectData]
    metadata : dict[str, Any]
    """

    subjects: Sequence[SubjectData]
    metadata: Json = field(default_factory=dict)
