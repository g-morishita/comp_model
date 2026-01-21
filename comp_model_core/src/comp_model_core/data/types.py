from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..spec import TaskSpec

Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Trial:
    t: int
    state: Any
    choice: int | None

    observed_outcome: float | None  # observed outcome (what subject sees)
    outcome: float | None           # environment outcome (always present if step happened)

    info: Json = field(default_factory=dict)

    others_choices: Sequence[int] | None = None
    others_outcomes: Sequence[float] | None = None          # environment outcome (always present if step happened)
    observed_others_outcomes: Sequence[float] | None = None # observed outcome (what subject sees)
    social_info: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Block:
    """A contiguous run. Latents typically reset at block start."""
    block_id: str
    trials: Sequence[Trial]
    task_spec: TaskSpec | None = None
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubjectData:
    subject_id: str
    blocks: Sequence[Block]
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StudyData:
    """Whole dataset for hierarchical inference."""
    subjects: Sequence[SubjectData]
    metadata: Json = field(default_factory=dict)
