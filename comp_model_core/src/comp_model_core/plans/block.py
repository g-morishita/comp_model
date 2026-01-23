"""
Plan schemas for simulation studies.

Plans are typically loaded from JSON/YAML and then used to build tasks/bandits and
(optional) demonstrators via registries in an implementation package.

See Also
--------
comp_model_core.plans.io
    Helpers for reading plans from JSON/YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """
    Specification for simulating a single block.

    A block plan is a declarative description of what to simulate. It is meant to be
    JSON/YAML friendly.

    Parameters
    ----------
    block_id : str
        Identifier for the block (unique within a subject).
    n_trials : int
        Number of trials to simulate in this block.
    bandit_type : str
        Registry key selecting which bandit/task class to use.
    bandit_config : Mapping[str, Any]
        Configuration passed to the bandit/task constructor.
    demonstrator_type : str or None, optional
        Registry key selecting a demonstrator class for social tasks.
        If both demonstrator fields are ``None``, the block is treated as asocial
        unless the bandit itself is social.
    demonstrator_config : Mapping[str, Any] or None, optional
        Configuration passed to the demonstrator constructor.
    metadata : dict[str, Any], optional
        Arbitrary user metadata (e.g., condition labels).

    Notes
    -----
    The plan does not construct any objects by itself. Construction is delegated to
    an implementation package that owns the registries.
    """

    block_id: str
    n_trials: int

    bandit_type: str
    bandit_config: Mapping[str, Any]

    demonstrator_type: str | None = None
    demonstrator_config: Mapping[str, Any] | None = None

    metadata: Json = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate basic invariants.

        Raises
        ------
        ValueError
            If required fields are missing or inconsistent.
        """
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("BlockPlan.block_id must be a non-empty string.")
        if int(self.n_trials) <= 0:
            raise ValueError("BlockPlan.n_trials must be > 0.")
        if not isinstance(self.bandit_type, str) or not self.bandit_type:
            raise ValueError("BlockPlan.bandit_type must be a non-empty string.")
        if self.bandit_config is None:
            raise ValueError("BlockPlan.bandit_config must not be None.")

        if (self.demonstrator_type is None) ^ (self.demonstrator_config is None):
            raise ValueError(
                "BlockPlan demonstrator fields must be both set or both None: "
                "demonstrator_type and demonstrator_config."
            )
        if self.demonstrator_type is not None and not isinstance(self.demonstrator_type, str):
            raise ValueError("BlockPlan.demonstrator_type must be a string if provided.")


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """
    Simulation plans grouped by subject.

    Parameters
    ----------
    subjects : Mapping[str, list[BlockPlan]]
        Mapping from subject id to a list of block plans.
    metadata : dict[str, Any], optional
        Arbitrary metadata.

    Attributes
    ----------
    subjects : Mapping[str, list[BlockPlan]]
    metadata : dict[str, Any]
    """

    subjects: Mapping[str, list[BlockPlan]]
    metadata: Json = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate basic invariants.

        Raises
        ------
        ValueError
            If the study plan has no subjects.
        """
        if not self.subjects:
            raise ValueError("StudyPlan.subjects must be non-empty.")
