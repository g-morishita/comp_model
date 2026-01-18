from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """
    Block simulation specification (schema).

    bandit_type + bandit_config:
      - bandit_type chooses which bandit constructor/validator to use (via registry)
      - bandit_config is passed to that constructor (validated by registry)

    demonstrator_type + demonstrator_config (optional):
      - if omitted, block is asocial unless bandit itself is SocialBandit
      - if present, generator can wrap base bandit with SocialBanditWrapper
    """
    block_id: str
    n_trials: int

    bandit_type: str
    bandit_config: Mapping[str, Any]

    demonstrator_type: str | None = None
    demonstrator_config: Mapping[str, Any] | None = None

    metadata: Json = field(default_factory=dict)

    def __post_init__(self) -> None:
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
    Plans per subject. This is what you typically load from JSON/YAML.
    """
    subjects: Mapping[str, list[BlockPlan]]
    metadata: Json = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subjects:
            raise ValueError("StudyPlan.subjects must be non-empty.")