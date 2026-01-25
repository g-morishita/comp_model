"""comp_model_core.plans.block

Declarative plan schemas for simulation studies.

A plan is **stateless** and **serializable** (YAML/JSON). It describes what should
be run, but does not instantiate any runtime objects.

Key concept
-----------
A :class:`BlockPlan` is the blueprint for simulating a single block:
- environment type + configuration
- number of trials
- optional demonstrator type + configuration
- trial-by-trial interface schedule (trial specs)

See Also
--------
comp_model_core.interfaces.block_runner.BlockRunner
comp_model_core.spec.TrialSpec
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """Declarative description of a single block.

    Parameters
    ----------
    block_id : str
        Block identifier.
    n_trials : int
        Number of trials in the block.
    bandit_type : str
        Registry key for the environment type.
    bandit_config : Mapping[str, Any]
        Configuration dict passed to the environment factory.
    demonstrator_type : str or None, optional
        Registry key for the demonstrator type (social blocks only).
    demonstrator_config : Mapping[str, Any] or None, optional
        Configuration dict passed to the demonstrator factory (social blocks only).
    trial_specs : Sequence[Json] or None, optional
        Trial-by-trial interface schedule. Must have length ``n_trials`` if provided.
        Each element is a dict that is parsed into :class:`comp_model_core.spec.TrialSpec`.
    metadata : Json, optional
        Arbitrary user metadata.

    Raises
    ------
    ValueError
        If required fields are missing, inconsistent, or if trial_specs length mismatches n_trials.

    Notes
    -----
    ``trial_specs`` is intentionally stored as dicts for YAML/JSON friendliness.
    The runtime builder converts these into structured specs.
    """

    block_id: str
    n_trials: int

    bandit_type: str
    bandit_config: Mapping[str, Any]

    demonstrator_type: str | None = None
    demonstrator_config: Mapping[str, Any] | None = None

    trial_specs: Sequence[Json] | None = None

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

        if self.trial_specs is not None:
            if not isinstance(self.trial_specs, (list, tuple)):
                raise ValueError("BlockPlan.trial_specs must be a list/sequence of dicts.")
            if len(self.trial_specs) != int(self.n_trials):
                raise ValueError("BlockPlan.trial_specs length must equal n_trials.")
            for i, ts in enumerate(self.trial_specs):
                if not isinstance(ts, dict):
                    raise ValueError(f"BlockPlan.trial_specs[{i}] must be a dict.")


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """Declarative description of a study.

    Parameters
    ----------
    subjects : Mapping[str, list[BlockPlan]]
        Mapping from subject id to a list of block plans.
    metadata : Json, optional
        Arbitrary user metadata.

    Raises
    ------
    ValueError
        If subjects is empty.
    """

    subjects: Mapping[str, list[BlockPlan]]
    metadata: Json = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subjects:
            raise ValueError("StudyPlan.subjects must be non-empty.")
