"""
plan_schemas.py

Plan schemas for simulation studies.

These schemas define **declarative**, JSON/YAML-friendly plans used to describe
simulation studies. Plans specify configuration and constraints, but do not
instantiate runtime objects (e.g., environments or runners).

Notes
-----
- A :class:`~BlockPlan` describes one block of trials for a subject.
- A :class:`~StudyPlan` maps subject IDs to a sequence of blocks.
- Trial-by-trial interface constraints are expressed via ``trial_specs``, a list
  of JSON/YAML-style dictionaries (one per trial).

Examples
--------
A minimal asocial block with explicit trial specifications:

>>> plan = BlockPlan(
...     block_id="b1",
...     n_trials=3,
...     bandit_type="BernoulliBanditEnv",
...     bandit_config={"probs": [0.2, 0.8]},
...     trial_specs=[
...         {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
...         {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
...         {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
...     ],
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """
    Declarative description of a single block to simulate.

    A block defines the environment (bandit) dynamics and a per-trial interface
    schedule describing what information is observable and which actions are
    available on each trial.

    Parameters
    ----------
    block_id
        Identifier for the block (non-empty string).
    n_trials
        Number of trials in the block. Must be > 0.
    bandit_type
        String key identifying which bandit/environment implementation to use.
    bandit_config
        JSON/YAML-friendly mapping used to configure the bandit/environment.
    trial_specs
        Per-trial interface schedule. Must be a sequence of length ``n_trials``,
        where each entry is a dict describing trial-level constraints.

        Supported keys (by convention)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self_outcome : dict
            How the subject observes its own outcome on that trial, e.g.
            ``{"kind": "HIDDEN"}``, ``{"kind": "VERIDICAL"}``,
            ``{"kind": "GAUSSIAN", "sigma": 0.1}``.
        demo_outcome : dict, optional
            (Social blocks) How the subject observes the demonstrator outcome.
            If present, should be explicit even when hidden, e.g.
            ``{"kind": "HIDDEN"}``.
        available_actions : list[int], optional
            Forced-choice action set for that trial (allowed action indices).
    demonstrator_type
        Optional string key identifying the demonstrator (social blocks). If
        provided, ``demonstrator_config`` must also be provided.
    demonstrator_config
        Optional JSON/YAML-friendly mapping used to configure the demonstrator.
        Must be provided together with ``demonstrator_type``.
    metadata
        Arbitrary JSON/YAML-friendly metadata for downstream bookkeeping.

    Attributes
    ----------
    block_id : str
    n_trials : int
    bandit_type : str
    bandit_config : Mapping[str, Any]
    trial_specs : Sequence[Json]
    demonstrator_type : str or None
    demonstrator_config : Mapping[str, Any] or None
    metadata : Json

    Notes
    -----
    - This class is **declarative**. It does not instantiate environments or
      runners; it only records configuration and constraints.
    - ``trial_specs`` is intentionally stored as raw JSON-like dicts so it can
      round-trip through JSON/YAML easily. Validation of the *contents* (e.g.,
      outcome visibility kinds) is typically performed elsewhere.

    Raises
    ------
    ValueError
        If required fields are missing/invalid, if demonstrator fields are
        inconsistently provided, or if ``trial_specs`` is not a sequence of
        dicts of length ``n_trials``.
    """

    block_id: str
    n_trials: int

    bandit_type: str
    bandit_config: Mapping[str, Any]

    # Per-trial interface schedule (fully explicit; length must equal n_trials)
    trial_specs: Sequence[Json]

    # Optional demonstrator (social blocks). When absent, the block is asocial.
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

        # Either both demonstrator_type and demonstrator_config are provided,
        # or neither (asocial block).
        if (self.demonstrator_type is None) != (self.demonstrator_config is None):
            raise ValueError(
                "BlockPlan.demonstrator_type and BlockPlan.demonstrator_config must be provided together (or both omitted)."
            )

        if not isinstance(self.trial_specs, (list, tuple)):
            raise ValueError("BlockPlan.trial_specs must be a list/sequence of dicts.")
        if len(self.trial_specs) != int(self.n_trials):
            raise ValueError("BlockPlan.trial_specs length must equal n_trials.")
        for i, ts in enumerate(self.trial_specs):
            if not isinstance(ts, dict):
                raise ValueError(f"BlockPlan.trial_specs[{i}] must be a dict.")


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """
    Declarative description of a full simulation study.

    A study plan maps each subject identifier to an ordered list of blocks
    (:class:`~BlockPlan`). Like :class:`~BlockPlan`, this structure is
    JSON/YAML-friendly and contains no instantiated runtime objects.

    Parameters
    ----------
    subjects
        Mapping from subject ID (string) to a list of :class:`~BlockPlan`
        instances describing that subject's blocks (in order).
    metadata
        Arbitrary JSON/YAML-friendly metadata for downstream bookkeeping.

    Attributes
    ----------
    subjects : Mapping[str, list[BlockPlan]]
        Subject-to-blocks mapping.
    metadata : Json
        Additional metadata.

    Raises
    ------
    ValueError
        If ``subjects`` is empty.
    """

    subjects: Mapping[str, list[BlockPlan]]
    metadata: Json = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subjects:
            raise ValueError("StudyPlan.subjects must be non-empty.")
