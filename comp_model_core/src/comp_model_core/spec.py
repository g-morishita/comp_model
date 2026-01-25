"""comp_model_core.spec

Environment- and trial-level specifications.

This library separates:

- **Environment contract** (:class:`EnvironmentSpec`)
  What the world *is* (number of actions, outcome semantics, state structure, etc.).

- **Trial interface schedule** (:class:`TrialSpec` / :class:`ResolvedTrialSpec`)
  What the subject/model can do/see on each trial (available actions; how outcomes
  are observed, including hidden or noisy feedback).

- **Outcome observation model** (:class:`OutcomeObservationSpec`)
  A small, explicit observation model used to transform a true outcome into what
  the subject/model observes (hidden/veridical/noisy).

This design makes it easy to represent trial-varying feedback conditions, such as:
- partial feedback (hidden outcomes) on selected trials
- noisy feedback (Gaussian noise) on selected trials
- binary feedback flips (bit-flip noise) for Bernoulli outcomes

See Also
--------
comp_model_core.interfaces.block_runner.BlockRunner
comp_model_core.plans.block.BlockPlan
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


class OutcomeType(Enum):
    """Semantic type of outcomes."""

    BINARY = auto()
    CONTINUOUS = auto()


class StateKind(Enum):
    """Kind of state/context the environment exposes."""

    NONE = auto()
    DISCRETE = auto()
    CONTINUOUS = auto()


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Minimal contract exposed by an environment.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions (arms).
    outcome_type : OutcomeType
        Semantic type of outcomes.
    outcome_range : tuple[float, float] or None, optional
        Optional numeric range for outcomes (e.g., (0, 1)).
    outcome_is_bounded : bool, optional
        True if outcomes are known to lie within a finite range.
    is_social : bool, optional
        True if the environment/run supports demonstrator/other-agent observations.
    state_kind : StateKind, optional
        NONE, DISCRETE, or CONTINUOUS.
    n_states : int or None, optional
        Required if ``state_kind == DISCRETE``. Use 1 for a single-context bandit.
        Must be ``None`` for CONTINUOUS and NONE.
    state_shape : tuple[int, ...] or None, optional
        Required if ``state_kind == CONTINUOUS`` (e.g., (d,) for a vector state).
        Must be ``None`` for DISCRETE and NONE.

    Raises
    ------
    ValueError
        If the state fields are inconsistent with ``state_kind``.
    """

    n_actions: int
    outcome_type: OutcomeType

    outcome_range: Optional[Tuple[float, float]] = None
    outcome_is_bounded: bool = False

    is_social: bool = False

    state_kind: StateKind = StateKind.DISCRETE
    n_states: Optional[int] = 1
    state_shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if int(self.n_actions) <= 0:
            raise ValueError("EnvironmentSpec.n_actions must be > 0")

        if self.state_kind is StateKind.NONE:
            if self.n_states is not None or self.state_shape is not None:
                raise ValueError("StateKind.NONE requires n_states=None and state_shape=None")
        elif self.state_kind is StateKind.DISCRETE:
            if self.n_states is None or int(self.n_states) <= 0:
                raise ValueError("StateKind.DISCRETE requires n_states > 0")
            if self.state_shape is not None:
                raise ValueError("StateKind.DISCRETE requires state_shape=None")
        elif self.state_kind is StateKind.CONTINUOUS:
            if not self.state_shape or any(int(x) <= 0 for x in self.state_shape):
                raise ValueError("StateKind.CONTINUOUS requires a positive state_shape tuple")
            if self.n_states is not None:
                raise ValueError("StateKind.CONTINUOUS requires n_states=None")
        else:
            raise ValueError(f"Unknown StateKind: {self.state_kind}")


class OutcomeObsKind(Enum):
    """How the subject observes an outcome on a trial."""

    HIDDEN = auto()
    VERIDICAL = auto()
    GAUSSIAN = auto()
    FLIP = auto()


@dataclass(frozen=True, slots=True)
class OutcomeObservationSpec:
    """Outcome observation model applied to a true outcome.

    Parameters
    ----------
    kind : OutcomeObsKind, optional
        Observation model kind.
    sigma : float or None, optional
        Standard deviation for GAUSSIAN noise. Required if ``kind == GAUSSIAN``.
    flip_p : float or None, optional
        Flip probability for FLIP noise (binary outcomes). Required if ``kind == FLIP``.
    clip_to_range : bool, optional
        If True and the environment defines ``outcome_range``, clip observed outcomes
        into that range.

    Notes
    -----
    - ``FLIP`` is only valid for environments with ``OutcomeType.BINARY``.
    - ``VERIDICAL`` returns the true outcome (optionally clipped).
    - ``HIDDEN`` returns ``None`` when applied.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid for the chosen kind.
    """

    kind: OutcomeObsKind = OutcomeObsKind.VERIDICAL
    sigma: float | None = None
    flip_p: float | None = None
    clip_to_range: bool = True

    def __post_init__(self) -> None:
        if self.kind is OutcomeObsKind.GAUSSIAN:
            if self.sigma is None or float(self.sigma) < 0.0:
                raise ValueError("OutcomeObservationSpec(GAUSSIAN) requires sigma >= 0")
        if self.kind is OutcomeObsKind.FLIP:
            if self.flip_p is None or not (0.0 <= float(self.flip_p) <= 1.0):
                raise ValueError("OutcomeObservationSpec(FLIP) requires flip_p in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """Trial-level *interface* constraints.

    This is the plan-facing (YAML/JSON friendly) representation. Outcome observation
    fields are stored as dicts and parsed during resolution.

    Parameters
    ----------
    available_actions : Sequence[int] or None, optional
        If provided, only these actions are legal on this trial.
    self_outcome : Mapping[str, Any] or None, optional
        Dict configuration for how the subject observes their own outcome on this trial.
        If None, defaults will be used by resolution.
    demo_outcome : Mapping[str, Any] or None, optional
        Dict configuration for how the subject observes demonstrator outcomes on this trial.
        Only relevant for social runners. If None, defaults will be used by resolution.
    metadata : Mapping[str, Any] or None, optional
        Optional user-defined data (condition labels, etc.).
    """

    available_actions: Sequence[int] | None = None
    self_outcome: Mapping[str, Any] | None = None
    demo_outcome: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None

    def resolve(
        self,
        *,
        default_self_outcome: OutcomeObservationSpec,
        default_demo_outcome: OutcomeObservationSpec | None,
    ) -> "ResolvedTrialSpec":
        """Resolve optional fields using defaults.

        Parameters
        ----------
        default_self_outcome : OutcomeObservationSpec
            Default observation model for self outcomes if ``self_outcome`` is None.
        default_demo_outcome : OutcomeObservationSpec or None
            Default observation model for demonstrator outcomes if ``demo_outcome`` is None.
            Use None for non-social runners.

        Returns
        -------
        ResolvedTrialSpec
            Concrete trial spec with defaults applied.
        """
        aa: tuple[int, ...] | None = None
        if self.available_actions is not None:
            seen: set[int] = set()
            out: list[int] = []
            for a in self.available_actions:
                ai = int(a)
                if ai not in seen:
                    out.append(ai)
                    seen.add(ai)
            aa = tuple(out)

        self_obs = _parse_outcome_obs(self.self_outcome, default_self_outcome)

        if default_demo_outcome is None:
            demo_obs: OutcomeObservationSpec | None = None
        else:
            demo_obs = _parse_outcome_obs(self.demo_outcome, default_demo_outcome)

        return ResolvedTrialSpec(
            available_actions=aa,
            self_outcome=self_obs,
            demo_outcome=demo_obs,
            metadata=dict(self.metadata) if self.metadata else {},
        )


@dataclass(frozen=True, slots=True)
class ResolvedTrialSpec:
    """Concrete trial interface spec with defaults applied.

    Attributes
    ----------
    available_actions : tuple[int, ...] or None
        Legal actions for this trial, or None if unconstrained.
    self_outcome : OutcomeObservationSpec
        Observation model applied to self outcomes.
    demo_outcome : OutcomeObservationSpec or None
        Observation model applied to demonstrator outcomes (social runners only).
    metadata : dict[str, Any]
        User-defined metadata.
    """

    available_actions: tuple[int, ...] | None
    self_outcome: OutcomeObservationSpec
    demo_outcome: OutcomeObservationSpec | None
    metadata: dict[str, Any]


def _parse_outcome_obs(
    raw: Mapping[str, Any] | None,
    default: OutcomeObservationSpec,
) -> OutcomeObservationSpec:
    """Parse an outcome observation configuration dict.

    Parameters
    ----------
    raw : Mapping[str, Any] or None
        Configuration dict. Keys may include:
        ``kind`` ("HIDDEN" | "VERIDICAL" | "GAUSSIAN" | "FLIP"),
        ``sigma``, ``flip_p``, ``clip_to_range``.
    default : OutcomeObservationSpec
        Default spec used when fields are missing.

    Returns
    -------
    OutcomeObservationSpec
        Parsed observation spec.
    """
    if raw is None:
        return default

    kind_raw = raw.get("kind", None)
    if kind_raw is None:
        kind = default.kind
    else:
        kind = kind_raw if isinstance(kind_raw, OutcomeObsKind) else OutcomeObsKind[str(kind_raw)]

    sigma = raw.get("sigma", default.sigma)
    flip_p = raw.get("flip_p", default.flip_p)
    clip_to_range = bool(raw.get("clip_to_range", default.clip_to_range))

    return OutcomeObservationSpec(kind=kind, sigma=sigma, flip_p=flip_p, clip_to_range=clip_to_range)


def observe_outcome(
    *,
    true_outcome: float,
    spec: EnvironmentSpec,
    obs: OutcomeObservationSpec,
    rng: np.random.Generator,
) -> float | None:
    """Apply an outcome observation model to a true outcome.

    Parameters
    ----------
    true_outcome : float
        The true outcome generated by the environment.
    spec : EnvironmentSpec
        Environment contract (used for outcome type and optional clipping range).
    obs : OutcomeObservationSpec
        Observation model to apply.
    rng : numpy.random.Generator
        Random number generator used for stochastic observation noise.

    Returns
    -------
    float or None
        Observed outcome after applying the observation model. Returns None if the
        outcome is hidden.

    Raises
    ------
    ValueError
        If ``obs.kind == FLIP`` but ``spec.outcome_type`` is not binary.

    Notes
    -----
    - For ``VERIDICAL``, the observed outcome equals the true outcome (optionally clipped).
    - For ``GAUSSIAN``, adds Normal(0, sigma) noise.
    - For ``FLIP``, converts the true outcome into {0,1} by thresholding at 0.5,
      then flips with probability ``flip_p``.

    Examples
    --------
    >>> import numpy as np
    >>> from comp_model_core.spec import EnvironmentSpec, OutcomeType, OutcomeObservationSpec, OutcomeObsKind, observe_outcome
    >>> rng = np.random.default_rng(0)
    >>> spec = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.CONTINUOUS, outcome_range=(0.0, 1.0), outcome_is_bounded=True)
    >>> obs = OutcomeObservationSpec(kind=OutcomeObsKind.GAUSSIAN, sigma=0.1)
    >>> y = observe_outcome(true_outcome=0.7, spec=spec, obs=obs, rng=rng)
    """
    if obs.kind is OutcomeObsKind.HIDDEN:
        return None

    y = float(true_outcome)

    if obs.kind is OutcomeObsKind.VERIDICAL:
        pass
    elif obs.kind is OutcomeObsKind.GAUSSIAN:
        sigma = 0.0 if obs.sigma is None else float(obs.sigma)
        if sigma > 0.0:
            y = float(y + rng.normal(loc=0.0, scale=sigma))
    elif obs.kind is OutcomeObsKind.FLIP:
        if spec.outcome_type is not OutcomeType.BINARY:
            raise ValueError("OutcomeObsKind.FLIP is only valid for OutcomeType.BINARY.")
        flip_p = 0.0 if obs.flip_p is None else float(obs.flip_p)
        b = 1.0 if float(y) > 0.5 else 0.0
        if rng.random() < flip_p:
            y = 0.0 if b == 1.0 else 1.0
        else:
            y = b
    else:
        raise ValueError(f"Unknown OutcomeObsKind: {obs.kind}")

    if obs.clip_to_range and spec.outcome_range is not None:
        lo, hi = spec.outcome_range
        y = float(min(max(y, float(lo)), float(hi)))

    return y
