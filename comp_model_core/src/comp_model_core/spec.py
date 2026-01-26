"""comp_model_core.spec

Environment- and trial-level specifications.

This library separates **what the environment is** from **what is observed on each trial**.

The split is important in computational modeling because many experiments manipulate
the *interface* (feedback visibility, noisy feedback, forced-choice sets) while keeping
the underlying generative dynamics unchanged.

Key types
---------
* :class:`EnvironmentSpec` – a stable *contract* describing the environment.
* :class:`TrialSpec` – trial-varying interface constraints (fully explicit, no defaults).
* :class:`OutcomeObservationSpec` – how a true outcome becomes an observed outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


class OutcomeType(Enum):
    """Semantic type of outcomes produced by the environment."""

    BINARY = auto()
    CONTINUOUS = auto()


class StateKind(Enum):
    """What kind of state/context the environment exposes."""

    NONE = auto()
    DISCRETE = auto()
    CONTINUOUS = auto()


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Minimal contract exposed by an environment.

    This describes **what the environment is capable of generating** and what
    shapes/models should expect (action count, outcome type, state structure).

    It intentionally does *not* include trial-level visibility or action-availability,
    because those are properties of the experimental *interface*.

    Parameters
    ----------
    n_actions:
        Number of discrete actions (arms).
    outcome_type:
        Semantic type of outcomes.
    outcome_range:
        Optional numeric range for outcomes (e.g., (0, 1)).
    outcome_is_bounded:
        True if outcomes are known to lie within a finite range.
    is_social:
        True if the block provides demonstrator/other-agent observations.
    state_kind:
        NONE, DISCRETE, or CONTINUOUS.
    n_states:
        Required if ``state_kind==DISCRETE``. Use 1 for a single-context bandit.
    state_shape:
        Required if ``state_kind==CONTINUOUS`` (e.g., (d,) for a vector state).
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


# ---------------------------------------------------------------------------
# Outcome observation specification
# ---------------------------------------------------------------------------


class OutcomeObservationKind(Enum):
    """How a true outcome becomes an observed outcome."""

    HIDDEN = auto()
    VERIDICAL = auto()
    GAUSSIAN = auto()
    FLIP = auto()


@dataclass(frozen=True, slots=True)
class OutcomeObservationSpec:
    """Specification of the observation channel for an outcome.

    This is used for self outcomes and (when social) demonstrator outcomes.

    Observation kinds
    -----------------
    * ``HIDDEN`` – outcome is not observed (returns ``None``).
    * ``VERIDICAL`` – observed outcome equals true outcome.
    * ``GAUSSIAN`` – observed = true + Normal(0, sigma).
    * ``FLIP`` – for binary outcomes only; observed = 1-true with prob ``flip_p``.

    Notes
    -----
    * Gaussian noise is applied to the numeric outcome and can produce values
      outside the environment's ``outcome_range``. If ``clip_to_range=True`` and
      ``EnvironmentSpec.outcome_range`` is provided, the observation is clipped.
    * For binary outcomes, Gaussian noise yields a float (not automatically
      thresholded). Use ``FLIP`` if you want a discrete noisy channel.
    """

    kind: OutcomeObservationKind = OutcomeObservationKind.VERIDICAL
    sigma: Optional[float] = None
    flip_p: Optional[float] = None
    clip_to_range: bool = True

    def __post_init__(self) -> None:
        if self.kind is OutcomeObservationKind.GAUSSIAN:
            if self.sigma is None or float(self.sigma) < 0:
                raise ValueError("GAUSSIAN requires sigma >= 0")
        elif self.kind is OutcomeObservationKind.FLIP:
            if self.flip_p is None or not (0.0 <= float(self.flip_p) <= 1.0):
                raise ValueError("FLIP requires flip_p in [0, 1]")

    def provides_outcome(self) -> bool:
        """Return True if this channel provides an outcome observation."""
        return self.kind is not OutcomeObservationKind.HIDDEN

    def observe(self, *, true_outcome: float, env: EnvironmentSpec, rng: np.random.Generator) -> float | None:
        """Generate an observed outcome from a true outcome.

        Parameters
        ----------
        true_outcome:
            The environment-generated outcome.
        env:
            Environment contract; used for validation and optional clipping.
        rng:
            RNG used to sample observation noise.
        """

        y = float(true_outcome)

        if self.kind is OutcomeObservationKind.HIDDEN:
            return None

        if self.kind is OutcomeObservationKind.VERIDICAL:
            obs = y

        elif self.kind is OutcomeObservationKind.GAUSSIAN:
            sigma = float(self.sigma or 0.0)
            obs = y + float(rng.normal(0.0, sigma))

        elif self.kind is OutcomeObservationKind.FLIP:
            if env.outcome_type is not OutcomeType.BINARY:
                raise ValueError("FLIP observation requires env.outcome_type == BINARY")
            # Interpret the true outcome as {0,1}. We accept 0/1 floats.
            yt = 1.0 if y >= 0.5 else 0.0
            if float(rng.random()) < float(self.flip_p or 0.0):
                obs = 1.0 - yt
            else:
                obs = yt

        else:
            raise ValueError(f"Unknown OutcomeObservationKind: {self.kind}")

        if self.clip_to_range and env.outcome_range is not None:
            lo, hi = float(env.outcome_range[0]), float(env.outcome_range[1])
            obs = float(np.clip(obs, lo, hi))

        return float(obs)


OutcomeObservationLike = Union[OutcomeObservationSpec, Mapping[str, Any]]


def _parse_outcome_observation(x: OutcomeObservationLike) -> OutcomeObservationSpec:
    """Parse an OutcomeObservationSpec from either a spec or a dict.

    This is primarily used for JSON/YAML-friendly trial specifications.
    """

    if isinstance(x, OutcomeObservationSpec):
        return x

    if not isinstance(x, Mapping):
        raise TypeError(f"Outcome observation must be OutcomeObservationSpec or mapping, got {type(x)}")

    kind_raw = x.get("kind", "VERIDICAL")
    if isinstance(kind_raw, OutcomeObservationKind):
        kind = kind_raw
    else:
        k = str(kind_raw).strip().upper()
        kind = OutcomeObservationKind[k]

    return OutcomeObservationSpec(
        kind=kind,
        sigma=None if x.get("sigma") is None else float(x.get("sigma")),
        flip_p=None if x.get("flip_p") is None else float(x.get("flip_p")),
        clip_to_range=bool(x.get("clip_to_range", True)),
    )


def parse_outcome_observation(x: OutcomeObservationLike) -> OutcomeObservationSpec:
    """Public wrapper for parsing an outcome observation spec from YAML/JSON.

    Parameters
    ----------
    x:
        Either an :class:`OutcomeObservationSpec` instance or a mapping such as
        ``{"kind": "GAUSSIAN", "sigma": 0.1}``.
    """

    return _parse_outcome_observation(x)


# ---------------------------------------------------------------------------
# Trial-level interface specification
# ---------------------------------------------------------------------------



@dataclass(frozen=True, slots=True)
class TrialSpec:
    """Trial-level *interface* constraints.

    Unlike environment dynamics, trial specs describe what the *subject/model*
    can see and do on each trial.

    This repo intentionally enforces **no implicit defaults** at the trial level:

    * ``self_outcome`` must be explicitly specified for every trial.
    * If the block is social, ``demo_outcome`` must be explicitly specified for
      every trial (use ``{"kind": "HIDDEN"}`` to hide it).

    ``available_actions`` is optional; when omitted, all actions are available.
    """

    self_outcome: OutcomeObservationSpec
    available_actions: tuple[int, ...] | None = None
    demo_outcome: OutcomeObservationSpec | None = None
    metadata: Mapping[str, Any] | None = None


def _dedupe_ints(seq: Sequence[Any]) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for a in seq:
        ai = int(a)
        if ai not in seen:
            out.append(ai)
            seen.add(ai)
    return tuple(out)


def parse_trial_spec_dict(
    d: Mapping[str, Any],
    *,
    is_social: bool,
    trial_index: int | None = None,
) -> TrialSpec:
    """Parse one trial spec dict into a :class:`TrialSpec`.

    Parameters
    ----------
    d:
        Mapping (e.g., YAML/JSON dict) describing a trial.
    is_social:
        Whether the block includes a demonstrator channel.
    trial_index:
        Optional trial index used only for clearer error messages.
    """

    tmsg = f" (trial {trial_index})" if trial_index is not None else ""

    if not isinstance(d, Mapping):
        raise TypeError(f"Trial spec must be a mapping{tmsg}, got {type(d)}")

    if "self_outcome" not in d or d.get("self_outcome") is None:
        raise ValueError(f"Missing required key 'self_outcome'{tmsg}.")

    self_outcome = _parse_outcome_observation(d["self_outcome"])

    # Action availability is optional.
    available_actions: tuple[int, ...] | None = None
    if d.get("available_actions") is not None:
        aa_raw = d["available_actions"]
        if not isinstance(aa_raw, Sequence) or isinstance(aa_raw, (str, bytes)):
            raise TypeError(f"'available_actions' must be a sequence of ints{tmsg}.")
        if len(aa_raw) == 0:
            raise ValueError(f"'available_actions' cannot be empty{tmsg}.")
        available_actions = _dedupe_ints(aa_raw)

    # Demonstrator outcome observation
    demo_outcome: OutcomeObservationSpec | None = None
    if is_social:
        if "demo_outcome" not in d or d.get("demo_outcome") is None:
            raise ValueError(f"Missing required key 'demo_outcome' for social block{tmsg}.")
        demo_outcome = _parse_outcome_observation(d["demo_outcome"])
    else:
        if d.get("demo_outcome") is not None:
            raise ValueError(f"'demo_outcome' is not allowed for asocial blocks{tmsg}.")

    metadata = d.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(f"'metadata' must be a mapping if provided{tmsg}.")

    return TrialSpec(
        self_outcome=self_outcome,
        available_actions=available_actions,
        demo_outcome=demo_outcome,
        metadata=None if metadata is None else dict(metadata),
    )


def parse_trial_specs_schedule(
    *,
    n_trials: int,
    raw_trial_specs: Sequence[TrialSpec | Mapping[str, Any]],
    is_social: bool,
) -> list[TrialSpec]:
    """Parse a full per-trial schedule.

    This is the canonical "no defaults" entry point. The schedule must be fully
    specified and have length equal to ``n_trials``.
    """

    if raw_trial_specs is None:
        raise ValueError("trial_specs must be provided (no implicit defaults).")

    if len(raw_trial_specs) != int(n_trials):
        raise ValueError("trial_specs length must equal n_trials")

    out: list[TrialSpec] = []
    for t, item in enumerate(raw_trial_specs):
        if isinstance(item, TrialSpec):
            # Still enforce social/asocial constraints.
            if is_social and item.demo_outcome is None:
                raise ValueError(f"trial_specs[{t}] missing demo_outcome for social block")
            if (not is_social) and item.demo_outcome is not None:
                raise ValueError(f"trial_specs[{t}] has demo_outcome but block is asocial")
            out.append(item)
        else:
            out.append(parse_trial_spec_dict(item, is_social=is_social, trial_index=t))
    return out
