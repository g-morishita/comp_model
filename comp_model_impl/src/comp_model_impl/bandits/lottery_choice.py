"""Lottery-choice bandit environment.

This environment is designed for one-shot lottery-choice tasks where each trial
offers a fixed menu of lottery options (actions). Choosing an action samples an
outcome from the corresponding lottery distribution.

State carries per-action moments so utility-based choice models can evaluate
options before choosing:

- mean
- variance
- standardized skewness
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.bandit import BanditEnv, EnvStep
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind


def _as_float_tuple(xs: Sequence[Any], *, name: str) -> tuple[float, ...]:
    if not isinstance(xs, Sequence) or isinstance(xs, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers.")
    out = tuple(float(v) for v in xs)
    if len(out) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(np.asarray(out, dtype=float))):
        raise ValueError(f"{name} must contain only finite values.")
    return out


def _normalize_probs(ps: Sequence[Any]) -> tuple[float, ...]:
    p = np.asarray(_as_float_tuple(ps, name="probs"), dtype=float)
    if np.any(p < 0.0):
        raise ValueError("Lottery probabilities must be >= 0.")
    s = float(np.sum(p))
    if s <= 0.0:
        raise ValueError("Lottery probabilities must sum to a positive value.")
    p = p / s
    return tuple(float(x) for x in p)


def _moment_stats(*, outcomes: Sequence[float], probs: Sequence[float]) -> tuple[float, float, float]:
    x = np.asarray(outcomes, dtype=float)
    p = np.asarray(probs, dtype=float)
    mu = float(np.sum(p * x))
    dev = x - mu
    var = float(np.sum(p * (dev ** 2)))
    if var <= 1e-12:
        skew = 0.0
    else:
        skew = float(np.sum(p * (dev ** 3)) / (var ** 1.5))
    return mu, var, skew


@dataclass(frozen=True, slots=True)
class LotteryOption:
    """One lottery option available as a single action."""

    outcomes: tuple[float, ...]
    probs: tuple[float, ...]
    mean: float
    variance: float
    skewness: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LotteryOption":
        if not isinstance(raw, Mapping):
            raise TypeError("Each lottery option must be a mapping with keys 'outcomes' and 'probs'.")
        if "outcomes" not in raw or "probs" not in raw:
            raise ValueError("Lottery option must include keys 'outcomes' and 'probs'.")
        outcomes = _as_float_tuple(raw["outcomes"], name="outcomes")
        probs = _normalize_probs(raw["probs"])
        if len(outcomes) != len(probs):
            raise ValueError("Lottery option requires len(outcomes) == len(probs).")
        mu, var, skew = _moment_stats(outcomes=outcomes, probs=probs)
        return cls(
            outcomes=outcomes,
            probs=probs,
            mean=mu,
            variance=var,
            skewness=skew,
        )


@dataclass(frozen=True, slots=True)
class LotteryTrial:
    """All options and metadata for one trial."""

    options: tuple[LotteryOption, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LotteryTrial":
        if not isinstance(raw, Mapping):
            raise TypeError("Each trial must be a mapping with key 'lotteries'.")
        lotteries_raw = raw.get("lotteries", None)
        if not isinstance(lotteries_raw, Sequence) or isinstance(lotteries_raw, (str, bytes)):
            raise ValueError("Trial must provide a sequence under key 'lotteries'.")
        options = tuple(LotteryOption.from_mapping(x) for x in lotteries_raw)
        if len(options) < 2:
            raise ValueError("Each trial must include at least two lottery options.")
        md_raw = raw.get("metadata", {})
        if md_raw is None:
            metadata = {}
        elif isinstance(md_raw, Mapping):
            metadata = dict(md_raw)
        else:
            raise TypeError("trial.metadata must be a mapping if provided.")
        return cls(options=options, metadata=metadata)


@dataclass(slots=True)
class LotteryChoiceBanditEnv(BanditEnv):
    """Asocial bandit environment for trial-wise lottery-choice menus.

    Configuration format
    --------------------
    ``from_config`` expects:

    .. code-block:: yaml

       trials:
         - lotteries:
             - outcomes: [0, 20]
               probs: [0.95, 0.05]
             - outcomes: [5]
               probs: [1.0]
           metadata:
             representation: high
             bonus: low
         - lotteries:
             ...

    All trials must have the same number of options (actions).
    """

    trials: tuple[LotteryTrial, ...]
    _t: int = 0
    _n_actions: int = 0
    _min_outcome: float = 0.0
    _max_outcome: float = 0.0

    def __post_init__(self) -> None:
        if len(self.trials) == 0:
            raise ValueError("LotteryChoiceBanditEnv requires at least one trial.")

        n_actions = len(self.trials[0].options)
        if n_actions < 2:
            raise ValueError("LotteryChoiceBanditEnv requires at least two actions.")
        for i, trial in enumerate(self.trials):
            if len(trial.options) != n_actions:
                raise ValueError(
                    f"Inconsistent number of lottery options across trials: "
                    f"trial 0 has {n_actions}, trial {i} has {len(trial.options)}."
                )

        all_outcomes = [
            y
            for trial in self.trials
            for opt in trial.options
            for y in opt.outcomes
        ]
        self._n_actions = int(n_actions)
        self._min_outcome = float(np.min(np.asarray(all_outcomes, dtype=float)))
        self._max_outcome = float(np.max(np.asarray(all_outcomes, dtype=float)))
        self._t = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LotteryChoiceBanditEnv":
        if not isinstance(cfg, Mapping):
            raise TypeError("LotteryChoiceBanditEnv config must be a mapping.")
        trials_raw = cfg.get("trials", None)
        if not isinstance(trials_raw, Sequence) or isinstance(trials_raw, (str, bytes)):
            raise ValueError("LotteryChoiceBanditEnv config must include a sequence 'trials'.")
        trials = tuple(LotteryTrial.from_mapping(x) for x in trials_raw)
        return cls(trials=trials)

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            n_actions=int(self._n_actions),
            outcome_type=OutcomeType.CONTINUOUS,
            outcome_range=(float(self._min_outcome), float(self._max_outcome)),
            outcome_is_bounded=True,
            is_social=False,
            state_kind=StateKind.NONE,
            n_states=None,
            state_shape=None,
        )

    def _state_for_trial(self, t: int) -> dict[str, Any]:
        tt = int(t)
        if tt < 0 or tt >= len(self.trials):
            return {"trial_index": tt, "action_moments": tuple(), "metadata": {}}
        trial = self.trials[tt]
        action_moments = tuple(
            (float(opt.mean), float(opt.variance), float(opt.skewness))
            for opt in trial.options
        )
        return {
            "trial_index": tt,
            "action_moments": action_moments,
            "metadata": dict(trial.metadata),
        }

    def reset(self, *, rng: np.random.Generator) -> Any:
        self._t = 0
        return self.get_state()

    def step(self, *, action: int, rng: np.random.Generator) -> EnvStep:
        if self._t >= len(self.trials):
            raise IndexError("LotteryChoiceBanditEnv.step called after final trial.")

        trial = self.trials[self._t]
        a = int(action)
        if a < 0 or a >= self._n_actions:
            raise IndexError(f"Action index out of range: {a} for n_actions={self._n_actions}.")

        option = trial.options[a]
        idx = int(rng.choice(len(option.outcomes), p=np.asarray(option.probs, dtype=float)))
        out = float(option.outcomes[idx])

        info = {
            "trial_index": int(self._t),
            "metadata": dict(trial.metadata),
            "action_moments": [
                [float(opt.mean), float(opt.variance), float(opt.skewness)]
                for opt in trial.options
            ],
            "lotteries": [
                {
                    "outcomes": [float(x) for x in opt.outcomes],
                    "probs": [float(p) for p in opt.probs],
                }
                for opt in trial.options
            ],
        }

        self._t += 1
        done = bool(self._t >= len(self.trials))
        return EnvStep(outcome=out, done=done, info=info)

    def get_state(self) -> Any:
        return self._state_for_trial(self._t)
