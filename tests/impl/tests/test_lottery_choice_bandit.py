"""Tests for LotteryChoiceBanditEnv."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model_impl.bandits.lottery_choice import LotteryChoiceBanditEnv


def _env() -> LotteryChoiceBanditEnv:
    return LotteryChoiceBanditEnv.from_config(
        {
            "trials": [
                {
                    "lotteries": [
                        {"outcomes": [0.0, 10.0], "probs": [0.9, 0.1]},
                        {"outcomes": [2.0], "probs": [1.0]},
                    ],
                    "metadata": {"representation": "low", "bonus": "low"},
                },
                {
                    "lotteries": [
                        {"outcomes": [0.0, 20.0], "probs": [0.95, 0.05]},
                        {"outcomes": [1.0], "probs": [1.0]},
                    ],
                    "metadata": {"representation": "high", "bonus": "high"},
                },
            ]
        }
    )


def test_lottery_env_state_contains_action_moments() -> None:
    env = _env()
    state = env.reset(rng=np.random.default_rng(0))
    assert isinstance(state, dict)
    assert state["trial_index"] == 0
    assert len(state["action_moments"]) == 2
    # Trial 0 action 1 is deterministic at 2.0.
    mu, var, skew = state["action_moments"][1]
    assert mu == pytest.approx(2.0)
    assert var == pytest.approx(0.0)
    assert skew == pytest.approx(0.0)


def test_lottery_env_step_advances_trial_and_emits_info() -> None:
    env = _env()
    rng = np.random.default_rng(123)
    _ = env.reset(rng=rng)

    # Deterministic action: outcome must be exactly 2.0.
    step0 = env.step(action=1, rng=rng)
    assert step0.outcome == pytest.approx(2.0)
    assert step0.done is False
    assert step0.info is not None
    assert step0.info["trial_index"] == 0
    assert step0.info["metadata"]["representation"] == "low"

    state1 = env.get_state()
    assert state1["trial_index"] == 1

    step1 = env.step(action=1, rng=rng)
    assert step1.outcome == pytest.approx(1.0)
    assert step1.done is True

    with pytest.raises(IndexError):
        _ = env.step(action=0, rng=rng)


def test_lottery_env_rejects_inconsistent_number_of_options() -> None:
    with pytest.raises(ValueError):
        _ = LotteryChoiceBanditEnv.from_config(
            {
                "trials": [
                    {"lotteries": [{"outcomes": [0.0], "probs": [1.0]}, {"outcomes": [1.0], "probs": [1.0]}]},
                    {"lotteries": [{"outcomes": [0.0], "probs": [1.0]}]},
                ]
            }
        )
