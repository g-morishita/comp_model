"""Tests for config-driven likelihood program parsing."""

from __future__ import annotations

import pytest

from comp_model.inference import (
    ActionReplayLikelihood,
    ActorSubsetReplayLikelihood,
    likelihood_program_from_config,
)


def test_likelihood_program_from_config_defaults_to_action_replay() -> None:
    """Missing likelihood config should default to action replay."""

    program = likelihood_program_from_config(None)
    assert isinstance(program, ActionReplayLikelihood)


def test_likelihood_program_from_config_parses_actor_subset_replay() -> None:
    """Parser should construct actor-subset replay likelihood from config."""

    program = likelihood_program_from_config(
        {
            "type": "actor_subset_replay",
            "fitted_actor_id": "subject",
            "scored_actor_ids": ["subject"],
            "auto_fill_unmodeled_actors": True,
        }
    )
    assert isinstance(program, ActorSubsetReplayLikelihood)


def test_likelihood_program_from_config_rejects_unknown_type() -> None:
    """Parser should fail on unsupported likelihood type."""

    with pytest.raises(ValueError, match="likelihood.type must be one of"):
        likelihood_program_from_config({"type": "not_supported"})

