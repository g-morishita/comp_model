"""Tests for event-trace generator components."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.demonstrators import FixedSequenceDemonstrator
from comp_model.generators import (
    AsocialBlockSpec,
    EventTraceAsocialGenerator,
    EventTraceSocialPostOutcomeGenerator,
    EventTraceSocialPreChoiceGenerator,
    SocialBlockSpec,
)
from comp_model.models import AsocialQValueSoftmaxModel, UniformRandomPolicyModel
from comp_model.plugins import build_default_registry


def test_asocial_generator_simulates_subject_blocks() -> None:
    """Asocial generator should produce subject data with attached traces."""

    generator = EventTraceAsocialGenerator()
    subject = generator.simulate_subject(
        subject_id="s1",
        model=AsocialQValueSoftmaxModel(),
        blocks=(
            AsocialBlockSpec(n_trials=5, problem_kwargs={"reward_probabilities": [0.2, 0.8]}, block_id="b1"),
            AsocialBlockSpec(n_trials=4, problem_kwargs={"reward_probabilities": [0.7, 0.3]}, block_id="b2"),
        ),
        rng=np.random.default_rng(0),
    )

    assert subject.subject_id == "s1"
    assert len(subject.blocks) == 2
    assert subject.blocks[0].event_trace is not None
    assert subject.blocks[0].n_trials == 5
    assert subject.blocks[1].n_trials == 4


def test_asocial_generator_simulates_study() -> None:
    """Asocial generator should simulate multiple subjects into a study."""

    generator = EventTraceAsocialGenerator()
    study = generator.simulate_study(
        subject_models={"s1": UniformRandomPolicyModel(), "s2": UniformRandomPolicyModel()},
        blocks=(AsocialBlockSpec(n_trials=3, problem_kwargs={"reward_probabilities": [0.5, 0.5]}),),
        rng=np.random.default_rng(1),
    )

    assert study.n_subjects == 2
    assert all(subject.blocks[0].event_trace is not None for subject in study.subjects)


def test_social_pre_choice_generator_timing() -> None:
    """Pre-choice social generator should emit demonstrator then subject decisions."""

    generator = EventTraceSocialPreChoiceGenerator()
    block = generator.simulate_block(
        subject_model=UniformRandomPolicyModel(),
        demonstrator_model=FixedSequenceDemonstrator(sequence=[1, 1, 1]),
        block=SocialBlockSpec(n_trials=3, program_kwargs={"reward_probabilities": [0.2, 0.8]}),
        rng=np.random.default_rng(2),
    )

    trace = block.event_trace
    assert trace is not None
    for trial_index in range(3):
        decisions = [e for e in trace.by_trial(trial_index) if e.phase.value == "decision"]
        assert decisions[0].payload["actor_id"] == "demonstrator"
        assert decisions[1].payload["actor_id"] == "subject"


def test_social_post_outcome_generator_timing() -> None:
    """Post-outcome social generator should emit subject then demonstrator decisions."""

    generator = EventTraceSocialPostOutcomeGenerator()
    block = generator.simulate_block(
        subject_model=UniformRandomPolicyModel(),
        demonstrator_model=UniformRandomPolicyModel(),
        block=SocialBlockSpec(n_trials=3, program_kwargs={"reward_probabilities": [0.2, 0.8]}),
        rng=np.random.default_rng(3),
    )

    trace = block.event_trace
    assert trace is not None
    for trial_index in range(3):
        decisions = [e for e in trace.by_trial(trial_index) if e.phase.value == "decision"]
        assert decisions[0].payload["actor_id"] == "subject"
        assert decisions[1].payload["actor_id"] == "demonstrator"


def test_social_generator_study_with_single_shared_demonstrator() -> None:
    """Social study simulation should allow one demonstrator for all subjects."""

    generator = EventTraceSocialPreChoiceGenerator()
    study = generator.simulate_study(
        subject_models={"s1": UniformRandomPolicyModel(), "s2": UniformRandomPolicyModel()},
        demonstrator_models=FixedSequenceDemonstrator(sequence=[0, 1, 0]),
        blocks=(SocialBlockSpec(n_trials=3, program_kwargs={"reward_probabilities": [0.5, 0.5]}),),
        rng=np.random.default_rng(5),
    )

    assert study.n_subjects == 2
    assert all(subject.blocks[0].event_trace is not None for subject in study.subjects)


def test_registry_discovers_and_creates_generators() -> None:
    """Default registry should expose built-in generator components."""

    registry = build_default_registry()
    generator_ids = {manifest.component_id for manifest in registry.list(kind="generator")}

    assert {
        "event_trace_asocial_generator",
        "event_trace_social_pre_choice_generator",
        "event_trace_social_post_outcome_generator",
    }.issubset(generator_ids)

    asocial = registry.create_generator("event_trace_asocial_generator")
    pre_choice = registry.create_generator("event_trace_social_pre_choice_generator")
    post_outcome = registry.create_generator("event_trace_social_post_outcome_generator")

    assert isinstance(asocial, EventTraceAsocialGenerator)
    assert isinstance(pre_choice, EventTraceSocialPreChoiceGenerator)
    assert isinstance(post_outcome, EventTraceSocialPostOutcomeGenerator)


def test_social_generator_rejects_missing_subject_demonstrator_mapping() -> None:
    """Per-subject demonstrator mapping must cover all subject IDs."""

    generator = EventTraceSocialPreChoiceGenerator()

    with pytest.raises(ValueError, match="missing demonstrator model"):
        generator.simulate_study(
            subject_models={"s1": UniformRandomPolicyModel(), "s2": UniformRandomPolicyModel()},
            demonstrator_models={"s1": UniformRandomPolicyModel()},
            blocks=(SocialBlockSpec(n_trials=2, program_kwargs={"reward_probabilities": [0.5, 0.5]}),),
            rng=np.random.default_rng(9),
        )
