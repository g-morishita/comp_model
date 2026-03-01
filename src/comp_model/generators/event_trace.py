"""Study generators backed by canonical runtime event traces.

These generators provide source-compatible simulation surfaces while using the
new runtime as the single source of trial semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionProblem
from comp_model.core.data import BlockData, StudyData, SubjectData, trial_decisions_from_trace
from comp_model.plugins import ComponentManifest
from comp_model.problems import (
    StationaryBanditProblem,
    TwoStageSocialBanditProgram,
    TwoStageSocialPostOutcomeBanditProgram,
)
from comp_model.runtime import SimulationConfig, TrialProgram, run_episode, run_trial_program


@dataclass(frozen=True, slots=True)
class AsocialBlockSpec:
    """Configuration for one asocial simulated block.

    Parameters
    ----------
    n_trials : int
        Number of trials in the block.
    problem_kwargs : Mapping[str, Any], optional
        Keyword arguments for problem factory.
    block_id : str | int | None, optional
        Optional block identifier.
    seed : int | None, optional
        Optional deterministic block seed.
    metadata : Mapping[str, Any], optional
        Additional metadata attached to resulting ``BlockData``.
    """

    n_trials: int
    problem_kwargs: Mapping[str, Any] = field(default_factory=dict)
    block_id: str | int | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_trials <= 0:
            raise ValueError("n_trials must be > 0")


@dataclass(frozen=True, slots=True)
class SocialBlockSpec:
    """Configuration for one social simulated block.

    Parameters
    ----------
    n_trials : int
        Number of trials in the block.
    program_kwargs : Mapping[str, Any], optional
        Keyword arguments for social trial-program factory.
    block_id : str | int | None, optional
        Optional block identifier.
    seed : int | None, optional
        Optional deterministic block seed.
    metadata : Mapping[str, Any], optional
        Additional metadata attached to resulting ``BlockData``.
    """

    n_trials: int
    program_kwargs: Mapping[str, Any] = field(default_factory=dict)
    block_id: str | int | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_trials <= 0:
            raise ValueError("n_trials must be > 0")


class EventTraceAsocialGenerator:
    """Generator for asocial decision tasks.

    Parameters
    ----------
    problem_factory : Callable[..., DecisionProblem], optional
        Factory for block-specific problem instances.
    """

    def __init__(self, problem_factory: Callable[..., DecisionProblem] = StationaryBanditProblem) -> None:
        self._problem_factory = problem_factory

    def simulate_block(
        self,
        *,
        model: AgentModel,
        block: AsocialBlockSpec,
        rng: np.random.Generator,
    ) -> BlockData:
        """Simulate one asocial block and return canonical block data."""

        seed = _resolve_block_seed(block.seed, rng)
        problem = self._problem_factory(**dict(block.problem_kwargs))
        trace = run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=block.n_trials, seed=seed))
        trials = trial_decisions_from_trace(trace)

        metadata = {
            "generator": self.__class__.__name__,
            "timing": "asocial",
            "seed": seed,
            **dict(block.metadata),
        }
        return BlockData(block_id=block.block_id, trials=trials, event_trace=trace, metadata=metadata)

    def simulate_subject(
        self,
        *,
        subject_id: str,
        model: AgentModel,
        blocks: Sequence[AsocialBlockSpec],
        rng: np.random.Generator,
        metadata: Mapping[str, Any] | None = None,
    ) -> SubjectData:
        """Simulate one subject across multiple asocial blocks."""

        block_data = tuple(
            self.simulate_block(model=model, block=block, rng=rng)
            for block in blocks
        )
        return SubjectData(subject_id=subject_id, blocks=block_data, metadata=dict(metadata or {}))

    def simulate_study(
        self,
        *,
        subject_models: Mapping[str, AgentModel],
        blocks: Sequence[AsocialBlockSpec],
        rng: np.random.Generator,
        metadata: Mapping[str, Any] | None = None,
    ) -> StudyData:
        """Simulate a full asocial study with one model per subject."""

        subjects = tuple(
            self.simulate_subject(subject_id=subject_id, model=model, blocks=blocks, rng=rng)
            for subject_id, model in subject_models.items()
        )
        return StudyData(subjects=subjects, metadata=dict(metadata or {}))


class _SocialGeneratorBase:
    """Common social generator logic for different timing schemes."""

    def __init__(self, program_factory: Callable[..., TrialProgram], *, timing: str) -> None:
        self._program_factory = program_factory
        self._timing = timing

    def simulate_block(
        self,
        *,
        subject_model: AgentModel,
        demonstrator_model: AgentModel,
        block: SocialBlockSpec,
        rng: np.random.Generator,
    ) -> BlockData:
        """Simulate one social block and return canonical block data."""

        seed = _resolve_block_seed(block.seed, rng)
        program = self._program_factory(**dict(block.program_kwargs))
        trace = run_trial_program(
            program=program,
            models={
                "subject": subject_model,
                "demonstrator": demonstrator_model,
            },
            config=SimulationConfig(n_trials=block.n_trials, seed=seed),
        )

        self._validate_timing(trace)
        trials = trial_decisions_from_trace(trace)
        metadata = {
            "generator": self.__class__.__name__,
            "timing": self._timing,
            "seed": seed,
            **dict(block.metadata),
        }
        return BlockData(block_id=block.block_id, trials=trials, event_trace=trace, metadata=metadata)

    def simulate_subject(
        self,
        *,
        subject_id: str,
        subject_model: AgentModel,
        demonstrator_model: AgentModel,
        blocks: Sequence[SocialBlockSpec],
        rng: np.random.Generator,
        metadata: Mapping[str, Any] | None = None,
    ) -> SubjectData:
        """Simulate one subject across multiple social blocks."""

        block_data = tuple(
            self.simulate_block(
                subject_model=subject_model,
                demonstrator_model=demonstrator_model,
                block=block,
                rng=rng,
            )
            for block in blocks
        )
        return SubjectData(subject_id=subject_id, blocks=block_data, metadata=dict(metadata or {}))

    def simulate_study(
        self,
        *,
        subject_models: Mapping[str, AgentModel],
        demonstrator_models: Mapping[str, AgentModel] | AgentModel,
        blocks: Sequence[SocialBlockSpec],
        rng: np.random.Generator,
        metadata: Mapping[str, Any] | None = None,
    ) -> StudyData:
        """Simulate full social study with subject/demonstrator models."""

        subjects: list[SubjectData] = []
        for subject_id, subject_model in subject_models.items():
            demo_model = _resolve_demonstrator_model(demonstrator_models, subject_id)
            subjects.append(
                self.simulate_subject(
                    subject_id=subject_id,
                    subject_model=subject_model,
                    demonstrator_model=demo_model,
                    blocks=blocks,
                    rng=rng,
                )
            )

        return StudyData(subjects=tuple(subjects), metadata=dict(metadata or {}))

    def _validate_timing(self, trace: Any) -> None:
        """Validate per-trial actor order. Subclasses must override."""

        raise NotImplementedError


class EventTraceSocialPreChoiceGenerator(_SocialGeneratorBase):
    """Social generator where demonstrator acts before subject choice."""

    def __init__(
        self,
        program_factory: Callable[..., TrialProgram] = TwoStageSocialBanditProgram,
    ) -> None:
        super().__init__(program_factory=program_factory, timing="social_pre_choice")

    def _validate_timing(self, trace: Any) -> None:
        """Ensure actor order is demonstrator then subject for each trial."""

        _assert_trial_actor_order(trace, expected_order=("demonstrator", "subject"))


class EventTraceSocialPostOutcomeGenerator(_SocialGeneratorBase):
    """Social generator where demonstrator acts after subject outcome."""

    def __init__(
        self,
        program_factory: Callable[..., TrialProgram] = TwoStageSocialPostOutcomeBanditProgram,
    ) -> None:
        super().__init__(program_factory=program_factory, timing="social_post_outcome")

    def _validate_timing(self, trace: Any) -> None:
        """Ensure actor order is subject then demonstrator for each trial."""

        _assert_trial_actor_order(trace, expected_order=("subject", "demonstrator"))


def _assert_trial_actor_order(trace: Any, *, expected_order: tuple[str, str]) -> None:
    """Validate first two decision actor IDs in each trial."""

    trial_to_decisions: dict[int, list[str]] = {}
    for event in trace.events:
        if event.phase.value != "decision":
            continue
        actor = event.payload.get("actor_id")
        trial_to_decisions.setdefault(event.trial_index, []).append(str(actor))

    for trial_index, actor_ids in sorted(trial_to_decisions.items()):
        if tuple(actor_ids[:2]) != expected_order:
            raise ValueError(
                f"trial {trial_index} actor order {tuple(actor_ids[:2])!r} does not match expected {expected_order!r}"
            )


def _resolve_demonstrator_model(
    demonstrator_models: Mapping[str, AgentModel] | AgentModel,
    subject_id: str,
) -> AgentModel:
    """Resolve demonstrator model for a subject."""

    if isinstance(demonstrator_models, Mapping):
        if subject_id not in demonstrator_models:
            available = ", ".join(sorted(demonstrator_models))
            raise ValueError(
                f"missing demonstrator model for subject {subject_id!r}; available={available}"
            )
        return demonstrator_models[subject_id]

    return demonstrator_models


def _resolve_block_seed(seed: int | None, rng: np.random.Generator) -> int:
    """Resolve deterministic block seed."""

    if seed is not None:
        return int(seed)
    return int(rng.integers(0, 2**31 - 1))


def create_event_trace_asocial_generator() -> EventTraceAsocialGenerator:
    """Factory used by plugin discovery."""

    return EventTraceAsocialGenerator()


def create_event_trace_social_pre_choice_generator() -> EventTraceSocialPreChoiceGenerator:
    """Factory used by plugin discovery."""

    return EventTraceSocialPreChoiceGenerator()


def create_event_trace_social_post_outcome_generator() -> EventTraceSocialPostOutcomeGenerator:
    """Factory used by plugin discovery."""

    return EventTraceSocialPostOutcomeGenerator()


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="generator",
        component_id="event_trace_asocial_generator",
        factory=create_event_trace_asocial_generator,
        description="Asocial study generator producing canonical event traces",
    ),
    ComponentManifest(
        kind="generator",
        component_id="event_trace_social_pre_choice_generator",
        factory=create_event_trace_social_pre_choice_generator,
        description="Social pre-choice study generator producing canonical event traces",
    ),
    ComponentManifest(
        kind="generator",
        component_id="event_trace_social_post_outcome_generator",
        factory=create_event_trace_social_post_outcome_generator,
        description="Social post-outcome study generator producing canonical event traces",
    ),
]
