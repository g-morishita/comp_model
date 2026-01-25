"""comp_model_core.interfaces.generator

Generator interface.

A generator produces synthetic datasets by simulating a
:class:`~comp_model_core.interfaces.block_runner.BlockRunner` and a computational model.

The generator defines the timing/order of operations (e.g., when social observations
happen relative to choice and private outcome updates), and returns standardized data
containers (:class:`~comp_model_core.data.types.SubjectData`, :class:`~comp_model_core.data.types.StudyData`).

See Also
--------
comp_model_core.interfaces.block_runner.BlockRunner
comp_model_core.interfaces.model.ComputationalModel
comp_model_core.plans.block.BlockPlan
comp_model_core.data.types.StudyData
comp_model_core.data.types.SubjectData
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Mapping, Sequence

import numpy as np

from ..data.types import StudyData, SubjectData
from ..plans.block import BlockPlan
from .model import ComputationalModel
from .block_runner import BlockRunner

#: Callable used by generators to build a runtime block runner from a :class:`~comp_model_core.plans.block.BlockPlan`.
#:
#: Notes
#: -----
#: Generators are intentionally decoupled from *how* runners are constructed. The builder
#: typically pulls from registries and applies plan-level defaults.
BlockRunnerBuilder = Callable[[BlockPlan], BlockRunner]


class Generator(ABC):
    """Abstract base class for dataset generators.

    A generator simulates one or more subjects by repeatedly:
    1) building a :class:`~comp_model_core.interfaces.block_runner.BlockRunner` from a plan,
    2) resetting both runner and model,
    3) sampling choices from the model,
    4) stepping the runner to produce outcomes and observations,
    5) updating the model,
    6) collecting data into standard containers.

    Implementations may differ in *when* social observations occur, how events are logged,
    or what is stored in trial records.
    """

    @abstractmethod
    def simulate_subject(
        self,
        *,
        subject_id: str,
        block_runner_builder: BlockRunnerBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        """Simulate a single subject over multiple blocks.

        Parameters
        ----------
        subject_id : str
            Subject identifier to store in the returned data.
        block_runner_builder : BlockRunnerBuilder
            Function that builds a runtime runner from a block plan.
        model : ComputationalModel
            Computational model used to generate choices and updates.
        params : Mapping[str, float]
            Model parameters for this subject.
        block_plans : Sequence[BlockPlan]
            Block blueprints specifying environments and trial schedules.
        rng : numpy.random.Generator
            Random number generator for all stochastic components.

        Returns
        -------
        SubjectData
            Simulated data for the subject.
        """
        ...

    def simulate_study(
        self,
        *,
        block_runner_builder: BlockRunnerBuilder,
        model: ComputationalModel,
        subj_params: Mapping[str, Mapping[str, float]],
        subject_block_plans: Mapping[str, Sequence[BlockPlan]],
        rng: np.random.Generator,
    ) -> StudyData:
        """Simulate a study consisting of multiple subjects.

        Parameters
        ----------
        block_runner_builder : BlockRunnerBuilder
            Function that builds a runtime runner from a block plan.
        model : ComputationalModel
            Computational model used to generate choices and updates.
        subj_params : Mapping[str, Mapping[str, float]]
            Mapping from subject_id to that subject's parameter dict.
        subject_block_plans : Mapping[str, Sequence[BlockPlan]]
            Mapping from subject_id to that subject's block plan sequence.
        rng : numpy.random.Generator
            Random number generator for all stochastic components.

        Returns
        -------
        StudyData
            Simulated data for the whole study.

        Raises
        ------
        ValueError
            If parameters are missing for a subject, if no subjects are simulated,
            or if a subject simulation returns a mismatched subject_id.
        """
        subjects: list[SubjectData] = []

        for subject_id, block_plans in subject_block_plans.items():
            if subject_id not in subj_params:
                raise ValueError(f"Missing subj_params for {subject_id}")

            subj = self.simulate_subject(
                subject_id=subject_id,
                block_runner_builder=block_runner_builder,
                model=model,
                params=subj_params[subject_id],
                block_plans=block_plans,
                rng=rng,
            )
            if subj.subject_id != subject_id:
                raise ValueError(
                    f"simulate_subject returned subject_id={subj.subject_id!r}, expected {subject_id!r}"
                )
            subjects.append(subj)

        if not subjects:
            raise ValueError("No subjects/blocks were simulated.")

        return StudyData(subjects=subjects, metadata={})
