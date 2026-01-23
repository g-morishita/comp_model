"""
Generator interface.

A generator produces synthetic datasets by simulating a task environment and a
computational model. The generator is responsible for enforcing the event/order
of operations (e.g., when social observations happen relative to choice and update).

The generator interface is defined at the level of :class:`~comp_model_core.data.types.SubjectData`
and :class:`~comp_model_core.data.types.StudyData`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Mapping, Sequence

import numpy as np

from ..data.types import StudyData, SubjectData
from ..interfaces.bandit import Bandit
from ..plans.block import BlockPlan
from .model import ComputationalModel

#: Callable used by generators to build a bandit/task instance from a :class:`BlockPlan`.
TaskBuilder = Callable[[BlockPlan], Bandit]


class Generator(ABC):
    """
    Abstract base class for dataset generators.
    """

    @abstractmethod
    def simulate_subject(
        self,
        *,
        subject_id: str,
        task_builder: TaskBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        """
        Simulate one subject across multiple blocks.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        task_builder : TaskBuilder
            Callable that constructs a bandit/task from a block plan.
        model : ComputationalModel
            Model instance to simulate.
        params : Mapping[str, float]
            Model parameters for this subject.
        block_plans : Sequence[BlockPlan]
            Ordered block plans for this subject.
        rng : numpy.random.Generator
            RNG for stochastic simulation.

        Returns
        -------
        SubjectData
            Simulated subject dataset.

        Notes
        -----
        Implementations should call :meth:`~comp_model_core.interfaces.model.ComputationalModel.reset_block`
        at the beginning of each block (or equivalently reset via an event log).
        """
        ...

    def simulate_study(
        self,
        *,
        task_builder: TaskBuilder,
        model: ComputationalModel,
        subj_params: Mapping[str, Mapping[str, float]],
        subject_block_plans: Mapping[str, Sequence[BlockPlan]],
        rng: np.random.Generator,
    ) -> StudyData:
        """
        Simulate a full study (multiple subjects).

        Parameters
        ----------
        task_builder : TaskBuilder
            Callable that constructs a bandit/task from a block plan.
        model : ComputationalModel
            Model instance used for all subjects.
        subj_params : Mapping[str, Mapping[str, float]]
            Mapping from subject id to parameter dict.
        subject_block_plans : Mapping[str, Sequence[BlockPlan]]
            Mapping from subject id to a list of block plans.
        rng : numpy.random.Generator
            RNG for simulation.

        Returns
        -------
        StudyData
            Simulated study.

        Raises
        ------
        ValueError
            If required subject parameters are missing or if no subjects were simulated.
        """
        subjects: list[SubjectData] = []

        for subject_id, block_plans in subject_block_plans.items():
            if subject_id not in subj_params:
                raise ValueError(f"Missing subj_params for {subject_id}")

            subj = self.simulate_subject(
                subject_id=subject_id,
                task_builder=task_builder,
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
