"""
comp_model_core.interfaces.generator

Generator interface.

A generator produces synthetic datasets by simulating a
:class:`~comp_model_core.interfaces.block_runner.BlockRunner` together with a
computational model.

The generator defines the timing and order of operations, for example:

- when (and whether) social observations occur relative to choice,
- whether outcome observations are provided before/after model updates,
- how hidden/noisy outcomes are represented in the resulting dataset.

Notes
-----
- Generators are responsible for producing :class:`~comp_model_core.data.types.SubjectData`
  and :class:`~comp_model_core.data.types.StudyData` objects.
- The runtime environment state is encapsulated in a :class:`~comp_model_core.interfaces.block_runner.BlockRunner`
  built from a declarative :class:`~comp_model_core.plans.block.BlockPlan`.

See Also
--------
comp_model_core.interfaces.block_runner.BlockRunner
    Runtime execution interface for one block.
comp_model_core.interfaces.model.ComputationalModel
    Model interface used by generators during simulation.
comp_model_core.plans.block.BlockPlan
    Declarative description of a block.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Mapping, Sequence

import numpy as np

from ..data.types import StudyData, SubjectData
from ..plans.block import BlockPlan
from .model import ComputationalModel
from .block_runner import BlockRunner

#: Callable used by generators to build a runtime block runner from a :class:`~BlockPlan`.
BlockRunnerBuilder = Callable[[BlockPlan], BlockRunner]


class Generator(ABC):
    """
    Abstract base class for dataset generators.

    A generator defines a simulation protocol: given a computational model,
    subject parameters, and a sequence of planned blocks, it runs a
    :class:`~comp_model_core.interfaces.block_runner.BlockRunner` to produce
    synthetic data.

    Subclasses must implement :meth:`~Generator.simulate_subject`, which
    simulates a single subject across a sequence of blocks. The default
    :meth:`~Generator.simulate_study` implementation iterates subjects and
    assembles the resulting :class:`~comp_model_core.data.types.StudyData`.

    Notes
    -----
    - Generators are the place to encode "what happens when" within a trial
      (e.g., whether demonstrator observations are obtained before choice).
    - The generator should treat :class:`~BlockPlan` as declarative input and
      build runtime runners via the provided ``block_runner_builder``.

    See Also
    --------
    simulate_subject
        Simulate one subject and return :class:`~SubjectData`.
    simulate_study
        Simulate multiple subjects and return :class:`~StudyData`.
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
        """
        Simulate all blocks for a single subject.

        Parameters
        ----------
        subject_id
            Identifier for the subject being simulated.
        block_runner_builder
            Callable that constructs a :class:`~comp_model_core.interfaces.block_runner.BlockRunner`
            from a :class:`~comp_model_core.plans.block.BlockPlan`.
        model
            Computational model used to generate choices and update internal state.
        params
            Model parameter mapping for this subject (e.g., ``{"alpha": 0.2, "beta": 5.0}``).
        block_plans
            Sequence of blocks to simulate for this subject (in order).
        rng
            RNG for stochastic simulation, passed through to the runner and model
            as appropriate.

        Returns
        -------
        SubjectData
            The simulated dataset for this subject.

        Raises
        ------
        Exception
            Subclasses may raise if parameters or plans are invalid, or if the
            runner/model encounter an unrecoverable error.
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
        """
        Simulate a full study across multiple subjects.

        This method iterates over ``subject_block_plans``, looks up each subject's
        parameters in ``subj_params``, calls :meth:`~simulate_subject`, and returns
        a :class:`~comp_model_core.data.types.StudyData` containing all subjects.

        Parameters
        ----------
        block_runner_builder
            Callable that constructs a :class:`~comp_model_core.interfaces.block_runner.BlockRunner`
            from a :class:`~comp_model_core.plans.block.BlockPlan`.
        model
            Computational model used to generate choices and update internal state.
        subj_params
            Mapping from subject ID to model parameter mapping.
        subject_block_plans
            Mapping from subject ID to a sequence of :class:`~BlockPlan` objects.
        rng
            RNG for stochastic simulation.

        Returns
        -------
        StudyData
            Simulated dataset for the full study.

        Raises
        ------
        ValueError
            If parameters are missing for a subject, if no subjects are simulated,
            or if :meth:`~simulate_subject` returns an inconsistent subject ID.
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
