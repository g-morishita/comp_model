from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Mapping, Sequence

import numpy as np

from ..data.types import StudyData, SubjectData
from ..interfaces.bandit import Bandit
from ..plans.block import BlockPlan
from .model import ComputationalModel


TaskBuilder = Callable[[BlockPlan], Bandit]


class Generator(ABC):
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
