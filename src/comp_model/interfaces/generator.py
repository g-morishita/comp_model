from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from ..data.types import StudyData
from .model import ComputationalModel
from ..plans.block import BlockPlan

class Generator(ABC):
    @abstractmethod
    def simulate_subject(
        self,
        *,
        bandit_factory: Any,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plan: Sequence[BlockPlan],
        rng: np.random.Generator,
    ):
        """
        bandit_factory: callable(block_cfg)->Bandit (so blocks can differ)
        block_plan: list of block configs (e.g., n_trials, probs, volatility, partner id)
        """
        ...

    @abstractmethod
    def simulate_study(
        self,
        *,
        bandit_factory: Any,
        model: ComputationalModel,
        subj_params: Mapping[str, Mapping[str, float]],
        subject_block_plans: Mapping[str, Sequence[BlockPlan]],
        rng: np.random.Generator,
    ) -> StudyData:
        ...
