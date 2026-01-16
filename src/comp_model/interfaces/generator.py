from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from ..data.types import StudyData
from .bandit import Bandit
from .model import ComputationalModel


class Generator(ABC):
    @abstractmethod
    def simulate_subject(
        self,
        *,
        bandit_factory: Any,
        model: ComputationalModel,
        params: Mapping[str, float],
        blocks: Sequence[Mapping[str, Any]],
        rng: np.random.Generator,
    ):
        """
        bandit_factory: callable(block_cfg)->Bandit (so blocks can differ)
        blocks: list of block configs (e.g., n_trials, probs, volatility, partner id)
        """
        ...

    @abstractmethod
    def simulate_study(
        self,
        *,
        bandit_factory: Any,
        model: ComputationalModel,
        subj_params: Mapping[str, Mapping[str, float]],
        subject_block_plans: Mapping[str, Sequence[Mapping[str, Any]]],
        rng: np.random.Generator,
    ) -> StudyData:
        ...
