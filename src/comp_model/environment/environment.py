from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..errors import CompatibilityError
from ..data.types import StudyData
from ..interfaces.estimator import Estimator, FitResult
from ..interfaces.model import ComputationalModel
from ..interfaces.generator import Generator


@dataclass(slots=True)
class Environment:
    generator: Generator
    estimator: Estimator

    def ensure_compatible(self, study: StudyData, model: ComputationalModel) -> None:
        # check model supports every block spec
        for subj in study.subjects:
            for blk in subj.blocks:
                spec = study.task_for_block(blk)
                if not model.supports(spec):
                    raise CompatibilityError(
                        f"{type(model).__name__} incompatible with block spec {spec} "
                        f"(subject={subj.subject_id}, block={blk.block_id})"
                    )
        if not self.estimator.supports(study, model):
            raise CompatibilityError(f"{type(self.estimator).__name__} cannot fit {type(model).__name__} for this study")

    def fit(self, *, study: StudyData, model: ComputationalModel, rng: np.random.Generator) -> FitResult:
        self.ensure_compatible(study, model)
        return self.estimator.fit(study=study, model=model, rng=rng)
