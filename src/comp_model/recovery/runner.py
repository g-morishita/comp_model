from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Callable
import uuid

import numpy as np

from ..data.types import StudyData, SubjectData, Block, Trial
from ..environment.environment import Environment
from ..interfaces.model import ComputationalModel
from ..interfaces.generator import Generator
from ..recovery.schema import RecoveryTruth, RecoveryEstimate, RecoveryResult
from .sampling import IndependentSubjectSampler, HierarchicalSampler


Sampler = IndependentSubjectSampler | HierarchicalSampler


@dataclass(slots=True)
class RecoveryRunner:
    """
    Handles parameter recovery / model recovery experiments.

    - generating_model: used for simulation
    - fitted_model: used for fitting (can differ for misspecification)
    """
    generator: Generator
    env: Environment
    bandit_factory: Any  # callable(block_cfg) -> Bandit

    def run(
        self,
        *,
        seed: int,
        task_name: str,
        generating_model: ComputationalModel,
        fitted_model: ComputationalModel,
        subject_ids: Sequence[str],
        subject_block_plans: Mapping[str, Sequence[Mapping[str, Any]]],
        sampler: Sampler,
        generating_model_name: str | None = None,
        fitted_model_name: str | None = None,
        run_id: str | None = None,
    ) -> RecoveryResult:
        rng = np.random.default_rng(seed)
        run_id = run_id or str(uuid.uuid4())

        # 1) sample truth
        if isinstance(sampler, HierarchicalSampler):
            pop = sampler.sample_population(rng)
            subj_true = sampler.sample_subject_params(rng, subject_ids, pop)
        else:
            pop = None
            subj_true = sampler.sample_subject_params(rng, subject_ids)

        truth = RecoveryTruth(population=pop, subjects=subj_true)

        # 2) simulate study using generating model + true subject params
        study = self.generator.simulate_study(
            bandit_factory=self.bandit_factory,
            model=generating_model,
            subj_params=subj_true,
            subject_block_plans=subject_block_plans,
            rng=rng,
        )

        # 3) fit with fitted model (may differ)
        fit = self.env.fit(study=study, model=fitted_model, rng=rng)

        est = RecoveryEstimate(
            population=fit.population_hat,
            subjects=fit.subject_hats,
        )

        return RecoveryResult(
            run_id=run_id,
            seed=seed,
            generating_model=generating_model_name or type(generating_model).__name__,
            fitted_model=fitted_model_name or type(fitted_model).__name__,
            task=task_name,
            truth=truth,
            estimate=est,
            success=fit.success,
            message=fit.message,
            diagnostics=fit.diagnostics,
            metadata={
                "n_subjects": len(study.subjects),
                "n_blocks_total": sum(len(s.blocks) for s in study.subjects),
            },
        )
