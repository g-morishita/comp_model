"""Convenience estimator for within-subject (block-level) conditions.

This estimator wires together:

1. A shared+delta condition-aware model wrapper
   (:func:`comp_model_impl.models.within_subject_shared_delta.wrap_model_with_shared_delta_conditions`).
2. The standard event-log MLE estimator
   (:class:`comp_model_impl.estimators.mle_event_log.TransformedMLESubjectwiseEstimator`).

The fitted parameters live in unconstrained z-space for the shared and delta
terms (Identity transforms). Derived constrained parameters per condition are
returned in the FitResult diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.data.types import StudyData
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.estimators.mle_event_log import TransformedMLESubjectwiseEstimator
from comp_model_impl.models.within_subject_shared_delta import (
    wrap_model_with_shared_delta_conditions,
    _condition_params_from_z,
    _z_vectors_from_params,
)


def _infer_conditions_from_study(study: StudyData) -> list[str]:
    conds: list[str] = []
    seen: set[str] = set()
    for subj in study.subjects:
        for blk in subj.blocks:
            c = getattr(blk, "condition", None)
            if c is None:
                raise ValueError("Block is missing required field 'condition'.")
            c = str(c)
            if c not in seen:
                seen.add(c)
                conds.append(c)
    if not conds:
        raise ValueError("No blocks/conditions found in study")
    return conds


def _ensure_within_subject_structure(study: StudyData, conditions: Sequence[str]) -> None:
    target = set(str(c) for c in conditions)
    for subj in study.subjects:
        subj_conds = {str(blk.condition) for blk in subj.blocks}
        if subj_conds != target:
            raise ValueError(
                "Within-subject estimator expects every subject to contain the same set of conditions. "
                f"Study conditions={sorted(target)}, subject {subj.subject_id!r} has={sorted(subj_conds)}"
            )


@dataclass(slots=True)
class WithinSubjectSharedDeltaTransformedMLEEstimator(Estimator):
    """Fit shared+delta within-subject parameters using transformed MLE."""

    base_model: ComputationalModel
    baseline_condition: str
    conditions: Sequence[str] | None = None

    n_starts: int = 5
    method: str = "L-BFGS-B"
    maxiter: int = 2_000
    z_init_scale: float = 0.5

    model: ComputationalModel = field(init=False)

    def supports(self, study: StudyData) -> bool:
        # We require block-level conditions.
        try:
            _ = _infer_conditions_from_study(study)
        except Exception:
            return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        conds = list(self.conditions) if self.conditions is not None else _infer_conditions_from_study(study)
        baseline = str(self.baseline_condition)
        if baseline not in set(map(str, conds)):
            raise ValueError(f"baseline_condition {baseline!r} must be present in conditions={conds}")

        # enforce within-subject structure (same set of conditions per subject)
        _ensure_within_subject_structure(study, conds)

        # Wrap the base model and delegate fitting to the standard transformed MLE estimator.
        self.model = wrap_model_with_shared_delta_conditions(
            model=self.base_model,
            conditions=conds,
            baseline_condition=baseline,
        )

        inner = TransformedMLESubjectwiseEstimator(
            model=self.model,
            n_starts=int(self.n_starts),
            method=str(self.method),
            maxiter=int(self.maxiter),
            z_init_scale=float(self.z_init_scale),
        )

        res = inner.fit(study=study, rng=rng)

        # Derive constrained (per-condition) parameters for convenience.
        derived: dict[str, dict[str, dict[str, float]]] = {}
        if res.subject_hats is not None:
            base_schema = getattr(self.base_model, "param_schema", None)
            if base_schema is not None:
                for sid, z_params in res.subject_hats.items():
                    z_shared, z_delta = _z_vectors_from_params(
                        base_schema=base_schema,
                        params=z_params,
                        conditions=conds,
                        baseline_condition=baseline,
                    )
                    derived[sid] = {
                        c: _condition_params_from_z(
                            base_schema=base_schema,
                            z_shared=z_shared,
                            z_delta_by_condition=z_delta,
                            condition=c,
                        )
                        for c in conds
                    }

        diag = dict(res.diagnostics or {})
        diag.update(
            {
                "within_subject": True,
                "baseline_condition": baseline,
                "conditions": list(conds),
                "derived_params_by_condition": derived,
            }
        )

        return FitResult(
            params_hat=res.params_hat,
            population_hat=res.population_hat,
            subject_hats=res.subject_hats,
            value=res.value,
            success=res.success,
            message=res.message,
            diagnostics=diag,
        )
