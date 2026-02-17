"""Convenience estimator for within-subject (block-level) conditions.

This estimator composes two pieces:

1. A shared+delta condition-aware model wrapper created by
   :func:`comp_model_impl.models.within_subject_shared_delta.wrap_model_with_shared_delta_conditions`.
2. The standard event-log MLE estimator
   :class:`comp_model_impl.estimators.mle_event_log.TransformedMLESubjectwiseEstimator`.

Notes
-----
The fitted parameters live in unconstrained z-space for the shared and delta
terms (Identity transforms). For convenience, constrained parameters for each
condition are derived and returned in :attr:`FitResult.diagnostics`.

See Also
--------
comp_model_impl.models.within_subject_shared_delta.wrap_model_with_shared_delta_conditions
comp_model_impl.estimators.mle_event_log.TransformedMLESubjectwiseEstimator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.data.types import StudyData
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.model import ComputationalModel

from ..estimators.mle_event_log import TransformedMLESubjectwiseEstimator
from ..models.within_subject_shared_delta import (
    wrap_model_with_shared_delta_conditions,
    _condition_params_from_z,
    _z_vectors_from_params,
)


def _infer_conditions_from_study(study: StudyData) -> list[str]:
    """Infer unique condition labels from a study in order of appearance.

    Parameters
    ----------
    study : comp_model_core.data.types.StudyData
        Study data containing subjects and block-level condition labels.

    Returns
    -------
    list[str]
        Unique condition labels in the order they first appear.

    Raises
    ------
    ValueError
        If any block is missing a condition or no conditions are found.
    """
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
    """Validate that every subject contains the same set of conditions.

    Parameters
    ----------
    study : comp_model_core.data.types.StudyData
        Study data with subject blocks.
    conditions : Sequence[str]
        Target condition set expected for each subject.

    Raises
    ------
    ValueError
        If any subject's condition set differs from the target set.
    """
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
    """Fit shared+delta within-subject parameters using transformed MLE.

    Parameters
    ----------
    base_model : comp_model_core.interfaces.model.ComputationalModel
        Base model to be wrapped with shared+delta within-subject parameters.
    baseline_condition : str
        Condition label treated as the baseline (delta = 0).
    conditions : Sequence[str] or None, optional
        Explicit condition order. If ``None``, conditions are inferred from
        the study data.
    n_starts : int, optional
        Number of optimization restarts in z-space.
    method : str, optional
        SciPy optimizer name (default: ``"L-BFGS-B"``).
    maxiter : int, optional
        Maximum number of optimizer iterations per start.
    z_init_scale : float, optional
        Scale of random z initializations around the default.

    Notes
    -----
    This estimator wraps ``base_model`` with
    :func:`wrap_model_with_shared_delta_conditions` and delegates fitting to
    :class:`TransformedMLESubjectwiseEstimator`. Derived per-condition
    constrained parameters are placed in ``FitResult.diagnostics`` under
    ``derived_params_by_condition``.
    """

    base_model: ComputationalModel
    baseline_condition: str
    conditions: Sequence[str] | None = None

    n_starts: int = 5
    method: str = "L-BFGS-B"
    maxiter: int = 2_000
    z_init_scale: float = 0.5

    model: ComputationalModel = field(init=False)

    def supports(self, study: StudyData) -> bool:
        """Return ``True`` if block-level conditions are present.

        Parameters
        ----------
        study : comp_model_core.data.types.StudyData
            Study data to be checked.

        Returns
        -------
        bool
            ``True`` if conditions can be inferred; ``False`` otherwise.
        """
        # We require block-level conditions.
        try:
            _ = _infer_conditions_from_study(study)
        except Exception:
            return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        """Fit shared+delta parameters using event-log MLE in z-space.

        Parameters
        ----------
        study : comp_model_core.data.types.StudyData
            Study data with event logs and block-level conditions.
        rng : numpy.random.Generator
            Random number generator for initialization.

        Returns
        -------
        comp_model_core.interfaces.estimator.FitResult
            Fit result with subject-level z-parameters and derived constrained
            parameters per condition in diagnostics.

        Raises
        ------
        ValueError
            If the baseline condition is not present or if subjects do not
            share the same condition set.
        """
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
