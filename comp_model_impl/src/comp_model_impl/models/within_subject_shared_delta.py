"""Within-subject (block-level) condition handling via shared+delta parameters.

This module provides lightweight wrappers around an existing
:class:`~comp_model_core.interfaces.model.ComputationalModel` (or
:class:`~comp_model_core.interfaces.model.SocialComputationalModel`) to support
within-subject experimental designs where **each block has an explicit
condition label**.

Design
------
We implement the preferred *shared + delta* parameterization:

- Each base parameter has a **shared** unconstrained value ``z_shared``.
- Each non-baseline condition has a **delta** unconstrained value ``z_delta``.
- For a given condition ``c`` and parameter ``p``:

  ``z(p, c) = z_shared(p) + z_delta(p, c)``  (baseline deltas are fixed to 0)

  ``theta(p, c) = transform_p.forward(z(p, c))``

Here ``transform_p`` is taken from the wrapped model's
:class:`~comp_model_core.params.schema.ParameterSchema`.

Important
---------
This wrapper has a parameter schema over *unconstrained* z-variables
(``Identity`` transforms). Estimators that optimize in z-space (e.g.
``TransformedMLESubjectwiseEstimator``) can therefore fit the wrapper directly.

The active condition must be set explicitly via :meth:`set_condition`.
Generators and event-log replay in this repository have been updated to call it
at each ``BLOCK_START``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.params import Identity, ParamDef, ParameterSchema


def _dedupe_preserve_order(xs: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        x = str(x)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _build_shared_delta_schema(
    *,
    base_schema: ParameterSchema,
    conditions: Sequence[str],
    baseline_condition: str,
) -> ParameterSchema:
    conds = _dedupe_preserve_order([str(c) for c in conditions])
    baseline = str(baseline_condition)
    if not conds:
        raise ValueError("conditions must be non-empty")
    if baseline not in conds:
        raise ValueError(f"baseline_condition {baseline!r} must be included in conditions")

    # Shared defaults live in base z-space at the base defaults.
    base_z0 = base_schema.default_z()
    base_params = base_schema.params

    params: list[ParamDef] = []

    # shared z's
    for i, p in enumerate(base_params):
        params.append(
            ParamDef(
                name=f"{p.name}__shared_z",
                default=float(base_z0[i]),
                bound=None,
                transform=Identity(),
            )
        )

    # delta z's for non-baseline conditions
    for c in conds:
        if c == baseline:
            continue
        for p in base_params:
            params.append(
                ParamDef(
                    name=f"{p.name}__delta_z__{c}",
                    default=0.0,
                    bound=None,
                    transform=Identity(),
                )
            )

    return ParameterSchema(params=tuple(params))


def _z_vectors_from_params(
    *,
    base_schema: ParameterSchema,
    params: Mapping[str, float],
    conditions: Sequence[str],
    baseline_condition: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Parse wrapper param dict into (z_shared, z_delta_by_condition)."""
    base_params = base_schema.params
    D = len(base_params)

    z_shared = np.empty((D,), dtype=float)
    for i, p in enumerate(base_params):
        z_shared[i] = float(params[f"{p.name}__shared_z"])

    z_delta: dict[str, np.ndarray] = {}
    for c in conditions:
        c = str(c)
        if c == baseline_condition:
            z_delta[c] = np.zeros((D,), dtype=float)
            continue
        v = np.empty((D,), dtype=float)
        for i, p in enumerate(base_params):
            v[i] = float(params[f"{p.name}__delta_z__{c}"])
        z_delta[c] = v

    return z_shared, z_delta


def _condition_params_from_z(
    *,
    base_schema: ParameterSchema,
    z_shared: np.ndarray,
    z_delta_by_condition: Mapping[str, np.ndarray],
    condition: str,
) -> dict[str, float]:
    """Compute constrained base-model params for a single condition."""
    base_params = base_schema.params
    transforms = base_schema.transforms()
    z = np.asarray(z_shared, dtype=float) + np.asarray(z_delta_by_condition[str(condition)], dtype=float)
    out: dict[str, float] = {}
    for i, p in enumerate(base_params):
        out[p.name] = float(transforms[i].forward(float(z[i])))
    return out


@dataclass(slots=True)
class ConditionedSharedDeltaModel(ComputationalModel):
    """Wrap an asocial :class:`~comp_model_core.interfaces.model.ComputationalModel`."""

    base_model: ComputationalModel
    conditions: Sequence[str]
    baseline_condition: str

    # built at init
    param_schema: ParameterSchema = field(init=False)

    # runtime
    _active_condition: str | None = field(default=None, init=False, repr=False)
    _z_shared: np.ndarray | None = field(default=None, init=False, repr=False)
    _z_delta: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _params_by_condition: dict[str, dict[str, float]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.conditions = _dedupe_preserve_order([str(c) for c in self.conditions])
        self.baseline_condition = str(self.baseline_condition)
        self.param_schema = _build_shared_delta_schema(
            base_schema=self.base_model.param_schema,
            conditions=self.conditions,
            baseline_condition=self.baseline_condition,
        )

    @classmethod
    def requirements(cls):  # type: ignore[override]
        # The wrapper itself has no additional requirements. When validating a
        # plan, prefer using the wrapped base model's requirements.
        return ()

    # --------------------
    # condition management
    # --------------------
    @property
    def active_condition(self) -> str | None:
        return self._active_condition

    def set_condition(self, condition: str) -> None:
        c = str(condition)
        if c not in set(self.conditions):
            raise ValueError(f"Unknown condition {c!r}. Known: {list(self.conditions)}")
        self._active_condition = c
        # If parameters have already been set, apply this condition's constrained params now.
        if self._params_by_condition is not None:
            self.base_model.set_params(self._params_by_condition[c])

    def params_by_condition(self) -> Mapping[str, Mapping[str, float]]:
        if self._params_by_condition is None:
            raise ValueError("Parameters have not been set yet.")
        return self._params_by_condition

    # --------------------
    # core model interface
    # --------------------
    def supports(self, spec: Any) -> bool:
        return self.base_model.supports(spec)

    def set_params(self, params: Mapping[str, float]) -> None:
        # validate + store as attributes (for introspection / FitResult)
        validated = self.param_schema.validate(params, strict=True, check_bounds=False)
        for k, v in validated.items():
            setattr(self, k, float(v))

        z_shared, z_delta = _z_vectors_from_params(
            base_schema=self.base_model.param_schema,
            params=validated,
            conditions=self.conditions,
            baseline_condition=self.baseline_condition,
        )
        self._z_shared = z_shared
        self._z_delta = z_delta

        # precompute constrained parameters per condition
        self._params_by_condition = {
            c: _condition_params_from_z(
                base_schema=self.base_model.param_schema,
                z_shared=z_shared,
                z_delta_by_condition=z_delta,
                condition=c,
            )
            for c in self.conditions
        }

        # apply active condition if already set
        if self._active_condition is not None:
            self.base_model.set_params(self._params_by_condition[self._active_condition])

    def reset_block(self, *, spec: Any) -> None:
        if self._active_condition is None:
            raise ValueError(
                "Active condition is not set. Call set_condition(condition) before reset_block()."
            )
        self.base_model.reset_block(spec=spec)

    def action_probs(self, *, state: Any, spec: Any) -> np.ndarray:
        return self.base_model.action_probs(state=state, spec=spec)

    def update(self, *, state: Any, action: int, outcome: float | None, spec: Any, info: Any | None = None) -> None:
        self.base_model.update(state=state, action=action, outcome=outcome, spec=spec, info=info)


@dataclass(slots=True)
class ConditionedSharedDeltaSocialModel(SocialComputationalModel):
    """Wrap a social :class:`~comp_model_core.interfaces.model.SocialComputationalModel`."""

    base_model: SocialComputationalModel
    conditions: Sequence[str]
    baseline_condition: str

    param_schema: ParameterSchema = field(init=False)

    _active_condition: str | None = field(default=None, init=False, repr=False)
    _z_shared: np.ndarray | None = field(default=None, init=False, repr=False)
    _z_delta: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _params_by_condition: dict[str, dict[str, float]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.conditions = _dedupe_preserve_order([str(c) for c in self.conditions])
        self.baseline_condition = str(self.baseline_condition)
        self.param_schema = _build_shared_delta_schema(
            base_schema=self.base_model.param_schema,
            conditions=self.conditions,
            baseline_condition=self.baseline_condition,
        )

    @classmethod
    def requirements(cls):  # type: ignore[override]
        return ()

    @property
    def active_condition(self) -> str | None:
        return self._active_condition

    def set_condition(self, condition: str) -> None:
        c = str(condition)
        if c not in set(self.conditions):
            raise ValueError(f"Unknown condition {c!r}. Known: {list(self.conditions)}")
        self._active_condition = c
        if self._params_by_condition is not None:
            self.base_model.set_params(self._params_by_condition[c])

    def params_by_condition(self) -> Mapping[str, Mapping[str, float]]:
        if self._params_by_condition is None:
            raise ValueError("Parameters have not been set yet.")
        return self._params_by_condition

    def supports(self, spec: Any) -> bool:
        return self.base_model.supports(spec)

    def set_params(self, params: Mapping[str, float]) -> None:
        validated = self.param_schema.validate(params, strict=True, check_bounds=False)
        for k, v in validated.items():
            setattr(self, k, float(v))

        z_shared, z_delta = _z_vectors_from_params(
            base_schema=self.base_model.param_schema,
            params=validated,
            conditions=self.conditions,
            baseline_condition=self.baseline_condition,
        )
        self._z_shared = z_shared
        self._z_delta = z_delta
        self._params_by_condition = {
            c: _condition_params_from_z(
                base_schema=self.base_model.param_schema,
                z_shared=z_shared,
                z_delta_by_condition=z_delta,
                condition=c,
            )
            for c in self.conditions
        }
        if self._active_condition is not None:
            self.base_model.set_params(self._params_by_condition[self._active_condition])

    def reset_block(self, *, spec: Any) -> None:
        if self._active_condition is None:
            raise ValueError(
                "Active condition is not set. Call set_condition(condition) before reset_block()."
            )
        self.base_model.reset_block(spec=spec)

    def social_update(self, *, state: Any, social: Any, spec: Any, info: Any | None = None) -> None:
        self.base_model.social_update(state=state, social=social, spec=spec, info=info)

    def action_probs(self, *, state: Any, spec: Any) -> np.ndarray:
        return self.base_model.action_probs(state=state, spec=spec)

    def update(self, *, state: Any, action: int, outcome: float | None, spec: Any, info: Any | None = None) -> None:
        self.base_model.update(state=state, action=action, outcome=outcome, spec=spec, info=info)


def wrap_model_with_shared_delta_conditions(
    *,
    model: ComputationalModel,
    conditions: Sequence[str],
    baseline_condition: str,
) -> ComputationalModel:
    """Return a condition-aware wrapper around ``model``.

    The returned object preserves whether the model is social (i.e., is an
    instance of :class:`~comp_model_core.interfaces.model.SocialComputationalModel`).
    """
    if isinstance(model, SocialComputationalModel):
        return ConditionedSharedDeltaSocialModel(
            base_model=model,
            conditions=conditions,
            baseline_condition=baseline_condition,
        )
    return ConditionedSharedDeltaModel(
        base_model=model,
        conditions=conditions,
        baseline_condition=baseline_condition,
    )
