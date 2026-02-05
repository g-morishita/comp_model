"""Within-subject condition handling via shared+delta parameters.

This module provides lightweight wrappers around an existing computational
model to support **within-subject** designs where each block has an explicit
condition label (e.g., A/B). The wrapper exposes an **unconstrained** parameter
schema in *z-space*, enabling fast optimization and Bayesian inference in a
single space while still enforcing the base model's constraints.

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
The wrapper's parameter schema is over **unconstrained** z-variables (all
``Identity`` transforms). Estimators that optimize in z-space (e.g., Stan/HMC or
transformed MLE) can therefore fit the wrapper directly.

The active condition must be set explicitly via :meth:`set_condition`.
Generators and event-log replay in this repository call it at each
``BLOCK_START`` event.

Examples
--------
Wrap an asocial model for a two-condition within-subject design:

>>> from comp_model_impl.models import QRL, wrap_model_with_shared_delta_conditions
>>> ws_model = wrap_model_with_shared_delta_conditions(
...     model=QRL(),
...     conditions=["A", "B"],
...     baseline_condition="A",
... )
>>> ws_model.param_schema.names[:3]  # shared z-params for base model
['alpha__shared_z', 'beta__shared_z']

Set a condition and z-parameters (unconstrained) before running a block:

>>> ws_model.set_condition("B")
>>> ws_model.set_params(
...     {
...         "alpha__shared_z": 0.0,
...         "beta__shared_z": 0.5,
...         "alpha__delta_z__B": 0.5,
...         "beta__delta_z__B": -0.25,
...     }
... )

See Also
--------
comp_model_impl.generators.event_log
    Event-log generators set the active condition at ``BLOCK_START``.
comp_model_impl.likelihood.event_log_replay
    Event-log replay uses the same condition switching protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.params import Identity, ParamDef, ParameterSchema


def _dedupe_preserve_order(xs: Sequence[str]) -> list[str]:
    """Return unique items in order of first appearance.

    Parameters
    ----------
    xs
        Sequence of items that can be stringified.

    Returns
    -------
    list[str]
        De-duplicated list preserving the first-seen order.

    Examples
    --------
    >>> _dedupe_preserve_order(["A", "B", "A"])
    ['A', 'B']
    """
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
    """Build a shared+delta z-parameter schema from a base schema.

    Parameters
    ----------
    base_schema
        Base model parameter schema (constrained parameters).
    conditions
        Condition labels for the within-subject design.
    baseline_condition
        Condition label used as the baseline (no deltas for this condition).

    Returns
    -------
    ParameterSchema
        Schema over unconstrained z-parameters. Shared z-parameters are created
        for all base parameters, and delta z-parameters are created for each
        non-baseline condition.

    Raises
    ------
    ValueError
        If ``conditions`` is empty or if ``baseline_condition`` is not in
        ``conditions``.

    Examples
    --------
    >>> from comp_model_core.params import ParameterSchema, ParamDef, Identity
    >>> base = ParameterSchema(params=(ParamDef("alpha", 0.0, None, Identity()),))
    >>> schema = _build_shared_delta_schema(
    ...     base_schema=base,
    ...     conditions=["A", "B"],
    ...     baseline_condition="A",
    ... )
    >>> schema.names
    ('alpha__shared_z', 'alpha__delta_z__B')
    """
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
    """Parse wrapper z-parameters into shared and delta vectors.

    Parameters
    ----------
    base_schema
        Base model schema used to determine parameter ordering.
    params
        Mapping with keys like ``<name>__shared_z`` and
        ``<name>__delta_z__<cond>``.
    conditions
        Condition labels in the within-subject design.
    baseline_condition
        Baseline condition label (delta vector is zero for this condition).

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        ``(z_shared, z_delta_by_condition)`` where ``z_shared`` is a 1D array of
        length ``D`` (number of base parameters) and ``z_delta_by_condition``
        maps each condition to a 1D array of length ``D``.

    Examples
    --------
    >>> from comp_model_core.params import ParameterSchema, ParamDef, Identity
    >>> base = ParameterSchema(params=(ParamDef("alpha", 0.0, None, Identity()),))
    >>> z_shared, z_delta = _z_vectors_from_params(
    ...     base_schema=base,
    ...     params={"alpha__shared_z": 0.1, "alpha__delta_z__B": -0.2},
    ...     conditions=["A", "B"],
    ...     baseline_condition="A",
    ... )
    >>> z_shared.tolist(), z_delta["A"].tolist(), z_delta["B"].tolist()
    ([0.1], [0.0], [-0.2])
    """
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
    """Compute constrained base-model parameters for a single condition.

    Parameters
    ----------
    base_schema
        Base model schema defining transforms.
    z_shared
        Shared unconstrained parameter vector (length ``D``).
    z_delta_by_condition
        Mapping from condition labels to delta vectors (length ``D``).
    condition
        Condition label to evaluate.

    Returns
    -------
    dict[str, float]
        Constrained parameter values for the base model under ``condition``.

    Examples
    --------
    >>> from comp_model_core.params import ParameterSchema, ParamDef, Identity
    >>> base = ParameterSchema(params=(ParamDef("alpha", 0.0, None, Identity()),))
    >>> out = _condition_params_from_z(
    ...     base_schema=base,
    ...     z_shared=np.array([0.1]),
    ...     z_delta_by_condition={"A": np.array([0.0])},
    ...     condition="A",
    ... )
    >>> out["alpha"]
    0.1
    """
    base_params = base_schema.params
    transforms = base_schema.transforms()
    z = np.asarray(z_shared, dtype=float) + np.asarray(z_delta_by_condition[str(condition)], dtype=float)
    out: dict[str, float] = {}
    for i, p in enumerate(base_params):
        out[p.name] = float(transforms[i].forward(float(z[i])))
    return out


@dataclass(slots=True)
class ConditionedSharedDeltaModel(ComputationalModel):
    """Within-subject wrapper for an asocial model (shared + delta).

    Parameters
    ----------
    base_model
        Asocial computational model to wrap.
    conditions
        Condition labels in the study (e.g., ``["A", "B"]``).
    baseline_condition
        Baseline condition label (no delta parameters are created for this
        condition).

    Attributes
    ----------
    param_schema : ParameterSchema
        Unconstrained z-parameter schema (shared + delta) used by estimators.
    active_condition : str or None
        Condition currently applied to the base model (set via
        :meth:`set_condition`).

    Notes
    -----
    - The wrapper itself is *asocial* and forwards all dynamics to ``base_model``.
    - Call :meth:`set_condition` before :meth:`reset_block`.

    Examples
    --------
    >>> from comp_model_impl.models import QRL
    >>> wrapper = ConditionedSharedDeltaModel(
    ...     base_model=QRL(),
    ...     conditions=["A", "B"],
    ...     baseline_condition="A",
    ... )
    >>> wrapper.set_condition("A")
    """

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
        """Normalize conditions and build the shared+delta schema."""
        self.conditions = _dedupe_preserve_order([str(c) for c in self.conditions])
        self.baseline_condition = str(self.baseline_condition)
        self.param_schema = _build_shared_delta_schema(
            base_schema=self.base_model.param_schema,
            conditions=self.conditions,
            baseline_condition=self.baseline_condition,
        )

    @classmethod
    def requirements(cls):  # type: ignore[override]
        """Return no additional plan requirements.

        Returns
        -------
        tuple
            Empty tuple (the wrapper adds no new requirements).
        """
        # The wrapper itself has no additional requirements. When validating a
        # plan, prefer using the wrapped base model's requirements.
        return ()

    # --------------------
    # condition management
    # --------------------
    @property
    def active_condition(self) -> str | None:
        """Return the active condition (or ``None`` if unset).

        Returns
        -------
        str or None
            Current active condition.
        """
        return self._active_condition

    def set_condition(self, condition: str) -> None:
        """Set the active condition and apply its parameters if available.

        Parameters
        ----------
        condition
            Condition label to activate.

        Raises
        ------
        ValueError
            If ``condition`` is not one of ``self.conditions``.
        """
        c = str(condition)
        if c not in set(self.conditions):
            raise ValueError(f"Unknown condition {c!r}. Known: {list(self.conditions)}")
        self._active_condition = c
        # If parameters have already been set, apply this condition's constrained params now.
        if self._params_by_condition is not None:
            self.base_model.set_params(self._params_by_condition[c])

    def params_by_condition(self) -> Mapping[str, Mapping[str, float]]:
        """Return constrained parameters for all conditions.

        Returns
        -------
        Mapping[str, Mapping[str, float]]
            Mapping from condition to constrained base-model parameters.

        Raises
        ------
        ValueError
            If parameters have not been set yet.
        """
        if self._params_by_condition is None:
            raise ValueError("Parameters have not been set yet.")
        return self._params_by_condition

    # --------------------
    # core model interface
    # --------------------
    def supports(self, spec: Any) -> bool:
        """Delegate compatibility check to the base model.

        Parameters
        ----------
        spec
            Environment specification.

        Returns
        -------
        bool
            True if the base model supports ``spec``.
        """
        return self.base_model.supports(spec)

    def set_params(self, params: Mapping[str, float]) -> None:
        """Validate and apply shared+delta z-parameters.

        Parameters
        ----------
        params
            Mapping of z-parameters (``__shared_z`` and ``__delta_z__<cond>``).

        Notes
        -----
        The base model is updated immediately if an active condition is set.
        """
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
        """Reset base model state for a new block.

        Parameters
        ----------
        spec
            Environment specification for the block.

        Raises
        ------
        ValueError
            If no active condition has been set.
        """
        if self._active_condition is None:
            raise ValueError(
                "Active condition is not set. Call set_condition(condition) before reset_block()."
            )
        self.base_model.reset_block(spec=spec)

    def action_probs(self, *, state: Any, spec: Any) -> np.ndarray:
        """Return action probabilities from the base model.

        Parameters
        ----------
        state
            Current state identifier.
        spec
            Environment specification.

        Returns
        -------
        numpy.ndarray
            Action probability vector.
        """
        return self.base_model.action_probs(state=state, spec=spec)

    def update(self, *, state: Any, action: int, outcome: float | None, spec: Any, info: Any | None = None) -> None:
        """Delegate learning update to the base model.

        Parameters
        ----------
        state
            Current state identifier.
        action
            Action index taken by the agent.
        outcome
            Observed outcome (possibly ``None`` if unobserved).
        spec
            Environment specification.
        info
            Optional metadata.
        """
        self.base_model.update(state=state, action=action, outcome=outcome, spec=spec, info=info)


@dataclass(slots=True)
class ConditionedSharedDeltaSocialModel(SocialComputationalModel):
    """Within-subject wrapper for a social model (shared + delta).

    Parameters
    ----------
    base_model
        Social computational model to wrap.
    conditions
        Condition labels in the study (e.g., ``["A", "B"]``).
    baseline_condition
        Baseline condition label (no delta parameters are created for this
        condition).

    Attributes
    ----------
    param_schema : ParameterSchema
        Unconstrained z-parameter schema (shared + delta).
    active_condition : str or None
        Condition currently applied to the base model.
    """

    base_model: SocialComputationalModel
    conditions: Sequence[str]
    baseline_condition: str

    param_schema: ParameterSchema = field(init=False)

    _active_condition: str | None = field(default=None, init=False, repr=False)
    _z_shared: np.ndarray | None = field(default=None, init=False, repr=False)
    _z_delta: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _params_by_condition: dict[str, dict[str, float]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize conditions and build the shared+delta schema."""
        self.conditions = _dedupe_preserve_order([str(c) for c in self.conditions])
        self.baseline_condition = str(self.baseline_condition)
        self.param_schema = _build_shared_delta_schema(
            base_schema=self.base_model.param_schema,
            conditions=self.conditions,
            baseline_condition=self.baseline_condition,
        )

    @classmethod
    def requirements(cls):  # type: ignore[override]
        """Return no additional plan requirements.

        Returns
        -------
        tuple
            Empty tuple (the wrapper adds no new requirements).
        """
        return ()

    @property
    def active_condition(self) -> str | None:
        """Return the active condition (or ``None`` if unset).

        Returns
        -------
        str or None
            Current active condition.
        """
        return self._active_condition

    def set_condition(self, condition: str) -> None:
        """Set the active condition and apply its parameters if available.

        Parameters
        ----------
        condition
            Condition label to activate.

        Raises
        ------
        ValueError
            If ``condition`` is not one of ``self.conditions``.
        """
        c = str(condition)
        if c not in set(self.conditions):
            raise ValueError(f"Unknown condition {c!r}. Known: {list(self.conditions)}")
        self._active_condition = c
        if self._params_by_condition is not None:
            self.base_model.set_params(self._params_by_condition[c])

    def params_by_condition(self) -> Mapping[str, Mapping[str, float]]:
        """Return constrained parameters for all conditions.

        Returns
        -------
        Mapping[str, Mapping[str, float]]
            Mapping from condition to constrained base-model parameters.

        Raises
        ------
        ValueError
            If parameters have not been set yet.
        """
        if self._params_by_condition is None:
            raise ValueError("Parameters have not been set yet.")
        return self._params_by_condition

    def supports(self, spec: Any) -> bool:
        """Delegate compatibility check to the base model.

        Parameters
        ----------
        spec
            Environment specification.

        Returns
        -------
        bool
            True if the base model supports ``spec``.
        """
        return self.base_model.supports(spec)

    def set_params(self, params: Mapping[str, float]) -> None:
        """Validate and apply shared+delta z-parameters.

        Parameters
        ----------
        params
            Mapping of z-parameters (``__shared_z`` and ``__delta_z__<cond>``).
        """
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
        """Reset base model state for a new block.

        Parameters
        ----------
        spec
            Environment specification.

        Raises
        ------
        ValueError
            If no active condition has been set.
        """
        if self._active_condition is None:
            raise ValueError(
                "Active condition is not set. Call set_condition(condition) before reset_block()."
            )
        self.base_model.reset_block(spec=spec)

    def social_update(self, *, state: Any, social: Any, spec: Any, info: Any | None = None) -> None:
        """Delegate social update to the base model.

        Parameters
        ----------
        state
            Current state identifier.
        social
            Social observation container.
        spec
            Environment specification.
        info
            Optional metadata.
        """
        self.base_model.social_update(state=state, social=social, spec=spec, info=info)

    def action_probs(self, *, state: Any, spec: Any) -> np.ndarray:
        """Return action probabilities from the base model.

        Parameters
        ----------
        state
            Current state identifier.
        spec
            Environment specification.

        Returns
        -------
        numpy.ndarray
            Action probability vector.
        """
        return self.base_model.action_probs(state=state, spec=spec)

    def update(self, *, state: Any, action: int, outcome: float | None, spec: Any, info: Any | None = None) -> None:
        """Delegate learning update to the base model.

        Parameters
        ----------
        state
            Current state identifier.
        action
            Action index taken by the agent.
        outcome
            Observed outcome (possibly ``None`` if unobserved).
        spec
            Environment specification.
        info
            Optional metadata.
        """
        self.base_model.update(state=state, action=action, outcome=outcome, spec=spec, info=info)


def wrap_model_with_shared_delta_conditions(
    *,
    model: ComputationalModel,
    conditions: Sequence[str],
    baseline_condition: str,
) -> ComputationalModel:
    """Return a condition-aware wrapper around ``model``.

    Parameters
    ----------
    model
        Base computational model to wrap.
    conditions
        Condition labels in the study (e.g., ``["A", "B"]``).
    baseline_condition
        Baseline condition label (no deltas for this condition).

    Returns
    -------
    ComputationalModel
        A :class:`ConditionedSharedDeltaModel` or
        :class:`ConditionedSharedDeltaSocialModel`, preserving whether ``model``
        is social.

    Examples
    --------
    >>> from comp_model_impl.models import QRL
    >>> wrapped = wrap_model_with_shared_delta_conditions(
    ...     model=QRL(),
    ...     conditions=["A", "B"],
    ...     baseline_condition="A",
    ... )
    >>> type(wrapped).__name__
    'ConditionedSharedDeltaModel'
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


# ----------------------------
# Public helpers (for recovery)
# ----------------------------

def constrained_params_by_condition_from_z(
    wrapper: ConditionedSharedDeltaModel | ConditionedSharedDeltaSocialModel,
    params_z: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Derive constrained base-model parameters per condition from wrapper z-params.

    This is a *pure* helper: it does not mutate the wrapper or its base model.

    Parameters
    ----------
    wrapper
        Conditioned wrapper defining the base schema and conditions.
    params_z
        Mapping of wrapper z-parameters.

    Returns
    -------
    dict[str, dict[str, float]]
        Constrained base-model parameters by condition.

    Examples
    --------
    >>> from comp_model_impl.models import QRL
    >>> wrapper = wrap_model_with_shared_delta_conditions(
    ...     model=QRL(),
    ...     conditions=["A", "B"],
    ...     baseline_condition="A",
    ... )
    >>> out = constrained_params_by_condition_from_z(
    ...     wrapper,
    ...     {
    ...         "alpha__shared_z": 0.0,
    ...         "beta__shared_z": 0.0,
    ...         "alpha__delta_z__B": 0.0,
    ...         "beta__delta_z__B": 0.0,
    ...     },
    ... )
    >>> sorted(out.keys())
    ['A', 'B']
    """
    validated = wrapper.param_schema.validate(params_z, strict=True, check_bounds=False)
    z_shared, z_delta = _z_vectors_from_params(
        base_schema=wrapper.base_model.param_schema,
        params=validated,
        conditions=wrapper.conditions,
        baseline_condition=wrapper.baseline_condition,
    )
    return {
        str(c): _condition_params_from_z(
            base_schema=wrapper.base_model.param_schema,
            z_shared=z_shared,
            z_delta_by_condition=z_delta,
            condition=str(c),
        )
        for c in wrapper.conditions
    }


def flatten_params_by_condition(params_by_condition: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Flatten nested parameter dicts into ``{param__cond: value}``.

    Parameters
    ----------
    params_by_condition
        Mapping from condition to base-model parameter dict.

    Returns
    -------
    dict[str, float]
        Flat mapping with keys like ``alpha__A``.

    Examples
    --------
    >>> flatten_params_by_condition({"A": {"alpha": 0.2}, "B": {"alpha": 0.5}})
    {'alpha__A': 0.2, 'alpha__B': 0.5}
    """
    out: dict[str, float] = {}
    for cond, d in params_by_condition.items():
        for p, v in d.items():
            out[f"{p}__{cond}"] = float(v)
    return out
