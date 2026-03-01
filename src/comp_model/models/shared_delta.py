"""Within-subject shared+delta model wrappers.

These wrappers provide condition-wise parameter tying via shared and delta
unconstrained parameters. They are generic wrappers around a model factory and
fit the new architecture without requiring a prior parameter-schema system.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from comp_model.core.contracts import AgentModel, DecisionContext


def _identity(value: float) -> float:
    """Return input unchanged as float."""

    return float(value)


@dataclass(frozen=True, slots=True)
class SharedDeltaParameterSpec:
    """Parameter definition for shared+delta transformation.

    Parameters
    ----------
    name : str
        Base parameter name consumed by ``model_factory``.
    transform : Callable[[float], float], optional
        Function mapping unconstrained ``z`` value to constrained parameter
        value passed to the model factory.

    Notes
    -----
    If no transform is provided, identity transform is used.
    """

    name: str
    transform: Callable[[float], float] = _identity


def _dedupe_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return de-duplicated labels preserving first occurrence order."""

    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


@dataclass(slots=True)
class ConditionedSharedDeltaModel:
    """Condition-aware wrapper with shared+delta tied parameters.

    Model Contract
    --------------
    Decision Rule
        For each condition ``c`` and parameter ``p``:
        ``z[p,c] = z_shared[p] + z_delta[p,c]`` (baseline condition uses 0 delta),
        ``theta[p,c] = transform_p(z[p,c])``.
        The active condition model decides actions using ``theta[:, active]``.
    Update Rule
        Every update delegates to the active condition model only.
        Condition can be switched explicitly via :meth:`set_condition` or
        inferred from observation/outcome payload keys.

    Parameters
    ----------
    model_factory : Callable[[dict[str, float]], AgentModel[Any, Any, Any]]
        Factory that creates a fresh base model from constrained parameters.
    parameter_specs : tuple[SharedDeltaParameterSpec, ...]
        Parameter specs defining the shared+delta transform graph.
    conditions : tuple[str, ...]
        Ordered condition labels in the within-subject design.
    baseline_condition : str
        Baseline condition label (delta fixed to zero).
    condition_resolver : Callable[[Any, Any, DecisionContext[Any]], str | None] | None, optional
        Custom function to resolve condition for each decision/update.
    initial_condition : str | None, optional
        Initial active condition. Defaults to baseline condition.

    Notes
    -----
    Automatic condition inference checks payload keys in this order:
    ``"condition"``, ``"block_condition"``, ``"condition_label"``.
    """

    model_factory: Callable[[dict[str, float]], AgentModel[Any, Any, Any]]
    parameter_specs: tuple[SharedDeltaParameterSpec, ...]
    conditions: tuple[str, ...]
    baseline_condition: str
    condition_resolver: Callable[[Any, Any, DecisionContext[Any]], str | None] | None = None
    initial_condition: str | None = None

    _active_condition: str = field(init=False, repr=False)
    _params_z: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _params_by_condition: dict[str, dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    _models_by_condition: dict[str, AgentModel[Any, Any, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate wrapper configuration and initialize active condition."""

        self.conditions = _dedupe_preserve_order(tuple(str(c) for c in self.conditions))
        if not self.conditions:
            raise ValueError("conditions must not be empty")

        self.baseline_condition = str(self.baseline_condition)
        if self.baseline_condition not in self.conditions:
            raise ValueError(
                f"baseline_condition {self.baseline_condition!r} must be one of {self.conditions}"
            )

        if not self.parameter_specs:
            raise ValueError("parameter_specs must not be empty")

        names = [spec.name for spec in self.parameter_specs]
        if len(names) != len(set(names)):
            raise ValueError("parameter_specs names must be unique")

        if self.initial_condition is not None:
            initial = str(self.initial_condition)
            if initial not in self.conditions:
                raise ValueError(
                    f"initial_condition {initial!r} must be one of {self.conditions}"
                )
            self._active_condition = initial
        else:
            self._active_condition = self.baseline_condition

    @property
    def active_condition(self) -> str:
        """Return the current active condition label."""

        return self._active_condition

    def parameter_keys(self) -> tuple[str, ...]:
        """Return expected shared+delta parameter keys in deterministic order."""

        keys: list[str] = []
        for spec in self.parameter_specs:
            keys.append(f"{spec.name}__shared_z")

        for condition in self.conditions:
            if condition == self.baseline_condition:
                continue
            for spec in self.parameter_specs:
                keys.append(f"{spec.name}__delta_z__{condition}")

        return tuple(keys)

    def default_params_z(self) -> dict[str, float]:
        """Return zero defaults for all shared+delta parameters."""

        return {name: 0.0 for name in self.parameter_keys()}

    def set_condition(self, condition: str) -> None:
        """Set active condition explicitly.

        Parameters
        ----------
        condition : str
            Condition label to activate.

        Raises
        ------
        ValueError
            If condition is unknown.
        """

        candidate = str(condition)
        if candidate not in self.conditions:
            raise ValueError(f"unknown condition {candidate!r}; expected one of {self.conditions}")
        self._active_condition = candidate

    def set_params(self, params_z: Mapping[str, float]) -> None:
        """Set shared+delta z-space parameters and rebuild condition models.

        Parameters
        ----------
        params_z : Mapping[str, float]
            Shared+delta unconstrained parameters.

        Raises
        ------
        ValueError
            If required keys are missing or unknown keys are provided.
        """

        required = set(self.parameter_keys())
        provided = set(params_z)
        missing = sorted(required - provided)
        unknown = sorted(provided - required)
        if missing:
            raise ValueError(f"missing shared+delta parameters: {missing}")
        if unknown:
            raise ValueError(f"unknown shared+delta parameters: {unknown}")

        self._params_z = {name: float(params_z[name]) for name in self.parameter_keys()}

        shared: dict[str, float] = {}
        for spec in self.parameter_specs:
            shared[spec.name] = self._params_z[f"{spec.name}__shared_z"]

        self._params_by_condition = {}
        self._models_by_condition = {}

        for condition in self.conditions:
            constrained_params: dict[str, float] = {}
            for spec in self.parameter_specs:
                z_value = shared[spec.name]
                if condition != self.baseline_condition:
                    z_value += self._params_z[f"{spec.name}__delta_z__{condition}"]
                constrained_params[spec.name] = float(spec.transform(float(z_value)))

            self._params_by_condition[condition] = constrained_params
            self._models_by_condition[condition] = self.model_factory(dict(constrained_params))

    def params_by_condition(self) -> dict[str, dict[str, float]]:
        """Return constrained parameter values by condition.

        Returns
        -------
        dict[str, dict[str, float]]
            Constrained parameter mapping for each condition.

        Raises
        ------
        ValueError
            If :meth:`set_params` has not been called.
        """

        if not self._params_by_condition:
            raise ValueError("shared+delta parameters are not set; call set_params() first")
        return {condition: dict(params) for condition, params in self._params_by_condition.items()}

    def start_episode(self) -> None:
        """Start episode for all condition models.

        Raises
        ------
        ValueError
            If condition models have not been initialized via :meth:`set_params`.
        """

        if not self._models_by_condition:
            raise ValueError("condition models are not initialized; call set_params() before start_episode()")

        for model in self._models_by_condition.values():
            model.start_episode()

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> Mapping[Any, float]:
        """Return action distribution from active condition model."""

        self._sync_condition(observation=observation, outcome=None, context=context)
        return self._active_model().action_distribution(observation, context=context)

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Apply update to active condition model only."""

        self._sync_condition(observation=observation, outcome=outcome, context=context)
        self._active_model().update(observation, action, outcome, context=context)

    def _active_model(self) -> AgentModel[Any, Any, Any]:
        """Return model for current active condition."""

        if not self._models_by_condition:
            raise ValueError("condition models are not initialized; call set_params() first")
        return self._models_by_condition[self._active_condition]

    def _sync_condition(
        self,
        *,
        observation: Any,
        outcome: Any,
        context: DecisionContext[Any],
    ) -> None:
        """Resolve and apply condition from runtime payloads when available."""

        resolved = self._resolve_condition(observation=observation, outcome=outcome, context=context)
        if resolved is not None:
            self.set_condition(resolved)

    def _resolve_condition(
        self,
        *,
        observation: Any,
        outcome: Any,
        context: DecisionContext[Any],
    ) -> str | None:
        """Resolve condition using custom resolver and payload conventions."""

        if self.condition_resolver is not None:
            resolved = self.condition_resolver(observation, outcome, context)
            if resolved is not None:
                return str(resolved)

        for payload in (observation, outcome):
            if not isinstance(payload, Mapping):
                continue
            for key in ("condition", "block_condition", "condition_label"):
                if key in payload:
                    return str(payload[key])

        if context.decision_label is not None and context.decision_label in self.conditions:
            return str(context.decision_label)

        return None


@dataclass(slots=True)
class ConditionedSharedDeltaSocialModel(ConditionedSharedDeltaModel):
    """Social-model variant of shared+delta condition wrapper.

    Model Contract
    --------------
    Decision Rule
        Identical to :class:`ConditionedSharedDeltaModel`; decision is delegated
        to the active condition-specific social model.
    Update Rule
        Identical to :class:`ConditionedSharedDeltaModel`; updates only affect
        the currently active condition model.

    Notes
    -----
    This class currently reuses the exact implementation from the asocial
    wrapper because the new architecture's ``AgentModel`` contract is generic
    across social and asocial observations/outcomes.
    """


def create_conditioned_shared_delta_model(
    *,
    model_factory: Callable[[dict[str, float]], AgentModel[Any, Any, Any]],
    parameter_specs: tuple[SharedDeltaParameterSpec, ...],
    conditions: tuple[str, ...],
    baseline_condition: str,
    condition_resolver: Callable[[Any, Any, DecisionContext[Any]], str | None] | None = None,
    initial_condition: str | None = None,
) -> ConditionedSharedDeltaModel:
    """Factory helper for conditioned shared+delta wrapper."""

    return ConditionedSharedDeltaModel(
        model_factory=model_factory,
        parameter_specs=parameter_specs,
        conditions=conditions,
        baseline_condition=baseline_condition,
        condition_resolver=condition_resolver,
        initial_condition=initial_condition,
    )


def create_conditioned_shared_delta_social_model(
    *,
    model_factory: Callable[[dict[str, float]], AgentModel[Any, Any, Any]],
    parameter_specs: tuple[SharedDeltaParameterSpec, ...],
    conditions: tuple[str, ...],
    baseline_condition: str,
    condition_resolver: Callable[[Any, Any, DecisionContext[Any]], str | None] | None = None,
    initial_condition: str | None = None,
) -> ConditionedSharedDeltaSocialModel:
    """Factory helper for conditioned shared+delta social wrapper."""

    return ConditionedSharedDeltaSocialModel(
        model_factory=model_factory,
        parameter_specs=parameter_specs,
        conditions=conditions,
        baseline_condition=baseline_condition,
        condition_resolver=condition_resolver,
        initial_condition=initial_condition,
    )


__all__ = [
    "ConditionedSharedDeltaModel",
    "ConditionedSharedDeltaSocialModel",
    "SharedDeltaParameterSpec",
    "create_conditioned_shared_delta_model",
    "create_conditioned_shared_delta_social_model",
]
