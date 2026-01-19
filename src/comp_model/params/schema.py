from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import ParameterValidationError
from .bounds import Bound, ParameterBoundsSpace


@dataclass(frozen=True, slots=True)
class ParamDef:
    """One parameter definition in the model schema."""
    name: str
    default: float
    bound: Bound | None = None

    def coerce(self, value: Any) -> float:
        try:
            x = float(value)
        except Exception as e:  # noqa: BLE001
            raise ParameterValidationError(
                f"Parameter '{self.name}' must be a real number; got {value!r}."
            ) from e
        if not np.isfinite(x):
            raise ParameterValidationError(
                f"Parameter '{self.name}' must be finite; got {value!r}."
            )
        return x

    def validate_bound(self, x: float) -> None:
        if self.bound is None:
            return
        if x < self.bound.lo or x > self.bound.hi:
            raise ParameterValidationError(
                f"Parameter '{self.name}'={x} out of bounds [{self.bound.lo}, {self.bound.hi}]."
            )


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    """
    Single source of truth for model parameters.

    Responsibilities:
    - canonical parameter order (names)
    - defaults
    - validation (unknown names, float/finite checks)
    - optional box bounds (for both validation and building an optimizer space)

    Later, this is where you'd add:
    - transforms (sigmoid/softplus)
    - priors (MAP / hierarchical)
    """

    params: tuple[ParamDef, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def defaults(self) -> dict[str, float]:
        return {p.name: float(p.default) for p in self.params}

    def _by_name(self) -> dict[str, ParamDef]:
        return {p.name: p for p in self.params}

    def validate(
        self,
        incoming: Mapping[str, Any],
        *,
        strict: bool = True,
        check_bounds: bool = False,
    ) -> dict[str, float]:
        """Validate a parameter mapping, returning coerced floats."""
        by = self._by_name()
        out: dict[str, float] = {}

        for k, v in incoming.items():
            p = by.get(k)
            if p is None:
                if strict:
                    known = ", ".join(self.names)
                    raise ParameterValidationError(
                        f"Unknown parameter '{k}'. Known parameters: {known}."
                    )
                continue
            x = p.coerce(v)
            if check_bounds:
                p.validate_bound(x)
            out[k] = x

        return out

    def bounds_space(
        self,
        *,
        names: Sequence[str] | None = None,
        require_bounds: bool = True,
    ) -> ParameterBoundsSpace:
        """Derive a ParameterBoundsSpace from this schema."""
        by = self._by_name()
        if names is None:
            names = self.names

        bounds: dict[str, Bound] = {}
        unknown: list[str] = []
        missing: list[str] = []

        for n in names:
            p = by.get(n)
            if p is None:
                unknown.append(n)
                continue
            if p.bound is None:
                if require_bounds:
                    missing.append(n)
                continue
            bounds[n] = p.bound

        if unknown:
            raise ValueError("Unknown parameter names in bounds_space: " + ", ".join(sorted(unknown)))
        if missing:
            raise ValueError("Missing bounds for parameters: " + ", ".join(sorted(missing)))

        return ParameterBoundsSpace(names=tuple(names), bounds=bounds)
