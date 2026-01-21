from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import ParameterValidationError
from .bounds import Bound, ParameterBoundsSpace
from .transforms import Transform, Identity, Sigmoid, Softplus, BoundedTanh


@dataclass(frozen=True, slots=True)
class ParamDef:
    name: str
    default: float
    bound: Bound | None = None
    transform: Transform | None = None  # if None, inferred

    def _infer_transform(self) -> Transform:
        # If user provided a transform, respect it.
        if self.transform is not None:
            return self.transform

        # Otherwise infer a reasonable default.
        if self.bound is None:
            return Identity()

        lo, hi = float(self.bound.lo), float(self.bound.hi)

        # Common special case: (0,1) -> sigmoid
        if lo == 0.0 and hi == 1.0:
            return Sigmoid()

        # Generic finite interval -> bounded tanh
        if np.isfinite(lo) and np.isfinite(hi):
            return BoundedTanh(lo=lo, hi=hi)

        # If someone ever uses semi-infinite bounds, prefer softplus-ish patterns.
        # (Your Bound is currently finite-only, so this is mostly future-proofing.)
        if np.isfinite(lo) and not np.isfinite(hi):
            # (lo, +inf)
            return Softplus()

        return Identity()

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

    def transform_obj(self) -> Transform:
        return self._infer_transform()


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    params: tuple[ParamDef, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def defaults(self) -> dict[str, float]:
        return {p.name: float(p.default) for p in self.params}

    def _by_name(self) -> dict[str, ParamDef]:
        return {p.name: p for p in self.params}

    def transforms(self) -> tuple[Transform, ...]:
        return tuple(p.transform_obj() for p in self.params)

    def validate(
        self,
        incoming: Mapping[str, Any],
        *,
        strict: bool = True,
        check_bounds: bool = False,
    ) -> dict[str, float]:
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

    # --- optimizer helpers (derived) ---

    def bounds_space(
        self,
        *,
        names: Sequence[str] | None = None,
        require_bounds: bool = True,
    ) -> ParameterBoundsSpace:
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

    def z_from_params(self, params: Mapping[str, Any]) -> np.ndarray:
        """Use transforms inverses to map a parameter dict -> unconstrained z vector."""
        validated = self.validate(params, strict=True, check_bounds=False)
        z = np.empty((len(self.params),), dtype=float)
        for i, p in enumerate(self.params):
            t = p.transform_obj()
            z[i] = float(t.inverse(float(validated[p.name])))
        return z

    def params_from_z(self, z: np.ndarray) -> dict[str, float]:
        """Map unconstrained z vector -> constrained parameter dict."""
        if z.shape != (len(self.params),):
            raise ValueError(f"Expected z.shape == ({len(self.params)},), got {z.shape}.")
        out: dict[str, float] = {}
        for i, p in enumerate(self.params):
            t = p.transform_obj()
            out[p.name] = float(t.forward(float(z[i])))
        return out

    def default_z(self) -> np.ndarray:
        """Unconstrained vector corresponding to defaults."""
        return self.z_from_params(self.defaults())

    def sample_z_init(
        self,
        rng: np.random.Generator,
        *,
        center: str = "default",  # "default" or "zero"
        scale: float = 1.0,
    ) -> np.ndarray:
        """Random init in z-space (Gaussian around center)."""
        if center == "default":
            mu = self.default_z()
        elif center == "zero":
            mu = np.zeros((len(self.params),), dtype=float)
        else:
            raise ValueError("center must be 'default' or 'zero'")
        return mu + rng.normal(loc=0.0, scale=float(scale), size=mu.shape)
