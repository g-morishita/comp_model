"""
Parameter schema definitions.

A :class:`ParameterSchema` declares the parameters used by a model, including their
defaults, optional bounds, and the transform used to map between constrained model
space and an unconstrained optimization space.

This module is designed to prevent common "human error" bugs by making parameter
handling explicit and centralized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import ParameterValidationError
from .bounds import Bound, ParameterBoundsSpace
from .transforms import (
    Transform,
    Identity,
    Sigmoid,
    Softplus,
    LowerBoundedSoftplus,
    UpperBoundedSoftplus,
    BoundedTanh,
)


@dataclass(frozen=True, slots=True)
class ParamDef:
    """
    Definition of a single model parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    default : float
        Default value used for initialization.
    bound : Bound or None, optional
        Optional box bound. If provided, it is used for clipping/validation and for
        inferring a default transform.
    transform : Transform or None, optional
        Optional explicit transform. If ``None``, a transform is inferred from
        ``bound`` (see :meth:`transform_obj`).

    Attributes
    ----------
    name : str
    default : float
    bound : Bound or None
    transform : Transform or None
    """

    name: str
    default: float
    bound: Bound | None = None
    transform: Transform | None = None  # if None, inferred

    def _infer_transform(self) -> Transform:
        """
        Infer a reasonable transform based on bounds.

        Returns
        -------
        Transform
            Inferred transform. If ``self.transform`` is provided, it is returned
            unchanged.

        Notes
        -----
        Current inference rules:

        - No bound -> :class:`~comp_model_core.params.transforms.Identity`
        - (0, 1) -> :class:`~comp_model_core.params.transforms.Sigmoid`
        - Finite (lo, hi) -> :class:`~comp_model_core.params.transforms.BoundedTanh`
        - (0, +inf) -> :class:`~comp_model_core.params.transforms.Softplus`
        - (lo, +inf) -> :class:`~comp_model_core.params.transforms.LowerBoundedSoftplus`
        - (-inf, hi) -> :class:`~comp_model_core.params.transforms.UpperBoundedSoftplus`
        - (-inf, +inf) -> :class:`~comp_model_core.params.transforms.Identity`
        """
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

        # Lower-bounded half-line.
        if np.isfinite(lo) and not np.isfinite(hi):
            if lo == 0.0:
                return Softplus()
            return LowerBoundedSoftplus(lo=lo)

        # Upper-bounded half-line.
        if not np.isfinite(lo) and np.isfinite(hi):
            return UpperBoundedSoftplus(hi=hi)

        # Fully unbounded.
        if not np.isfinite(lo) and not np.isfinite(hi):
            return Identity()

        return Identity()

    def coerce(self, value: Any) -> float:
        """
        Coerce an incoming value to a finite float.

        Parameters
        ----------
        value : Any
            Incoming value.

        Returns
        -------
        float
            Coerced finite float.

        Raises
        ------
        ParameterValidationError
            If the value cannot be converted to a finite float.
        """
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
        """
        Validate a value against this parameter's bounds (if any).

        Parameters
        ----------
        x : float
            Parameter value.

        Raises
        ------
        ParameterValidationError
            If bounds are present and ``x`` is outside ``[lo, hi]``.
        """
        if self.bound is None:
            return
        if x < self.bound.lo or x > self.bound.hi:
            raise ParameterValidationError(
                f"Parameter '{self.name}'={x} out of bounds [{self.bound.lo}, {self.bound.hi}]."
            )

    def transform_obj(self) -> Transform:
        """
        Return the transform to be used for this parameter.

        Returns
        -------
        Transform
            Explicit transform if provided; otherwise an inferred transform.
        """
        return self._infer_transform()


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    """
    Schema for a model's parameters.

    Parameters
    ----------
    params : tuple[ParamDef, ...]
        Parameter definitions in a fixed order.

    Attributes
    ----------
    params : tuple[ParamDef, ...]
        Parameter definitions.
    """

    params: tuple[ParamDef, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """
        Parameter names in schema order.

        Returns
        -------
        tuple[str, ...]
            Names in fixed order.
        """
        return tuple(p.name for p in self.params)

    def defaults(self) -> dict[str, float]:
        """
        Default parameter values.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to default value.
        """
        return {p.name: float(p.default) for p in self.params}

    def _by_name(self) -> dict[str, ParamDef]:
        """
        Internal helper mapping parameter name -> definition.

        Returns
        -------
        dict[str, ParamDef]
            Lookup dict.
        """
        return {p.name: p for p in self.params}

    def transforms(self) -> tuple[Transform, ...]:
        """
        Return transforms in schema order.

        Returns
        -------
        tuple[Transform, ...]
            Transform objects, one per parameter.
        """
        return tuple(p.transform_obj() for p in self.params)

    def validate(
        self,
        incoming: Mapping[str, Any],
        *,
        strict: bool = True,
        check_bounds: bool = False,
    ) -> dict[str, float]:
        """
        Validate an incoming parameter mapping.

        Parameters
        ----------
        incoming : Mapping[str, Any]
            Incoming values keyed by parameter name.
        strict : bool, optional
            If ``True``, unknown parameter names raise an error. If ``False``, unknown
            names are ignored.
        check_bounds : bool, optional
            If ``True``, validate values against declared bounds.

        Returns
        -------
        dict[str, float]
            Validated mapping containing only schema parameters present in ``incoming``.

        Raises
        ------
        ParameterValidationError
            If values cannot be coerced to finite floats or violate bounds.
        ValueError
            If unknown names are provided and ``strict=True``.
        """
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
        """
        Build a :class:`~comp_model_core.params.bounds.ParameterBoundsSpace`.

        Parameters
        ----------
        names : Sequence[str] or None, optional
            Subset of parameters to include. If ``None``, uses all schema names.
        require_bounds : bool, optional
            If ``True``, raise an error if any requested parameters lack bounds.

        Returns
        -------
        ParameterBoundsSpace
            Bounds space with vector order matching ``names``.

        Raises
        ------
        ValueError
            If unknown parameter names are requested or bounds are missing.
        """
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
        """
        Map a constrained parameter dict to an unconstrained vector ``z``.

        Parameters
        ----------
        params : Mapping[str, Any]
            Parameter values keyed by name.

        Returns
        -------
        numpy.ndarray
            Unconstrained vector of shape ``(D,)`` in schema order.

        Notes
        -----
        Uses each parameter's transform inverse.
        """
        validated = self.validate(params, strict=True, check_bounds=False)
        z = np.empty((len(self.params),), dtype=float)
        for i, p in enumerate(self.params):
            t = p.transform_obj()
            z[i] = float(t.inverse(float(validated[p.name])))
        return z

    def params_from_z(self, z: np.ndarray) -> dict[str, float]:
        """
        Map an unconstrained vector ``z`` to a constrained parameter dict.

        Parameters
        ----------
        z : numpy.ndarray
            Vector of shape ``(D,)`` where ``D=len(self.params)``.

        Returns
        -------
        dict[str, float]
            Constrained parameter values keyed by name.

        Raises
        ------
        ValueError
            If ``z`` does not have the expected shape.
        """
        if z.shape != (len(self.params),):
            raise ValueError(f"Expected z.shape == ({len(self.params)},), got {z.shape}.")
        out: dict[str, float] = {}
        for i, p in enumerate(self.params):
            t = p.transform_obj()
            out[p.name] = float(t.forward(float(z[i])))
        return out

    def default_z(self) -> np.ndarray:
        """
        Unconstrained vector corresponding to :meth:`defaults`.

        Returns
        -------
        numpy.ndarray
            Unconstrained defaults in schema order.
        """
        return self.z_from_params(self.defaults())

    def sample_z_init(
        self,
        rng: np.random.Generator,
        *,
        center: str = "default",
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Sample a random initialization in ``z``-space.

        Parameters
        ----------
        rng : numpy.random.Generator
            RNG used for sampling.
        center : {"default", "zero"}, optional
            Center of the initialization distribution.
        scale : float, optional
            Standard deviation of the Gaussian noise around ``center``.

        Returns
        -------
        numpy.ndarray
            Random vector in ``z`` space.

        Raises
        ------
        ValueError
            If ``center`` is not one of ``{"default", "zero"}``.
        """
        if center == "default":
            mu = self.default_z()
        elif center == "zero":
            mu = np.zeros((len(self.params),), dtype=float)
        else:
            raise ValueError("center must be 'default' or 'zero'")
        return mu + rng.normal(loc=0.0, scale=float(scale), size=mu.shape)
