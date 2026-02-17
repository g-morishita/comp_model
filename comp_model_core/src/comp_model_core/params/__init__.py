"""
Parameter schemas, bounds, and transforms.

This subpackage provides utilities for:

- Declaring model parameters with defaults and (optional) box bounds.
- Validating and coercing incoming parameter dictionaries.
- Mapping constrained parameters to/from an unconstrained optimization space.

The central class is :class:`comp_model_core.params.schema.ParameterSchema`.

See Also
--------
comp_model_core.params.schema.ParameterSchema
    Schema definition and validation utilities.
comp_model_core.params.transforms
    Transform implementations for constrained parameters.
"""

from .bounds import Bound, ParameterBoundsSpace
from .schema import ParamDef, ParameterSchema
from .transforms import (
    Transform,
    Identity,
    Sigmoid,
    Softplus,
    LowerBoundedSoftplus,
    UpperBoundedSoftplus,
    BoundedTanh,
)

__all__ = [
    "Bound",
    "ParameterBoundsSpace",
    "ParamDef",
    "ParameterSchema",
    "Transform",
    "Identity",
    "Sigmoid",
    "Softplus",
    "LowerBoundedSoftplus",
    "UpperBoundedSoftplus",
    "BoundedTanh",
]
