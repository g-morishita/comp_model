"""
Custom exceptions used throughout :mod:`comp_model_core`.

The library distinguishes between:

- *Compatibility* issues (e.g., a model cannot be applied to a given task spec).
- *Parameter validation* issues (e.g., invalid types or out-of-bounds values).
"""

from __future__ import annotations


class CompatibilityError(ValueError):
    """
    Raised when components are incompatible.

    Examples include:

    - Applying a model to a :class:`~comp_model_core.spec.TaskSpec` it does not support.
    - Using a social model with an asocial task (or vice versa).

    This exception subclasses :class:`ValueError` so it integrates naturally with
    argument validation patterns.
    """


class ParameterValidationError(ValueError):
    """
    Raised when a parameter value fails validation.

    Typical causes:

    - Non-numeric values where floats are required.
    - Non-finite values (NaN, inf).
    - Values outside declared bounds when bounds checking is requested.
    """
