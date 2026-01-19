from __future__ import annotations

class CompatibilityError(ValueError):
    """Raised when a model/estimator is incompatible with a task spec."""

class ParameterValidationError(ValueError):
    """Raised when a parameter is incompatible with a task spec."""