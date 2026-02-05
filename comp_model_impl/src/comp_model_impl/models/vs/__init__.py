"""Value Shaping (VS) model package.
"""

from .vs import VS
from .schema import vs_schema
from .bounds import vs_bounds_space

__all__ = ["VS", "vs_schema", "vs_bounds_space"]
