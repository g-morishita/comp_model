"""Vicarious Value Shaping (Vicarious_VS) model package.
"""

from .vicarious_vs_stay import Vicarious_VS_Stay
from .schema import vicarious_vs_stay_schema
from .bounds import vicarious_vs_stay_bounds_space

__all__ = ["Vicarious_VS_Stay", "vicarious_vs_stay_schema", "vicarious_vs_stay_bounds_space"]
