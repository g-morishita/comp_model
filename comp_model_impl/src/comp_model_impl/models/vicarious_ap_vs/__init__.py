"""Action-Policy-gated Value Shaping (VS) model package.
"""

from .vicarious_ap_vs import Vicarious_AP_VS
from .schema import vicarious_ap_vs_schema

__all__ = ["Vicarious_AP_VS", "vicarious_ap_vs_schema"]
