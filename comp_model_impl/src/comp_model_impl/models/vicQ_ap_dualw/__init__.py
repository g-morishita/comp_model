"""Vicarious Value Shaping (Vicarious_VS) model package.
"""

from .vicQ_ap_dualw import VicQ_AP_DualW
from .schema import vicQ_ap_dualw_schema

__all__ = ["VicQ_AP_DualW", "vicQ_ap_dualw_schema"]
