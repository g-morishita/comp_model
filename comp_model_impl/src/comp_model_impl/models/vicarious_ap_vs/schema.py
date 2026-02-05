"""Parameter schema for the Vicarious AP-VS model.
"""

from __future__ import annotations

from comp_model_core.params import Bound, ParamDef, ParameterSchema, Sigmoid, BoundedTanh


def vicarious_ap_vs_schema(
    *,
    alpha_o_default: float = 0.2,
    alpha_vs_base_default: float = 0.2,
    alpha_a_default: float = 0.2,
    beta_default: float = 3.0,
    kappa_default: float = 0.0,
    beta_max: float = 20.0,
    kappa_abs_max: float = 5.0,
) -> ParameterSchema:

    """
    Construct the Vicarious AP-VS parameter schema.
    
    Parameters
    ----------
    alpha_o_default : float, optional
        Default learning rate of the demonstrator's outcome
    alpha_vs_base_default : float, optional
        Default social value-shape (VS) base learning rate.
    alpha_a_default : float, optional
        Default learning rate of the demonstrator's action
    beta_default : float, optional
        Default inverse temperature.
    kappa_default : float, optional
        Default perseveration parameter.
    beta_max : float, optional
        Maximum allowed beta.
    kappa_abs_max : float, optional
        Maximum absolute kappa.
    
    Returns
    -------
    comp_model_core.params.ParameterSchema
        Schema describing parameter bounds and transforms.
    """

    return ParameterSchema(
        params=(
            ParamDef("alpha_o", float(alpha_o_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_vs_base", float(alpha_vs_base_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_a", float(alpha_a_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("beta", float(beta_default), Bound(1e-6, float(beta_max)),
                     transform=BoundedTanh(1e-6, float(beta_max))),
            ParamDef("kappa", float(kappa_default), Bound(-float(kappa_abs_max), float(kappa_abs_max)),
                     transform=BoundedTanh(-float(kappa_abs_max), float(kappa_abs_max))),
        )
    )
