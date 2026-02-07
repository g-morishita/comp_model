"""Parameter schema for the Vicarious DB + stay model.
"""

from __future__ import annotations

from comp_model_core.params import Bound, ParamDef, ParameterSchema, Sigmoid, BoundedTanh


def vicarious_db_stay_schema(
    *,
    alpha_o_default: float = 0.2,
    demo_bias_default: float = 1.0,
    beta_default: float = 3.0,
    kappa_default: float = 0.0,
    demo_bias_abs_max: float = 5.0,
    beta_max: float = 20.0,
    kappa_abs_max: float = 5.0,
) -> ParameterSchema:

    """
    Construct the Vicarious DB-Stay parameter schema.
    
    Parameters
    ----------
    alpha_o_default : float, optional
        Default learning rate of the demonstrator's outcome
    demo_bias_default : float, optional
        Default demonstrator-choice decision bias (constant; no reliability scaling)
    beta_default : float, optional
        Default inverse temperature.
    kappa_default : float, optional
        Default perseveration parameter.
    demo_bias_abs_max : float, optional
        Maximum absolute value for the demonstrator-choice bias parameter.
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
            ParamDef(
                "demo_bias",
                float(demo_bias_default),
                Bound(-float(demo_bias_abs_max), float(demo_bias_abs_max)),
                transform=BoundedTanh(-float(demo_bias_abs_max), float(demo_bias_abs_max)),
            ),
            ParamDef("beta", float(beta_default), Bound(1e-6, float(beta_max)),
                     transform=BoundedTanh(1e-6, float(beta_max))),
            ParamDef("kappa", float(kappa_default), Bound(-float(kappa_abs_max), float(kappa_abs_max)),
                     transform=BoundedTanh(-float(kappa_abs_max), float(kappa_abs_max))),
        )
    )
