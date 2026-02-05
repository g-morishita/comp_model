"""Parameter schema for the QRL model.
"""

from __future__ import annotations

from comp_model_core.params import Bound, ParamDef, ParameterSchema, Sigmoid, BoundedTanh


def unidentifiable_qrl_schema(
    *,
    alpha_1_default: float = 0.2,
    alpha_2_default: float = 0.2,
    beta_default: float = 5.0,
    beta_max: float = 20.0,
) -> ParameterSchema:

    """
    Construct the QRL parameter schema.
    
    Parameters
    ----------
    alpha_default : float, optional
        Default learning rate.
    beta_default : float, optional
        Default inverse temperature.
    beta_max : float, optional
        Maximum allowed beta.
    
    Returns
    -------
    comp_model_core.params.ParameterSchema
        Schema describing parameter bounds and transforms.
    """

    return ParameterSchema(
        params=(
            # alpha in (0,1)
            ParamDef("alpha_1", float(alpha_1_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_2", float(alpha_2_default), Bound(0.0, 1.0), transform=Sigmoid()),
            # beta in (1e-6, beta_max) (bounded to avoid crazy temps)
            ParamDef("beta", float(beta_default), Bound(1e-6, float(beta_max)),
                     transform=BoundedTanh(1e-6, float(beta_max))),
        )
    )
