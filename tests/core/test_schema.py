import numpy as np
import pytest

from comp_model_core.errors import ParameterValidationError
from comp_model_core.params.bounds import Bound
from comp_model_core.params.schema import ParamDef, ParameterSchema
from comp_model_core.params.transforms import (
    Identity,
    LowerBoundedSoftplus,
    UpperBoundedSoftplus,
)


def test_schema_names_defaults():
    schema = ParameterSchema(
        params=(
            ParamDef("alpha", default=0.2, bound=Bound(0.0, 1.0)),
            ParamDef("beta", default=2.0, bound=Bound(0.1, 10.0)),
        )
    )
    assert schema.names == ("alpha", "beta")
    assert schema.defaults() == {"alpha": 0.2, "beta": 2.0}


def test_schema_validate_strict_unknown_raises():
    schema = ParameterSchema(params=(ParamDef("alpha", default=0.2),))
    with pytest.raises(ParameterValidationError):
        schema.validate({"alpha": 0.1, "unknown": 1.0}, strict=True)


def test_schema_validate_nonstrict_ignores_unknown():
    schema = ParameterSchema(params=(ParamDef("alpha", default=0.2),))
    out = schema.validate({"alpha": 0.1, "unknown": 1.0}, strict=False)
    assert out == {"alpha": 0.1}


def test_schema_validate_bounds_check():
    schema = ParameterSchema(params=(ParamDef("alpha", default=0.2, bound=Bound(0.0, 1.0)),))
    # ok
    schema.validate({"alpha": 0.7}, check_bounds=True)

    # out of bounds
    with pytest.raises(ParameterValidationError):
        schema.validate({"alpha": 1.2}, check_bounds=True)


def test_schema_bounds_space_errors():
    schema = ParameterSchema(params=(ParamDef("alpha", default=0.2, bound=Bound(0.0, 1.0)),))

    with pytest.raises(ValueError):
        schema.bounds_space(names=["unknown"])

    schema2 = ParameterSchema(params=(ParamDef("x", default=0.0, bound=None),))
    with pytest.raises(ValueError):
        schema2.bounds_space(require_bounds=True)


def test_schema_z_roundtrip():
    schema = ParameterSchema(
        params=(
            ParamDef("alpha", default=0.2, bound=Bound(0.0, 1.0)),   # inferred Sigmoid
            ParamDef("beta", default=2.0, bound=Bound(0.1, 10.0)),   # inferred BoundedTanh
        )
    )

    params = {"alpha": 0.33, "beta": 7.7}
    z = schema.z_from_params(params)
    assert z.shape == (2,)

    params2 = schema.params_from_z(z)
    assert set(params2.keys()) == {"alpha", "beta"}
    assert np.isclose(params2["alpha"], params["alpha"], rtol=1e-7, atol=1e-7)
    assert np.isclose(params2["beta"], params["beta"], rtol=1e-7, atol=1e-7)


def test_schema_sample_z_init_shapes():
    rng = np.random.default_rng(0)
    schema = ParameterSchema(
        params=(ParamDef("alpha", default=0.2, bound=Bound(0.0, 1.0)),)
    )
    z1 = schema.sample_z_init(rng, center="default", scale=1.0)
    z2 = schema.sample_z_init(rng, center="zero", scale=1.0)
    assert z1.shape == (1,)
    assert z2.shape == (1,)

    with pytest.raises(ValueError):
        schema.sample_z_init(rng, center="nope")


def test_schema_infers_semi_infinite_and_unbounded_transforms():
    schema = ParameterSchema(
        params=(
            ParamDef("lo_only", default=1.0, bound=Bound(0.7, np.inf)),
            ParamDef("hi_only", default=-1.0, bound=Bound(-np.inf, 2.3)),
            ParamDef("free", default=0.0, bound=Bound(-np.inf, np.inf)),
        )
    )

    t_lo, t_hi, t_free = schema.transforms()
    assert isinstance(t_lo, LowerBoundedSoftplus)
    assert isinstance(t_hi, UpperBoundedSoftplus)
    assert isinstance(t_free, Identity)

    params = {"lo_only": 2.1, "hi_only": -0.4, "free": 3.3}
    z = schema.z_from_params(params)
    params2 = schema.params_from_z(z)
    assert np.isclose(params2["lo_only"], params["lo_only"], rtol=1e-7, atol=1e-7)
    assert np.isclose(params2["hi_only"], params["hi_only"], rtol=1e-7, atol=1e-7)
    assert np.isclose(params2["free"], params["free"], rtol=1e-7, atol=1e-7)
