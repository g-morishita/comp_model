import numpy as np
import pytest

from comp_model_core.errors import ParameterValidationError
from comp_model_core.params.transforms import (
    Identity,
    Sigmoid,
    Softplus,
    LowerBoundedSoftplus,
    UpperBoundedSoftplus,
    BoundedTanh,
)


def _assert_inverse_roundtrip(transform, zs):
    for z in zs:
        x = transform.forward(z)
        z2 = transform.inverse(x)
        assert np.isfinite(x)
        assert np.isfinite(z2)
        assert np.isclose(z2, z, rtol=1e-6, atol=1e-6)


def test_identity_roundtrip():
    t = Identity()
    zs = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    _assert_inverse_roundtrip(t, zs)


def test_sigmoid_roundtrip():
    t = Sigmoid()
    zs = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])
    _assert_inverse_roundtrip(t, zs)

    # edge values should not explode
    assert np.isfinite(t.inverse(0.0))
    assert np.isfinite(t.inverse(1.0))


def test_softplus_roundtrip():
    t = Softplus()
    zs = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])
    _assert_inverse_roundtrip(t, zs)

    with pytest.raises(ParameterValidationError):
        t.inverse(0.0)

    with pytest.raises(ParameterValidationError):
        t.inverse(-1.0)


def test_bounded_tanh_roundtrip():
    t = BoundedTanh(lo=-3.0, hi=5.0)
    zs = np.array([-6.0, -2.0, 0.0, 2.0, 6.0])
    _assert_inverse_roundtrip(t, zs)

    # forward always in [lo, hi]
    for z in zs:
        x = t.forward(z)
        assert -3.0 <= x <= 5.0


def test_lower_bounded_softplus_roundtrip():
    t = LowerBoundedSoftplus(lo=2.5)
    zs = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])
    _assert_inverse_roundtrip(t, zs)
    for z in zs:
        assert t.forward(z) > 2.5
    with pytest.raises(ParameterValidationError):
        t.inverse(2.5)


def test_upper_bounded_softplus_roundtrip():
    t = UpperBoundedSoftplus(hi=1.2)
    zs = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])
    _assert_inverse_roundtrip(t, zs)
    for z in zs:
        assert t.forward(z) < 1.2
    with pytest.raises(ParameterValidationError):
        t.inverse(1.2)
