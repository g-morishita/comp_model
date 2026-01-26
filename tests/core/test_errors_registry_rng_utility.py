import numpy as np
import pytest

from comp_model_core.errors import CompatibilityError, ParameterValidationError
from comp_model_core.params.bounds import Bound, ParameterBoundsSpace
from comp_model_core.registry import NamedRegistry
from comp_model_core.rng import RNG
from comp_model_core.utility import _as_scipy_bounds, _softmax


def test_custom_errors_are_valueerrors():
    assert issubclass(CompatibilityError, ValueError)
    assert issubclass(ParameterValidationError, ValueError)


def test_rng_wrapper_reproducible():
    r1 = RNG(seed=123).numpy()
    r2 = RNG(seed=123).numpy()
    assert isinstance(r1, np.random.Generator)
    assert r1.random() == r2.random()


def test_named_registry_register_get_names_sorted():
    reg: NamedRegistry[int] = NamedRegistry()
    reg.register("b", 2)
    reg.register("a", 1)
    assert reg.get("a") == 1
    assert reg["b"] == 2
    assert reg.names() == ["a", "b"]

    with pytest.raises(ValueError):
        reg.register("a", 999)

    with pytest.raises(KeyError):
        reg.get("missing")


def test_softmax_numerically_stable_and_sums_to_one():
    # Large utilities shouldn't overflow.
    u = np.array([1000.0, 1000.0], dtype=float)
    p = _softmax(u, beta=1.0)
    assert p.shape == (2,)
    assert np.isfinite(p).all()
    assert np.isclose(p.sum(), 1.0)
    assert np.isclose(p[0], 0.5)
    assert np.isclose(p[1], 0.5)

    # Deterministic as beta->inf: argmax gets prob ~1.
    u2 = np.array([0.0, 1.0], dtype=float)
    p2 = _softmax(u2, beta=1000.0)
    assert np.isclose(p2.sum(), 1.0)
    assert p2[1] > 0.999


def test_as_scipy_bounds_respects_order():
    space = ParameterBoundsSpace(
        names=("beta", "alpha"),
        bounds={"alpha": Bound(0.0, 1.0), "beta": Bound(0.1, 10.0)},
    )
    b = _as_scipy_bounds(space)
    assert b == [(0.1, 10.0), (0.0, 1.0)]
