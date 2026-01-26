import numpy as np
import pytest

from comp_model_core.params.bounds import Bound, ParameterBoundsSpace


def test_bound_clip():
    b = Bound(lo=0.0, hi=1.0)
    assert b.clip(-1.0) == 0.0
    assert b.clip(0.2) == 0.2
    assert b.clip(2.0) == 1.0


def test_parameter_bounds_space_dim_clip_to_params():
    space = ParameterBoundsSpace(
        names=("a", "b"),
        bounds={"a": Bound(0.0, 1.0), "b": Bound(-2.0, 2.0)},
    )

    assert space.dim == 2

    x = np.array([-1.0, 3.0], dtype=float)
    y = space.clip_vec(x)
    assert y.shape == (2,)
    assert y[0] == 0.0
    assert y[1] == 2.0

    params = space.to_params(x)
    assert params == {"a": 0.0, "b": 2.0}


def test_parameter_bounds_space_clip_vec_shape_error():
    space = ParameterBoundsSpace(
        names=("a",),
        bounds={"a": Bound(0.0, 1.0)},
    )
    with pytest.raises(ValueError):
        space.clip_vec(np.array([0.1, 0.2], dtype=float))


def test_parameter_bounds_space_sample_init_within_bounds():
    rng = np.random.default_rng(0)
    space = ParameterBoundsSpace(
        names=("a", "b", "c"),
        bounds={"a": Bound(0.0, 1.0), "b": Bound(-2.0, 2.0), "c": Bound(5.0, 6.0)},
    )

    for _ in range(100):
        x = space.sample_init(rng)
        assert x.shape == (3,)
        assert 0.0 <= x[0] <= 1.0
        assert -2.0 <= x[1] <= 2.0
        assert 5.0 <= x[2] <= 6.0
