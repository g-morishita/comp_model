"""Tests for parameter recovery sampling utilities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.models.within_subject_shared_delta import ConditionedSharedDeltaModel
from comp_model_impl.recovery.parameter.config import ConditionSamplingSpec, DistSpec, SamplingSpec
from comp_model_impl.recovery.parameter.sampling import (
    _clip_params,
    _dist_from_spec,
    _param_bounds_from_model,
    _param_names_from_model,
    _rvs,
    sample_subject_params,
)


def test_dist_from_spec_alias_and_unknown() -> None:
    """Distribution specs should accept aliases and reject unknown names."""
    dist = _dist_from_spec(DistSpec(name="normal", args={"loc": 0.0, "scale": 1.0}))
    draws = dist.rvs(size=2, random_state=np.random.default_rng(0))
    assert len(draws) == 2

    with pytest.raises(ValueError):
        _ = _dist_from_spec(DistSpec(name="not_a_dist"))


def test_dist_from_spec_constant() -> None:
    """Constant distribution should always return the configured value."""
    dist = _dist_from_spec(DistSpec(name="constant", args={"value": 0.2}))
    draws = dist.rvs(size=3, random_state=np.random.default_rng(0))
    assert np.allclose(draws, 0.2)

    with pytest.raises(ValueError):
        _ = _dist_from_spec(DistSpec(name="constant", args={}))


def test_rvs_returns_float_array() -> None:
    """_rvs should return a float array of requested size."""
    rng = np.random.default_rng(1)
    samples = _rvs(stats.norm(loc=0.0, scale=1.0), rng, size=3)
    assert samples.shape == (3,)
    assert samples.dtype == float


def test_param_helpers_bounds_and_clip() -> None:
    """Parameter helpers should resolve names, bounds, and clipping."""
    model = QRL()
    names = _param_names_from_model(model)
    assert names == ["alpha", "beta"]

    bounds = _param_bounds_from_model(model)
    assert bounds["alpha"] == (0.0, 1.0)
    assert np.isinf(bounds["beta"][1])

    clipped = _clip_params({"alpha": 2.0, "beta": 30.0}, bounds)
    assert clipped["alpha"] == pytest.approx(1.0)
    assert clipped["beta"] == pytest.approx(30.0)


def test_param_names_missing_raises() -> None:
    """_param_names_from_model should error when metadata is unavailable."""
    class NoParams:
        pass

    with pytest.raises(ValueError):
        _ = _param_names_from_model(NoParams())


def test_sample_subject_params_fixed_and_clipped() -> None:
    """Fixed sampling should clip values into bounds when requested."""
    cfg = SamplingSpec(
        mode="fixed",
        fixed={"alpha": 2.0, "beta": 30.0},
        clip_to_bounds=True,
    )
    model = QRL()
    subj_params, pop_params = sample_subject_params(
        cfg=cfg,
        model=model,
        subject_ids=["s1", "s2"],
        rng=np.random.default_rng(0),
    )
    assert pop_params is None
    assert subj_params["s1"]["alpha"] == pytest.approx(1.0)
    assert subj_params["s1"]["beta"] == pytest.approx(30.0)
    assert subj_params["s2"]["alpha"] == pytest.approx(1.0)
    assert subj_params["s2"]["beta"] == pytest.approx(30.0)


def test_sample_subject_params_independent_param_space_is_deterministic() -> None:
    """Independent sampling in param space should follow RNG order."""
    cfg = SamplingSpec(
        mode="independent",
        space="param",
        individual={
            "alpha": DistSpec(name="norm", args={"loc": 0.0, "scale": 1.0}),
            "beta": DistSpec(name="norm", args={"loc": 1.0, "scale": 0.5}),
        },
        clip_to_bounds=False,
    )
    model = QRL()
    subject_ids = ["s1", "s2"]

    rng_expected = np.random.default_rng(0)
    expected: dict[str, dict[str, float]] = {}
    for sid in subject_ids:
        expected[sid] = {}
        for p in model.param_schema.names:
            dist = _dist_from_spec(cfg.individual[p])
            expected[sid][p] = float(_rvs(dist, rng_expected, size=1)[0])

    rng = np.random.default_rng(0)
    subj_params, pop_params = sample_subject_params(
        cfg=cfg,
        model=model,
        subject_ids=subject_ids,
        rng=rng,
    )
    assert pop_params is None
    for sid in subject_ids:
        for p in model.param_schema.names:
            assert subj_params[sid][p] == pytest.approx(expected[sid][p])


def test_sample_subject_params_independent_z_space_uses_schema_order() -> None:
    """Z-space sampling should draw in schema order and transform via params_from_z."""
    cfg = SamplingSpec(
        mode="independent",
        space="z",
        individual={
            "alpha": DistSpec(name="norm", args={"loc": 0.0, "scale": 1.0}),
            "beta": DistSpec(name="norm", args={"loc": 0.0, "scale": 1.0}),
        },
        clip_to_bounds=False,
    )
    model = QRL()
    schema = model.param_schema

    rng_expected = np.random.default_rng(1)
    z = []
    for p in schema.params:
        dist = _dist_from_spec(cfg.individual[p.name])
        z.append(float(_rvs(dist, rng_expected, size=1)[0]))
    expected = schema.params_from_z(np.asarray(z, dtype=float))

    rng = np.random.default_rng(1)
    subj_params, _ = sample_subject_params(
        cfg=cfg,
        model=model,
        subject_ids=["s1"],
        rng=rng,
    )
    for k, v in expected.items():
        assert subj_params["s1"][k] == pytest.approx(v)


def test_sample_subject_params_hierarchical_param_space_sd_zero() -> None:
    """Hierarchical sampling with zero SD should match population values."""
    cfg = SamplingSpec(
        mode="hierarchical",
        space="param",
        population={
            "alpha": DistSpec(name="norm", args={"loc": 0.2, "scale": 0.0}),
            "beta": DistSpec(name="norm", args={"loc": 3.0, "scale": 0.0}),
        },
        individual_sd={"alpha": 0.0, "beta": 0.0},
        clip_to_bounds=False,
    )
    model = QRL()
    subj_params, pop_params = sample_subject_params(
        cfg=cfg,
        model=model,
        subject_ids=["s1", "s2"],
        rng=np.random.default_rng(0),
    )
    assert pop_params is not None
    for sid in subj_params:
        assert subj_params[sid]["alpha"] == pytest.approx(pop_params["alpha"])
        assert subj_params[sid]["beta"] == pytest.approx(pop_params["beta"])


def test_sample_subject_params_hierarchical_z_space_sd_zero() -> None:
    """Hierarchical z-space sampling with zero SD should match population values."""
    cfg = SamplingSpec(
        mode="hierarchical",
        space="z",
        population={
            "alpha": DistSpec(name="norm", args={"loc": 0.0, "scale": 0.0}),
            "beta": DistSpec(name="norm", args={"loc": 0.0, "scale": 0.0}),
        },
        individual_sd={"alpha": 0.0, "beta": 0.0},
        clip_to_bounds=False,
    )
    model = QRL()
    subj_params, pop_params = sample_subject_params(
        cfg=cfg,
        model=model,
        subject_ids=["s1", "s2"],
        rng=np.random.default_rng(0),
    )
    assert pop_params is not None
    for sid in subj_params:
        assert subj_params[sid]["alpha"] == pytest.approx(pop_params["alpha"])
        assert subj_params[sid]["beta"] == pytest.approx(pop_params["beta"])


def test_sample_subject_params_by_condition_fixed_param_space() -> None:
    """Condition-based fixed sampling should map to shared+delta params."""
    base = QRL()
    wrapper = ConditionedSharedDeltaModel(
        base_model=base,
        conditions=["A", "B"],
        baseline_condition="A",
    )
    cfg = SamplingSpec(
        mode="fixed",
        space="param",
        by_condition={
            "A": ConditionSamplingSpec(fixed={"alpha": 0.2, "beta": 2.0}),
            "B": ConditionSamplingSpec(fixed={"alpha": 0.4, "beta": 3.0}),
        },
    )

    subj_params, pop_params = sample_subject_params(
        cfg=cfg,
        model=wrapper,
        subject_ids=["s1"],
        rng=np.random.default_rng(0),
    )
    assert pop_params is None

    wrapper.set_params(subj_params["s1"])
    params_by_condition = wrapper.params_by_condition()
    assert params_by_condition["A"]["alpha"] == pytest.approx(0.2)
    assert params_by_condition["A"]["beta"] == pytest.approx(2.0)
    assert params_by_condition["B"]["alpha"] == pytest.approx(0.4)
    assert params_by_condition["B"]["beta"] == pytest.approx(3.0)


def test_sample_subject_params_by_condition_independent_param_space_constant() -> None:
    """Independent sampling with constants should respect per-condition values."""
    base = QRL()
    wrapper = ConditionedSharedDeltaModel(
        base_model=base,
        conditions=["A", "B"],
        baseline_condition="A",
    )
    cfg = SamplingSpec(
        mode="independent",
        space="param",
        by_condition={
            "A": ConditionSamplingSpec(
                individual={
                    "alpha": DistSpec(name="constant", args={"value": 0.2}),
                    "beta": DistSpec(name="constant", args={"value": 2.0}),
                }
            ),
            "B": ConditionSamplingSpec(
                individual={
                    "alpha": DistSpec(name="constant", args={"value": 0.4}),
                    "beta": DistSpec(name="constant", args={"value": 3.0}),
                }
            ),
        },
    )

    subj_params, _ = sample_subject_params(
        cfg=cfg,
        model=wrapper,
        subject_ids=["s1"],
        rng=np.random.default_rng(0),
    )
    wrapper.set_params(subj_params["s1"])
    params_by_condition = wrapper.params_by_condition()
    assert params_by_condition["A"]["alpha"] == pytest.approx(0.2)
    assert params_by_condition["A"]["beta"] == pytest.approx(2.0)
    assert params_by_condition["B"]["alpha"] == pytest.approx(0.4)
    assert params_by_condition["B"]["beta"] == pytest.approx(3.0)


def test_sample_subject_params_by_condition_independent_z_space_constant() -> None:
    """Independent z-space sampling should map per-condition z values."""
    base = QRL()
    schema = base.param_schema
    wrapper = ConditionedSharedDeltaModel(
        base_model=base,
        conditions=["A", "B"],
        baseline_condition="A",
    )
    cfg = SamplingSpec(
        mode="independent",
        space="z",
        by_condition={
            "A": ConditionSamplingSpec(
                individual={
                    "alpha": DistSpec(name="constant", args={"value": 0.0}),
                    "beta": DistSpec(name="constant", args={"value": 0.0}),
                }
            ),
            "B": ConditionSamplingSpec(
                individual={
                    "alpha": DistSpec(name="constant", args={"value": 1.0}),
                    "beta": DistSpec(name="constant", args={"value": -1.0}),
                }
            ),
        },
    )

    subj_params, _ = sample_subject_params(
        cfg=cfg,
        model=wrapper,
        subject_ids=["s1"],
        rng=np.random.default_rng(0),
    )
    wrapper.set_params(subj_params["s1"])
    params_by_condition = wrapper.params_by_condition()

    param_names = [p.name for p in schema.params]
    z_a = np.asarray([0.0 if p == "alpha" else 0.0 for p in param_names], dtype=float)
    z_b = np.asarray([1.0 if p == "alpha" else -1.0 for p in param_names], dtype=float)
    expected_a = schema.params_from_z(z_a)
    expected_b = schema.params_from_z(z_b)

    assert params_by_condition["A"]["alpha"] == pytest.approx(expected_a["alpha"])
    assert params_by_condition["A"]["beta"] == pytest.approx(expected_a["beta"])
    assert params_by_condition["B"]["alpha"] == pytest.approx(expected_b["alpha"])
    assert params_by_condition["B"]["beta"] == pytest.approx(expected_b["beta"])


def test_sample_subject_params_by_condition_hierarchical_param_space_sd_zero() -> None:
    """Hierarchical per-condition sampling should match population values when SD=0."""
    base = QRL()
    wrapper = ConditionedSharedDeltaModel(
        base_model=base,
        conditions=["A", "B"],
        baseline_condition="A",
    )
    cfg = SamplingSpec(
        mode="hierarchical",
        space="param",
        by_condition={
            "A": ConditionSamplingSpec(
                population={
                    "alpha": DistSpec(name="constant", args={"value": 0.2}),
                    "beta": DistSpec(name="constant", args={"value": 2.0}),
                },
                individual_sd={"alpha": 0.0, "beta": 0.0},
            ),
            "B": ConditionSamplingSpec(
                population={
                    "alpha": DistSpec(name="constant", args={"value": 0.4}),
                    "beta": DistSpec(name="constant", args={"value": 3.0}),
                },
                individual_sd={"alpha": 0.0, "beta": 0.0},
            ),
        },
    )

    subj_params, pop_params = sample_subject_params(
        cfg=cfg,
        model=wrapper,
        subject_ids=["s1", "s2"],
        rng=np.random.default_rng(0),
    )
    assert pop_params is not None
    for sid in subj_params:
        wrapper.set_params(subj_params[sid])
        params_by_condition = wrapper.params_by_condition()
        assert params_by_condition["A"]["alpha"] == pytest.approx(0.2)
        assert params_by_condition["A"]["beta"] == pytest.approx(2.0)
        assert params_by_condition["B"]["alpha"] == pytest.approx(0.4)
        assert params_by_condition["B"]["beta"] == pytest.approx(3.0)


def test_sample_subject_params_by_condition_hierarchical_z_space_sd_zero() -> None:
    """Hierarchical z-space sampling should map per-condition z values when SD=0."""
    base = QRL()
    schema = base.param_schema
    wrapper = ConditionedSharedDeltaModel(
        base_model=base,
        conditions=["A", "B"],
        baseline_condition="A",
    )
    cfg = SamplingSpec(
        mode="hierarchical",
        space="z",
        by_condition={
            "A": ConditionSamplingSpec(
                population={
                    "alpha": DistSpec(name="constant", args={"value": 0.0}),
                    "beta": DistSpec(name="constant", args={"value": 0.0}),
                },
                individual_sd={"alpha": 0.0, "beta": 0.0},
            ),
            "B": ConditionSamplingSpec(
                population={
                    "alpha": DistSpec(name="constant", args={"value": 1.0}),
                    "beta": DistSpec(name="constant", args={"value": -1.0}),
                },
                individual_sd={"alpha": 0.0, "beta": 0.0},
            ),
        },
    )

    subj_params, pop_params = sample_subject_params(
        cfg=cfg,
        model=wrapper,
        subject_ids=["s1"],
        rng=np.random.default_rng(0),
    )
    assert pop_params is not None

    wrapper.set_params(subj_params["s1"])
    params_by_condition = wrapper.params_by_condition()

    param_names = [p.name for p in schema.params]
    z_a = np.asarray([0.0 if p == "alpha" else 0.0 for p in param_names], dtype=float)
    z_b = np.asarray([1.0 if p == "alpha" else -1.0 for p in param_names], dtype=float)
    expected_a = schema.params_from_z(z_a)
    expected_b = schema.params_from_z(z_b)

    assert params_by_condition["A"]["alpha"] == pytest.approx(expected_a["alpha"])
    assert params_by_condition["A"]["beta"] == pytest.approx(expected_a["beta"])
    assert params_by_condition["B"]["alpha"] == pytest.approx(expected_b["alpha"])
    assert params_by_condition["B"]["beta"] == pytest.approx(expected_b["beta"])


def test_sample_subject_params_by_condition_requires_wrapper() -> None:
    """Condition-based sampling should error for non-wrapper models."""
    cfg = SamplingSpec(
        mode="fixed",
        space="param",
        by_condition={
            "A": ConditionSamplingSpec(fixed={"alpha": 0.2, "beta": 2.0}),
        },
    )
    with pytest.raises(ValueError):
        _ = sample_subject_params(
            cfg=cfg,
            model=QRL(),
            subject_ids=["s1"],
            rng=np.random.default_rng(0),
        )
