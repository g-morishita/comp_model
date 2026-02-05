"""Sampling utilities for parameter recovery.

This module provides helpers for sampling subject parameters from
``scipy.stats`` distributions and mapping them into a model's parameter space.
It supports independent, hierarchical, and fixed sampling modes, as well as
sampling in constrained parameter space or unconstrained ``z`` space.

Examples
--------
Sample fixed parameters for two subjects:

>>> import numpy as np
>>> from comp_model_impl.models.qrl.qrl import QRL
>>> from comp_model_impl.recovery.parameter.config import SamplingSpec
>>> cfg = SamplingSpec(
...     mode="fixed",
...     fixed={"alpha": 0.2, "beta": 3.0},
... )
>>> subj_params, pop_params = sample_subject_params(
...     cfg=cfg,
...     model=QRL(),
...     subject_ids=["s1", "s2"],
...     rng=np.random.default_rng(0),
... )
>>> pop_params is None
True
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from .config import ConditionSamplingSpec, DistSpec, SamplingSpec
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_impl.models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)


# ----------------------------
# scipy.stats distribution helpers
# ----------------------------

def _dist_from_spec(spec: DistSpec) -> Any:
    """Create a scipy.stats distribution (frozen when possible).

    Parameters
    ----------
    spec : DistSpec
        Distribution specification (name + args).

    Returns
    -------
    Any
        A frozen scipy.stats distribution when possible, otherwise the
        distribution constructor.

    Raises
    ------
    ValueError
        If the distribution name is unknown.

    Examples
    --------
    >>> from comp_model_impl.recovery.parameter.config import DistSpec
    >>> dist = _dist_from_spec(DistSpec(name="normal", args={"loc": 0.0, "scale": 1.0}))
    >>> hasattr(dist, "rvs")
    True
    """
    name = spec.name.lower()
    if name in ("normal", "gaussian"):
        name = "norm"
    if name in ("constant", "const", "fixed"):
        if "value" not in spec.args:
            raise ValueError("DistSpec(name='constant') requires args={'value': <float>}.")
        value = float(spec.args["value"])

        class _ConstantDist:
            def __init__(self, v: float) -> None:
                self._v = float(v)

            def rvs(self, size: int = 1, random_state: np.random.Generator | None = None) -> np.ndarray:
                return np.full(size, self._v, dtype=float)

        return _ConstantDist(value)

    if not hasattr(stats, name):
        raise ValueError(f"Unknown scipy.stats distribution: {spec.name!r}")

    dist = getattr(stats, name)
    # Prefer a frozen distribution when possible
    try:
        return dist(**spec.args)
    except TypeError:
        return dist


def _rvs(dist_obj: Any, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Draw random samples from a scipy.stats distribution.

    Parameters
    ----------
    dist_obj : Any
        scipy.stats distribution or frozen distribution.
    rng : numpy.random.Generator
        RNG used for sampling.
    size : int, optional
        Number of samples to draw.

    Returns
    -------
    numpy.ndarray
        Array of drawn samples (dtype float).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng(0)
    >>> samples = _rvs(stats.norm(loc=0.0, scale=1.0), rng, size=2)
    >>> samples.shape
    (2,)
    """
    return np.asarray(dist_obj.rvs(size=size, random_state=rng), dtype=float)


# ----------------------------
# bounds/schema helpers
# ----------------------------

def _param_names_from_model(model: ComputationalModel) -> list[str]:
    """Return parameter names from a model.

    Parameters
    ----------
    model : ComputationalModel
        Model with either ``param_names`` or ``param_schema``.

    Returns
    -------
    list[str]
        Parameter names in model order.

    Raises
    ------
    ValueError
        If parameter names cannot be determined.
    """
    names = list(getattr(model, "param_names", []) or [])
    if names:
        return names

    schema = getattr(model, "param_schema", None)
    if schema is not None:
        return list(getattr(schema, "names", []) or [])

    raise ValueError("Could not determine model parameter names (model.param_names or model.param_schema.names).")


def _param_bounds_from_model(model: ComputationalModel) -> dict[str, tuple[float, float]]:
    """Infer parameter bounds from ``model.param_schema`` if available.

    Parameters
    ----------
    model : ComputationalModel
        Model providing a parameter schema.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from parameter name to (lo, hi) bounds. Returns an empty dict
        if bounds are unavailable.
    """
    bounds: dict[str, tuple[float, float]] = {}
    schema = getattr(model, "param_schema", None)
    if schema is None:
        return bounds

    params = getattr(schema, "params", None)
    if not params:
        return bounds

    for p in params:
        name = getattr(p, "name", None)
        b = getattr(p, "bound", None)
        if name is None or b is None:
            continue
        lo = float(getattr(b, "lo"))
        hi = float(getattr(b, "hi"))
        bounds[str(name)] = (lo, hi)

    return bounds


def _clip_params(params: dict[str, float], bounds: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    """Clip parameter values into their bounds.

    Parameters
    ----------
    params : dict[str, float]
        Parameter values to clip.
    bounds : Mapping[str, tuple[float, float]]
        Bounds mapping (lo, hi) for each parameter.

    Returns
    -------
    dict[str, float]
        Clipped parameter values.

    Examples
    --------
    >>> _clip_params({"alpha": 2.0}, {"alpha": (0.0, 1.0)})["alpha"]
    1.0
    """
    out = dict(params)
    for k, v in list(out.items()):
        if k in bounds:
            lo, hi = bounds[k]
            out[k] = float(min(max(float(v), lo), hi))
    return out


def _merge_condition_spec(
    base: SamplingSpec,
    override: ConditionSamplingSpec,
) -> ConditionSamplingSpec:
    """Merge top-level sampling settings with per-condition overrides."""
    return ConditionSamplingSpec(
        individual={**base.individual, **override.individual},
        population={**base.population, **override.population},
        individual_sd={**base.individual_sd, **override.individual_sd},
        fixed={**base.fixed, **override.fixed},
    )


def _shared_delta_params_from_z_by_condition(
    *,
    z_by_condition: Mapping[str, np.ndarray],
    param_names: Sequence[str],
    baseline_condition: str,
    conditions: Sequence[str],
) -> dict[str, float]:
    """Build shared+delta parameter mapping from per-condition z vectors."""
    baseline = str(baseline_condition)
    if baseline not in z_by_condition:
        raise ValueError(f"Baseline condition {baseline!r} missing in z_by_condition.")

    z_shared = z_by_condition[baseline]
    out: dict[str, float] = {}
    for i, name in enumerate(param_names):
        out[f"{name}__shared_z"] = float(z_shared[i])

    for cond in conditions:
        if cond == baseline:
            continue
        z_cond = z_by_condition[cond]
        z_delta = z_cond - z_shared
        for i, name in enumerate(param_names):
            out[f"{name}__delta_z__{cond}"] = float(z_delta[i])

    return out


def _sample_subject_params_by_condition(
    *,
    cfg: SamplingSpec,
    model: ComputationalModel,
    subject_ids: Sequence[str],
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, float]], dict[str, float] | None]:
    """Sample subject parameters using per-condition overrides."""
    if not isinstance(model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel)):
        raise ValueError("sampling.by_condition is only supported for within-subject shared+delta models.")

    base_model = model.base_model
    base_schema = getattr(base_model, "param_schema", None)
    if base_schema is None:
        raise ValueError("Base model has no param_schema; cannot use sampling.by_condition.")

    conditions = [str(c) for c in getattr(model, "conditions", []) or []]
    if not conditions:
        raise ValueError("Model has no conditions; cannot use sampling.by_condition.")
    baseline = str(getattr(model, "baseline_condition", ""))
    if baseline not in conditions:
        raise ValueError(f"Baseline condition {baseline!r} not found in model.conditions.")

    extra = [c for c in cfg.by_condition.keys() if c not in conditions]
    if extra:
        raise ValueError(f"sampling.by_condition contains unknown conditions: {sorted(extra)}")

    param_names = [p.name for p in base_schema.params]
    bounds = _param_bounds_from_model(base_model)
    mode = cfg.mode.lower()
    space = cfg.space.lower()

    cond_specs = {c: _merge_condition_spec(cfg, cfg.by_condition.get(c, ConditionSamplingSpec())) for c in conditions}

    if mode == "fixed":
        for cond, spec in cond_specs.items():
            missing = [p for p in param_names if p not in spec.fixed]
            if missing:
                raise ValueError(f"sampling.fixed missing parameters for condition {cond!r}: {missing}")
    elif mode == "independent":
        for cond, spec in cond_specs.items():
            missing = [p for p in param_names if p not in spec.individual]
            if missing:
                raise ValueError(f"sampling.individual missing parameters for condition {cond!r}: {missing}")
    elif mode == "hierarchical":
        for cond, spec in cond_specs.items():
            missing_pop = [p for p in param_names if p not in spec.population]
            if missing_pop:
                raise ValueError(f"sampling.population missing parameters for condition {cond!r}: {missing_pop}")
            missing_sd = [p for p in param_names if p not in spec.individual_sd]
            if missing_sd:
                raise ValueError(f"sampling.individual_sd missing parameters for condition {cond!r}: {missing_sd}")
    else:
        raise ValueError(f"Unknown sampling.mode: {cfg.mode!r}")

    def _z_from_param_dict(theta: dict[str, float]) -> np.ndarray:
        if cfg.clip_to_bounds and bounds:
            theta = _clip_params(theta, bounds)
        return base_schema.z_from_params(theta)

    def _z_from_z_dict(z_vals: dict[str, float]) -> np.ndarray:
        z_vec = np.asarray([float(z_vals[p]) for p in param_names], dtype=float)
        if cfg.clip_to_bounds and bounds:
            theta = base_schema.params_from_z(z_vec)
            theta = _clip_params(theta, bounds)
            z_vec = base_schema.z_from_params(theta)
        return z_vec

    subj_params: dict[str, dict[str, float]] = {}

    if mode == "fixed":
        z_by_condition = {}
        for cond, spec in cond_specs.items():
            if space == "param":
                z_by_condition[cond] = _z_from_param_dict(
                    {p: float(spec.fixed[p]) for p in param_names}
                )
            else:
                z_by_condition[cond] = _z_from_z_dict(
                    {p: float(spec.fixed[p]) for p in param_names}
                )

        base_params = _shared_delta_params_from_z_by_condition(
            z_by_condition=z_by_condition,
            param_names=param_names,
            baseline_condition=baseline,
            conditions=conditions,
        )
        for sid in subject_ids:
            subj_params[sid] = dict(base_params)
        return subj_params, None

    if mode == "independent":
        for sid in subject_ids:
            z_by_condition: dict[str, np.ndarray] = {}
            for cond, spec in cond_specs.items():
                if space == "param":
                    theta = {}
                    for p in param_names:
                        dist = _dist_from_spec(spec.individual[p])
                        theta[p] = float(_rvs(dist, rng, size=1)[0])
                    z_by_condition[cond] = _z_from_param_dict(theta)
                else:
                    z_vals = {}
                    for p in param_names:
                        dist = _dist_from_spec(spec.individual[p])
                        z_vals[p] = float(_rvs(dist, rng, size=1)[0])
                    z_by_condition[cond] = _z_from_z_dict(z_vals)

            subj_params[sid] = _shared_delta_params_from_z_by_condition(
                z_by_condition=z_by_condition,
                param_names=param_names,
                baseline_condition=baseline,
                conditions=conditions,
            )

        return subj_params, None

    # hierarchical
    pop_z_by_condition: dict[str, np.ndarray] = {}
    pop_theta_by_condition: dict[str, dict[str, float]] = {}
    for cond, spec in cond_specs.items():
        if space == "param":
            pop_theta: dict[str, float] = {}
            for p in param_names:
                dist = _dist_from_spec(spec.population[p])
                pop_theta[p] = float(_rvs(dist, rng, size=1)[0])
            pop_theta_by_condition[cond] = pop_theta
            pop_z_by_condition[cond] = base_schema.z_from_params(pop_theta)
        else:
            z_vals: dict[str, float] = {}
            for p in param_names:
                dist = _dist_from_spec(spec.population[p])
                z_vals[p] = float(_rvs(dist, rng, size=1)[0])
            pop_z_by_condition[cond] = np.asarray([float(z_vals[p]) for p in param_names], dtype=float)

    for sid in subject_ids:
        z_by_condition: dict[str, np.ndarray] = {}
        for cond, spec in cond_specs.items():
            if space == "param":
                pop_theta = pop_theta_by_condition[cond]
                theta = {}
                for p in param_names:
                    theta[p] = float(rng.normal(loc=pop_theta[p], scale=float(spec.individual_sd[p])))
                z_by_condition[cond] = _z_from_param_dict(theta)
            else:
                z_vec = pop_z_by_condition[cond].copy()
                for i, p in enumerate(param_names):
                    z_vec[i] += float(rng.normal(loc=0.0, scale=float(spec.individual_sd[p])))
                if cfg.clip_to_bounds and bounds:
                    theta = base_schema.params_from_z(z_vec)
                    theta = _clip_params(theta, bounds)
                    z_vec = base_schema.z_from_params(theta)
                z_by_condition[cond] = z_vec

        subj_params[sid] = _shared_delta_params_from_z_by_condition(
            z_by_condition=z_by_condition,
            param_names=param_names,
            baseline_condition=baseline,
            conditions=conditions,
        )

    pop_params = _shared_delta_params_from_z_by_condition(
        z_by_condition=pop_z_by_condition,
        param_names=param_names,
        baseline_condition=baseline,
        conditions=conditions,
    )
    return subj_params, pop_params


# ----------------------------
# main sampler
# ----------------------------

def sample_subject_params(
    *,
    cfg: SamplingSpec,
    model: ComputationalModel,
    subject_ids: Sequence[str],
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, float]], dict[str, float] | None]:
    """Sample true parameters for each subject.

    Parameters
    ----------
    cfg : SamplingSpec
        Sampling configuration (mode, space, distributions).
    model : ComputationalModel
        Model whose parameter schema defines names and transforms.
    subject_ids : Sequence[str]
        Subject identifiers to sample.
    rng : numpy.random.Generator
        RNG used for sampling.

    Returns
    -------
    tuple
        ``(subj_params, pop_params)`` where:

        - ``subj_params`` maps ``subject_id -> {param: value}``
        - ``pop_params`` is a population-level dict (hierarchical) or ``None``

    Raises
    ------
    ValueError
        For unknown modes, missing distributions, or missing schema support for
        ``space="z"``.

    Examples
    --------
    Independent sampling in parameter space:

    >>> import numpy as np
    >>> from comp_model_impl.models.qrl.qrl import QRL
    >>> from comp_model_impl.recovery.parameter.config import SamplingSpec, DistSpec
    >>> cfg = SamplingSpec(
    ...     mode="independent",
    ...     space="param",
    ...     individual={
    ...         "alpha": DistSpec(name="norm", args={"loc": 0.2, "scale": 0.05}),
    ...         "beta": DistSpec(name="norm", args={"loc": 3.0, "scale": 0.5}),
    ...     },
    ... )
    >>> subj_params, pop_params = sample_subject_params(
    ...     cfg=cfg,
    ...     model=QRL(),
    ...     subject_ids=["s1"],
    ...     rng=np.random.default_rng(0),
    ... )
    >>> pop_params is None
    True
    """
    if cfg.by_condition:
        return _sample_subject_params_by_condition(
            cfg=cfg,
            model=model,
            subject_ids=subject_ids,
            rng=rng,
        )

    param_names = _param_names_from_model(model)
    bounds = _param_bounds_from_model(model)

    mode = cfg.mode.lower()
    space = cfg.space.lower()
    schema = getattr(model, "param_schema", None)

    # ---- fixed ----
    if mode == "fixed":
        missing = [p for p in param_names if p not in cfg.fixed]
        if missing:
            raise ValueError(f"sampling.fixed missing parameters: {missing}")
        subj_params = {sid: {p: float(cfg.fixed[p]) for p in param_names} for sid in subject_ids}
        if cfg.clip_to_bounds and bounds:
            subj_params = {sid: _clip_params(d, bounds) for sid, d in subj_params.items()}
        return subj_params, None

    # ---- independent ----
    if mode == "independent":
        if not cfg.individual:
            raise ValueError("sampling.mode=independent requires sampling.individual distributions.")
        missing = [p for p in param_names if p not in cfg.individual]
        if missing:
            raise ValueError(f"sampling.individual missing parameters: {missing}")

        subj_params: dict[str, dict[str, float]] = {}
        for sid in subject_ids:
            if space == "z":
                if schema is None or not hasattr(schema, "params_from_z"):
                    raise ValueError("sampling.space=z requires model.param_schema with params_from_z(z).")
                z = []
                # IMPORTANT: draw in schema.params order (not param_names) so z aligns with transforms
                for p in getattr(schema, "params"):
                    ds = cfg.individual[p.name]
                    dist = _dist_from_spec(ds)
                    z.append(float(_rvs(dist, rng, size=1)[0]))
                z_vec = np.asarray(z, dtype=float)
                theta = dict(schema.params_from_z(z_vec))  # type: ignore[arg-type]
            else:
                theta = {}
                for p in param_names:
                    ds = cfg.individual[p]
                    dist = _dist_from_spec(ds)
                    theta[p] = float(_rvs(dist, rng, size=1)[0])

            if cfg.clip_to_bounds and bounds:
                theta = _clip_params(theta, bounds)

            subj_params[sid] = theta

        return subj_params, None

    # ---- hierarchical ----
    if mode == "hierarchical":
        if not cfg.population:
            raise ValueError("sampling.mode=hierarchical requires sampling.population distributions.")
        if not cfg.individual_sd:
            raise ValueError("sampling.mode=hierarchical requires sampling.individual_sd (per-parameter).")

        missing_pop = [p for p in param_names if p not in cfg.population]
        if missing_pop:
            raise ValueError(f"sampling.population missing parameters: {missing_pop}")
        missing_sd = [p for p in param_names if p not in cfg.individual_sd]
        if missing_sd:
            raise ValueError(f"sampling.individual_sd missing parameters: {missing_sd}")

        if space == "z":
            if schema is None or not hasattr(schema, "params_from_z"):
                raise ValueError("sampling.space=z requires model.param_schema with params_from_z(z).")

            z_pop = []
            for p in getattr(schema, "params"):
                ds = cfg.population[p.name]
                dist = _dist_from_spec(ds)
                z_pop.append(float(_rvs(dist, rng, size=1)[0]))
            z_pop = np.asarray(z_pop, dtype=float)
            pop_theta = dict(schema.params_from_z(z_pop))  # type: ignore[arg-type]

            subj_params: dict[str, dict[str, float]] = {}
            for sid in subject_ids:
                z = z_pop.copy()
                for i, p in enumerate(getattr(schema, "params")):
                    z[i] += float(rng.normal(loc=0.0, scale=float(cfg.individual_sd[p.name])))
                theta = dict(schema.params_from_z(z))  # type: ignore[arg-type]
                if cfg.clip_to_bounds and bounds:
                    theta = _clip_params(theta, bounds)
                subj_params[sid] = theta

            return subj_params, pop_theta

        # parameter-space hierarchical
        pop_theta: dict[str, float] = {}
        for p in param_names:
            ds = cfg.population[p]
            dist = _dist_from_spec(ds)
            pop_theta[p] = float(_rvs(dist, rng, size=1)[0])

        subj_params: dict[str, dict[str, float]] = {}
        for sid in subject_ids:
            theta = {}
            for p in param_names:
                theta[p] = float(rng.normal(loc=pop_theta[p], scale=float(cfg.individual_sd[p])))
            if cfg.clip_to_bounds and bounds:
                theta = _clip_params(theta, bounds)
            subj_params[sid] = theta

        return subj_params, pop_theta

    raise ValueError(f"Unknown sampling.mode: {cfg.mode!r}")
