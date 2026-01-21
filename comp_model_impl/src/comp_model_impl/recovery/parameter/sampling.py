from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from .config import DistSpec, SamplingSpec
from comp_model_core.interfaces.model import ComputationalModel


# ----------------------------
# scipy.stats distribution helpers
# ----------------------------

def _dist_from_spec(spec: DistSpec) -> Any:
    name = spec.name.lower()
    if name in ("normal", "gaussian"):
        name = "norm"

    if not hasattr(stats, name):
        raise ValueError(f"Unknown scipy.stats distribution: {spec.name!r}")

    dist = getattr(stats, name)
    # Prefer a frozen distribution when possible
    try:
        return dist(**spec.args)
    except TypeError:
        return dist


def _rvs(dist_obj: Any, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    return np.asarray(dist_obj.rvs(size=size, random_state=rng), dtype=float)


# ----------------------------
# bounds/schema helpers
# ----------------------------

def _param_names_from_model(model: ComputationalModel) -> list[str]:
    names = list(getattr(model, "param_names", []) or [])
    if names:
        return names

    schema = getattr(model, "param_schema", None)
    if schema is not None:
        return list(getattr(schema, "names", []) or [])

    raise ValueError("Could not determine model parameter names (model.param_names or model.param_schema.names).")


def _param_bounds_from_model(model: ComputationalModel) -> dict[str, tuple[float, float]]:
    """
    Try to infer bounds from model.param_schema (if present).
    Returns {} if unavailable.
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
    out = dict(params)
    for k, v in list(out.items()):
        if k in bounds:
            lo, hi = bounds[k]
            out[k] = float(min(max(float(v), lo), hi))
    return out


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
    """
    Returns:
      subj_params: subject_id -> {param: value}
      pop_params: {param: value} or None
    """
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
