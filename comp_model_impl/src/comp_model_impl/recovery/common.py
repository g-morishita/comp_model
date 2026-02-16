"""Shared helpers for recovery runners."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import inspect
import secrets
import shutil

import numpy as np

from comp_model_core.interfaces.estimator import Estimator
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.plans.block import StudyPlan


def make_unique_run_dir(base_out_dir: str | Path) -> Path:
    """Create a unique run directory under a base output directory."""

    base = Path(base_out_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = secrets.token_hex(2)  # 4 hex chars

    run_id = f"{ts}__{suffix}"
    out_dir = base / run_id
    out_dir.mkdir(parents=False, exist_ok=False)
    return out_dir


def plan_summary(plan: StudyPlan) -> dict[str, Any]:
    """Summarize a study plan for run manifests."""

    n_subjects = int(len(plan.subjects))

    blocks_per_subject: list[int] = []
    trials_per_subject: list[int] = []

    block_trials_by_id: dict[str, list[int]] = {}
    block_seen_by_subject: dict[str, set[str]] = {}

    canonical_subject_id = next(iter(plan.subjects.keys()))
    canonical_blocks = list(plan.subjects[canonical_subject_id])
    canonical_block_ids = [bp.block_id for bp in canonical_blocks]
    canonical_trials_by_block = [int(bp.n_trials) for bp in canonical_blocks]

    for sid, block_plans in plan.subjects.items():
        block_plans = list(block_plans)
        blocks_per_subject.append(int(len(block_plans)))

        tot = 0
        for bp in block_plans:
            ntr = int(bp.n_trials)
            tot += ntr

            block_trials_by_id.setdefault(bp.block_id, []).append(ntr)
            block_seen_by_subject.setdefault(bp.block_id, set()).add(sid)

        trials_per_subject.append(int(tot))

    n_blocks_unique = int(len(block_trials_by_id))

    bmin = int(min(blocks_per_subject)) if blocks_per_subject else 0
    bmax = int(max(blocks_per_subject)) if blocks_per_subject else 0
    bmean = float(np.mean(blocks_per_subject)) if blocks_per_subject else 0.0

    tmin = int(min(trials_per_subject)) if trials_per_subject else 0
    tmax = int(max(trials_per_subject)) if trials_per_subject else 0
    tmean = float(np.mean(trials_per_subject)) if trials_per_subject else 0.0

    blocks_table = []
    for block_id, trials_list in sorted(block_trials_by_id.items(), key=lambda kv: kv[0]):
        blocks_table.append(
            {
                "block_id": block_id,
                "n_trials_min": int(min(trials_list)),
                "n_trials_max": int(max(trials_list)),
                "n_trials_mean": float(np.mean(trials_list)),
                "n_subjects_with_block": int(len(block_seen_by_subject.get(block_id, set()))),
            }
        )

    return {
        "n_subjects": n_subjects,
        "n_blocks_unique": n_blocks_unique,
        "blocks_per_subject_min": bmin,
        "blocks_per_subject_mean": bmean,
        "blocks_per_subject_max": bmax,
        "total_trials_per_subject_min": tmin,
        "total_trials_per_subject_mean": tmean,
        "total_trials_per_subject_max": tmax,
        "total_trials_all_subjects": int(sum(trials_per_subject)),
        "canonical_subject_id": canonical_subject_id,
        "canonical_block_ids": canonical_block_ids,
        "canonical_trials_by_block": canonical_trials_by_block,
        "blocks_table": blocks_table,
    }


def safe_copy_file(src: str | Path, dst_dir: str | Path) -> str | None:
    """Copy a file into a destination directory if the source exists."""

    src = Path(src)
    dst_dir = Path(dst_dir)
    try:
        if src.exists() and src.is_file():
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            return str(dst.name)
    except Exception:
        return None
    return None


def subject_ids_from_plan(plan: StudyPlan) -> list[str]:
    """Extract subject identifiers from a study plan."""

    return [str(sid) for sid in (plan.subjects.keys() if plan.subjects else [])]


def _resolve_registry_component(
    *,
    reference: str,
    registry: Any,
    kind: str,
) -> Any:
    """Resolve a component from a named registry."""

    try:
        return registry.get(reference)
    except KeyError as e:
        available = ", ".join(registry.names())
        if available:
            msg = (
                f"Could not resolve {kind} {reference!r}. "
                f"Use a registered {kind} key. "
                f"Available {kind} keys: {available}."
            )
        else:
            msg = (
                f"Could not resolve {kind} {reference!r}. "
                f"Use a registered {kind} key."
            )
        raise ValueError(msg) from e


def resolve_model_callable(reference: str, *, registries: Any) -> Any:
    """Resolve a model class/factory by registry key."""

    return _resolve_registry_component(
        reference=reference,
        registry=registries.models,
        kind="model",
    )


def resolve_estimator_callable(reference: str, *, registries: Any) -> Any:
    """Resolve an estimator class/factory by registry key."""

    return _resolve_registry_component(
        reference=reference,
        registry=registries.estimators,
        kind="estimator",
    )


def resolve_generator_callable(reference: str, *, registries: Any) -> Any:
    """Resolve a generator class/factory by registry key."""

    return _resolve_registry_component(
        reference=reference,
        registry=registries.generators,
        kind="generator",
    )


def build_nested(value: Any, *, registries: Any) -> Any:
    """Build nested inline factory mappings found in kwargs."""

    if isinstance(value, Mapping) and "factory" in value:
        nested_kwargs_raw = value.get("kwargs", {})
        if nested_kwargs_raw is None:
            nested_kwargs: dict[str, Any] = {}
        elif isinstance(nested_kwargs_raw, Mapping):
            nested_kwargs = {str(k): v for k, v in nested_kwargs_raw.items()}
        else:
            raise TypeError("Inline factory kwargs must be a mapping")
        return build_from_reference(
            reference=str(value["factory"]),
            kwargs=nested_kwargs,
            registries=registries,
            kind="model_or_other",
        )
    return value


def build_kwargs(kwargs: Mapping[str, Any] | None, *, registries: Any) -> dict[str, Any]:
    """Build kwargs recursively for inline factory mappings."""

    if kwargs is None:
        return {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"kwargs must be a mapping, got {type(kwargs)}")

    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        out[str(k)] = build_nested(v, registries=registries)
    return out


def build_from_reference(
    *,
    reference: str,
    kwargs: Mapping[str, Any] | None,
    registries: Any,
    kind: str,
) -> Any:
    """Build an object from a reference string and kwargs."""

    resolved_kwargs = build_kwargs(kwargs, registries=registries)

    if kind in ("model", "model_or_other"):
        cls = resolve_model_callable(reference, registries=registries)
        return cls(**resolved_kwargs)

    if kind == "estimator":
        cls_or_fn = resolve_estimator_callable(reference, registries=registries)
        return cls_or_fn(**resolved_kwargs)

    if kind == "generator":
        cls_or_fn = resolve_generator_callable(reference, registries=registries)
        return cls_or_fn(**resolved_kwargs)

    raise ValueError(f"Unknown kind: {kind!r}")


def build_model(
    model: str,
    *,
    model_kwargs: Mapping[str, Any] | None,
    registries: Any,
) -> ComputationalModel:
    """Build a computational model instance."""

    built = build_from_reference(
        reference=str(model),
        kwargs=model_kwargs,
        registries=registries,
        kind="model",
    )
    if not isinstance(built, ComputationalModel):
        raise TypeError(f"Built object is not a ComputationalModel: {type(built)}")
    return built


def build_estimator(
    estimator: str,
    *,
    estimator_kwargs: Mapping[str, Any] | None,
    registries: Any,
    model: ComputationalModel | None = None,
    inject_model: str = "model",
) -> Estimator:
    """Build an estimator and optionally inject model/base_model."""

    kwargs = build_kwargs(estimator_kwargs, registries=registries)
    cls_or_fn = resolve_estimator_callable(str(estimator), registries=registries)

    if model is not None:
        try:
            sig = inspect.signature(cls_or_fn)
            params = sig.parameters
        except Exception:
            params = None

        inject_mode = str(inject_model).lower()
        if inject_mode not in {"model", "base_model", "auto"}:
            raise ValueError(f"Unknown inject_model mode: {inject_model!r}")

        if inject_mode == "model":
            if (params is None or "model" in params) and "model" not in kwargs:
                kwargs["model"] = model
        elif inject_mode == "base_model":
            if (params is None or "base_model" in params) and "base_model" not in kwargs:
                kwargs["base_model"] = model
        else:
            if params is not None:
                if "base_model" in params and "base_model" not in kwargs:
                    kwargs["base_model"] = model
                elif "model" in params and "model" not in kwargs:
                    kwargs["model"] = model

    built = cls_or_fn(**kwargs)
    if not isinstance(built, Estimator):
        raise TypeError(f"Built object is not an Estimator: {type(built)}")
    return built


def build_generator(
    generator: str,
    *,
    generator_kwargs: Mapping[str, Any] | None,
    registries: Any,
) -> Generator:
    """Build a generator instance from a registry reference."""

    built = build_from_reference(
        reference=str(generator),
        kwargs=generator_kwargs,
        registries=registries,
        kind="generator",
    )
    if not hasattr(built, "simulate_study"):
        raise TypeError(f"Built object does not provide simulate_study: {type(built)}")
    return built


def build_model_checked(
    model: str,
    *,
    model_kwargs: Mapping[str, Any] | None,
    registries: Any,
    label: str = "model",
) -> ComputationalModel:
    """Build a model with context-rich instantiation errors."""

    kwargs_dict = dict(model_kwargs or {})
    try:
        return build_model(
            model,
            model_kwargs=model_kwargs,
            registries=registries,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to instantiate {label} {model!r} "
            f"with kwargs={kwargs_dict!r}: {exc}"
        ) from exc


def build_estimator_checked(
    estimator: str,
    *,
    estimator_kwargs: Mapping[str, Any] | None,
    registries: Any,
    model: ComputationalModel | None = None,
    inject_model: str = "model",
    label: str = "estimator",
) -> Estimator:
    """Build an estimator with context-rich instantiation errors."""

    kwargs_dict = dict(estimator_kwargs or {})
    try:
        return build_estimator(
            estimator,
            estimator_kwargs=estimator_kwargs,
            registries=registries,
            model=model,
            inject_model=inject_model,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to instantiate {label} {estimator!r} "
            f"with kwargs={kwargs_dict!r}: {exc}"
        ) from exc


def build_generator_checked(
    generator: str,
    *,
    generator_kwargs: Mapping[str, Any] | None,
    registries: Any,
    label: str = "generator",
) -> Generator:
    """Build a generator with context-rich instantiation errors."""

    kwargs_dict = dict(generator_kwargs or {})
    try:
        return build_generator(
            generator,
            generator_kwargs=generator_kwargs,
            registries=registries,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to instantiate {label} {generator!r} "
            f"with kwargs={kwargs_dict!r}: {exc}"
        ) from exc
