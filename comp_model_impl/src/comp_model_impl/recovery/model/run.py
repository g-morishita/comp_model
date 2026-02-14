"""End-to-end model recovery runner.

Model recovery simulates datasets from each generating model and fits a set of
candidate models to each dataset, then selects the best candidate using a
model selection criterion (log-likelihood, AIC, BIC).

The runner writes a self-contained run directory with configuration snapshots,
a per-candidate fit table, a winners table, optional diagnostics, and optional
pickled simulated datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import json
import time
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from comp_model_core.plans.io import load_study_plan_yaml, load_study_plan_json
from comp_model_core.plans.block import StudyPlan
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.interfaces.estimator import Estimator, FitResult

from ...register import make_registry
from ...tasks.build import build_runner_for_plan

from ..parameter.run import (
    make_unique_run_dir,
    _plan_summary,
    git_info_for_module,
)
from .config import ModelRecoveryConfig
from .criteria import get_criterion, ModelCriterion
from .likelihood import compute_likelihood_summary
from .resolution import resolve_estimator_callable, resolve_model_callable


@dataclass(frozen=True, slots=True)
class ModelRecoveryOutputs:
    """Outputs from a model recovery run.

    Attributes
    ----------
    fit_table : pandas.DataFrame
        Long table with one row per (rep, generating_model, candidate_model).
    winners : pandas.DataFrame
        One row per (rep, generating_model) with the selected model and score gaps.
    out_dir : str
        Output directory containing artifacts written by the run.
    """
    fit_table: pd.DataFrame
    winners: pd.DataFrame
    out_dir: str


def write_model_recovery_manifest(
    out_dir: Path,
    *,
    config: ModelRecoveryConfig,
    generator: Any,
    plan_path_copied: str | None,
    plan_summary: dict[str, Any],
) -> None:
    """Write the model recovery manifest JSON.

    Parameters
    ----------
    out_dir : pathlib.Path
        Run output directory.
    config : ModelRecoveryConfig
        Model recovery configuration used for this run.
    generator : Any
        Generator instance used for data simulation.
    plan_path_copied : str or None
        Relative path for the copied study plan inside ``out_dir``.
    plan_summary : dict[str, Any]
        Compact summary of the loaded study plan.
    """
    git_impl = git_info_for_module("comp_model_impl")

    manifest = {
        "run_id": out_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **git_impl,
        "generator": f"{generator.__class__.__module__}.{generator.__class__.__name__}",
        "plan_file_copied": plan_path_copied,
        "plan_summary": plan_summary,
        "generating_models": [
            {
                "name": g.name,
                "model": g.model,
                "model_kwargs": g.model_kwargs,
                "sampling": "see config_dict",
            }
            for g in config.generating
        ],
        "candidate_models": [
            {
                "name": c.name,
                "model": c.model,
                "model_kwargs": c.model_kwargs,
                "estimator": c.estimator,
                "estimator_kwargs": c.estimator_kwargs,
            }
            for c in config.candidates
        ],
        "selection": asdict(config.selection) if hasattr(config.selection, "__dataclass_fields__") else None,
        "config_dict": asdict(config) if hasattr(config, "__dataclass_fields__") else None,
    }

    (out_dir / "model_recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

def _build_nested(value: Any, *, registries: Any) -> Any:
    """Build nested inline factory mappings found in kwargs.

    Parameters
    ----------
    value : Any
        Raw kwarg value.
    registries : Any
        Registry object returned by :func:`comp_model_impl.register.make_registry`.

    Returns
    -------
    Any
        Built value for inline factory mappings; original value otherwise.
    """

    if isinstance(value, Mapping) and "factory" in value:
        nested_kwargs_raw = value.get("kwargs", {})
        if nested_kwargs_raw is None:
            nested_kwargs: dict[str, Any] = {}
        elif isinstance(nested_kwargs_raw, Mapping):
            nested_kwargs = {str(k): v for k, v in nested_kwargs_raw.items()}
        else:
            raise TypeError("Inline factory kwargs must be a mapping")
        return _build_from_reference(
            reference=str(value["factory"]),
            kwargs=nested_kwargs,
            registries=registries,
            kind="model_or_other",
        )
    return value


def _build_kwargs(kwargs: Mapping[str, Any] | None, *, registries: Any) -> dict[str, Any]:
    """Build kwargs recursively.

    Parameters
    ----------
    kwargs : Mapping[str, Any] or None
        Raw kwargs mapping.
    registries : Any
        Registry object returned by :func:`comp_model_impl.register.make_registry`.

    Returns
    -------
    dict[str, Any]
        Prepared kwargs dictionary.
    """

    if kwargs is None:
        return {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"kwargs must be a mapping, got {type(kwargs)}")

    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        out[str(k)] = _build_nested(v, registries=registries)
    return out


def _build_from_reference(
    *,
    reference: str,
    kwargs: Mapping[str, Any] | None,
    registries: Any,
    kind: str,
) -> Any:
    """Build an object from a reference string and kwargs.

    Parameters
    ----------
    reference : str
        Registry key.
    kwargs : Mapping[str, Any] or None
        Constructor/factory kwargs.
    registries : Any
        Registry object returned by :func:`comp_model_impl.register.make_registry`.
    kind : {"model", "model_or_other", "estimator"}
        Build mode.

    Returns
    -------
    Any
        Instantiated object.
    """

    resolved_kwargs = _build_kwargs(kwargs, registries=registries)

    if kind in ("model", "model_or_other"):
        cls = resolve_model_callable(reference, registries=registries)
        return cls(**resolved_kwargs)

    if kind == "estimator":
        cls_or_fn = resolve_estimator_callable(reference, registries=registries)
        return cls_or_fn(**resolved_kwargs)

    raise ValueError(f"Unknown kind: {kind!r}")


def _build_model(model: str, *, model_kwargs: Mapping[str, Any] | None, registries: Any) -> ComputationalModel:
    """Build a computational model instance.

    Parameters
    ----------
    model : str
        Model registry key.
    model_kwargs : Mapping[str, Any] or None
        Model constructor/factory kwargs.
    registries : Any
        Registry object returned by :func:`comp_model_impl.register.make_registry`.

    Returns
    -------
    ComputationalModel
        Instantiated model.
    """

    built = _build_from_reference(
        reference=str(model),
        kwargs=model_kwargs,
        registries=registries,
        kind="model",
    )
    if not isinstance(built, ComputationalModel):
        raise TypeError(f"Built object is not a ComputationalModel: {type(built)}")
    return built


def _build_estimator(
    estimator: str,
    *,
    estimator_kwargs: Mapping[str, Any] | None,
    model: ComputationalModel,
    registries: Any,
) -> Estimator:
    """Build an estimator and inject ``model`` when supported.

    Parameters
    ----------
    estimator : str
        Estimator reference.
    estimator_kwargs : Mapping[str, Any] or None
        Estimator constructor/factory kwargs.
    model : ComputationalModel
        Candidate model instance.
    registries : Any
        Registry object returned by :func:`comp_model_impl.register.make_registry`.

    Returns
    -------
    Estimator
        Instantiated estimator.
    """

    kwargs = _build_kwargs(estimator_kwargs, registries=registries)
    cls_or_fn = resolve_estimator_callable(str(estimator), registries=registries)

    try:
        import inspect

        sig = inspect.signature(cls_or_fn)
        if "model" in sig.parameters and "model" not in kwargs:
            kwargs["model"] = model
    except Exception:
        if "model" not in kwargs:
            kwargs["model"] = model

    built = cls_or_fn(**kwargs)
    if not isinstance(built, Estimator):
        raise TypeError(f"Built object is not an Estimator: {type(built)}")
    return built


def _subject_ids_from_plan(plan: StudyPlan) -> list[str]:
    """Extract subject identifiers from a study plan.

    Parameters
    ----------
    plan : StudyPlan
        Study plan instance.

    Returns
    -------
    list[str]
        Subject IDs as strings.
    """

    return [str(sid) for sid in (plan.subjects.keys() if plan.subjects else [])]


def _params_hat_by_subject(fit: FitResult, subject_ids: list[str]) -> dict[str, dict[str, float]]:
    """Normalize fit output to a subject-parameter mapping.

    Parameters
    ----------
    fit : FitResult
        Fit result object.
    subject_ids : list[str]
        Expected subject identifiers.

    Returns
    -------
    dict[str, dict[str, float]]
        Per-subject parameter dictionary.
    """

    if getattr(fit, "subject_hats", None):
        return {str(k): dict(v) for k, v in (fit.subject_hats or {}).items()}

    if getattr(fit, "params_hat", None):
        p = dict(fit.params_hat or {})
        return {sid: dict(p) for sid in subject_ids}

    return {}


def _count_free_params(
    *,
    model: ComputationalModel,
    fit: FitResult | None,
    n_subjects: int,
    fixed_params: Mapping[str, float] | None,
) -> tuple[int, int]:
    """Estimate free parameter counts for model selection.

    Parameters
    ----------
    model : ComputationalModel
        Candidate model.
    fit : FitResult or None
        Fit result used for fallback dimensionality inference.
    n_subjects : int
        Number of subjects.
    fixed_params : Mapping[str, float] or None
        Fixed parameter values passed to the estimator.

    Returns
    -------
    tuple[int, int]
        ``(k_per_subject, k_total)``.
    """

    fixed = set((fixed_params or {}).keys())
    k_per_sub: int | None = None

    schema = getattr(model, "param_schema", None)
    if schema is not None and getattr(schema, "names", None):
        try:
            names = list(schema.names)
            k_per_sub = int(len([n for n in names if n not in fixed]))
        except Exception:
            k_per_sub = None

    if k_per_sub is None and fit is not None and getattr(fit, "subject_hats", None):
        try:
            sizes = [len(d) for d in (fit.subject_hats or {}).values()]
            if sizes:
                k_per_sub = int(np.median(sizes)) - int(len(fixed))
        except Exception:
            k_per_sub = None

    if k_per_sub is None:
        k_per_sub = 0

    k_pop = 0
    if fit is not None and getattr(fit, "population_hat", None):
        try:
            k_pop = int(len(fit.population_hat or {}))
        except Exception:
            k_pop = 0

    k_total = int(k_per_sub * int(n_subjects) + k_pop)
    return int(k_per_sub), int(k_total)


def _extract_waic_from_fit_diagnostics(fit: FitResult) -> dict[str, float] | None:
    """Extract WAIC diagnostics from a fit result when available.

    Supports either:
    - top-level diagnostics keys (``waic``, ``elpd_waic``, optional ``p_waic``),
    - per-subject diagnostics under ``diagnostics['per_subject']`` where each
      subject has WAIC metrics (aggregated by summation).
    """
    d = getattr(fit, "diagnostics", None)
    if not isinstance(d, Mapping):
        return None

    # Direct study-level WAIC (typical for hierarchical Stan estimators).
    if "waic" in d and "elpd_waic" in d:
        out: dict[str, float] = {
            "waic": float(d["waic"]),
            "elpd_waic": float(d["elpd_waic"]),
        }
        if "p_waic" in d:
            out["p_waic"] = float(d["p_waic"])
        if "waic_n_obs" in d:
            out["waic_n_obs"] = float(d["waic_n_obs"])
        if all(np.isfinite(list(out.values()))):
            return out
        return None

    # Subject-wise WAIC (typical for independent Stan fits).
    per_subject = d.get("per_subject", None)
    if not isinstance(per_subject, Mapping):
        return None

    waic_rows: list[Mapping[str, Any]] = []
    for v in per_subject.values():
        if isinstance(v, Mapping) and "waic" in v and "elpd_waic" in v:
            waic_rows.append(v)
    if not waic_rows:
        return None

    out = {
        "waic": float(np.sum([float(v["waic"]) for v in waic_rows])),
        "elpd_waic": float(np.sum([float(v["elpd_waic"]) for v in waic_rows])),
    }
    if all("p_waic" in v for v in waic_rows):
        out["p_waic"] = float(np.sum([float(v["p_waic"]) for v in waic_rows]))
    if all("waic_n_obs" in v for v in waic_rows):
        out["waic_n_obs"] = float(np.sum([float(v["waic_n_obs"]) for v in waic_rows]))

    if all(np.isfinite(list(out.values()))):
        return out
    return None


def _select_winner(
    rows: list[dict[str, Any]],
    *,
    criterion: ModelCriterion,
    tie_break: str,
    atol: float,
) -> dict[str, Any]:
    """Select the winning candidate row for one replication.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Candidate rows for one ``(rep, generating_model)`` group.
    criterion : ModelCriterion
        Criterion instance defining score direction.
    tie_break : str
        Tie strategy in ``{"first", "simpler"}``.
    atol : float
        Absolute tolerance used to detect ties.

    Returns
    -------
    dict[str, Any]
        Winner summary row.
    """

    if not rows:
        raise ValueError("No candidate rows to select from.")

    # Filter to finite scores first; if all invalid, keep all.
    scores = np.array([r.get("score", np.nan) for r in rows], dtype=float)
    finite_mask = np.isfinite(scores)
    cand_rows = [r for r, ok in zip(rows, finite_mask) if ok]
    if not cand_rows:
        cand_rows = rows

    # Determine best score direction.
    def better(a: float, b: float) -> bool:
        return a > b if criterion.higher_is_better() else a < b

    best = cand_rows[0]
    for r in cand_rows[1:]:
        if better(float(r["score"]), float(best["score"])):
            best = r

    # Tie set within atol
    tie_set = [r for r in cand_rows if np.isfinite(r.get("score", np.nan)) and abs(float(r["score"]) - float(best["score"])) <= float(atol)]
    if tie_set and tie_break.lower() == "simpler":
        # prefer smaller k_total
        best = min(tie_set, key=lambda r: (int(r.get("k_total", 10**9)), cand_rows.index(r)))
    elif tie_set and tie_break.lower() == "first":
        # keep order already implied
        best = tie_set[0]

    # Second best for delta.
    sorted_rows = sorted(
        [r for r in cand_rows if np.isfinite(r.get("score", np.nan))],
        key=lambda r: float(r["score"]),
        reverse=criterion.higher_is_better(),
    )
    second = sorted_rows[1] if len(sorted_rows) > 1 else None

    if second is None:
        delta = np.nan
        second_name = None
        second_score = np.nan
    else:
        winner_score = float(best["score"])
        second_score = float(second["score"])
        if criterion.higher_is_better():
            delta = winner_score - second_score
        else:
            delta = second_score - winner_score
        second_name = str(second["candidate_model"])

    return {
        "selected_model": str(best["candidate_model"]),
        "winner_score": float(best.get("score", np.nan)),
        "second_best_model": second_name,
        "second_best_score": float(second_score),
        "delta_to_second": float(delta),
        "winner_ll_total": float(best.get("ll_total", np.nan)),
        "winner_k_total": int(best.get("k_total", -1)),
    }


def run_model_recovery(
    *,
    config: ModelRecoveryConfig,
    generator: Generator,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ModelRecoveryOutputs:
    """Run a full model recovery experiment.

    Parameters
    ----------
    config : ModelRecoveryConfig
        Model recovery configuration.
    generator : Generator
        Generator used to simulate studies (should attach event logs).
    progress_callback : callable or None, default=None
        Optional callback invoked after each replication with (completed, total).

    Returns
    -------
    ModelRecoveryOutputs
        Fit table, winners table, and output directory.
    """
    # Load plan
    plan_path = Path(config.plan_path)
    if plan_path.suffix.lower() in (".yaml", ".yml"):
        plan: StudyPlan = load_study_plan_yaml(str(plan_path))
    elif plan_path.suffix.lower() == ".json":
        plan = load_study_plan_json(str(plan_path))
    else:
        raise ValueError("plan_path must end with .yaml/.yml or .json")

    subject_ids = _subject_ids_from_plan(plan)
    if not subject_ids:
        raise ValueError("Study plan has no subjects.")

    if not config.generating:
        raise ValueError("config.generating is empty.")
    if not config.candidates:
        raise ValueError("config.candidates is empty.")

    criterion = get_criterion(config.selection.criterion)

    # Registry + block runner builder
    registries = make_registry()
    block_runner_builder = lambda p: build_runner_for_plan(plan=p, registries=registries)

    # Output directory
    git_impl = git_info_for_module("comp_model_impl")
    out_dir = make_unique_run_dir(config.output.out_dir, git_commit=git_impl.get("comp_model_impl_git_commit"))

    # Copy plan into output directory (provenance)
    plan_copy_rel = None
    try:
        plan_copy_rel = f"plan{plan_path.suffix.lower()}"
        (out_dir / plan_copy_rel).write_bytes(plan_path.read_bytes())
    except Exception:
        plan_copy_rel = None

    plan_summary = _plan_summary(plan)
    if config.output.save_config:
        (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")
        write_model_recovery_manifest(
            out_dir,
            config=config,
            generator=generator,
            plan_path_copied=plan_copy_rel,
            plan_summary=plan_summary,
        )

    # Main loop
    fit_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    total = int(config.n_reps) * int(len(config.generating))
    completed = 0

    base_rng = np.random.default_rng(int(config.seed))

    for gen_spec in config.generating:
        # Build generating model once per generating spec, but fresh per rep for safety.
        gen_name = str(gen_spec.name)

        for rep in tqdm(
            range(int(config.n_reps)),
            desc=f"Model recovery: gen={gen_name}",
            leave=False,
            disable=not sys.stderr.isatty(),
        ):
            rep_seed = int(base_rng.integers(0, 2**32 - 1))
            rep_rng = np.random.default_rng(rep_seed)

            # Build generating model fresh each replication.
            gen_model = _build_model(
                gen_spec.model,
                model_kwargs=gen_spec.model_kwargs,
                registries=registries,
            )

            # Sample true params per subject
            subj_true, pop_true = None, None
            from comp_model_impl.recovery.parameter.sampling import sample_subject_params
            subj_true, pop_true = sample_subject_params(
                cfg=gen_spec.sampling,
                model=gen_model,
                subject_ids=subject_ids,
                rng=rep_rng,
            )

            # Simulate study
            study = generator.simulate_study(
                block_runner_builder=block_runner_builder,
                model=gen_model,
                subj_params=subj_true,
                subject_block_plans=plan.subjects,
                rng=rep_rng,
            )

            if config.output.save_simulated_study:
                import pickle
                with (out_dir / f"study_{gen_name}_rep_{rep:04d}.pkl").open("wb") as f:
                    pickle.dump(study, f)

            # Fit all candidates
            rep_candidate_rows: list[dict[str, Any]] = []

            for cand_spec in config.candidates:
                cand_name = str(cand_spec.name)
                t0 = time.time()

                try:
                    cand_model = _build_model(
                        cand_spec.model,
                        model_kwargs=cand_spec.model_kwargs,
                        registries=registries,
                    )
                    estimator = _build_estimator(
                        cand_spec.estimator,
                        estimator_kwargs=cand_spec.estimator_kwargs,
                        model=cand_model,
                        registries=registries,
                    )

                    # Run fit; pass fixed_params if supported
                    fit_kwargs: dict[str, Any] = {"study": study, "rng": rep_rng}
                    if cand_spec.fixed_params:
                        try:
                            import inspect
                            if "fixed_params" in inspect.signature(estimator.fit).parameters:
                                fit_kwargs["fixed_params"] = dict(cand_spec.fixed_params)
                        except Exception:
                            # Safe attempt; estimator.fit may accept **kwargs
                            fit_kwargs["fixed_params"] = dict(cand_spec.fixed_params)

                    fit: FitResult = estimator.fit(**fit_kwargs)

                    subj_hat = _params_hat_by_subject(fit, subject_ids)
                    ll_summary = compute_likelihood_summary(
                        study=study,
                        model=cand_model,
                        subject_params=subj_hat,
                    )

                    k_per_sub, k_total = _count_free_params(
                        model=cand_model,
                        fit=fit,
                        n_subjects=len(subject_ids),
                        fixed_params=cand_spec.fixed_params,
                    )

                    score = criterion.score(ll=ll_summary.ll_total, k=k_total, n_obs=ll_summary.n_obs_total)

                    success = bool(getattr(fit, "success", True)) and np.isfinite(score)
                    message = str(getattr(fit, "message", ""))
                    value = float(getattr(fit, "value", np.nan)) if getattr(fit, "value", None) is not None else np.nan

                    runtime_s = float(time.time() - t0)

                    row = {
                        "rep": int(rep),
                        "rep_seed": int(rep_seed),
                        "generating_model": gen_name,
                        "candidate_model": cand_name,
                        "criterion": str(criterion.name),
                        "success": bool(success),
                        "message": message,
                        "fit_value": value,
                        "ll_total": float(ll_summary.ll_total),
                        "n_obs_total": int(ll_summary.n_obs_total),
                        "k_per_subject": int(k_per_sub),
                        "k_total": int(k_total),
                        "score": float(score),
                        "runtime_s": runtime_s,
                    }
                    waic_diag = _extract_waic_from_fit_diagnostics(fit)
                    if waic_diag is not None:
                        row.update(waic_diag)
                    fit_rows.append(row)
                    rep_candidate_rows.append(row)

                    if config.output.save_fit_diagnostics:
                        diag_rows.append(
                            {
                                "rep": int(rep),
                                "rep_seed": int(rep_seed),
                                "generating_model": gen_name,
                                "candidate_model": cand_name,
                                "true_params": subj_true,
                                "population_true": pop_true,
                                "success": bool(getattr(fit, "success", True)),
                                "message": str(getattr(fit, "message", "")),
                                "fit_value": value if np.isfinite(value) else None,
                                "fit_diagnostics": getattr(fit, "diagnostics", None),
                                "params_hat_by_subject": subj_hat,
                                "ll_by_subject": ll_summary.ll_by_subject,
                                "n_obs_by_subject": ll_summary.n_obs_by_subject,
                                "ll_total": float(ll_summary.ll_total),
                                "n_obs_total": int(ll_summary.n_obs_total),
                                "k_total": int(k_total),
                                "score": float(score),
                                "criterion": str(criterion.name),
                                "runtime_s": runtime_s,
                            }
                        )

                except Exception as e:
                    runtime_s = float(time.time() - t0)
                    row = {
                        "rep": int(rep),
                        "rep_seed": int(rep_seed),
                        "generating_model": gen_name,
                        "candidate_model": cand_name,
                        "criterion": str(criterion.name),
                        "success": False,
                        "message": f"EXCEPTION: {type(e).__name__}: {e}",
                        "fit_value": np.nan,
                        "ll_total": float("-inf"),
                        "n_obs_total": np.nan,
                        "k_per_subject": np.nan,
                        "k_total": np.nan,
                        "score": float("inf") if not criterion.higher_is_better() else float("-inf"),
                        "runtime_s": runtime_s,
                    }
                    fit_rows.append(row)
                    rep_candidate_rows.append(row)

                    if config.output.save_fit_diagnostics:
                        diag_rows.append(
                            {
                                "rep": int(rep),
                                "rep_seed": int(rep_seed),
                                "generating_model": gen_name,
                                "candidate_model": cand_name,
                                "true_params": subj_true,
                                "population_true": pop_true,
                                "success": False,
                                "message": row["message"],
                                "fit_value": None,
                                "fit_diagnostics": None,
                                "params_hat_by_subject": None,
                                "ll_by_subject": None,
                                "n_obs_by_subject": None,
                                "ll_total": float("-inf"),
                                "n_obs_total": None,
                                "k_total": None,
                                "score": row["score"],
                                "criterion": str(criterion.name),
                                "runtime_s": runtime_s,
                            }
                        )

            # Winner selection for this rep
            winner = _select_winner(
                rep_candidate_rows,
                criterion=criterion,
                tie_break=config.selection.tie_break,
                atol=float(config.selection.atol),
            )
            winner_rows.append(
                {
                    "rep": int(rep),
                    "rep_seed": int(rep_seed),
                    "generating_model": gen_name,
                    **winner,
                }
            )

            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    fit_df = pd.DataFrame.from_records(fit_rows)
    winners_df = pd.DataFrame.from_records(winner_rows)

    # Save tables
    fmt = str(config.output.save_format).lower()
    if fmt == "csv":
        fit_df.to_csv(out_dir / "model_recovery_fit_table.csv", index=False)
    elif fmt == "parquet":
        fit_df.to_parquet(out_dir / "model_recovery_fit_table.parquet", index=False)
    else:
        raise ValueError("output.save_format must be 'csv' or 'parquet'")

    winners_df.to_csv(out_dir / "model_recovery_winners.csv", index=False)

    if config.output.save_fit_diagnostics:
        with (out_dir / "model_recovery_fit_diagnostics.jsonl").open("w", encoding="utf-8") as f:
            for row in diag_rows:
                f.write(json.dumps(row, default=str) + "\n")

    # Save a small summary file
    try:
        from .analysis import confusion_matrix, recovery_rates

        cm = confusion_matrix(winners_df)
        rr = recovery_rates(winners_df)
        cm.to_csv(out_dir / "model_recovery_confusion_matrix.csv")
        rr.to_csv(out_dir / "model_recovery_recovery_rates.csv", index=False)
    except Exception:
        pass

    return ModelRecoveryOutputs(
        fit_table=fit_df,
        winners=winners_df,
        out_dir=str(out_dir),
    )
