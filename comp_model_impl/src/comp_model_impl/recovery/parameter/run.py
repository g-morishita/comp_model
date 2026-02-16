"""End-to-end parameter recovery runner.

This module orchestrates simulation + fitting loops for parameter recovery
experiments. It writes a self-contained run directory with configuration
    snapshots, true-vs-estimated parameter records, summary metrics, and optional
    diagnostics. Helper utilities are included for manifest generation
and population-level recovery summaries in hierarchical settings.

Notes
-----
- Supports standard models and within-subject shared+delta wrappers.
- Population-level summaries depend on estimator diagnostics and are only
  computed when the sampling mode is hierarchical.

See Also
--------
comp_model_impl.recovery.parameter.config.ParameterRecoveryConfig
comp_model_impl.recovery.parameter.sampling.sample_subject_params
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import json
import secrets
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from comp_model_core.plans.io import load_study_plan
from comp_model_core.plans.block import StudyPlan, BlockPlan
from comp_model_core.data.types import StudyData
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.register import make_registry
from comp_model_impl.tasks.build import build_runner_for_plan

from comp_model_impl.recovery.parameter.analysis import (
    compute_parameter_recovery_metrics,
    compute_population_recovery_metrics,
)
from comp_model_impl.recovery.parameter.config import ParameterRecoveryConfig, save_config_auto
from comp_model_impl.recovery.parameter.sampling import sample_subject_params
from comp_model_impl.models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
    constrained_params_by_condition_from_z,
    flatten_params_by_condition,
)


@dataclass(frozen=True, slots=True)
class ParameterRecoveryOutputs:
    """Outputs from a parameter recovery run.

    Attributes
    ----------
    records : pandas.DataFrame
        Tidy table of true/estimated parameters per replication and subject.
    metrics : pandas.DataFrame
        Metrics per parameter and replication (correlation, RMSE, bias, etc.).
    out_dir : str
        Output directory where artifacts were written.
    population_records : pandas.DataFrame or None
        Population-level true/estimated records for hierarchical runs, if available.
    population_metrics : pandas.DataFrame or None
        Population-level recovery metrics pooled across replications, if available.
    """

    records: pd.DataFrame
    metrics: pd.DataFrame
    out_dir: str
    population_records: pd.DataFrame | None = None
    population_metrics: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class _ReplicationResult:
    """Per-replication outputs used by sequential/parallel execution."""

    rep: int
    records: list[dict[str, Any]]
    fit_diag: dict[str, Any] | None = None


_WORKER_CONTEXT: dict[str, Any] | None = None


def _compute_fixed_mode_error_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed-mode recovery errors pooled across reps/subjects.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy records table with at least ``param``, ``true``, and ``hat``.

    Returns
    -------
    pandas.DataFrame
        Per-parameter pooled error summary with bias and absolute/squared error
        statistics. If no usable rows are available, returns an empty table with
        the expected columns.
    """
    columns = [
        "param",
        "n",
        "true_unique",
        "mean_true",
        "mean_hat",
        "bias_mean",
        "bias_std",
        "mae",
        "mse",
        "rmse",
        "median_abs_error",
    ]

    if df.empty or not {"param", "true", "hat"}.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for param, g in df.groupby("param", sort=True):
        g2 = g.dropna(subset=["true", "hat"])
        if len(g2) == 0:
            rows.append(
                {
                    "param": param,
                    "n": 0,
                    "true_unique": 0,
                    "mean_true": np.nan,
                    "mean_hat": np.nan,
                    "bias_mean": np.nan,
                    "bias_std": np.nan,
                    "mae": np.nan,
                    "mse": np.nan,
                    "rmse": np.nan,
                    "median_abs_error": np.nan,
                }
            )
            continue

        x = g2["true"].to_numpy(dtype=float)
        y = g2["hat"].to_numpy(dtype=float)
        err = y - x
        abs_err = np.abs(err)
        sq_err = err ** 2

        rows.append(
            {
                "param": str(param),
                "n": int(len(g2)),
                "true_unique": int(np.unique(x).size),
                "mean_true": float(np.mean(x)),
                "mean_hat": float(np.mean(y)),
                "bias_mean": float(np.mean(err)),
                "bias_std": float(np.std(err, ddof=0)),
                "mae": float(np.mean(abs_err)),
                "mse": float(np.mean(sq_err)),
                "rmse": float(np.sqrt(np.mean(sq_err))),
                "median_abs_error": float(np.median(abs_err)),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values("param").reset_index(drop=True)


def _strip_hat_key(name: str) -> str:
    """Strip a trailing ``_hat`` suffix from a parameter name.

    Parameters
    ----------
    name : str
        Parameter name (possibly ending in ``_hat``).

    Returns
    -------
    str
        Name without the ``_hat`` suffix.
    """
    return str(name[:-4]) if str(name).endswith("_hat") else str(name)


def _maybe_compute_population_recovery(
    *,
    fit_diags: list[dict[str, Any]],
    config: ParameterRecoveryConfig,
    model: ComputationalModel,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Compute population-level recovery records and metrics when available.

    Requirements
    ------------
    - ``sampling.mode`` must be ``"hierarchical"``
    - diagnostics must include ``population_true`` and ``population_hat``

    Notes
    -----
    For standard hierarchical models (non within-subject), we align:
    - true: ``population_true["alpha_p"]`` vs hat: ``population_hat["alpha_p_pop"]``
    - true: ``sampling.individual_sd["alpha_p"]`` vs hat: ``population_hat["sd_alpha_p"]``

    For within-subject shared+delta wrappers, we also compute:
    - per-condition constrained population locations: ``alpha_p_pop__A``, ...
    - shared/delta hyperparameters on z-scale: ``mu_alpha_p_shared``, ``sd_alpha_p_shared``, ...

    Parameters
    ----------
    fit_diags : list[dict[str, Any]]
        Per-rep diagnostics from the estimator.
    config : ParameterRecoveryConfig
        Recovery configuration.
    model : ComputationalModel
        Model used in the recovery run.

    Returns
    -------
    tuple[pandas.DataFrame | None, pandas.DataFrame | None]
        Population-level records and metrics, or ``(None, None)`` if unavailable.
    """
    if str(config.sampling.mode).lower() != "hierarchical":
        return None, None

    # Build a tidy table with subject_id set to a sentinel value.
    rows: list[dict[str, Any]] = []

    def _is_ws(m: ComputationalModel) -> bool:
        return isinstance(m, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel))

    # True subject-level SDs (z-scale if sampling.space=z; param-scale otherwise).
    true_sds = dict(getattr(config.sampling, "individual_sd", {}) or {})

    for d in fit_diags:
        rep = int(d.get("rep", -1))
        pop_true = d.get("population_true")
        pop_hat = d.get("population_hat")
        if not isinstance(pop_true, dict) or not isinstance(pop_hat, dict):
            continue

        # Normalize estimator keys by stripping trailing _hat.
        pop_hat_n = {_strip_hat_key(k): float(v) for k, v in pop_hat.items() if v is not None}

        if _is_ws(model):
            # 1) Per-condition constrained population locations on natural scale
            try:
                pb = constrained_params_by_condition_from_z(model, pop_true)  # type: ignore[arg-type]
                for cond, pmap in pb.items():
                    for p, v in pmap.items():
                        key = f"{p}_pop__{cond}"
                        if key in pop_hat_n:
                            rows.append(
                                {
                                    "rep": rep,
                                    "subject_id": "POP",
                                    "param": key,
                                    "true": float(v),
                                    "hat": float(pop_hat_n.get(key, np.nan)),
                                }
                            )
            except Exception:
                pass

            # 2) z-scale hyperparameters (shared / delta) if present
            base_schema = getattr(getattr(model, "base_model", None), "param_schema", None)
            base_params = list(getattr(base_schema, "params", []) or [])
            conditions = list(getattr(model, "conditions", []) or [])
            baseline = str(getattr(model, "baseline_condition", ""))

            for pdef in base_params:
                pname = str(getattr(pdef, "name"))
                true_mu_key = f"{pname}__shared_z"
                mu_hat_key = f"mu_{pname}__shared"
                sd_hat_key = f"sd_{pname}__shared"
                alt_mu_hat_key = f"mu_{pname}_shared"
                alt_sd_hat_key = f"sd_{pname}_shared"
                mu_key = mu_hat_key if mu_hat_key in pop_hat_n else alt_mu_hat_key
                sd_key = sd_hat_key if sd_hat_key in pop_hat_n else alt_sd_hat_key
                if true_mu_key in pop_true and mu_key in pop_hat_n:
                    rows.append(
                        {
                            "rep": rep,
                            "subject_id": "POP",
                            "param": mu_key,
                            "true": float(pop_true[true_mu_key]),
                            "hat": float(pop_hat_n.get(mu_key, np.nan)),
                        }
                    )
                if true_mu_key in true_sds and sd_key in pop_hat_n:
                    rows.append(
                        {
                            "rep": rep,
                            "subject_id": "POP",
                            "param": sd_key,
                            "true": float(true_sds[true_mu_key]),
                            "hat": float(pop_hat_n.get(sd_key, np.nan)),
                        }
                    )

                for cond in conditions:
                    if cond == baseline:
                        continue
                    true_d_key = f"{pname}__delta_z__{cond}"
                    mu_d_hat_key = f"mu_{pname}__delta__{cond}"
                    sd_d_hat_key = f"sd_{pname}__delta__{cond}"
                    alt_mu_d_hat_key = f"mu_{pname}_delta__{cond}"
                    alt_sd_d_hat_key = f"sd_{pname}_delta__{cond}"
                    mu_d_key = mu_d_hat_key if mu_d_hat_key in pop_hat_n else alt_mu_d_hat_key
                    sd_d_key = sd_d_hat_key if sd_d_hat_key in pop_hat_n else alt_sd_d_hat_key
                    if true_d_key in pop_true and mu_d_key in pop_hat_n:
                        rows.append(
                            {
                                "rep": rep,
                                "subject_id": "POP",
                                "param": mu_d_key,
                                "true": float(pop_true[true_d_key]),
                                "hat": float(pop_hat_n.get(mu_d_key, np.nan)),
                            }
                        )
                    if true_d_key in true_sds and sd_d_key in pop_hat_n:
                        rows.append(
                            {
                                "rep": rep,
                                "subject_id": "POP",
                                "param": sd_d_key,
                                "true": float(true_sds[true_d_key]),
                                "hat": float(pop_hat_n.get(sd_d_key, np.nan)),
                            }
                        )

        else:
            # Standard hierarchical model (no within-subject conditioning)
            for p, v_true in pop_true.items():
                # Match to population-level natural-scale location if present
                key = f"{p}_pop"
                if key in pop_hat_n:
                    rows.append(
                        {
                            "rep": rep,
                            "subject_id": "POP",
                            "param": key,
                            "true": float(v_true),
                            "hat": float(pop_hat_n.get(key, np.nan)),
                        }
                    )

            # SDs (typically on z-scale) from the sampling config
            for p, sd_true in true_sds.items():
                sd_key = f"sd_{p}"
                if sd_key in pop_hat_n:
                    rows.append(
                        {
                            "rep": rep,
                            "subject_id": "POP",
                            "param": sd_key,
                            "true": float(sd_true),
                            "hat": float(pop_hat_n.get(sd_key, np.nan)),
                        }
                    )

    if not rows:
        return None, None
    pop_df = pd.DataFrame.from_records(rows)
    pop_df.sort_values(["param", "rep"], inplace=True)
    pop_metrics = compute_population_recovery_metrics(pop_df)
    return pop_df, pop_metrics


def _find_git_root(start: Path) -> Path | None:
    """Walk upward until a ``.git`` entry is found.

    Parameters
    ----------
    start : pathlib.Path
        Starting directory for the search.

    Returns
    -------
    pathlib.Path or None
        Repository root if found, otherwise ``None``.
    """
    p = start.resolve()
    for parent in [p, *p.parents]:
        git_entry = parent / ".git"
        if git_entry.exists():
            return parent
    return None


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    """Run a git command and return its output (or None on failure).

    Parameters
    ----------
    repo_root : pathlib.Path
        Root of the git repository.
    args : list[str]
        Git arguments (e.g., ``["rev-parse", "HEAD"]``).

    Returns
    -------
    str or None
        Command output (stripped) or ``None`` if the command fails.
    """
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return None


def make_unique_run_dir(base_out_dir: str | Path) -> Path:
    """Create a unique run directory under a base output directory.

    Parameters
    ----------
    base_out_dir : str or pathlib.Path
        Root output directory.

    Returns
    -------
    pathlib.Path
        Newly created run directory path.

    Notes
    -----
    The directory name encodes a timestamp and a short hash:
    ``YYYY-MM-DD_HHMMSS__<suffix>``.
    """
    base = Path(base_out_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = secrets.token_hex(2)  # 4 hex chars

    run_id = f"{ts}__{suffix}"
    out_dir = base / run_id
    out_dir.mkdir(parents=False, exist_ok=False)
    return out_dir


def _plan_summary(plan: StudyPlan) -> dict[str, Any]:
    """Summarize a StudyPlan for run manifests.

    Parameters
    ----------
    plan : StudyPlan
        Study plan containing per-subject block lists.

    Returns
    -------
    dict[str, Any]
        Summary statistics and a small block table.
    """
    n_subjects = int(len(plan.subjects))

    blocks_per_subject: list[int] = []
    trials_per_subject: list[int] = []

    block_trials_by_id: dict[str, list[int]] = {}
    block_seen_by_subject: dict[str, set[str]] = {}

    # Canonical view: derive block order + trials from first subject
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


def write_run_manifest(
    out_dir: Path,
    *,
    config_obj: Any,
    generator: Any,
    model: Any,
    estimator: Any,
    plan_path_copied: str | None,
    plan_summary: dict[str, Any],
) -> None:
    """Write a JSON manifest describing a recovery run.

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory for this run.
    config_obj : Any
        Configuration object (typically :class:`ParameterRecoveryConfig`).
    generator, model, estimator : Any
        Instantiated objects used in the run.
    plan_path_copied : str or None
        If the plan file was copied into ``out_dir``, the relative path.
    plan_summary : dict[str, Any]
        Small summary of the plan (e.g., number of blocks/trials).
    """
    manifest = {
        "run_id": out_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generator": f"{generator.__class__.__module__}.{generator.__class__.__name__}",
        "model": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "estimator": f"{estimator.__class__.__module__}.{estimator.__class__.__name__}",
        "plan_file_copied": plan_path_copied,
        "plan_summary": plan_summary,
        "config_dict": asdict(config_obj) if hasattr(config_obj, "__dataclass_fields__") else None,
    }

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def _is_within_subject_model(m: ComputationalModel) -> bool:
    """Return ``True`` if the model is a within-subject shared+delta wrapper."""
    return isinstance(m, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel))


def _derive_within_subject_targets(
    m: ComputationalModel,
    params: dict[str, float],
) -> dict[str, float]:
    """Derive constrained per-condition parameters from z-params."""
    pb = constrained_params_by_condition_from_z(m, params)  # type: ignore[arg-type]
    return flatten_params_by_condition(pb)


def _run_single_rep(
    *,
    rep: int,
    rep_seed: int,
    config: ParameterRecoveryConfig,
    generator: Generator,
    model: ComputationalModel,
    estimator: Estimator,
    subject_ids: Sequence[str],
    subject_block_plans: Mapping[str, Sequence[BlockPlan]],
    out_dir: str,
) -> _ReplicationResult:
    """Run one replication: sample -> simulate -> fit -> collect records."""
    rep_rng = np.random.default_rng(int(rep_seed))

    subj_params_true, pop_true = sample_subject_params(
        cfg=config.sampling,
        model=model,
        subject_ids=subject_ids,
        rng=rep_rng,
    )

    r = make_registry()

    def block_runner_builder(block_plan: BlockPlan):
        return build_runner_for_plan(plan=block_plan, registries=r)

    study: StudyData = generator.simulate_study(
        block_runner_builder=block_runner_builder,
        model=model,
        subj_params=subj_params_true,
        subject_block_plans=subject_block_plans,
        rng=rep_rng,
    )

    fit: FitResult = estimator.fit(study=study, rng=rep_rng)

    if config.output.save_simulated_study:
        import pickle

        with (Path(out_dir) / f"study_rep_{rep:04d}.pkl").open("wb") as f:
            pickle.dump(study, f)

    fit_diag: dict[str, Any] | None = None
    if config.output.save_fit_diagnostics:
        fit_diag = {
            "rep": rep,
            "success": bool(getattr(fit, "success", True)),
            "message": str(getattr(fit, "message", "")),
            "value": float(fit.value) if getattr(fit, "value", None) is not None else None,
            "diagnostics": getattr(fit, "diagnostics", None),
            "population_true": pop_true,
            "population_hat": getattr(fit, "population_hat", None),
        }

    records: list[dict[str, Any]] = []
    hats = fit.subject_hats or {}
    for sid in subject_ids:
        theta_true_raw = subj_params_true[sid]
        theta_hat_raw = hats.get(sid, {})

        if _is_within_subject_model(model):
            # Compare constrained per-condition params (param__cond)
            true_targets = _derive_within_subject_targets(model, dict(theta_true_raw))

            # Prefer already-constrained hats (Bayesian typically outputs these),
            # otherwise derive from z-space hats if present (e.g. MLE in z-space).
            if all(k in theta_hat_raw for k in true_targets.keys()):
                hat_targets = {k: float(theta_hat_raw.get(k, np.nan)) for k in true_targets.keys()}
            else:
                z_names = list(getattr(model.param_schema, "names", []) or [])
                if z_names and all(k in theta_hat_raw for k in z_names):
                    hat_targets = _derive_within_subject_targets(
                        model,
                        {k: float(theta_hat_raw[k]) for k in z_names},
                    )
                else:
                    hat_targets = {k: float(theta_hat_raw.get(k, np.nan)) for k in true_targets.keys()}

            for p, v_true in true_targets.items():
                v_hat = hat_targets.get(p, np.nan)
                records.append(
                    {
                        "rep": rep,
                        "subject_id": sid,
                        "param": p,
                        "true": float(v_true),
                        "hat": float(v_hat) if np.isfinite(v_hat) else np.nan,
                    }
                )
        else:
            theta_true = theta_true_raw
            theta_hat = theta_hat_raw
            for p, v_true in theta_true.items():
                v_hat = theta_hat.get(p, np.nan)
                records.append(
                    {
                        "rep": rep,
                        "subject_id": sid,
                        "param": p,
                        "true": float(v_true),
                        "hat": float(v_hat) if np.isfinite(v_hat) else np.nan,
                    }
                )

    return _ReplicationResult(rep=rep, records=records, fit_diag=fit_diag)


def _init_worker_context(
    config: ParameterRecoveryConfig,
    generator: Generator,
    model: ComputationalModel,
    estimator: Estimator,
    subject_ids: list[str],
    subject_block_plans: dict[str, list[BlockPlan]],
    out_dir: str,
) -> None:
    """Initialize global worker context for process-based replications."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "config": config,
        "generator": generator,
        "model": model,
        "estimator": estimator,
        "subject_ids": subject_ids,
        "subject_block_plans": subject_block_plans,
        "out_dir": out_dir,
    }


def _run_single_rep_from_worker(rep: int, rep_seed: int) -> _ReplicationResult:
    """Run one replication from process-worker global context."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context not initialized.")
    ctx = _WORKER_CONTEXT
    return _run_single_rep(
        rep=rep,
        rep_seed=rep_seed,
        config=ctx["config"],
        generator=ctx["generator"],
        model=ctx["model"],
        estimator=ctx["estimator"],
        subject_ids=ctx["subject_ids"],
        subject_block_plans=ctx["subject_block_plans"],
        out_dir=ctx["out_dir"],
    )


def _safe_copy_file(src: str | Path, dst_dir: str | Path) -> str | None:
    """
    Copy a file into a destination directory if the source file exists.

    Parameters
    ----------
    src : str | Path
        Source file path.
    dst_dir : str | Path
        Destination directory path.

    Returns
    -------
    str | None
        Copied filename on success, or ``None`` if ``src`` does not exist or
        is not a file or the copy operation fails.

    Notes
    -----
    This helper does not create ``dst_dir``. The caller is expected to ensure
    the destination directory already exists.
    """

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


def run_parameter_recovery(
    *,
    config: ParameterRecoveryConfig,
    generator: Generator,
    model: ComputationalModel,
    estimator: Estimator,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ParameterRecoveryOutputs:
    """Run a full parameter recovery experiment.

    Parameters
    ----------
    config : ParameterRecoveryConfig
        Recovery configuration.
    generator : Generator
        Data generator used to simulate studies.
    model : ComputationalModel
        Computational model used for simulation and fitting.
    estimator : Estimator
        Estimator used to fit the model to simulated data.
    progress_callback : Callable[[int, int], None] or None, optional
        Optional callback invoked after each replication with
        ``(completed, total)`` to report progress.

    Returns
    -------
    ParameterRecoveryOutputs
        Records, metrics, output directory, and optional population summaries.

    Raises
    ------
    ValueError
        If ``plan_path`` does not end in ``.yaml/.yml`` or ``.json``.
        Also raised if ``output.save_format`` is not ``"csv"`` or ``"parquet"``.

    Notes
    -----
    The run performs the following steps:

    - Load the study plan and summarize it for the run manifest.
    - Create a unique output directory tagged with timestamp and git commit.
    - Save the recovery config and a JSON manifest for provenance.
    - For each replication:
      * sample subject (and optional population) parameters
      * simulate a full study
      * fit the estimator and record diagnostics
      * collect true vs. estimated parameters (subject-level or within-subject
        constrained parameters, depending on the model wrapper)
    - Write records/metrics tables and optional diagnostics.
    - When ``sampling.mode="fixed"``, write pooled fixed-mode error summaries
      and set correlation metrics to ``NaN`` (correlation is not interpretable
      without true-parameter variation).

    Output files include (when enabled):
    ``parameter_recovery_records.(csv|parquet)``,
    ``parameter_recovery_metrics.csv``,
    ``parameter_recovery_fixed_mode_error_summary.csv`` (fixed mode only),
    ``parameter_recovery_fit_diagnostics.jsonl``,
    ``population_recovery_records.csv``,
    ``population_recovery_metrics.csv``.
    """
    # Load study plan FIRST (so we can compute plan summary + copy plan into out_dir)
    plan_path: Path = Path(config.plan_path)
    plan: StudyPlan = load_study_plan(Path(config.plan_path))

    subject_ids = list(plan.subjects.keys())

    out_dir = make_unique_run_dir(config.output.out_dir)

    # Save config (YAML)
    if config.output.save_config:
        save_config_auto(cfg=config, out_dir=out_dir, stem="parameter_recovery_config")

    # Copy plan file into out_dir for reproducibility
    plan_copied_name: str | None = _safe_copy_file(plan_path, out_dir)

    # Manifest: plan summary + copied plan path
    plan_summary = _plan_summary(plan)
    write_run_manifest(
        out_dir,
        config_obj=config,
        generator=generator,
        model=model,
        estimator=estimator,
        plan_path_copied=plan_copied_name,
        plan_summary=plan_summary,
    )

    n_reps = int(config.n_reps)
    n_jobs = max(1, int(config.n_jobs))

    rng = np.random.default_rng(int(config.seed))
    rep_seeds = [
        int(x)
        for x in rng.integers(
            0,
            2**32 - 1,
            size=max(n_reps, 0),
            dtype=np.uint32,
        )
    ]

    records: list[dict[str, Any]] = []
    fit_diags: list[dict[str, Any]] = []

    completed = 0
    pbar = None
    if progress_callback is not None:
        progress_callback(0, n_reps)
    else:
        pbar = tqdm(
            total=n_reps,
            desc="Parameter recovery",
            disable=not sys.stderr.isatty(),
        )

    def _accumulate_rep_result(res: _ReplicationResult) -> None:
        records.extend(res.records)
        if res.fit_diag is not None:
            fit_diags.append(res.fit_diag)

    def _run_reps_sequential() -> None:
        nonlocal completed
        for rep in range(n_reps):
            rep_result = _run_single_rep(
                rep=rep,
                rep_seed=rep_seeds[rep],
                config=config,
                generator=generator,
                model=model,
                estimator=estimator,
                subject_ids=subject_ids,
                subject_block_plans=plan.subjects,
                out_dir=str(out_dir),
            )
            _accumulate_rep_result(rep_result)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, n_reps)
            elif pbar is not None:
                pbar.update(1)

    try:
        if n_jobs <= 1 or n_reps <= 1:
            _run_reps_sequential()
        else:
            max_workers = min(n_jobs, n_reps)
            try:
                executor = ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_worker_context,
                    initargs=(
                        config,
                        generator,
                        model,
                        estimator,
                        subject_ids,
                        plan.subjects,
                        str(out_dir),
                    ),
                )
            except (PermissionError, OSError) as exc:
                warnings.warn(
                    (
                        f"Parallel parameter recovery unavailable ({exc}). "
                        "Falling back to sequential execution."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                _run_reps_sequential()
            else:
                with executor as ex:
                    futures = [
                        ex.submit(_run_single_rep_from_worker, rep, rep_seeds[rep])
                        for rep in range(n_reps)
                    ]
                    for fut in as_completed(futures):
                        rep_result = fut.result()
                        _accumulate_rep_result(rep_result)
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(completed, n_reps)
                        elif pbar is not None:
                            pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    df = pd.DataFrame.from_records(records)
    df.sort_values(["param", "rep", "subject_id"], inplace=True)

    # Save records
    fmt = config.output.save_format.lower()
    if fmt == "csv":
        df.to_csv(out_dir / "parameter_recovery_records.csv", index=False)
    elif fmt == "parquet":
        df.to_parquet(out_dir / "parameter_recovery_records.parquet", index=False)
    else:
        raise ValueError("output.save_format must be 'csv' or 'parquet'")

    # Save diagnostics
    if config.output.save_fit_diagnostics:
        fit_diags.sort(key=lambda x: int(x.get("rep", -1)))
        with (out_dir / "parameter_recovery_fit_diagnostics.jsonl").open("w", encoding="utf-8") as f:
            for row in fit_diags:
                f.write(json.dumps(row, default=str) + "\n")

    metrics = compute_parameter_recovery_metrics(df)
    if str(config.sampling.mode).lower() == "fixed":
        # In fixed mode true values are constant, so corr is undefined/uninformative.
        if "corr" in metrics.columns:
            metrics = metrics.copy()
            metrics["corr"] = np.nan
        fixed_summary = _compute_fixed_mode_error_summary(df)
        fixed_summary.to_csv(out_dir / "parameter_recovery_fixed_mode_error_summary.csv", index=False)

    metrics.to_csv(out_dir / "parameter_recovery_metrics.csv", index=False)

    # Population-level recovery (hierarchical runs only)
    pop_df, pop_metrics = _maybe_compute_population_recovery(
        fit_diags=fit_diags,
        config=config,
        model=model,
    )
    if pop_df is not None and pop_metrics is not None:
        pop_df.to_csv(out_dir / "population_recovery_records.csv", index=False)
        pop_metrics.to_csv(out_dir / "population_recovery_metrics.csv", index=False)

    return ParameterRecoveryOutputs(
        records=df,
        metrics=metrics,
        out_dir=str(out_dir),
        population_records=pop_df,
        population_metrics=pop_metrics,
    )
