from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import importlib
import json
import secrets
import shutil
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

from comp_model_core.plans.io import load_study_plan_yaml, load_study_plan_json
from comp_model_core.plans.block import StudyPlan, BlockPlan
from comp_model_core.data.types import StudyData
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.register import make_registry
from comp_model_impl.tasks.build import build_runner_for_plan

from comp_model_impl.recovery.parameter.analysis import compute_parameter_recovery_metrics
from comp_model_impl.recovery.parameter.config import ParameterRecoveryConfig, config_to_json
from comp_model_impl.recovery.parameter.plots import (
    plot_parameter_recovery_interactive,
    plot_parameter_recovery_scatter,
    plot_parameter_recovery_scatter_color,
)
from comp_model_impl.recovery.parameter.sampling import sample_subject_params


@dataclass(frozen=True, slots=True)
class ParameterRecoveryOutputs:
    records: pd.DataFrame
    metrics: pd.DataFrame
    out_dir: str


def _find_git_root(start: Path) -> Path | None:
    """Walk upward until we find a .git entry (dir OR file)."""
    p = start.resolve()
    for parent in [p, *p.parents]:
        git_entry = parent / ".git"
        if git_entry.exists():
            return parent
    return None


def _run_git(repo_root: Path, args: list[str]) -> str | None:
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


def git_info_for_module(module_name: str) -> dict[str, Any]:
    """
    Returns commit/branch/dirty for the git repo that contains `module_name`.
    Never raises; returns Nones if not in a git checkout.
    """
    try:
        mod = importlib.import_module(module_name)
        mod_path = Path(mod.__file__).resolve()
    except Exception:
        return {
            f"{module_name}_repo_root": None,
            f"{module_name}_git_commit": None,
            f"{module_name}_git_branch": None,
            f"{module_name}_git_dirty": None,
        }

    repo_root = _find_git_root(mod_path.parent)
    if repo_root is None:
        return {
            f"{module_name}_repo_root": None,
            f"{module_name}_git_commit": None,
            f"{module_name}_git_branch": None,
            f"{module_name}_git_dirty": None,
        }

    commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(repo_root, ["status", "--porcelain"])
    dirty = None if status is None else (len(status) > 0)

    return {
        f"{module_name}_repo_root": str(repo_root),
        f"{module_name}_git_commit": commit,
        f"{module_name}_git_branch": branch,
        f"{module_name}_git_dirty": dirty,
    }


def make_unique_run_dir(base_out_dir: str | Path, *, git_commit: str | None = None) -> Path:
    """
    Creates a unique run directory under base_out_dir and returns it.
    Example name: 2026-01-26_143012__a1b2c3d__7f9c
    """
    base = Path(base_out_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    short = (git_commit or "nogit")[:7]
    suffix = secrets.token_hex(2)  # 4 hex chars

    run_id = f"{ts}__{short}__{suffix}"
    out_dir = base / run_id
    out_dir.mkdir(parents=False, exist_ok=False)
    return out_dir


def _plan_summary(plan: StudyPlan) -> dict[str, Any]:
    """
    Robust summary for StudyPlan where structure is:
      StudyPlan.subjects: Mapping[str, list[BlockPlan]]
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
    git_impl = git_info_for_module("comp_model_impl")

    manifest = {
        "run_id": out_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **git_impl,
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


def run_parameter_recovery(
    *,
    config: ParameterRecoveryConfig,
    generator: Generator,
    model: ComputationalModel,
    estimator: Estimator,
) -> ParameterRecoveryOutputs:
    """
    Full parameter recovery experiment.

    - loads study plan
    - creates a unique output directory for this run
    - writes config + manifest (git + plan summary + copied plan file)
    - for each replication:
        * sample true subject parameters
        * simulate data
        * fit estimator
        * store true vs estimated per subject/param
    - saves records + metrics + optional diagnostics + plots
    """
    # Load study plan FIRST (so we can compute plan summary + copy plan into out_dir)
    plan_path = Path(config.plan_path)
    if plan_path.suffix.lower() in (".yaml", ".yml"):
        plan: StudyPlan = load_study_plan_yaml(str(plan_path))
    elif plan_path.suffix.lower() == ".json":
        plan = load_study_plan_json(str(plan_path))
    else:
        raise ValueError("plan_path must be .yaml/.yml or .json")

    subject_ids = list(plan.subjects.keys())

    # Unique output directory (use comp_model_impl git commit if available)
    git_impl = git_info_for_module("comp_model_impl")
    git_commit = git_impl.get("comp_model_impl_git_commit")
    out_dir = make_unique_run_dir(config.output.out_dir, git_commit=git_commit)

    # Save config
    if config.output.save_config:
        (out_dir / "parameter_recovery_config.json").write_text(
            config_to_json(config),
            encoding="utf-8",
        )

    # Copy plan file into out_dir for reproducibility
    plan_copied_name: str | None = None
    try:
        if plan_path.exists() and plan_path.is_file():
            dest = out_dir / plan_path.name
            shutil.copy2(plan_path, dest)
            plan_copied_name = dest.name
    except Exception:
        plan_copied_name = None

    # Manifest: git + plan summary + copied plan path
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

    rng = np.random.default_rng(int(config.seed))

    records: list[dict[str, Any]] = []
    fit_diags: list[dict[str, Any]] = []

    r = make_registry()

    def block_runner_builder(block_plan: BlockPlan):
        return build_runner_for_plan(plan=block_plan, registries=r)

    for rep in tqdm(range(int(config.n_reps)), desc="Parameter recovery"):
        rep_rng = np.random.default_rng(
            rng.integers(0, 2**32 - 1, dtype=np.uint32)
        )

        subj_params_true, pop_true = sample_subject_params(
            cfg=config.sampling,
            model=model,
            subject_ids=subject_ids,
            rng=rep_rng,
        )

        study: StudyData = generator.simulate_study(
            block_runner_builder=block_runner_builder,
            model=model,
            subj_params=subj_params_true,
            subject_block_plans=plan.subjects,
            rng=rep_rng,
        )

        fit: FitResult = estimator.fit(study=study, rng=rep_rng)

        if config.output.save_simulated_study:
            import pickle
            with (out_dir / f"study_rep_{rep:04d}.pkl").open("wb") as f:
                pickle.dump(study, f)

        if config.output.save_fit_diagnostics:
            fit_diags.append(
                {
                    "rep": rep,
                    "success": bool(getattr(fit, "success", True)),
                    "message": str(getattr(fit, "message", "")),
                    "value": float(fit.value) if getattr(fit, "value", None) is not None else None,
                    "diagnostics": getattr(fit, "diagnostics", None),
                    "population_true": pop_true,
                    "population_hat": getattr(fit, "population_hat", None),
                }
            )

        hats = fit.subject_hats or {}
        for sid in subject_ids:
            theta_true = subj_params_true[sid]
            theta_hat = hats.get(sid, {})
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
        with (out_dir / "parameter_recovery_fit_diagnostics.jsonl").open("w", encoding="utf-8") as f:
            for row in fit_diags:
                f.write(json.dumps(row, default=str) + "\n")

    metrics = compute_parameter_recovery_metrics(df)
    metrics.to_csv(out_dir / "parameter_recovery_metrics.csv", index=False)

    # Plots
    if config.plots.make_plots:
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        alpha = config.plots.scatter_alpha
        max_points = config.plots.max_points

        plot_parameter_recovery_scatter(df=df, out_dir=plot_dir, alpha=alpha, max_points=max_points)
        for mode in ["true", "abs_error", "error"]:
            plot_parameter_recovery_scatter_color(
                df=df, out_dir=plot_dir, color_by=mode, alpha=alpha, max_points=max_points
            )

        try:
            plot_parameter_recovery_interactive(df=df, out_path=plot_dir / "recovery_interactive.html")
        except ImportError:
            pass

    return ParameterRecoveryOutputs(records=df, metrics=metrics, out_dir=str(out_dir))
