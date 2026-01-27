from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm

import json
import secrets
import shutil
import subprocess
import importlib

import numpy as np
import pandas as pd

from .config import ParameterRecoveryConfig, config_to_json
from .sampling import sample_subject_params
from .analysis import compute_parameter_recovery_metrics

from comp_model_core.plans.io import load_study_plan_yaml, load_study_plan_json
from comp_model_core.plans.block import StudyPlan, BlockPlan

from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.data.types import StudyData

from ...tasks.build import build_runner_for_plan
from ...register import make_registry

from .plots import (
    plot_parameter_recovery_scatter,
    plot_parameter_recovery_scatter_color,
    plot_parameter_recovery_interactive,
)


@dataclass(frozen=True, slots=True)
class ParameterRecoveryOutputs:
    records: pd.DataFrame
    metrics: pd.DataFrame
    out_dir: str


def _find_git_root(start: Path) -> Path | None:
    p = start.resolve()
    for parent in [p, *p.parents]:
        git_entry = parent / ".git"
        if git_entry.exists():  # dir or file (submodules often use a file)
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
    n_subjects = len(plan.subjects)
    n_blocks = len(plan.blocks)

    trials_by_block = []
    for b in plan.blocks:
        trials_by_block.append(int(getattr(b, "n_trials", 0)))

    # total trials per subject (robust to list/dict)
    totals: list[int] = []
    for sid, bps in plan.subjects.items():
        if isinstance(bps, dict):
            blocks = list(bps.values())
        else:
            blocks = list(bps)
        totals.append(int(sum(getattr(bp, "n_trials", 0) for bp in blocks)))

    total_trials_min = int(min(totals)) if totals else 0
    total_trials_max = int(max(totals)) if totals else 0
    total_trials_mean = float(np.mean(totals)) if totals else 0.0

    return {
        "n_subjects": int(n_subjects),
        "n_blocks": int(n_blocks),
        "trials_by_block": trials_by_block,
        "total_trials_per_subject_min": total_trials_min,
        "total_trials_per_subject_mean": total_trials_mean,
        "total_trials_per_subject_max": total_trials_max,
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
    # ---- Load study plan first (so we can compute summary + copy the plan) ----
    plan_path = Path(config.plan_path)
    if plan_path.suffix.lower() in (".yaml", ".yml"):
        plan: StudyPlan = load_study_plan_yaml(str(plan_path))
    elif plan_path.suffix.lower() == ".json":
        plan = load_study_plan_json(str(plan_path))
    else:
        raise ValueError("plan_path must be .yaml/.yml or .json")

    subject_ids = list(plan.subjects.keys())

    # ---- Unique output folder (include comp_model_impl commit in folder name) ----
    git_impl = git_info_for_module("comp_model_impl")
    git_commit = git_impl.get("comp_model_impl_git_commit")
    out_dir = make_unique_run_dir(config.output.out_dir, git_commit=git_commit)

    # ---- Save config ----
    if config.output.save_config:
        (out_dir / "parameter_recovery_config.json").write_text(config_to_json(config), encoding="utf-8")

    # ---- Copy plan file into out_dir for reproducibility ----
    plan_copied_name = None
    try:
        if plan_path.exists() and plan_path.is_file():
            dest = out_dir / plan_path.name
            shutil.copy2(plan_path, dest)
            plan_copied_name = dest.name
    except Exception:
        plan_copied_name = None

    # ---- Manifest with summary (blocks/subjects/trials) + git commit ----
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
        rep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1, dtype=np.uint32))

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

    fmt = config.output.save_format.lower()
    if fmt == "csv":
        df.to_csv(out_dir / "parameter_recovery_records.csv", index=False)
    elif fmt == "parquet":
        df.to_parquet(out_dir / "parameter_recovery_records.parquet", index=False)
    else:
        raise ValueError("output.save_format must be 'csv' or 'parquet'")

    if config.output.save_fit_diagnostics:
        with (out_dir / "parameter_recovery_fit_diagnostics.jsonl").open("w", encoding="utf-8") as f:
            for row in fit_diags:
                f.write(json.dumps(row, default=str) + "\n")

    metrics = compute_parameter_recovery_metrics(df)
    metrics.to_csv(out_dir / "parameter_recovery_metrics.csv", index=False)

    if config.plots.make_plots:
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        alpha = config.plots.scatter_alpha
        max_points = config.plots.max_points

        plot_parameter_recovery_scatter(df=df, out_dir=plot_dir, alpha=alpha, max_points=max_points)
        for mode in ["true", "abs_error", "error"]:
            plot_parameter_recovery_scatter_color(df=df, out_dir=plot_dir, color_by=mode, alpha=alpha, max_points=max_points)
        try:
            plot_parameter_recovery_interactive(df=df, out_path=plot_dir / "recovery_interactive.html")
        except ImportError:
            pass

    return ParameterRecoveryOutputs(records=df, metrics=metrics, out_dir=str(out_dir))
