"""Run parameter recovery for the MVS lottery-choice model.

Example:
  python example/lottery_mvs/run_lottery_mvs_recovery.py \
    --config example/lottery_mvs/lottery_mvs_recovery.yaml --plots
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from comp_model_analysis import plot_parameter_recovery
from comp_model_impl.estimators import (
    BoxMLESubjectwiseEstimator,
    StanHierarchicalNUTSEstimator,
    StanNUTSSubjectwiseEstimator,
    TransformedMLESubjectwiseEstimator,
)
from comp_model_impl.generators.event_log import EventLogAsocialGenerator
from comp_model_impl.models import MVS
from comp_model_impl.recovery.parameter.config import load_parameter_recovery_config
from comp_model_impl.recovery.parameter.run import run_parameter_recovery

DEFAULT_CONFIG = Path(__file__).with_name("lottery_mvs_recovery.yaml")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML is required. Install: pip install pyyaml") from e

    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/object: {p}")
    return obj


def _build_model(model_cfg: Mapping[str, Any]) -> MVS:
    base_name = str(model_cfg.get("base_model", "MVS"))
    if base_name != "MVS":
        raise ValueError(
            "This runner is written for base_model=MVS. "
            f"Got: {base_name!r}"
        )
    base_kwargs = dict(model_cfg.get("base_model_config", {}) or {})
    return MVS(**base_kwargs)


def _build_estimator(est_cfg: Mapping[str, Any], *, model: MVS):
    est_type = str(est_cfg.get("type", "transformed_mle")).lower()
    if est_type == "transformed_mle":
        return TransformedMLESubjectwiseEstimator(
            model=model,
            n_starts=int(est_cfg.get("n_starts", 10)),
            method=str(est_cfg.get("method", "L-BFGS-B")),
            maxiter=int(est_cfg.get("maxiter", 300)),
            z_init_scale=float(est_cfg.get("z_init_scale", 1.0)),
            return_uncertainty=bool(est_cfg.get("return_uncertainty", False)),
            uncertainty_ci=float(est_cfg.get("uncertainty_ci", 0.95)),
            uncertainty_fd_step=float(est_cfg.get("uncertainty_fd_step", 1e-5)),
        )
    if est_type == "box_mle":
        return BoxMLESubjectwiseEstimator(
            model=model,
            n_starts=int(est_cfg.get("n_starts", 10)),
            method=str(est_cfg.get("method", "L-BFGS-B")),
            maxiter=int(est_cfg.get("maxiter", 300)),
            validate_bounds_on_set=bool(est_cfg.get("validate_bounds_on_set", False)),
            return_uncertainty=bool(est_cfg.get("return_uncertainty", False)),
            uncertainty_ci=float(est_cfg.get("uncertainty_ci", 0.95)),
        )
    if est_type in (
        "stan_nuts_subjectwise",
        "stan_subjectwise_nuts",
        "stan_nuts_indiv",
        "bayes_subjectwise",
    ):
        priors = est_cfg.get("priors", None)
        if priors is None:
            raise ValueError(
                "Subjectwise Stan NUTS requires estimator.priors in config "
                "(keys: lambda_var, delta, beta)."
            )
        return StanNUTSSubjectwiseEstimator(
            model=model,
            priors=priors,
            chains=int(est_cfg.get("chains", 4)),
            iter_warmup=int(est_cfg.get("iter_warmup", 500)),
            iter_sampling=int(est_cfg.get("iter_sampling", 1000)),
            adapt_delta=float(est_cfg.get("adapt_delta", 0.9)),
            max_treedepth=int(est_cfg.get("max_treedepth", 12)),
            forbid_extra_priors=bool(est_cfg.get("forbid_extra_priors", True)),
            show_progress=bool(est_cfg.get("show_progress", False)),
            return_posterior_summary=bool(est_cfg.get("return_posterior_summary", False)),
        )
    if est_type in (
        "stan_hierarchical_nuts",
        "stan_nuts_hierarchical",
        "bayes_hierarchical",
        "bayesian_hierarchical",
    ):
        hyper_priors = est_cfg.get("hyper_priors", None)
        if hyper_priors is None:
            raise ValueError(
                "Hierarchical Stan NUTS requires estimator.hyper_priors in config "
                "(keys: mu_*/sd_* for lambda_var, delta, beta)."
            )
        return StanHierarchicalNUTSEstimator(
            model=model,
            hyper_priors=hyper_priors,
            chains=int(est_cfg.get("chains", 4)),
            iter_warmup=int(est_cfg.get("iter_warmup", 800)),
            iter_sampling=int(est_cfg.get("iter_sampling", 1200)),
            adapt_delta=float(est_cfg.get("adapt_delta", 0.92)),
            max_treedepth=int(est_cfg.get("max_treedepth", 12)),
            forbid_extra_priors=bool(est_cfg.get("forbid_extra_priors", True)),
            show_progress=bool(est_cfg.get("show_progress", False)),
            subject_point_estimate=str(est_cfg.get("subject_point_estimate", "mean")),
            cond_map_method=str(est_cfg.get("cond_map_method", "L-BFGS-B")),
            cond_map_maxiter=int(est_cfg.get("cond_map_maxiter", 300)),
            cond_map_n_starts=int(est_cfg.get("cond_map_n_starts", 1)),
            cond_map_jitter=float(est_cfg.get("cond_map_jitter", 0.1)),
            return_posterior_summary=bool(est_cfg.get("return_posterior_summary", False)),
        )
    raise ValueError(
        f"Unsupported estimator.type={est_type!r}. "
        "Expected one of: 'transformed_mle', 'box_mle', "
        "'stan_nuts_subjectwise', 'stan_hierarchical_nuts'."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the lottery MVS parameter recovery YAML.",
    )
    ap.add_argument(
        "--n-reps",
        type=int,
        default=None,
        help="Optional override for config n_reps.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for config seed.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional override for config output.out_dir.",
    )
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Generate recovery plots after the run.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    raw = _load_yaml(cfg_path)

    rec_cfg = load_parameter_recovery_config(cfg_path)
    plan_p = Path(rec_cfg.plan_path)
    if not plan_p.is_absolute():
        plan_p = (cfg_path.parent / plan_p).resolve()
    rec_cfg = replace(rec_cfg, plan_path=str(plan_p))

    if args.n_reps is not None:
        rec_cfg = replace(rec_cfg, n_reps=int(args.n_reps))
    if args.seed is not None:
        rec_cfg = replace(rec_cfg, seed=int(args.seed))
    if args.out_dir is not None:
        rec_cfg = replace(rec_cfg, output=replace(rec_cfg.output, out_dir=str(args.out_dir)))

    model = _build_model(raw.get("model", {}) or {})
    estimator = _build_estimator(raw.get("estimator", {}) or {}, model=model)
    generator = EventLogAsocialGenerator()

    out = run_parameter_recovery(
        config=rec_cfg,
        generator=generator,
        model=model,
        estimator=estimator,
    )

    print("\nDone.")
    print(f"Run directory: {out.out_dir}")
    if args.plots:
        plot_cfg = raw.get("plots", {}) or {}
        plot_paths = plot_parameter_recovery(
            run_dir=out.out_dir,
            scatter_alpha=float(plot_cfg.get("scatter_alpha", 0.6)),
            max_points=int(plot_cfg.get("max_points", 50000)),
            split_by_rep=True,
        )
        if plot_paths:
            print("Plots:")
            for k, p in plot_paths.items():
                print(f"  - {k}: {p}")
    print("Artifacts:")
    print("  - parameter_recovery_records.csv")
    print("  - parameter_recovery_metrics.csv")
    print("  - parameter_recovery_fit_diagnostics.jsonl (if enabled)")
    print("  - plots/ (if enabled)")


if __name__ == "__main__":
    main()
