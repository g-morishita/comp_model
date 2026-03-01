"""CLI helpers for config-driven model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from comp_model.core import load_config_mapping

from .tabular import fit_study_csv_from_config, fit_trial_csv_from_config


def run_fit_cli(argv: Sequence[str] | None = None) -> int:
    """Run config-driven fitting from tabular CSV input.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        CLI argument list. When ``None``, process arguments are used.

    Returns
    -------
    int
        Exit code (`0` on success).
    """

    parser = argparse.ArgumentParser(description="Run comp_model fitting from CSV and config.")
    parser.add_argument("--config", required=True, help="Path to fitting JSON or YAML config.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--input-kind",
        choices=("trial", "study"),
        default="trial",
        help="Input CSV type.",
    )
    parser.add_argument(
        "--level",
        choices=("auto", "dataset", "subject", "study"),
        default="auto",
        help="Fitting aggregation level.",
    )
    parser.add_argument(
        "--subject-id",
        default=None,
        help="Subject ID used when --level subject and --input-kind study.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output summary JSON.",
    )
    parser.add_argument(
        "--prefix",
        default="fit",
        help="Output summary filename prefix.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = _load_config_mapping(args.config)
    input_kind = str(args.input_kind)
    level = _resolve_level(input_kind=input_kind, level=str(args.level))

    if input_kind == "trial":
        result = fit_trial_csv_from_config(str(args.input_csv), config=config)
    else:
        study_level: Literal["study", "subject"] = "study" if level == "study" else "subject"
        result = fit_study_csv_from_config(
            str(args.input_csv),
            config=config,
            level=study_level,
            subject_id=args.subject_id,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{str(args.prefix)}_summary.json"
    summary = {
        "input_kind": input_kind,
        "level": level,
        **_fit_result_summary(result),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Fit complete: input_kind={input_kind}, level={level}")
    print(f"Summary JSON: {summary_path}")
    return 0


def _resolve_level(*, input_kind: str, level: str) -> str:
    """Resolve level defaults and validate level/input combinations."""

    if input_kind == "trial":
        if level in {"auto", "dataset"}:
            return "dataset"
        raise ValueError("--level must be 'auto' or 'dataset' for --input-kind trial")

    if level == "auto":
        return "study"
    if level not in {"study", "subject"}:
        raise ValueError("--level must be one of {'auto', 'study', 'subject'} for --input-kind study")
    return level


def _fit_result_summary(result: Any) -> dict[str, Any]:
    """Build compact JSON-serializable summary from heterogeneous fit outputs."""

    # MLE-style results.
    if hasattr(result, "best") and hasattr(result.best, "params"):
        best = result.best
        return {
            "result_type": type(result).__name__,
            "best_log_likelihood": float(best.log_likelihood),
            "best_params": {key: float(value) for key, value in best.params.items()},
        }

    # MAP-style single-dataset results.
    if hasattr(result, "map_candidate") and hasattr(result.map_candidate, "params"):
        candidate = result.map_candidate
        return {
            "result_type": type(result).__name__,
            "map_log_likelihood": float(candidate.log_likelihood),
            "map_log_prior": float(candidate.log_prior),
            "map_log_posterior": float(candidate.log_posterior),
            "map_params": {key: float(value) for key, value in candidate.params.items()},
        }

    # MCMC single-dataset results.
    if hasattr(result, "posterior_samples") and hasattr(result, "draws"):
        map_candidate = result.map_candidate
        return {
            "result_type": type(result).__name__,
            "n_draws": int(len(result.draws)),
            "parameter_names": list(result.posterior_samples.parameter_names),
            "map_log_posterior": float(map_candidate.log_posterior),
        }

    # Study-level aggregate results (MLE/MAP/MCMC/hierarchical wrappers).
    if hasattr(result, "subject_results"):
        out: dict[str, Any] = {
            "result_type": type(result).__name__,
            "n_subjects": int(len(result.subject_results)),
        }
        if hasattr(result, "total_log_likelihood"):
            out["total_log_likelihood"] = float(result.total_log_likelihood)
        if hasattr(result, "total_log_posterior"):
            out["total_log_posterior"] = float(result.total_log_posterior)
        if hasattr(result, "total_map_log_posterior"):
            out["total_map_log_posterior"] = float(result.total_map_log_posterior)
        return out

    # Subject-level aggregate results.
    if hasattr(result, "subject_id"):
        out = {
            "result_type": type(result).__name__,
            "subject_id": str(result.subject_id),
        }
        if hasattr(result, "total_log_likelihood"):
            out["total_log_likelihood"] = float(result.total_log_likelihood)
        if hasattr(result, "total_log_posterior"):
            out["total_log_posterior"] = float(result.total_log_posterior)
        if hasattr(result, "draws"):
            out["n_draws"] = int(len(result.draws))
        return out

    return {"result_type": type(result).__name__}


def _load_config_mapping(path: str | Path) -> dict[str, Any]:
    """Load one config object from JSON or YAML path."""

    return load_config_mapping(path)


def main() -> None:
    """Execute fit CLI and exit with returned code."""

    raise SystemExit(run_fit_cli())


if __name__ == "__main__":
    main()


__all__ = ["main", "run_fit_cli"]
