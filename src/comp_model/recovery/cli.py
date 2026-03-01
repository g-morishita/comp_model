"""CLI helpers for config-driven recovery workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .config import load_json_config, run_model_recovery_from_config, run_parameter_recovery_from_config
from .serialization import (
    write_model_recovery_cases_csv,
    write_model_recovery_confusion_csv,
    write_parameter_recovery_csv,
)


def run_recovery_cli(argv: Sequence[str] | None = None) -> int:
    """Run parameter/model recovery from a JSON config path.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        CLI argument list. When ``None``, process arguments are used.

    Returns
    -------
    int
        Exit code (`0` on success).
    """

    parser = argparse.ArgumentParser(description="Run comp_model recovery workflow from JSON config.")
    parser.add_argument("--config", required=True, help="Path to recovery JSON config.")
    parser.add_argument(
        "--mode",
        choices=("auto", "parameter", "model"),
        default="auto",
        help="Recovery mode. 'auto' infers from config keys.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="recovery",
        help="Output filename prefix.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_json_config(args.config)
    mode = _resolve_mode(config, requested_mode=str(args.mode))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)

    if mode == "parameter":
        result = run_parameter_recovery_from_config(config)
        case_path = write_parameter_recovery_csv(
            result,
            output_dir / f"{prefix}_parameter_cases.csv",
        )
        summary_path = _write_json_summary(
            output_dir / f"{prefix}_parameter_summary.json",
            {
                "mode": "parameter",
                "n_cases": len(result.cases),
                "mean_absolute_error": result.mean_absolute_error,
                "mean_signed_error": result.mean_signed_error,
            },
        )
        print(f"Parameter recovery complete: n_cases={len(result.cases)}")
        print(f"Cases CSV: {case_path}")
        print(f"Summary JSON: {summary_path}")
        return 0

    result = run_model_recovery_from_config(config)
    case_path = write_model_recovery_cases_csv(
        result,
        output_dir / f"{prefix}_model_cases.csv",
    )
    confusion_path = write_model_recovery_confusion_csv(
        result,
        output_dir / f"{prefix}_model_confusion.csv",
    )
    summary_path = _write_json_summary(
        output_dir / f"{prefix}_model_summary.json",
        {
            "mode": "model",
            "criterion": result.criterion,
            "n_cases": len(result.cases),
            "confusion_matrix": result.confusion_matrix,
        },
    )
    print(
        "Model recovery complete: "
        f"n_cases={len(result.cases)}, criterion={result.criterion}"
    )
    print(f"Cases CSV: {case_path}")
    print(f"Confusion CSV: {confusion_path}")
    print(f"Summary JSON: {summary_path}")
    return 0


def _resolve_mode(config: dict[str, Any], *, requested_mode: str) -> str:
    """Resolve recovery mode from requested value and config shape."""

    if requested_mode in {"parameter", "model"}:
        return requested_mode

    # Auto mode uses top-level key signatures from strict config schemas.
    if {"generating_model", "fitting_model", "true_parameter_sets"}.issubset(config):
        return "parameter"
    if {"generating", "candidates"}.issubset(config):
        return "model"
    raise ValueError(
        "could not infer recovery mode from config; pass --mode explicitly"
    )


def _write_json_summary(path: Path, payload: dict[str, Any]) -> Path:
    """Write summary JSON payload to disk."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    """Execute recovery CLI and exit with returned code."""

    raise SystemExit(run_recovery_cli())


if __name__ == "__main__":
    main()


__all__ = ["main", "run_recovery_cli"]
