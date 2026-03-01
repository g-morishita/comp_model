"""CLI helpers for exporting model parity matrix artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .parity_matrix import (
    build_model_parity_matrix,
    summarize_model_parity_matrix,
    write_model_parity_matrix_csv,
    write_model_parity_matrix_json,
)


def run_model_parity_matrix_cli(argv: Sequence[str] | None = None) -> int:
    """Run parity-matrix export CLI flow.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        CLI argument list. When ``None``, process arguments are used.

    Returns
    -------
    int
        Exit code (`0` when matrix is fully valid, `1` otherwise).
    """

    parser = argparse.ArgumentParser(description="Export model parity matrix artifacts.")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path.")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_json = str(args.output_json).strip() if args.output_json is not None else ""
    output_csv = str(args.output_csv).strip() if args.output_csv is not None else ""
    if not output_json and not output_csv:
        raise ValueError("at least one of --output-json or --output-csv is required")

    rows = build_model_parity_matrix()
    summary = summarize_model_parity_matrix(rows)

    if output_json:
        json_path = write_model_parity_matrix_json(rows, Path(output_json))
        print(f"Parity matrix JSON: {json_path}")
    if output_csv:
        csv_path = write_model_parity_matrix_csv(rows, Path(output_csv))
        print(f"Parity matrix CSV: {csv_path}")

    print(
        "Parity matrix summary: "
        f"rows={summary.n_rows}, implemented={summary.n_implemented}, "
        f"planned={summary.n_planned}, invalid={summary.n_invalid}"
    )
    return 0 if summary.n_invalid == 0 else 1


def main() -> None:
    """Execute parity matrix CLI and exit with returned code."""

    raise SystemExit(run_model_parity_matrix_cli())


if __name__ == "__main__":
    main()


__all__ = ["main", "run_model_parity_matrix_cli"]
