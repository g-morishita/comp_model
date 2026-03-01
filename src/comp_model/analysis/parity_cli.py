"""CLI helpers for running parity benchmarks from fixture files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .parity_benchmark import (
    load_parity_fixture_file,
    run_parity_benchmark,
    write_parity_benchmark_csv,
)


def run_parity_benchmark_cli(argv: Sequence[str] | None = None) -> int:
    """Run parity benchmark CLI flow.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        CLI argument list. When ``None``, reads process args.

    Returns
    -------
    int
        Exit code (`0` on success, `1` on benchmark failures).
    """

    parser = argparse.ArgumentParser(description="Run v1 fixture parity benchmark.")
    parser.add_argument("--fixture", required=True, help="Path to parity fixture JSON file.")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV report.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cases = load_parity_fixture_file(Path(args.fixture))
    result = run_parity_benchmark(
        cases,
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    output_path = write_parity_benchmark_csv(result, Path(args.output_csv))

    print(
        "Parity benchmark complete: "
        f"{result.n_passed}/{result.n_cases} passed "
        f"(failed={result.n_failed}, atol={result.atol}, rtol={result.rtol})"
    )
    print(f"CSV report: {output_path}")
    return 0 if result.n_failed == 0 else 1


def main() -> None:
    """Execute the parity benchmark CLI and exit with its return code.

    This helper exists so the module can be used with both script entry points
    and ``python -m`` execution.
    """

    raise SystemExit(run_parity_benchmark_cli())


if __name__ == "__main__":
    main()


__all__ = ["main", "run_parity_benchmark_cli"]
