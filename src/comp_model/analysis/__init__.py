"""Analysis utilities for model comparison and diagnostics."""

from .information_criteria import PSISLOOResult, WAICResult, aic, bic, psis_loo, waic
from .profile_likelihood import (
    ProfileLikelihood1DResult,
    ProfileLikelihood2DResult,
    ProfilePoint1D,
    ProfilePoint2D,
    profile_likelihood_1d,
    profile_likelihood_2d,
)
from .parity_benchmark import (
    ParityBenchmarkResult,
    ParityCaseResult,
    ParityFixtureCase,
    load_parity_fixture_file,
    run_parity_benchmark,
    write_parity_benchmark_csv,
)
from .parity_cli import run_parity_benchmark_cli
from .parity_matrix import (
    ModelParityMatrixRow,
    ModelParityMatrixSummary,
    build_model_parity_matrix,
    summarize_model_parity_matrix,
    write_model_parity_matrix_csv,
    write_model_parity_matrix_json,
)
from .parity_matrix_cli import run_model_parity_matrix_cli

__all__ = [
    "PSISLOOResult",
    "ProfileLikelihood1DResult",
    "ProfileLikelihood2DResult",
    "ProfilePoint1D",
    "ProfilePoint2D",
    "ModelParityMatrixRow",
    "ModelParityMatrixSummary",
    "ParityBenchmarkResult",
    "ParityCaseResult",
    "ParityFixtureCase",
    "WAICResult",
    "aic",
    "bic",
    "build_model_parity_matrix",
    "load_parity_fixture_file",
    "psis_loo",
    "profile_likelihood_1d",
    "profile_likelihood_2d",
    "run_parity_benchmark",
    "run_parity_benchmark_cli",
    "run_model_parity_matrix_cli",
    "summarize_model_parity_matrix",
    "waic",
    "write_parity_benchmark_csv",
    "write_model_parity_matrix_csv",
    "write_model_parity_matrix_json",
]
