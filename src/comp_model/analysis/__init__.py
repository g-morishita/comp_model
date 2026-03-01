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

__all__ = [
    "PSISLOOResult",
    "ProfileLikelihood1DResult",
    "ProfileLikelihood2DResult",
    "ProfilePoint1D",
    "ProfilePoint2D",
    "ParityBenchmarkResult",
    "ParityCaseResult",
    "ParityFixtureCase",
    "WAICResult",
    "aic",
    "bic",
    "load_parity_fixture_file",
    "psis_loo",
    "profile_likelihood_1d",
    "profile_likelihood_2d",
    "run_parity_benchmark",
    "waic",
    "write_parity_benchmark_csv",
]
