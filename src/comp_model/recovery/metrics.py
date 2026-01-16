from __future__ import annotations

from typing import Mapping, Sequence
import math

from .schema import RecoveryResult


def rmse(xs: Sequence[float], ys: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(xs, ys, strict=True)) / len(xs))


def subject_param_rmse(res: RecoveryResult, param: str) -> float | None:
    if res.estimate.subjects is None:
        return None
    xs, ys = [], []
    for sid, true_p in res.truth.subjects.items():
        if sid not in res.estimate.subjects:
            continue
        if param not in true_p or param not in res.estimate.subjects[sid]:
            continue
        xs.append(float(true_p[param]))
        ys.append(float(res.estimate.subjects[sid][param]))
    return rmse(xs, ys) if xs else None


def population_param_error(res: RecoveryResult, pop_key: str) -> float | None:
    if res.truth.population is None or res.estimate.population is None:
        return None
    if pop_key not in res.truth.population or pop_key not in res.estimate.population:
        return None
    return float(res.estimate.population[pop_key] - res.truth.population[pop_key])
