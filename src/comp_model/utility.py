import numpy as np
from .params.bounds import ParameterBoundsSpace


def _softmax(u: np.ndarray, beta: float) -> np.ndarray:
    z = beta * u
    z -= float(np.max(z))
    expz = np.exp(z)
    return expz / float(np.sum(expz))


def _as_scipy_bounds(space: ParameterBoundsSpace) -> list[tuple[float, float]]:
    return [(space.bounds[name].lo, space.bounds[name].hi) for name in space.names]

