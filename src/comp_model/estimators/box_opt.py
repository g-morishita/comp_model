from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class RandomRestartCoordinateAscentBox:
    """
    Box-constrained coordinate ascent (NumPy-only).

    - random restarts from uniform-in-bounds
    - coordinate search with step shrinking
    - each proposal is clipped to bounds
    """
    n_starts: int = 50
    n_iters: int = 200
    step0: float = 0.2
    step_shrink: float = 0.85
    min_step: float = 1e-4

    def maximize(self, f, rng: np.random.Generator, space) -> tuple[np.ndarray, float]:
        dim = space.dim()
        best_x = None
        best_val = -float("inf")

        # random starts
        for _ in range(self.n_starts):
            x = space.sample_init(rng)
            x = space.clip_vec(x)
            v = float(f(x))
            if v > best_val:
                best_val = v
                best_x = x

        assert best_x is not None
        x = best_x.copy()
        val = best_val
        step = self.step0

        for _ in range(self.n_iters):
            improved = False
            for j in range(dim):
                for sgn in (+1.0, -1.0):
                    x2 = x.copy()
                    x2[j] += sgn * step
                    x2 = space.clip_vec(x2)
                    v2 = float(f(x2))
                    if v2 > val:
                        x, val = x2, v2
                        improved = True
            if not improved:
                step *= self.step_shrink
                if step < self.min_step:
                    break

        return x, val
