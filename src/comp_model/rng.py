from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class RNG:
    """Simple wrapper to standardize RNG usage across the library."""
    seed: int | None = None

    def numpy(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)