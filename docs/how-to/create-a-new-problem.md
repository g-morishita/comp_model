# How-to: Create a New Problem

Use this guide to add a new `DecisionProblem` implementation.

## When to Use This

- Your task has one decision per trial.
- You can express task dynamics with `available_actions`, `observe`, and
  `transition`.

For multi-phase tasks with multiple decision nodes per trial, use
[Build a Multi-Phase Social Problem](build-a-multi-phase-social-problem.md).

## 1. Implement `DecisionProblem`

Create a new module under `src/comp_model/problems/`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from comp_model.core.contracts import DecisionContext, DecisionProblem


@dataclass(slots=True)
class VolatileBanditProblem(DecisionProblem[int, int, dict[str, float]]):
    reward_probabilities: tuple[float, float]
    volatility: float = 0.01

    def reset(self, *, rng: np.random.Generator) -> None:
        self._rng = rng

    def available_actions(self, *, trial_index: int) -> tuple[int, int]:
        return (0, 1)

    def observe(self, *, context: DecisionContext[int]) -> int:
        return 0

    def transition(
        self,
        action: int,
        *,
        context: DecisionContext[int],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        # Example: slow random drift in reward probabilities.
        p0, p1 = self.reward_probabilities
        p0 = float(np.clip(p0 + rng.normal(0.0, self.volatility), 0.0, 1.0))
        p1 = float(np.clip(p1 + rng.normal(0.0, self.volatility), 0.0, 1.0))
        self.reward_probabilities = (p0, p1)
        reward = 1.0 if rng.random() < self.reward_probabilities[action] else 0.0
        return {"reward": reward, "state": 0.0}
```

## 2. Add Plugin Manifest

Expose factory + manifest in the same module:

```python
from comp_model.plugins import ComponentManifest

def create_volatile_bandit_problem(
    reward_probabilities: tuple[float, float] = (0.4, 0.6),
    volatility: float = 0.01,
) -> VolatileBanditProblem:
    return VolatileBanditProblem(
        reward_probabilities=reward_probabilities,
        volatility=volatility,
    )

PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="problem",
        component_id="volatile_bandit",
        factory=create_volatile_bandit_problem,
        description="Two-armed bandit with drifting reward probabilities",
    ),
]
```

## 3. Add Tests

Minimum tests:

- problem reset/transition deterministic behavior under seed,
- `run_episode` executes without errors,
- trace validates under canonical event semantics.

## 4. Verify Integration

```bash
python -m pytest -q
```

Then verify plugin creation:

```python
from comp_model.plugins import build_default_registry

registry = build_default_registry()
problem = registry.create_problem("volatile_bandit")
print(type(problem).__name__)
```
