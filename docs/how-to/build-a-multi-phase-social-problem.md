# How-to: Build a Multi-Phase Social Problem

Use this guide when a trial includes multiple decision phases (for example
demonstrator phase plus subject phase).

## Why Trial Programs

Multi-phase problems should be represented by a `TrialProgram` with explicit
decision nodes and actor identities. This keeps timing semantics auditable and
replay-consistent.

## 1. Implement a `TrialProgram`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.events import SimulationEvent
from comp_model.runtime.program import DecisionNode, TrialProgram


@dataclass(slots=True)
class ThreePhaseSocialProgram(TrialProgram):
    reward_probabilities: tuple[float, float]

    def reset(self, *, rng: np.random.Generator) -> None:
        self._rng = rng

    def decision_nodes(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[SimulationEvent],
    ) -> tuple[DecisionNode, ...]:
        del trial_events
        return (
            DecisionNode(node_id="demo_phase", actor_id="demonstrator"),
            DecisionNode(node_id="subject_phase", actor_id="subject"),
            DecisionNode(node_id="post_phase", actor_id="subject"),
        )

    def available_actions(
        self,
        *,
        trial_index: int,
        node: DecisionNode,
        trial_events: Sequence[SimulationEvent],
    ) -> tuple[int, int]:
        del trial_index, node, trial_events
        return (0, 1)

    def observe(
        self,
        *,
        trial_index: int,
        node: DecisionNode,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
    ) -> dict[str, Any]:
        return {"trial": trial_index, "node_id": node.node_id}

    def transition(
        self,
        action: Any,
        *,
        trial_index: int,
        node: DecisionNode,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        reward = 1.0 if rng.random() < self.reward_probabilities[int(action)] else 0.0
        return {"reward": reward, "node_id": node.node_id}
```

## 2. Run with Multiple Actors

```python
from comp_model.models import UniformRandomPolicyModel
from comp_model.runtime import SimulationConfig, run_trial_program

program = ThreePhaseSocialProgram(reward_probabilities=(0.3, 0.7))
trace = run_trial_program(
    program=program,
    models={
        "subject": UniformRandomPolicyModel(),
        "demonstrator": UniformRandomPolicyModel(),
    },
    config=SimulationConfig(n_trials=20, seed=42),
)
```

## 3. Verify Timing Semantics

Check each trial has node order you intend:

- demonstrator node first,
- subject nodes after.

Add tests asserting actor/node order and event-phase order.

## 4. Use in Recovery/Generation Workflows

For config-driven social simulation, connect this logic through a generator
component and register it in plugins.
