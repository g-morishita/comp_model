# How-to: Build a Multi-Phase Social Problem

Use this guide when a trial includes multiple decision phases (for example
demonstrator phase plus subject phase).

## Why Trial Programs

Multi-phase problems should be represented by a `TrialProgram` with explicit
ordered steps, node identities, and actor identities. This keeps timing
semantics auditable and replay-consistent, even when one actor observes or
updates before another actor makes a decision.

## 1. Implement a `TrialProgram`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.events import EventPhase, SimulationEvent
from comp_model.runtime.program import ProgramStep, TrialProgram


@dataclass(slots=True)
class ThreePhaseSocialProgram(TrialProgram):
    reward_probabilities: tuple[float, float]

    def reset(self, *, rng: np.random.Generator) -> None:
        self._rng = rng

    def trial_steps(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[SimulationEvent],
    ) -> tuple[ProgramStep, ...]:
        del trial_index, trial_events
        return (
            ProgramStep(EventPhase.OBSERVATION, node_id="demo_phase", actor_id="demonstrator"),
            ProgramStep(EventPhase.DECISION, node_id="demo_phase", actor_id="demonstrator"),
            ProgramStep(EventPhase.OUTCOME, node_id="demo_phase", actor_id="demonstrator"),
            ProgramStep(
                EventPhase.UPDATE,
                node_id="demo_phase",
                actor_id="demonstrator",
                learner_id="subject",
            ),
            ProgramStep(EventPhase.OBSERVATION, node_id="subject_phase", actor_id="subject"),
            ProgramStep(EventPhase.DECISION, node_id="subject_phase", actor_id="subject"),
            ProgramStep(EventPhase.OUTCOME, node_id="subject_phase", actor_id="subject"),
            ProgramStep(EventPhase.UPDATE, node_id="subject_phase", actor_id="subject"),
        )

    def available_actions(
        self,
        *,
        trial_index: int,
        step: ProgramStep,
        trial_events: Sequence[SimulationEvent],
    ) -> tuple[int, int]:
        del trial_index, step, trial_events
        return (0, 1)

    def observe(
        self,
        *,
        trial_index: int,
        step: ProgramStep,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
    ) -> dict[str, Any]:
        del context
        if step.actor_id == "demonstrator":
            return {"trial": trial_index, "stage": "demonstrator"}

        demonstrator_action = next(
            event.payload["action"]
            for event in trial_events
            if event.phase is EventPhase.DECISION and event.payload["node_id"] == "demo_phase"
        )
        return {
            "trial": trial_index,
            "stage": "subject",
            "demonstrator_action": demonstrator_action,
        }

    def transition(
        self,
        action: Any,
        *,
        trial_index: int,
        step: ProgramStep,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        del trial_index, trial_events
        reward = 1.0 if rng.random() < self.reward_probabilities[int(action)] else 0.0
        return {"reward": reward, "node_id": step.node_id, "actor_id": context.actor_id}
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

Check each trial has step order you intend:

- demonstrator observation/decision/outcome first,
- any social update steps next,
- subject observation/decision after that.

Add tests asserting actor/node order and event-phase order.

## 4. Use in Recovery/Generation Workflows

For config-driven social simulation, connect this logic through a generator
component and register it in plugins.
