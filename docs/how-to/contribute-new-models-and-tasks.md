# How-to: Step-by-Step Contribution (Task + Model + Hierarchical Bayesian)

Use this page as a strict checklist from local setup to merge.

## 1. Check where you are

Open a terminal and run these commands to check if you are at the project root:

```bash
pwd  # Show where you are
git rev-parse --show-toplevel  # Show the path to the project root
git status --short  # Check if there is no modified file.
```

Expected:

- `pwd` is inside the `comp_model` repository.
- `git rev-parse --show-toplevel` points to the repo root.
- `git status --short` shows what is already modified before you start.

## 2. Create a Git Branch

For the pull request (PR), make a new branch. 

```bash
git checkout -b feat/add-reversal-task-and-sticky-rl-model
```

Use one branch for one coherent contribution.

## 3. Decide the Task to Implement (Concept + Math)

Example task: `reversal_bandit`.

Concept:

- 2-armed bandit.
- Reward contingencies switch (reverse) at fixed trial indices.

Math:

- Contingency state:
  `z_t in {0, 1}`.
- Reversal dynamics:
  `z_t` flips at scheduled reversal trials.
- Reward:
  `p_t(a=0) = p_high` if `z_t=0`, else `p_low`;
  `p_t(a=1) = p_low` if `z_t=0`, else `p_high`.
- Outcome sampling:
  `r_t ~ Bernoulli(p_{t,a_t})`.

## 4. Create Task Files

Create or edit:

- `src/comp_model/problems/reversal_bandit.py`
- `src/comp_model/problems/__init__.py`
- `tests/test_reversal_bandit_problem.py`

In `reversal_bandit.py`, add:

- `ReversalBanditProblem` class
- `create_reversal_bandit_problem(...)` factory
- `PLUGIN_MANIFESTS` with `component_id="reversal_bandit"`

Task registration is required.
If `PLUGIN_MANIFESTS` is missing, `build_default_registry()` cannot discover
your task, and config-driven workflows will fail on unknown component ID.

Why registration is needed:

- The registry discovers components by scanning each module for
  `PLUGIN_MANIFESTS` entries.
- Config files only provide a string `component_id` (for example
  `"reversal_bandit"`), so runtime needs a registered mapping from that string
  to your factory function.
- The same registry mapping is also used by direct ID-based creation
  (`registry.create_problem("reversal_bandit")`), not only by config parsing.

Registration snippet (`src/comp_model/problems/reversal_bandit.py`):

```python
from comp_model.plugins import ComponentManifest


def create_reversal_bandit_problem(...) -> ReversalBanditProblem:
    return ReversalBanditProblem(...)


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="problem",
        component_id="reversal_bandit",
        factory=create_reversal_bandit_problem,
        description="Two-armed bandit with scheduled contingency reversals",
    ),
]
```

Public export snippet (`src/comp_model/problems/__init__.py`):

```python
from .reversal_bandit import ReversalBanditProblem, create_reversal_bandit_problem
```

Quick registration check:

```python
from comp_model.plugins import build_default_registry

registry = build_default_registry()
problem = registry.create_problem("reversal_bandit")
print(type(problem).__name__)  # ReversalBanditProblem
```

## 5. Implement Required Task Methods and Meanings

`ReversalBanditProblem` should implement these methods:

- `reset(rng)`:
  initialize episode-level state.
- `available_actions(trial_index)`:
  return legal actions for that trial.
- `observe(context)`:
  return observation payload for the model.
- `transition(action, context, rng)`:
  apply task dynamics and return outcome (must include reward signal).

These methods must keep canonical event semantics:
`observation -> decision -> outcome -> update`.

Example implementation:

```python
from __future__ import annotations  # Delay evaluation of type hints; helps avoid forward-reference issues.

from dataclasses import dataclass  # Generates constructor and repr automatically for the class fields.

import numpy as np  # Provides RNG typing and random-number utilities.

from comp_model.core.contracts import DecisionContext  # Runtime passes per-trial metadata through this type.
# Optional explicit typing form (not required in this project):
# from comp_model.core.contracts import DecisionProblem
# class ReversalBanditProblem(DecisionProblem[dict[str, int], int, dict[str, float]]): ...


@dataclass(slots=True)  # Use slots for lower memory usage and to prevent accidental new attributes.
class ReversalBanditProblem:
    reward_probability_high: float = 0.8  # Reward probability for the currently better action.
    reward_probability_low: float = 0.2  # Reward probability for the currently worse action.
    reversal_trials: tuple[int, ...] = (40, 80, 120)  # Trial indices where the good and bad actions swap.

    def __post_init__(self) -> None:
        if not (0.0 <= self.reward_probability_low <= 1.0):  # Ensure low probability is valid.
            raise ValueError("reward_probability_low must be in [0, 1]")  # Fail fast on invalid low value.
        if not (0.0 <= self.reward_probability_high <= 1.0):  # Ensure high probability is valid.
            raise ValueError("reward_probability_high must be in [0, 1]")  # Fail fast on invalid high value.
        if self.reward_probability_low >= self.reward_probability_high:  # Ensure low is actually lower than high.
            raise ValueError("reward_probability_low must be < reward_probability_high")  # Enforce parameter meaning.
        if tuple(sorted(set(self.reversal_trials))) != self.reversal_trials:  # Require sorted, unique reversals.
            raise ValueError("reversal_trials must be sorted unique trial indices")  # Keep reversal schedule deterministic.
        if any(trial < 0 for trial in self.reversal_trials):  # Disallow negative trial indices.
            raise ValueError("reversal_trials must be >= 0")  # Enforce valid timeline indexing.

    def reset(self, *, rng: np.random.Generator) -> None:
        self._rng = rng  # Store runtime RNG for consistency/debugging (even if not used directly later).
        self._contingency_state = 0  # Start in state 0: action-0 high, action-1 low.
        self._reversal_set = set(self.reversal_trials)  # Convert schedule to set for fast membership checks.

    def available_actions(self, *, trial_index: int) -> tuple[int, int]:
        _ = trial_index  # This task has the same legal actions every trial; trial index is unused.
        return (0, 1)  # Always expose the two arms.

    def observe(self, *, context: DecisionContext[int]) -> dict[str, int]:
        if context.trial_index in self._reversal_set:  # Check whether this trial is a scheduled reversal point.
            self._contingency_state = 1 - self._contingency_state  # Flip between state 0 and state 1.
        return {  # Return observation payload consumed by the model for this decision step.
            "state": 0,  # Provide one latent state ID for state-indexed models.
            "trial_index": context.trial_index,  # Expose current trial index for diagnostics/reproducibility.
            "contingency_state": self._contingency_state,  # Expose whether current mapping is normal or reversed.
        }

    def transition(
        self,
        action: int,
        *,
        context: DecisionContext[int],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        if action not in context.available_actions:  # Validate the chosen action against runtime action set.
            raise ValueError(f"action {action!r} is not available on trial {context.trial_index}")  # Stop on invalid action.

        if self._contingency_state == 0:  # State 0: action 0 has high reward, action 1 has low reward.
            p_action_0 = self.reward_probability_high  # Assign high reward probability to action 0.
            p_action_1 = self.reward_probability_low  # Assign low reward probability to action 1.
        else:  # State 1: mapping is reversed.
            p_action_0 = self.reward_probability_low  # Assign low reward probability to action 0.
            p_action_1 = self.reward_probability_high  # Assign high reward probability to action 1.

        p_reward = p_action_0 if action == 0 else p_action_1  # Select reward probability for the chosen action.
        reward = 1.0 if rng.random() < p_reward else 0.0  # Sample binary reward from Bernoulli(p_reward).

        return {  # Return outcome payload used by model update step and logs.
            "reward": reward,  # Scalar reward consumed by RL update.
            "reward_probability": p_reward,  # True generating probability for diagnostics.
            "contingency_state": float(self._contingency_state),  # Store current reversal state in numeric form.
        }
```

## 6. Write Task Tests

Add tests in `tests/test_reversal_bandit_problem.py`.

Minimum tests:

- deterministic behavior with fixed seed,
- invalid action handling,
- integration smoke test with `run_episode(...)`.

Example command:

```bash
python3 -m pytest -q tests/test_reversal_bandit_problem.py
```

Example test file:

```python
from __future__ import annotations

import numpy as np
import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.models import UniformRandomPolicyModel
from comp_model.problems.reversal_bandit import ReversalBanditProblem
from comp_model.runtime import SimulationConfig, run_episode


def _context(trial_index: int) -> DecisionContext[int]:
    return DecisionContext(trial_index=trial_index, available_actions=(0, 1))


def test_reversal_bandit_is_deterministic_with_fixed_seed() -> None:
    problem_a = ReversalBanditProblem(
        reward_probability_high=0.8,
        reward_probability_low=0.2,
        reversal_trials=(2, 4),
    )
    problem_b = ReversalBanditProblem(
        reward_probability_high=0.8,
        reward_probability_low=0.2,
        reversal_trials=(2, 4),
    )

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    problem_a.reset(rng=rng_a)
    problem_b.reset(rng=rng_b)

    records_a: list[tuple[int, float, float]] = []
    records_b: list[tuple[int, float, float]] = []

    for trial_index in range(6):
        context = _context(trial_index)
        observation_a = problem_a.observe(context=context)
        observation_b = problem_b.observe(context=context)
        assert observation_a == observation_b

        action = 0 if trial_index % 2 == 0 else 1
        outcome_a = problem_a.transition(action, context=context, rng=rng_a)
        outcome_b = problem_b.transition(action, context=context, rng=rng_b)

        records_a.append(
            (
                int(observation_a["contingency_state"]),
                float(outcome_a["reward"]),
                float(outcome_a["reward_probability"]),
            )
        )
        records_b.append(
            (
                int(observation_b["contingency_state"]),
                float(outcome_b["reward"]),
                float(outcome_b["reward_probability"]),
            )
        )

    assert records_a == records_b


def test_reversal_bandit_rejects_invalid_action() -> None:
    problem = ReversalBanditProblem(
        reward_probability_high=0.8,
        reward_probability_low=0.2,
        reversal_trials=(2,),
    )
    rng = np.random.default_rng(0)
    problem.reset(rng=rng)
    context = _context(0)
    problem.observe(context=context)

    with pytest.raises(ValueError, match="not available"):
        problem.transition(2, context=context, rng=rng)


def test_reversal_bandit_runs_with_runtime_engine() -> None:
    problem = ReversalBanditProblem(
        reward_probability_high=0.8,
        reward_probability_low=0.2,
        reversal_trials=(3,),
    )
    model = UniformRandomPolicyModel()

    trace = run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=5, seed=7))

    assert len(trace.events) == 5 * 4
    first_trial = trace.by_trial(0)
    assert first_trial[0].payload["observation"]["state"] == 0
    assert "reward" in first_trial[2].payload["outcome"]
```

## 7. Git Add + Commit (Task Part)

```bash
git add src/comp_model/problems/reversal_bandit.py
git add src/comp_model/problems/__init__.py
git add tests/test_reversal_bandit_problem.py
git commit
```

`git commit` invokes a commit message prompt.

Commit message tips:

- Use imperative mood: `Add`, `Fix`, `Refactor`, `Update`.
- Mention scope and outcome: `problem`, `model`, `tests`, `docs`.
- Keep first line concise (about 50-72 chars) and specific.

Good examples:

- `Add reversal_bandit problem and task tests`

Commit body tips (the "why"):

- First line: problem/context.
- Second line: why this change is needed.
- Third line: what behavior changed.
- Fourth line: how you validated it (tests/docs).

Body template:

```text
Add reversal_bandit problem
Context: reversal behavior was missing from the current task set.
Why: we need a benchmark task for testing sticky RL under contingency switches.
Change: added reversal_bandit problem, plugin registration, and task tests.
Validation: ran tests/test_reversal_bandit_problem.py and runtime integration checks.
```

## 8. Decide the Model to Implement (Concept + Math)

Example model: `sticky_state_q_softmax`.

Concept:

- State-indexed Q-learning model.
- Softmax policy for action selection.
- Stickiness/perseveration term that biases repeating the previous action.

Math:

- Utility:
  `U_t(a) = beta * Q_t(s,a) + kappa * I[a = a_{t-1}]`.
- Policy:
  `P(a|s) = exp(U_t(a)) / sum_{a'} exp(U_t(a'))`.
- Value update:
  `Q(s,a_t) <- Q(s,a_t) + alpha * (r_t - Q(s,a_t))`.

## 9. Create Model Files (Repeat)

Create or edit:

- `src/comp_model/models/sticky_state_q_softmax.py`
- `src/comp_model/models/__init__.py`
- `tests/test_sticky_state_q_softmax_model.py`

In `sticky_state_q_softmax.py`, add:

- `StickyStateQSoftmaxModel` class
- `create_sticky_state_q_softmax_model(...)` factory
- `PLUGIN_MANIFESTS` with `component_id="sticky_state_q_softmax"`
- `requirements=ComponentRequirements(...)` if your model expects specific observation/outcome fields

## 10. Implement Required Model Methods and Meanings (Repeat)

`StickyStateQSoftmaxModel` should implement:

- `start_episode()`:
  reset latent state.
- `action_distribution(observation, context)`:
  return action probabilities.
- `update(observation, action, outcome, context)`:
  update latent parameters from outcome.

Also include docstring sections in the model class:

- `Model Contract`
- `Decision Rule`
- `Update Rule`

Example implementation:

```python
from __future__ import annotations  # This defers type-hint evaluation until runtime.

from collections.abc import Mapping  # This lets us check dict-like observation and outcome payloads.
from dataclasses import dataclass, field  # This creates dataclass fields and internal state containers.
from typing import Any  # This keeps action and payload types generic.

import numpy as np  # This provides vector math for softmax computation.

from comp_model.core.contracts import DecisionContext  # This provides trial metadata and available actions.

@dataclass(slots=True)  # This reduces per-instance memory and prevents accidental dynamic attributes.
class StickyStateQSoftmaxModel:
    """State-indexed Q-learning model with sticky softmax action policy.

    Model Contract
    --------------
    Decision Rule
        For each available action ``a`` in state ``s``, compute
        ``U_t(a) = beta * Q_t(s, a) + kappa * I[a = a_(t-1)]`` and choose with
        ``P_t(a|s) = softmax(U_t(a))`` over current available actions.
    Update Rule
        After observing reward ``r_t`` for chosen action ``a_t``, update
        ``Q_{t+1}(s, a_t) = Q_t(s, a_t) + alpha * (r_t - Q_t(s, a_t))`` and
        store ``a_t`` as the previous action for stickiness on the next trial.

    Parameters
    ----------
    alpha : float, optional
        Learning rate in ``[0, 1]``.
    beta : float, optional
        Inverse-temperature for softmax action selection. ``beta=0`` produces
        uniform probabilities over available actions.
    kappa : float, optional
        Stickiness bonus added to the utility of the previously selected action
        in the same state.
    initial_value : float, optional
        Initial Q-value assigned when a state-action pair is first encountered.

    Raises
    ------
    ValueError
        If ``alpha`` is outside ``[0, 1]`` or ``beta`` is negative.
    """

    alpha: float = 0.2  # This is the learning rate used by the value update.
    beta: float = 4.0  # This is the inverse temperature that controls choice sharpness.
    kappa: float = 1.0  # This is the stickiness bonus for repeating the previous action.
    initial_value: float = 0.0  # This is the default Q-value for unseen state-action pairs.
    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)  # This stores per-state action values.
    _last_action_by_state: dict[int, Any] = field(default_factory=dict, init=False, repr=False)  # This stores previous action memory per state.

    def __post_init__(self) -> None:  # This validates hyperparameters immediately after initialization.
        if not (0.0 <= self.alpha <= 1.0):  # This enforces the valid unit-interval range for alpha.
            raise ValueError("alpha must be in [0, 1]")  # This fails fast when alpha is invalid.
        if self.beta < 0.0:  # This disallows negative inverse-temperature values.
            raise ValueError("beta must be >= 0")  # This fails fast when beta is invalid.

    def start_episode(self) -> None:  # This resets latent state at the start of each episode.
        self._q_values = {}  # This clears all learned Q-values from prior episodes.
        self._last_action_by_state = {}  # This clears previous-action memory for stickiness.

    def action_distribution(self, observation: Any, *, context: DecisionContext[Any]) -> dict[Any, float]:  # This computes action probabilities for the current trial.
        state = int(observation["state"]) if isinstance(observation, Mapping) and "state" in observation else 0  # This extracts state from observation with fallback to zero.
        q_state = self._q_values.setdefault(state, {})  # This gets or creates the value table for the current state.

        for action in context.available_actions:  # This iterates over all legal actions for this decision.
            q_state.setdefault(action, float(self.initial_value))  # This initializes unseen actions to initial_value.

        last_action = self._last_action_by_state.get(state)  # This reads the previous action in this state if available.
        utilities = []  # This allocates a list for per-action utilities.

        for action in context.available_actions:  # This computes utility for each legal action.
            stay_bonus = float(self.kappa) if action == last_action else 0.0  # This applies stickiness only to repeated action.
            utilities.append(float(self.beta) * q_state[action] + stay_bonus)  # This combines value and stickiness terms.

        logits = np.asarray(utilities, dtype=float)  # This converts utilities into a numeric vector.
        shifted = logits - np.max(logits)  # This applies max-shift for numerical stability in softmax.
        probs = np.exp(shifted)  # This exponentiates shifted logits.
        probs = probs / np.sum(probs)  # This normalizes probabilities so they sum to one.

        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}  # This maps each legal action to its probability.

    def update(self, observation: Any, action: Any, outcome: Any, *, context: DecisionContext[Any]) -> None:  # This updates latent values after observing outcome.
        if action not in context.available_actions:  # This validates that the chosen action is legal.
            raise ValueError(f"action {action!r} not in available_actions")  # This fails fast for invalid action input.

        state = int(observation["state"]) if isinstance(observation, Mapping) and "state" in observation else 0  # This extracts state using the same rule as policy.
        q_state = self._q_values.setdefault(state, {})  # This gets or creates the state value table for update.

        for candidate in context.available_actions:  # This ensures each legal action has an initialized value.
            q_state.setdefault(candidate, float(self.initial_value))  # This fills missing actions with initial_value.

        reward = float(outcome["reward"]) if isinstance(outcome, Mapping) and "reward" in outcome else float(getattr(outcome, "reward"))  # This reads reward from mapping or object payload.
        prediction_error = reward - q_state[action]  # This computes reward prediction error for chosen action.
        q_state[action] += float(self.alpha) * prediction_error  # This applies the RL value update to chosen action.
        self._last_action_by_state[state] = action  # This stores chosen action for next-trial stickiness.
```

## 11. Write Model Tests (Repeat)

Add tests in `tests/test_sticky_state_q_softmax_model.py`.

Minimum tests:

- distribution sums to 1 over available actions,
- update changes Q-values as expected,
- parameter validation/edge cases,
- deterministic seeded behavior when applicable.

Example command:

```bash
python3 -m pytest -q tests/test_sticky_state_q_softmax_model.py
```

Example test file:

```python
from __future__ import annotations

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.core.events import EventPhase
from comp_model.models.sticky_state_q_softmax import StickyStateQSoftmaxModel
from comp_model.problems import StationaryBanditProblem
from comp_model.runtime import SimulationConfig, run_episode


def _context(trial_index: int = 0) -> DecisionContext[int]:
    return DecisionContext(trial_index=trial_index, available_actions=(0, 1))


def test_distribution_sums_to_one_over_available_actions() -> None:
    model = StickyStateQSoftmaxModel(alpha=0.2, beta=2.0, kappa=0.5, initial_value=0.0)
    model.start_episode()

    distribution = model.action_distribution(
        {"state": 0},
        context=DecisionContext(trial_index=0, available_actions=(0, 1, 2)),
    )

    assert set(distribution) == {0, 1, 2}
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(prob >= 0.0 for prob in distribution.values())


def test_update_changes_q_values_as_expected() -> None:
    model = StickyStateQSoftmaxModel(alpha=0.5, beta=3.0, kappa=0.0, initial_value=0.0)
    model.start_episode()
    context = _context(0)

    before = model.action_distribution({"state": 0}, context=context)[1]
    model.update({"state": 0}, action=1, outcome={"reward": 1.0}, context=context)
    after = model.action_distribution({"state": 0}, context=context)[1]

    assert model._q_values[0][1] == pytest.approx(0.5)
    assert after > before


def test_parameter_validation_and_edge_cases() -> None:
    with pytest.raises(ValueError, match="alpha"):
        StickyStateQSoftmaxModel(alpha=-0.1)

    with pytest.raises(ValueError, match="alpha"):
        StickyStateQSoftmaxModel(alpha=1.1)

    with pytest.raises(ValueError, match="beta"):
        StickyStateQSoftmaxModel(beta=-1.0)


def test_deterministic_seeded_behavior() -> None:
    problem = StationaryBanditProblem([0.2, 0.8])
    model_a = StickyStateQSoftmaxModel(alpha=0.2, beta=3.0, kappa=0.8, initial_value=0.0)
    model_b = StickyStateQSoftmaxModel(alpha=0.2, beta=3.0, kappa=0.8, initial_value=0.0)

    trace_a = run_episode(problem=problem, model=model_a, config=SimulationConfig(n_trials=12, seed=77))
    trace_b = run_episode(problem=problem, model=model_b, config=SimulationConfig(n_trials=12, seed=77))

    actions_a = [event.payload["action"] for event in trace_a.events if event.phase is EventPhase.DECISION]
    actions_b = [event.payload["action"] for event in trace_b.events if event.phase is EventPhase.DECISION]
    rewards_a = [event.payload["outcome"]["reward"] for event in trace_a.events if event.phase is EventPhase.OUTCOME]
    rewards_b = [event.payload["outcome"]["reward"] for event in trace_b.events if event.phase is EventPhase.OUTCOME]

    assert actions_a == actions_b
    assert rewards_a == rewards_b
```

## 12. Git Add (Model Part)

```bash
git add src/comp_model/models/sticky_state_q_softmax.py
git add src/comp_model/models/__init__.py
git add tests/test_sticky_state_q_softmax_model.py
```

Let's commit and think of your own commit message this time.

## 13. Run Full Checks

```bash
python3 -m pytest -q
mkdocs build --strict
```

## 15. Git Push

```bash
git push origin feat/add-reversal-task-and-sticky-rl-model
```

Then open a PR, pass CI, and merge.

## Related Guides

- [Create a New Problem](create-a-new-problem.md)
- [Fit a Bayesian Hierarchical Model](how-to-fit-bayesian-hierarchical-model.md)
- [Contribution Rules](../reference/contribution-rules.md)
