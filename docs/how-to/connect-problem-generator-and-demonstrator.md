# How-to: Connect a Problem, Generator, and Demonstrator

Use this guide for advanced social-task pipelines where a generator constructs
study data and demonstrators provide social behavior.

## 1. Choose Components

Example component IDs:

- problem: `stationary_bandit`
- generator: `event_trace_social_pre_choice_generator`
- demonstrator model: `fixed_sequence_demonstrator`
- subject model: any registered learner model

## 2. Build Components from Registry

```python
from comp_model.plugins import build_default_registry

registry = build_default_registry()

generator = registry.create_generator(
    "event_trace_social_pre_choice_generator"
)
subject_model = registry.create_model(
    "social_observed_outcome_q",
    alpha=0.3,
    beta=3.0,
    initial_value=0.0,
)
demonstrator_model = registry.create_demonstrator(
    "fixed_sequence_demonstrator",
    sequence=[1, 1, 0, 1, 0, 1],
)
```

## 3. Define Block Specs and Simulate

```python
import numpy as np

from comp_model.generators import SocialBlockSpec

study = generator.simulate_study(
    subject_models={"s1": subject_model, "s2": subject_model},
    demonstrator_models=demonstrator_model,
    blocks=(
        SocialBlockSpec(
            n_trials=40,
            block_id="b1",
            program_kwargs={"reward_probabilities": [0.2, 0.8]},
        ),
    ),
    rng=np.random.default_rng(123),
)
```

## 4. Fit Generated Data

Feed `study` directly into inference or write/read CSV via `comp_model.io`.

## 5. Advanced Tips

- Use different demonstrator models per subject by passing a mapping.
- Keep actor timing explicit (`pre_choice` vs `post_outcome`) by selecting the
  matching generator.
- Add generator-level tests for actor-order validation and deterministic seeds.
