# comp_model

A lightweight Python toolkit for **computational modeling**, **simulation**, and **parameter recovery**, designed to reduce “generator–estimator mismatch” via a **portable event-log contract**.

This repository contains two packages:

- **`comp_model_core`**: core abstractions and data structures (models, bandits/tasks, generators, estimators, plans, parameter schemas).
- **`comp_model_impl`**: concrete implementations (tasks/bandits, models, generators, MLE and Stan estimators, utilities).

---

## Why event logs?

A common failure mode in parameter recovery is when the **data generator** and the **estimator likelihood** implement slightly different timing/ordering (e.g., social observation happening **pre-choice** vs **post-outcome**). This repo addresses that by making the “flow” explicit:

- The **generator** emits an **event log** (a sequence of typed events).
- Estimators (MLE replay, Stan Bayesian) consume that same event log.
- This makes timing a **data contract**, not duplicated control flow.

---

## Installation

Editable install (recommended for development):

    python -m pip install -U pip
    python -m pip install -e ./comp_model_core
    python -m pip install -e ./comp_model_impl

Install a single package:

    python -m pip install -e ./comp_model_core
    # or
    python -m pip install -e ./comp_model_impl

Note: `comp_model_impl` depends on `comp_model_core`.

---

## Minimal workflow

### 1) Define a plan (blocks / trials)

    from comp_model_core.plans.block import BlockPlan

    plans = [
        BlockPlan(block_id="b1", n_trials=100),
        BlockPlan(block_id="b2", n_trials=100),
    ]

### 2) Simulate with an event-log generator

    import numpy as np
    from comp_model_impl.generators.event_log import EventLogSocialPreChoiceGenerator
    from comp_model_impl.bandits.social_bernoulli import SocialBernoulliBandit
    from comp_model_impl.models.vs.vs import VS

    rng = np.random.default_rng(0)

    generator = EventLogSocialPreChoiceGenerator()
    model = VS()

    def task_builder(plan):
        return SocialBernoulliBandit(...)  # configure your task spec here

    true_params = {"alpha_p": 0.2, "alpha_i": 0.3, "beta": 4.0, "kappa": 0.5}

    subject = generator.simulate_subject(
        subject_id="S001",
        task_builder=task_builder,
        model=model,
        params=true_params,
        block_plans=plans,
        rng=rng,
    )

The resulting `SubjectData` contains normal `Trial` records **and** a block-level event log stored in `Block.metadata["event_log"]`.

### 3) Fit with MLE (event-log replay)

    from comp_model_core.data.types import StudyData
    from comp_model_impl.estimators.mle_event_log import BoxMLESubjectwiseEstimator
    from comp_model_impl.models.vs.vs import VS

    study = StudyData(subjects=[subject])

    est = BoxMLESubjectwiseEstimator(model=VS(), n_starts=20)
    fit = est.fit(study=study, rng=rng)

    print(fit.subject_hats["S001"])

### 4) Fit with Bayesian inference (Stan / CmdStanPy)

    from comp_model_impl.estimators.stan.nuts import StanNUTSSubjectwiseEstimator
    from comp_model_impl.models.vs.vs import VS

    stan_est = StanNUTSSubjectwiseEstimator(
        model=VS(),
        priors={
            "alpha_p": {"family": "beta", "a": 2, "b": 2},
            "alpha_i": {"family": "beta", "a": 2, "b": 2},
            "beta": {"family": "lognormal", "mu": 0, "sigma": 1},
            "kappa": {"family": "normal", "mu": 0, "sigma": 1},
        },
    )
    fit = stan_est.fit(study=study, rng=rng)
    print(fit.subject_hats["S001"])

Stan estimators use the **same event log** (exported to Stan arrays once) to avoid flow mismatch.

---

## Data format for real experiments

For real experimental data, the recommended format is:

- `StudyData` → `SubjectData` → `Block` → `Trial`
- and a **block-level** event log in `Block.metadata["event_log"]`

The event log should include:
- `BLOCK_START` to trigger block resets
- `CHOICE` events with `payload["choice"]`
- `OUTCOME` events with `payload["action"]` and `payload["observed_outcome"]`
- optional `SOCIAL_OBSERVED` events with demonstrated actions/outcomes

This makes missing/irregular trials representable (timeouts, skipped feedback, etc.).

---

## Priors (Stan)

Stan priors are configurable from estimator config and support:

- beta
- normal
- lognormal
- gamma
- exponential
- half-normal
- student-t
- cauchy

Priors are passed as **Stan data**, not hard-coded in `.stan` files.

---

## Repository layout

    comp_model_core/     # core interfaces + data types
    comp_model_impl/     # implementations (models/generators/estimators/stan)

---

## Contributing / development notes

- Prefer event-log generators for simulation.
- Estimators should consume the event log (not reconstruct timing from trials).
- Add unit tests that compare log-likelihood between Python replay and Stan for fixed parameters.

---
