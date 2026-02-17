# Welcome to comp_model

**comp_model** is an ecosystem of Python packages for reproducible computational modeling of behavioral decision-making tasks.
It’s designed to support common research workflows such as **model fitting**, **parameter recovery**, and **model recovery**—with a focus on *clear task specifications* and *auditable likelihood computation*.

## What you can do with comp_model

- **Specify tasks and experiments** using serializable plans (JSON/YAML-friendly).
- **Simulate synthetic studies** (including social tasks with demonstrators) while recording an explicit **event log**.
- **Fit models** by replaying event logs for likelihood evaluation (MLE) and optionally using **Stan NUTS** for Bayesian inference.
- Run **parameter recovery** and **model recovery** experiments with utilities built around the same plan → simulate → fit pipeline.
- Produce **diagnostics and plots** (optional analysis utilities).

## Packages

### `comp_model_core`
Lightweight, dependency-minimal “core” definitions:

- Interfaces (ABCs) for models, tasks/bandits, generators, estimators, block runners, demonstrators
- Data containers for trials/blocks/subjects/studies
- Plan/spec schemas (JSON/YAML oriented)
- Parameter schemas + transforms (bounds, constrained/unconstrained parameterizations)
- Registries and validation helpers

### `comp_model_impl`
Reference implementations and end-to-end pipelines:

- Concrete models (e.g., RL variants and social RL models)
- Bandit environments + block runners
- Demonstrators for social learning tasks
- Event-log generators (simulation)
- Estimators (MLE via event-log replay; optional Stan-based estimators)
- Recovery utilities (parameter/model recovery runners)

### `comp_model_analysis` (optional)
Small analysis helpers (profiling/diagnostics/plotting), intended to remain separate from core simulation/estimation code.

## Design in a nutshell

`Plan (what happens)` → `Runner (executes task)` → `Event log (what happened, in order)` →  
`Likelihood replay (how the model explains it)` → `Estimator (fit parameters)`

The **event log** is a first-class object: it makes the timing of resets, observations, choices, and outcomes explicit, and it ensures simulation and fitting share the same semantics.

## Getting Started

We highly recommend following the following tutorials:

- [Install comp_model packages]()
- [Fit a standard reinforcement learning model to a public dataset]()
- [Perform parameter recovery on a standard reinforcement learning model]()
- [Perform model recovery on a set of reinforcement learning models]()