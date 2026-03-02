# Reference: API Overview

This page summarizes the public package-level modules and their primary API
surfaces.

## `comp_model.core`

- Contracts: `AgentModel`, `DecisionProblem`, `DecisionContext`
- Canonical events: `SimulationEvent`, `EpisodeTrace`, `EventPhase`
- Data containers: `TrialDecision`, `BlockData`, `SubjectData`, `StudyData`

## `comp_model.runtime`

- Simulation engines:
  - `run_episode` (single-agent episode wrapper)
  - `run_trial_program` (multi-phase/multi-actor trial program execution)
- Replay engine:
  - canonical replay semantics for likelihood evaluation

## `comp_model.models`

- Canonical model implementations across asocial and social families.
- Shared-delta wrappers for within-subject parameter tying:
  - `ConditionedSharedDeltaModel`
  - `ConditionedSharedDeltaSocialModel`

## `comp_model.problems`

- Concrete problem/task implementations, including bandit variants as specific
  implementations.

## `comp_model.inference`

- MLE fitting (`fit_model`, config-driven and dataset/study variants)
- Stan-backed MAP and posterior sampling
- Subject/study block handling via `block_fit_strategy` (`independent` or `joint`)
- Hierarchical within-subject Stan workflows
- Candidate model comparison APIs
- CSV-driven fit/comparison CLIs

## `comp_model.recovery`

- Parameter recovery runners and serializers
- Model recovery runners and serializers
- Model-recovery support for subject/study `block_fit_strategy`
- Config-driven recovery CLI

## `comp_model.analysis`

- Information criteria: `aic`, `bic`, `waic`, `psis_loo`
- Profile likelihood: 1D/2D utilities

## `comp_model.io`

- Tabular CSV I/O for trial and study datasets.
- Bridges between external CSV datasets and canonical in-memory data structures.

## `comp_model.plugins`

- Registry and manifest-based component discovery for models/problems/
  demonstrators/generators.

## Schema Reference

For complete config contracts and allowed keys, see
[Configuration Schemas](config_schemas.md).
