# Explanation: Architecture Decisions

This page explains key architectural decisions and their tradeoffs.

## Decision: Keep `Bandit` Out of Core Naming

Core protocols use generic terms (`DecisionProblem`, `AgentModel`) instead of
task-specific names.

Why:

- avoids misleading abstraction boundaries,
- enables reuse across non-bandit tasks,
- keeps interface semantics stable as new problems are added.

Tradeoff:

- requires slightly more domain translation in quick examples.

## Decision: Trial Programs for Multi-Phase Timing

Complex tasks (for example social tasks with additional decision phases) are
expressed as explicit trial programs rather than ad-hoc flags.

Why:

- timing semantics become inspectable and testable,
- replay and simulation remain aligned,
- multi-actor ordering is explicit.

Tradeoff:

- more upfront structure for simple tasks.

## Decision: Unified Fitting Surface

MLE, MAP, MCMC, and hierarchical methods share canonical replay likelihood
paths and fit-result extraction conventions.

Why:

- downstream workflows (comparison/recovery/serialization) can operate
  estimator-agnostically,
- less duplicated orchestration logic.

Tradeoff:

- some abstractions are necessarily broader than a single estimator needs.

## Decision: Registry-Driven Components

Models/problems/generators/demonstrators are resolved by `component_id` through
a registry.

Why:

- declarative config workflows remain concise,
- plugin discovery can scale to larger model suites.

Tradeoff:

- registry consistency and smoke tests are mandatory to prevent runtime
  resolution failures.
