# Explanation: Design Philosophy

`comp_model` is intentionally designed as a general decision-modeling framework,
not a task-specific toolkit.

## 1. Problem-Model Interaction Is Generic

The core interaction loop is domain-agnostic:

- environment/problem emits context,
- model selects an action,
- environment emits outcome/observation,
- model updates internal state.

This abstraction supports bandits, social multi-phase tasks, and broader
sequential decision problems without renaming core types.

## 2. One Canonical Event Language

Simulation, replay, inference, and recovery all consume the same canonical
event representation. This avoids semantic drift where generation and fitting
disagree about what happened.

## 3. Declarative, Strict Configs

Config APIs are strict by design:

- unknown keys raise errors,
- required sections are explicit,
- estimator/likelihood/prior wiring is validated upfront.

The goal is fail-fast behavior and reproducibility under automation.

## 4. Composition Over Entanglement

Modules are separated by responsibility:

- runtime executes interactions,
- inference optimizes/posterior-samples,
- analysis computes diagnostics,
- recovery orchestrates simulation-plus-fitting experiments,
- I/O handles persistence boundaries.

This separation keeps each layer testable and easier to evolve.

## 5. Reproducibility as a First-Class Constraint

Deterministic seeds, stable tabular I/O, and explicit artifact writers exist to
support scientific workflows where exact reruns and auditability matter.
