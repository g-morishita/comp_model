# compmod

A lightweight Python 3.13+ library skeleton for **parameter recovery** and **model recovery** in computational modeling (e.g., human reinforcement learning), with first-class support for:

- **Multiple blocks per subject** (one parameter set can generate many blocks; fit jointly across blocks)
- **Hierarchical modeling** (population → individual parameters), compatible with **Stan**
- **Model misspecification** (simulate with one model, fit with another)
- **Asocial + social tasks/models** under unified interfaces

This repository is intentionally “skeleton-first”: it provides clean interfaces, data structures, experiment runners, and Stan-ready packing utilities. You plug in your own tasks/bandits, models, and Stan code.

---

## Contents

- [Quick start](#quick-start)
- [Concepts](#concepts)
  - [Study → Subject → Block → Trial](#study--subject--block--trial)
  - [Fit-only Environment vs RecoveryRunner](#fit-only-environment-vs-recoveryrunner)
  - [Hierarchical sampling for recovery](#hierarchical-sampling-for-recovery)
  - [Model misspecification](#model-misspecification)
- [Stan integration](#stan-integration)
  - [Ragged block packing](#ragged-block-packing)
  - [Estimator contract](#estimator-contract)
- [Project structure](#project-structure)
- [Extending the library](#extending-the-library)
- [License](#license)

---

## Quick start

### Install (editable)

```bash
pip install -e .
