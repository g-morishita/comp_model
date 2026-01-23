# comp_model

A lightweight Python toolkit for **computational modeling**, **simulation**, and **parameter recovery** with a strong emphasis on
avoiding “generator–estimator mismatch” via a **portable event-log contract**.

This repository contains two packages:

- **`comp_model_core`**: core abstractions and data structures (models, bandits/tasks, generators, estimators, plans, param schemas).
- **`comp_model_impl`**: concrete implementations (tasks/bandits, models, generators, MLE and Stan estimators, utilities).

---

## Why event logs?

A common failure mode in parameter recovery is when the **data generator** and the **estimator likelihood** implement slightly different
timing/ordering (e.g., social observation happening pre-choice vs post-outcome). This repo addresses that by making the “flow” explicit:

- The **generator** emits an **event log** (a sequence of typed events).
- **Estimators** (MLE replay, Stan Bayesian) consume that same event log.
- This makes timing a *data contract*, not duplicated control flow.

---

## Installation

### Option A: editable install (recommended for development)

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -e ./comp_model_core
python -m pip install -e ./comp_model_impl
