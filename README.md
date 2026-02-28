# comp_model

A clean-slate computational decision modeling library.

This repository starts from generic decision-problem abstractions where:

1. an agent observes a problem state,
2. selects an action,
3. receives an outcome,
4. updates internal memory.

`Bandit` appears only as one concrete problem implementation under `comp_model.problems`.

## Model Naming

Canonical model names are descriptive and mechanism-first (for example,
`AsocialQValueSoftmaxModel` and
`AsocialStateQValueSoftmaxPerseverationModel`).

Deprecated aliases from v1 naming (for example, `QLearningAgent`, `QRL`,
`RandomAgent`, and legacy component IDs like `q_learning`) remain available
only for migration and emit `DeprecationWarning`.

These aliases are scheduled for removal in **v0.3.0**.
