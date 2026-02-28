# comp_model

A clean-slate computational decision modeling library.

This repository starts from generic decision-problem abstractions where:

1. an agent observes a problem state,
2. selects an action,
3. receives an outcome,
4. updates internal memory.

`Bandit` appears only as one concrete problem implementation under `comp_model.problems`.
