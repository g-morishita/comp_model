# comp_model

A lightweight Python 3.13+ library skeleton for **parameter recovery** and **model recovery** in computational modeling (e.g., human reinforcement learning), with first-class support for:

- **Multiple blocks per subject** (one parameter set can generate many blocks; fit jointly across blocks)
- **Hierarchical modeling** (population → individual parameters), compatible with **Stan**
- **Model misspecification** (simulate with one model, fit with another)
- **Asocial + social tasks/models** under unified interfaces

This repository is intentionally “skeleton-first”: it provides clean interfaces, data structures, experiment runners, and Stan-ready packing utilities. You plug in your own tasks/bandits, models, and Stan code.

---

## Implemented computational models

### Value-Shaping (VS) — social reinforcement learning
Imitation model where observing another agent’s action acts as a **pseudo-reward** that shapes the observer’s action values.

**Features**
- **K-armed** tasks (`n_actions >= 2`)
- **Chosen-only updates** (private learning updates chosen action only; social shaping updates demonstrated action only)
- **Softmax** choice with inverse temperature `beta`
- Optional **perseveration** (`kappa`) as a repeat-choice bonus
- Works with **multi-block** datasets (latent state resets per block; parameters shared across blocks)

**Parameters**
- `alpha_p` — private learning rate
- `alpha_i` — social/value-shaping learning rate
- `beta` — inverse temperature
- `kappa` — perseveration strength

**Reference**
- Najar et al. (2020). *The actions of others act as a pseudo-reward to drive imitation in the context of social reinforcement learning.* PLOS Biology.

See [`src/comp_model/models/vs/`](src/comp_model/models/vs/) for implementation details and notes.
