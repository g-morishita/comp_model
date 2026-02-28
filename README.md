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

Legacy alias names and IDs were removed; use canonical model classes and
canonical plugin component IDs only.

## v1 Capability Parity Matrix

The repository now includes an explicit parity matrix in
`comp_model.models.V1_MODEL_PARITY` mapping internal v1 model names to
canonical class names and plugin IDs.

Current status:
- Implemented: all base asocial/social model families from v1.
- Implemented: within-subject shared+delta wrappers (`ConditionedSharedDeltaModel`, `ConditionedSharedDeltaSocialModel`).

For wrapper models, canonical class mappings are provided in the parity matrix.
They are intentionally not registered as zero-argument plugin components because
wrapper construction requires explicit base-model factory and condition metadata.

## Easy Model Fitting API

Use `comp_model.inference.fit_model` to fit a model directly from:
- canonical `EpisodeTrace`,
- `BlockData`, or
- `TrialDecision` rows.

This fitting module is now the shared base used by recovery workflows.

For multi-subject datasets, use:
- `comp_model.inference.fit_block_data`
- `comp_model.inference.fit_subject_data`
- `comp_model.inference.fit_study_data`

These APIs run independent fits per block and aggregate summaries at subject and
study levels.

You can also fit from declarative config mappings via:
- `comp_model.inference.fit_dataset_from_config`
- `comp_model.inference.fit_study_from_config`
