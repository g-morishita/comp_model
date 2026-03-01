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
Recovery workflows can consume both MLE-style and MAP-style fit outputs.
Config-driven recovery runners support MAP estimators when a `prior` section is provided.

For multi-subject datasets, use:
- `comp_model.inference.fit_block_data`
- `comp_model.inference.fit_subject_data`
- `comp_model.inference.fit_study_data`

These APIs run independent fits per block and aggregate summaries at subject and
study levels.

You can also fit from declarative config mappings via:
- `comp_model.inference.fit_dataset_from_config`
- `comp_model.inference.fit_study_from_config`
- `comp_model.inference.fit_dataset_auto_from_config`
- `comp_model.inference.fit_block_auto_from_config`
- `comp_model.inference.fit_subject_auto_from_config`
- `comp_model.inference.fit_study_auto_from_config`

To export fit outputs:
- `comp_model.inference.write_study_fit_records_csv`
- `comp_model.inference.write_study_fit_summary_csv`
- `comp_model.inference.write_map_study_fit_records_csv`
- `comp_model.inference.write_map_study_fit_summary_csv`
- `comp_model.inference.write_hierarchical_study_block_records_csv`
- `comp_model.inference.write_hierarchical_study_summary_csv`
- `comp_model.inference.write_model_comparison_csv`

For candidate-model comparison on one dataset:
- `comp_model.inference.compare_candidate_models`
- `comp_model.inference.compare_registry_candidate_models`
- `comp_model.inference.compare_dataset_candidates_from_config`
- `comp_model.inference.compare_subject_candidate_models`
- `comp_model.inference.compare_study_candidate_models`
- `comp_model.inference.compare_subject_candidates_from_config`
- `comp_model.inference.compare_study_candidates_from_config`

These return per-candidate log-likelihood/AIC/BIC summaries and selected model
labels under a chosen criterion.

## Bayesian Inference (MAP First)

Bayesian scaffolding now includes MAP estimators built on SciPy optimization:
- `comp_model.inference.ScipyMapBayesEstimator`
- `comp_model.inference.TransformedScipyMapBayesEstimator`
- `comp_model.inference.fit_map_model`
- `comp_model.inference.fit_map_model_from_registry`
- `comp_model.inference.fit_map_dataset_from_config`
- `comp_model.inference.fit_map_block_data`
- `comp_model.inference.fit_map_subject_data`
- `comp_model.inference.fit_map_study_data`
- `comp_model.inference.fit_map_block_from_config`
- `comp_model.inference.fit_map_subject_from_config`
- `comp_model.inference.fit_map_study_from_config`

Prior utilities:
- `comp_model.inference.IndependentPriorProgram`
- `comp_model.inference.normal_log_prior`
- `comp_model.inference.uniform_log_prior`
- `comp_model.inference.beta_log_prior`
- `comp_model.inference.log_normal_log_prior`

These are the first step toward full Bayesian and hierarchical workflows using
the same canonical replay likelihood semantics as MLE.

Within-subject hierarchical MAP is now available via:
- `comp_model.inference.fit_subject_hierarchical_map`
- `comp_model.inference.fit_study_hierarchical_map`
- `comp_model.inference.fit_subject_hierarchical_map_from_config`
- `comp_model.inference.fit_study_hierarchical_map_from_config`

This performs partial pooling across blocks for named parameters in transformed
space, using the same replay likelihood path as all other estimators.
