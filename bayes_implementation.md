# Bayesian Estimator Implementation Plan

Branch: `feat/subject-bayesian-hierarchies`

## Goal

Implement a Bayesian estimator surface that distinguishes the actual hierarchy
being fit, instead of overloading:

- `sample`
- `within_subject_hierarchical`
- `pooled`

The target surface must cover four distinct model structures:

1. `SubjectData`, one parameter set shared across blocks.
   Structure: subject
2. `SubjectData`, block-specific parameters pooled within subject.
   Structure: subject -> block
3. `StudyData`, one parameter set per subject pooled at population level.
   Structure: population -> subject
4. `StudyData`, subject-level parameters and block-level parameters pooled at
   population level.
   Structure: population -> subject -> block

No backward compatibility: remove the old function names, estimator type names,
tests, docs wording, and Stan filename conventions in the same change.

## Current State

- `sample_subject_hierarchical_posterior_stan(...)` on `SubjectData` is
  actually case 2 (`subject -> block`).
- `sample_subject_pooled_posterior_stan(...)` on `SubjectData` is
  actually case 1 (`subject` shared across blocks).
- `sample_study_hierarchical_posterior_stan(...)` on `StudyData` is not a
  population hierarchy. It loops over subjects and runs the subject-level case
  independently.
- `sample_study_pooled_posterior_stan(...)` on `StudyData` is likewise only a
  per-subject batch wrapper.
- `mcmc_config.py` is misnamed because it dispatches both NUTS and MAP.
- Current Stan files only distinguish "non-pooled" vs "_pooled", which is too
  vague once study-level hierarchies are added.

## Target Estimator Matrix

### Direct Stan APIs

| Input | Structure | Posterior draws | MAP | Estimator type |
| --- | --- | --- | --- | --- |
| `SubjectData` | subject | `draw_subject_shared_posterior_stan` | `estimate_subject_shared_map_stan` | `subject_shared_stan_nuts` / `subject_shared_stan_map` |
| `SubjectData` | subject -> block | `draw_subject_block_hierarchy_posterior_stan` | `estimate_subject_block_hierarchy_map_stan` | `subject_block_hierarchy_stan_nuts` / `subject_block_hierarchy_stan_map` |
| `StudyData` | population -> subject | `draw_study_subject_hierarchy_posterior_stan` | `estimate_study_subject_hierarchy_map_stan` | `study_subject_hierarchy_stan_nuts` / `study_subject_hierarchy_stan_map` |
| `StudyData` | population -> subject -> block | `draw_study_subject_block_hierarchy_posterior_stan` | `estimate_study_subject_block_hierarchy_map_stan` | `study_subject_block_hierarchy_stan_nuts` / `study_subject_block_hierarchy_stan_map` |

### Config Entry Points

Rename the current config helpers so their names do not pretend there is only
one hierarchical estimator family:

- `sample_subject_hierarchical_posterior_from_config`
  -> `infer_subject_stan_from_config`
- `sample_study_hierarchical_posterior_from_config`
  -> `infer_study_stan_from_config`

`fit_subject_auto_from_config` and `fit_study_auto_from_config` remain as the
top-level auto-dispatch entry points, but must accept the new estimator type
names only.

### Block/Dataset Bayesian Dispatch

`fit_block_auto_from_config` and `fit_dataset_auto_from_config` should support:

- `subject_shared_stan_*`
- `subject_block_hierarchy_stan_*`

For one block / one dataset, the block-hierarchy case degenerates to a
single-block hierarchy. Study-level estimator types must be rejected for
`BlockData` / `EpisodeTrace`.

## Result Type Plan

Current `HierarchicalSubjectPosteriorResult` and
`HierarchicalStudyPosteriorResult` are too narrow for the new matrix.

Introduce explicit result families:

- `SubjectSharedPosteriorResult`
- `SubjectBlockHierarchyPosteriorResult`
- `StudySubjectHierarchyPosteriorResult`
- `StudySubjectBlockHierarchyPosteriorResult`

Parallel MAP result families:

- `SubjectSharedMapResult`
- `SubjectBlockHierarchyMapResult`
- `StudySubjectHierarchyMapResult`
- `StudySubjectBlockHierarchyMapResult`

Implementation note:

- Case 2 can reuse much of the current
  `HierarchicalSubjectPosteriorResult` structure.
- Cases 3 and 4 need new population-level latent parameters in the result
  object, rather than only a tuple of per-subject outputs.

Serialization helpers must be split accordingly so population-, subject-, and
block-level summaries are explicit in both draw records and MAP summaries.

## Stan File Plan

Replace the ambiguous current naming:

- `*.stan`
- `*_pooled.stan`

with explicit structural suffixes for every supported model family:

- `*_subject_shared.stan`
- `*_subject_block_hierarchy.stan`
- `*_study_subject_hierarchy.stan`
- `*_study_subject_block_hierarchy.stan`

This applies to:

- all currently supported asocial Stan models
- all currently supported social Stan models

Implementation rule:

- do not keep the old filename conventions
- update every loader, cache tag, test, and package-data expectation to the new
  names in the same change

## Module/File Restructure

### Keep

- `hierarchical_stan.py` as the Stan backend entry module
- `hierarchical_stan_social.py` for social-model-specific Stan adapters

### Rename / repurpose

- `mcmc_config.py` -> `stan_config.py`
  Reason: this module dispatches both NUTS and MAP.

### Add

- result modules that match the new estimator matrix if the current
  `hierarchical_posterior.py` / `hierarchical_map.py` become too overloaded

## Implementation Phases

### Phase 1: Naming and API split

1. Add the new direct Stan function names.
2. Add the new estimator type names to config parsing and dispatch.
3. Remove the old `sample_*_hierarchical_posterior_*`,
   `sample_*_pooled_posterior_*`, `optimize_*_hierarchical_posterior_*`, and
   `optimize_*_pooled_posterior_*` names.
4. Rename `mcmc_config.py` and its imports.

### Phase 2: Subject-level estimator cleanup

1. Rename current subject-shared Stan path (`*_pooled`) to explicit
   `subject_shared`.
2. Rename current subject-block Stan path to explicit
   `subject_block_hierarchy`.
3. Replace all docs/tests/config examples accordingly.

### Phase 3: Study-level population -> subject

1. Add new Stan programs for study-level subject pooling.
2. Add population-level result types and decoding logic.
3. Add direct APIs, config dispatch, serializers, and tests.

### Phase 4: Study-level population -> subject -> block

1. Add new Stan programs for the full three-level structure.
2. Add result types with population, subject, and block latent structure.
3. Add direct APIs, config dispatch, serializers, and tests.

### Phase 5: Documentation and tutorial rewrite

1. Rewrite `docs/reference/config_schemas.md`.
2. Rewrite Bayesian how-to and tutorial pages to use the new names.
3. Replace wording like "within-subject hierarchical" with the explicit
   hierarchy structure being fit.

## Testing Plan

Add or update tests for:

- config parsing for all 8 estimator types
- auto-dispatch acceptance / rejection by input data type
- result decoding for all 4 structures
- Stan loader and cache-tag selection for all 4 structures
- serialization outputs for population / subject / block summaries
- docs examples and tutorial code snippets

At minimum, update:

- `tests/test_hierarchical_stan.py`
- `tests/test_mcmc_config.py`
- `tests/test_fit_dispatch_config.py`
- `tests/test_fit_result.py`
- `tests/test_hierarchical_serialization.py`
- tutorial/how-to references that still mention `within_subject_*`

## Documentation Rules For This Work

- Never use `sample` for posterior-draw APIs. Use `draw`.
- Never use `pooled` when the actual structure is "shared across blocks" or
  "population -> subject". Name the structure directly.
- Never use `within_subject_hierarchical` as the primary public label.
  Spell out the hierarchy:
  - `subject`
  - `subject -> block`
  - `population -> subject`
  - `population -> subject -> block`

## Notes

- The current study-level Bayesian surface is a batch wrapper, not a true
  population hierarchy. Cases 3 and 4 require real new Stan models, not just
  Python wrapper renames.
- Because the repository policy is no backward compatibility, this should land
  as one coordinated API change across code, tests, docs, and Stan assets.
