# v1 vs Current Gap Audit (2026-03-04)

## Scope

- v1 audited: `/Users/morishitag/comp_model_v1`
- current audited: `/Users/morishitag/comp_model`

Focus:

- architecture and core contracts
- model/inference/recovery parity
- analysis/docs/test/CI parity and regression risk

---

## Validation Snapshot (current repo)

- `pytest -q`: pass
- `ruff check src tests`: pass
- `mypy src/comp_model`: pass
- `mkdocs build --strict`: fail (broken link)
  - `docs/tutorials/first-end-to-end-fit.md:423` links to missing `docs/how-to/how-to-fit-bayesian-hierarchical-model.md`

---

## Findings (Severity Ordered)

### CRITICAL

1. **Parameter-recovery parity gap: v1 rich true-parameter sampling modes are not fully ported**

- v1 supports:
  - sampling modes: `independent | hierarchical | fixed`
  - sampling space: `param | z`
  - per-condition overrides (`by_condition`) for within-subject shared+delta
  - refs:
    - `comp_model_v1/comp_model_impl/src/comp_model_impl/recovery/parameter/config.py:65`
    - `.../config.py:72`
    - `.../config.py:85`
    - `.../sampling.py:271`
    - `.../sampling.py:305`
    - `.../sampling.py:389`
- current recovery config only accepts:
  - explicit sets (`true_parameter_sets`) or independent per-parameter distributions (`true_parameter_distributions`)
  - refs:
    - `src/comp_model/recovery/config.py:119`
    - `src/comp_model/recovery/config.py:120`
    - `src/comp_model/recovery/config.py:695`
    - `src/comp_model/recovery/config.py:729`
- impact:
  - cannot replicate v1-style hierarchical/fixed/by-condition recovery designs directly via config.

2. **Parameter-recovery simulation scope is restricted to block-level, while v1 ran study-level recovery loops**

- v1 runs `simulate_study(...)` during parameter recovery:
  - `comp_model_v1/comp_model_impl/src/comp_model_impl/recovery/parameter/run.py:465`
  - v1 loads full study plan subjects:
    - `.../run.py:655`
    - `.../run.py:657`
- current hard-stops unless `simulation.level == "block"`:
  - `src/comp_model/recovery/config.py:146`
  - `src/comp_model/recovery/config.py:148`
- impact:
  - parity gap for subject/study-level parameter recovery workflows.

### HIGH

3. **Within-subject shared+delta wrappers are not integrated into current plugin/config pipelines**

- v1 registry exposed wrapper models directly:
  - `comp_model_v1/comp_model_impl/src/comp_model_impl/register.py:85`
  - `.../register.py:86`
- v1 Stan adapter registry handled wrapped models:
  - `comp_model_v1/comp_model_impl/src/comp_model_impl/estimators/stan/adapters/registry.py:91`
  - `.../registry.py:117`
- current has wrapper classes but no plugin manifests for them:
  - wrappers exist: `src/comp_model/models/shared_delta.py:58`, `:329`
  - file ends with `__all__`, no `PLUGIN_MANIFESTS`: `src/comp_model/models/shared_delta.py:391`
  - plugin discovery only registers modules exposing `PLUGIN_MANIFESTS`:
    - `src/comp_model/plugins/registry.py:163`
    - `src/comp_model/plugins/registry.py:185`
- impact:
  - wrappers are usable manually in Python, but not first-class in config-driven registry workflows (v1 parity not complete).

4. **Current public inference surface still exports removed MAP/Bayes APIs that fail at runtime**

- exported from package:
  - `src/comp_model/inference/__init__.py:3`
  - `src/comp_model/inference/__init__.py:14`
- removed runtime behavior:
  - `src/comp_model/inference/bayes.py:214`
  - `src/comp_model/inference/bayes.py:253`
  - `src/comp_model/inference/bayes.py:294`
  - `src/comp_model/inference/bayes_config.py:106`
  - `src/comp_model/inference/bayes_config.py:172`
- map study helpers still call removed API:
  - `src/comp_model/inference/map_study_fitting.py:112`
  - `src/comp_model/inference/map_study_fitting.py:209`
- tests intentionally assert removed behavior:
  - `tests/test_map_study_fitting.py:61`
- impact:
  - user-facing API has dead entry points; easy to hit runtime traps.

5. **Current repo violates its own “no backward compatibility” rule in multiple places**

- hard rule states no backward compatibility aliases/wrappers:
  - `AGENTS.md:3`
- current still includes alias:
  - `src/comp_model/recovery/config.py:70` (`load_json_config`)
  - exported in `src/comp_model/recovery/__init__.py:7`, `:40`
- additional backward-compatible alias property:
  - `src/comp_model/inference/hierarchical_mcmc.py:95`
- impact:
  - architecture policy inconsistency; increases maintenance cost and ambiguity.

### MEDIUM

6. **Analysis parity gap: v1 exposed parameter-recovery plotting helpers; current analysis package does not**

- v1 exports parameter recovery plotting:
  - `comp_model_v1/comp_model_analysis/src/comp_model_analysis/__init__.py:4`
  - plotting implementation:
    - `comp_model_v1/comp_model_analysis/src/comp_model_analysis/parameter_recovery.py:286`
- current analysis exports only information criteria + profile likelihood:
  - `src/comp_model/analysis/__init__.py:3`
- impact:
  - regression for post-recovery visualization workflows.

7. **Docs/API drift: README code sample uses obsolete `FitSpec` field**

- README uses `estimator_type`:
  - `README.md:93`
- actual `FitSpec` API is `inference` + `solver`:
  - `src/comp_model/inference/fitting.py:69`
  - `src/comp_model/inference/fitting.py:70`
- impact:
  - onboarding friction; copy-paste example breaks.

### LOW

8. **Documentation strict build is currently broken due to one bad internal link**

- broken link location:
  - `docs/tutorials/first-end-to-end-fit.md:423`
- impact:
  - docs CI fails in strict mode despite code/tests/lint/type passing.

---

## v1 Issues Re-Check: Solved vs Still Present

### Solved in current

1. **Bandit-centric core abstraction in v1**

- v1 core is explicitly bandit/block-runner centric:
  - `comp_model_v1/comp_model_core/src/comp_model_core/interfaces/bandit.py:73`
  - `.../interfaces/block_runner.py:129`
  - `.../plans/block.py:124`
- current core is generic decision-problem + agent protocols:
  - `src/comp_model/core/contracts.py:63`
  - `src/comp_model/core/contracts.py:137`
- status: **solved**

2. **Hidden coupling in v1 social timing/state mutation**

- v1 social wrapper advances env in both social observation and subject step:
  - `comp_model_v1/comp_model_impl/src/comp_model_impl/tasks/block_runner_wrappers.py:345`
  - `.../block_runner_wrappers.py:389`
- current uses explicit trial-node phase pipeline:
  - `src/comp_model/runtime/program.py:41`
  - `src/comp_model/runtime/engine.py:113`
  - `src/comp_model/runtime/engine.py:177`
- status: **solved**

3. **Event contract fragility in v1 (int codes + loosely-keyed payloads)**

- v1 int-coded event types:
  - `comp_model_v1/comp_model_core/src/comp_model_core/events/types.py:25`
  - key-based checks:
    - `.../types.py:219`
- current typed phases + strict per-trial phase validation:
  - `src/comp_model/core/events.py:14`
  - `src/comp_model/core/events.py:116`
  - `src/comp_model/core/events.py:164`
- status: **solved**

4. **Duplicate mask/renormalize logic in v1 generation vs replay**

- duplicated in v1:
  - `comp_model_v1/comp_model_impl/src/comp_model_impl/generators/event_log.py:180`
  - `.../likelihood/event_log_replay.py:35`
- centralized current utility:
  - `src/comp_model/runtime/probabilities.py:15`
  - used in runtime:
    - `src/comp_model/runtime/engine.py:136`
    - `src/comp_model/runtime/replay.py:143`
- status: **solved**

5. **v1 registry inconsistency risk (exported model not registered)**

- v1 exports `Vicarious_AP_VS`:
  - `comp_model_v1/comp_model_impl/src/comp_model_impl/models/__init__.py:67`
  - `.../__init__.py:83`
- but v1 registry registration list omits it:
  - `comp_model_v1/comp_model_impl/src/comp_model_impl/register.py:69`
  - `.../register.py:87`
- current uses plugin manifests + discovery + smoke tests:
  - `src/comp_model/plugins/registry.py:163`
  - `tests/test_plugins.py:223`
- status: **solved**

### Still present / partially present

- findings #1, #2, #3, #4, #5, #6, #7, #8 above.

---

## Model Parity Matrix (v1 -> current)

Source of v1 model set (registered):

- `comp_model_v1/comp_model_impl/src/comp_model_impl/register.py:69`
- `.../register.py:86`

Additional v1 exported but not registered:

- `comp_model_v1/comp_model_impl/src/comp_model_impl/models/__init__.py:83`

| v1 model | current mapping | status | notes/risk |
|---|---|---|---|
| `QRL` | `asocial_state_q_value_softmax` | mapped | same family (state Q + softmax) |
| `QRL_Stay` | `asocial_state_q_value_softmax_perseveration` | mapped | perseveration retained |
| `UnidentifiableQRL` | `asocial_state_q_value_softmax_split_alpha` | mapped | non-identifiable split-alpha retained |
| `VS` | `social_self_outcome_value_shaping` | mapped | self outcome + social shaping + perseveration |
| `Vicarious_RL` | `social_observed_outcome_q` | mapped | observed-demo-outcome RL |
| `Vicarious_RL_Stay` | `social_observed_outcome_q_perseveration` | mapped | + perseveration |
| `Vicarious_VS` | `social_observed_outcome_value_shaping` | mapped | observed outcome + value shaping |
| `Vicarious_VS_Stay` | `social_observed_outcome_value_shaping_perseveration` | mapped | + perseveration |
| `Vicarious_AP_DB_STAY` | `social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration` | mapped | reliability-gated demo-bias family |
| `Vicarious_Dir_DB_Stay` | `social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration` | mapped | Dirichlet reliability variant |
| `Vicarious_DB_Stay` | `social_constant_demo_bias_observed_outcome_q_perseveration` | mapped | constant demo-bias variant |
| `VicQ_AP_DualW_Stay` | `social_observed_outcome_policy_shared_mix_perseveration` | mapped | shared mix + perseveration |
| `VicQ_AP_DualW_NoStay` | `social_observed_outcome_policy_shared_mix` | mapped | shared mix, no stay |
| `VicQ_AP_IndepDualW` | `social_observed_outcome_policy_independent_mix_perseveration` | mapped | independent Q/policy temperatures |
| `AP_RL_Stay` | `social_policy_learning_only_perseveration` | mapped | policy-learning-only + perseveration |
| `AP_RL_NoStay` | `social_policy_learning_only` | mapped | policy-learning-only |
| `ConditionedSharedDeltaModel` | `ConditionedSharedDeltaModel` | partial | class exists but no plugin/config integration |
| `ConditionedSharedDeltaSocialModel` | `ConditionedSharedDeltaSocialModel` | partial | class exists but no plugin/config integration |
| `Vicarious_AP_VS` (v1 exported only) | `social_policy_reliability_gated_value_shaping` | mapped | now first-class registered component |

Stan support for current mapped models:

- asocial + social families supported:
  - `src/comp_model/inference/hierarchical_stan.py:63`
  - `src/comp_model/inference/hierarchical_stan.py:72`
  - `src/comp_model/inference/hierarchical_stan_social.py:127`
- per-model Stan sources present:
  - `src/comp_model/inference/stan/within_subject/*.stan`

---

## Implementation Plan to Close Remaining Gaps

### Phase 1 (highest priority): Recovery parity

1. Extend parameter-recovery config schema with v1-equivalent sampling controls:
   - `sampling.mode = independent | hierarchical | fixed`
   - `sampling.space = param | z`
   - `sampling.by_condition` for shared+delta wrappers
2. Remove block-only guard:
   - support `simulation.level` for `block | subject | study` in parameter recovery.
3. Add tests:
   - independent/fixed/hierarchical sampling
   - by-condition overrides
   - subject/study level generation paths

### Phase 2: Shared+delta first-class integration

1. Decide canonical wrapper registration strategy (no legacy aliases):
   - either plugin manifests for wrapper constructors, or strict config helpers that build wrappers deterministically.
2. Add config-driven wrapper fitting/recovery tests.
3. Add Stan route for wrappers if required for parity use-cases.

### Phase 3: API cleanup (remove dead/legacy paths)

1. Remove `load_json_config` alias and update imports/tests/docs.
2. Remove removed-MAP placeholder APIs from public exports:
   - `inference.__init__`
   - `bayes.py`, `bayes_config.py`, `map_study_fitting.py` dead routes
3. Keep only supported surfaces:
   - MLE (`fit_dataset`, config MLE)
   - Stan Bayesian (`within_subject_hierarchical_stan_map|nuts`)

### Phase 4: Analysis parity + docs hardening

1. Reintroduce parameter-recovery plotting helper(s) in `comp_model.analysis` (optional dependency style).
2. Fix docs/API drift:
   - README `FitSpec` sample
   - broken how-to link in tutorial
3. Add a docs link check test or CI step to prevent regressions.

### Phase 5: Explicit parity assurance

1. Add model-level parity tests (behavioral invariants and parameter naming maps).
2. Add workflow parity smoke tests:
   - generate -> fit -> recover under v1-equivalent scenarios.
3. Keep parity report artifact in CI for traceability.

---

## Recommended Execution Order

1. Phase 3 quick hygiene (dead APIs + aliases + docs sample/link).
2. Phase 1 recovery parity (largest functional gap).
3. Phase 2 wrapper integration.
4. Phase 4 analysis plotting parity.
5. Phase 5 parity test harness.

This order minimizes user-facing confusion first, then closes substantive parity gaps.
