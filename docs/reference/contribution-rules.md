# Reference: Contribution Rules

This page defines repository rules for adding new computational models and
tasks.

## Core Rules

1. Use generic decision abstractions; do not introduce task-specific terms into
   core interfaces.
2. Preserve canonical event semantics:
   observation -> decision -> outcome -> update.
3. Do not add backward-compatibility aliases for removed internal names.
4. Keep plugin component IDs descriptive, stable, and snake_case.

## Model Rules

1. Every model must implement the full model interface methods.
2. Every public model class docstring must include:
   - `Model Contract`
   - `Decision Rule`
   - `Update Rule`
3. Parameter names must be explicit and scientifically meaningful.
4. Decision policy outputs must be valid probability distributions after runtime
   normalization.

## Task/Program Rules

1. Multi-phase tasks must be represented by explicit trial-program nodes and
   actor IDs.
2. Actor and learner identities must be explicit in events when they diverge.
3. Task implementations must be replay-compatible through canonical traces.

## Config and Validation Rules

1. Public config parsers must fail on unknown keys.
2. New config fields require:
   - parser updates,
   - schema doc updates,
   - tests for valid and invalid cases.
3. Estimator dispatch must remain explicit and auditable.

## Testing Rules

1. Add positive-path tests for new behavior.
2. Add negative-path tests for invalid inputs.
3. Add deterministic seeded tests where randomness is involved.
4. Full test suite must pass before merge.

## Documentation Rules

1. Research-facing overview content belongs in top-level `README.md`.
2. Technical details belong in `docs/` pages by document type:
   - tutorials,
   - how-to guides,
   - reference,
   - explanation.
3. Public API changes must update reference docs in the same PR.
