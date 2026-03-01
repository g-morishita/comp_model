# How-to: Contribute New Models and Tasks

Welcome contributors building new computational models and tasks.

Use this guide when you want to add:

- a new model family,
- a new decision task/problem,
- a new generator or demonstrator tied to task timing.

## Workflow Overview

1. Define the scientific behavior you need (decision rule and update rule).
2. Implement task/model code in the appropriate package.
3. Register components in the plugin registry.
4. Add tests for simulation, replay, and fitting behavior.
5. Update docs and schemas when public APIs/configs change.
6. Run checks locally before pushing.

## Add a New Model

1. Create a model class in `src/comp_model/models/`.
2. Implement required model interface methods:
   - `start_episode`
   - `action_distribution`
   - `update`
3. Use canonical constructor names and parameter names.
4. Add NumPy-style docstrings with consistent contract sections:
   - `Model Contract`
   - `Decision Rule`
   - `Update Rule`
5. Add plugin manifest entry with a canonical `component_id`.
6. Export from `src/comp_model/models/__init__.py`.
7. Add model tests (deterministic behavior and edge cases).

## Add a New Task/Problem

1. Add problem or trial-program implementation under `src/comp_model/problems/`
   (or `src/comp_model/runtime/program.py`-compatible class).
2. Keep event semantics explicit:
   - observation -> decision -> outcome -> update per decision node.
3. For multi-phase social tasks, encode phases via trial-program nodes rather
   than ad-hoc flags.
4. Add plugin manifest and factory for config-driven workflows.
5. Add tests:
   - simulation trace validity,
   - actor ordering and phase ordering,
   - replay compatibility.

## Add Config-Driven Support (If Needed)

If your model/task adds new public config fields:

1. Extend config parsers in `src/comp_model/inference/*_config.py` or recovery
   config module.
2. Validate unknown keys strictly.
3. Update `docs/reference/config_schemas.md`.
4. Add both positive and negative tests for schema parsing.

## Testing Checklist

Run these before creating a PR:

```bash
python -m pytest -q
mkdocs build --strict
```

If you enabled repository hooks:

```bash
./scripts/install_git_hooks.sh
```

`pre-push` will run tests automatically.

## Where to Put Documentation

- Task-oriented instructions: `docs/how-to/`
- API and schema facts: `docs/reference/`
- Rationale and architecture: `docs/explanation/`
- Learning walkthroughs: `docs/tutorials/`

For contribution standards, see
[Contribution Rules](../reference/contribution-rules.md).
