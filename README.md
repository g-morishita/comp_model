# comp_model

`comp_model` is a clean-slate computational decision-modeling library for
simulation, replay, fitting, model comparison, and recovery workflows.

It is designed around a generic decision loop:

1. a model receives context from a problem/environment,
2. the model chooses an action,
3. the environment returns outcomes/observations,
4. the model updates internal state (or no-op updates for static policies).

`Bandit` is only one concrete problem family, not the core abstraction.

## Design Philosophy

- Generic first: core interfaces describe decision problems and agent models,
  not a specific task family.
- One canonical event semantics: simulation, replay likelihood, inference, and
  recovery all operate on the same event representation.
- Strict contracts: config schemas and component requirements fail fast on
  malformed inputs.
- Separation of concerns: runtime, models, inference, analysis, recovery, I/O,
  and plugin registry live in separate packages with explicit boundaries.
- Practical reproducibility: deterministic seeds and tabular I/O are first-class
  to support iterative scientific workflows.

## Package Map

- `comp_model.core`: contracts, event/data structures, validation primitives
- `comp_model.runtime`: simulation engines and replay execution
- `comp_model.models`: canonical decision-model implementations
- `comp_model.problems`: concrete task/problem implementations
- `comp_model.inference`: MLE/MAP/MCMC fitting and model comparison
- `comp_model.recovery`: parameter/model recovery workflows
- `comp_model.analysis`: information criteria and profile-likelihood tools
- `comp_model.io`: tabular CSV import/export
- `comp_model.plugins`: registry and component manifests

## Documentation

Documentation follows the Divio system (Diataxis):

- Tutorials: `docs/tutorials/`
- How-to guides: `docs/how-to/`
- Reference: `docs/reference/`
- Explanation: `docs/explanation/`

MkDocs config is provided in `mkdocs.yml`.

## Automation

- CI is defined in `.github/workflows/ci.yml` and runs tests plus docs build on
  push/pull request.
- Docs deployment is defined in `.github/workflows/deploy-docs.yml` and
  publishes MkDocs to GitHub Pages from `main`.
- Local pre-push test hook is provided in `.githooks/pre-push`.

Install local hooks with:

```bash
./scripts/install_git_hooks.sh
```

## Quick Start

```python
from comp_model.problems import StationaryBanditProblem
from comp_model.models import AsocialStateQValueSoftmaxModel
from comp_model.runtime import SimulationConfig, run_episode
from comp_model.inference import fit_model, FitSpec

problem = StationaryBanditProblem(reward_probabilities=[0.2, 0.8])
model = AsocialStateQValueSoftmaxModel(alpha=0.2, beta=3.0, initial_value=0.0)
trace = run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=100, seed=7))

result = fit_model(
    trace,
    model_factory=lambda p: AsocialStateQValueSoftmaxModel(**p),
    fit_spec=FitSpec(
        estimator_type="grid_search",
        parameter_grid={
            "alpha": [0.1, 0.2, 0.3],
            "beta": [2.0, 3.0, 4.0],
            "initial_value": [0.0],
        },
    ),
)
print(result.best.params)
```

## Credits

- Original internal model suite and research framing: Morishita Lab.
