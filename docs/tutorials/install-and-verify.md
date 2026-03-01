# Tutorial: Install and Verify

This tutorial gets you from a clean environment to a verified `comp_model`
installation.

By the end, you will:

1. install the package,
2. run a smoke-check simulation,
3. verify CLI and documentation tooling.

## Step 1: Create an Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Step 2: Install `comp_model`

Install from the repository root:

```bash
pip install -e ".[dev,docs]"
```

## Step 3: Verify Python Import

```bash
python -c "import comp_model; print('comp_model import OK')"
```

Expected output includes:

```text
comp_model import OK
```

## Step 4: Verify a Minimal Simulation

```python
from comp_model.problems import StationaryBanditProblem
from comp_model.models import UniformRandomPolicyModel
from comp_model.runtime import SimulationConfig, run_episode

trace = run_episode(
    problem=StationaryBanditProblem(reward_probabilities=[0.4, 0.6]),
    model=UniformRandomPolicyModel(),
    config=SimulationConfig(n_trials=10, seed=1),
)
print(len(trace.events))
```

You should see a positive event count.

## Step 5: Verify CLI Tools

```bash
comp-model-fit --help
comp-model-compare --help
comp-model-recovery --help
```

## Step 6: Verify Documentation Build

```bash
mkdocs build --strict
```

## Step 7: Verify Test Suite

```bash
python -m pytest -q
```

## Next Tutorial

Continue with [First End-to-End Fit](first-end-to-end-fit.md).
