# How-to: Run Recovery Workflows

Use this guide to run parameter recovery or model recovery from config.

## 1. Prepare Recovery Config

Use strict config schemas from:

- [`reference/config_schemas.md`](../reference/config_schemas.md)

At minimum:

- parameter recovery requires `generating_model`, `fitting_model`,
  `true_parameter_sets`, and fitting setup.
- model recovery requires `generating`, `candidates`, and a comparison
  criterion.

## 2. Run Recovery CLI

```bash
comp-model-recovery \
  --config recovery_config.json \
  --mode auto \
  --output-dir recovery_out \
  --prefix run1
```

`--mode auto` infers parameter vs model recovery from top-level config keys.

## 3. Inspect Artifacts

Parameter recovery writes:

- `*_parameter_cases.csv`
- `*_parameter_summary.json`

Model recovery writes:

- `*_model_cases.csv`
- `*_model_confusion.csv`
- `*_model_summary.json`

## Practical Tips

- Start with deterministic seeds.
- Keep simulation and fitting likelihood settings aligned.
- For social workflows, prefer generator-based simulation configs with explicit
  actor timing semantics.
- For subject/study model recovery, set `block_fit_strategy` to `joint` when
  you want one shared parameter set per subject across blocks.
