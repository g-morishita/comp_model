# Lottery MVS Parameter Recovery Example

This example adds a first-pass lottery-choice workflow using:

- `LotteryChoiceBanditEnv` (trial-wise lottery menus)
- `MVS` model (`E + lambda_var * Var + delta * Skew`)
- event-log simulation + MLE fitting via the existing recovery runner

## Files

- `lottery_between_subject_plan_64.yaml`: 64-trial between-subject plan
- `lottery_between_subject_plan_96.yaml`: 96-trial between-subject plan (recommended default)
- `lottery_within_subject_plan.yaml`: legacy alias currently mirroring the 96-trial between-subject plan
- `lottery_mvs_recovery.yaml`: recovery/sampling configuration
- `run_lottery_mvs_recovery.py`: runnable script

## Run

From repository root:

```bash
python example/lottery_mvs/run_lottery_mvs_recovery.py \
  --config example/lottery_mvs/lottery_mvs_recovery.yaml --plots
```

Output is written to:

- `recovery_out_lottery_mvs/`

## Trial Count Guidance

- Use `96` trials/subject when estimating `lambda_var`, `delta`, and `beta` jointly.
- Use `64` trials/subject as a lighter option when runtime is a concern.
