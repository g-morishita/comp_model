# Parameter Recovery GUI (Streamlit)

Local, lightweight GUI for running parameter recovery.

## Requirements

- Python environment with `streamlit` and `pyyaml` installed.
- All project dependencies needed to run parameter recovery.

## Run

From the repo root:

```bash
streamlit run apps/parameter_recovery_gui/app.py
```

## Notes

- Dropdowns populate a default plan/config, but the YAML editors are the source of truth at run time.
- For custom bandit/demonstrator/model settings, edit the YAML directly.
- Output runs are stored under the `Output root directory` (default: `apps/parameter_recovery_gui/runs`).
