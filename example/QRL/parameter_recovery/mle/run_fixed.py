from comp_model_analysis import plot_parameter_recovery
from comp_model_impl.recovery.parameter.config import load_parameter_recovery_config
from comp_model_impl.recovery.parameter.run import run_parameter_recovery


def main():
    param_config = load_parameter_recovery_config("param_recovery_configs/fixed.yaml")
    results = run_parameter_recovery(config=param_config)
    plot_paths = plot_parameter_recovery(outputs=results, out_dir=f"{results.out_dir}/plots")
    if plot_paths:
        print("Saved recovery plots:")
        for key, path in plot_paths.items():
            print(f"- {key}: {path}")
    return results


if __name__ == "__main__":
    main()
