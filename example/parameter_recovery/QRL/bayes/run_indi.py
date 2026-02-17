from comp_model_analysis import plot_parameter_recovery
from comp_model_impl.recovery.parameter.config import load_parameter_recovery_config
from comp_model_impl.recovery.parameter.run import run_parameter_recovery


def main():
    # Load parameter recovery configure
    param_config = load_parameter_recovery_config("param_recovery_configs/indi.yaml")
    # Run the parameter recovery analysis
    results = run_parameter_recovery(config=param_config)
    # Plot the results
    plot_parameter_recovery(outputs=results, out_dir=f"{results.out_dir}/plots", split_by_rep=True)


if __name__ == "__main__":
    main()
