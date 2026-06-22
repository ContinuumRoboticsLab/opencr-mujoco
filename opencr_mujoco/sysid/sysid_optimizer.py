"""Main system identification optimizer using parallel multi-start optimization."""

import copy
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

from .parameters import ParameterRegistry
from .data_loader import TrajectoryDataLoader
from .trajectory_simulator import TrajectorySimulator
from .error_metrics import create_metrics_from_config
from .visualization import SysIDVisualizer
from ..generators.unified_tdcr_generator import create_tdcr_from_config


class SystemIdentificationOptimizer:
    """Main optimizer for TDCR system identification."""

    def __init__(
        self,
        config: Dict[str, Any],
        enable_viewer: bool = False,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        """Initialize system identification optimizer.

        Args:
            config: System identification configuration
            enable_viewer: If True, launch passive MuJoCo viewer for visualization
            output_dir: If provided, use this directory for output instead of auto-generating
            verbose: If True, print per-iteration logs (iteration banner,
                parameter values, metric). Default off: the parallel multistart
                runs the objective in many worker processes at once, so
                per-iteration prints interleave into an unreadable firehose.
        """
        self.config = config
        self.verbose = verbose
        self.base_generation_config = self._load_base_config()
        self.parameter_registry = ParameterRegistry(config, verbose=verbose)
        self.data_loader = None
        self.simulator = TrajectorySimulator(
            config.get("data", {}), enable_viewer=enable_viewer, verbose=verbose
        )
        self.metrics = create_metrics_from_config(config.get("metrics", {}))
        self.visualizer = None
        self.output_dir = None
        self.temp_dir = None

        # Optimization tracking
        self.iteration = 0
        self.best_error = float("inf")
        self.best_params = None
        self.best_config = None
        self.param_history = []
        self.error_history = []

        # Setup output directory
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir = self.output_dir / "temp_models"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            # Save configuration
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(self.config, f, indent=2)
            if self.verbose:
                print(f"Output directory: {self.output_dir}")
        else:
            self._setup_output_dir()

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base generation configuration.

        Returns:
            Base TDCR generation config
        """
        base_config_path = Path(self.config["base_generation_config"])
        if not base_config_path.exists():
            # Try relative to project root
            from ..utils.config_loader import PROJECT_ROOT

            base_config_path = PROJECT_ROOT / base_config_path

        with open(base_config_path, "r") as f:
            return json.load(f)

    def _setup_output_dir(self):
        """Setup output directory for results."""
        output_base = self.config.get("output", {}).get("base_dir", "sysid_results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = self.config.get("name", "unnamed")

        self.output_dir = Path(output_base) / f"{config_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create temp directory for generated models
        self.temp_dir = self.output_dir / "temp_models"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        if self.verbose:
            print(f"Output directory: {self.output_dir}")

    def load_data(self, data_file: str):
        """Load trajectory data.

        Args:
            data_file: Path to CSV data file
        """
        self.data_loader = TrajectoryDataLoader(
            data_file, self.config.get("data", {}), verbose=self.verbose
        )

        # Get trajectory data
        self.trajectory = self.data_loader.get_full_trajectory()
        self.timestamps = self.trajectory["timestamps"]
        self.actuator_commands = self.trajectory["actuator_commands"]
        self.real_marker_positions = self.trajectory["marker_positions"]

        if self.verbose:
            print(f"Loaded {len(self.timestamps)} trajectory samples")

    def objective_function(self, params: np.ndarray) -> float:
        """Objective function for optimization.

        Args:
            params: Parameter values to evaluate

        Returns:
            Error value (RMSE or combined metric)
        """
        self.iteration += 1
        if self.verbose:
            print(f"\n--- Iteration {self.iteration} ---")

        # Apply parameters to generation config. Deep copy: the parameter
        # classes mutate nested dicts (actuator_properties, sysid_params,
        # material_properties), so a shallow copy would pollute the base
        # config across iterations and alias best_config.
        try:
            generation_config = self.parameter_registry.apply_to_config(
                params, copy.deepcopy(self.base_generation_config)
            )
        except Exception as e:
            print(f"Error applying parameters: {e}")
            return 1e6  # Return large error

        # Generate MJCF model
        model_path = self.temp_dir / f"model_iter{self.iteration}.xml"
        try:
            create_tdcr_from_config(generation_config, str(model_path))
        except Exception as e:
            print(f"Error generating model: {e}")
            return 1e6

        # Load model and simulate
        try:
            # Extract slack scaling if present (can be scalar or array)
            slack_scaling = generation_config.get("sysid_params", {}).get(
                "tendon_slack_scaling", 1.0
            )
            if isinstance(slack_scaling, list):
                self.simulator.slack_scaling = np.array(slack_scaling)
            else:
                self.simulator.slack_scaling = slack_scaling

            self.simulator.load_model(str(model_path))

            # Pass tendon friction params to simulator
            sysid_params = generation_config.get("sysid_params", {})
            num_tendons = len(self.simulator.actuator_ids)
            friction_const = sysid_params.get("tendon_friction_const", [])
            friction_linear = sysid_params.get("tendon_friction_linear", [])
            if friction_const:
                # Expand per-segment values to per-tendon (tendons are
                # seg-major, so tendons-per-segment = total / num_segments)
                tps = num_tendons // len(friction_const)
                self.simulator.config["_sysid_friction_const"] = [
                    friction_const[j // tps] for j in range(num_tendons)
                ]
            if friction_linear:
                tps = num_tendons // len(friction_linear)
                self.simulator.config["_sysid_friction_linear"] = [
                    friction_linear[j // tps] for j in range(num_tendons)
                ]

            # Quasi-static rollout over the recorded tendon commands
            settling_time = self.config.get("simulation", {}).get("settling_time", 1.0)
            tip_positions, simulated_marker_positions = (
                self.simulator.simulate_trajectory(
                    self.actuator_commands, self.timestamps, settling_time
                )
            )

        except Exception as e:
            print(f"Error during simulation: {e}")
            return 1e6

        # Align: subtract mean offset to remove constant position bias
        # This changes each iteration as model parameters shift the neutral position
        offset = np.mean(
            simulated_marker_positions - self.real_marker_positions, axis=0
        )
        simulated_marker_positions = simulated_marker_positions - offset

        # Compute error
        error = self.metrics.compute(
            self.real_marker_positions, simulated_marker_positions
        )

        # Update tracking
        self.param_history.append(params.copy())
        self.error_history.append(error)

        # Check if best (before updating self.best_error)
        is_new_best = error < self.best_error

        if is_new_best:
            self.best_error = error
            self.best_params = params.copy()
            self.best_config = copy.deepcopy(generation_config)

            # Save best model
            best_model_path = self.output_dir / "best_model.xml"
            shutil.copy(model_path, best_model_path)

            # Save best config
            with open(self.output_dir / "best_generation_config.json", "w") as f:
                json.dump(generation_config, f, indent=2)

            if self.verbose:
                print(f"  New best! RMSE: {error*1000:.3f} mm")

        if self.verbose:
            # Print parameter values
            print(self.parameter_registry.format_values(params))
            print(f"  RMSE: {error*1000:.3f} mm (Best: {self.best_error*1000:.3f} mm)")

        # Visualization update
        if self.visualizer:
            self.visualizer.update_optimization_history(self.iteration, error, params)

            # Plot only when new best is found
            if is_new_best:
                self.visualizer.plot_trajectory_comparison(
                    self.real_marker_positions,
                    simulated_marker_positions,
                    title=f"Iter {self.iteration} - BEST (RMSE: {error*1000:.1f}mm)",
                    save_name=f"traj_best_iter{self.iteration}.png",
                )

        # Clean up old model files to save space
        if self.iteration > 10:
            old_model = self.temp_dir / f"model_iter{self.iteration-10}.xml"
            if old_model.exists():
                old_model.unlink()

        return error

    def optimize(self):
        """Run optimization process."""
        if self.data_loader is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Setup visualizer
        viz_config = self.config.get("visualization", {})
        self.visualizer = SysIDVisualizer(
            self.output_dir / "plots", viz_config.get("show_live", False)
        )

        # Get optimization config
        opt_config = self.config.get("optimization", {})

        # Get parameter bounds and initial values
        bounds = self.parameter_registry.get_bounds()
        initial_values = self.parameter_registry.get_initial_values()
        dim_names = self.parameter_registry.get_dimension_names()

        print(f"\n{'='*60}")
        print("SYSTEM IDENTIFICATION OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Parameters: {self.parameter_registry.get_parameter_names()}")
        print(f"Total dimensions: {len(bounds)}")
        print(f"{'='*60}\n")

        # parallel_multistart is the only implemented optimizer; warn (rather
        # than silently ignore) if a config requests something else. The
        # kp-fixed problem is unimodal, so multistart converges reliably and
        # Bayesian optimization offers no benefit (see docs/optimizer notes).
        algorithm = opt_config.get("algorithm", "parallel_multistart")
        if algorithm != "parallel_multistart":
            print(
                f"  [warning] optimization.algorithm='{algorithm}' is not "
                f"implemented; using parallel_multistart."
            )

        # Run optimization (parallel multi-start Powell; see parallel_optimizer)
        self._optimize_parallel_multistart(
            opt_config, bounds, dim_names, initial_values
        )

        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Best RMSE: {self.best_error*1000:.3f} mm")
        print("\nBest parameters:")
        print(self.parameter_registry.format_values(self.best_params))

        # Save final results
        self._save_results()

    def _optimize_parallel_multistart(
        self, opt_config, bounds, dim_names, initial_values
    ):
        """Parallel multi-start Powell search (uses all cores; see parallel_optimizer).

        Each worker rebuilds its own MuJoCo model from self.config (the simulator
        isn't picklable). N Sobol starts run concurrently; the best is evaluated
        once here on the FULL trajectory to set best_* and save the model/config.

        Config (under 'optimization'):
            n_starts: int (default 11)         number of Powell starts
            workers: int (default cpu-1)       parallel processes
            search_stride: int (default 1)     subsample trajectory for speed (rmse is stride-robust)
            maxfev: int (default 120)          Powell evals per start
        """
        import shutil
        from .parallel_optimizer import parallel_multistart

        ms_dir = self.output_dir / "_ms"
        best_theta, best_err_search, ms_results = parallel_multistart(
            self.config,
            bounds,
            dim_names,
            n_starts=opt_config.get("n_starts", 11),
            workers=opt_config.get("workers"),
            search_stride=opt_config.get("search_stride", 1),
            maxfev=opt_config.get("maxfev", 120),
            seed_x0=initial_values,
            workdir=ms_dir,
            verbose=self.verbose,
        )
        shutil.rmtree(ms_dir, ignore_errors=True)
        self._multistart_results = ms_results
        # Evaluate the winner on the FULL trajectory in this process: this sets
        # best_*/best_config and saves best_model.xml via objective_function.
        print("  re-evaluating winner on the full trajectory...")
        full_err = self.objective_function(np.asarray(best_theta))
        self.best_params = np.asarray(best_theta)
        self.best_error = min(full_err, self.best_error)
        print(
            f"  parallel multi-start best (full data): {self.best_error * 1000:.3f} mm"
        )

    def _save_results(self):
        """Save optimization results."""
        # Save parameter history
        param_history = np.array(self.param_history)
        np.savetxt(
            self.output_dir / "parameter_history.csv",
            param_history,
            delimiter=",",
            header=",".join(self.parameter_registry.get_dimension_names()),
        )

        # Save error history
        np.savetxt(self.output_dir / "error_history.csv", self.error_history)

        # Save best parameters
        results = {
            "best_error": float(self.best_error),
            "best_error_mm": float(self.best_error * 1000),
            "best_parameters": self.best_params.tolist(),
            "parameter_names": self.parameter_registry.get_dimension_names(),
            "iterations": self.iteration,
            "final_generation_config": self.best_config,
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Create final visualizations
        if self.visualizer:
            # Simulate with best parameters one more time for final plots
            best_model = self.output_dir / "best_model.xml"
            if best_model.exists():
                self.simulator.load_model(str(best_model))
                settling_time = self.config.get("simulation", {}).get(
                    "settling_time", 1.0
                )
                _, best_simulated = self.simulator.simulate_trajectory(
                    self.actuator_commands, self.timestamps, settling_time
                )

                # Create summary plot
                self.visualizer.create_summary_plot(
                    self.real_marker_positions, best_simulated, save_name="summary.png"
                )

                # Plot optimization progress
                self.visualizer.plot_optimization_progress(
                    save_name="optimization_progress.png"
                )

                # Plot parameter evolution
                self.visualizer.plot_parameter_evolution(
                    self.param_history,
                    self.parameter_registry.get_dimension_names(),
                    save_name="parameter_evolution.png",
                )

        print(f"\nResults saved to: {self.output_dir}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        if self.simulator:
            self.simulator.cleanup()
