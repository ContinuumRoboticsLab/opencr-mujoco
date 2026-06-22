"""Pipeline orchestrator for 3-step TDCR system identification.

Step 1: Geometric identification (seg_offsets, tendon_angle_deltas) from tendon pulls
Step 2: Tendon parameter optimization (pretension, kp, constraint factor) on tendon pulls
Step 3: Refinement on train data, validation on val data
"""

import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .data_loader import TrajectoryDataLoader
from .geometric_identifier import GeometricIdentifier
from .pipeline_data_loader import PipelineDataLoader
from .sysid_optimizer import SystemIdentificationOptimizer
from .trajectory_simulator import TrajectorySimulator
from ..generators.unified_tdcr_generator import create_tdcr_from_config

# Output sub-directory names for each pipeline step.
STEP_DIR_NAMES = {1: "step1_geometric", 2: "step2_tendon_opt", 3: "step3_refinement"}


class PipelineOrchestrator:
    """Orchestrates the 3-step sysid pipeline."""

    def __init__(
        self, config: Dict[str, Any], enable_viewer: bool = False, verbose: bool = False
    ):
        """Initialize pipeline orchestrator.

        Args:
            config: Pipeline configuration dict
            enable_viewer: If True, launch MuJoCo viewer during simulation
            verbose: If True, the optimizers print raw per-iteration logs
                (sysid_pipeline.py --debug); default is a compact live
                progress display
        """
        self.config = config
        self.enable_viewer = enable_viewer
        self.verbose = verbose
        self.working_config = self._load_base_generation_config()

        # Setup output directory
        self.output_dir = self._setup_output_dir()

        # Data paths
        data_cfg = config["data"]
        data_dir = Path(data_cfg["data_dir"])
        self.tendon_pull_file = data_dir / data_cfg["tendon_pull_file"]
        self.train_file = data_dir / data_cfg["train_file"]
        self.val_file = data_dir / data_cfg["val_file"]

        # Validate data files exist
        for f in [self.tendon_pull_file, self.train_file, self.val_file]:
            if not f.exists():
                raise FileNotFoundError(f"Data file not found: {f}")

        # Pipeline state
        self.step_results = {}
        self.servo_mapping = None
        self.position_bias = None

        # Save pipeline config
        with open(self.output_dir / "pipeline_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Pipeline output directory: {self.output_dir}")

    def _load_base_generation_config(self) -> Dict:
        """Load the base generation config."""
        config_path = Path(self.config["base_generation_config"])
        if not config_path.exists():
            from ..utils.config_loader import PROJECT_ROOT

            config_path = PROJECT_ROOT / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Base generation config not found: {config_path}")
        with open(config_path) as f:
            return json.load(f)

    def _setup_output_dir(self) -> Path:
        """Create timestamped output directory."""
        output_base = self.config.get("output", {}).get("base_dir", "sysid_results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.config.get("name", "pipeline")
        output_dir = Path(output_base) / f"{name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def run(self, start_step: int = 1, end_step: int = 3):
        """Run the pipeline from start_step to end_step.

        Args:
            start_step: First step to run (1, 2, or 3)
            end_step: Last step to run (1, 2, or 3)
        """
        print(f"\n{'='*60}")
        print("SYSID PIPELINE")
        print(f"{'='*60}")
        print(f"Running steps {start_step} to {end_step}")
        print(f"{'='*60}\n")

        # Load results from prior steps if starting mid-pipeline
        if start_step > 1:
            for step in range(1, start_step):
                results = self._load_step_results(step)
                if results is None:
                    raise RuntimeError(
                        f"Cannot start at step {start_step}: "
                        f"step {step} results not found in {self.output_dir}"
                    )
                self.step_results[step] = results
                self._apply_step_results(step, results)

        # Run requested steps
        if start_step <= 1 and end_step >= 1:
            step1_cfg = self.config.get("step1_geometric", {})
            if step1_cfg.get("enabled", True):
                self.step_results[1] = self.run_step1_geometric_id()

        if start_step <= 2 and end_step >= 2:
            step2_cfg = self.config.get("step2_tendon_optimization", {})
            if step2_cfg.get("enabled", True):
                self.step_results[2] = self.run_step2_tendon_optimization()

        if start_step <= 3 and end_step >= 3:
            step3_cfg = self.config.get("step3_refinement", {})
            if step3_cfg.get("enabled", True):
                self.step_results[3] = self.run_step3_refinement_validation()

        # Save final artifacts
        self._save_final_results()

    def run_step1_geometric_id(self) -> Dict:
        """Step 1: Geometric identification from tendon pull data.

        Returns:
            Dict with seg_offsets, tendon_angle_deltas, servo_mapping, position_bias
        """
        print(f"\n{'='*60}")
        print("STEP 1: GEOMETRIC IDENTIFICATION")
        print(f"{'='*60}")

        step_dir = self.output_dir / STEP_DIR_NAMES[1]
        step_dir.mkdir(exist_ok=True)

        # Load tendon pull data
        loader = PipelineDataLoader(
            str(self.tendon_pull_file), self.config.get("data", {})
        )
        df = loader.load_raw_dataframe()

        # Run geometric identification
        kin_config = self.config.get("data", {}).get("kinematics", {})
        identifier = GeometricIdentifier(kin_config)
        results = identifier.identify(df)

        # Store mapping and bias for subsequent steps
        self.servo_mapping = results["servo_mapping"]
        self.position_bias = np.array(results["position_bias"])

        # Fix Z bias: Z should be relative to expected tip height, not zero.
        # expected_z = sum(segment_lengths) + marker_transform_z, all in mm
        expected_z_mm = self._compute_expected_z_mm()
        self.position_bias[2] = self.position_bias[2] - expected_z_mm
        results["position_bias"] = self.position_bias.tolist()
        print(f"  Expected Z height: {expected_z_mm:.2f} mm")
        print(
            f"  Adjusted Z bias: {self.position_bias[2]:.2f} mm (so neutral Z ≈ {expected_z_mm:.2f} mm after bias removal)"
        )

        # No angular correction needed: sim and data frames now share the same
        # backbone (+Z) and perpendicular plane (XY), so identified seg_offsets
        # are directly usable as generation config values.

        # Apply to working config
        self.working_config = identifier.apply_to_generation_config(
            results, self.working_config
        )

        # Save results
        self._save_step_results(1, results, step_dir)

        with open(step_dir / "generation_config.json", "w") as f:
            json.dump(self.working_config, f, indent=2)

        # Visualize geometric identification
        self._visualize_step1(results, step_dir)

        print(f"\nStep 1 complete. Results saved to {step_dir}")
        return results

    def run_step2_tendon_optimization(self) -> Dict:
        """Step 2: Optimize tendon parameters on tendon pull data.

        Returns:
            Dict with best_error, best_params, generation_config
        """
        print(f"\n{'='*60}")
        print("STEP 2: TENDON PARAMETER OPTIMIZATION")
        print(f"{'='*60}")

        step_dir = self.output_dir / STEP_DIR_NAMES[2]
        step_dir.mkdir(exist_ok=True)

        if self.servo_mapping is None:
            raise RuntimeError("Step 1 must run before Step 2 (need servo_mapping)")

        # Preprocess tendon_pull CSV
        preprocessed_csv = step_dir / "preprocessed_tendon_pull.csv"
        self._preprocess_csv(self.tendon_pull_file, preprocessed_csv)

        # Save current working config to temp file for optimizer
        gen_config_path = step_dir / "input_generation_config.json"
        with open(gen_config_path, "w") as f:
            json.dump(self.working_config, f, indent=2)

        # Build optimizer config
        step2_cfg = self.config.get("step2_tendon_optimization", {})
        optimizer_config = self._build_optimizer_config(
            step2_cfg, str(gen_config_path), str(preprocessed_csv)
        )

        # Run optimization
        optimizer = SystemIdentificationOptimizer(
            optimizer_config,
            enable_viewer=self.enable_viewer,
            output_dir=step_dir,
            verbose=self.verbose,
        )
        optimizer.load_data(str(preprocessed_csv))
        optimizer.optimize()

        # Extract results
        results = {
            "best_error": float(optimizer.best_error),
            "best_error_mm": float(optimizer.best_error * 1000),
            "best_params": (
                optimizer.best_params.tolist()
                if optimizer.best_params is not None
                else []
            ),
            "parameter_names": optimizer.parameter_registry.get_dimension_names(),
            "iterations": optimizer.iteration,
        }

        # Update working config with optimized parameters
        if optimizer.best_config is not None:
            self.working_config = optimizer.best_config

        optimizer.cleanup()

        with open(step_dir / "generation_config.json", "w") as f:
            json.dump(self.working_config, f, indent=2)

        self._save_step_results(2, results, step_dir)

        # Visualize: static plot + animated video
        self._visualize_trajectory_comparison(
            step_dir, preprocessed_csv, "Step 2: Tendon Pull Optimization"
        )
        self._render_comparison_video(
            step_dir,
            preprocessed_csv,
            "Step 2: Tendon Pull Optimization",
            video_name="step2_comparison.mp4",
        )

        print("\nStep 2 complete.")
        print(f"  RMSE: {results['best_error_mm']:.3f} mm")
        print(f"Results saved to {step_dir}")
        return results

    def run_step3_refinement_validation(self) -> Dict:
        """Step 3: Refine on train data, validate on val data.

        Returns:
            Dict with train_error, val_error, generation_config
        """
        print(f"\n{'='*60}")
        print("STEP 3: REFINEMENT & VALIDATION")
        print(f"{'='*60}")

        step_dir = self.output_dir / STEP_DIR_NAMES[3]
        step_dir.mkdir(exist_ok=True)

        if self.servo_mapping is None:
            raise RuntimeError("Step 1 must run before Step 3 (need servo_mapping)")

        # Preprocess train CSV
        preprocessed_train = step_dir / "preprocessed_train.csv"
        self._preprocess_csv(self.train_file, preprocessed_train)

        # Save current working config
        gen_config_path = step_dir / "input_generation_config.json"
        with open(gen_config_path, "w") as f:
            json.dump(self.working_config, f, indent=2)

        # Build optimizer config with tightened bounds
        step3_cfg = self.config.get("step3_refinement", {})
        optimizer_config = self._build_optimizer_config(
            step3_cfg, str(gen_config_path), str(preprocessed_train)
        )

        # Tighten bounds around Step 2 best values
        margin = step3_cfg.get("bounds_margin", 0.2)
        optimizer_config = self._tighten_bounds(optimizer_config, margin)

        # Run optimization on train data
        optimizer = SystemIdentificationOptimizer(
            optimizer_config,
            enable_viewer=self.enable_viewer,
            output_dir=step_dir,
            verbose=self.verbose,
        )
        optimizer.load_data(str(preprocessed_train))
        optimizer.optimize()

        train_error = float(optimizer.best_error)
        train_error_mm = train_error * 1000

        # Update working config
        if optimizer.best_config is not None:
            self.working_config = optimizer.best_config

        # Validate on val data
        val_error_mm = self._evaluate_on_validation(step_dir)

        optimizer.cleanup()

        results = {
            "train_error": train_error,
            "train_error_mm": train_error_mm,
            "val_error_mm": val_error_mm,
            "best_params": (
                optimizer.best_params.tolist()
                if optimizer.best_params is not None
                else []
            ),
            "parameter_names": optimizer.parameter_registry.get_dimension_names(),
            "iterations": optimizer.iteration,
        }

        with open(step_dir / "generation_config.json", "w") as f:
            json.dump(self.working_config, f, indent=2)

        # Visualize: trajectory comparisons for train and val
        self._visualize_trajectory_comparison(
            step_dir,
            preprocessed_train,
            "Step 3: Train Data",
            save_name="trajectory_train.png",
        )
        preprocessed_val = step_dir / "preprocessed_val.csv"
        if preprocessed_val.exists():
            self._visualize_trajectory_comparison(
                step_dir,
                preprocessed_val,
                "Step 3: Validation Data",
                save_name="trajectory_val.png",
            )

        # Animated videos
        self._render_comparison_video(
            step_dir,
            preprocessed_train,
            "Step 3: Train Data",
            video_name="step3_train.mp4",
        )
        if preprocessed_val.exists():
            self._render_comparison_video(
                step_dir,
                preprocessed_val,
                "Step 3: Validation Data",
                video_name="step3_val.mp4",
            )

        self._save_step_results(3, results, step_dir)

        print("\nStep 3 complete.")
        print(f"  Train RMSE: {train_error_mm:.3f} mm")
        print(f"  Val RMSE:   {val_error_mm:.3f} mm")
        print(f"Results saved to {step_dir}")
        return results

    @staticmethod
    def _build_metrics_config(metrics: Dict) -> Dict:
        """Normalize the metrics config. Only RMSE is implemented."""
        if "rmse" not in metrics:
            metrics["rmse"] = {"weight": 1.0}
        return metrics

    def _evaluate_on_validation(self, step_dir: Path) -> float:
        """Evaluate the best model on validation data.

        Standard (Euclidean, pointwise) RMSE with mean-offset alignment —
        the exact same convention as the training objective, so train and
        validation numbers are directly comparable.

        Args:
            step_dir: Step 3 output directory (contains best_model.xml)

        Returns:
            Validation RMSE in mm
        """
        print("\n--- Evaluating on validation data ---")

        # Preprocess val CSV
        preprocessed_val = step_dir / "preprocessed_val.csv"
        self._preprocess_csv(self.val_file, preprocessed_val)

        # Simulate the best model over the validation trajectory
        best_model_path = step_dir / "best_model.xml"
        sim = self._simulate_on_csv(best_model_path, preprocessed_val)
        if sim is None:
            print("  Warning: best_model.xml not found, skipping validation")
            return float("nan")
        traj, real_mm, sim_mm, _ = sim

        # Mean-offset alignment (same as the optimizer objective)
        sim_mm = sim_mm - np.mean(sim_mm - real_mm, axis=0)
        errors = np.linalg.norm(real_mm - sim_mm, axis=1)
        val_error_mm = float(np.sqrt(np.mean(errors**2)))

        print(f"  Validation RMSE: {val_error_mm:.3f} mm")
        print(f"  Validation mean error: {np.mean(errors):.3f} mm")
        print(f"  Validation samples: {len(traj['timestamps'])}")

        return val_error_mm

    def _compute_expected_z_mm(self) -> float:
        """Compute expected neutral tip Z height in mm.

        Sum of all segment lengths (from generation config) plus
        the marker_transform Z offset (from pipeline config).

        Returns:
            Expected Z height in mm.
        """
        # Sum segment lengths from generation config (in meters)
        seg_lengths = self.working_config.get("segment_lengths", {})
        total_length_m = sum(seg_lengths.values())

        # Marker transform Z offset (in meters)
        marker_z_m = (
            self.config.get("data", {})
            .get("marker_transform", {})
            .get("translation", [0, 0, 0])[2]
        )

        return (total_length_m + marker_z_m) * 1000.0  # convert to mm

    def _preprocess_csv(self, input_csv: Path, output_csv: Path):
        """Preprocess a raw CSV into standard format.

        Uses servo_mapping from Step 1 and per-CSV position bias
        (first data point) for alignment.
        """
        loader = PipelineDataLoader(str(input_csv), self.config.get("data", {}))
        df = loader.load_raw_dataframe()
        tip_cols = loader.detect_tip_columns(df)

        # Per-CSV bias from first data point
        first_point = np.array(
            [df[tip_cols[0]].iloc[0], df[tip_cols[1]].iloc[0], df[tip_cols[2]].iloc[0]]
        )
        expected_z_mm = self._compute_expected_z_mm()
        bias = first_point.copy()
        bias[2] = first_point[2] - expected_z_mm

        loader.preprocess_to_standard_csv(
            str(output_csv),
            servo_mapping=self.servo_mapping,
            position_bias=bias,
        )

    def _build_optimizer_config(
        self,
        step_config: Dict,
        gen_config_path: str,
        data_file: str,
    ) -> Dict:
        """Build a config dict compatible with SystemIdentificationOptimizer.

        Args:
            step_config: Step-specific config (step2 or step3 section)
            gen_config_path: Path to the generation config JSON file
            data_file: Path to preprocessed CSV file

        Returns:
            Config dict for SystemIdentificationOptimizer
        """
        data_cfg = self.config.get("data", {})

        optimizer_config = {
            "name": self.config.get("name", "pipeline"),
            "base_generation_config": gen_config_path,
            "data": {
                "trajectory_file": data_file,
                "marker_units": data_cfg.get("marker_units", "mm"),
                "marker_transform": data_cfg.get("marker_transform", {}),
                "kinematics": data_cfg.get("kinematics", {}),
            },
            "parameters": step_config.get("parameters", {}),
            "optimization": step_config.get(
                "optimization",
                {
                    "algorithm": "parallel_multistart",
                    "n_starts": 8,
                    "maxfev": 120,
                },
            ),
            "simulation": step_config.get(
                "simulation",
                {
                    "settling_time": 0.5,
                },
            ),
            "metrics": self._build_metrics_config(
                step_config.get(
                    "metrics",
                    {
                        "rmse": {"weight": 1.0},
                    },
                )
            ),
            "visualization": self.config.get(
                "visualization",
                {
                    "save_plots": True,
                    "show_live": False,
                },
            ),
        }

        # Copy num_segments/tendons_per_segment to parameter configs that need them
        kin = data_cfg.get("kinematics", {})
        num_segments = kin.get("num_segments", 3)
        tendons_per_segment = kin.get("tendons_per_segment", 3)

        for param_name, param_cfg in optimizer_config["parameters"].items():
            if param_cfg.get("enabled", False):
                if "num_segments" not in param_cfg:
                    param_cfg["num_segments"] = num_segments
                if "tendons_per_segment" not in param_cfg:
                    param_cfg["tendons_per_segment"] = tendons_per_segment

        return optimizer_config

    def _tighten_bounds(self, optimizer_config: Dict, margin: float = 0.2) -> Dict:
        """Tighten parameter bounds around current working config values.

        For each enabled parameter, finds the current value in self.working_config
        and narrows the bounds to ±margin around that value.

        Args:
            optimizer_config: Optimizer config to modify
            margin: Fraction for bound tightening (0.2 = ±20%)

        Returns:
            Modified optimizer config
        """
        params = optimizer_config.get("parameters", {})

        for param_name, param_cfg in params.items():
            if not param_cfg.get("enabled", False):
                continue

            # Get current values from working config
            current_values = self._get_current_param_values(param_name, param_cfg)
            if current_values is None:
                continue

            # Get original bounds (from step2 if available)
            step2_params = self.config.get("step2_tendon_optimization", {}).get(
                "parameters", {}
            )
            original_bounds = step2_params.get(param_name, {}).get(
                "bounds", param_cfg.get("bounds")
            )
            if original_bounds is None:
                continue

            orig_lo, orig_hi = original_bounds
            orig_range = orig_hi - orig_lo

            # Tighten bounds around current values
            if isinstance(current_values, (list, np.ndarray)):
                val_min = min(current_values)
                val_max = max(current_values)
            else:
                val_min = val_max = current_values

            # Add margin as fraction of original range around the extremes
            pad = orig_range * margin
            new_lo = max(orig_lo, val_min - pad)
            new_hi = min(orig_hi, val_max + pad)

            # Ensure some minimum range (5% of original)
            if new_hi - new_lo < orig_range * 0.05:
                center = (val_min + val_max) / 2
                min_range = orig_range * 0.05
                new_lo = max(orig_lo, center - min_range / 2)
                new_hi = min(orig_hi, center + min_range / 2)

            param_cfg["bounds"] = [float(new_lo), float(new_hi)]
            print(
                f"  Tightened {param_name} bounds: [{orig_lo}, {orig_hi}] → [{new_lo:.4f}, {new_hi:.4f}]"
            )

        return optimizer_config

    @staticmethod
    def _select_by_mode(values, param_cfg: Dict):
        """Reduce a per-tendon array to the representation the optimizer expects.

        'per_segment' returns the first tendon's value for each segment,
        'global' returns the single mean, any other mode returns values as-is.
        """
        mode = param_cfg.get("mode", "per_segment")
        if mode == "per_segment":
            num_seg = param_cfg.get("num_segments", 3)
            tps = param_cfg.get("tendons_per_segment", 3)
            return [values[s * tps] for s in range(num_seg)]
        elif mode == "global":
            return [np.mean(values)]
        return values

    def _get_current_param_values(self, param_name: str, param_cfg: Dict):
        """Extract current parameter values from working config.

        Args:
            param_name: Parameter type name
            param_cfg: Parameter config dict

        Returns:
            Current value(s) or None if not found
        """
        act_props = self.working_config.get("actuator_properties", {})

        if param_name == "pretension":
            vals = act_props.get("tendon_pretension")
            if vals is None:
                return None
            return self._select_by_mode(vals, param_cfg)

        elif param_name in ("tendon_kp", "tendon_stiffness"):
            kp_array = act_props.get("tendon_kp_array")
            if kp_array is not None:
                return self._select_by_mode(kp_array, param_cfg)
            kp = act_props.get("tendon_kp")
            return [kp] if kp is not None else None

        elif param_name == "tendon_constraint_factor":
            val = act_props.get("tendon_constraint_factor")
            if val is None:
                val = self.working_config.get("tendon_config", {}).get(
                    "tendon_constraint_factor"
                )
            return [val] if val is not None else None

        elif param_name == "joint_deadband":
            val = self.working_config.get("joint_deadband")
            if val is None:
                return None
            if isinstance(val, (int, float)):
                num_seg = param_cfg.get("num_segments", 3)
                return [val] * num_seg
            return val

        return None

    def _save_step_results(self, step: int, results: Dict, step_dir: Path):
        """Save step results to disk."""
        with open(step_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    def _load_step_results(self, step: int) -> Optional[Dict]:
        """Load step results from disk for resume capability."""
        step_dir = self.output_dir / STEP_DIR_NAMES[step]

        results_file = step_dir / "results.json"
        if not results_file.exists():
            return None

        with open(results_file) as f:
            results = json.load(f)

        # Also load the generation config from that step
        gen_config_file = step_dir / "generation_config.json"
        if gen_config_file.exists():
            with open(gen_config_file) as f:
                self.working_config = json.load(f)

        return results

    def _apply_step_results(self, step: int, results: Dict):
        """Apply loaded step results to pipeline state."""
        if step == 1:
            self.servo_mapping = results.get("servo_mapping")
            # Convert string keys back to int keys
            if self.servo_mapping:
                self.servo_mapping = {int(k): v for k, v in self.servo_mapping.items()}
            bias = results.get("position_bias")
            self.position_bias = np.array(bias) if bias is not None else None

    # ---- Simulation helpers ----

    def _inject_friction_params(self, simulator):
        """Copy per-tendon friction params from working_config onto the simulator.

        Must be called before simulator.load_model(); a no-op when the
        working config carries no friction params.
        """
        sysid_params = self.working_config.get("sysid_params", {})
        fc = sysid_params.get("tendon_friction_const", [])
        fl = sysid_params.get("tendon_friction_linear", [])
        # Friction arrays are per-segment; expand to per-tendon (seg-major)
        # using the actuation details rather than hardcoding a 3x3 robot.
        segments = self.working_config.get("actuation_details", {}).get("segments", [])
        if segments:
            tendons_per_segment = [s_.get("number_of_tendons", 3) for s_ in segments]
        else:
            tendons_per_segment = [3] * (len(fc) or len(fl) or 3)
        if fc:
            simulator.config["_sysid_friction_const"] = [
                fc[seg] for seg, n in enumerate(tendons_per_segment) for _ in range(n)
            ]
        if fl:
            simulator.config["_sysid_friction_linear"] = [
                fl[seg] for seg, n in enumerate(tendons_per_segment) for _ in range(n)
            ]

    def _simulate_on_csv(self, model_path, preprocessed_csv, inject_friction=False):
        """Simulate ``model_path`` over a preprocessed trajectory CSV.

        Returns ``(traj, real_mm, sim_mm, pattern_labels)`` with marker
        positions converted to mm, or ``None`` if the model file is missing.
        Callers apply their own alignment (none / mean-offset / first-point)
        to ``sim_mm``.
        """
        if not Path(model_path).exists():
            return None

        loader = TrajectoryDataLoader(
            str(preprocessed_csv), self.config.get("data", {})
        )
        traj = loader.get_full_trajectory()

        raw_df = pd.read_csv(str(preprocessed_csv))
        pattern_labels = (
            raw_df["pattern_label"].values
            if "pattern_label" in raw_df.columns
            else None
        )

        simulator = TrajectorySimulator(
            self.config.get("data", {}), enable_viewer=False
        )
        if inject_friction:
            self._inject_friction_params(simulator)
        simulator.load_model(str(model_path))
        # Use the same settling time as the optimization itself, so reported
        # metrics/plots are computed under identical quasi-static conditions.
        settling_time = 1.0
        for step_key in ("step3_refinement", "step2_tendon_optimization"):
            step_sim = self.config.get(step_key, {}).get("simulation", {})
            if "settling_time" in step_sim:
                settling_time = step_sim["settling_time"]
                break
        _, sim_marker = simulator.simulate_trajectory(
            traj["actuator_commands"], traj["timestamps"], settling_time
        )
        simulator.cleanup()

        real_mm = traj["marker_positions"] * 1000
        sim_mm = sim_marker * 1000
        return traj, real_mm, sim_mm, pattern_labels

    # ---- Visualization methods ----

    def _visualize_step1(self, results: Dict, step_dir: Path):
        """Generate model from Step 1 params, simulate tendon pulls, and create
        an animated XY comparison video of real vs simulated tip motion."""
        try:
            # Generate a model with step 1 geometric params (default pretension/kp)
            model_path = step_dir / "step1_model.xml"
            create_tdcr_from_config(self.working_config, str(model_path))

            # Preprocess tendon pull CSV
            preprocessed = step_dir / "preprocessed_tendon_pull.csv"
            self._preprocess_csv(self.tendon_pull_file, preprocessed)

            # Static summary plot + animated video
            self._visualize_trajectory_comparison(
                step_dir,
                preprocessed,
                "Step 1: Geometric ID (default pretension/kp)",
                save_name="trajectory_comparison.png",
                model_path=model_path,
            )
            self._render_comparison_video(
                step_dir,
                preprocessed,
                "Step 1: Geometric Verification",
                video_name="step1_verification.mp4",
                model_path=model_path,
            )
        except Exception as e:
            print(f"  Warning: Step 1 visualization failed: {e}")
            traceback.print_exc()

    def _render_comparison_video(
        self,
        step_dir: Path,
        preprocessed_csv: Path,
        title: str,
        video_name: str = "comparison.mp4",
        model_path: Optional[Path] = None,
    ):
        """Render animated XY comparison video of real vs simulated tip motion."""
        if not self.config.get("visualization", {}).get("render_videos", True):
            return
        try:
            if model_path is None:
                model_path = step_dir / "best_model.xml"
            sim = self._simulate_on_csv(model_path, preprocessed_csv)
            if sim is None:
                print(f"  No model at {model_path}, skipping video")
                return
            _, real_mm, sim_mm, pattern_labels = sim

            # Align first points
            real_mm = real_mm - real_mm[0]
            sim_mm = sim_mm - sim_mm[0]
            n_samples = len(real_mm)

            # Standard pointwise errors for display
            errors = np.linalg.norm(real_mm - sim_mm, axis=1)

            fig, (ax_xy, ax_err) = plt.subplots(1, 2, figsize=(14, 6))

            all_xy = np.vstack([real_mm[:, :2], sim_mm[:, :2]])
            pad = 10
            ax_xy.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
            ax_xy.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
            ax_xy.set_xlabel("X (mm)")
            ax_xy.set_ylabel("Y (mm)")
            ax_xy.set_aspect("equal")
            ax_xy.grid(True, alpha=0.3)

            (real_line,) = ax_xy.plot([], [], "g-", alpha=0.4, linewidth=1)
            (sim_line,) = ax_xy.plot([], [], "k-", alpha=0.4, linewidth=1)
            (real_dot,) = ax_xy.plot([], [], "go", markersize=10, label="Real")
            (sim_dot,) = ax_xy.plot([], [], "ks", markersize=8, label="Sim")
            ax_xy.legend(loc="upper right")
            title_text = ax_xy.set_title(title)

            ax_err.set_xlim(0, n_samples)
            ax_err.set_ylim(0, max(np.nanmax(errors) * 1.1, 1))
            ax_err.set_xlabel("Sample")
            ax_err.set_ylabel("Error (mm)")
            ax_err.set_title("Position Error")
            ax_err.grid(True, alpha=0.3)
            (err_line,) = ax_err.plot([], [], "r-", linewidth=1)
            mean_text = ax_err.text(
                0.02, 0.95, "", transform=ax_err.transAxes, va="top"
            )

            def update(frame):
                i = frame
                real_line.set_data(real_mm[: i + 1, 0], real_mm[: i + 1, 1])
                sim_line.set_data(sim_mm[: i + 1, 0], sim_mm[: i + 1, 1])
                real_dot.set_data([real_mm[i, 0]], [real_mm[i, 1]])
                sim_dot.set_data([sim_mm[i, 0]], [sim_mm[i, 1]])
                err_line.set_data(np.arange(i + 1), errors[: i + 1])
                mean_text.set_text(
                    f"Mean: {np.nanmean(errors[:i+1]):.1f}mm\nCurrent: {errors[i]:.1f}mm"
                )
                label = pattern_labels[i] if pattern_labels is not None else ""
                title_text.set_text(f"{title}  [{i+1}/{n_samples}]  {label}")
                return (
                    real_line,
                    sim_line,
                    real_dot,
                    sim_dot,
                    err_line,
                    mean_text,
                    title_text,
                )

            anim = animation.FuncAnimation(
                fig, update, frames=n_samples, interval=50, blit=True
            )
            video_path = step_dir / video_name
            anim.save(str(video_path), writer="ffmpeg", fps=20, dpi=100)
            plt.close(fig)
            print(f"  Saved: {video_path}")
        except Exception as e:
            print(f"  Warning: Video rendering failed: {e}")
            traceback.print_exc()

    def _visualize_trajectory_comparison(
        self,
        step_dir: Path,
        preprocessed_csv: Path,
        title: str,
        save_name: str = "trajectory_comparison.png",
        model_path: Optional[Path] = None,
    ):
        """Simulate best model and plot trajectory comparison with real data.

        Error is the standard (Euclidean, pointwise) RMSE with mean-offset
        alignment — the same convention as the optimizer objective and the
        reported train/val numbers, so the plot annotation matches
        results.json.
        """
        try:
            if model_path is None:
                model_path = step_dir / "best_model.xml"
            sim = self._simulate_on_csv(model_path, preprocessed_csv)
            if sim is None:
                print(f"  No model at {model_path}, skipping visualization")
                return
            _, real_mm, sim_mm, _ = sim

            # Mean-offset alignment (same as the optimizer objective)
            sim_mm = sim_mm - np.mean(sim_mm - real_mm, axis=0)
            errors = np.linalg.norm(real_mm - sim_mm, axis=1)
            rmse = np.sqrt(np.mean(errors**2))

            # Create 2x2 figure: 3D, XY, XZ, error
            fig = plt.figure(figsize=(14, 12))

            # 3D trajectory
            ax1 = fig.add_subplot(221, projection="3d")
            ax1.plot(
                real_mm[:, 0],
                real_mm[:, 1],
                real_mm[:, 2],
                "g.-",
                label="Real",
                alpha=0.7,
                markersize=3,
            )
            ax1.plot(
                sim_mm[:, 0],
                sim_mm[:, 1],
                sim_mm[:, 2],
                "k.-",
                label="Sim",
                alpha=0.7,
                markersize=3,
            )
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            ax1.set_zlabel("Z (mm)")
            ax1.legend()
            ax1.set_title("3D Trajectory")

            # XY projection
            ax2 = fig.add_subplot(222)
            ax2.plot(
                real_mm[:, 0],
                real_mm[:, 1],
                "g.-",
                label="Real",
                alpha=0.7,
                markersize=3,
            )
            ax2.plot(
                sim_mm[:, 0], sim_mm[:, 1], "k.-", label="Sim", alpha=0.7, markersize=3
            )
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.legend()
            ax2.set_title("XY Projection")
            ax2.set_aspect("equal")
            ax2.grid(True, alpha=0.3)

            # XZ projection
            ax3 = fig.add_subplot(223)
            ax3.plot(
                real_mm[:, 0],
                real_mm[:, 2],
                "g.-",
                label="Real",
                alpha=0.7,
                markersize=3,
            )
            ax3.plot(
                sim_mm[:, 0], sim_mm[:, 2], "k.-", label="Sim", alpha=0.7, markersize=3
            )
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Z (mm)")
            ax3.legend()
            ax3.set_title("XZ Projection")
            ax3.grid(True, alpha=0.3)

            # Error over samples (standard RMSE, the reported convention)
            ax4 = fig.add_subplot(224)
            ax4.plot(errors, "b-", alpha=0.7, label=f"RMSE={rmse:.1f}mm")
            ax4.set_xlabel("Sample")
            ax4.set_ylabel("Error (mm)")
            ax4.set_title("Position Error")
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)

            fig.suptitle(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plot_path = step_dir / save_name
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {plot_path} (RMSE={rmse:.1f}mm)")
        except Exception as e:
            print(f"  Warning: Trajectory visualization failed: {e}")
            traceback.print_exc()

    def _save_final_results(self):
        """Save final pipeline artifacts."""
        # Copy best model and config to top level
        last_step = max(self.step_results.keys()) if self.step_results else 0

        if last_step >= 2:
            step_dir = self.output_dir / STEP_DIR_NAMES[last_step]

            # Copy best model
            best_model = step_dir / "best_model.xml"
            if best_model.exists():
                shutil.copy2(best_model, self.output_dir / "final_model.xml")

            # Copy generation config
            gen_config = step_dir / "generation_config.json"
            if gen_config.exists():
                shutil.copy2(
                    gen_config, self.output_dir / "final_generation_config.json"
                )

        # Write summary
        summary = {
            "pipeline_name": self.config.get("name", "pipeline"),
            "steps_completed": sorted(self.step_results.keys()),
            "timestamp": datetime.now().isoformat(),
        }

        for step_num, results in sorted(self.step_results.items()):
            step_key = f"step{step_num}"
            if step_num == 1:
                summary[step_key] = {
                    "seg_offsets": results.get("seg_offsets"),
                    "tendon_angle_deltas": results.get("tendon_angle_deltas"),
                }
            elif step_num == 2:
                summary[step_key] = {
                    "best_error_mm": results.get("best_error_mm"),
                    "iterations": results.get("iterations"),
                }
            elif step_num == 3:
                summary[step_key] = {
                    "train_error_mm": results.get("train_error_mm"),
                    "val_error_mm": results.get("val_error_mm"),
                    "iterations": results.get("iterations"),
                }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Output: {self.output_dir}")
        if 3 in self.step_results:
            r = self.step_results[3]
            train_err = r.get("train_error_mm")
            if train_err is None:
                print("Train RMSE: N/A")
            else:
                print(f"Train RMSE: {train_err:.3f} mm")
            print(f"Val RMSE:   {r.get('val_error_mm', 'N/A'):.3f} mm")
        elif 2 in self.step_results:
            r = self.step_results[2]
            print(f"Best RMSE:  {r.get('best_error_mm', 'N/A'):.3f} mm")
        print(f"{'='*60}")
