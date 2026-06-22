#!/usr/bin/env python3
"""Run SoroSim reference evaluations for opencr-mujoco.

This script drives the shared evaluation engine used by the paper-result
helpers. It generates MuJoCo models from the evaluation config, runs statics or
tip-release dynamics against bundled SoroSim reference data, stores CSV/pickle
outputs, and optionally creates summary plots.

Usage:
    python paper_results/evaluate.py --list-configs
    python paper_results/evaluate.py --show-config
    python paper_results/evaluate.py --config spring_steel_statics --n-values 50
    python paper_results/evaluate.py --config spring_steel_statics \\
        --n-values 25 50 100 --early-stop 20 --no-visualize

SoroSim configs carry an explicit frame_conversion.file_to_mujoco matrix. Keep
that convention intact when adding new reference data; otherwise reference and
simulation shapes are compared in different frames.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

# This script lives in paper_results/; add the repo root to sys.path so the
# `src` package imports resolve regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencr_mujoco.utils.config_loader import (  # noqa: E402
    add_config_args,
    handle_config_args,
    PROJECT_ROOT,
)
from opencr_mujoco.evaluation import (  # noqa: E402
    TrajectoryEvaluator,
    validate_frame_conversion,
)
from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config  # noqa: E402


def _parse_frame_conversion(config: dict) -> Optional[np.ndarray]:
    """Pull frame_conversion.file_to_mujoco out of the config as a 3x3 matrix.

    Returns None if the field is absent. Raises if the field is malformed.
    """
    fc = config.get("frame_conversion")
    if fc is None:
        return None
    matrix = fc.get("file_to_mujoco") if isinstance(fc, dict) else None
    if matrix is None:
        raise ValueError(
            "frame_conversion must be a dict with a 'file_to_mujoco' 3x3 matrix"
        )
    return validate_frame_conversion(matrix)


def _is_sorosim_format(config: dict) -> bool:
    """True when the eval reads SoroSim reference data (CSV statics or the
    13-column dynamics files), which requires a frame_conversion to compare
    reference and simulation in the same frame."""
    test_type = str(config.get("test_type", ""))
    reference_dir = str(config.get("reference_dir", ""))
    return (
        "MuJoCo" in test_type
        or "sorosim" in test_type.lower()
        or "sorosim" in reference_dir.lower()
    )


# Default configuration — a SoroSim spring-steel static evaluation. Used only as
# the bare-invocation fallback; the paper runs always pass --config (see
# configs/evaluation/). Mirrors spring_steel_statics.json.
DEFAULT_CONFIG = {
    "test_type": "SpringSteelRodMuJoCo",
    "n_values": [25, 35, 50, 70, 100],
    "sim_time": 10.0,
    "sim_timestep": 0.01,
    "force_ramp_time": 5.0,
    "early_stop": None,
    "output_dir": "paper_results/evaluation_results/sorosim_statics",
    "reference_dir": "data/reference/sorosim",
    "plot_dir": "plots/sorosim_statics",
    "frame_conversion": {"file_to_mujoco": [[0, 0, 1], [0, -1, 0], [1, 0, 0]]},
    "generator_config": {
        "joints_per_link": 3,
        "joint_config_mode": "material",
        "material_properties": {
            "density": 7870,
            "youngs_modulus": 200e9,
            "poisson_ratio": 0.33,
            "inner_radius": 0,
            "outer_radius": 0.0008,
            "damping_ratio": 0.1,
        },
        "actuation_mode": "none",
        "gravity": "0 -9.81 0",
        "plane": False,
        "total_length": 0.6,
        "radius": 0.0008,
        "disable_contact": True,
    },
    "visualize": True,
    "save_positions": True,
    "description": "SoroSim spring-steel static evaluation (default)",
}


def create_model_generator(generator_config: Dict[str, Any]):
    """Create a model generator function with the given configuration.

    Args:
        generator_config: Configuration for the unified generator

    Returns:
        Function that generates models for a given number of links
    """

    def model_generator(n: int) -> str:
        # Create a copy of the config and update with current N
        config = generator_config.copy()
        config["total_links"] = n

        # Ensure num_segments is set if not provided
        if "num_segments" not in config:
            config["num_segments"] = 1

        # Handle links_per_segment for single segment case
        if config["num_segments"] == 1 and "links_per_segment" not in config:
            config["links_per_segment"] = {"1": n}
            config["segment_lengths"] = {"1": config.get("total_length", 0.6)}

        # Generate the model
        output_path = f"tdcr_n{n}.xml"
        create_tdcr_from_config(config, output_path)
        return output_path

    return model_generator


def run_evaluation(config: dict):  # noqa: C901
    """Run evaluation based on configuration."""
    print("\nEvaluation Configuration:")
    print(f"  - Test type: {config['test_type']}")
    print(f"  - N values: {config['n_values']}")

    # Tip-release (dynamics, time series) when the reference dir is the SoroSim
    # dynamics bank; otherwise a static-equilibrium evaluation.
    reference_dir_str = config["reference_dir"]
    is_tip_release = "dynamics" in reference_dir_str.lower()

    if is_tip_release:
        print(f"  - Sim Hz: {config.get('sim_hz', 500.0)} Hz")
        print(f"  - Force ramp time: {config.get('force_ramp_time', 1.0)} seconds")
        print(f"  - Hold time: {config.get('hold_time', 30.0)} seconds")
        if config.get("debug_visualize", False):
            print("  - Debug visualization: ENABLED (realtime playback)")
    else:
        print(f"  - Sim time: {config.get('sim_time', 'not specified')} seconds")
        if config.get("early_stop"):
            print(f"  - Early stop: {config['early_stop']} tests")
    print(f"  - Output: {config['output_dir']}")

    # Print generator configuration
    gen_config = config.get("generator_config", {})
    print("\nGenerator Configuration:")
    print(f"  - Joints per link: {gen_config.get('joints_per_link', 3)}")
    print(f"  - Joint config mode: {gen_config.get('joint_config_mode', 'material')}")
    print(f"  - Actuation mode: {gen_config.get('actuation_mode', 'direct_torque')}")
    if gen_config.get("joint_config_mode") == "direct":
        print(f"  - Stiffness: {gen_config.get('stiffness', 'not specified')}")
        print(f"  - Damping: {gen_config.get('damping', 'not specified')}")

    # Resolve paths relative to project root if they're not absolute
    reference_dir = Path(config["reference_dir"])
    if not reference_dir.is_absolute():
        reference_dir = PROJECT_ROOT / reference_dir

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # Initialize evaluator
    frame_conversion = _parse_frame_conversion(config)
    if frame_conversion is None and _is_sorosim_format(config):
        raise ValueError(
            "This looks like a SoroSim evaluation "
            f"(test_type={config.get('test_type')!r}, "
            f"reference_dir={config.get('reference_dir')!r}) but the config has "
            "no 'frame_conversion.file_to_mujoco' matrix. Without it, reference "
            "data stays in file frame while the simulation is in MuJoCo frame "
            "and the errors are meaningless. Add a frame_conversion block "
            "(the live configs ship [[0,0,1],[0,-1,0],[1,0,0]])."
        )
    if frame_conversion is not None:
        print(f"\nFrame conversion (file → MuJoCo):\n{frame_conversion}")
    integrator = config.get("integrator")
    if integrator is not None:
        print(f"Integrator override: {integrator}")
    evaluator = TrajectoryEvaluator(
        str(reference_dir),
        str(output_dir),
        frame_conversion=frame_conversion,
        integrator=integrator,
    )

    # Create model generator with unified configuration
    model_generator = create_model_generator(config.get("generator_config", {}))

    # Get config name for output
    config_name = config.get("_config_name", "unified_evaluation")

    # Run parameter sweep
    print(f"\nRunning parameter sweep for config '{config_name}'...")

    if is_tip_release:
        # Run tip release evaluation
        print("\nRunning tip release dynamic test...")
        eval_kwargs = {
            "sim_hz": config.get("sim_hz", 500.0),
            "show_progress": True,
            "force_ramp_time": config.get("force_ramp_time", 1.0),
            "hold_time": config.get("hold_time", 30.0),
            "visualize": config.get("debug_visualize", False),
        }
        csv_path, positions_path = evaluator.run_tip_release_sweep(
            model_generator=model_generator,
            n_values=config["n_values"],
            test_type=config["test_type"],
            config_name=config_name,
            **eval_kwargs,
        )
    else:
        # Build kwargs for static evaluation
        eval_kwargs = {"early_stop": config.get("early_stop"), "show_progress": True}

        # Add sim_time
        if "sim_time" in config:
            eval_kwargs["sim_time"] = config["sim_time"]

        # Add timestep override if specified
        if config.get("sim_timestep"):
            eval_kwargs["sim_timestep"] = config["sim_timestep"]

        # Add force ramp time if specified
        if "force_ramp_time" in config:
            eval_kwargs["force_ramp_time"] = config["force_ramp_time"]

        csv_path, positions_path = evaluator.run_parameter_sweep(
            model_generator=model_generator,
            test_types=[config["test_type"]],
            n_values=config["n_values"],
            config_name=config_name,
            **eval_kwargs,
        )

    print("\nResults saved to:")
    print(f"  - CSV: {csv_path}")
    if positions_path:
        print(f"  - Positions: {positions_path}")

    # Visualize if requested
    if config["visualize"]:
        print("\nGenerating visualizations...")

        # Check if this is a tip release test
        if is_tip_release:
            # Tip release visualization is handled within the evaluator
            print(f"Tip release plots saved to: {evaluator.session_dir / 'plots'}")
        else:
            from opencr_mujoco.evaluation import PaperVisualizer

            # Use session-specific plot directory if available
            plot_dir = (
                evaluator.session_dir / "plots"
                if evaluator.session_dir
                else config["plot_dir"]
            )
            visualizer = PaperVisualizer(
                config["output_dir"],
                str(plot_dir),
                frame_conversion=frame_conversion,
            )

            # Error vs N plot
            visualizer.plot_error_vs_n(csv_path, config_name)

            # Shape comparisons (use all N values)
            if positions_path:
                visualizer.plot_random_shapes_comparison(
                    csv_path,
                    positions_path,
                    str(reference_dir),
                    config_name,
                    n_values=config["n_values"],
                )

            # Runtime performance
            visualizer.plot_runtime_performance(csv_path, config_name)

            # Error histograms
            visualizer.plot_error_histograms(csv_path, config_name, config["n_values"])

            # Worst-case shape plots
            if positions_path:
                print("\nGenerating worst-case shape plots...")
                visualizer.plot_highest_error_shapes(
                    csv_path,
                    positions_path,
                    str(reference_dir),
                    config_name,
                    n_values=config["n_values"],
                    num_cases=10,  # Plot top 10 worst cases for each N
                )

            print(f"\nPlots saved to: {plot_dir}")

    # Clean up generated XML files
    print("\nCleaning up generated XML files...")
    for n in config["n_values"]:
        xml_path = Path(f"tdcr_n{n}.xml")
        if xml_path.exists():
            xml_path.unlink()


def main():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Evaluation interface for TDCR system with unified generator"
    )

    # Add config arguments. No default config name: a bare invocation falls
    # back to the in-module DEFAULT_CONFIG (SoroSim spring-steel statics).
    add_config_args(parser, "evaluation")

    # Add evaluation-specific arguments
    parser.add_argument("--test-type", "-t", type=str, help="Override test type")
    parser.add_argument(
        "--n-values", "-n", nargs="+", type=int, help="Override N values to test"
    )
    parser.add_argument(
        "--sim-time", type=float, help="Override simulation time (seconds)"
    )
    parser.add_argument(
        "--sim-hz",
        type=float,
        help="Override simulation rate for tip release tests (Hz)",
    )
    parser.add_argument("--early-stop", type=int, help="Override early stop count")
    parser.add_argument(
        "--hold-time",
        type=float,
        help="Override hold time for tip release tests (seconds)",
    )
    parser.add_argument(
        "--force-ramp-time",
        type=float,
        help="Override force/gravity ramp time for tip release tests (seconds)",
    )
    parser.add_argument(
        "--integrator",
        choices=["euler", "rk4", "implicit", "implicitfast"],
        help=(
            "Override the MuJoCo integrator. Use 'implicit' for low-damping "
            "stiff systems (e.g. TPU dynamics) where Euler diverges."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, help="Override output directory"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_const",
        const=True,
        default=None,
        help="Skip visualization plots",
    )
    parser.add_argument(
        "--visualize",
        action="store_const",
        const=True,
        default=None,
        help="Generate post-sweep summary plots",
    )
    parser.add_argument(
        "--debug-viewer",
        action="store_const",
        const=True,
        default=None,
        help="Launch the MuJoCo viewer during evaluation (realtime debugging)",
    )

    # Generator-specific overrides
    parser.add_argument(
        "--joints-per-link",
        type=int,
        choices=[2, 3],
        help="Override joints per link (2 or 3)",
    )
    parser.add_argument(
        "--joint-mode",
        choices=["direct", "material"],
        help="Override joint configuration mode",
    )
    parser.add_argument(
        "--actuation",
        choices=["direct_torque", "parallel_tendons"],
        help="Override actuation mode",
    )

    args = parser.parse_args()

    # Load configuration
    config = handle_config_args(args, "evaluation", DEFAULT_CONFIG)

    # Apply overrides
    if args.test_type:
        config["test_type"] = args.test_type
    if args.n_values:
        config["n_values"] = args.n_values
    if args.sim_time is not None:
        config["sim_time"] = args.sim_time
    if args.sim_hz is not None:
        config["sim_hz"] = args.sim_hz
    if args.early_stop is not None:
        config["early_stop"] = args.early_stop
    if args.hold_time is not None:
        config["hold_time"] = args.hold_time
    if args.force_ramp_time is not None:
        config["force_ramp_time"] = args.force_ramp_time
    if args.integrator is not None:
        config["integrator"] = args.integrator
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.no_visualize:
        config["visualize"] = False
    if args.debug_viewer:
        config["debug_visualize"] = True

    # Apply generator overrides
    if "generator_config" not in config:
        config["generator_config"] = {}

    if args.joints_per_link:
        config["generator_config"]["joints_per_link"] = args.joints_per_link
    if args.joint_mode:
        config["generator_config"]["joint_config_mode"] = args.joint_mode
    if args.actuation:
        config["generator_config"]["actuation_mode"] = args.actuation

    # Run evaluation
    run_evaluation(config)


if __name__ == "__main__":
    main()
