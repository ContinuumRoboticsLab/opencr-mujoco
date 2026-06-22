#!/usr/bin/env python3
"""Run the TDCR system-identification pipeline.

The pipeline calibrates simulation parameters against bundled motion-capture
data. It is organized as three explicit stages:

1. Geometry identification from tendon-pull trajectories.
2. Tendon parameter optimization for pretension, stiffness, and constraints.
3. Refinement on training trajectories with validation on held-out data.

Usage:
    # Discover bundled pipeline configs.
    python sysid_pipeline.py --list-configs

    # Run the full bundled pipeline.
    python sysid_pipeline.py --config ftdcr_v4_pipeline

    # Run or resume a single step.
    python sysid_pipeline.py --config ftdcr_v4_pipeline --step 1
    python sysid_pipeline.py --config ftdcr_v4_pipeline --step 2
    python sysid_pipeline.py --config ftdcr_v4_pipeline --step 3

    # Resume from an existing output directory.
    python sysid_pipeline.py --config ftdcr_v4_pipeline --step 3 \\
        --output-dir sysid_results/ftdcr_v4/ftdcr_v4_pipeline_20260322_143215

    # Validate config without running optimization; use --debug for raw logs.
    python sysid_pipeline.py --config ftdcr_v4_pipeline --dry-run
    python sysid_pipeline.py --config ftdcr_v4_pipeline --debug

Configs can be names resolved from configs/sysid/<name>.json or explicit JSON
paths, including files saved from previous runs.
"""

import argparse
import json
import sys
from pathlib import Path

from opencr_mujoco.utils.config_loader import ConfigLoader, PROJECT_ROOT

CONFIG_TYPE = "sysid"


def resolve_config_path(config_arg: str) -> Path:
    """Resolve --config to a file path.

    A bare name maps to configs/sysid/<name>.json (the generate.py / teleop.py
    convention); anything that already exists or ends in .json is treated as a
    path so explicit config files keep working.
    """
    candidate = Path(config_arg)
    if candidate.suffix == ".json" or candidate.exists():
        return candidate
    return PROJECT_ROOT / "configs" / CONFIG_TYPE / f"{config_arg}.json"


def load_config(config_arg: str) -> dict:
    """Load pipeline configuration by name or path."""
    path = resolve_config_path(config_arg)
    if not path.exists():
        print(f"Error: Config not found: {config_arg}")
        available = ConfigLoader().list_configs(CONFIG_TYPE)
        if available:
            print(f"Available {CONFIG_TYPE} configs: {', '.join(available)}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def list_configs():
    """Print available sysid configs with their descriptions."""
    loader = ConfigLoader()
    names = loader.list_configs(CONFIG_TYPE)
    if not names:
        print(f"No {CONFIG_TYPE} configs found in configs/{CONFIG_TYPE}/")
        return
    print(f"Available {CONFIG_TYPE} configs:")
    for name in names:
        try:
            desc = loader.load_config(CONFIG_TYPE, name).get("description", "")
        except (OSError, json.JSONDecodeError):
            desc = "(unreadable)"
        print(f"  - {name}" + (f": {desc}" if desc else ""))


def validate_config(config: dict) -> bool:
    """Validate pipeline config has required fields."""
    errors = []

    if "base_generation_config" not in config:
        errors.append("Missing 'base_generation_config'")

    data_cfg = config.get("data", {})
    if "data_dir" not in data_cfg:
        errors.append("Missing 'data.data_dir'")
    for key in ["tendon_pull_file", "train_file", "val_file"]:
        if key not in data_cfg:
            errors.append(f"Missing 'data.{key}'")

    # Validate data files exist
    if "data_dir" in data_cfg:
        data_dir = Path(data_cfg["data_dir"])
        for key in ["tendon_pull_file", "train_file", "val_file"]:
            if key in data_cfg:
                fpath = data_dir / data_cfg[key]
                if not fpath.exists():
                    errors.append(f"Data file not found: {fpath}")

    # Validate base generation config
    if "base_generation_config" in config:
        gen_path = Path(config["base_generation_config"])
        if not gen_path.exists():
            errors.append(f"Base generation config not found: {gen_path}")

    if errors:
        print("Config validation errors:")
        for e in errors:
            print(f"  - {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="3-step TDCR system identification pipeline"
    )
    parser.add_argument(
        "--config",
        "-c",
        help=f"Config name to load from configs/{CONFIG_TYPE}/ "
        "(or a path to a pipeline config JSON file)",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help=f"List available {CONFIG_TYPE} configs and exit",
    )
    parser.add_argument(
        "--show-config", action="store_true", help="Print loaded configuration and exit"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        help="Run only a specific step (loads prior results from disk)",
    )
    parser.add_argument(
        "--output-dir", help="Resume from existing output directory (for --step > 1)"
    )
    parser.add_argument(
        "--viewer", action="store_true", help="Enable MuJoCo viewer during simulation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running optimization",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override the per-start function-evaluation budget (maxfev) of "
        "the parallel-multistart optimizer for steps 2 and 3",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip comparison-video rendering (faster; ~half the runtime)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Raw per-iteration optimizer logs from every worker process "
        "(default: compact live progress display)",
    )

    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return

    if not args.config:
        parser.error("--config is required (use --list-configs to see options)")

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Pipeline name: {config.get('name', 'unnamed')}")

    if args.show_config:
        print(json.dumps(config, indent=2))
        return

    # Override the optimizer's per-start evaluation budget if specified
    if args.iterations is not None:
        for step_key in ["step2_tendon_optimization", "step3_refinement"]:
            if step_key in config:
                opt = config[step_key].setdefault("optimization", {})
                opt["maxfev"] = args.iterations
        print(f"Override optimizer maxfev: {args.iterations}")

    if args.no_videos:
        config.setdefault("visualization", {})["render_videos"] = False
        print("Skipping comparison-video rendering (--no-videos)")

    # Validate
    if not validate_config(config):
        sys.exit(1)

    if args.dry_run:
        print("\nDry run: config is valid.")
        data_dir = Path(config["data"]["data_dir"])
        for key in ["tendon_pull_file", "train_file", "val_file"]:
            fpath = data_dir / config["data"][key]
            import pandas as pd

            df = pd.read_csv(fpath)
            print(f"  {key}: {len(df)} rows, columns: {list(df.columns)}")
        return

    # Import here to avoid slow imports for --dry-run
    from opencr_mujoco.sysid.pipeline_orchestrator import PipelineOrchestrator

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config, enable_viewer=args.viewer, verbose=args.debug
    )

    # Override output dir if resuming from existing run
    if args.output_dir:
        orchestrator.output_dir = Path(args.output_dir)

    # Run
    try:
        if args.step:
            orchestrator.run(start_step=args.step, end_step=args.step)
        else:
            orchestrator.run()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        print(f"Partial results may be in: {orchestrator.output_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
