#!/usr/bin/env python3
"""Load an opencr-mujoco scene in MuJoCo.

Use this for quick inspection of generated or hand-authored MJCF files. Missing
generated TDCR scenes are rebuilt automatically from `configs/generation/` when
the scene path follows the standard repository naming convention.

Usage:
    python viewer.py --list-configs
    python viewer.py --show-config
    python viewer.py --scene assets/franka_scene.xml
    python viewer.py --config franka_no_ee
    python viewer.py --headless --duration 2

On macOS, GUI viewer windows should be launched with `mjpython`. Headless
commands and config inspection work with ordinary `python`.
"""

import argparse
import sys

import mujoco
import mujoco.viewer

from opencr_mujoco.utils.config_loader import (
    add_config_args,
    handle_config_args,
    PROJECT_ROOT,
)

# Default configuration
DEFAULT_CONFIG = {
    "scene": "assets/franka_scene.xml",
    "description": "Basic Franka scene with end-effector",
}


def main():
    parser = argparse.ArgumentParser(
        description="Basic MuJoCo viewer for opencr-mujoco scenes"
    )

    # Add config arguments
    add_config_args(parser, "viewer", default_config="default")

    # Add viewer-specific arguments
    parser.add_argument("--scene", "-s", type=str, help="Path to MuJoCo XML scene file")
    parser.add_argument(
        "--headless",
        action="store_const",
        const=True,
        default=None,
        help="Run in headless mode (no viewer window)",
    )
    parser.add_argument(
        "--duration", type=float, help="Duration to run in headless mode (seconds)"
    )

    args = parser.parse_args()

    # Handle config loading
    config = handle_config_args(args, "viewer", DEFAULT_CONFIG)

    # Resolve the scene, generating it from its generation config when the
    # XML is absent (generated scenes are not tracked in git)
    from generate import ensure_scene

    try:
        scene_path = ensure_scene(config["scene"])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Project root: {PROJECT_ROOT}")
        return 1

    # Load model
    print(f"\nLoading scene: {scene_path}")
    if "description" in config:
        print(f"Description: {config['description']}")

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)

        # Apply pretension keyframe if available (for TDCR models)
        for i in range(model.nkey):
            key_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name == "pretension":
                mujoco.mj_resetDataKeyframe(model, data, i)
                print(f"\nApplied pretension keyframe to simulation")
                break

        # Print scene info
        print(f"\nScene information:")
        print(f"  - Bodies: {model.nbody}")
        print(f"  - Joints: {model.njnt}")
        print(f"  - DOFs: {model.nv}")
        print(f"  - Actuators: {model.nu}")
        print(f"  - Timestep: {model.opt.timestep}")

        if args.headless:
            # Headless mode for testing
            duration = args.duration or 1.0
            print(f"\nRunning in headless mode for {duration} seconds...")

            steps = int(duration / model.opt.timestep)
            for _ in range(steps):
                mujoco.mj_step(model, data)

            print("Scene loaded successfully")
        else:
            # Launch viewer
            print("Controls:")
            print("  - Left mouse: Rotate")
            print("  - Right mouse: Pan")
            print("  - Scroll: Zoom")
            print("  - Space: Pause/Resume")
            print("  - Backspace: Reset")
            print("  - Close window to quit")

            mujoco.viewer.launch(model, data)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
