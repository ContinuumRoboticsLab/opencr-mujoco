#!/usr/bin/env python3
"""Teleoperation interface for opencr-mujoco.

Drive a Franka arm and/or a TDCR with a keyboard or DualSense controller,
selected via configuration files in configs/teleop/.

Usage:
    # The GUI uses MuJoCo's passive viewer, so run it under mjpython on macOS
    # (plain `python` is fine for --list-configs and --headless).
    python teleop.py --list-configs                    # list available configs
    mjpython teleop.py                                 # default: franka_tdcr_combined

    # Combined Franka + TDCR (default)
    mjpython teleop.py --config franka_tdcr_combined     # combined arm + continuum control
    mjpython teleop.py --config ftdcr_taskspace_position # multi-point task-space (position)
    mjpython teleop.py --config ftdcr_taskspace_tension  # multi-point task-space (tension/force)

    # Franka arm only
    mjpython teleop.py --config franka_keyboard_ik       # Cartesian (IK) control, keyboard

    # TDCR only
    mjpython teleop.py --config tdcr_keyboard            # coupled TDCR, joint control
    mjpython teleop.py --config tdcr_keyboard_modular    # heterogeneous modular TDCR
    mjpython teleop.py --config tdcr_keyboard_modular_tension  # modular TDCR, tension (force) control

    # Overrides
    python teleop.py --config tdcr_keyboard --headless --duration 2   # run without a viewer window

    # Mirror to a real TDCR over Dynamixel (requires the ".[hardware]" extra)
    python teleop.py --config tdcr_keyboard --with-real --device-name /dev/ttyUSB0

Keyboard controls (hold LSHIFT for all teleop keys):

    Keys                Franka modes              TDCR modes
    ----                ------------              ----------
    W/A/S/D             end-effector X/Y          control-point X/Y (task-space modes)
    Q/E                 end-effector up/down      control-point up/down (task-space modes)
    I/K, J/L, U/O       roll / pitch / yaw        control-point rotation (task-space modes)
    T/G, F/H            -                         bend current segment up/down, left/right
    Z/X/C/V/B           -                         select segment 1-5 (combined mode: Z/X/C only)
    N/M                 gripper close/open        insert/extract (multi-point mode)
    H                   reset Franka to home      -
    R                   -                         reset TDCR to home (combined: R=Franka, Y=TDCR)

DualSense controls:

    Left stick          X/Y position              D-pad up/down: Z
    Right stick         orientation (task-space) / wrist (joint mode)
    D-pad left/right    gripper close/open
    PS button (hold)    return to home
    Gyroscope           tip orientation when enable_gyroscope is true
"""

import argparse
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from opencr_mujoco.utils.config_loader import add_config_args, handle_config_args, PROJECT_ROOT
from opencr_mujoco.controllers.ik_controller import read_franka_home_from_model

# Default configuration
DEFAULT_CONFIG = {
    "scene": "assets/franka_scene.xml",
    "input_device": "dualsense",
    "control_mode": "task_space",
    "controller": "ik",
    "width": 1200,
    "height": 900,
    "show_info": True,
    "sim_steps_per_frame": 1,
    "enable_gyroscope": False,
    "fps": 100,
    "description": "DualSense controller with IK control",
}


def get_input_device(device_type: str):
    """Get input device instance based on type."""
    if device_type == "dualsense":
        from opencr_mujoco.utils.dualsense_input_device import DualSenseInputDevice

        try:
            return DualSenseInputDevice()
        except Exception as e:
            raise RuntimeError(
                "Could not connect to a DualSense controller "
                f"({e}). Plug one in via USB/Bluetooth, or use a keyboard "
                "config instead (e.g. --config franka_keyboard_ik)."
            ) from e
    elif device_type == "keyboard":
        from opencr_mujoco.utils.keyboard_input_device import KeyboardInputDevice

        return KeyboardInputDevice()
    elif device_type == "tdcr_keyboard":
        from opencr_mujoco.controllers.tdcr_keyboard_input_mapper import TDCRKeyboardInputMapper

        return TDCRKeyboardInputMapper()
    elif device_type == "combined_keyboard":
        from opencr_mujoco.controllers.combined_keyboard_input_mapper import (
            CombinedKeyboardInputMapper,
        )

        return CombinedKeyboardInputMapper()
    elif device_type == "multipt_keyboard":
        from opencr_mujoco.controllers.multipt_taskspace_keyboard_mapper import (
            MultiPointTaskSpaceKeyboardMapper,
        )

        return MultiPointTaskSpaceKeyboardMapper()
    else:
        raise ValueError(f"Unknown input device: {device_type}")


def get_scene_info(model, scene_path):
    """Extract scene information from the loaded model.

    TDCR/gripper presence is detected from the MODEL, not the scene path:
    development checkout names can contain "tdcr", so a path-based check can
    match every scene -- which previously made even a plain Franka scene look
    like a 10-segment TDCR (num_robot_joints=30) and crash reset_to_home.
    """
    actuator = mujoco.mjtObj.mjOBJ_ACTUATOR

    def has_actuator(name):
        # mj_name2id returns -1 (it does not raise) when the actuator is absent.
        return mujoco.mj_name2id(model, actuator, name) >= 0

    info = {
        "num_bodies": model.nbody,
        "num_joints": model.njnt,
        "num_actuators": model.nu,
        "has_gripper": False,
        "num_robot_joints": 7,  # Default for Franka
        "is_tdcr": False,
    }

    # Count TDCR segments by their seg_X_ten_0 actuators.
    num_segments = 0
    while has_actuator(f"seg_{num_segments}_ten_0"):
        num_segments += 1

    if num_segments > 0:
        info["is_tdcr"] = True
        info["num_tdcr_segments"] = num_segments
        info["num_robot_joints"] = num_segments * 3  # 3 tendons per segment
        # Combined = a Franka arm is present alongside the TDCR.
        info["is_combined"] = has_actuator("panda_joint1") or has_actuator(
            "panda0_joint1"
        )
    else:
        # Franka(-like) scene: detect a gripper from the actuator names.
        act_names = [
            mujoco.mj_id2name(model, actuator, i) or "" for i in range(model.nu)
        ]
        info["has_gripper"] = any(
            ("finger" in n) or ("gripper" in n) for n in act_names
        )

    return info


def apply_initial_state(model, data):
    """Reset the sim to the 'pretension' (home) keyframe if the scene defines one.

    Single source of truth for the initial robot state -- replaces the per-branch
    keyframe-application blocks. Returns True if a keyframe was applied.
    """
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("Applied 'pretension' (home) keyframe")
        return True
    return False


def is_tdcr_single_command(controller_type):
    """TDCR controllers driven by a single command dict (not the combined arm)."""
    return controller_type in [
        "tdcr_joint",
        "tdcr_ik",
        "tdcr_multipt",
        "tdcr_multipt_tension",
    ]


def apply_control_frame(
    model, data, joint_controller, input_mapper, config, dynamixel_bridge=None
):
    """Read inputs, compute targets, and write data.ctrl for ONE frame.

    Does not step physics -- the caller (headless loop or passive viewer) steps.
    This is the single control path shared by every controller type, so all
    teleop runs identically under launch_passive."""
    controller_type = config.get("controller")
    if controller_type == "combined":
        # Combined Franka + TDCR (two command dicts).
        franka_cmd, tdcr_cmd = input_mapper.get_command()
        joint_targets = joint_controller.compute_target_qpos(franka_cmd, tdcr_cmd)
        assert len(joint_targets) == model.nu, (
            f"controller returned {len(joint_targets)} ctrl values, "
            f"expected model.nu={model.nu}"
        )
        data.ctrl[:] = joint_targets
    elif is_tdcr_single_command(controller_type):
        command = input_mapper.get_command()
        joint_targets = joint_controller.compute_target_qpos(command, data)
        assert len(joint_targets) == model.nu, (
            f"controller returned {len(joint_targets)} ctrl values, "
            f"expected model.nu={model.nu}"
        )
        data.ctrl[:] = joint_targets

        # Mirror to the real robot if a bridge is connected.
        if dynamixel_bridge and dynamixel_bridge.enabled:
            dynamixel_bridge.send_sim_tendons(
                joint_targets[joint_controller.tendon_actuator_ids]
            )
    else:
        # Standard Franka (joint / IK) via dualsense or keyboard.
        dualsense = getattr(input_mapper, "dualsense", None)
        if hasattr(dualsense, "update_state"):
            dualsense.update_state()
        elif hasattr(dualsense, "update"):
            dualsense.update()
        joint_targets = input_mapper.read_controller_inputs()
        joint_controller.set_joint_targets(joint_targets)


def main():
    """Main function for teleoperation interface."""
    parser = argparse.ArgumentParser(
        description="Teleoperation interface for opencr-mujoco"
    )

    # Add config arguments. No-arg `python teleop.py` loads this config.
    add_config_args(parser, "teleop", default_config="franka_tdcr_combined")

    # Add teleop-specific arguments
    parser.add_argument("--scene", "-s", type=str, help="Path to MuJoCo XML scene file")

    parser.add_argument(
        "--input-device",
        "-i",
        type=str,
        choices=[
            "dualsense",
            "keyboard",
            "tdcr_keyboard",
            "combined_keyboard",
            "multipt_keyboard",
        ],
        help="Input device type",
    )

    parser.add_argument(
        "--control-mode",
        "-m",
        type=str,
        choices=["joint_space", "task_space"],
        help="Control mode",
    )

    parser.add_argument(
        "--controller",
        type=str,
        choices=[
            "joint",
            "ik",
            "tdcr_joint",
            "tdcr_ik",
            "combined",
            "tdcr_multipt",
            "tdcr_multipt_tension",
        ],
        help="Controller type",
    )

    parser.add_argument(
        "--enable-gyroscope",
        action="store_const",
        const=True,
        default=None,
        help="Enable gyroscope control (DualSense only)",
    )

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

    parser.add_argument(
        "--sim-steps",
        type=int,
        dest="sim_steps_per_frame",
        help="Simulation steps per frame",
    )

    parser.add_argument("--fps", type=int, help="Control loop frequency in Hz")

    parser.add_argument(
        "--with-real",
        action="store_const",
        const=True,
        default=None,
        help="Mirror simulation to real robot via Dynamixel bridge",
    )

    parser.add_argument(
        "--device-name",
        type=str,
        default=None,
        help="Serial device name for Dynamixel connection " "(default: /dev/ttyUSB0)",
    )

    args = parser.parse_args()

    # Handle config loading
    config = handle_config_args(args, "teleop", DEFAULT_CONFIG)

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

        # Get scene information
        scene_info = get_scene_info(model, scene_path)

        print(f"\nScene information:")
        print(f"  - Bodies: {scene_info['num_bodies']}")
        print(f"  - Joints: {scene_info['num_joints']}")
        print(f"  - Actuators: {scene_info['num_actuators']}")
        print(f"  - Has gripper: {scene_info['has_gripper']}")
        if scene_info.get("is_tdcr", False):
            print(f"  - TDCR segments: {scene_info.get('num_tdcr_segments', 0)}")

        # Apply the home/'pretension' keyframe once (single source of truth for the
        # initial robot state) before creating any controller.
        keyframe_applied = apply_initial_state(model, data)

        # Initialize input device
        print(f"\nInitializing {config['input_device']} input device...")
        input_device = get_input_device(config["input_device"])

        # Fill missing geometric TDCR params (tendon distance + per-segment
        # angle offsets) from the scene's generation config. These are facts
        # of the asset, not teleop preferences, so deriving them keeps them
        # from silently drifting when the scene is rerouted to a different
        # robot; an explicit controller_params value still overrides.
        from generate import tdcr_geometry_from_scene

        tdcr_geom = tdcr_geometry_from_scene(scene_path)
        if tdcr_geom:
            cp = config.setdefault("controller_params", {})
            # the combined controller nests TDCR params under "tdcr"; the
            # standalone TDCR controllers read them at the top level
            target = (
                cp.setdefault("tdcr", {})
                if config.get("controller") == "combined"
                else cp
            )
            for key, value in tdcr_geom.items():
                if target.get(key) is None:  # explicit config value wins
                    target[key] = value
            print(f"TDCR geometry from generation config: {tdcr_geom}")

        # Create appropriate controller based on config
        joint_controller = None
        input_mapper = None

        if config.get("controller") == "combined":
            # Combined Franka + TDCR control
            from opencr_mujoco.controllers.combined_controller import CombinedController

            # Get controller parameters from config
            controller_params = config.get("controller_params", {})
            tdcr_params = controller_params.get("tdcr", {})
            tdcr_params["fps"] = config.get("fps", 100)

            joint_controller = CombinedController(model, data, tdcr_params)

            # Input device is the mapper
            input_mapper = input_device
            if hasattr(input_device, "start"):
                input_device.start()
        elif config.get("controller") == "tdcr_joint":
            # TDCR joint control
            from opencr_mujoco.controllers.tdcr_joint_controller import TDCRJointController

            # Get controller parameters from config
            controller_params = config.get("controller_params", {})

            joint_controller = TDCRJointController(
                model,
                data=data,
                n_tendons_per_segment=controller_params.get("n_tendons_per_segment"),
                n_segments=controller_params.get("n_segments"),
                tendon_distance_mm=controller_params.get("tendon_distance_mm", 4.0),
                angle_offset_rad_ccw=controller_params.get("angle_offset_rad_ccw"),
                clark_speed_scale=controller_params.get("clark_speed_scale", 0.001),
                fps=config.get("fps", 100),
                tension_mode=controller_params.get("tension_mode", False),
                independent_segments=controller_params.get(
                    "independent_segments", False
                ),
                command_frame_offset_rad=controller_params.get(
                    "command_frame_offset_rad", 0.0
                ),
                command_mirror_x=controller_params.get("command_mirror_x", False),
            )

            # For TDCR, the input device IS the mapper
            input_mapper = input_device
            if hasattr(input_device, "start"):
                input_device.start()

            # Print TDCR info
            print(
                f"Found {len(joint_controller.tendon_actuator_ids)} TDCR tendon actuators"
            )
        elif config.get("controller") == "tdcr_ik":
            # TDCR task-space control with IK
            from opencr_mujoco.controllers.tdcr_ik_controller import TDCRIKController

            # Disable gravity if specified
            if config.get("disable_gravity", False):
                model.opt.gravity[:] = 0
                print("Gravity disabled")

            # Get controller parameters from config
            controller_params = config.get("controller_params", {})
            joint_controller = TDCRIKController(
                model,
                data,
                tendon_distance_mm=controller_params.get("tendon_distance_mm", 4.0),
                angle_offset_rad_ccw=controller_params.get("angle_offset_rad_ccw"),
                velocity_scale=config.get("velocity_scale", 0.5),
                damping_factor=config.get("damping_factor", 0.01),
                fps=config.get("fps", 100),
                verbose=config.get("verbose", False),
            )

            # For TDCR IK, use task-space keyboard mapper
            from opencr_mujoco.controllers.tdcr_taskspace_keyboard_mapper import (
                TDCRTaskSpaceKeyboardMapper,
            )

            input_mapper = TDCRTaskSpaceKeyboardMapper(
                velocity_scale=1.0, verbose=config.get("verbose", False)
            )

            # Print TDCR info
            print(f"TDCR Task-Space Control initialized")
        elif config.get("controller") == "tdcr_multipt":
            # Multi-point TDCR task-space control
            from opencr_mujoco.controllers.tdcr_multipt_taskspace_controller import (
                TDCRMultiPointTaskSpaceController,
            )

            # Disable gravity if specified
            if config.get("disable_gravity", False):
                model.opt.gravity[:] = 0
                print("Gravity disabled")

            # Get controller parameters from config
            controller_params = config.get("controller_params", {})
            joint_controller = TDCRMultiPointTaskSpaceController(
                model,
                data,
                tendon_distance_mm=controller_params.get("tendon_distance_mm", 4.0),
                angle_offset_rad_ccw=controller_params.get("angle_offset_rad_ccw"),
                velocity_scale=config.get("velocity_scale", 0.5),
                damping_factor=config.get("damping_factor", 0.01),
                fps=config.get("fps", 100),
                verbose=config.get("verbose", False),
                franka_linear_scale=controller_params.get("franka_linear_scale", 0.1),
                franka_angular_scale=controller_params.get("franka_angular_scale", 0.5),
                tdcr_linear_scale=controller_params.get("tdcr_linear_scale", 0.1),
                tdcr_angular_scale=controller_params.get("tdcr_angular_scale", 2.0),
                clark_direct_scale=controller_params.get("clark_direct_scale", 15.0),
                jacobian_refresh_hz=controller_params.get("jacobian_refresh_hz", 10.0),
                settle_horizon_s=controller_params.get("settle_horizon_s", 0.1),
                jacobian_perturbation_mm=controller_params.get(
                    "jacobian_perturbation_mm", 1.5
                ),
                jacobian_cols_per_refresh=controller_params.get(
                    "jacobian_cols_per_refresh", 2
                ),
                contact_settle_horizon_s=controller_params.get(
                    "contact_settle_horizon_s"
                ),
            )

            # Use multi-point keyboard mapper
            input_mapper = input_device

            # Print controller info
            print(f"Multi-Point TDCR Task-Space Control initialized")
        elif config.get("controller") == "tdcr_multipt_tension":
            # Multi-point TDCR tension control
            from opencr_mujoco.controllers.tdcr_multipt_tension_controller import (
                TDCRMultiPointTensionController,
            )

            # Disable gravity if specified
            if config.get("disable_gravity", False):
                model.opt.gravity[:] = 0
                print("Gravity disabled")

            # Get controller parameters from config
            controller_params = config.get("controller_params", {})
            joint_controller = TDCRMultiPointTensionController(
                model,
                data,
                tendon_distance_mm=controller_params.get("tendon_distance_mm", 4.0),
                angle_offset_rad_ccw=controller_params.get("angle_offset_rad_ccw"),
                velocity_scale=config.get("velocity_scale", 0.5),
                damping_factor=config.get("damping_factor", 0.01),
                fps=config.get("fps", 100),
                verbose=config.get("verbose", False),
                franka_linear_scale=controller_params.get("franka_linear_scale", 0.1),
                franka_angular_scale=controller_params.get("franka_angular_scale", 0.5),
                tdcr_linear_scale=controller_params.get("tdcr_linear_scale", 1.0),
                tdcr_angular_scale=controller_params.get("tdcr_angular_scale", 1.0),
                clark_direct_scale=controller_params.get("clark_direct_scale", 100.0),
                tension_scale=controller_params.get("tension_scale", 0.1),
                jacobian_refresh_hz=controller_params.get("jacobian_refresh_hz", 10.0),
                settle_horizon_s=controller_params.get("settle_horizon_s", 0.1),
                jacobian_perturbation_mm=controller_params.get(
                    "jacobian_perturbation_mm", 1.5
                ),
                jacobian_cols_per_refresh=controller_params.get(
                    "jacobian_cols_per_refresh", 2
                ),
                # Tension is compliant -> settle longer under contact for an
                # accurate contact Jacobian (free-space stays at settle_horizon_s).
                contact_settle_horizon_s=controller_params.get(
                    "contact_settle_horizon_s", 0.2
                ),
            )

            # Use multi-point keyboard mapper
            input_mapper = input_device

            # Print controller info
            print(f"Multi-Point TDCR Tension Control initialized")
        else:
            # Standard Franka control
            from opencr_mujoco.controllers.joint_controller import JointController

            joint_controller = JointController(
                model,
                data,
                num_joints=scene_info["num_robot_joints"],
                has_gripper=scene_info["has_gripper"],
            )

        # Initialize IK controller if needed (for Franka only, not TDCR)
        ik_controller = None
        if (
            config.get("controller") == "ik"
            or config.get("control_mode") == "task_space"
        ) and config.get("controller") not in ["tdcr_ik", "tdcr_joint"]:
            from opencr_mujoco.controllers.ik_controller import IKController

            print(f"Initializing IK controller...")
            # Check which end-effector reference to use
            ik_controller = IKController(model, data)

        # Initialize based on controller type
        if config.get("controller") not in [
            "tdcr_joint",
            "tdcr_ik",
            "tdcr_multipt",
            "tdcr_multipt_tension",
            "combined",
        ]:
            # The home keyframe (applied above) defines the start pose. Only fall
            # back to reset_to_home for a Franka scene with no keyframe.
            if not keyframe_applied:
                joint_controller.reset_to_home()
            # Step simulation to settle
            for _ in range(200):
                mujoco.mj_step(model, data)

        if config.get("controller") == "combined":
            # For combined control, Franka position is already set by pretension keyframe
            # Don't reset to home as it would override the pretension frame positions
            # Just step simulation to settle
            for _ in range(200):
                mujoco.mj_step(model, data)
        elif config.get("controller") not in [
            "tdcr_joint",
            "tdcr_ik",
            "tdcr_multipt",
            "tdcr_multipt_tension",
        ]:
            # Initialize input mapper
            print(f"Setting up input mapping...")
            from opencr_mujoco.controllers.dualsense_input_mapper import InputMapper

            # Determine control mode. With a DualSense in task-space control,
            # enable_gyroscope selects the gyro-augmented task-pose mode.
            use_gyro = config["input_device"] == "dualsense" and config.get(
                "enable_gyroscope", False
            )
            if (
                config.get("controller") == "ik"
                or config.get("control_mode") == "task_space"
            ):
                control_mode = (
                    InputMapper.GYROSCOPE_TASK_POSE
                    if use_gyro
                    else InputMapper.TASK_SPACE_POSE
                )
            else:
                control_mode = InputMapper.DIRECT_JOINT_CONTROL

            # Create input mapper
            input_mapper = InputMapper(
                dualsense_device=input_device,
                control_mode=control_mode,
                num_joints=scene_info["num_robot_joints"],
                has_gripper=scene_info["has_gripper"],
                ik_controller=ik_controller,
                fps=config.get("fps", 100),
            )

            # Reset target (keyboard "H" / DualSense PS) = the scene's home pose.
            input_mapper.set_home_positions(read_franka_home_from_model(model))

            if use_gyro:
                print("  - Gyroscope: enabled (task-space orientation control)")

        # Print control info
        print(f"\nControl Configuration:")
        print(f"  - Input: {config['input_device']}")
        if "control_mode" in config:
            print(f"  - Mode: {config['control_mode']}")
        print(f"  - Controller: {config.get('controller', 'joint')}")

        # Device-specific instructions
        if config["input_device"] == "dualsense":
            print("\nDualSense Controls:")
            print("  - Left Stick: X/Y position")
            print("  - Right Stick: Z position / Joint control")
            if config.get("enable_gyroscope"):
                print("  - Gyroscope: Orientation control")
            print("  - L2/R2: Gripper close/open")
            print("  - PS Button: Reset to home")
        elif config["input_device"] == "keyboard":
            print("\nKeyboard Controls (hold LSHIFT):")
            print("  - W/A/S/D: X/Y position")
            print("  - Q/E: Z position")
            print("  - I/J/K/L: Roll/pitch rotation")
            print("  - U/O: Yaw rotation")
            if scene_info["has_gripper"]:
                print("  - N/M: Gripper close/open")
            print("  - H: Home position")
        elif config["input_device"] == "tdcr_keyboard":
            print("\nTDCR Keyboard Controls (hold LSHIFT):")
            print("  - T/F/G/H: Control current segment (up/left/down/right)")
            print("  - Z/X/C/V/B: Select segments 1-5")
            print("  - R: Reset to home position")
        elif config["input_device"] == "combined_keyboard":
            # Combined control instructions are printed by the mapper itself
            pass

        # Initialize Dynamixel bridge if requested
        dynamixel_bridge = None
        if args.with_real and scene_info.get("is_tdcr", False):
            print("\nInitializing Dynamixel bridge for real robot control...")
            from opencr_mujoco.dynamixel_bridge.dynamixel_bridge import DynamixelBridge

            bridge_config = {
                "device_name": args.device_name or "/dev/ttyUSB0",
                # Add other config options here if needed based on robot model
            }

            dynamixel_bridge = DynamixelBridge(bridge_config)
            if dynamixel_bridge.connect():
                print("Real robot bridge connected successfully")

                # Pass pretension lengths once at initialization
                if config.get("controller") in [
                    "tdcr_joint",
                    "tdcr_ik",
                ] and hasattr(joint_controller, "pretension_lengths"):
                    # Store pretension in bridge (only needs to be done once)
                    dynamixel_bridge.pretension_lengths = (
                        joint_controller.pretension_lengths.copy()
                    )
                    # Send initial position with pretension
                    success = dynamixel_bridge.send_sim_tendons(
                        data.ctrl[joint_controller.tendon_actuator_ids],
                        joint_controller.pretension_lengths,
                    )
                    if success:
                        print("Initialized bridge with pretension lengths")
                    else:
                        print(
                            "ERROR: Failed to initialize real robot with pretension, exiting..."
                        )
                        dynamixel_bridge.disconnect()
                        return 1
            else:
                print("ERROR: Failed to connect to real robot, exiting...")
                return 1

        print("\nLaunching teleoperation interface...")
        print("Close window to quit")

        # Control runs in the main thread every frame (see apply_control_frame),
        # so every controller uses the same launch_passive loop below -- no
        # managed launch(), no background control thread. The standard-Franka
        # mapper needs the current joint positions seeded once up front.
        controller_type = config.get("controller")
        if controller_type != "combined" and not is_tdcr_single_command(
            controller_type
        ):
            input_mapper.set_current_positions(joint_controller.get_current_positions())

        if args.headless:
            # Headless smoke mode: the same per-frame control + stepping as the
            # GUI loop, just without a viewer window -- so plain `python` works
            # (no mjpython needed) for CI / quick config checks.
            duration = args.duration or 1.0
            print(f"\nRunning in headless mode for {duration} seconds...")
            sim_timestep = model.opt.timestep
            steps_per_frame = config.get("sim_steps_per_frame", 8)
            end_time = time.time() + duration
            try:
                while time.time() < end_time:
                    step_start = time.time()
                    apply_control_frame(
                        model,
                        data,
                        joint_controller,
                        input_mapper,
                        config,
                        dynamixel_bridge,
                    )
                    for _ in range(steps_per_frame):
                        mujoco.mj_step(model, data)
                    leftover = steps_per_frame * sim_timestep - (
                        time.time() - step_start
                    )
                    if leftover > 0:
                        time.sleep(leftover)
            except KeyboardInterrupt:
                print("\nTeleoperation interrupted by user")
        else:
            # GUI: ONE passive-viewer loop for every controller (consistent;
            # requires mjpython on macOS). The main thread computes control,
            # applies ctrl, steps physics, and syncs -- no managed launch(), no
            # background control thread.
            render_fps = config.get("render_fps", config.get("fps", 60))
            render_period = 1.0 / render_fps
            sim_timestep = model.opt.timestep  # 0.002 for a 500Hz sim
            steps_per_frame = config.get("sim_steps_per_frame", 8)
            is_multipt = config.get("controller") in (
                "tdcr_multipt",
                "tdcr_multipt_tension",
            )
            is_tdcr = is_tdcr_single_command(config.get("controller"))

            # The (expensive) Jacobian re-estimate is spread across control steps
            # inside the multi-point controllers, so the per-frame solve is cheap
            # and the loop never stalls on one big re-estimate.
            cols = getattr(joint_controller, "jacobian_cols_per_refresh", None)
            msg = (
                f"\nPassive viewer: Simulation {1/sim_timestep:.0f}Hz, "
                f"Viewer {render_fps}Hz, Steps/frame {steps_per_frame}"
            )
            if cols is not None:
                msg += f", Jacobian cols/step {cols}"
            print(msg)
            print("(launch_passive requires running under mjpython on macOS)")

            try:
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    step_count = 0
                    while viewer.is_running():
                        current_time = time.time()

                        # Compute + apply control for this frame (all controllers).
                        apply_control_frame(
                            model,
                            data,
                            joint_controller,
                            input_mapper,
                            config,
                            dynamixel_bridge,
                        )

                        # Step physics (fixed steps/frame for consistent timing).
                        for _ in range(steps_per_frame):
                            mujoco.mj_step(model, data)
                            step_count += 1

                        # Periodic debug for the TDCR controllers. tip_body_id
                        # exists only on the IK-based controllers (tdcr_ik /
                        # multipt), not TDCRJointController, so guard for it.
                        if (
                            is_tdcr
                            and step_count % 500 == 0
                            and hasattr(joint_controller, "kinematics")
                        ):
                            clark = joint_controller.kinematics.goal_clark_coords
                            msg = (
                                f"Step {step_count}: "
                                f"Clark={np.array2string(clark, precision=2)}"
                            )
                            tip_id = getattr(joint_controller, "tip_body_id", -1)
                            if tip_id is not None and tip_id >= 0:
                                msg += (
                                    f", Tip="
                                    f"{np.array2string(data.xpos[tip_id], precision=3)}"
                                )
                            print(msg)

                        # Control-point sphere for the multi-point controllers.
                        if (
                            is_multipt
                            and hasattr(joint_controller, "get_control_point_position")
                            and hasattr(viewer, "user_scn")
                            and hasattr(viewer.user_scn, "ngeom")
                        ):
                            cp_pos = joint_controller.get_control_point_position()
                            viewer.user_scn.ngeom = 0
                            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                                g.size[:] = [0.01, 0.01, 0.01]  # 1cm radius
                                g.pos[:] = cp_pos
                                g.mat[:, :] = np.eye(3)
                                g.rgba[:] = [0, 1, 0, 0.3]  # green, 30% opacity
                                viewer.user_scn.ngeom = 1

                        viewer.sync()
                        elapsed = time.time() - current_time
                        if elapsed < render_period:
                            time.sleep(render_period - elapsed)
            except KeyboardInterrupt:
                print("\nTeleoperation interrupted by user")

        # Cleanup
        print("\nShutting down...")

        # Disconnect from real robot if connected
        if dynamixel_bridge:
            dynamixel_bridge.disconnect()

        if hasattr(input_device, "close"):
            input_device.close()
        elif hasattr(input_device, "stop"):
            input_device.stop()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
