"""Trajectory simulator for TDCR system identification."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import mujoco

try:
    import mujoco.viewer

    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False


class TrajectorySimulator:
    """Simulates TDCR trajectories for given actuator commands."""

    def __init__(
        self,
        config: Dict[str, Any],
        enable_viewer: bool = False,
        slack_scaling: float = 1.0,
        verbose: bool = False,
    ):
        """Initialize trajectory simulator.

        Args:
            config: Simulation configuration
            enable_viewer: If True, launch passive MuJoCo viewer for visualization
            slack_scaling: Tendon slack scaling factor (1.0 = no scaling)
            verbose: If True, print informational messages on every model load
                (the optimizer loads a fresh model per iteration, so these are
                gated off by default; warnings always print)
        """
        self.config = config
        self.verbose = verbose
        self.model = None
        self.data = None
        self.tip_site_id = None
        self.tip_body_id = None
        self.actuator_ids = []
        self.marker_transform = self._parse_marker_transform()
        self.enable_viewer = enable_viewer and VIEWER_AVAILABLE
        self.viewer = None
        self.pretension_keyframe_idx = 0  # Default to keyframe 0
        self.slack_scaling = slack_scaling  # Tendon slack scaling factor

        if self.enable_viewer and not VIEWER_AVAILABLE:
            print("Warning: MuJoCo viewer not available, running without visualization")

    def _parse_marker_transform(self) -> np.ndarray:
        """Parse marker transformation matrix from config.

        Returns:
            4x4 transformation matrix from tip to marker
        """
        transform_config = self.config.get("marker_transform", {})

        # Get translation
        translation = np.array(transform_config.get("translation", [0, 0, 0]))

        # Get rotation (as 3x3 matrix or euler angles)
        if "rotation" in transform_config:
            rotation_data = transform_config["rotation"]
            if isinstance(rotation_data, list) and len(rotation_data) == 3:
                # Check if it's euler angles or rotation matrix rows
                if isinstance(rotation_data[0], (float, int)):
                    # Euler angles (in radians)
                    from scipy.spatial.transform import Rotation

                    rotation = Rotation.from_euler("xyz", rotation_data).as_matrix()
                else:
                    # Rotation matrix rows
                    rotation = np.array(rotation_data)
            else:
                rotation = np.eye(3)
        else:
            rotation = np.eye(3)

        # Build 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        return transform

    def _get_simulation_to_data_transform(self) -> np.ndarray:
        """Get coordinate frame transformation from simulation to real data.

        Both frames now use +Z as the backbone direction and XY as the
        perpendicular plane, so no rotation is needed.

        Returns:
            3x3 identity matrix (no rotation)
        """
        return np.eye(3)

    def load_model(self, xml_path: str):
        """Load MuJoCo model from XML file.

        Args:
            xml_path: Path to MJCF XML file
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Launch viewer if enabled
        if self.enable_viewer:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Find the tip tracking point. mj_name2id returns -1 when the name is
        # missing (it does NOT raise), so check explicitly: previously the
        # try/except always "succeeded" with -1 and site_xpos[-1] silently
        # tracked the LAST site in the model — a tendon-routing site that sits
        # off the backbone axis and rotates with the tip link.
        self.tip_site_id = None
        for name in ["tip", "force_site_tip", "end_effector", "ee", "tcp"]:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                self.tip_site_id = site_id
                break

        # Fallback: EE_pos body (backbone tip center), then last body.
        self.tip_body_id = None
        if self.tip_site_id is None:
            ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
            if ee_id >= 0:
                self.tip_body_id = ee_id
            else:
                self.tip_body_id = self.model.nbody - 1
                print(
                    "Warning: no tip site or EE_pos body found; tracking the "
                    "last body in the model"
                )

        # Find actuator IDs for TDCR tendons in seg-major order. Any hardware
        # servo ordering is handled upstream by PipelineDataLoader's
        # servo_mapping during preprocessing.
        self.actuator_ids = []
        seg = 0
        while True:
            seg_ids = []
            ten = 0
            while True:
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"seg_{seg}_ten_{ten}"
                )
                if act_id < 0:
                    break
                seg_ids.append(act_id)
                ten += 1
            if not seg_ids:
                break
            self.actuator_ids.extend(seg_ids)
            seg += 1
        if not self.actuator_ids:
            print("Warning: no seg_X_ten_Y tendon actuators found in model")

        if self.verbose:
            print(f"Loaded model with {len(self.actuator_ids)} actuators")

        # Get pretension baseline from keyframe
        self.pretension_baseline = self._get_pretension_baseline()
        if self.verbose and self.pretension_baseline is not None:
            print(
                f"Using pretension baseline: {self.pretension_baseline[:3]}... (first 3 values)"
            )

    def _get_pretension_baseline(self) -> Optional[np.ndarray]:
        """Get pretension control values from keyframe.

        Returns:
            Tuple of (pretension_values, keyframe_index) or (None, None) if not found
        """
        if self.model is None:
            return None

        # Look for pretension keyframe
        for i in range(self.model.nkey):
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name == "pretension":
                # Extract control values from keyframe
                pretension = np.zeros(len(self.actuator_ids))
                for j, act_id in enumerate(self.actuator_ids):
                    pretension[j] = self.model.key_ctrl[i][act_id]
                self.pretension_keyframe_idx = i
                return pretension

        return None

    def simulate_trajectory(
        self,
        actuator_commands: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        settling_time: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate trajectory with given actuator commands.

        Args:
            actuator_commands: Array of relative actuator commands from pretension (N x num_actuators)
            timestamps: Optional timestamps for commands
            settling_time: Time to wait for settling at each command

        Returns:
            Tuple of (tip_positions, marker_positions) both as N x 3 arrays
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        num_samples = len(actuator_commands)
        tip_positions = np.zeros((num_samples, 3))
        marker_positions = np.zeros((num_samples, 3))

        # Determine simulation timestep
        sim_dt = self.model.opt.timestep
        settle_steps = int(settling_time / sim_dt)

        # Reset to pretension state
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.pretension_keyframe_idx)

        # Forward dynamics to initialize the state properly
        mujoco.mj_forward(self.model, self.data)

        # Tendon friction parameters (hysteresis model)
        friction_const = np.array(
            self.config.get("_sysid_friction_const", [0.0] * len(self.actuator_ids))
        )
        friction_linear = np.array(
            self.config.get("_sysid_friction_linear", [0.0] * len(self.actuator_ids))
        )
        use_friction = np.any(friction_const > 0) or np.any(friction_linear > 0)
        prev_cmd = np.zeros(len(self.actuator_ids))

        # Simulate each commanded position
        for i, cmd in enumerate(actuator_commands):
            # Apply slack scaling to commands
            scaled_cmd = cmd * self.slack_scaling

            # Set actuator targets (add pretension baseline to relative commands)
            # Note: Flip sign because positive servo motion means pulling (shortening tendon)
            # but in MuJoCo, decreasing control value pulls the tendon
            for j, act_id in enumerate(self.actuator_ids):
                if j < len(scaled_cmd):
                    effective_cmd = scaled_cmd[j]

                    # Apply tendon friction (hysteresis)
                    if use_friction:
                        tension = abs(scaled_cmd[j])
                        friction = max(friction_const[j], friction_linear[j] * tension)
                        delta = scaled_cmd[j] - prev_cmd[j]
                        direction = -np.tanh(1000.0 * delta)
                        effective_cmd = scaled_cmd[j] + direction * friction
                        prev_cmd[j] = scaled_cmd[j]

                    if self.pretension_baseline is not None:
                        absolute_cmd = self.pretension_baseline[j] - effective_cmd
                    else:
                        absolute_cmd = -effective_cmd

                    self.data.ctrl[act_id] = absolute_cmd

            # Simulate until settled with stability check
            for step in range(settle_steps):
                mujoco.mj_step(self.model, self.data)

                # Update viewer if enabled (only every 10 steps for performance)
                if self.viewer is not None and step % 10 == 0:
                    self.viewer.sync()

                # Check for instability
                if step % 100 == 0:  # Check every 100 steps
                    if (
                        not np.isfinite(self.data.qacc).all()
                        or not np.isfinite(self.data.qvel).all()
                    ):
                        print(
                            f"Warning: Simulation unstable at sample {i}, step {step}"
                        )
                        print(f"  Relative cmd: {cmd[:3]}... (first 3)")
                        print(
                            f"  Absolute ctrl: {self.data.ctrl[self.actuator_ids[:3]]}... (first 3)"
                        )
                        # Return what we have so far with NaN for remaining
                        tip_positions[i:] = np.nan
                        marker_positions[i:] = np.nan
                        return tip_positions, marker_positions

            # Get tip position
            if self.tip_site_id is not None:
                tip_pos_sim = self.data.site_xpos[self.tip_site_id].copy()
            else:
                # EE_pos body (or last body) fallback resolved in load_model
                tip_pos_sim = self.data.xpos[self.tip_body_id].copy()

            # Transform from simulation frame to data frame
            sim_to_data_rot = self._get_simulation_to_data_transform()
            tip_pos = sim_to_data_rot @ tip_pos_sim

            tip_positions[i] = tip_pos

            # Apply marker transformation (in data frame)
            tip_homogeneous = np.append(tip_pos, 1.0)
            marker_homogeneous = self.marker_transform @ tip_homogeneous
            marker_positions[i] = marker_homogeneous[:3]

        return tip_positions, marker_positions

    def reset(self):
        """Reset simulation to initial state."""
        if self.model is not None and self.data is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def cleanup(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.model = None
        self.data = None
        self.tip_site_id = None
        self.tip_body_id = None
        self.actuator_ids = []
