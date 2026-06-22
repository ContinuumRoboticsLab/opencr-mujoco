"""Joint-space controller for TDCR robots using Clark coordinates.

This controller maps user inputs to TDCR joint (tendon) positions using
the Clark transformation, which provides an intuitive 2D representation
of each segment's bending state.
"""

import numpy as np
from typing import Any, Optional, Dict, Tuple
import mujoco

from opencr_mujoco.tdcr_kinematics import (
    MultiSegmentTDCRKinematics,
    MultiSegmentTDCRIndependentKinematics,
)
from opencr_mujoco.tdcr_kinematics.multi_segment_tdcr_tension import (
    MultiSegmentTDCRTensionKinematics,
)

from .homing import step_toward


class TDCRJointController:
    """Joint-space controller for multi-segment TDCR.

    Maps user inputs (e.g., joystick commands) to tendon positions/tensions using
    Clark coordinates for intuitive control of each segment's bending.

    Supports arbitrary numbers of segments and tendons per segment.
    Defaults to 3 segments with 3 tendons each for backward compatibility.

    Control Modes:
        - Position Mode (default): Coupled segments where upper segment tendons
          accumulate contributions from lower segments. Uses MultiSegmentTDCRKinematics.
        - Tension Mode: Independent segments with no cross-segment coupling.
          Uses MultiSegmentTDCRTensionKinematics for independent force control.
        - Independent Segments Mode: Decoupled position control where each segment's
          tendons only route through that segment. Uses MultiSegmentTDCRIndependentKinematics.

    Attributes:
        model: MuJoCo model
        kinematics: TDCR kinematics object (position, tension, or independent mode)
        current_segment: Currently controlled segment index
        clark_speed_scale: Speed scaling for Clark coordinate changes
        tendon_actuator_ids: Actuator IDs for all tendons
        n_segments: Number of segments
        n_tendons_per_segment: Array of tendon counts per segment
        tension_mode: Whether using tension control mode
        independent_segments: Whether using independent segments mode
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: Optional[mujoco.MjData] = None,
        tendon_distance_mm: float = 4.0,
        angle_offset_rad_ccw: Optional[np.ndarray] = None,
        clark_speed_scale: float = 0.001,
        fps: float = 100.0,
        n_tendons_per_segment: Optional[np.ndarray] = None,
        n_segments: Optional[int] = None,
        tension_mode: bool = False,
        independent_segments: bool = False,
        command_frame_offset_rad: float = 0.0,
        command_mirror_x: bool = False,
    ):
        """Initialize TDCR joint controller.

        Args:
            model: MuJoCo model containing TDCR
            data: MuJoCo data (optional, for reading current state)
            tendon_distance_mm: Distance from backbone to tendons (same for all segments or per-segment)
            angle_offset_rad_ccw: Angular offsets for each segment (optional, auto-generated if None)
            clark_speed_scale: Speed scaling for Clark coordinate changes (per second)
            fps: Control loop frequency for speed scaling
            n_tendons_per_segment: Number of tendons per segment (e.g., [3,3,3] or [3,4,3])
                                  If None, auto-detects from model
            n_segments: Number of segments (optional, for validation)
            tension_mode: If True, use tension control kinematics (independent force control)
                         If False, use position control kinematics
            independent_segments: If True, use independent position control (tendons only in own segment)
                                 Overrides tension_mode if both are True
            command_frame_offset_rad: rotate the incoming x/y command by this angle before it
                         becomes a Clark increment. Use it to align "up/down/left/right" with the
                         viewer when the robot is mounted at an angle (the mount rotates the bend
                         plane in the world; angle_offset_rad_ccw cannot fix this — it only matches
                         tendon assignment to the routing, it does not rotate the bend direction).
            command_mirror_x: if True, negate the left/right (x) command before the frame
                         rotation. A mount can leave the bend plane mirror-flipped relative to the
                         viewer so that left/right reads inverted while up/down is correct; a pure
                         rotation (command_frame_offset_rad) cannot fix that (it would flip both
                         axes). Negating x before the rotation flips ONLY left/right -- the rotation
                         is linear, so it negates exactly the x-command's contribution to both Clark
                         axes and leaves the up/down (y) command untouched.
        """
        self.model = model
        self.data = data
        self.clark_speed_scale = clark_speed_scale
        self.command_frame_offset_rad = command_frame_offset_rad
        self.command_mirror_x = command_mirror_x
        self.fps = fps
        self.dt = 1.0 / fps  # Time step for speed scaling
        self.tension_mode = tension_mode
        self.independent_segments = independent_segments

        # Auto-detect or use provided segment/tendon configuration
        if n_tendons_per_segment is None:
            # Auto-detect from model actuators
            self.n_tendons_per_segment, detected_n_segments = (
                self._detect_configuration()
            )
            self.n_segments = (
                n_segments if n_segments is not None else detected_n_segments
            )
            if self.n_segments != len(self.n_tendons_per_segment):
                raise ValueError(
                    f"n_segments={self.n_segments} doesn't match the "
                    f"{len(self.n_tendons_per_segment)} segments auto-detected "
                    "from the model's seg_X_ten_Y actuators"
                )
        else:
            # Use provided configuration
            self.n_tendons_per_segment = np.array(n_tendons_per_segment)
            self.n_segments = len(self.n_tendons_per_segment)
            if n_segments is not None and n_segments != self.n_segments:
                raise ValueError(
                    f"n_segments={n_segments} doesn't match length of "
                    f"n_tendons_per_segment={len(self.n_tendons_per_segment)}"
                )

        self.current_segment = 0  # Start controlling segment 0

        # Initialize kinematics with appropriate class based on mode
        if angle_offset_rad_ccw is None:
            # Generate default offsets (30-degree increments for segments)
            angle_offset_rad_ccw = np.array(
                [i * np.pi / 6 for i in range(self.n_segments)]
            )

        # Handle scalar tendon distance (apply to all segments)
        if np.isscalar(tendon_distance_mm):
            tendon_distances = np.ones(self.n_segments) * tendon_distance_mm
        else:
            tendon_distances = tendon_distance_mm

        # Select appropriate kinematics class based on mode
        if independent_segments:
            self.kinematics = MultiSegmentTDCRIndependentKinematics(
                n_tendons_per_segment=self.n_tendons_per_segment,
                tendon_distances_mm=tendon_distances,
                angle_offsets_rad_ccw=angle_offset_rad_ccw,
            )
            print(
                "Initialized TDCR controller in INDEPENDENT SEGMENTS mode (decoupled position control)"
            )
        elif tension_mode:
            self.kinematics = MultiSegmentTDCRTensionKinematics(
                n_tendons_per_segment=self.n_tendons_per_segment,
                tendon_distances_mm=tendon_distances,
                angle_offsets_rad_ccw=angle_offset_rad_ccw,
            )
            print(
                "Initialized TDCR controller in TENSION mode (independent force control)"
            )
        else:
            self.kinematics = MultiSegmentTDCRKinematics(
                n_tendons_per_segment=self.n_tendons_per_segment,
                tendon_distances_mm=tendon_distances,
                angle_offsets_rad_ccw=angle_offset_rad_ccw,
            )
            print("Initialized TDCR controller in POSITION mode (coupled segments)")

        # Find tendon actuator IDs
        self._find_tendon_actuators()

        # Initialize from pretension keyframe if available
        self.pretension_lengths = None
        self._initialize_from_pretension()

        # Initialize goal to current position
        self.reset_to_current()

    def _detect_configuration(self) -> Tuple[np.ndarray, int]:
        """Auto-detect TDCR configuration from model actuators.

        Scans actuator names for pattern seg_X_ten_Y to determine number
        of segments and tendons per segment.

        Returns:
            Tuple of (n_tendons_per_segment array, n_segments)
        """
        # Parse all actuator names to find seg_X_ten_Y pattern
        segment_tendons = {}  # seg_idx -> list of tendon indices

        for act_id in range(self.model.nu):
            act_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id
            )
            if act_name and act_name.startswith("seg_") and "_ten_" in act_name:
                try:
                    parts = act_name.split("_")
                    seg_idx = int(parts[1])
                    ten_idx = int(parts[3])

                    if seg_idx not in segment_tendons:
                        segment_tendons[seg_idx] = []
                    segment_tendons[seg_idx].append(ten_idx)
                except (ValueError, IndexError):
                    continue

        if not segment_tendons:
            # No TDCR actuators found, use default 3×3 configuration
            print("No TDCR actuators found, using default 3 segments x 3 tendons")
            return np.array([3, 3, 3]), 3

        # Build n_tendons_per_segment array
        n_segments = max(segment_tendons.keys()) + 1
        n_tendons_per_segment = []

        for seg_idx in range(n_segments):
            if seg_idx in segment_tendons:
                n_tendons = len(segment_tendons[seg_idx])
                n_tendons_per_segment.append(n_tendons)
            else:
                raise ValueError(f"Missing tendons for segment {seg_idx}")

        print(
            f"Auto-detected TDCR configuration: {n_segments} segments, "
            f"tendons per segment: {n_tendons_per_segment}"
        )
        return np.array(n_tendons_per_segment), n_segments

    def _find_tendon_actuators(self):
        """Find actuator IDs for TDCR tendons."""
        self.tendon_actuator_ids = []

        # Look for actuators named seg_X_ten_Y based on detected configuration
        for seg_idx in range(self.n_segments):
            n_tendons = self.n_tendons_per_segment[seg_idx]
            for ten_idx in range(n_tendons):
                actuator_name = f"seg_{seg_idx}_ten_{ten_idx}"
                actuator_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )

                if actuator_id == -1:
                    raise ValueError(f"Actuator '{actuator_name}' not found in model")

                self.tendon_actuator_ids.append(actuator_id)

        print(f"Found {len(self.tendon_actuator_ids)} TDCR tendon actuators")

    def _initialize_from_pretension(self):
        """Initialize from pretension keyframe if available."""
        total_tendons = int(np.sum(self.n_tendons_per_segment))

        # If we have data, check current control values first
        if self.data is not None:
            self.pretension_lengths = np.zeros(total_tendons)
            for j, act_id in enumerate(self.tendon_actuator_ids):
                self.pretension_lengths[j] = self.data.ctrl[act_id]
            if np.any(self.pretension_lengths != 0):
                print(f"Initialized from current control values (pretension applied)")
                print(f"Pretension values: {self.pretension_lengths}")
                return

        # Otherwise look for pretension keyframe
        for i in range(self.model.nkey):
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name == "pretension":
                # Extract pretension values from keyframe
                key_ctrl = self.model.key_ctrl[i]
                self.pretension_lengths = np.zeros(total_tendons)
                for j, act_id in enumerate(self.tendon_actuator_ids):
                    self.pretension_lengths[j] = key_ctrl[act_id]
                print(f"Initialized from pretension keyframe")
                return

        # If no pretension keyframe, use zeros
        self.pretension_lengths = np.zeros(total_tendons)
        print("No pretension keyframe found, using zero pretension")

    def reset_to_current(self):
        """Reset goal to current tendon positions (relative to pretension)."""
        total_tendons = int(np.sum(self.n_tendons_per_segment))

        # If we have data, read actual current values
        if self.data is not None:
            current_tendons = np.zeros(total_tendons)
            for i, act_id in enumerate(self.tendon_actuator_ids):
                current_tendons[i] = self.data.ctrl[act_id]
            # Convert to relative values for kinematics
            if self.pretension_lengths is not None:
                relative_tendons = current_tendons - self.pretension_lengths
            else:
                relative_tendons = current_tendons
            self.kinematics.set_goal_clark_coords_to_current(relative_tendons)
        else:
            # No data and no pretension: start from zero (home)
            self.kinematics.set_goal_clark_coords_to_current(np.zeros(total_tendons))

    def reset_to_home(self):
        """Reset all segments to home position (pretension state)."""
        # Home is zero relative to pretension
        self.kinematics.set_goal_clark_coords_to_home()
        print("Reset to home position")

    def set_current_segment(self, segment_idx: int):
        """Set which segment to control.

        Args:
            segment_idx: Segment index (0 to n_segments-1)
        """
        if segment_idx < 0 or segment_idx >= self.n_segments:
            raise ValueError(
                f"Invalid segment index: {segment_idx}. "
                f"Must be 0 to {self.n_segments - 1}"
            )
        if self.current_segment != segment_idx:
            self.current_segment = segment_idx
            print(f"Now controlling segment {segment_idx + 1}")

    def compute_target_qpos(
        self, command: Dict[str, float], data: Optional[mujoco.MjData] = None
    ) -> np.ndarray:
        """Compute target joint positions from user command.

        Args:
            command: Dictionary with keys:
                - 'x': X-axis command (-1 to 1)
                - 'y': Y-axis command (-1 to 1)
                - 'segment': Which segment to control (0 to n_segments-1)
                - 'reset_home': Reset to home if True
            data: MuJoCo data (optional, for reading current state)

        Returns:
            Array of target positions for all actuators
        """
        # Handle segment switching
        if "segment" in command:
            self.set_current_segment(int(command["segment"]))

        # Build the Clark-coordinate increment for this step.
        if command.get("reset_home", False):
            # Hold-to-home: nudge every segment's goal Clark coords toward zero at
            # the teleop speed (clark_speed_scale * dt per step), synced so they
            # straighten together. Releasing the key stops the motion.
            goal_clark = self.kinematics.goal_clark_coords
            target = step_toward(
                goal_clark, np.zeros_like(goal_clark), self.clark_speed_scale * self.dt
            )
            clark_increment = target - goal_clark
        else:
            clark_increment = np.zeros(2 * self.n_segments)
            if "x" in command and "y" in command:
                seg_idx = self.current_segment
                # Rotate the command into the bend frame so up/down/left/right
                # match the viewer for a mounted (angled) robot.
                cx, cy = command["x"], command["y"]
                if self.command_mirror_x:
                    cx = -cx  # mirror left/right only (before the rotation)
                if self.command_frame_offset_rad:
                    co = np.cos(self.command_frame_offset_rad)
                    si = np.sin(self.command_frame_offset_rad)
                    cx, cy = cx * co - cy * si, cx * si + cy * co
                # Scale by speed and time step for consistent speed across FPS
                clark_increment[seg_idx * 2] = cx * self.clark_speed_scale * self.dt
                clark_increment[seg_idx * 2 + 1] = cy * self.clark_speed_scale * self.dt

        # Update kinematics and get tendon lengths (relative to zero)
        target_tendons = self.kinematics.clark_coords_increment_to_tendon(
            clark_increment
        )

        # Build full control array (may include other actuators)
        target_qpos = np.zeros(self.model.nu)

        # Set tendon actuator targets (add pretension to get absolute values)
        for i, (act_id, tendon_length) in enumerate(
            zip(self.tendon_actuator_ids, target_tendons)
        ):
            # Add pretension to get absolute control value
            if self.pretension_lengths is not None:
                target_qpos[act_id] = self.pretension_lengths[i] + tendon_length
            else:
                target_qpos[act_id] = tendon_length

        return target_qpos

    def get_info(self) -> Dict[str, Any]:
        """Get controller information for display.

        Returns:
            Dictionary with controller state information
        """
        clark_coords = self.kinematics.goal_clark_coords

        # Build clark_coords dict dynamically for all segments
        clark_coords_dict = {}
        clark_magnitudes_dict = {}

        for seg_idx in range(self.n_segments):
            seg_key = f"seg{seg_idx + 1}"
            seg_clark = clark_coords[seg_idx * 2 : (seg_idx + 1) * 2]
            clark_coords_dict[seg_key] = seg_clark
            clark_magnitudes_dict[seg_key] = np.linalg.norm(seg_clark)

        # Determine control mode
        if self.independent_segments:
            control_mode = "Independent Segments"
        elif self.tension_mode:
            control_mode = "Tension"
        else:
            control_mode = "Position (Coupled)"

        return {
            "type": f"TDCR Joint Control ({control_mode} Mode)",
            "current_segment": self.current_segment + 1,
            "n_segments": self.n_segments,
            "n_tendons_per_segment": self.n_tendons_per_segment.tolist(),
            "clark_coords": clark_coords_dict,
            "clark_magnitudes": clark_magnitudes_dict,
        }
