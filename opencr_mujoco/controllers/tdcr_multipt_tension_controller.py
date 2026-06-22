"""Multi-point tension controller for TDCR with variable Franka contribution.

This controller uses tension control instead of position control for TDCR segments,
allowing task-space control at different points along the TDCR.
"""

from typing import Optional, Dict, Tuple, Any

import numpy as np
import mujoco

from .tdcr_ik_controller import TDCRIKController
from .ik_controller import IKController
from .homing import step_toward


class TDCRMultiPointTensionController(TDCRIKController):
    """Multi-point tension controller with variable Franka contribution."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tendon_distance_mm: float = 4.0,
        angle_offset_rad_ccw: Optional[np.ndarray] = None,
        velocity_scale: float = 0.1,
        fps: float = 100.0,
        damping_factor: float = 0.01,
        verbose: bool = False,
        franka_linear_scale: float = 0.1,
        franka_angular_scale: float = 0.5,
        tdcr_linear_scale: float = 1.0,
        tdcr_angular_scale: float = 1.0,
        clark_direct_scale: float = 100.0,
        tension_scale: float = 0.1,
        jacobian_refresh_hz: float = 10.0,
        settle_horizon_s: float = 0.1,
        jacobian_perturbation_mm: float = 1.5,
        velocity_boost: float = 50.0,
        rotation_boost: float = 1000.0,
        jacobian_cols_per_refresh: int = 2,
        contact_settle_horizon_s: float = 0.2,
        insertion_angular_scale: float = 2.0,
        position_weight: float = 1.0,
        orientation_weight: float = 0.1,
    ):
        """Initialize multi-point tension controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            tendon_distance_mm: Distance from backbone to tendons
            angle_offset_rad_ccw: Angular offsets for each segment
            velocity_scale: Scaling factor for velocity commands
            fps: Control loop frequency
            damping_factor: Damping for pseudo-inverse computation
            verbose: Enable verbose output
            franka_linear_scale: Linear velocity scaling for Franka control
            franka_angular_scale: Angular velocity scaling for Franka control
            tdcr_linear_scale: Linear velocity scaling for TDCR control
            tdcr_angular_scale: Angular velocity scaling for TDCR control
            clark_direct_scale: Scaling for direct Clark coordinate control
            tension_scale: Scaling factor for tension control (mm to tension)
        """
        # tension_scale must exist before super().__init__, which calls our
        # _tendons_mm_to_ctrl override (S-shape application and the initial
        # numerical Jacobian).
        self.tension_scale = tension_scale

        # Initialize parent TDCR IK controller with tension_mode=True, forwarding
        # the live-Jacobian knobs so the estimator + cache are configured.
        super().__init__(
            model,
            data,
            tendon_distance_mm,
            angle_offset_rad_ccw,
            velocity_scale,
            fps,
            damping_factor,
            verbose,
            clark_direct_scale=clark_direct_scale,
            tension_mode=True,
            jacobian_refresh_hz=jacobian_refresh_hz,
            settle_horizon_s=settle_horizon_s,
            jacobian_perturbation_mm=jacobian_perturbation_mm,
            velocity_boost=velocity_boost,
            rotation_boost=rotation_boost,
            jacobian_cols_per_refresh=jacobian_cols_per_refresh,
            contact_settle_horizon_s=contact_settle_horizon_s,
        )

        # Store speed scaling parameters
        self.franka_linear_scale = franka_linear_scale
        self.franka_angular_scale = franka_angular_scale
        self.tdcr_linear_scale = tdcr_linear_scale
        self.tdcr_angular_scale = tdcr_angular_scale
        self.clark_direct_scale = clark_direct_scale
        self.insertion_angular_scale = insertion_angular_scale
        self.tension_position_weight = position_weight
        self.tension_orientation_weight = orientation_weight

        # Initialize Franka IK controller
        self.franka_controller = IKController(model, data)

        # Find Franka actuator IDs
        self._find_franka_actuators()

        # Define control points: the segment-end bodies are derived from the
        # model's tendon routing (last routing site of seg_{s}_tendon_0), so
        # this adapts to any link count or chain mode without hardcoding.
        seg_ends = self._derive_segment_end_body_names()
        seg_end_fallbacks = ["link_10", "link_20", "link_30"]
        self.control_points = {
            "base": {
                "body_name": "link_0",  # TDCR base attachment point
                "franka_weight": 1.0,  # 100% Franka
                "tdcr_segments": [],  # No TDCR segments
                "body_id": None,
            },
        }
        for s in range(3):
            body_name = (
                seg_ends[s]
                if s < len(seg_ends) and seg_ends[s]
                else seg_end_fallbacks[s]
            )
            self.control_points[f"seg{s + 1}"] = {
                "body_name": body_name,  # End of segment s+1
                "franka_weight": 0.0,  # 0% Franka, 100% TDCR
                "tdcr_segments": [s],  # Control only this segment
                "body_id": None,
            }

        # Find body IDs for each control point
        self._find_control_point_bodies()

        # Current control point (default to tip)
        self.current_control_point = "seg3"

        # Base-local backbone (insertion) axis, captured at the settled init
        # pose (see _capture_base_forward_axis).
        self.base_forward_local = self._capture_base_forward_axis()

        # Visual indicator for control point
        self.visual_site_id = None
        self._setup_visual_indicator()

        # Enable the base-insertion DOF now that control points + Franka
        # actuators exist; invalidate the init Jacobian so the first control step
        # re-estimates with the insertion column.
        self._has_insertion_dof = True
        self._cached_J_pos = None

        if self.verbose:
            print("\nMulti-Point Tension Controller initialized")
            print(f"  Tension scale: {self.tension_scale}")
            print("Control points:")
            for name, info in self.control_points.items():
                print(
                    f"  {name}: {info['body_name']} "
                    f"(Franka: {info['franka_weight']*100:.0f}%, "
                    f"TDCR: {(1-info['franka_weight'])*100:.0f}%)"
                )

    def _find_franka_actuators(self):
        """Find Franka actuator IDs."""
        self.franka_actuator_ids = []
        for i in range(1, 8):  # Franka has 7 joints
            act_name = f"panda_joint{i}"
            # mj_name2id returns -1 (it does not raise) when absent; skip those.
            act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
            )
            if act_id >= 0:
                self.franka_actuator_ids.append(act_id)

        if self.verbose:
            print(f"Found {len(self.franka_actuator_ids)} Franka actuators")

    def _tendons_mm_to_ctrl(self, tendons_mm: np.ndarray) -> np.ndarray:
        """Tension mode: ctrl values are forces, scaled by tension_scale.

        Overrides the position-mode parent so the inherited numerical Jacobian
        perturbs the actuators with the same mm->ctrl mapping that command
        execution uses. Command execution adds the pretension baseline
        (compute_target_qpos: pretension_lengths[i] + tension), so the Jacobian
        perturbation must add it too or the linearization is taken about a
        different operating point than where commands actually drive.
        (pretension_lengths is None during the parent __init__ that first calls
        this; the guard handles that bootstrap.)
        """
        forces = tendons_mm * self.tension_scale
        if self.pretension_lengths is not None:
            return self.pretension_lengths + forces
        return forces

    def _setup_visual_indicator(self):
        """Setup visual indicator for control point."""
        # mj_name2id returns -1 (it does not raise) when the site is absent.
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "control_point_vis"
        )
        self.visual_site_id = site_id if site_id >= 0 else None
        if self.visual_site_id is None and self.verbose:
            print("Note: No visualization site found.")

    def _capture_base_forward_axis(self) -> np.ndarray:
        """Backbone (insertion) axis expressed in the base body frame."""
        base_id = self.control_points["base"]["body_id"]
        seg1_id = self.control_points["seg1"]["body_id"]
        if base_id is None or seg1_id is None or base_id < 0 or seg1_id < 0:
            return np.array([0.0, 0.0, 1.0])  # generator builds backbones along +Z
        vec = self.data.xpos[seg1_id] - self.data.xpos[base_id]
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, self.data.xquat[base_id])
        return rot_mat.reshape(3, 3).T @ (vec / norm)

    def _find_control_point_bodies(self):
        """Find body IDs for each control point."""
        for name, info in self.control_points.items():
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, info["body_name"]
            )
            if body_id >= 0:
                info["body_id"] = body_id
                if self.verbose:
                    print(
                        f"Found control point '{name}' body: "
                        f"{info['body_name']} (ID: {body_id})"
                    )
            else:
                print(
                    f"Warning: Could not find body '{info['body_name']}' "
                    f"for control point '{name}'"
                )
                info["body_id"] = None

    def set_control_point(self, point_name: str):
        """Set the active control point."""
        if point_name not in self.control_points:
            print(f"Warning: Unknown control point '{point_name}'")
            return

        self.current_control_point = point_name
        info = self.control_points[point_name]

        if self.verbose:
            print(f"\nSwitched to control point: {point_name}")
            print(f"  Body: {info['body_name']}")
            print(f"  Franka: {info['franka_weight']*100:.0f}%")
            print(f"  TDCR: {(1-info['franka_weight'])*100:.0f}%")

    def compute_franka_jacobian(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Franka Jacobian for a given body."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

        # Extract only Franka joint columns (first 7 DOF)
        jacp_franka = jacp[:, :7]
        jacr_franka = jacr[:, :7]

        return jacp_franka, jacr_franka

    def compute_orientation_error(
        self, current_quat: np.ndarray, target_dir: np.ndarray
    ) -> np.ndarray:
        """Compute angular velocity to align the backbone axis with a direction.

        The backbone (forward) axis in the base frame is captured at init from
        the actual scene geometry (see _capture_base_forward_axis), so this is
        independent of whether the chain was built along +X or +Z.

        Args:
            current_quat: Current orientation quaternion [w, x, y, z]
            target_dir: Target direction vector (unit vector)

        Returns:
            Angular velocity vector to align with target
        """
        # Convert quaternion to rotation matrix
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, current_quat)
        rot_mat = rot_mat.reshape(3, 3)

        # Current backbone (forward) axis in world frame
        current_forward = rot_mat @ self.base_forward_local

        # Compute rotation axis (cross product)
        rot_axis = np.cross(current_forward, target_dir)
        rot_axis_norm = np.linalg.norm(rot_axis)

        if rot_axis_norm < 1e-6:
            # Already aligned or opposite direction
            return np.zeros(3)

        # Normalize rotation axis
        rot_axis = rot_axis / rot_axis_norm

        # Compute rotation angle
        cos_angle = np.dot(current_forward, target_dir)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Angular velocity (axis-angle representation)
        # Scale by angle to get proportional control
        return rot_axis * angle

    def _apply_insertion_motion(
        self,
        v_insert_cmd: float,
        control_body_id: int,
        target_qpos: np.ndarray,
        verbose_prefix: str = "",
    ) -> np.ndarray:
        """Apply insertion/extraction motion via Franka control.

        This is the unified insertion logic used by both keyboard and Jacobian control.

        Args:
            v_insert_cmd: Insertion velocity command (positive = insert, negative = extract)
            control_body_id: ID of the control point body
            target_qpos: Target position array to update
            verbose_prefix: Prefix for verbose output (e.g., "via Jacobian")

        Returns:
            Updated target_qpos array
        """
        # Get base body ID
        base_body_id = self.control_points["base"]["body_id"]
        if base_body_id is None or base_body_id < 0:
            if self.verbose:
                print("Warning: Base body not found for insertion control")
            return target_qpos

        # Compute insertion direction (from base to segment 1 endpoint)
        # Always use segment 1 endpoint (link_10), not the control point
        seg1_endpoint_id = self.control_points["seg1"]["body_id"]
        base_pos = self.data.xpos[base_body_id]
        endpoint_pos = self.data.xpos[seg1_endpoint_id]
        insertion_vec = endpoint_pos - base_pos
        insertion_dist = np.linalg.norm(insertion_vec)

        if insertion_dist < 1e-6:
            return target_qpos

        insertion_dir = insertion_vec / insertion_dist

        # Get current base orientation
        base_quat = self.data.xquat[base_body_id]

        # Compute angular velocity based on current orientation error
        # adjust scaling to approx follow the leader motion
        angular_vel_desired = (
            self.compute_orientation_error(base_quat, insertion_dir)
            * self.insertion_angular_scale
        )

        # Flip angular velocity for extraction (moving away from endpoint)
        if v_insert_cmd < 0:
            angular_vel_desired = -angular_vel_desired

        # Create Cartesian velocity for base motion (linear + angular)
        v_insert_val = v_insert_cmd * self.velocity_scale * self.franka_linear_scale
        v_insert_cartesian = np.zeros(6)
        v_insert_cartesian[:3] = insertion_dir * v_insert_val
        v_insert_cartesian[3:] = angular_vel_desired * self.franka_angular_scale

        # Compute Franka Jacobian for base
        j_pos_franka, j_rot_franka = self.compute_franka_jacobian(base_body_id)
        j_franka_full = np.vstack([j_pos_franka, j_rot_franka])

        # Compute joint velocities
        j_pinv = np.linalg.pinv(j_franka_full, rcond=self.damping_factor)
        joint_velocities = j_pinv @ v_insert_cartesian
        joint_increments = joint_velocities * self.dt

        # Update Franka joints
        current_joints = np.zeros(7)
        for i, act_id in enumerate(self.franka_actuator_ids):
            current_joints[i] = self.data.ctrl[act_id]

        new_joints = current_joints + joint_increments

        for i, act_id in enumerate(self.franka_actuator_ids):
            target_qpos[act_id] = new_joints[i]

        if self.verbose and abs(v_insert_val) > 1e-4:
            mode_str = "INSERT" if v_insert_cmd > 0 else "EXTRACT"
            print(f"\n{mode_str}{verbose_prefix}: {v_insert_val:.4f}")
            print(f"  Direction: {insertion_dir}")
            print(f"  Distance to endpoint: {insertion_dist:.4f}")
            print(f"  Angular velocity: {angular_vel_desired}")

        return target_qpos

    def compute_target_qpos(
        self, command: Dict[str, float], data: Optional[mujoco.MjData] = None
    ) -> np.ndarray:
        """Compute target joint positions/tensions from Cartesian velocity.

        For TDCR: Uses tension control (direct force application)
        For Franka: Uses position control (standard IK)
        """
        # Handle control point switching
        if "control_point" in command:
            self.set_control_point(command["control_point"])

        # Get current control point info
        cp_info = self.control_points[self.current_control_point]
        franka_weight = cp_info["franka_weight"]
        tdcr_segments = cp_info["tdcr_segments"]

        # Hold-to-home (independent): reset_franka (R) steps the Franka joints
        # toward home; reset_tdcr (Y) straightens the TDCR (goal Clark -> 0 and
        # the matching tendon tensions are applied, so Y actually straightens in
        # tension mode). Separate keys in teleop, matching franka_tdcr_combined;
        # legacy reset_home does both. Reset dominates other motion this frame.
        reset_home = command.get("reset_home", False)
        do_franka = command.get("reset_franka", False) or reset_home
        do_tdcr = command.get("reset_tdcr", False) or reset_home
        if do_franka or do_tdcr:
            target_qpos = (
                self.data.ctrl.copy() if self.data else np.zeros(self.model.nu)
            )
            if do_franka:
                cur_f = np.array([self.data.ctrl[a] for a in self.franka_actuator_ids])
                new_f = step_toward(
                    cur_f,
                    np.asarray(self.franka_controller.home_position),
                    self.home_joint_step,
                )
                for i, act_id in enumerate(self.franka_actuator_ids):
                    target_qpos[act_id] = new_f[i]
            if do_tdcr:
                gc = self.kinematics.goal_clark_coords
                new_clark = step_toward(gc, np.zeros_like(gc), self.home_clark_step)
                self.kinematics.goal_clark_coords = new_clark
                target_tensions = (
                    self.kinematics.clark_to_tendons_mm(new_clark) * self.tension_scale
                )
                for i, act_id in enumerate(self.tendon_actuator_ids):
                    target_qpos[act_id] = (
                        self.pretension_lengths[i]
                        if self.pretension_lengths is not None
                        else 0.0
                    ) + target_tensions[i]
            self._note_command_active(False)  # reset is not Cartesian motion
            return target_qpos

        # Build Cartesian velocity vector
        if franka_weight > 0:
            linear_scale = self.franka_linear_scale
            angular_scale = self.franka_angular_scale
        else:
            linear_scale = self.tdcr_linear_scale
            angular_scale = self.tdcr_angular_scale

        v_cartesian = np.zeros(6)
        v_cartesian[0] = command.get("vx", 0.0) * self.velocity_scale * linear_scale
        v_cartesian[1] = command.get("vy", 0.0) * self.velocity_scale * linear_scale
        v_cartesian[2] = command.get("vz", 0.0) * self.velocity_scale * linear_scale
        v_cartesian[3] = command.get("wx", 0.0) * self.velocity_scale * angular_scale
        v_cartesian[4] = command.get("wy", 0.0) * self.velocity_scale * angular_scale
        v_cartesian[5] = command.get("wz", 0.0) * self.velocity_scale * angular_scale

        # Whether Cartesian (Jacobian-driven) motion is commanded this frame —
        # the signal for command-onset Jacobian refresh, recorded on every exit.
        cartesian_active = np.linalg.norm(v_cartesian) > 1e-6

        # Get control point body ID
        control_body_id = cp_info["body_id"]
        if control_body_id is None or control_body_id < 0:
            print(f"Error: No valid body ID for '{self.current_control_point}'")
            self._note_command_active(cartesian_active)
            return self.data.ctrl.copy() if self.data else np.zeros(self.model.nu)

        # Initialize target array
        target_qpos = self.data.ctrl.copy() if self.data else np.zeros(self.model.nu)

        # Compute based on control point
        if franka_weight == 1.0:
            # Pure Franka control (base point) - position control
            j_pos_franka, j_rot_franka = self.compute_franka_jacobian(control_body_id)
            j_franka_full = np.vstack([j_pos_franka, j_rot_franka])

            # Compute joint velocities using pseudo-inverse
            j_pinv = np.linalg.pinv(j_franka_full, rcond=self.damping_factor)
            joint_velocities = j_pinv @ v_cartesian
            joint_increments = joint_velocities * self.dt

            # Update Franka joints
            current_joints = np.zeros(7)
            for i, act_id in enumerate(self.franka_actuator_ids):
                current_joints[i] = self.data.ctrl[act_id]

            new_joints = current_joints + joint_increments

            for i, act_id in enumerate(self.franka_actuator_ids):
                target_qpos[act_id] = new_joints[i]

        else:
            # Pure TDCR control - TENSION MODE
            has_clark = (
                abs(command.get("clark_x", 0.0)) > 0.01
                or abs(command.get("clark_y", 0.0)) > 0.01
            )
            has_cartesian = cartesian_active
            has_insertion = abs(command.get("v_insert", 0.0)) > 1e-6

            # Handle insertion/extraction control (move base towards/away from endpoint)
            if has_insertion:
                v_insert_cmd = command.get("v_insert", 0.0)
                # Use unified insertion logic
                target_qpos = self._apply_insertion_motion(
                    v_insert_cmd,
                    control_body_id,
                    target_qpos,
                    verbose_prefix=" (keyboard)",
                )

            if has_clark and not has_cartesian:
                # Direct Clark coordinate control
                clark_velocities = np.zeros(6)
                if len(tdcr_segments) > 0:
                    seg_idx = tdcr_segments[0]
                    clark_velocities[seg_idx * 2] = (
                        command.get("clark_x", 0.0) * self.clark_direct_scale
                    )
                    clark_velocities[seg_idx * 2 + 1] = (
                        command.get("clark_y", 0.0) * self.clark_direct_scale
                    )

                clark_increment = clark_velocities * self.dt
                new_clark = self.kinematics.goal_clark_coords + clark_increment

            else:
                # Jacobian-based task-space control about this control point: the
                # IK solves the active Clark bending synergies plus the base
                # insertion DOF (now a per-mm column consistent with the Clark
                # columns). Restore tip_body_id even if the estimate raises.
                saved_tip_id = self.tip_body_id
                self.tip_body_id = control_body_id
                try:
                    self._maybe_refresh_jacobian(has_cartesian)
                finally:
                    self.tip_body_id = saved_tip_id

                # Position-biased weights (orientation relaxed); both exposed.
                clark_velocities, insertion_vel = self._solve_tdcr_ik(
                    v_cartesian,
                    tdcr_segments,
                    self.tension_position_weight,
                    self.tension_orientation_weight,
                )
                new_clark = (
                    self.kinematics.goal_clark_coords + clark_velocities * self.dt
                )

                # Auto-insertion (base translation) only while commanding
                # task-space motion AND not manually inserting, so it never
                # clobbers the Franka targets set by manual Y/N insertion.
                if has_cartesian and not has_insertion:
                    target_qpos = self._apply_ik_insertion(
                        insertion_vel * self.dt, target_qpos
                    )

            self.kinematics.goal_clark_coords = new_clark

            # Convert to tendon lengths
            target_tendons_mm = self.kinematics.clark_to_tendons_mm(new_clark)

            # TENSION MODE: Scale by tension_scale instead of 0.001
            target_tensions = target_tendons_mm * self.tension_scale

            # Set TDCR tension targets
            for i, (act_id, tension) in enumerate(
                zip(self.tendon_actuator_ids, target_tensions)
            ):
                if self.pretension_lengths is not None:
                    target_qpos[act_id] = self.pretension_lengths[i] + tension
                else:
                    target_qpos[act_id] = tension

        # Verbose output
        if self.verbose:
            has_motion = (
                np.linalg.norm(v_cartesian) > 1e-6
                or abs(command.get("clark_x", 0.0)) > 0.01
                or abs(command.get("clark_y", 0.0)) > 0.01
            )
            if has_motion:
                print(
                    f"\n--- Multi-Point Tension Control "
                    f"({self.current_control_point}) ---"
                )
                print(f"Franka weight: {franka_weight*100:.0f}%")
                if np.linalg.norm(v_cartesian) > 1e-6:
                    print(f"Cartesian velocity: {v_cartesian[:3]}")
                if (
                    abs(command.get("clark_x", 0.0)) > 0.01
                    or abs(command.get("clark_y", 0.0)) > 0.01
                ):
                    print(
                        f"Clark control: X={command.get('clark_x', 0.0):.1f}"
                        f", Y={command.get('clark_y', 0.0):.1f}"
                    )
                    print(f"Active segments: {tdcr_segments}")
                if control_body_id is not None and control_body_id >= 0:
                    body_pos = self.data.xpos[control_body_id]
                    print(f"Control point position: {body_pos}")

        # Record this frame's Cartesian-active state for next-frame onset detection.
        self._note_command_active(cartesian_active)
        return target_qpos

    def get_control_point_position(self) -> np.ndarray:
        """Get the current control point position for visualization."""
        cp_info = self.control_points[self.current_control_point]
        if (
            cp_info["body_id"] is not None
            and cp_info["body_id"] >= 0
            and self.data is not None
        ):
            return self.data.xpos[cp_info["body_id"]].copy()
        return np.zeros(3)

    def get_info(self) -> Dict[str, Any]:
        """Get controller information for display."""
        base_info = super().get_info()

        # Add multi-point specific info
        cp_info = self.control_points[self.current_control_point]
        base_info["control_point"] = {
            "name": self.current_control_point,
            "body": cp_info["body_name"],
            "franka_weight": cp_info["franka_weight"],
            "tdcr_weight": 1.0 - cp_info["franka_weight"],
            "position": self.get_control_point_position(),
            "mode": "tension",
        }

        return base_info
