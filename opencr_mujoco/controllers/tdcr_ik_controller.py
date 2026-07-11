"""Task-space controller for TDCR using Jacobian-based inverse kinematics.

This controller computes the tip Jacobian matrix with respect to Clark coordinates
and uses it for velocity-based teleoperation control.
"""

import numpy as np
import mujoco
from typing import Any, Optional, Dict, Tuple

from opencr_mujoco.tdcr_kinematics import ThreeTendonThreeSegmentTDCRKinematics
from opencr_mujoco.tdcr_kinematics.multi_segment_tdcr_tension import (
    MultiSegmentTDCRTensionKinematics,
)

from .homing import step_toward


def build_synergy_matrix(n_segments: int) -> np.ndarray:
    """Synergy basis over the Clark DOF (order [s0X, s0Y, s1X, s1Y, ...]): per
    bending axis, the n-1 adjacent opposing-bend differences (e_i - e_{i+1}) plus
    the distal segment's absolute bend; 2*n_segments modes, square and invertible.

    Solving the tip DLS in this basis (damping the synergy activations, not the
    raw per-segment bends) gives every tip direction comparable authority. The raw
    basis is ill-conditioned -- the base has several times the distal tip
    authority -- so one damping value either lags the distal segment or goes
    unstable at the base. n_segments == 1 reduces to the identity (raw fallback).
    """
    m = np.zeros((2 * n_segments, 2 * n_segments))
    col = 0
    for axis in range(2):  # 0 = X bending, 1 = Y bending
        for i in range(n_segments - 1):
            m[2 * i + axis, col] = 1.0
            m[2 * (i + 1) + axis, col] = -1.0
            col += 1
        m[2 * (n_segments - 1) + axis, col] = 1.0
        col += 1
    return m


class TDCRIKController:
    """Task-space controller for TDCR using Jacobian-based IK.

    Maps Cartesian velocity commands to Clark coordinate changes via
    the tip Jacobian matrix.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tendon_distance_mm: float = 4.0,
        angle_offset_rad_ccw: Optional[np.ndarray] = None,
        velocity_scale: float = 0.1,
        fps: float = 100.0,
        damping_factor: float = 0.01,
        verbose: bool = True,
        clark_direct_scale: float = 15.0,
        tension_mode: bool = False,
        jacobian_refresh_hz: float = 10.0,
        settle_horizon_s: float = 0.1,
        jacobian_perturbation_mm: float = 1.5,
        velocity_boost: float = 50.0,
        rotation_boost: float = 1000.0,
        jacobian_cols_per_refresh: int = 2,
        contact_settle_horizon_s: float = None,
        synergy_basis: bool = True,
    ):
        """Initialize TDCR task-space controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            tendon_distance_mm: Distance from backbone to tendons
            angle_offset_rad_ccw: Angular offsets for each segment
            velocity_scale: Scaling factor for velocity commands
            fps: Control loop frequency
            damping_factor: Damping for pseudo-inverse computation
            verbose: Enable verbose output for debugging
            clark_direct_scale: Gain for direct Clark coordinate control (T/F/G/H
                keys). No longer affects the numerical Jacobian (see
                jacobian_perturbation_mm).
            tension_mode: If True, use tension kinematics (opposing forces in adjacent segments)
            jacobian_refresh_hz: Max rate at which the local Jacobian is
                re-estimated by cloning + settling. Between refreshes the cached
                Jacobian drives the cheap per-frame pinv-IK, so the control/render
                loop stays responsive. The estimate also refreshes on command
                onset, control-point switch, and contact-topology change.
            settle_horizon_s: Wall-clock duration the perturbed clone is
                integrated to settle before reading the tip pose. Converted to
                an integer step count via model.opt.timestep.
            jacobian_perturbation_mm: Finite-difference step (mm of Clark
                coordinate) used to estimate the Jacobian. Decoupled from
                clark_direct_scale so both controllers share one stencil.
            velocity_boost: Linear command gain feeding the IK velocity.
            rotation_boost: Angular command gain feeding the IK velocity.
            jacobian_cols_per_refresh: How many Jacobian columns to re-estimate
                per control step while moving/in contact. The estimate is spread
                across steps (there are 9 columns: 8 Clark + insertion) so no
                single step stalls the loop. Lower = smoother loop but a slightly
                staler Jacobian; higher = fresher but heavier per step.
            contact_settle_horizon_s: Settle horizon used only when in contact
                (ncon>0), where the perturbation response settles more slowly.
                None (default) keeps the free-space horizon. Compliant tension
                control wants a longer value (~0.3 s) for an accurate contact
                Jacobian; stiff position control gains nothing (it settles fast
                but is capped by its own contact chatter) so it leaves this None.
            synergy_basis: If True (default), solve the tip DLS in the synergy
                basis (build_synergy_matrix) rather than the ill-conditioned raw
                per-segment basis; needed for crisp multi-segment tip tracking.
        """
        self.model = model
        self.data = data
        self.velocity_scale = velocity_scale
        self.fps = fps
        self.dt = 1.0 / fps
        self.damping_factor = damping_factor  # Use original damping factor
        self.verbose = verbose
        self.clark_direct_scale = clark_direct_scale
        self.tension_mode = tension_mode
        self.synergy_basis = synergy_basis

        # Live-Jacobian estimation knobs
        self.jacobian_refresh_hz = jacobian_refresh_hz  # kept for config/log compat
        self.jacobian_perturbation_mm = jacobian_perturbation_mm
        self.velocity_boost = velocity_boost
        self.rotation_boost = rotation_boost
        self.jacobian_cols_per_refresh = max(1, int(jacobian_cols_per_refresh))
        # Settle horizon as a step count derived from the actual sim timestep,
        # so the same wall-clock settle is used regardless of scene timestep.
        self.equilibrium_steps = max(
            1, int(round(settle_horizon_s / model.opt.timestep))
        )
        self.settle_horizon_s = self.equilibrium_steps * model.opt.timestep
        # Under contact the perturbation response settles more slowly (the
        # backbone deflects against the obstacle), so when in contact we settle
        # longer for an accurate estimate. Matters most for compliant tension
        # control; position control settles fast and is capped by its own
        # contact chatter, so it leaves this at the free-space horizon (None).
        if contact_settle_horizon_s is None:
            self.contact_equilibrium_steps = self.equilibrium_steps
        else:
            self.contact_equilibrium_steps = max(
                self.equilibrium_steps,
                int(round(contact_settle_horizon_s / model.opt.timestep)),
            )

        # Cached Jacobian + the state it was estimated at. A full re-estimate
        # runs only on first use / control-point switch (_needs_full_refresh);
        # while moving or in contact, _refresh_incremental updates a few columns
        # per step (round-robin via _refresh_cursor) so the loop never stalls.
        self._cached_J_pos = None
        self._cached_J_rot = None
        self._cached_tip_id = None
        self._refresh_cursor = 0
        self._jacobian_frozen = False  # hold the cache fixed (debugging / A-B)
        self._prev_command_active = False

        # Insertion DOF (base translation along the backbone axis). Off for the
        # plain TDCR controller; the multi-point controllers enable it once their
        # control points + Franka actuators exist. When on, refresh_jacobian also
        # estimates an insertion column in the same per-mm units as the Clark
        # columns, and _solve_tdcr_ik augments the IK with it.
        self._has_insertion_dof = False
        self._cached_J_ins_pos = None
        self._cached_J_ins_rot = None

        # Create separate MjData instance for Jacobian computation
        # This prevents perturbations from affecting the main simulation state
        self.data_jacobian = mujoco.MjData(model)
        if self.verbose:
            print("Created separate MjData instance for Jacobian computation")

        # Find tip body for Jacobian computation. mj_name2id returns -1 when
        # the name is missing (it does not raise), so check explicitly.
        self.tip_body_id = -1
        for tip_name in ("EE_pos", "link_end"):
            tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tip_name)
            if tip_id >= 0:
                self.tip_body_id = tip_id
                if self.verbose:
                    print(f"Found tip body '{tip_name}' with ID {tip_id}")
                break
        if self.tip_body_id < 0:
            raise RuntimeError("Could not find tip body 'EE_pos' or 'link_end'")

        # Initialize kinematics
        if angle_offset_rad_ccw is None:
            angle_offset_rad_ccw = np.array([0, np.pi / 6, np.pi / 3])

        # Choose kinematics class based on tension mode
        if self.tension_mode:
            self.kinematics = MultiSegmentTDCRTensionKinematics(
                n_tendons_per_segment=[3, 3, 3],
                tendon_distances_mm=tendon_distance_mm,
                angle_offsets_rad_ccw=angle_offset_rad_ccw,
                max_bending_angles_rad=np.pi,
            )
            if self.verbose:
                print("Using tension mode kinematics (opposing forces)")
        else:
            self.kinematics = ThreeTendonThreeSegmentTDCRKinematics(
                tendon_distance_mm=tendon_distance_mm,
                angle_offset_rad_ccw=angle_offset_rad_ccw,
                max_bending_angle_rad=np.pi,
            )
            if self.verbose:
                print("Using position mode kinematics (cumulative coupling)")

        # Find tendon actuators
        self._find_tendon_actuators()

        # Initialize from pretension
        self.pretension_lengths = None
        self._initialize_from_pretension()

        # Reset to current
        self.reset_to_current()

        # Hold-to-home rates (per control step): Clark coords (mm) toward zero, and
        # for the combined task-space controllers, Franka joints (rad) toward home.
        self.home_clark_step = 1.0
        self.home_joint_step = 0.01

        # Start with more pronounced S-shaped Clark coordinates for better workspace
        # Larger S-shape provides more range of motion and visibility
        initial_s_shape = np.array([-6.0, -6.0, 15.0, 10.0, -8.0, -8.0])  # mm
        self.kinematics.goal_clark_coords = initial_s_shape

        # Apply the S-shape to the actual robot before computing Jacobian
        # Preserve non-TDCR actuator values (e.g., Franka joints)
        tendons_mm = self.kinematics.clark_to_tendons_mm(initial_s_shape)
        ctrl_values = self._tendons_mm_to_ctrl(tendons_mm)
        for j, act_id in enumerate(self.tendon_actuator_ids):
            self.data.ctrl[act_id] = ctrl_values[j]

        # Step to let S-shape settle
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        # Initialize Jacobian computation variables. Columns are the RAW Clark
        # coordinates (2 per segment: X/Y bending), one column per DOF -- the
        # multi-point controllers mask distal columns, so a proximal control point
        # only actuates its own + more-proximal segments. (The tip DLS separately
        # solves in the synergy basis; see build_synergy_matrix.)
        self.n_clark = len(self.kinematics.goal_clark_coords)  # 2 * n_segments
        self.jacobian_pos = np.zeros((3, self.n_clark))  # d(tip_pos)/d(clark_i)
        self.jacobian_rot = np.zeros((3, self.n_clark))  # d(tip_rot)/d(clark_i)
        self.jacobian_full = np.zeros((6, self.n_clark))

        # Synergy basis for the tip DLS (build_synergy_matrix); None = raw basis.
        self._synergy_M = (
            build_synergy_matrix(self.n_clark // 2) if self.synergy_basis else None
        )

        # Compute initial Jacobian with actual stepping for accuracy
        # Now the robot is in S-shape, so Jacobian should be better
        if self.verbose:
            print("Computing initial Jacobian from S-shape...")
        self.refresh_jacobian()

        if self.verbose:
            print("TDCR IK Controller initialized")
            print(f"  Velocity scale: {self.velocity_scale}")
            print(f"  Damping factor: {self.damping_factor}")
            print(f"  Control frequency: {self.fps} Hz")
            print(f"\nInitial Position Jacobian (3x{self.n_clark}):")
            print("  Rows: [X, Y, Z] tip position")
            print("  Cols: per-segment Clark DOF [s1X, s1Y, s2X, s2Y, ...]")
            for i in range(3):
                print(f"  {['X','Y','Z'][i]}: ", end="")
                for j in range(self.n_clark):
                    print(f"{self.jacobian_pos[i,j]:7.3f} ", end="")
                print()
            print(f"\nInitial Rotation Jacobian (3x{self.n_clark}):")
            print("  Rows: [RX, RY, RZ] tip rotation")
            for i in range(3):
                print(f"  {['RX','RY','RZ'][i]}: ", end="")
                for j in range(self.n_clark):
                    print(f"{self.jacobian_rot[i,j]:7.3f} ", end="")
                print()

    def _find_tendon_actuators(self):
        """Find actuator IDs for TDCR tendons."""
        self.tendon_actuator_ids = []

        for seg_idx in range(3):
            for ten_idx in range(3):
                actuator_name = f"seg_{seg_idx}_ten_{ten_idx}"
                actuator_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                if actuator_id >= 0:
                    self.tendon_actuator_ids.append(actuator_id)
                elif self.verbose:
                    print(f"Warning: Actuator '{actuator_name}' not found")

        if self.verbose:
            print(f"Found {len(self.tendon_actuator_ids)} TDCR tendon actuators")

    def _initialize_from_pretension(self):
        """Initialize from pretension keyframe if available."""
        # Check for pretension keyframe
        for i in range(self.model.nkey):
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name == "pretension":
                key_ctrl = self.model.key_ctrl[i]
                self.pretension_lengths = np.zeros(len(self.tendon_actuator_ids))
                for j, act_id in enumerate(self.tendon_actuator_ids):
                    self.pretension_lengths[j] = key_ctrl[act_id]
                # Let the pretension settle once
                for _ in range(200):
                    mujoco.mj_step(self.model, self.data)
                if self.verbose:
                    print(f"Initialized from pretension keyframe")
                    print(f"Pretension values: {self.pretension_lengths}")
                return

        self.pretension_lengths = np.zeros(len(self.tendon_actuator_ids))
        if self.verbose:
            print("No pretension keyframe found, using zero pretension")

    def reset_to_current(self):
        """Reset goal to current tendon positions."""
        if self.data is not None and len(self.tendon_actuator_ids) > 0:
            current_tendons = np.zeros(len(self.tendon_actuator_ids))
            for i, act_id in enumerate(self.tendon_actuator_ids):
                current_tendons[i] = self.data.ctrl[act_id]

            # Convert to relative values (in meters) and then to mm for kinematics
            if self.pretension_lengths is not None:
                relative_tendons_m = current_tendons - self.pretension_lengths
            else:
                relative_tendons_m = current_tendons

            # Convert to mm for kinematics
            relative_tendons_mm = relative_tendons_m * 1000
            self.kinematics.set_goal_clark_coords_to_current(relative_tendons_mm)
        else:
            self.kinematics.set_goal_clark_coords_to_home()

    def reset_to_home(self):
        """Reset all segments to home position."""
        self.kinematics.set_goal_clark_coords_to_home()
        if self.verbose:
            print("Reset to home position")

    def _tendons_mm_to_ctrl(self, tendons_mm: np.ndarray) -> np.ndarray:
        """Convert kinematics tendon length changes (mm) to actuator ctrl values.

        Position-mode default: meters relative to the pretension lengths.
        Tension-mode controllers override this so the Jacobian perturbations
        use the same mm->ctrl mapping as command execution.
        """
        tendons_m = tendons_mm * 0.001
        if self.pretension_lengths is not None:
            return self.pretension_lengths + tendons_m
        return tendons_m

    def _derive_segment_end_body_names(self) -> list:
        """Derive the body name at the end of each segment from the model.

        For each spatial tendon ``seg_{s}_tendon_0`` the last routing site
        sits on that segment's end link, so this works for any link count or
        chain mode (standard, modular, constraint-factor sites) without
        hardcoding link indices.
        """
        names = []
        for s in range(
            self.kinematics.n_segments if hasattr(self.kinematics, "n_segments") else 3
        ):
            ten_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_TENDON, f"seg_{s}_tendon_0"
            )
            if ten_id < 0:
                names.append(None)
                continue
            # tendon_adr/tendon_num are the static wrap-path arrays on MjModel
            # (the ten_wrapadr/ten_wrapnum spelling only exists on MjData).
            adr = self.model.tendon_adr[ten_id]
            num = self.model.tendon_num[ten_id]
            last_body = None
            for w in range(adr, adr + num):
                if self.model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_SITE:
                    site_id = self.model.wrap_objid[w]
                    last_body = self.model.site_bodyid[site_id]
            if last_body is None:
                names.append(None)
            else:
                names.append(
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, int(last_body)
                    )
                )
        return names

    def compute_jacobian_numerical(self, cols=None) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the local tip Jacobian by central finite differences.

        cols: optional iterable of Clark-DOF column indices (0..n_clark-1) to
        (re)estimate; the rest of self.jacobian_pos/rot are left as-is. Used by
        the incremental refresh to spread the estimate over several control
        steps. Default (None) estimates all n_clark columns.

        For each control mode the Clark goal is perturbed by
        ``±jacobian_perturbation_mm``, applied to a cloned MjData, and the clone
        is integrated ``equilibrium_steps`` (≈ settle_horizon_s) to settle before
        the control point's pose is read. Columns are divided by the actual
        perturbation magnitude (2·epsilon mm), so the Jacobian is in physical
        units (m-tip / mm-Clark and rad-tip / mm-Clark) and is independent of the
        user-facing clark_direct_scale gain. Re-estimating from the live state
        captures Jacobian changes under contact.

        Returns:
            Tuple of (position_jacobian, rotation_jacobian)
        """
        # Copy current simulation state to the separate Jacobian data instance
        # This ensures Jacobian computation starts from the current robot state.
        # act (stateful-actuator activation) matters for force/position actuators
        # with internal state, so copy it too or the clone linearizes about the
        # wrong actuator state.
        self.data_jacobian.qpos[:] = self.data.qpos
        self.data_jacobian.qvel[:] = self.data.qvel
        self.data_jacobian.ctrl[:] = self.data.ctrl
        if self.model.na:
            self.data_jacobian.act[:] = self.data.act
        mujoco.mj_forward(self.model, self.data_jacobian)

        # Save state for restoration after each perturbation
        saved_qpos = self.data_jacobian.qpos.copy()
        saved_qvel = self.data_jacobian.qvel.copy()
        saved_act = self.data_jacobian.act.copy() if self.model.na else None

        # Get current Clark coordinates
        clark_coords = self.kinematics.goal_clark_coords.copy()

        # Finite-difference step (mm of Clark coordinate) and settle horizon
        # (longer under contact, where the response settles more slowly).
        epsilon = self.jacobian_perturbation_mm
        equilibrium_steps = self._settle_steps()

        # One column per raw Clark coordinate: column i = d(tip)/d(clark_i),
        # estimated by central differences (perturb only coordinate i by ±eps).
        for col_idx in range(self.n_clark) if cols is None else cols:
            # Positive perturbation of Clark coordinate col_idx only
            clark_perturbed = clark_coords.copy()
            clark_perturbed[col_idx] += epsilon

            # Apply perturbed configuration
            tendon_lengths_mm = self.kinematics.clark_to_tendons_mm(clark_perturbed)
            ctrl_values = self._tendons_mm_to_ctrl(tendon_lengths_mm)

            # Restore state
            self.data_jacobian.qpos[:] = saved_qpos
            self.data_jacobian.qvel[:] = saved_qvel
            if saved_act is not None:
                self.data_jacobian.act[:] = saved_act

            # Apply control
            for j, act_id in enumerate(self.tendon_actuator_ids):
                self.data_jacobian.ctrl[act_id] = ctrl_values[j]

            # Step to equilibrium
            for _ in range(equilibrium_steps):
                mujoco.mj_step(self.model, self.data_jacobian)

            tip_pos_plus = self.data_jacobian.xpos[self.tip_body_id].copy()
            tip_quat_plus = self.data_jacobian.xquat[self.tip_body_id].copy()

            # Negative perturbation of the same coordinate
            clark_perturbed = clark_coords.copy()
            clark_perturbed[col_idx] -= epsilon

            tendon_lengths_mm = self.kinematics.clark_to_tendons_mm(clark_perturbed)
            ctrl_values = self._tendons_mm_to_ctrl(tendon_lengths_mm)

            # Restore state
            self.data_jacobian.qpos[:] = saved_qpos
            self.data_jacobian.qvel[:] = saved_qvel
            if saved_act is not None:
                self.data_jacobian.act[:] = saved_act

            # Apply control
            for j, act_id in enumerate(self.tendon_actuator_ids):
                self.data_jacobian.ctrl[act_id] = ctrl_values[j]

            # Step to equilibrium
            for _ in range(equilibrium_steps):
                mujoco.mj_step(self.model, self.data_jacobian)

            tip_pos_minus = self.data_jacobian.xpos[self.tip_body_id].copy()
            tip_quat_minus = self.data_jacobian.xquat[self.tip_body_id].copy()

            # Jacobian columns in physical units: divide tip displacement by the
            # actual perturbation magnitude (2*epsilon mm), NOT a fictitious time.
            self.jacobian_pos[:, col_idx] = (tip_pos_plus - tip_pos_minus) / (
                2 * epsilon
            )

            quat_diff = self._quat_multiply(
                tip_quat_plus, self._quat_conjugate(tip_quat_minus)
            )
            angle = 2 * np.arccos(np.clip(quat_diff[0], -1, 1))
            if angle > 1e-6:
                axis = quat_diff[1:4] / np.sin(angle / 2)
                self.jacobian_rot[:, col_idx] = axis * angle / (2 * epsilon)
            else:
                self.jacobian_rot[:, col_idx] = 0

        # Note: No need to restore state since we used a separate data_jacobian instance
        # The main self.data remains untouched throughout the computation

        # Combine into full Jacobian
        self.jacobian_full[:3, :] = self.jacobian_pos
        self.jacobian_full[3:, :] = self.jacobian_rot

        return self.jacobian_pos, self.jacobian_rot

    def _settle_steps(self) -> int:
        """Settle-step count for the current state: the longer contact horizon
        when the robot is in contact (response settles slowly against the
        obstacle), the free-space horizon otherwise."""
        return (
            self.contact_equilibrium_steps
            if self.data.ncon > 0
            else self.equilibrium_steps
        )

    def _needs_full_refresh(self) -> bool:
        """A full (all-column) re-estimate is required only when the cache can't
        be incrementally trusted: no cache yet, or the control point changed
        (every column then refers to a different body)."""
        return (
            self._cached_J_pos is None
            or (self._has_insertion_dof and self._cached_J_ins_pos is None)
            or self._cached_tip_id != self.tip_body_id
        )

    def refresh_jacobian(self):
        """Full re-estimate of all columns about the current state (used at init
        and on control-point switch). Expensive (~all columns × settle); the
        steady-state path is _refresh_incremental."""
        self.compute_jacobian_numerical()
        self._cached_J_pos = self.jacobian_pos.copy()
        self._cached_J_rot = self.jacobian_rot.copy()
        if self._has_insertion_dof:
            self._cached_J_ins_pos, self._cached_J_ins_rot = (
                self._compute_insertion_jacobian_column()
            )
        self._cached_tip_id = self.tip_body_id
        self._refresh_cursor = 0

    def _refresh_incremental(self, n_cols: int):
        """Re-estimate n_cols of the cached Jacobian (rotating through all
        columns) about the CURRENT state, updating them in place. Spreading the
        full estimate over several control steps keeps each step cheap, so the
        teleop loop never stalls on a single big re-estimate. Each column is
        measured at the (slightly different) state of the step it runs on, so
        the cache is a rolling local estimate that tracks the moving config and
        contact."""
        nc = self.n_clark
        n_total = nc + 1 if self._has_insertion_dof else nc
        for _ in range(n_cols):
            col = self._refresh_cursor
            if col < nc:
                self.compute_jacobian_numerical(cols=[col])
                self._cached_J_pos[:, col] = self.jacobian_pos[:, col]
                self._cached_J_rot[:, col] = self.jacobian_rot[:, col]
            elif self._has_insertion_dof:
                self._cached_J_ins_pos, self._cached_J_ins_rot = (
                    self._compute_insertion_jacobian_column()
                )
            self._refresh_cursor = (self._refresh_cursor + 1) % n_total

    # ----- Insertion DOF (base translation along the backbone) -------------- #
    # These helpers reference self.control_points / self.franka_actuator_ids,
    # which only exist on the multi-point subclasses; they run only when
    # _has_insertion_dof is True (set by those subclasses).

    def _franka_pos_jacobian_on(self, data_inst, body_id: int) -> np.ndarray:
        """3x7 position Jacobian of `body_id` w.r.t. the 7 Franka joints."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data_inst, jacp, jacr, body_id)
        return jacp[:, :7]

    def _insertion_dir_world(self, data_inst) -> np.ndarray:
        """Unit world vector from the TDCR base toward segment-1's end (the
        backbone/insertion axis), evaluated on the given data instance."""
        base_id = self.control_points["base"]["body_id"]
        seg1_id = self.control_points["seg1"]["body_id"]
        vec = data_inst.xpos[seg1_id] - data_inst.xpos[base_id]
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    def _franka_delta_for_translation(
        self, data_inst, body_id: int, world_translation: np.ndarray
    ) -> np.ndarray:
        """7-joint delta that translates `body_id` by world_translation (m), via
        the damped pseudo-inverse of the Franka position Jacobian."""
        jacp = self._franka_pos_jacobian_on(data_inst, body_id)
        return np.linalg.pinv(jacp, rcond=self.damping_factor) @ world_translation

    def _compute_insertion_jacobian_column(self):
        """Estimate d(control_point)/d(base insertion) by translating the base
        +/- jacobian_perturbation_mm along the backbone axis on the clone and
        settling. Returned in the SAME per-mm units as the Clark columns so the
        augmented pseudo-inverse weights insertion and bending consistently."""
        dj = self.data_jacobian
        dj.qpos[:] = self.data.qpos
        dj.qvel[:] = self.data.qvel
        dj.ctrl[:] = self.data.ctrl
        if self.model.na:
            dj.act[:] = self.data.act
        mujoco.mj_forward(self.model, dj)
        saved_qpos = dj.qpos.copy()
        saved_qvel = dj.qvel.copy()
        saved_ctrl = dj.ctrl.copy()
        saved_act = dj.act.copy() if self.model.na else None

        base_id = self.control_points["base"]["body_id"]
        cp_id = self.tip_body_id
        eps_m = self.jacobian_perturbation_mm * 0.001  # base translation (m)
        steps = self._settle_steps()

        def settle_and_read(sign):
            dj.qpos[:] = saved_qpos
            dj.qvel[:] = saved_qvel
            dj.ctrl[:] = saved_ctrl
            if saved_act is not None:
                dj.act[:] = saved_act
            mujoco.mj_forward(self.model, dj)
            idir = self._insertion_dir_world(dj)
            djoint = self._franka_delta_for_translation(
                dj, base_id, sign * eps_m * idir
            )
            for k, aid in enumerate(self.franka_actuator_ids):
                dj.ctrl[aid] = saved_ctrl[aid] + djoint[k]
            for _ in range(steps):
                mujoco.mj_step(self.model, dj)
            return dj.xpos[cp_id].copy(), dj.xquat[cp_id].copy()

        cp_plus, q_plus = settle_and_read(+1.0)
        cp_minus, q_minus = settle_and_read(-1.0)

        j_pos = (cp_plus - cp_minus) / (2 * self.jacobian_perturbation_mm)
        quat_diff = self._quat_multiply(q_plus, self._quat_conjugate(q_minus))
        angle = 2 * np.arccos(np.clip(quat_diff[0], -1, 1))
        if angle > 1e-6:
            j_rot = (
                quat_diff[1:4]
                / np.sin(angle / 2)
                * angle
                / (2 * self.jacobian_perturbation_mm)
            )
        else:
            j_rot = np.zeros(3)
        return j_pos, j_rot

    def _apply_ik_insertion(
        self, insertion_increment_mm: float, target_qpos: np.ndarray
    ) -> np.ndarray:
        """Translate the base by insertion_increment_mm along the backbone axis
        (consistent with the insertion column), writing Franka joint targets."""
        base_id = self.control_points["base"]["body_id"]
        idir = self._insertion_dir_world(self.data)
        djoint = self._franka_delta_for_translation(
            self.data, base_id, idir * insertion_increment_mm * 0.001
        )
        for k, aid in enumerate(self.franka_actuator_ids):
            target_qpos[aid] = self.data.ctrl[aid] + djoint[k]
        return target_qpos

    def _solve_tdcr_ik(
        self, v_cartesian, tdcr_segments, position_weight, orientation_weight
    ):
        """Damped least-squares IK over the active per-segment Clark DOF plus the
        base-insertion DOF (when enabled). Uses the cached Jacobian (the caller
        is responsible for refreshing it). Returns (clark_velocities[n_clark],
        insertion_velocity) where insertion_velocity is mm/s (0 if no DOF).

        For control point seg_S only segments 1..S are actuated: the distal
        segments don't move seg_S's endpoint and actuating them would flail the
        tip, so they're held fixed (excluded from the solve)."""
        nc = self.n_clark
        j_pos, j_rot = self._cached_J_pos, self._cached_J_rot
        has_ins = self._has_insertion_dof and self._cached_J_ins_pos is not None
        if has_ins:
            j_pos = np.hstack([j_pos, self._cached_J_ins_pos.reshape(3, 1)])
            j_rot = np.hstack([j_rot, self._cached_J_ins_rot.reshape(3, 1)])
        n = nc + 1 if has_ins else nc

        # Active Clark DOF = the 2 coords of every segment up to the control
        # point's segment (segments 1..S); distal segments are held fixed.
        n_active = 2 * (max(tdcr_segments) + 1) if tdcr_segments else 0
        mask = np.zeros(n, dtype=bool)
        mask[:n_active] = True
        if has_ins:
            mask[nc] = True  # insertion always active

        j_active = np.vstack(
            [j_pos[:, mask] * position_weight, j_rot[:, mask] * orientation_weight]
        )
        v_desired = v_cartesian.copy()
        v_desired[:3] *= position_weight
        v_desired[3:] *= orientation_weight
        active_vel = np.linalg.pinv(j_active, rcond=self.damping_factor) @ v_desired

        control_velocities = np.zeros(n)
        control_velocities[mask] = active_vel
        clark_velocities = control_velocities[:nc]
        insertion_vel = control_velocities[nc] if has_ins else 0.0
        return clark_velocities, insertion_vel

    def _maybe_refresh_jacobian(
        self, command_active: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keep the cached Jacobian current, cheaply. Full re-estimate only when
        required (first use / control-point switch); otherwise, while moving or
        in contact, re-estimate a few columns per call (_refresh_incremental) so
        no single control step stalls the loop. Idle + contact-free reuses the
        cache (~free). Returns (J_pos, J_rot)."""
        if self._jacobian_frozen:
            pass  # hold the current cache (debugging / A-B tests)
        elif self._needs_full_refresh():
            self.refresh_jacobian()
        elif command_active or self.data.ncon > 0:
            self._refresh_incremental(self.jacobian_cols_per_refresh)
        # Mirror into the public attributes so get_info()/debug still see current J.
        self.jacobian_pos = self._cached_J_pos
        self.jacobian_rot = self._cached_J_rot
        return self._cached_J_pos, self._cached_J_rot

    def _note_command_active(self, command_active: bool):
        """Retained for the subclass exit-path calls; the incremental refresh no
        longer needs command-onset detection (it runs every motion/contact step),
        but tracking the flag is harmless and cheap."""
        self._prev_command_active = command_active

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _solve_clark_dls(self, J: np.ndarray, v_desired: np.ndarray) -> np.ndarray:
        """Damped least-squares for Clark velocities. With the synergy basis M
        (default) damp the synergy activations and map back (clark_vel = M y);
        otherwise damp the raw per-segment bends directly."""
        if self._synergy_M is not None:
            M = self._synergy_M
            Js = J @ M
            A = Js.T @ Js + self.damping_factor * np.eye(M.shape[1])
            try:
                y = np.linalg.solve(A, Js.T @ v_desired)
            except np.linalg.LinAlgError:
                if self.verbose:
                    print("Warning: Singular matrix in IK, using SVD")
                y = np.linalg.pinv(Js, rcond=self.damping_factor) @ v_desired
            return M @ y

        JtJ = J.T @ J
        damped_JtJ = JtJ + self.damping_factor * np.eye(self.n_clark)
        try:
            return np.linalg.solve(damped_JtJ, J.T @ v_desired)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("Warning: Singular matrix in IK, using SVD")
            return np.linalg.pinv(J, rcond=self.damping_factor) @ v_desired

    def compute_target_qpos(
        self, command: Dict[str, float], data: Optional[mujoco.MjData] = None
    ) -> np.ndarray:
        """Compute target joint positions from Cartesian velocity command.

        Args:
            command: Dictionary with keys:
                - 'vx', 'vy', 'vz': Linear velocity commands
                - 'wx', 'wy', 'wz': Angular velocity commands (optional)
                - 'reset_home': Reset to home if True
            data: MuJoCo data (optional)

        Returns:
            Array of target positions for all actuators
        """
        # Hold-to-home: while reset is held, step the goal Clark coords toward zero
        # (synced) and drive the tendons there; releasing the key stops the motion.
        if command.get("reset_home", False):
            gc = self.kinematics.goal_clark_coords
            new_clark = step_toward(gc, np.zeros_like(gc), self.home_clark_step)
            self.kinematics.goal_clark_coords = new_clark
            tendons_m = self.kinematics.clark_to_tendons_mm(new_clark) * 0.001
            target_qpos = (
                self.data.ctrl.copy() if self.data else np.zeros(self.model.nu)
            )
            for i, act_id in enumerate(self.tendon_actuator_ids):
                target_qpos[act_id] = (
                    self.pretension_lengths[i]
                    if self.pretension_lengths is not None
                    else 0.0
                ) + tendons_m[i]
            self._note_command_active(False)  # reset is not Cartesian motion
            return target_qpos

        # Build Cartesian velocity vector (scaled for faster motion)
        v_cartesian = np.zeros(6)
        v_cartesian[0] = (
            command.get("vx", 0.0) * self.velocity_scale * self.velocity_boost
        )
        v_cartesian[1] = (
            command.get("vy", 0.0) * self.velocity_scale * self.velocity_boost
        )
        v_cartesian[2] = (
            command.get("vz", 0.0) * self.velocity_scale * self.velocity_boost
        )
        v_cartesian[3] = (
            command.get("wx", 0.0) * self.velocity_scale * self.rotation_boost
        )
        v_cartesian[4] = (
            command.get("wy", 0.0) * self.velocity_scale * self.rotation_boost
        )
        v_cartesian[5] = (
            command.get("wz", 0.0) * self.velocity_scale * self.rotation_boost
        )

        # Live local Jacobian: re-estimate only when stale (throttled to
        # jacobian_refresh_hz, plus command-onset / contact-change events);
        # reuse the cache otherwise so this call stays cheap most frames.
        command_active = np.linalg.norm(v_cartesian) > 1e-6
        J_pos, J_rot = self._maybe_refresh_jacobian(command_active)

        # Adaptive weighted control based on user command
        # Linear only: 100% position, 10% orientation (relax orientation)
        # Angular only: 50% position, 50% orientation (balanced)
        linear_vel_magnitude = np.linalg.norm(v_cartesian[:3])
        angular_vel_magnitude = np.linalg.norm(v_cartesian[3:])

        # Determine control mode based on what user is commanding
        commanding_linear = linear_vel_magnitude > 1e-6
        commanding_angular = angular_vel_magnitude > 1e-6

        if commanding_linear and not commanding_angular:
            # Pure linear motion: position priority, relax orientation (10-90)
            position_weight = 1.0
            orientation_weight = 0.1
        elif commanding_angular and not commanding_linear:
            # Pure angular motion: balanced (50-50)
            position_weight = 0.5
            orientation_weight = 0.5
        elif commanding_linear and commanding_angular:
            # Both commanded: use weighted control based on magnitudes
            # But ensure minimum weight for stability
            total_magnitude = linear_vel_magnitude + angular_vel_magnitude
            position_weight = max(0.1, linear_vel_magnitude / total_magnitude)
            orientation_weight = max(0.1, angular_vel_magnitude / total_magnitude)
        else:
            # No motion commanded, use position-biased weights
            position_weight = 1.0
            orientation_weight = 0.1

        # Apply weights to Jacobian
        J_pos_weighted = J_pos * position_weight
        J_rot_weighted = J_rot * orientation_weight
        J = np.vstack([J_pos_weighted, J_rot_weighted])

        # Apply weights to velocity commands
        v_desired = v_cartesian.copy()
        v_desired[:3] *= position_weight
        v_desired[3:] *= orientation_weight

        # Damped least squares over the Clark DOF (synergy basis by default).
        clark_velocities = self._solve_clark_dls(J, v_desired)

        # Scale by time step
        clark_increment = clark_velocities * self.dt

        # Update Clark coordinates
        new_clark = self.kinematics.goal_clark_coords + clark_increment

        # Apply limits (optional - can be tuned)
        max_clark_value = (
            100.0  # Maximum Clark coordinate magnitude in mm (increased for more range)
        )
        new_clark = np.clip(new_clark, -max_clark_value, max_clark_value)

        # Update kinematics goal
        self.kinematics.goal_clark_coords = new_clark

        # Convert to tendon lengths (convert mm to m)
        target_tendons_mm = self.kinematics.clark_to_tendons_mm(new_clark)
        target_tendons_m = target_tendons_mm * 0.001

        # Build full control array - preserve current control values for non-TDCR actuators
        target_qpos = self.data.ctrl.copy() if self.data else np.zeros(self.model.nu)

        # Set tendon actuator targets
        for i, (act_id, tendon_length) in enumerate(
            zip(self.tendon_actuator_ids, target_tendons_m)
        ):
            if self.pretension_lengths is not None:
                target_qpos[act_id] = self.pretension_lengths[i] + tendon_length
            else:
                target_qpos[act_id] = tendon_length

        self._note_command_active(command_active)

        # Verbose output for debugging
        if self.verbose and np.linalg.norm(v_desired) > 1e-6:
            print(f"\n--- TDCR IK Step ---")
            print(f"Cartesian velocity XYZ RXYZ: {v_desired}")
            print(f"Clark velocities: {clark_velocities}")
            print(f"Clark coords: {new_clark}")
            print(f"Jacobian condition: {np.linalg.cond(J):.2f}")
            print(f"Tip position: {self.data.xpos[self.tip_body_id]}")

            # Print full matrices for detailed analysis
            print(f"\nFull Position Jacobian:")
            for i in range(3):
                print(f"  {['X','Y','Z'][i]}: ", end="")
                for j in range(self.n_clark):
                    print(f"{J_pos[i,j]:8.4f} ", end="")
                print()
            print(f"Full Rotation Jacobian:")
            for i in range(3):
                print(f"  {['RX','RY','RZ'][i]}: ", end="")
                for j in range(self.n_clark):
                    print(f"{J_rot[i,j]:8.4f} ", end="")
                print()

        return target_qpos

    def get_info(self) -> Dict[str, Any]:
        """Get controller information for display.

        Returns:
            Dictionary with controller state information
        """
        clark_coords = self.kinematics.goal_clark_coords
        tip_pos = self.data.xpos[self.tip_body_id] if self.data else np.zeros(3)

        return {
            "type": "TDCR Task-Space Control",
            "tip_position": tip_pos,
            "clark_coords": {
                "seg1": clark_coords[0:2],
                "seg2": clark_coords[2:4],
                "seg3": clark_coords[4:6],
            },
            "jacobian_condition": np.linalg.cond(self.jacobian_pos),
            "velocity_scale": self.velocity_scale,
        }
