"""Combined controller for Franka + TDCR robots.

This controller manages both robots simultaneously:
- Franka: IK-based task-space control
- TDCR: Joint-space control with Clark coordinates
"""

import numpy as np
from typing import Any, Dict, Optional
import mujoco
from scipy.spatial.transform import Rotation

from .ik_controller import IKController
from .tdcr_joint_controller import TDCRJointController
from .homing import step_toward


class CombinedController:
    """Combined controller for dual robot system.

    Manages control for both Franka (IK) and TDCR (joint) robots.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tdcr_params: Optional[Dict] = None,
    ):
        """Initialize combined controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            tdcr_params: TDCR controller parameters
        """
        self.model = model
        self.data = data

        # Initialize Franka IK controller
        print("Initializing Franka IK controller...")
        self.franka_controller = IKController(model, data)

        # Initialize TDCR joint controller
        print("Initializing TDCR joint controller...")
        tdcr_params = tdcr_params or {}
        self.tdcr_controller = TDCRJointController(
            model,
            data=data,
            tendon_distance_mm=tdcr_params.get("tendon_distance_mm", 4.0),
            angle_offset_rad_ccw=tdcr_params.get("angle_offset_rad_ccw"),
            clark_speed_scale=tdcr_params.get("clark_speed_scale", 0.1),
            fps=tdcr_params.get("fps", 100),
            n_tendons_per_segment=tdcr_params.get("n_tendons_per_segment"),
            n_segments=tdcr_params.get("n_segments"),
            tension_mode=tdcr_params.get("tension_mode", False),
            independent_segments=tdcr_params.get("independent_segments", False),
            command_frame_offset_rad=tdcr_params.get("command_frame_offset_rad", 0.0),
            command_mirror_x=tdcr_params.get("command_mirror_x", False),
        )

        # Find Franka actuator IDs
        self._find_franka_actuators()

        # Hold-to-home: while r is held the Franka arm steps toward home at a
        # teleop-comparable joint increment (the combined Franka teleop is
        # task-space, so there is no joint-speed config). The TDCR homes via its
        # own controller (y) at clark_speed_scale. Releasing the key stops it.
        self.franka_home_joint_step = tdcr_params.get("franka_home_joint_step", 0.01)
        # Scales the per-tick task-space step for Franka keyboard teleop. 1.0 is
        # the default feel; lower = slower/finer (e.g. 0.5 = half speed).
        self.franka_speed_scale = tdcr_params.get("franka_speed_scale", 1.0)
        self._home_ctrl = self._read_home_ctrl()
        # Rotation command frame: roll/pitch/yaw are pre-rotated by the fixed
        # franka->TDCR mount before the (unchanged) Jacobian IK, so they line up
        # with the TDCR instead of the EE's skewed body axes. The mount chain has
        # no joints, so this offset is a constant set in the MJCF -- deterministic,
        # no settling, no dependence on the live pose / qpos0.
        self._franka_rot_offset = self._mount_rotation()

        print(f"Combined controller initialized:")
        print(f"  - Franka joints: {len(self.franka_actuator_ids)}")
        print(f"  - TDCR tendons: {len(self.tdcr_controller.tendon_actuator_ids)}")

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

    def _read_home_ctrl(self):
        """Full-actuator home target: the 'pretension' keyframe ctrl, or a
        fallback that puts the Franka block at its home pose."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
        if key_id >= 0:
            return self.model.key_ctrl[key_id].copy()
        home = self.data.ctrl.copy()
        for i, act_id in enumerate(self.franka_actuator_ids):
            home[act_id] = self.franka_controller.home_position[i]
        return home

    def _mount_rotation(self):
        """Fixed rotation from the end-effector frame to the TDCR base frame.

        This is the franka->TDCR mount (the ~45 deg bracket plus the base link's
        orientation). The whole mount chain has no joints, so the relative
        orientation EE->base is constant -- a forward-kinematics pass at *any*
        pose returns the same value. We sample it from a fresh scratch copy so it
        never depends on the live sim state or init timing. Returns identity (no
        offset = original behaviour) if the TDCR base body can't be found.
        """
        ee = self.franka_controller.body_id
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_0")
        if base < 0:
            return Rotation.identity()
        try:
            scratch = mujoco.MjData(self.model)
            mujoco.mj_forward(
                self.model, scratch
            )  # qpos0 is fine: M is pose-independent
            qe = np.asarray(scratch.body(ee).xquat, dtype=float)  # [w,x,y,z]
            qb = np.asarray(scratch.body(base).xquat, dtype=float)
        except Exception:  # noqa: BLE001 - fall back to the live pose
            qe = np.asarray(self.data.body(ee).xquat, dtype=float)
            qb = np.asarray(self.data.body(base).xquat, dtype=float)
        if not (np.any(qe) and np.any(qb)):
            return Rotation.identity()
        r_ee = Rotation.from_quat([qe[1], qe[2], qe[3], qe[0]])
        r_base = Rotation.from_quat([qb[1], qb[2], qb[3], qb[0]])
        return r_ee.inv() * r_base

    def compute_target_qpos(
        self, franka_command: Dict[str, float], tdcr_command: Dict[str, float]
    ) -> np.ndarray:
        """Compute target positions for both robots.

        Args:
            franka_command: Command for Franka (dx, dy, dz, droll, dpitch, dyaw)
            tdcr_command: Command for TDCR (x, y, segment, reset_home)

        Returns:
            Array of target positions for all actuators
        """
        target_qpos = np.zeros(self.model.nu)

        franka_motion = any(
            franka_command.get(k, 0) != 0
            for k in ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
        )

        # ---------- Franka: hold-to-home (r), else IK / hold ----------
        if franka_command.get("reset_home", False):
            cur = np.array([self.data.ctrl[a] for a in self.franka_actuator_ids])
            goal = np.array([self._home_ctrl[a] for a in self.franka_actuator_ids])
            franka_target = step_toward(cur, goal, self.franka_home_joint_step)
            for i, act_id in enumerate(self.franka_actuator_ids):
                target_qpos[act_id] = franka_target[i]
        elif franka_motion:
            # Task-space (IK) control
            current_joints = np.array(
                [self.data.ctrl[a] for a in self.franka_actuator_ids]
            )
            rot_step = 0.01 * self.franka_speed_scale
            pos_step = 0.002 * self.franka_speed_scale
            droll = franka_command.get("droll", 0) * rot_step
            dpitch = franka_command.get("dpitch", 0) * rot_step
            dyaw = franka_command.get("dyaw", 0) * rot_step
            if abs(droll) > 0 or abs(dpitch) > 0 or abs(dyaw) > 0:
                # Pre-rotate the command by the fixed mount so panda_ik (which
                # applies the increment in the EE body frame) rotates about the
                # TDCR's axes rather than the EE's skewed ones. Same Jacobian IK
                # as before, just a rotated command frame (see __init__).
                body_rotvec = self._franka_rot_offset.apply([droll, dpitch, dyaw])
                dquat = Rotation.from_rotvec(body_rotvec).as_quat()
            else:
                dquat = np.array([0, 0, 0, 1])
            panda_action = np.array(
                [
                    franka_command.get("dx", 0) * pos_step,
                    franka_command.get("dy", 0) * pos_step,
                    franka_command.get("dz", 0) * pos_step,
                    dquat[0],
                    dquat[1],
                    dquat[2],
                    dquat[3],
                ]
            )
            target_joints = current_joints + self.franka_controller.panda_ik(
                panda_action
            )
            for i, act_id in enumerate(self.franka_actuator_ids):
                target_qpos[act_id] = target_joints[i]
        else:
            # Hold current Franka position
            for act_id in self.franka_actuator_ids:
                target_qpos[act_id] = self.data.ctrl[act_id]

        # ---------- TDCR: delegate (its controller handles hold-to-home for y) ----------
        tdcr_targets = self.tdcr_controller.compute_target_qpos(tdcr_command, self.data)
        for act_id in self.tdcr_controller.tendon_actuator_ids:
            target_qpos[act_id] = tdcr_targets[act_id]

        return target_qpos

    def reset_to_home(self):
        """Reset both robots to home position."""
        # Reset Franka to home using IK controller's home position
        home_joints = self.franka_controller.home_position
        for i, act_id in enumerate(self.franka_actuator_ids):
            self.data.ctrl[act_id] = home_joints[i]

        # Reset TDCR to home
        self.tdcr_controller.reset_to_home()

    def get_current_positions(self) -> np.ndarray:
        """Get current joint positions for Franka (compatibility method)."""
        current_joints = np.zeros(7)
        for i, act_id in enumerate(self.franka_actuator_ids):
            current_joints[i] = self.data.ctrl[act_id]
        return current_joints

    def set_joint_targets(self, targets: np.ndarray):
        """Set joint targets for Franka (compatibility method)."""
        for i, act_id in enumerate(self.franka_actuator_ids[: len(targets)]):
            self.data.ctrl[act_id] = targets[i]

    def get_info(self) -> Dict[str, Any]:
        """Get combined controller information.

        Returns:
            Dictionary with controller state for both robots
        """
        # Get Franka end-effector pose (fall back to the end-effector body
        # when the scene defines no 'ee_site' site)
        site_id = self.franka_controller.ee_site_id
        if site_id is not None:
            ee_pos = self.data.site_xpos[site_id].copy()
            ee_mat = self.data.site_xmat[site_id].reshape(3, 3)
        else:
            body_id = self.franka_controller.body_id
            ee_pos = self.data.xpos[body_id].copy()
            ee_mat = self.data.xmat[body_id].reshape(3, 3)

        # Get TDCR info
        tdcr_info = self.tdcr_controller.get_info()

        return {
            "franka": {"ee_position": ee_pos, "ee_orientation_matrix": ee_mat},
            "tdcr": tdcr_info,
        }
