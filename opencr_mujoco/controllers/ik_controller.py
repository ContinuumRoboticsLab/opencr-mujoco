import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

# Canonical Franka "ready" home pose (7 arm joints). This is the fallback used
# only when a scene has no 'pretension' keyframe; the runtime source of truth is
# the keyframe itself (see read_franka_home_from_model). Matches the value baked
# into every generated Franka+TDCR scene's pretension keyframe.
FRANKA_HOME_QPOS = np.array(
    [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
)


def read_franka_home_from_model(model):
    """Return the 7 Franka arm-joint home values for the loaded model.

    Reads the model's 'pretension' keyframe ctrl, mapped by actuator name so it
    works for both 'panda_jointN' (combined scenes) and 'panda0_jointN'
    (standalone scenes). Falls back to FRANKA_HOME_QPOS when no keyframe or
    Franka actuator is present.
    """
    home = FRANKA_HOME_QPOS.copy()
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    if key_id < 0:
        return home
    for i in range(1, 8):
        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"panda_joint{i}"
        )
        if act_id < 0:
            act_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"panda0_joint{i}"
            )
        if act_id >= 0:
            home[i - 1] = float(model.key_ctrl[key_id, act_id])
    return home


class IKController:
    """Inverse Kinematics controller for Franka robot using quaternions."""

    def __init__(self, model, data):
        """
        Initialize the IK controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

        # Initialize the model
        mujoco.mj_forward(model, data)

        # Get the body ID for the end effector
        try:
            # Try the standard end_effector body first
            self.body_id = data.body("end_effector").id
        except KeyError:
            try:
                self.body_id = data.body("panda_link8").id
            except KeyError:
                self.body_id = data.body("panda_link7").id
                print(
                    "Warning: end-effector bodies 'end_effector' and "
                    "'panda_link8' not found. Using 'panda_link7' instead."
                )

        # Store end effector site ID (mj_name2id returns -1 when missing —
        # it does not raise)
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.ee_site_id = ee_site_id if ee_site_id >= 0 else None

        # Home position for Franka (7 joints), read from the scene's keyframe
        # (single source of truth) with a constant fallback.
        self.home_position = read_franka_home_from_model(model)

    def panda_ik(self, panda_action):
        """
        Compute joint increments for desired Cartesian pose increment.

        Args:
            panda_action: [pos_dx, pos_dy, pos_dz, quat_dx, quat_dy, quat_dz, quat_dw]
                         where pos_d* are position increments and quat_d* are quaternion increments

        Returns:
            Joint increments for the 7 Franka joints
        """
        pos_d = panda_action[:3]
        quat_d = panda_action[3:7]  # Quaternion increment, scipy order [x, y, z, w]

        # Current pose (use the body resolved at init, with its fallbacks)
        body = self.data.body(self.body_id)
        xpos = body.xpos
        xquat = body.xquat  # MuJoCo order [w, x, y, z]

        # Check if we have valid poses
        if np.allclose(xpos, 0) or np.allclose(xquat, 0):
            print(
                "Warning: End effector pose is zero. Model may not be properly initialized."
            )
            return np.zeros(7)

        # Compute Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.body_id)
        JAC = np.concatenate((jacp, jacr))  # 6 x num joints

        # Desired position
        xpos_d = xpos + pos_d

        # Desired quaternion (compose current quaternion with increment)
        # Normalize quaternion to ensure it's valid
        xquat_norm = xquat / np.linalg.norm(xquat)
        quat_d_norm = (
            quat_d / np.linalg.norm(quat_d)
            if np.linalg.norm(quat_d) > 0
            else np.array([0, 0, 0, 1])
        )

        # Convert MuJoCo's [w, x, y, z] to scipy's [x, y, z, w]
        current_rot = Rotation.from_quat(
            np.concatenate([xquat_norm[1:], xquat_norm[:1]])
        )
        increment_rot = Rotation.from_quat(quat_d_norm)
        desired_rot = current_rot * increment_rot

        # Compute error
        error = np.zeros(6)
        error[:3] = xpos_d - xpos  # Position error

        # Quaternion error - use scipy for quaternion operations
        rot_diff = desired_rot * current_rot.inv()
        rot_error = rot_diff.as_rotvec()  # Convert to rotation vector
        error[3:] = rot_error

        # Compute joint increments using pseudo-inverse
        grad_coef = 1.0
        grad = grad_coef * np.linalg.pinv(JAC) @ error

        return np.round(grad[:7], 5)
