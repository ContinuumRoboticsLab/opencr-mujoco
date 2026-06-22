"""Browser teleop runtime: builds the REAL controllers and drives them per frame.

This mirrors teleop.py's controller/input setup and control loop, but:
  * the MuJoCo model/data come from mujoco-js via the `mujoco` shim, and
  * keyboard state is injected from JavaScript (no pynput OS listener).

The JS bridge sets ``globalThis.__MJ`` (model/data/engine), then calls
``make_runtime(...)`` once per scene and ``runtime.step()`` every control tick.
"""

import time

import numpy as np
import mujoco  # the in-browser shim
from pynput import keyboard as _kb  # the in-browser stub


# --------------------------------------------------------------------------
# Scene info (subset of teleop.get_scene_info, model-based)
# --------------------------------------------------------------------------


def _scene_info(model):
    actuator = mujoco.mjtObj.mjOBJ_ACTUATOR

    def has_act(name):
        return mujoco.mj_name2id(model, actuator, name) >= 0

    info = {"has_gripper": False, "num_robot_joints": 7, "is_tdcr": False}
    n_seg = 0
    while has_act(f"seg_{n_seg}_ten_0"):
        n_seg += 1
    if n_seg > 0:
        info["is_tdcr"] = True
        info["num_robot_joints"] = n_seg * 3
    else:
        names = [mujoco.mj_id2name(model, actuator, i) or "" for i in range(model.nu)]
        info["has_gripper"] = any(("finger" in n) or ("gripper" in n) for n in names)
    return info


def _pretension_key(model):
    for i in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "pretension":
            return i
    return -1


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


class Runtime:
    """Holds a controller + input mapper and applies it to data.ctrl per tick."""

    def __init__(self, controller_type, input_device_type, controller_params, fps):
        ctx = mujoco.context()
        self.model = mujoco.MjModel(ctx.model)
        self.data = mujoco.MjData._wrap(self.model, ctx.data)
        self.controller_type = controller_type
        self.input_device_type = input_device_type
        self.params = controller_params or {}
        self.fps = fps or 100
        self.unsupported = None

        # Start every scene from its home keyframe + let it settle, like teleop.
        key = _pretension_key(self.model)
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        self._discover_props()  # interactive knock-around props (if any)
        self.scene_info = _scene_info(self.model)
        self.device = None  # object whose key-state we inject
        self.input_mapper = None  # object we query for commands
        self.controller = None
        # Chars that send the robot home for this controller (combined uses both
        # the Franka 'r' and the TDCR 'y'); set in _build.
        self.home_chars = {"r"}
        self._raw_keys = set()
        self._home_ticks = 0  # >0 while a one-press "go home" is latched
        self._build()

    # ---- construction (mirrors teleop.py dispatch) ----
    def _build(self):
        ct = self.controller_type
        p = self.params
        m, d = self.model, self.data

        if self.input_device_type == "dualsense":
            self.unsupported = "DualSense controller isn't available in the browser."
            return

        if ct == "tdcr_joint":
            from opencr_mujoco.controllers.tdcr_joint_controller import TDCRJointController
            from opencr_mujoco.controllers.tdcr_keyboard_input_mapper import (
                TDCRKeyboardInputMapper,
            )

            self.controller = TDCRJointController(
                m,
                data=d,
                tendon_distance_mm=p.get("tendon_distance_mm", 4.0),
                angle_offset_rad_ccw=p.get("angle_offset_rad_ccw"),
                clark_speed_scale=p.get("clark_speed_scale", 0.001),
                fps=self.fps,
                tension_mode=p.get("tension_mode", False),
                independent_segments=p.get("independent_segments", False),
                command_frame_offset_rad=p.get("command_frame_offset_rad", 0.0),
                command_mirror_x=p.get("command_mirror_x", False),
            )
            self.device = self.input_mapper = TDCRKeyboardInputMapper()

        elif ct == "combined":
            from opencr_mujoco.controllers.combined_controller import CombinedController
            from opencr_mujoco.controllers.combined_keyboard_input_mapper import (
                CombinedKeyboardInputMapper,
            )

            tdcr_params = dict(p.get("tdcr", {}))
            tdcr_params["fps"] = self.fps
            self.controller = CombinedController(m, d, tdcr_params)
            self.device = self.input_mapper = CombinedKeyboardInputMapper()
            self.home_chars = {"r", "y"}  # 'r' homes Franka, 'y' homes TDCR

        elif ct == "tdcr_ik":
            from opencr_mujoco.controllers.tdcr_ik_controller import TDCRIKController
            from opencr_mujoco.controllers.tdcr_taskspace_keyboard_mapper import (
                TDCRTaskSpaceKeyboardMapper,
            )

            self.controller = TDCRIKController(
                m,
                d,
                tendon_distance_mm=p.get("tendon_distance_mm", 4.0),
                angle_offset_rad_ccw=p.get("angle_offset_rad_ccw"),
                velocity_scale=p.get("velocity_scale", 0.5),
                damping_factor=p.get("damping_factor", 0.01),
                fps=self.fps,
                verbose=False,
            )
            self.device = self.input_mapper = TDCRTaskSpaceKeyboardMapper(
                velocity_scale=1.0, verbose=False
            )

        elif ct == "tdcr_multipt":
            from opencr_mujoco.controllers.tdcr_multipt_taskspace_controller import (
                TDCRMultiPointTaskSpaceController,
            )
            from opencr_mujoco.controllers.multipt_taskspace_keyboard_mapper import (
                MultiPointTaskSpaceKeyboardMapper,
            )

            self.controller = TDCRMultiPointTaskSpaceController(
                m,
                d,
                tendon_distance_mm=p.get("tendon_distance_mm", 4.0),
                angle_offset_rad_ccw=p.get("angle_offset_rad_ccw"),
                fps=self.fps,
                verbose=False,
            )
            self.device = self.input_mapper = MultiPointTaskSpaceKeyboardMapper()

        elif ct == "tdcr_multipt_tension":
            from opencr_mujoco.controllers.tdcr_multipt_tension_controller import (
                TDCRMultiPointTensionController,
            )
            from opencr_mujoco.controllers.multipt_taskspace_keyboard_mapper import (
                MultiPointTaskSpaceKeyboardMapper,
            )

            self.controller = TDCRMultiPointTensionController(
                m,
                d,
                tendon_distance_mm=p.get("tendon_distance_mm", 4.0),
                angle_offset_rad_ccw=p.get("angle_offset_rad_ccw"),
                fps=self.fps,
                verbose=False,
            )
            self.device = self.input_mapper = MultiPointTaskSpaceKeyboardMapper()

        elif ct in ("ik", "joint"):
            from opencr_mujoco.utils.keyboard_input_device import KeyboardInputDevice
            from opencr_mujoco.controllers.joint_controller import JointController
            from opencr_mujoco.controllers.ik_controller import IKController
            from opencr_mujoco.controllers.dualsense_input_mapper import InputMapper

            info = self.scene_info
            self.controller = JointController(
                m,
                d,
                num_joints=info["num_robot_joints"],
                has_gripper=info["has_gripper"],
            )
            ik = IKController(m, d) if ct == "ik" else None
            mode = (
                InputMapper.TASK_SPACE_POSE
                if ct == "ik"
                else InputMapper.DIRECT_JOINT_CONTROL
            )
            self.device = KeyboardInputDevice()
            self.input_mapper = InputMapper(
                dualsense_device=self.device,
                control_mode=mode,
                num_joints=info["num_robot_joints"],
                has_gripper=info["has_gripper"],
                ik_controller=ik,
                fps=self.fps,
            )
            self.home_chars = {"h"}  # KeyboardInputDevice homes on 'h'
        else:
            self.unsupported = (
                f"Controller '{ct}' is not supported in the browser demo."
            )

    HOME_TICKS = 240  # frames a single "go home" press keeps driving toward home

    # ---- interactive props (auto-respawn when knocked off) ----
    RESET_DELAY_S = 3.0  # let a knocked-off prop fall/roll this long before respawn

    def _discover_props(self):
        """Find prop_<i> bodies (injected by build_site) and remember where
        they rest, so a prop knocked off the pedestal can be teleported back."""
        self._props = []  # (body_id, qpos_adr, dof_adr, spawn_qpos7)
        self._prop_reset_z = 0.30  # below the pedestal top -> respawn
        self._fell_at = {}  # body_id -> monotonic time it dropped below
        for i in range(64):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"prop_{i}")
            if bid < 0:
                break
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"prop_{i}_free"
            )
            if jid < 0:
                continue
            qadr = int(self.model.jnt_qposadr[jid])
            dadr = int(self.model.jnt_dofadr[jid])
            spawn = np.array(self.data.qpos[qadr : qadr + 7], dtype=float)
            self._props.append((bid, qadr, dadr, spawn))

    def _reset_fallen_props(self):
        """Respawn a prop that has been off the pedestal for RESET_DELAY_S, so
        it gets to fall/roll on the floor first instead of teleporting the
        instant it's knocked off."""
        if not getattr(self, "_props", None):
            return
        now = time.monotonic()
        xpos = self.data.xpos  # flat (nbody*3); a body's z is index 3*bid+2
        q = self.data.qpos  # write-through mirrors
        v = self.data.qvel
        for bid, qadr, dadr, spawn in self._props:
            if xpos[bid * 3 + 2] >= self._prop_reset_z:
                self._fell_at.pop(bid, None)  # on the table -> cancel pending reset
                continue
            t0 = self._fell_at.get(bid)
            if t0 is None:
                self._fell_at[bid] = now  # just fell -> start the delay
            elif now - t0 >= self.RESET_DELAY_S:
                q[qadr : qadr + 7] = spawn  # delay elapsed -> respawn
                v[dadr : dadr + 6] = 0.0
                self._fell_at.pop(bid, None)

    # ---- key state from JS (no modifier required) ----
    def set_keys(self, keys, shift=True):
        """Record the currently pressed letter keys. A press of any home char
        latches a one-press return-to-home; pressing a movement key cancels it."""
        try:
            keyset = set(str(k) for k in keys)
        except Exception:  # noqa: BLE001
            keyset = set()
        if keyset & self.home_chars:
            self._home_ticks = self.HOME_TICKS  # latch homing
        elif keyset - self.home_chars:
            self._home_ticks = 0  # any movement key cancels
        self._raw_keys = keyset

    def _inject(self, keyset):
        """Push the effective key set into the active device/mapper. Shift is
        always asserted so the (LSHIFT-gated) mappers act on plain keys."""
        dev = self.device
        if dev is None:
            return
        if hasattr(dev, "current_keys"):
            cur = {_kb.KeyCode.from_char(c) for c in keyset}
            cur.add(_kb.Key.shift_l)
            cur.add(_kb.Key.shift)
            cur.add(_kb.Key.shift_r)
            dev.current_keys = cur
        elif hasattr(dev, "keys_pressed"):
            dev.keys_pressed = set(keyset)
            if hasattr(dev, "shift_pressed"):
                dev.shift_pressed = True
        elif hasattr(dev, "pressed_keys"):
            dev.pressed_keys = set(keyset)
            if hasattr(dev, "shift_pressed"):
                dev.shift_pressed = True

    # ---- per control tick (mirrors teleop.control_loop) ----
    def step(self):
        self._reset_fallen_props()  # respawn anything knocked off the pedestal
        if self.unsupported or self.controller is None:
            return
        # While homing is latched, drive the home chars regardless of held keys.
        if self._home_ticks > 0:
            self._home_ticks -= 1
            self._inject(set(self.home_chars))
        else:
            self._inject(self._raw_keys)

        ct = self.controller_type
        d = self.data
        # A stray key can produce an out-of-range command (e.g. selecting a
        # segment a shorter robot doesn't have); ignore the frame rather than
        # letting it crash the controller.
        try:
            if ct == "combined":
                franka_cmd, tdcr_cmd = self.input_mapper.get_command()
                d.ctrl[:] = self.controller.compute_target_qpos(franka_cmd, tdcr_cmd)
            elif ct in (
                "tdcr_joint",
                "tdcr_ik",
                "tdcr_multipt",
                "tdcr_multipt_tension",
            ):
                command = self.input_mapper.get_command()
                d.ctrl[:] = self.controller.compute_target_qpos(command, d)
            else:  # ik / joint
                if hasattr(self.device, "update_state"):
                    self.device.update_state()
                targets = self.input_mapper.read_controller_inputs()
                self.controller.set_joint_targets(targets)
        except Exception:  # noqa: BLE001 - keep the demo running
            pass


def make_runtime(controller_type, input_device_type, controller_params, fps):
    """Entry point called from JS. controller_params is a plain dict (from JSON)."""
    return Runtime(controller_type, input_device_type, controller_params, fps)
