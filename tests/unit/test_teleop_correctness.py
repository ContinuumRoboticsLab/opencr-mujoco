#!/usr/bin/env python3
"""Behavioral correctness tests for teleoperation.

The smoke tests in test_teleop.py check that the pieces construct and run;
the tests here check they do the *right thing*:

- a Clark command on segment 1 bends the simulated robot in the commanded
  direction (closed loop through controller -> tendons -> physics)
- the Clark <-> tendon transforms invert each other, and the coupled-segment
  compensation scales with per-segment tendon distances
- the keyboard mappers translate keys to the documented commands, and the
  LSHIFT gating works (including shift_l)
- config merging follows the documented precedence (CLI > config file >
  defaults) and boolean config values survive unset CLI flags
- every shipped teleop config names a known input device + controller, and
  its scene either ships in git or is produced by a shipped generation config

Run nightly alongside test_generator_physics.py.
"""

import json
import math
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

mujoco = pytest.importorskip("mujoco")

from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config  # noqa: E402

TELEOP_CONFIG_DIR = PROJECT_ROOT / "configs" / "teleop"
GEN_CONFIG_DIR = PROJECT_ROOT / "configs" / "generation"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tdcr_model_path(tmp_path_factory):
    """A standalone 3-segment tendon TDCR generated from the shipped example."""
    cfg = json.load(open(GEN_CONFIG_DIR / "example_three_segment_franka.json"))
    if "total_links" not in cfg:
        cfg["total_links"] = sum(cfg["links_per_segment"].values())
    if "total_length" not in cfg:
        cfg["total_length"] = sum(cfg["segment_lengths"].values())
    path = tmp_path_factory.mktemp("tdcr") / "tdcr.xml"
    create_tdcr_from_config(cfg, str(path))
    return str(path)


def settled_controller(model_path):
    """TDCRJointController on a freshly loaded model at pretension."""
    from opencr_mujoco.controllers.tdcr_joint_controller import TDCRJointController

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    controller = TDCRJointController(
        model, data=data, clark_speed_scale=0.001, fps=100.0
    )
    # Settle at pretension
    for _ in range(1000):
        mujoco.mj_step(model, data)
    return model, data, controller


# --------------------------------------------------------------------------- #
# Closed-loop control direction
# --------------------------------------------------------------------------- #
class TestClosedLoopBending:
    def test_clark_x_command_bends_in_segment_one_bend_plane(self, tdcr_model_path):
        """Lock the control convention end-to-end.

        Segment 1's lead tendon sits at angle offset 0 (the +X side).
        +clark_x lengthens it, so the tip bends along the X axis toward -X —
        the convention the hardware sysid calibration and the keyboard
        mapping ('H' = +clark_x) are built on. The bend must be dominantly
        in the X-Z plane (|dy| small).
        """
        model, data, controller = settled_controller(tdcr_model_path)
        ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
        tip_before = data.xpos[ee].copy()

        command = {"x": 1.0, "y": 0.0, "segment": 0, "reset_home": False}
        for _ in range(300):
            data.ctrl[:] = controller.compute_target_qpos(command, data)
            for _ in range(5):
                mujoco.mj_step(model, data)
        for _ in range(2000):  # settle
            mujoco.mj_step(model, data)

        delta = data.xpos[ee] - tip_before
        assert (
            delta[0] < -0.005
        ), f"+clark_x should bend the tip toward -X, moved {delta}"
        assert abs(delta[0]) > 3 * abs(
            delta[1]
        ), f"bend should be dominantly in the X-Z plane, moved {delta}"

    def test_reset_home_returns_toward_straight(self, tdcr_model_path):
        model, data, controller = settled_controller(tdcr_model_path)
        ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
        home_tip = data.xpos[ee].copy()

        bend = {"x": 1.0, "y": 0.0, "segment": 0, "reset_home": False}
        for _ in range(300):
            data.ctrl[:] = controller.compute_target_qpos(bend, data)
            for _ in range(5):
                mujoco.mj_step(model, data)
        bent_offset = np.linalg.norm(data.xpos[ee] - home_tip)
        assert bent_offset > 0.005

        controller.reset_to_home()
        neutral = {"x": 0.0, "y": 0.0, "segment": 0, "reset_home": False}
        for _ in range(50):
            data.ctrl[:] = controller.compute_target_qpos(neutral, data)
            for _ in range(20):
                mujoco.mj_step(model, data)
        home_offset = np.linalg.norm(data.xpos[ee] - home_tip)
        assert home_offset < 0.35 * bent_offset, (
            f"after reset the tip should return toward home "
            f"(bent {bent_offset*1000:.1f} mm, now {home_offset*1000:.1f} mm)"
        )


# --------------------------------------------------------------------------- #
# Clark kinematics
# --------------------------------------------------------------------------- #
class TestClarkKinematics:
    def test_round_trip_uniform(self):
        from opencr_mujoco.tdcr_kinematics import MultiSegmentTDCRKinematics

        k = MultiSegmentTDCRKinematics([3, 3, 3], 4.0, [0, math.pi / 6, math.pi / 3])
        clark = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 1.0])
        assert np.allclose(k.tendons_mm_to_clark(k.clark_to_tendons_mm(clark)), clark)

    def test_coupling_scales_with_tendon_distance(self):
        from opencr_mujoco.tdcr_kinematics import MultiSegmentTDCRKinematics

        # Segment 1 bend of theta=0.5 rad (clark = theta * d1 = 1.5 mm at
        # d1 = 3 mm) must change a segment-2 tendon routed at d2 = 6 mm by
        # theta * d2 = 3.0 mm (its lead tendon is at angle 0).
        k = MultiSegmentTDCRKinematics([3, 3], [3.0, 6.0], [0.0, 0.0])
        tendons = k.clark_to_tendons_mm(np.array([1.5, 0.0, 0.0, 0.0]))
        assert np.isclose(abs(tendons[3]), 3.0, rtol=1e-9)
        # and the round trip must still invert exactly
        clark = np.array([1.5, 0.0, -0.7, 0.4])
        assert np.allclose(k.tendons_mm_to_clark(k.clark_to_tendons_mm(clark)), clark)

    def test_segment_sum_is_zero_for_three_tendons(self):
        from opencr_mujoco.tdcr_kinematics import NTendonSegmentKinematics

        k = NTendonSegmentKinematics(n=3, tendon_distance_mm=4.0)
        deltas = k.clark_to_tendons_mm(np.array([2.0, 1.0]))
        # Symmetric 3-tendon routing: length changes sum to ~0 (inextensible
        # backbone constraint)
        assert abs(np.sum(deltas)) < 1e-9


# --------------------------------------------------------------------------- #
# Keyboard mappers
# --------------------------------------------------------------------------- #
def make_mapper(cls, **kwargs):
    """Construct a pynput-backed mapper, skipping if no input backend exists
    (headless CI without a display)."""
    try:
        return cls(**kwargs)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"keyboard listener unavailable: {e}")


class TestTDCRKeyboardMapper:
    def test_key_to_command_table(self):
        pytest.importorskip("pynput")
        from pynput import keyboard
        from opencr_mujoco.controllers.tdcr_keyboard_input_mapper import (
            TDCRKeyboardInputMapper,
        )

        mapper = make_mapper(TDCRKeyboardInputMapper)
        try:
            shift = keyboard.Key.shift_l

            def cmd(*keys):
                mapper.current_keys = set(keys)
                return mapper.get_command()

            # No shift -> inert
            assert cmd(keyboard.KeyCode.from_char("t"))["y"] == 0.0
            # T/G = +Y/-Y, F/H = -X/+X (up/down/left/right)
            assert cmd(shift, keyboard.KeyCode.from_char("t"))["y"] == 1.0
            assert cmd(shift, keyboard.KeyCode.from_char("g"))["y"] == -1.0
            assert cmd(shift, keyboard.KeyCode.from_char("f"))["x"] == -1.0
            assert cmd(shift, keyboard.KeyCode.from_char("h"))["x"] == 1.0
            # R = hold-to-home
            assert cmd(shift, keyboard.KeyCode.from_char("r"))["reset_home"]
            # Z/X/C/V/B select segments 1-5 (clamped to max_segments)
            for char, seg in zip("zxc", range(3)):
                assert cmd(shift, keyboard.KeyCode.from_char(char))["segment"] == seg
        finally:
            mapper.stop() if hasattr(mapper, "stop") else None

    def test_taskspace_mapper_velocity_keys(self):
        pytest.importorskip("pynput")
        from opencr_mujoco.controllers.tdcr_taskspace_keyboard_mapper import (
            TDCRTaskSpaceKeyboardMapper,
        )

        mapper = make_mapper(TDCRTaskSpaceKeyboardMapper, verbose=False)
        try:
            with mapper._keys_lock:
                mapper.keys_pressed = {"w", "e"}
            command = mapper.get_command()
            assert command["vx"] == 1.0 * mapper.velocity_scale
            assert command["vz"] == -1.0 * mapper.velocity_scale
            assert command["vy"] == 0.0
        finally:
            mapper.stop()


class TestMultiPointTaskSpaceMapper:
    """The ftdcr_taskspace mapper (multipt_keyboard) must stay consistent with
    franka_tdcr_combined: T/G drive Clark X and F/H drive Clark Y (the combined
    command-frame transform baked into the table), insertion lives on N/M, and
    reset is split R = Franka, Y = TDCR."""

    def _mapper(self):
        pytest.importorskip("pynput")
        from unittest import mock
        from opencr_mujoco.controllers.multipt_taskspace_keyboard_mapper import (
            MultiPointTaskSpaceKeyboardMapper,
        )

        # Stub the OS keyboard listener: we drive keys_pressed directly, and the
        # real pynput darwin backend aborts when many listeners are spun up in
        # one test process. The mapper keeps a (mock) .listener for .stop().
        with mock.patch("pynput.keyboard.Listener"):
            mp = MultiPointTaskSpaceKeyboardMapper(velocity_scale=1.0, verbose=False)
        mp.shift_pressed = True
        mp.current_control_point = "seg3"  # a TDCR segment (not base)
        return mp

    def _cmd(self, mp, *keys):
        with mp._keys_lock:
            mp.keys_pressed = set(keys)
        return mp.get_command()

    def test_tfgh_matches_combined_clark_axes(self):
        mp = self._mapper()
        try:
            # T/G -> Clark X, F/H -> Clark Y: the exact output of combined's
            # command_frame_offset_rad(-90) + command_mirror_x transform. Keep
            # these in sync with combined_keyboard_input_mapper / the joint
            # controller if that transform ever changes.
            assert self._cmd(mp, "t")["clark_x"] == 1.0
            assert self._cmd(mp, "g")["clark_x"] == -1.0
            assert self._cmd(mp, "f")["clark_y"] == -1.0
            assert self._cmd(mp, "h")["clark_y"] == 1.0
            # ...and they do not bleed into the other Clark axis
            assert self._cmd(mp, "t")["clark_y"] == 0.0
            assert self._cmd(mp, "f")["clark_x"] == 0.0
        finally:
            mp.stop()

    def test_insertion_on_n_m_not_y(self):
        mp = self._mapper()
        try:
            assert self._cmd(mp, "n")["v_insert"] == 1.0
            assert self._cmd(mp, "m")["v_insert"] == -1.0
            # Y no longer inserts (it is reset-TDCR now)
            assert self._cmd(mp, "y")["v_insert"] == 0.0
        finally:
            mp.stop()

    def test_reset_split_r_franka_y_tdcr(self):
        mp = self._mapper()
        try:
            r = self._cmd(mp, "r")
            assert r["reset_franka"] is True and r["reset_tdcr"] is False
            y = self._cmd(mp, "y")
            assert y["reset_tdcr"] is True and y["reset_franka"] is False
            both = self._cmd(mp, "r", "y")
            assert both["reset_franka"] and both["reset_tdcr"]
        finally:
            mp.stop()


class TestCombinedMapperShiftGating:
    def test_shift_l_is_accepted(self):
        pytest.importorskip("pynput")
        from pynput import keyboard
        from opencr_mujoco.controllers.combined_keyboard_input_mapper import (
            CombinedKeyboardInputMapper,
        )

        mapper = make_mapper(CombinedKeyboardInputMapper)
        try:
            w = keyboard.KeyCode.from_char("w")
            # Without shift: inert
            mapper.current_keys = {w}
            franka_cmd, _ = mapper.get_command()
            assert all(v == 0.0 for v in franka_cmd.values())
            # With LEFT shift specifically (pynput reports shift_l on most
            # platforms): the Franka X command engages
            mapper.current_keys = {keyboard.Key.shift_l, w}
            franka_cmd, _ = mapper.get_command()
            assert franka_cmd["dx"] == 1.0
        finally:
            mapper.stop() if hasattr(mapper, "stop") else None


# --------------------------------------------------------------------------- #
# Config system semantics
# --------------------------------------------------------------------------- #
class TestConfigPrecedence:
    @staticmethod
    def _args(**overrides):
        base = dict(
            config=None,
            list_configs=False,
            save_config=None,
            show_config=False,
        )
        base.update(overrides)
        return Namespace(**base)

    def test_boolean_config_value_survives_unset_cli_flag(self):
        """Regression: store_true defaults used to clobber config booleans.

        Uses a temp config (no shipped config sets a store_const flag true) so
        the test doesn't depend on a specific config existing."""
        import json
        from opencr_mujoco.utils.config_loader import handle_config_args

        tmp = TELEOP_CONFIG_DIR / "_tmp_bool_regression.json"
        tmp.write_text(
            json.dumps(
                {
                    "input_device": "keyboard",
                    "controller": "ik",
                    "enable_gyroscope": True,
                }
            )
        )
        try:
            args = self._args(config="_tmp_bool_regression", enable_gyroscope=None)
            config = handle_config_args(args, "teleop", {})
            assert config["enable_gyroscope"] is True
        finally:
            tmp.unlink(missing_ok=True)

    def test_cli_overrides_config_file(self):
        from opencr_mujoco.utils.config_loader import handle_config_args

        args = self._args(config="tdcr_keyboard", fps=42)
        config = handle_config_args(args, "teleop", {})
        assert config["fps"] == 42

    def test_config_file_overrides_defaults(self):
        from opencr_mujoco.utils.config_loader import handle_config_args

        args = self._args(config="tdcr_keyboard")
        config = handle_config_args(args, "teleop", {"input_device": "dualsense"})
        assert config["input_device"] == "tdcr_keyboard"

    def test_defaults_used_when_unset(self):
        from opencr_mujoco.utils.config_loader import handle_config_args

        args = self._args()
        config = handle_config_args(args, "teleop", {"fps": 77})
        assert config["fps"] == 77


# --------------------------------------------------------------------------- #
# Shipped teleop config inventory
# --------------------------------------------------------------------------- #
KNOWN_DEVICES = {
    "dualsense",
    "keyboard",
    "tdcr_keyboard",
    "combined_keyboard",
    "multipt_keyboard",
}
KNOWN_CONTROLLERS = {
    "joint",
    "ik",
    "tdcr_joint",
    "tdcr_ik",
    "combined",
    "tdcr_multipt",
    "tdcr_multipt_tension",
}


class TestShippedTeleopConfigs:
    @pytest.mark.parametrize(
        "config_path",
        sorted(TELEOP_CONFIG_DIR.glob("*.json")),
        ids=lambda p: p.stem,
    )
    def test_config_is_runnable_as_documented(self, config_path):
        cfg = json.load(open(config_path))

        assert (
            cfg.get("input_device") in KNOWN_DEVICES
        ), f"{config_path.stem}: unknown input_device {cfg.get('input_device')}"
        controller = cfg.get("controller") or cfg.get("control_mode")
        assert (
            cfg.get("controller") is None or cfg["controller"] in KNOWN_CONTROLLERS
        ), f"{config_path.stem}: unknown controller {controller}"

        scene = cfg.get("scene")
        assert scene, f"{config_path.stem}: no scene configured"

        # Either the scene is a tracked base scene, or teleop.py regenerates
        # it on first use via generate.ensure_scene (generated XMLs are not
        # in git) — so every shipped config must be resolvable one way or
        # the other from a clean clone.
        from generate import resolve_scene_source

        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", scene],
                cwd=PROJECT_ROOT,
                capture_output=True,
            ).returncode
            == 0
        )
        derivable = resolve_scene_source(scene) is not None
        assert tracked or derivable, (
            f"{config_path.stem}: scene {scene} is neither tracked in git nor "
            f"derivable from a shipped configs/generation/ config"
        )
