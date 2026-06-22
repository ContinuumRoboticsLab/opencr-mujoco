#!/usr/bin/env python3
"""Unit tests for teleop.py teleoperation functionality."""

import pytest
import tempfile
from pathlib import Path
import sys
import os
import json
import numpy as np

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mujoco  # noqa: E402
from opencr_mujoco.controllers.joint_controller import JointController  # noqa: E402
from opencr_mujoco.controllers.ik_controller import IKController  # noqa: E402
from opencr_mujoco.utils.keyboard_input_device import KeyboardInputDevice  # noqa: E402


class _FakeKeyboardListener:
    """Test double for pynput's listener; avoids starting OS callback threads."""

    def __init__(self, *args, **kwargs):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class TestTeleop:
    """Test teleoperation functionality."""

    @pytest.fixture(autouse=True)
    def fake_keyboard_listener(self, monkeypatch):
        """Keep keyboard smoke tests deterministic and thread-free."""
        import opencr_mujoco.utils.keyboard_input_device as keyboard_input_device

        if keyboard_input_device.keyboard is not None:
            monkeypatch.setattr(
                keyboard_input_device.keyboard, "Listener", _FakeKeyboardListener
            )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def franka_scene_xml(self):
        """Use the actual Franka scene for testing."""
        scene_path = Path("assets/franka_scene.xml")
        if not scene_path.exists():
            # Fall back to a simpler scene if the full scene isn't available
            pytest.skip("Full Franka scene not available")
        return scene_path

    @pytest.fixture
    def teleop_config(self, temp_dir, franka_scene_xml):
        """Create a teleop configuration file."""
        config = {
            "scene": str(franka_scene_xml),
            "input_device": "keyboard",
            "control_mode": "task_space",
            "controller": "ik",
            "width": 800,
            "height": 600,
            "show_info": True,
            "sim_steps_per_frame": 1,
            "fps": 60,
            "description": "Test teleop configuration",
        }
        config_path = temp_dir / "teleop_config.json"
        config_path.write_text(json.dumps(config))
        return config_path

    def test_joint_controller_initialization(self, franka_scene_xml):
        """Test JointController initialization."""
        model = mujoco.MjModel.from_xml_path(str(franka_scene_xml))
        data = mujoco.MjData(model)

        controller = JointController(model, data)

        assert controller is not None
        assert hasattr(controller, "set_joint_targets")
        assert hasattr(controller, "apply_control")
        assert controller.num_joints >= 1

    def test_joint_controller_compute_target(self, franka_scene_xml):
        """Test JointController target setting."""
        model = mujoco.MjModel.from_xml_path(str(franka_scene_xml))
        data = mujoco.MjData(model)

        controller = JointController(model, data, num_joints=1)

        # Set joint targets
        joint_targets = np.array([0.1, 0.0])  # Joint position and gripper
        controller.set_joint_targets(joint_targets)

        # Get target positions
        target_pos = controller.get_target_positions()

        assert target_pos is not None
        assert len(target_pos) >= 1

    def test_ik_controller_initialization(self, franka_scene_xml):
        """Test IKController initialization."""
        model = mujoco.MjModel.from_xml_path(str(franka_scene_xml))
        data = mujoco.MjData(model)

        # IKController needs end effector site
        controller = IKController(model, data)

        assert controller is not None
        assert hasattr(controller, "panda_ik")

    def test_ik_controller_compute_target(self, franka_scene_xml):
        """Test IKController target computation."""
        model = mujoco.MjModel.from_xml_path(str(franka_scene_xml))
        data = mujoco.MjData(model)

        controller = IKController(model, data)

        # Create Cartesian action for IK
        # [pos_dx, pos_dy, pos_dz, quat_dx, quat_dy, quat_dz, quat_dw]
        panda_action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        # Compute joint increments
        joint_increments = controller.panda_ik(panda_action)

        assert joint_increments is not None
        assert len(joint_increments) == 7  # 7 DOF for Franka

    def test_keyboard_input_device(self):
        """Test KeyboardInputDevice functionality."""
        device = KeyboardInputDevice()
        try:
            assert device is not None
            assert hasattr(device, "update_state")
            assert hasattr(device, "left_joystick_state")
            assert hasattr(device, "right_joystick_state")

            # Test initial state
            assert device.left_joystick_state == [0, 0]  # Neutral
            assert device.right_joystick_state == [0, 0]
        finally:
            device.close()

    def test_keyboard_input_mapping(self):
        """Test keyboard input mapping to robot commands."""
        device = KeyboardInputDevice()
        try:
            # Test that the device has the expected state attributes
            assert hasattr(device, "left_joystick_state")
            assert hasattr(device, "right_joystick_state")

            # Test button states
            assert hasattr(device, "ps_down_state")
            assert hasattr(device, "l1_state")
            assert hasattr(device, "r1_state")

            # Verify initial neutral state
            assert device.left_joystick_state == [0, 0]
            assert device.right_joystick_state == [0, 0]
        finally:
            device.close()

    def test_control_loop_components(self, franka_scene_xml):
        """Test control loop components integration."""
        model = mujoco.MjModel.from_xml_path(str(franka_scene_xml))
        data = mujoco.MjData(model)

        # Test joint controller
        joint_controller = JointController(model, data, num_joints=model.nu)

        # Test setting joint targets
        joint_targets = np.zeros(model.nu + 1)  # joints + gripper
        joint_controller.set_joint_targets(joint_targets)
        target = joint_controller.get_target_positions()
        assert target is not None

        # Test IK controller
        ik_controller = IKController(model, data)

        # Test IK computation
        panda_action = np.zeros(7)  # pos(3) + quat(4)
        panda_action[6] = 1.0  # w component of quaternion
        joint_increments = ik_controller.panda_ik(panda_action)
        assert joint_increments is not None

    def test_scene_info_extraction(self, franka_scene_xml):
        """Test extraction of scene information for teleop."""
        model = mujoco.MjModel.from_xml_path(str(franka_scene_xml))

        # Extract scene info
        info = {
            "num_bodies": model.nbody,
            "num_joints": model.njnt,
            "num_actuators": model.nu,
            "has_gripper": False,
            "num_robot_joints": model.nu,
        }

        # Check for gripper bodies
        left_finger_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "panda0_leftfinger"
        )
        info["has_gripper"] = left_finger_id >= 0

        assert info["num_actuators"] >= 1
        assert info["has_gripper"]  # Our test scene has gripper bodies

    def test_tdcr_scene_detection(self, temp_dir):
        """Test detection of TDCR scenes."""
        tdcr_scene_content = """
        <mujoco>
            <worldbody>
                <body name="tdcr_base">
                    <joint name="joint_0" type="hinge"/>
                    <geom type="cylinder" size="0.01 0.05"/>
                </body>
            </worldbody>
            <actuator>
                <position name="seg_0_ten_0" joint="joint_0" kp="100"/>
                <position name="seg_0_ten_1" joint="joint_0" kp="100"/>
                <position name="seg_0_ten_2" joint="joint_0" kp="100"/>
            </actuator>
        </mujoco>
        """

        tdcr_path = temp_dir / "ftdcr_scene.xml"
        tdcr_path.write_text(tdcr_scene_content)

        model = mujoco.MjModel.from_xml_path(str(tdcr_path))

        # Check for TDCR actuators
        is_tdcr = False
        num_segments = 0

        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name and "seg_" in actuator_name and "_ten_" in actuator_name:
                is_tdcr = True
                # Extract segment number
                seg_num = int(actuator_name.split("_")[1])
                num_segments = max(num_segments, seg_num + 1)

        assert is_tdcr
        assert num_segments == 1

    def test_control_mode_switching(self, franka_scene_xml):
        """Test switching between different control modes."""
        model = mujoco.MjModel.from_xml_path(str(franka_scene_xml))
        data = mujoco.MjData(model)

        # Test joint control mode
        joint_controller = JointController(model, data, num_joints=model.nu)
        joint_targets = np.zeros(model.nu + 1)
        joint_controller.set_joint_targets(joint_targets)
        joint_target = joint_controller.get_target_positions()

        # Test IK control mode
        ik_controller = IKController(model, data)
        panda_action = np.zeros(7)
        panda_action[6] = 1.0  # w component
        ik_result = ik_controller.panda_ik(panda_action)

        # Both should produce valid results
        assert joint_target is not None
        assert ik_result is not None

    def test_simulation_rate_control(self, franka_scene_xml):
        """Test simulation rate and frame skip."""
        model = mujoco.MjModel.from_xml_path(str(franka_scene_xml))
        data = mujoco.MjData(model)

        initial_time = data.time
        sim_steps_per_frame = 5

        # Simulate frame skip
        for _ in range(sim_steps_per_frame):
            mujoco.mj_step(model, data)

        # Check time advancement
        expected_time = initial_time + sim_steps_per_frame * model.opt.timestep
        assert data.time == pytest.approx(expected_time)

    def test_tdcr_controller_independent_segments(self):
        """Test TDCRJointController with independent segments mode."""
        from opencr_mujoco.controllers.tdcr_joint_controller import TDCRJointController
        from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config
        import tempfile

        config = {
            "description": "Test TDCR with independent segments",
            "num_segments": 3,
            "links_per_segment": {"1": 5, "2": 5, "3": 5},
            "segment_lengths": {"1": 0.1, "2": 0.1, "3": 0.1},
            "total_links": 15,
            "total_length": 0.3,
            "radius": 0.006,
            "mass": 0.2,
            "joints_per_link": 2,
            "joint_config_mode": "direct",
            "stiffness": 100.0,
            "damping": 1.0,
            "actuation_mode": "parallel_tendons",
            "independent_segments": True,
            "actuation_details": {
                "segments": [
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                ]
            },
            "actuator_properties": {
                "tendon_ctrlrange": "-0.05 0.05",
                "tendon_actuator_type": "position",
                "tendon_kp": 10000,
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            temp_path = f.name

        try:
            # Generate model
            create_tdcr_from_config(config, temp_path)

            # Load model
            model = mujoco.MjModel.from_xml_path(temp_path)
            data = mujoco.MjData(model)

            # Initialize TDCR controller with independent segments
            controller = TDCRJointController(
                model,
                data=data,
                tendon_distance_mm=4.0,
                clark_speed_scale=0.005,
                independent_segments=True,
                fps=100,
            )

            # Verify controller was initialized correctly
            assert controller.independent_segments is True
            assert controller.n_segments == 3

            # Test computing target positions
            command = {"x": 1.0, "y": 0.0, "segment": 0, "reset_home": False}
            target_qpos = controller.compute_target_qpos(command, data)

            # Should return array with correct size
            assert target_qpos.shape == (model.nu,)

            # Test that controller info shows independent mode
            info = controller.get_info()
            assert "Independent Segments" in info["type"]

        finally:
            # Clean up temp file
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
