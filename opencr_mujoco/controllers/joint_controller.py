import numpy as np
import mujoco

from .ik_controller import read_franka_home_from_model


class JointController:
    """Joint controller that uses MuJoCo's actuator control system."""

    def __init__(self, model, data, num_joints=7, has_gripper=True):
        """
        Initialize the joint controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            num_joints: Number of robot joints (default: 7 for Franka)
            has_gripper: Whether the robot has a gripper (default: True)
        """
        self.model = model
        self.data = data
        self.num_joints = num_joints
        self.has_gripper = has_gripper

        # Franka home pose: single source of truth is the scene's keyframe (with a
        # constant fallback), sliced to num_joints.
        self.home_positions = read_franka_home_from_model(model)[:num_joints]

        # Initialize target positions
        self.target_positions = data.qpos[:num_joints].copy()

        # Apply initial positions using actuator control
        self._apply_actuator_control(self.target_positions)

        # Update MuJoCo
        mujoco.mj_forward(model, data)

        print(
            f"Joint controller initialized with {num_joints} joints using actuator control"
        )
        if has_gripper:
            print("Gripper enabled - values will be normalized to [0, 0.04]")

    def _apply_actuator_control(self, joint_targets):
        """
        Apply joint targets using MuJoCo's actuator control system.

        Args:
            joint_targets: Array of joint target positions
        """
        # Set actuator controls for position control
        # MuJoCo will handle the control internally
        for i in range(min(len(joint_targets), self.model.nu)):
            if i < len(self.data.ctrl):
                # Use position control - set the target position
                self.data.ctrl[i] = joint_targets[i]

    def set_joint_targets(self, joint_targets):
        """
        Set target joint positions using actuator control.

        Args:
            joint_targets: Array of joint target positions
                          - If has_gripper=True: 8 values (7 joints + 1 gripper)
                          - If has_gripper=False: 7 values (7 joints only)
        """
        if len(joint_targets) != self.num_joints + (1 if self.has_gripper else 0):
            raise ValueError(
                f"Expected {self.num_joints + (1 if self.has_gripper else 0)} values, got {len(joint_targets)}"
            )

        # Update target positions
        self.target_positions[: self.num_joints] = joint_targets[: self.num_joints]

        # Apply using actuator control
        self._apply_actuator_control(self.target_positions)

        # Handle gripper if present
        if self.has_gripper:
            gripper_value = joint_targets[self.num_joints]
            # Clamp to the finger actuators' ctrl range (0 = closed, 0.04 = open)
            gripper_normalized = np.clip(gripper_value, 0.0, 0.04)

            # Drive the two finger position actuators (which follow the 7 arm
            # actuators in data.ctrl). Writing data.qpos directly would be undone
            # by the next mj_step, since the actuators would still command 0.
            if self.num_joints + 2 <= self.model.nu:
                self.data.ctrl[self.num_joints] = gripper_normalized
                self.data.ctrl[self.num_joints + 1] = gripper_normalized

    def apply_control(self):
        """Apply target positions using actuator control."""
        # Apply using actuator control - this works with MuJoCo's control system
        self._apply_actuator_control(self.target_positions)

        # Update MuJoCo
        mujoco.mj_forward(self.model, self.data)

    def get_current_positions(self):
        """Get current joint positions."""
        return self.data.qpos[: self.num_joints].copy()

    def get_target_positions(self):
        """Get current target positions."""
        return self.target_positions.copy()

    def reset_to_home(self):
        """Reset robot to the home pose (from the scene keyframe, set in __init__)."""
        home_positions = self.home_positions.copy()

        # Add gripper value if needed (0.0 = closed)
        if self.has_gripper:
            home_positions = np.append(home_positions, 0.0)

        self.set_joint_targets(home_positions)
        self.apply_control()
