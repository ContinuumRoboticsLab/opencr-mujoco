import numpy as np

from .ik_controller import FRANKA_HOME_QPOS
from .homing import step_toward


class InputMapper:
    """Maps DualSense controller inputs to robot joint positions with different control modes."""

    # Control modes
    DIRECT_JOINT_CONTROL = "direct_joint"
    TASK_SPACE_POSE = "task_space_pose"
    GYROSCOPE_ORIENTATION = (
        "gyroscope_orientation"  # New mode for gyroscope-based control
    )
    GYROSCOPE_TASK_POSE = (
        "gyroscope_task_pose"  # New mode for gyroscope-based task-space control
    )

    def __init__(
        self,
        dualsense_device=None,
        control_mode=DIRECT_JOINT_CONTROL,
        num_joints=7,
        has_gripper=True,
        ik_controller=None,
        fps=100,
    ):
        """
        Initialize the input mapper.

        Args:
            dualsense_device: DualSenseInputDevice instance (optional)
            control_mode: Control mode ("direct_joint", "task_space_pose", or "gyroscope_orientation")
            num_joints: Number of robot joints (default: 7 for Franka)
            has_gripper: Whether robot has gripper (default: True)
            ik_controller: IKController instance for task-space control (optional)
        """
        self.dualsense = dualsense_device
        self.available = dualsense_device is not None
        self.control_mode = control_mode
        self.num_joints = num_joints
        self.has_gripper = has_gripper
        self.ik_controller = ik_controller

        # Control parameters - easily adjustable by users
        self.position_speed = 0.01  # rad/s for position movements
        self.rotation_speed = 0.005  # rad/s for rotation movements
        self.task_space_position_speed = 0.1 / fps  # m/s for task-space position
        self.task_space_rotation_speed = 0.2 / fps  # rad/s for task-space rotation
        self.joystick_deadzone = 0.05  # Deadzone for joystick drift

        # Gyroscope control parameters
        self.gyro_sensitivity = 0.5 / fps  # Sensitivity for gyroscope data
        self.gyro_deadzone = 0.1  # Deadzone for gyroscope noise
        self.gyro_position_speed = (
            0.1 / fps
        )  # rad/s for position movements with gyroscope mode

        # Gyroscope absolute orientation tracking
        self.gyro_reference = None  # Initial gyroscope reference (0,0,0)
        self.gyro_initialized = False  # Whether gyroscope reference has been set

        # Initialize desired joint positions
        self.desired_positions = np.zeros(num_joints)
        self.joint_increment = np.zeros(num_joints)

        # Home position for Franka. Overridden at runtime via set_home_positions
        # (the scene keyframe is the single source of truth); this is the fallback.
        self.home_positions = (
            FRANKA_HOME_QPOS.copy()
            if num_joints == len(FRANKA_HOME_QPOS)
            else np.zeros(num_joints)
        )

        # Gripper state
        self.gripper_value = 0.0

        print(f"Input mapper initialized with {control_mode} mode")
        if self.available:
            print("DualSense controller connected")
        else:
            print("No DualSense controller - using neutral inputs")

        if self.control_mode == self.TASK_SPACE_POSE and self.ik_controller is None:
            print("Warning: Task-space mode selected but no IK controller provided")

    def read_controller_inputs(self):
        """
        Read controller inputs and return desired joint positions.

        Returns:
            joint_positions: Array of desired joint positions (7 or 8 with gripper)
        """
        if not self.available or self.dualsense is None:
            return self._get_neutral_joint_positions()

        try:
            # Read raw controller inputs
            inputs = self._read_raw_inputs()

            # Hold-to-home: while the home button is held, step the desired pose
            # toward home at the teleop joint speed (position_speed). Releasing it
            # stops the motion, like any other teleop key.
            if self.should_reset_to_home(inputs):
                self.desired_positions = step_toward(
                    self.desired_positions, self.home_positions, self.position_speed
                )
                self.gripper_value = 0.0  # Close gripper
                if self.available and self.dualsense is not None:
                    self.set_gyroscope_reference_from_home()
                return self._get_joint_positions_with_gripper()

            # Update desired positions based on control mode
            if self.control_mode == self.DIRECT_JOINT_CONTROL:
                self._update_direct_joint_control(inputs)
            elif self.control_mode == self.TASK_SPACE_POSE:
                self._update_task_space_pose(inputs)
            elif self.control_mode in (
                self.GYROSCOPE_ORIENTATION,
                self.GYROSCOPE_TASK_POSE,
            ):
                # Both gyro modes use the gyro-augmented task-pose handler
                self._update_gyroscope_task_pose_control(inputs)
            else:
                print(f"Unknown control mode: {self.control_mode}")
                return self._get_neutral_joint_positions()

            return self._get_joint_positions_with_gripper()

        except Exception as e:
            print(f"Error reading controller: {e}")
            return self._get_neutral_joint_positions()

    def _read_raw_inputs(self):
        """Read raw inputs from DualSense controller."""
        # Read joystick values
        left_joystick = np.array(self.dualsense.left_joystick_state)
        right_joystick = np.array(self.dualsense.right_joystick_state)

        # Normalize joystick values to [-1, 1] (pydualsense reports -128..127)
        left_x = (left_joystick[0]) / 128.0
        left_y = (left_joystick[1]) / 128.0
        right_x = (right_joystick[0]) / 128.0
        right_y = (right_joystick[1]) / 128.0

        # Apply deadzone
        if abs(left_x) < self.joystick_deadzone:
            left_x = 0.0
        if abs(left_y) < self.joystick_deadzone:
            left_y = 0.0
        if abs(right_x) < self.joystick_deadzone:
            right_x = 0.0
        if abs(right_y) < self.joystick_deadzone:
            right_y = 0.0

        # Read button states
        triangle = self.dualsense.triangle_down_state
        cross = self.dualsense.cross_down_state
        square = self.dualsense.square_down_state
        circle = self.dualsense.circle_down_state
        dpad_up = self.dualsense.dpad_up_state
        dpad_down = self.dualsense.dpad_down_state
        dpad_left = self.dualsense.dpad_left_state
        dpad_right = self.dualsense.dpad_right_state
        l1 = self.dualsense.l1_state
        r1 = self.dualsense.r1_state
        ps = self.dualsense.ps_down_state

        # Read gyroscope and accelerometer data
        gyro_data = np.array(self.dualsense.gyro_state)
        accel_data = np.array(self.dualsense.accel_state)

        return {
            "left_joystick": (left_x, left_y),
            "right_joystick": (right_x, right_y),
            "triangle": triangle,
            "cross": cross,
            "square": square,
            "circle": circle,
            "dpad_up": dpad_up,
            "dpad_down": dpad_down,
            "dpad_left": dpad_left,
            "dpad_right": dpad_right,
            "l1": l1,
            "r1": r1,
            "ps": ps,
            "gyro": gyro_data,
            "accel": accel_data,
        }

    def _update_direct_joint_control(self, inputs):
        """
        Update desired positions using direct joint control mapping:
        - Left side: Position control (X, Y, Z)
        - Right side: Rotation control (wrist joints)
        - D-pad Left/Right: Gripper control
        """
        left_x, left_y = inputs["left_joystick"]
        right_x, right_y = inputs["right_joystick"]

        # Reset joint increment
        self.joint_increment = np.zeros(self.num_joints)

        # LEFT SIDE - Position Control (X, Y, Z)
        # Left joystick X: Base rotation (joint 1) - X position
        if abs(left_x) > self.joystick_deadzone:
            self.joint_increment[0] = left_x * self.position_speed

        # Left joystick Y: Shoulder joint (joint 2) - Y position
        if abs(left_y) > self.joystick_deadzone:
            self.joint_increment[1] = left_y * self.position_speed

        # D-pad Up/Down: Elbow joint (joint 3) - Z position
        if inputs["dpad_up"]:
            self.joint_increment[2] = self.position_speed  # Elbow up
        if inputs["dpad_down"]:
            self.joint_increment[2] = -self.position_speed  # Elbow down

        # RIGHT SIDE - Rotation Control (wrist joints)
        # Right joystick X: Wrist 1 rotation (joint 5)
        if abs(right_x) > self.joystick_deadzone:
            self.joint_increment[4] = right_x * self.rotation_speed

        # Right joystick Y: Wrist 2 rotation (joint 6)
        if abs(right_y) > self.joystick_deadzone:
            self.joint_increment[5] = right_y * self.rotation_speed

        # Additional rotation controls
        # Triangle/Cross: Wrist 3 rotation (joint 7) - roll
        if inputs["triangle"]:
            self.joint_increment[6] = self.rotation_speed  # Wrist 3 up
        if inputs["cross"]:
            self.joint_increment[6] = -self.rotation_speed  # Wrist 3 down

        # Square/Circle: Joint 4 rotation (forearm) - yaw
        if inputs["square"]:
            self.joint_increment[3] = self.rotation_speed  # Forearm left
        if inputs["circle"]:
            self.joint_increment[3] = -self.rotation_speed  # Forearm right

        # GRIPPER CONTROL
        # Left/Right keypad for gripper
        if inputs["dpad_left"]:
            self.gripper_value = max(0.0, self.gripper_value - 0.002)  # Close gripper
        if inputs["dpad_right"]:
            self.gripper_value = min(0.04, self.gripper_value + 0.002)  # Open gripper

        # Apply increments to desired positions
        self.desired_positions += self.joint_increment

    def _update_task_space_pose(self, inputs):
        """
        Update desired positions using task-space pose increment control with IK.
        - Left side: Position control (joystick for plane, up/down for vertical)
        - Right side: Rotation control using quaternions
        - Left/Right keypad: Gripper control
        """
        if self.ik_controller is None:
            print(
                "Warning: No IK controller available, falling back to direct joint control"
            )
            self._update_direct_joint_control(inputs)
            return

        left_x, left_y = inputs["left_joystick"]
        right_x, right_y = inputs["right_joystick"]

        # Initialize pose increment [dx, dy, dz, dqx, dqy, dqz, dqw]
        pose_increment = np.zeros(7)

        # LEFT SIDE - Position Control
        # Left joystick X: X position (left/right)
        if abs(left_x) > self.joystick_deadzone:
            pose_increment[1] = -left_x * self.task_space_position_speed

        # Left joystick Y: Y position (forward/backward)
        if abs(left_y) > self.joystick_deadzone:
            pose_increment[0] = -left_y * self.task_space_position_speed

        # D-pad Up/Down: Z position (up/down)
        if inputs["dpad_up"]:
            pose_increment[2] = self.task_space_position_speed  # Move up
        if inputs["dpad_down"]:
            pose_increment[2] = -self.task_space_position_speed  # Move down

        # RIGHT SIDE - Rotation Control using Quaternions
        # Create rotation increment from controller inputs
        rotation_increment = self._compute_quaternion_increment(inputs)
        pose_increment[3:7] = rotation_increment

        # GRIPPER CONTROL
        # Left/Right keypad for gripper
        if inputs["dpad_left"]:
            self.gripper_value = max(0.0, self.gripper_value - 0.002)  # Close gripper
        if inputs["dpad_right"]:
            self.gripper_value = min(0.04, self.gripper_value + 0.002)  # Open gripper

        # Use IK controller to compute joint increments
        try:
            joint_increments = self.ik_controller.panda_ik(pose_increment)

            # Apply joint increments to desired positions
            self.desired_positions += joint_increments

        except Exception as e:
            print(f"Error in IK computation: {e}")
            # Fall back to direct joint control
            self._update_direct_joint_control(inputs)

    def _compute_quaternion_increment(self, inputs):
        """
        Compute quaternion increment from controller inputs.

        Args:
            inputs: Dictionary of controller inputs

        Returns:
            quaternion_increment: [qx, qy, qz, qw] quaternion increment
        """
        right_x, right_y = inputs["right_joystick"]

        # Initialize rotation angles (in radians)
        roll_angle = 0.0
        pitch_angle = 0.0
        yaw_angle = 0.0

        # Right joystick X: Roll rotation (around X axis)
        if abs(right_x) > self.joystick_deadzone:
            roll_angle = right_x * self.task_space_rotation_speed

        # Right joystick Y: Pitch rotation (around Y axis)
        if abs(right_y) > self.joystick_deadzone:
            pitch_angle = right_y * self.task_space_rotation_speed

        # Triangle/Cross: Yaw rotation (around Z axis)
        if inputs["triangle"]:
            yaw_angle = self.task_space_rotation_speed  # Yaw left
        if inputs["cross"]:
            yaw_angle = -self.task_space_rotation_speed  # Yaw right

        # Convert Euler angles to quaternion increment
        # Use small angle approximation for quaternion increment
        if abs(roll_angle) < 1e-6 and abs(pitch_angle) < 1e-6 and abs(yaw_angle) < 1e-6:
            # No rotation, return identity quaternion
            return np.array([0.0, 0.0, 0.0, 1.0])

        return self.rpy_to_quaternion(roll_angle, pitch_angle, yaw_angle)

    def rpy_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion.

        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians

        Returns:
            quaternion: [qx, qy, qz, qw] quaternion
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)

        return np.array([qx, qy, qz, qw])

    def _update_gyroscope_task_pose_control(self, inputs):
        """
        Update desired positions using gyroscope for task-space orientation control with IK:
        - Gyroscope: Task-space orientation control (roll, pitch, yaw) from gravity vector
        - Buttons/Joysticks: Position control (X, Y, Z)
        - D-pad Left/Right: Gripper control
        """

        left_x, left_y = inputs["left_joystick"]
        right_x, right_y = inputs["right_joystick"]

        # Initialize pose increment [dx, dy, dz, dqx, dqy, dqz, dqw]
        pose_increment = np.zeros(7)

        # POSITION CONTROL (using buttons and joysticks)
        # Left joystick X: X position (left/right)
        if abs(left_x) > self.joystick_deadzone:
            pose_increment[1] = -left_x * self.task_space_position_speed

        # Left joystick Y: Y position (forward/backward)
        if abs(left_y) > self.joystick_deadzone:
            pose_increment[0] = -left_y * self.task_space_position_speed

        # D-pad Up/Down: Z position (up/down)
        if inputs["dpad_up"]:
            pose_increment[2] = self.task_space_position_speed  # Move up
        if inputs["dpad_down"]:
            pose_increment[2] = -self.task_space_position_speed  # Move down

        # GYROSCOPE TASK-SPACE ORIENTATION CONTROL
        # Gyroscope data is absolute gravity vector in range [-10000, 10000]
        gyro_x, gyro_y, gyro_z = inputs["gyro"]

        # Get relative orientation from gyroscope reference (home position)
        relative_gyro = self.get_gyroscope_relative_orientation(
            [gyro_x, gyro_y, gyro_z]
        )
        # Suppress sensor noise around the reference orientation
        relative_gyro = np.where(
            np.abs(relative_gyro) < self.gyro_deadzone, 0.0, relative_gyro
        )
        gyro_roll, gyro_pitch, gyro_yaw = relative_gyro * self.gyro_sensitivity

        # Compute orientation increment from gyroscope data
        orientation_increment = self.rpy_to_quaternion(gyro_roll, gyro_pitch, gyro_yaw)
        pose_increment[3:7] = orientation_increment

        # GRIPPER CONTROL
        # Left/Right keypad for gripper
        if inputs["dpad_left"]:
            self.gripper_value = max(0.0, self.gripper_value - 0.002)  # Close gripper
        if inputs["dpad_right"]:
            self.gripper_value = min(0.04, self.gripper_value + 0.002)  # Open gripper

        # Use IK controller to compute joint increments
        joint_increments = self.ik_controller.panda_ik(pose_increment)

        # Apply joint increments to desired positions
        self.desired_positions += joint_increments

    def _get_joint_positions_with_gripper(self):
        """Return joint positions with gripper value if needed."""
        if self.has_gripper:
            return np.append(self.desired_positions, self.gripper_value)
        else:
            return self.desired_positions.copy()

    def _get_neutral_joint_positions(self):
        """Return current desired positions when no controller is available."""
        return self._get_joint_positions_with_gripper()

    def should_reset_to_home(self, inputs):
        """Check if robot should reset to home position."""
        return inputs["ps"]

    def set_current_positions(self, current_positions):
        """
        Set the current desired positions (e.g., from robot's actual position).

        Args:
            current_positions: Array of current joint positions (with optional gripper value)
        """
        if len(current_positions) >= self.num_joints:
            self.desired_positions = current_positions[: self.num_joints].copy()

            # If gripper value is provided, set it
            if self.has_gripper and len(current_positions) > self.num_joints:
                self.gripper_value = current_positions[self.num_joints]
        else:
            print(
                f"Warning: Expected {self.num_joints} joint positions, got {len(current_positions)}"
            )

    def set_current_gripper_value(self, gripper_value):
        """
        Set the current gripper value (e.g., from robot's actual gripper state).

        Args:
            gripper_value: Current gripper value (0.04 = open, 0.0 = closed)
        """
        if self.has_gripper:
            self.gripper_value = np.clip(gripper_value, 0.0, 0.04)
        else:
            print("Warning: No gripper configured, ignoring gripper value")

    def get_desired_positions(self):
        """Get current desired joint positions."""
        return self.desired_positions.copy()

    def get_gripper_value(self):
        """Get current gripper value."""
        return self.gripper_value

    def set_gripper_value(self, value):
        """Set gripper value (0.04 = open, 0.0 = closed)."""
        self.gripper_value = np.clip(value, 0.0, 0.04)

    def reset_to_home(self):
        """Reset desired positions to home position."""
        self.desired_positions = self.home_positions.copy()
        self.gripper_value = 0.0  # Close gripper

    def set_home_positions(self, home_positions):
        """Set custom home positions."""
        if len(home_positions) == self.num_joints:
            self.home_positions = np.array(home_positions)
        else:
            print(
                f"Warning: Expected {self.num_joints} home positions, got {len(home_positions)}"
            )

    def set_control_mode(self, mode):
        """Set the control mode."""
        if mode in [
            self.DIRECT_JOINT_CONTROL,
            self.TASK_SPACE_POSE,
            self.GYROSCOPE_ORIENTATION,
            self.GYROSCOPE_TASK_POSE,
        ]:
            self.control_mode = mode
            print(f"Control mode set to: {mode}")
        else:
            print(f"Unknown control mode: {mode}")

    def set_ik_controller(self, ik_controller):
        """Set the IK controller for task-space control."""
        self.ik_controller = ik_controller
        print("IK controller set for task-space control")

    def set_task_space_speeds(self, position_speed, rotation_speed):
        """Set task-space movement speeds."""
        self.task_space_position_speed = position_speed
        self.task_space_rotation_speed = rotation_speed
        print(
            f"Task-space speeds set - Position: {position_speed} m/s, Rotation: {rotation_speed} rad/s"
        )

    def set_controller_light(self, player_id):
        """Set the controller light (for future feedback)."""
        if self.available and self.dualsense is not None:
            try:
                self.dualsense.set_light_playerid(player_id)
            except Exception as e:
                print(f"Error setting controller light: {e}")

    def set_position_speed(self, speed):
        """Set position movement speed."""
        self.position_speed = speed
        print(f"Position speed set to: {speed}")

    def set_rotation_speed(self, speed):
        """Set rotation movement speed."""
        self.rotation_speed = speed
        print(f"Rotation speed set to: {speed}")

    def set_joystick_deadzone(self, deadzone):
        """Set joystick deadzone."""
        self.joystick_deadzone = deadzone
        print(f"Joystick deadzone set to: {deadzone}")

    def set_gyro_sensitivity(self, sensitivity):
        """Set gyroscope sensitivity for orientation control."""
        self.gyro_sensitivity = sensitivity
        print(f"Gyroscope sensitivity set to: {sensitivity}")

    def set_gyro_deadzone(self, deadzone):
        """Set gyroscope deadzone to filter noise."""
        self.gyro_deadzone = deadzone
        print(f"Gyroscope deadzone set to: {deadzone}")

    def set_gyro_position_speed(self, speed):
        """Set position movement speed for gyroscope mode."""
        self.gyro_position_speed = speed
        print(f"Gyroscope position speed set to: {speed}")

    def get_gyro_data(self):
        """Get current gyroscope data if available."""
        if self.available and self.dualsense is not None:
            return {
                "gyro": self.dualsense.gyro_state.copy(),
                "accel": self.dualsense.accel_state.copy(),
            }
        else:
            return {"gyro": np.zeros(3), "accel": np.zeros(3)}

    def get_gyro_reference(self):
        """Get current gyroscope reference."""
        return (
            self.gyro_reference.copy()
            if self.gyro_reference is not None
            else np.zeros(3)
        )

    def get_gyro_delta(self):
        """Get current gyroscope delta from reference."""
        if (
            self.available
            and self.dualsense is not None
            and self.gyro_reference is not None
        ):
            current_gyro = np.array(self.dualsense.gyro_state)
            return current_gyro - self.gyro_reference
        else:
            return np.zeros(3)

    def set_gyroscope_reference_from_home(self):
        """Set gyroscope reference to current reading when robot is in home position."""
        if self.available and self.dualsense is not None:
            current_gyro = np.array(self.dualsense.gyro_state)
            self.gyro_reference = current_gyro.copy()
            self.gyro_initialized = True
        else:
            self.gyro_reference = np.zeros(3)
            self.gyro_initialized = False
            print("Gyroscope reference set to zero (no controller available)")

    def get_gyroscope_relative_orientation(self, gyro_data):
        """
        Get relative orientation from gyroscope data compared to reference.

        Args:
            gyro_data: Raw gyroscope data [x, y, z]

        Returns:
            relative_orientation: [dx, dy, dz] relative to reference
        """
        if not self.gyro_initialized or self.gyro_reference is None:
            # If no reference set, use current as reference
            self.gyro_reference = np.array(gyro_data)
            self.gyro_initialized = True
            return np.zeros(3)

        # Compute relative orientation
        relative = np.array(gyro_data) - self.gyro_reference

        # Normalize to [-1, 1] range
        relative_norm = relative / 10000.0

        return relative_norm
