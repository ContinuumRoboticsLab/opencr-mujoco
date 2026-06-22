"""Bridge for controlling real TDCR robot via Dynamixel servos."""

import numpy as np
import time
from typing import Optional, Dict, Any


class DynamixelBridge:
    """Bridge to mirror MuJoCo simulation tendon values to real TDCR robot."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dynamixel bridge.

        Args:
            config: Configuration dictionary with servo settings
        """
        self.enabled = False
        self.servos_controller = None
        self.initial_sim_tendons = None
        self.initial_real_tendons = None
        self.last_valid_movement = None  # Track last valid movement for each segment

        # Default configuration for ftdcr_v4
        default_config = {
            "servo_ids": [3, 10, 1, 7, 6, 5, 11, 2, 9],
            "spool_radii_mm": 10.0,
            "device_name": "/dev/ttyUSB0",
            "servo_dir": [-1, 1, -1, -1, 1, -1, -1, 1, -1],
            "max_speed_mm_per_sec": 8.0,
            "tendon_scale_factor": 0.7,  # Scale factor from sim to real if needed
            # Safety parameters
            "max_tendon_displacement_mm": 15.0,  # Maximum displacement from pretension
            "max_segment_sum_error_mm": 0.5,  # Maximum allowed error in segment sum
            "enable_safety_checks": True,
        }

        self.config = {**default_config, **(config or {})}

    def connect(self) -> bool:
        """
        Connect to the real robot and initialize.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import dependency only when connecting
            from .multi_tendon_sync_rw import MultiTendonSyncRW

            # Initialize servo controller
            servo_ids = np.array(self.config["servo_ids"])
            spool_radii = np.ones(len(servo_ids)) * self.config["spool_radii_mm"]
            zero_offsets = np.zeros(len(servo_ids))
            servo_dir = np.array(self.config["servo_dir"])

            self.servos_controller = MultiTendonSyncRW(
                servo_ids=servo_ids,
                spool_radii_mm=spool_radii,
                zero_offsets_tick=zero_offsets,
                device_name=self.config["device_name"],
                servo_dir=servo_dir,
            )

            # Set zero position to current position (pretension)
            print("Setting zero position to current (pretension) position...")
            self.servos_controller.set_zero_offsets_to_current_position()

            # Set movement speed
            speed = self.config["max_speed_mm_per_sec"]
            self.servos_controller.set_tendons_speeds_mm_per_sec(
                np.ones(len(servo_ids)) * speed
            )

            # Store initial real robot position (should be zero after setting offsets)
            self.initial_real_tendons = self.servos_controller.get_tendons_mm()

            self.enabled = True
            print(
                f"Successfully connected to real robot on {self.config['device_name']}"
            )
            print(f"Servo IDs: {servo_ids}")
            print(f"Initial real tendon positions: {self.initial_real_tendons}")
            return True

        except ImportError as e:
            print(f"Failed to import MultiTendonSyncRW: {e}")
            print('Install the hardware extra: pip install -e ".[hardware]"')
            return False
        except Exception as e:
            print(f"Failed to connect to real robot: {e}")
            return False

    def set_initial_sim_position(self, sim_tendons: np.ndarray):
        """
        Store the initial simulation tendon positions (at pretension).

        Args:
            sim_tendons: Initial tendon positions from simulation
        """
        self.initial_sim_tendons = sim_tendons.copy()
        print(f"Stored initial sim tendon positions: {self.initial_sim_tendons}")

    def check_and_clip_safety(
        self, movement_mm: np.ndarray
    ) -> tuple[np.ndarray, bool, str]:
        """
        Check safety and clip movement to safe limits if needed.
        When any tendon in a segment exceeds limits, the entire segment maintains its last valid position.

        Args:
            movement_mm: Commanded movement from pretension in mm

        Returns:
            (clipped_movement, was_clipped, message): Clipped movement, whether clipping occurred, and message
        """
        if not self.config["enable_safety_checks"]:
            return movement_mm, False, ""

        clipped = movement_mm.copy()
        was_clipped = False
        messages = []

        # Initialize last valid movement if not set
        if self.last_valid_movement is None:
            self.last_valid_movement = np.zeros_like(movement_mm)

        # Check 1: Segment-level limit checking
        # If any tendon in a segment exceeds limit, maintain last valid position for entire segment
        max_disp = self.config["max_tendon_displacement_mm"]
        num_tendons = len(clipped)
        num_segments = num_tendons // 3

        for seg in range(num_segments):
            start_idx = seg * 3
            end_idx = start_idx + 3
            segment_tendons = clipped[start_idx:end_idx]

            # Check if any tendon in this segment exceeds limit
            over_limit_mask = np.abs(segment_tendons) > max_disp
            if np.any(over_limit_mask):
                over_limit_idx = np.where(over_limit_mask)[0] + start_idx
                # Maintain last valid position for entire segment
                clipped[start_idx:end_idx] = self.last_valid_movement[start_idx:end_idx]
                was_clipped = True
                messages.append(
                    f"Segment {seg+1} holding last position: tendon(s) {over_limit_idx} exceeded ±{max_disp}mm"
                )
            else:
                # Update last valid movement for this segment
                self.last_valid_movement[start_idx:end_idx] = segment_tendons

        # Check 2: Sum of tendons in each segment should be zero (conservation)
        # For 3-tendon segments, check and correct if needed
        max_error = self.config["max_segment_sum_error_mm"]

        for seg in range(num_segments):
            start_idx = seg * 3
            end_idx = start_idx + 3
            segment_tendons = clipped[start_idx:end_idx]
            segment_sum = np.sum(segment_tendons)

            if np.abs(segment_sum) > max_error:
                # Distribute the error equally among the three tendons to maintain sum near zero
                correction = segment_sum / 3.0
                clipped[start_idx:end_idx] -= correction
                was_clipped = True
                messages.append(
                    f"Corrected segment {seg+1} sum from {segment_sum:.3f}mm to near-zero"
                )

        message = "; ".join(messages) if messages else ""
        return clipped, was_clipped, message

    def send_sim_tendons(
        self, sim_tendons: np.ndarray, pretension_lengths: np.ndarray = None
    ) -> bool:
        """
        Send simulation tendon values to the real robot.
        The simulation values are absolute positions (pretension + movement) in meters.
        The real robot expects relative positions from pretension in mm.

        Args:
            sim_tendons: Current tendon values from MuJoCo simulation (in meters)
            pretension_lengths: Pretension lengths from simulation (in meters) - only needed on first call

        Returns:
            True if command sent successfully
        """
        if not self.enabled:
            return False

        try:
            # Store pretension if provided (only on first call)
            if pretension_lengths is not None:
                self.pretension_lengths = pretension_lengths.copy()
                print(f"Stored pretension lengths (m): {self.pretension_lengths}")

            # On first call, capture the initial state
            if self.initial_sim_tendons is None:
                self.set_initial_sim_position(sim_tendons)

                # If we have pretension lengths, subtract them to get relative movement
                if (
                    hasattr(self, "pretension_lengths")
                    and self.pretension_lengths is not None
                ):
                    # Initial movement = initial_position - pretension
                    initial_movement_m = sim_tendons - self.pretension_lengths
                else:
                    # Assume sim_tendons are already relative to pretension
                    initial_movement_m = sim_tendons

                # Convert to mm
                initial_movement_mm = (
                    initial_movement_m * 1000.0 * self.config["tendon_scale_factor"]
                )

                print(f"Initial sim position (m): {sim_tendons}")
                if (
                    hasattr(self, "pretension_lengths")
                    and self.pretension_lengths is not None
                ):
                    print(f"Pretension (m): {self.pretension_lengths}")
                print(f"Initial movement from pretension (mm): {initial_movement_mm}")

                # Safety check and clip if needed
                safe_movement, was_clipped, msg = self.check_and_clip_safety(
                    initial_movement_mm
                )
                if was_clipped:
                    print(f"SAFETY CLIPPING: {msg}")
                    print(f"Original: {initial_movement_mm}")
                    print(f"Clipped:  {safe_movement}")

                print(f"Moving real robot to match initial shape...")

                # Send position to robot (relative to its zero/pretension)
                success = self.servos_controller.async_set_tendons_mm_together(
                    safe_movement
                )
                if not success:
                    print("Failed to set initial position on real robot")
                return success

            # For subsequent calls, compute relative movement from pretension
            if (
                hasattr(self, "pretension_lengths")
                and self.pretension_lengths is not None
            ):
                # Movement = current_position - pretension
                movement_m = sim_tendons - self.pretension_lengths
            else:
                # Assume sim_tendons are already relative
                movement_m = sim_tendons

            # Convert to mm
            movement_mm = movement_m * 1000.0 * self.config["tendon_scale_factor"]

            # Safety check and clip if needed
            safe_movement, was_clipped, msg = self.check_and_clip_safety(movement_mm)

            # Only print clipping messages once or periodically to avoid spam
            if was_clipped:
                # Use a simple counter to reduce message frequency
                if not hasattr(self, "_clip_msg_counter"):
                    self._clip_msg_counter = 0
                self._clip_msg_counter += 1

                # Print every 10th clipping or if it's the first
                if self._clip_msg_counter == 1 or self._clip_msg_counter % 10 == 0:
                    print(f"SAFETY CLIPPING ({self._clip_msg_counter}x): {msg}")
                    if self.config.get("debug", False):
                        print(f"  Original: {movement_mm}")
                        print(f"  Clipped:  {safe_movement}")
            else:
                # Reset counter when back in safe zone
                if hasattr(self, "_clip_msg_counter") and self._clip_msg_counter > 0:
                    print(
                        f"Back in safe zone after {self._clip_msg_counter} clipped commands"
                    )
                    self._clip_msg_counter = 0

            # Debug output (remove * 0.0 for actual operation)
            if self.config.get("debug", False) and not was_clipped:
                print(f"Sim tendons (m): {sim_tendons}")
                print(f"Movement from pretension (mm): {safe_movement}")
                # Print segment sums for verification
                for seg in range(len(safe_movement) // 3):
                    seg_sum = np.sum(safe_movement[seg * 3 : (seg + 1) * 3])
                    print(f"  Segment {seg+1} sum: {seg_sum:.3f}mm")

            # Send to servos
            success = self.servos_controller.async_set_tendons_mm_together(
                safe_movement
            )

            if not success:
                print(f"Failed to send tendons to real robot")

            return success

        except Exception as e:
            print(f"Error sending tendon values: {e}")
            return False

    def reset_to_home(self) -> bool:
        """
        Reset the real robot to home (pretension) position.

        Returns:
            True if reset successful
        """
        if not self.enabled:
            return False

        try:
            # Send zeros to return to pretension position
            home_tendons = np.zeros(len(self.config["servo_ids"]))
            success = self.servos_controller.async_set_tendons_mm_together(home_tendons)

            if success:
                print("Reset real robot to pretension position")

            return success

        except Exception as e:
            print(f"Error resetting to home: {e}")
            return False

    def set_safety_params(
        self, max_displacement_mm: float = None, max_segment_sum_error_mm: float = None
    ):
        """
        Update safety parameters at runtime.

        Args:
            max_displacement_mm: Maximum allowed displacement from pretension
            max_segment_sum_error_mm: Maximum allowed error in segment sum
        """
        if max_displacement_mm is not None:
            self.config["max_tendon_displacement_mm"] = max_displacement_mm
            print(f"Updated max displacement to {max_displacement_mm}mm")

        if max_segment_sum_error_mm is not None:
            self.config["max_segment_sum_error_mm"] = max_segment_sum_error_mm
            print(f"Updated max segment sum error to {max_segment_sum_error_mm}mm")

    def enable_safety_checks(self, enable: bool):
        """Enable or disable safety checks."""
        self.config["enable_safety_checks"] = enable
        print(f"Safety checks {'enabled' if enable else 'disabled'}")

    def disconnect(self):
        """Disconnect from the real robot."""
        was_connected = bool(self.servos_controller and self.enabled)
        if was_connected:
            try:
                # Return to home position before disconnecting
                print("Returning to pretension position before disconnect...")
                self.reset_to_home()
                time.sleep(1.0)  # Wait for motion to complete
            except Exception:
                pass

        self.enabled = False
        self.servos_controller = None
        if was_connected:
            print("Disconnected from real robot")

    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
