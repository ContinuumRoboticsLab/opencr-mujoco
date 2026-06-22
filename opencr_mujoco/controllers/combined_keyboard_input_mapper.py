"""Combined keyboard input mapper for Franka + TDCR control.

This mapper allows simultaneous control of both robots:
- Franka: Task-space control with WASD/QE/IJKL/UO
- TDCR: Joint control with TFGH and segment selection with ZXC
All controls work simultaneously - no switching needed.
"""

from typing import Dict, Tuple
from pynput import keyboard
import threading


class CombinedKeyboardInputMapper:
    """Combined keyboard mapper for Franka and TDCR control.

    Provides unified interface for controlling both robots via keyboard.
    """

    def __init__(self):
        """Initialize combined keyboard mapper."""
        self.current_keys = set()
        self.lock = threading.Lock()
        self.listener = None

        # Control states
        self.tdcr_segment = 0  # Current TDCR segment (0, 1, or 2)
        self.last_tdcr_segment = -1  # For detecting changes

        # Commands for both robots
        self.franka_command = {
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
            "droll": 0.0,
            "dpitch": 0.0,
            "dyaw": 0.0,
        }

        self.tdcr_command = {"x": 0.0, "y": 0.0, "segment": 0, "reset_home": False}

        print("\n=== Combined Keyboard Control ===")
        print("All controls work simultaneously - no switching needed!")
        print("\nFranka controls:")
        print("  LSHIFT + W/A/S/D: X/Y position")
        print("  LSHIFT + Q/E: Z position")
        print("  LSHIFT + I/J/K/L: Roll/pitch rotation")
        print("  LSHIFT + U/O: Yaw rotation")
        print("  LSHIFT + R: Reset to home position")
        print("\nTDCR controls:")
        print("  LSHIFT + T/F/G/H: Control current segment (up/left/down/right)")
        print("  LSHIFT + Z/X/C: Select segments 1/2/3")
        print("  LSHIFT + Y: Reset to home")

    def on_press(self, key):
        """Handle key press events."""
        with self.lock:
            self.current_keys.add(key)

    def on_release(self, key):
        """Handle key release events."""
        with self.lock:
            self.current_keys.discard(key)

    def start(self):
        """Start keyboard listener."""
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()
        print("Combined keyboard control started")

    def stop(self):
        """Stop keyboard listener."""
        if self.listener:
            self.listener.stop()
            self.listener.join()

    def get_franka_command(self) -> Dict[str, float]:
        """Get current Franka command based on keyboard state."""
        with self.lock:
            # Reset command
            for key in self.franka_command:
                self.franka_command[key] = 0.0

            # Process Franka controls if LSHIFT is held
            if (
                keyboard.Key.shift in self.current_keys
                or keyboard.Key.shift_l in self.current_keys
            ):
                # Position control
                if (
                    keyboard.KeyCode.from_char("w") in self.current_keys
                    or keyboard.KeyCode.from_char("W") in self.current_keys
                ):
                    self.franka_command["dx"] = 1.0
                if (
                    keyboard.KeyCode.from_char("s") in self.current_keys
                    or keyboard.KeyCode.from_char("S") in self.current_keys
                ):
                    self.franka_command["dx"] = -1.0
                if (
                    keyboard.KeyCode.from_char("a") in self.current_keys
                    or keyboard.KeyCode.from_char("A") in self.current_keys
                ):
                    self.franka_command["dy"] = 1.0
                if (
                    keyboard.KeyCode.from_char("d") in self.current_keys
                    or keyboard.KeyCode.from_char("D") in self.current_keys
                ):
                    self.franka_command["dy"] = -1.0
                if (
                    keyboard.KeyCode.from_char("q") in self.current_keys
                    or keyboard.KeyCode.from_char("Q") in self.current_keys
                ):
                    self.franka_command["dz"] = 1.0
                if (
                    keyboard.KeyCode.from_char("e") in self.current_keys
                    or keyboard.KeyCode.from_char("E") in self.current_keys
                ):
                    self.franka_command["dz"] = -1.0

                # Rotation control
                if (
                    keyboard.KeyCode.from_char("i") in self.current_keys
                    or keyboard.KeyCode.from_char("I") in self.current_keys
                ):
                    self.franka_command["droll"] = 1.0
                if (
                    keyboard.KeyCode.from_char("k") in self.current_keys
                    or keyboard.KeyCode.from_char("K") in self.current_keys
                ):
                    self.franka_command["droll"] = -1.0
                if (
                    keyboard.KeyCode.from_char("j") in self.current_keys
                    or keyboard.KeyCode.from_char("J") in self.current_keys
                ):
                    self.franka_command["dpitch"] = 1.0
                if (
                    keyboard.KeyCode.from_char("l") in self.current_keys
                    or keyboard.KeyCode.from_char("L") in self.current_keys
                ):
                    self.franka_command["dpitch"] = -1.0
                if (
                    keyboard.KeyCode.from_char("u") in self.current_keys
                    or keyboard.KeyCode.from_char("U") in self.current_keys
                ):
                    self.franka_command["dyaw"] = 1.0
                if (
                    keyboard.KeyCode.from_char("o") in self.current_keys
                    or keyboard.KeyCode.from_char("O") in self.current_keys
                ):
                    self.franka_command["dyaw"] = -1.0

                # Home position
                if (
                    keyboard.KeyCode.from_char("r") in self.current_keys
                    or keyboard.KeyCode.from_char("R") in self.current_keys
                ):
                    self.franka_command["reset_home"] = True
                else:
                    self.franka_command["reset_home"] = False

            return self.franka_command.copy()

    def get_tdcr_command(self) -> Dict[str, float]:
        """Get current TDCR command based on keyboard state."""
        with self.lock:
            # Reset command
            self.tdcr_command["x"] = 0.0
            self.tdcr_command["y"] = 0.0
            self.tdcr_command["reset_home"] = False

            # Process TDCR controls if LSHIFT is held
            if (
                keyboard.Key.shift in self.current_keys
                or keyboard.Key.shift_l in self.current_keys
            ):
                # Segment selection
                if (
                    keyboard.KeyCode.from_char("z") in self.current_keys
                    or keyboard.KeyCode.from_char("Z") in self.current_keys
                ):
                    self.tdcr_segment = 0
                elif (
                    keyboard.KeyCode.from_char("x") in self.current_keys
                    or keyboard.KeyCode.from_char("X") in self.current_keys
                ):
                    self.tdcr_segment = 1
                elif (
                    keyboard.KeyCode.from_char("c") in self.current_keys
                    or keyboard.KeyCode.from_char("C") in self.current_keys
                ):
                    self.tdcr_segment = 2

                # Movement control using TFGH
                if (
                    keyboard.KeyCode.from_char("t") in self.current_keys
                    or keyboard.KeyCode.from_char("T") in self.current_keys
                ):
                    self.tdcr_command["y"] = 1.0  # Up
                if (
                    keyboard.KeyCode.from_char("g") in self.current_keys
                    or keyboard.KeyCode.from_char("G") in self.current_keys
                ):
                    self.tdcr_command["y"] = -1.0  # Down
                if (
                    keyboard.KeyCode.from_char("f") in self.current_keys
                    or keyboard.KeyCode.from_char("F") in self.current_keys
                ):
                    self.tdcr_command["x"] = -1.0  # Left
                if (
                    keyboard.KeyCode.from_char("h") in self.current_keys
                    or keyboard.KeyCode.from_char("H") in self.current_keys
                ):
                    self.tdcr_command["x"] = 1.0  # Right

                # Reset
                if (
                    keyboard.KeyCode.from_char("y") in self.current_keys
                    or keyboard.KeyCode.from_char("Y") in self.current_keys
                ):
                    self.tdcr_command["reset_home"] = True

            # Always update segment
            self.tdcr_command["segment"] = self.tdcr_segment

            # Print segment change
            if self.tdcr_segment != self.last_tdcr_segment:
                print(f"Now controlling TDCR segment {self.tdcr_segment + 1}")
                self.last_tdcr_segment = self.tdcr_segment

            return self.tdcr_command.copy()

    def get_command(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get commands for both robots.

        Returns:
            Tuple of (franka_command, tdcr_command)
        """
        return self.get_franka_command(), self.get_tdcr_command()

    def close(self):
        """Clean up resources."""
        self.stop()
