"""Keyboard input mapper for TDCR task-space control."""

import numpy as np
from typing import Dict
from pynput import keyboard
import threading


class TDCRTaskSpaceKeyboardMapper:
    """Maps keyboard inputs to TDCR task-space velocity commands."""

    def __init__(self, velocity_scale: float = 1.0, verbose: bool = True):
        """Initialize keyboard mapper for task-space control.

        Args:
            velocity_scale: Overall velocity scaling factor
            verbose: Enable verbose output
        """
        self.velocity_scale = velocity_scale
        self.verbose = verbose

        # Current velocity commands
        self.linear_velocity = np.zeros(3)  # [vx, vy, vz]
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz]
        self.reset_requested = False

        # Key states (mutated by the pynput listener thread; guarded by a lock)
        self.keys_pressed = set()
        self._keys_lock = threading.Lock()

        # Key mappings for task-space control
        self.key_mappings = {
            # Linear motion
            "w": ("vx", 1.0),  # Move forward (+X)
            "s": ("vx", -1.0),  # Move backward (-X)
            "a": ("vy", 1.0),  # Move left (+Y)
            "d": ("vy", -1.0),  # Move right (-Y)
            "q": ("vz", 1.0),  # Move up (+Z)
            "e": ("vz", -1.0),  # Move down (-Z)
            # Angular motion (optional)
            "i": ("wx", 1.0),  # Rotate around X
            "k": ("wx", -1.0),
            "j": ("wy", 1.0),  # Rotate around Y
            "l": ("wy", -1.0),
            "u": ("wz", 1.0),  # Rotate around Z
            "o": ("wz", -1.0),
            # Reset
            "r": ("reset", True),
        }

        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release
        )
        self.listener.start()

        if self.verbose:
            print("TDCR Task-Space Keyboard Mapper initialized")
            print("Controls (Position):")
            print("  W/S: Move forward/backward (X)")
            print("  A/D: Move left/right (Y)")
            print("  Q/E: Move up/down (Z)")
            print("Controls (Rotation):")
            print("  I/K: Roll (rotate around X)")
            print("  J/L: Pitch (rotate around Y)")
            print("  U/O: Yaw (rotate around Z)")
            print("  R: Reset to home")
            print("  ESC: Exit")

    def _on_key_press(self, key):
        """Handle key press events (runs on the pynput listener thread)."""
        if key == keyboard.Key.esc:
            return False  # Stop listener
        key_char = getattr(key, "char", None)
        if key_char:
            with self._keys_lock:
                self.keys_pressed.add(key_char.lower())

    def _on_key_release(self, key):
        """Handle key release events (runs on the pynput listener thread)."""
        key_char = getattr(key, "char", None)
        if key_char:
            with self._keys_lock:
                self.keys_pressed.discard(key_char.lower())

    def get_command(self) -> Dict[str, float]:
        """Get current velocity command based on pressed keys.

        Returns:
            Dictionary with velocity commands
        """
        command = {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
            "reset_home": False,
        }

        # Snapshot under the lock; the listener thread mutates the set
        with self._keys_lock:
            keys_pressed = set(self.keys_pressed)

        # Check for reset
        if "r" in keys_pressed:
            command["reset_home"] = True  # held (hold-to-home; released on key up)
            if self.verbose:
                print("Reset to home requested")
            return command

        # Process velocity commands
        for key_char in keys_pressed:
            if key_char in self.key_mappings:
                action, value = self.key_mappings[key_char]
                if action in ["vx", "vy", "vz", "wx", "wy", "wz"]:
                    command[action] = value * self.velocity_scale

        return command

    def stop(self):
        """Stop the keyboard listener."""
        self.listener.stop()
        if self.verbose:
            print("Keyboard mapper stopped")
