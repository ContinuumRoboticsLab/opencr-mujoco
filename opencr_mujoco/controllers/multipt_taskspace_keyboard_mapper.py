"""Keyboard input mapper for multi-point TDCR task-space control."""

import numpy as np
from typing import Dict
from pynput import keyboard
import threading


class MultiPointTaskSpaceKeyboardMapper:
    """Maps keyboard inputs to multi-point task-space velocity commands."""

    def __init__(self, velocity_scale: float = 1.0, verbose: bool = True):
        """Initialize keyboard mapper for multi-point task-space control.

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

        # Control point selection
        self.current_control_point = "seg3"  # Default to tip
        self.control_point_changed = False

        # Key states (mutated by the pynput listener thread; guarded by a lock)
        self.keys_pressed = set()
        self._keys_lock = threading.Lock()
        self.shift_pressed = False

        # Key mappings for task-space control
        self.motion_mappings = {
            # Linear motion
            "w": ("vx", 1.0),  # Move forward (+X)
            "s": ("vx", -1.0),  # Move backward (-X)
            "a": ("vy", 1.0),  # Move left (+Y)
            "d": ("vy", -1.0),  # Move right (-Y)
            "q": ("vz", 1.0),  # Move up (+Z)
            "e": ("vz", -1.0),  # Move down (-Z)
            # Angular motion
            "i": ("wx", 1.0),  # Rotate around X
            "k": ("wx", -1.0),
            "j": ("wy", 1.0),  # Rotate around Y
            "l": ("wy", -1.0),
            "u": ("wz", 1.0),  # Rotate around Z
            "o": ("wz", -1.0),
        }

        # TDCR segment bend (Clark). These bake in the SAME command-frame
        # transform that franka_tdcr_combined applies in the joint controller
        # (command_frame_offset_rad = -90 deg + command_mirror_x) so T/F/G/H bend
        # the same physical way in both teleop modes on the ftdcr_v4_sysid mount:
        # T/G drive Clark X (the combined viewer's up/down) and F/H drive Clark Y
        # (left/right). Keep these consistent with combined_keyboard_input_mapper.
        self.tdcr_mappings = {
            "t": ("clark_x", 1.0),  # Up
            "g": ("clark_x", -1.0),  # Down
            "f": ("clark_y", -1.0),  # Left
            "h": ("clark_y", 1.0),  # Right
        }

        # Insertion/extraction (move base along the backbone). On N/M so Y is
        # free for "reset TDCR" (R = reset Franka, Y = reset TDCR, matching
        # franka_tdcr_combined).
        self.insertion_mappings = {
            "n": ("v_insert", 1.0),  # Insert (move base towards endpoint)
            "m": ("v_insert", -1.0),  # Extract (move base away from endpoint)
        }

        # Control point mappings (with LSHIFT)
        self.control_point_mappings = {
            "b": "base",  # TDCR base (Franka control)
            "z": "seg1",  # Segment 1 endpoint (TDCR control)
            "x": "seg2",  # Segment 2 endpoint (TDCR control)
            "c": "seg3",  # TDCR tip (TDCR control)
        }

        # Control point descriptions
        self.control_point_info = {
            "base": "TDCR Base (Franka control)",
            "seg1": "Segment 1 End (TDCR seg 1)",
            "seg2": "Segment 2 End (TDCR seg 1-2)",
            "seg3": "TDCR Tip (TDCR all segments)",
        }

        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release
        )
        self.listener.start()

        if self.verbose:
            print("\nMulti-Point Task-Space Keyboard Mapper initialized")
            print("\nMotion Controls (with LSHIFT):")
            print("  W/S: Move forward/backward (X)")
            print("  A/D: Move left/right (Y)")
            print("  Q/E: Move up/down (Z)")
            print("  I/K: Roll (rotate around X)")
            print("  J/L: Pitch (rotate around Y)")
            print("  U/O: Yaw (rotate around Z)")
            print("  R: Reset Franka to home    Y: Reset TDCR (straighten)")
            print("\nTDCR Segment Control (with LSHIFT, for seg1/2/3):")
            print("  T/G: Bend up/down")
            print("  F/H: Bend left/right")
            print("  N/M: Insert/extract (move base towards/away from endpoint)")
            print("\nControl Point Selection (with LSHIFT):")
            print("  B: TDCR Base (Franka control)")
            print("  Z: Segment 1 End (TDCR seg 1)")
            print("  X: Segment 2 End (TDCR seg 1-2)")
            print("  C: TDCR Tip (TDCR all segments)")
            print(
                f"\nCurrent control point: {self.control_point_info[self.current_control_point]}"
            )

    def _on_key_press(self, key):
        """Handle key press events (runs on the pynput listener thread)."""
        if key == keyboard.Key.esc:
            return False  # Stop listener

        # Check for shift key
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.shift_pressed = True
            return

        # Only process if LSHIFT is held
        if not self.shift_pressed:
            return

        key_char = getattr(key, "char", None)
        if key_char:
            key_char = key_char.lower()

            # Check for control point change
            if key_char in self.control_point_mappings:
                new_control_point = self.control_point_mappings[key_char]
                if new_control_point != self.current_control_point:
                    self.current_control_point = new_control_point
                    self.control_point_changed = True
                    if self.verbose:
                        print(
                            f"\nSwitched to: {self.control_point_info[self.current_control_point]}"
                        )
                return

            # Add to pressed keys for motion
            with self._keys_lock:
                self.keys_pressed.add(key_char)

    def _on_key_release(self, key):
        """Handle key release events (runs on the pynput listener thread)."""
        # Check for shift key release
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.shift_pressed = False
            # Clear all pressed keys when shift is released
            with self._keys_lock:
                self.keys_pressed.clear()
            return

        key_char = getattr(key, "char", None)
        if key_char:
            with self._keys_lock:
                self.keys_pressed.discard(key_char.lower())

    def get_command(self) -> Dict[str, float]:
        """Get current velocity command based on pressed keys.

        Returns:
            Dictionary with velocity commands and control point
        """
        command = {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
            "reset_franka": False,  # R: home the Franka arm
            "reset_tdcr": False,  # Y: straighten the TDCR
            "clark_x": 0.0,
            "clark_y": 0.0,
            "v_insert": 0.0,  # Insertion/extraction velocity
        }

        # Add control point if changed
        if self.control_point_changed:
            command["control_point"] = self.current_control_point
            self.control_point_changed = False

        # Only process motion if LSHIFT is held
        if not self.shift_pressed:
            return command

        # Snapshot under the lock; the listener thread mutates the set
        with self._keys_lock:
            keys_pressed = set(self.keys_pressed)

        # Reset (hold-to-home): R = Franka, Y = TDCR (independent; press both to
        # home both). Matches franka_tdcr_combined. Reset dominates other motion
        # this frame (released on key up).
        reset_franka = "r" in keys_pressed
        reset_tdcr = "y" in keys_pressed
        if reset_franka or reset_tdcr:
            command["reset_franka"] = reset_franka
            command["reset_tdcr"] = reset_tdcr
            if self.verbose:
                print(f"Reset held: franka={reset_franka} tdcr={reset_tdcr}")
            return command

        # Process velocity commands
        for key_char in keys_pressed:
            if key_char in self.motion_mappings:
                action, value = self.motion_mappings[key_char]
                if action in ["vx", "vy", "vz", "wx", "wy", "wz"]:
                    command[action] = value * self.velocity_scale

        # Process TDCR segment controls (only for seg1/2/3, not base)
        if self.current_control_point != "base":
            for key_char in keys_pressed:
                if key_char in self.tdcr_mappings:
                    action, value = self.tdcr_mappings[key_char]
                    if action in ["clark_x", "clark_y"]:
                        command[action] = value * self.velocity_scale

            # Process insertion/extraction controls
            for key_char in keys_pressed:
                if key_char in self.insertion_mappings:
                    action, value = self.insertion_mappings[key_char]
                    if action == "v_insert":
                        command["v_insert"] = value * self.velocity_scale

        return command

    def stop(self):
        """Stop the keyboard listener."""
        self.listener.stop()
        if self.verbose:
            print("Multi-point keyboard mapper stopped")
