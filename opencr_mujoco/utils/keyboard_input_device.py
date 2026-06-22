#!/usr/bin/env python3
"""
Keyboard Input Device for Franka MuJoCo Environment.

This module provides a keyboard-based input device that can be used with the InputMapper
to control the robot using keyboard keys instead of a gamepad.

Uses pynput for cross-platform keyboard input that works alongside MuJoCo viewer.
"""

from typing import Set

try:
    from pynput import keyboard
except ImportError:
    print("Warning: pynput not installed. Install with: pip install pynput")
    keyboard = None


class KeyboardInputDevice:
    """
    Keyboard input device that provides a similar interface to DualSense controller.

    Key mappings (all require holding LSHIFT to avoid MuJoCo conflicts):
    - LSHIFT + W/A/S/D: X/Y position control
    - LSHIFT + Q/E: Z position (up/down)
    - LSHIFT + I/J/K/L: Roll/pitch rotation
    - LSHIFT + U/O: Yaw rotation
    - LSHIFT + H: Reset to home position (Homing)
    - LSHIFT + N/M: Gripper control (close/open)
    """

    def __init__(self):
        """Initialize the keyboard input device."""
        # State variables that InputMapper expects
        self.left_joystick_state = [0, 0]  # Neutral at 0 (0-255 range)
        self.right_joystick_state = [0, 0]

        # Button states
        self.triangle_down_state = False
        self.cross_down_state = False
        self.square_down_state = False
        self.circle_down_state = False
        self.dpad_up_state = False
        self.dpad_down_state = False
        self.dpad_left_state = False
        self.dpad_right_state = False
        self.ps_down_state = False
        self.l1_state = False
        self.r1_state = False

        # Gyroscope state (not used for keyboard)
        self.gyro_state = [0.0, 0.0, 0.0]
        self.accel_state = [0.0, 0.0, 0.0]

        # Track currently pressed keys
        self.pressed_keys: Set[str] = set()

        # Track shift key state
        self.shift_pressed = False

        # Start keyboard listener if pynput is available
        self.listener = None
        if keyboard is not None:
            self.listener = keyboard.Listener(
                on_press=self._on_key_press, on_release=self._on_key_release
            )
            self.listener.start()

            print("Keyboard Input Device Initialized")
            print("Controls (all require holding LSHIFT):")
            print("  LSHIFT + W/A/S/D: Move end-effector in X/Y plane")
            print("  LSHIFT + Q/E: Move end-effector up/down (Z)")
            print("  LSHIFT + I/J/K/L: Rotate end-effector (roll/pitch)")
            print("  LSHIFT + U/O: Rotate end-effector (yaw)")
            print("  LSHIFT + N/M: Close/Open gripper")
            print("  LSHIFT + H: Reset to home position")
        else:
            print("Keyboard Input Device: pynput not available")

    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            # Check for shift key
            if (
                key == keyboard.Key.shift
                or key == keyboard.Key.shift_l
                or key == keyboard.Key.shift_r
            ):
                self.shift_pressed = True
            # Only register other keys if shift is pressed
            elif self.shift_pressed:
                # Handle regular character keys
                if hasattr(key, "char") and key.char:
                    key_char = key.char.lower() if key.char else None
                    if key_char:
                        self.pressed_keys.add(key_char)
        except AttributeError:
            pass

    def _on_key_release(self, key):
        """Handle key release events."""
        try:
            # Check for shift key release
            if (
                key == keyboard.Key.shift
                or key == keyboard.Key.shift_l
                or key == keyboard.Key.shift_r
            ):
                self.shift_pressed = False
                # Clear all pressed keys when shift is released
                self.pressed_keys.clear()
            # Handle regular character keys
            elif hasattr(key, "char") and key.char:
                key_char = key.char.lower() if key.char else None
                if key_char:
                    self.pressed_keys.discard(key_char)
        except AttributeError:
            pass

    def update_state(self):
        """Update internal state based on currently pressed keys."""
        # Reset joystick states to neutral
        left_x = 0
        left_y = 0
        right_x = 0
        right_y = 0

        # Left joystick (WASD for X/Y position)
        if "a" in self.pressed_keys:
            left_x = -128  # Full left
        elif "d" in self.pressed_keys:
            left_x = 128  # Full right

        if "w" in self.pressed_keys:
            left_y = -128  # Full forward
        elif "s" in self.pressed_keys:
            left_y = 128  # Full backward

        # Right joystick (IJKL for roll/pitch)
        if "j" in self.pressed_keys:
            right_x = -128  # Roll left
        elif "l" in self.pressed_keys:
            right_x = 128  # Roll right

        if "i" in self.pressed_keys:
            right_y = -128  # Pitch up
        elif "k" in self.pressed_keys:
            right_y = 128  # Pitch down

        # Update joystick states
        self.left_joystick_state = [left_x, left_y]
        self.right_joystick_state = [right_x, right_y]

        # D-pad states (Q/E for Z position)
        self.dpad_up_state = "q" in self.pressed_keys
        self.dpad_down_state = "e" in self.pressed_keys

        # Button states (U/O for yaw rotation)
        self.triangle_down_state = "u" in self.pressed_keys
        self.cross_down_state = "o" in self.pressed_keys

        # Gripper control (N/M)
        self.dpad_left_state = "n" in self.pressed_keys  # Close
        self.dpad_right_state = "m" in self.pressed_keys  # Open

        # PS button (H for reset - Homing)
        self.ps_down_state = "h" in self.pressed_keys

    def close(self):
        """Close the keyboard input device."""
        if self.listener:
            self.listener.stop()
