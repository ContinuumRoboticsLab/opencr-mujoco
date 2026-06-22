"""Keyboard input mapper for TDCR control.

Maps keyboard inputs to TDCR control commands using LSHIFT + keys:
- LSHIFT + T/F/G/H: Control current segment (up/left/down/right)
- LSHIFT + Z/X/C/V/B: Select segments 1-5
- LSHIFT + R: Reset to home position
"""

from typing import Dict
from pynput import keyboard


class TDCRKeyboardInputMapper:
    """Maps keyboard inputs to TDCR control commands.

    Supports up to 5 segments via Z/X/C/V/B keys.

    Attributes:
        current_keys: Set of currently pressed keys
        current_segment: Currently selected segment (0 to max_segments-1)
        max_segments: Maximum number of segments supported (default 5)
    """

    def __init__(self, max_segments: int = 5):
        """Initialize keyboard input mapper.

        Args:
            max_segments: Maximum number of segments to support (1-5, default 5)
        """
        self.current_keys = set()
        self.current_segment = 0
        self.listener = None
        self.max_segments = min(max(1, max_segments), 5)  # Clamp to 1-5

    def start(self):
        """Start listening to keyboard events."""
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()
        print("TDCR Keyboard control started. Controls:")
        print("  LSHIFT + T/F/G/H: Control current segment (up/left/down/right)")

        # Show available segment keys based on max_segments
        segment_keys = ["Z", "X", "C", "V", "B"]
        available_keys = segment_keys[: self.max_segments]
        print(
            f"  LSHIFT + {'/'.join(available_keys)}: Select segments 1-{self.max_segments}"
        )

        print("  LSHIFT + R: Reset to home position")

    def stop(self):
        """Stop listening to keyboard events."""
        if self.listener:
            self.listener.stop()

    def _on_press(self, key):
        """Handle key press events."""
        try:
            self.current_keys.add(key)
        except Exception:
            pass

    def _on_release(self, key):
        """Handle key release events."""
        try:
            self.current_keys.discard(key)
        except Exception:
            pass

    def get_command(self) -> Dict[str, float]:
        """Get current control command based on pressed keys.

        Returns:
            Dictionary with control commands:
                - 'x': X-axis command (-1 to 1)
                - 'y': Y-axis command (-1 to 1)
                - 'segment': Target segment (0-based, up to max_segments - 1)
                - 'reset_home': True if home reset requested
        """
        command = {
            "x": 0.0,
            "y": 0.0,
            "segment": self.current_segment,
            "reset_home": False,
        }

        # Check if LSHIFT is held
        lshift_held = (
            keyboard.Key.shift_l in self.current_keys
            or keyboard.Key.shift in self.current_keys
        )

        if not lshift_held:
            return command

        # TFGH control for current segment
        if (
            keyboard.KeyCode.from_char("f") in self.current_keys
            or keyboard.KeyCode.from_char("F") in self.current_keys
        ):
            command["x"] = -1.0  # Left
        elif (
            keyboard.KeyCode.from_char("h") in self.current_keys
            or keyboard.KeyCode.from_char("H") in self.current_keys
        ):
            command["x"] = 1.0  # Right

        if (
            keyboard.KeyCode.from_char("t") in self.current_keys
            or keyboard.KeyCode.from_char("T") in self.current_keys
        ):
            command["y"] = 1.0  # Up
        elif (
            keyboard.KeyCode.from_char("g") in self.current_keys
            or keyboard.KeyCode.from_char("G") in self.current_keys
        ):
            command["y"] = -1.0  # Down

        # Segment selection with ZXCVB keys (up to max_segments)
        segment_keys = ["z", "x", "c", "v", "b"]
        for i, key_char in enumerate(segment_keys):
            if i >= self.max_segments:  # Only support up to max_segments
                break
            if (
                keyboard.KeyCode.from_char(key_char) in self.current_keys
                or keyboard.KeyCode.from_char(key_char.upper()) in self.current_keys
            ):
                if self.current_segment != i:
                    print(f"Selected segment {i + 1}")
                self.current_segment = i
                command["segment"] = i

        # Reset to home position
        if (
            keyboard.KeyCode.from_char("r") in self.current_keys
            or keyboard.KeyCode.from_char("R") in self.current_keys
        ):
            command["reset_home"] = True

        return command
