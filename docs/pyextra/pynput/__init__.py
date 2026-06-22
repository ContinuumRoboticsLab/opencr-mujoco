"""Minimal pynput stub for the browser (Pyodide) runtime.

The real teleop input devices/mappers `from pynput import keyboard` and listen to
the OS keyboard. In the browser there is no OS listener -- the JS layer captures
keys and feeds them in -- but we still want to run the *unchanged* mapper code.
This stub provides just enough of the pynput keyboard API (Key, KeyCode,
Listener) for those modules to import and for their `get_command()` membership
checks (`KeyCode.from_char("f") in current_keys`, `Key.shift_l in current_keys`)
to work against keys we inject from JavaScript.
"""

from . import keyboard  # noqa: F401
