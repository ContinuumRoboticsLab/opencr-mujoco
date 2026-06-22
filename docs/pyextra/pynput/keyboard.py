"""Stub of ``pynput.keyboard`` sufficient for the teleop mappers in the browser.

Only the surface the repo touches is implemented:
  - ``Key`` named special keys (shift / shift_l / shift_r / ...), each a unique
    hashable sentinel, comparable by identity.
  - ``KeyCode.from_char(c)`` -> cached hashable object comparing equal by char,
    matching how the mappers build/compare their ``current_keys`` sets.
  - ``Listener`` no-op (the browser supplies key state directly).
"""


class KeyCode:
    _cache = {}

    def __init__(self, char=None, vk=None):
        self.char = char
        self.vk = vk

    @classmethod
    def from_char(cls, char):
        key = ("char", char)
        obj = cls._cache.get(key)
        if obj is None:
            obj = cls(char=char)
            cls._cache[key] = obj
        return obj

    @classmethod
    def from_vk(cls, vk):
        key = ("vk", vk)
        obj = cls._cache.get(key)
        if obj is None:
            obj = cls(vk=vk)
            cls._cache[key] = obj
        return obj

    def __eq__(self, other):
        return (
            isinstance(other, KeyCode)
            and self.char == other.char
            and self.vk == other.vk
        )

    def __hash__(self):
        return hash((self.char, self.vk))

    def __repr__(self):
        return f"KeyCode(char={self.char!r})"


class _SpecialKey:
    """A unique, hashable sentinel for a named special key."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Key.{self.name}"


class _KeyMeta(type):
    """Lazily mint a unique _SpecialKey for any attribute (esc, space, ...)."""

    _members = {}

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        obj = cls._members.get(name)
        if obj is None:
            obj = _SpecialKey(name)
            cls._members[name] = obj
        return obj


class Key(metaclass=_KeyMeta):
    pass


class Listener:
    """No-op stand-in for pynput.keyboard.Listener (browser feeds keys directly)."""

    def __init__(self, on_press=None, on_release=None, *args, **kwargs):
        self.on_press = on_press
        self.on_release = on_release
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def join(self, *args, **kwargs):
        pass


class Controller:  # pragma: no cover - not used in browser, present for imports
    def press(self, *a, **k):
        pass

    def release(self, *a, **k):
        pass
