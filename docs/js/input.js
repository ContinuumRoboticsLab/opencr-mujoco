// Browser keyboard capture for the demo. Captures plain letter keys while the
// viewer is hovered/focused (no modifier required) and reports the pressed set
// to the Python runtime, which injects them into the real teleop mappers.

// Physical-key -> letter. Using e.code (not e.key) makes WASD/TFGH map to the
// same physical positions on every keyboard layout (QWERTY, AZERTY, Dvorak,
// international Macs) — e.key returns the produced *character*, which differs
// by layout and is one reason a demo can work on one machine and not another.
function letterFor(e) {
  if (/^Key[A-Z]$/.test(e.code)) return e.code.slice(3).toLowerCase();
  const k = e.key;
  if (k && k.length === 1 && /[a-z]/i.test(k)) return k.toLowerCase();
  return null;
}

export class KeyboardCapture {
  constructor(targetEl) {
    this.keys = new Set();
    this._hovered = false;
    this._focused = false;
    this._target = targetEl;

    targetEl.tabIndex = 0; // focusable
    targetEl.addEventListener("pointerover", () => (this._hovered = true));
    targetEl.addEventListener("pointerout", () => {
      this._hovered = false;
      // If the viewer also isn't focused, drop held keys so a key released
      // off-element can't stay stuck.
      if (!this._focused) this.keys.clear();
    });
    targetEl.addEventListener("pointerdown", () => targetEl.focus());
    targetEl.addEventListener("focus", () => (this._focused = true));
    targetEl.addEventListener("blur", () => {
      this._focused = false;
      this.keys.clear();
    });

    window.addEventListener("keydown", (e) => this._onKey(e, true));
    window.addEventListener("keyup", (e) => this._onKey(e, false));
    // Anything that can swallow a keyup (tab switch, window blur, the macOS
    // press-and-hold accent popup) must drop held keys so they don't stick.
    window.addEventListener("blur", () => this.keys.clear());
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.keys.clear();
    });
  }

  get active() {
    return this._hovered || this._focused;
  }

  _onKey(e, down) {
    const c = letterFor(e);
    if (c === null) return;
    if (down) {
      // Only START tracking a key while the viewer is active...
      if (!this.active) return;
      this.keys.add(c);
    } else {
      // ...but ALWAYS honor the release, even if the pointer has since left
      // the viewer — otherwise the key sticks "pressed" and the robot keeps
      // moving on its own.
      this.keys.delete(c);
    }
    e.preventDefault(); // keep the page from scrolling / shortcuts firing
  }

  /** Currently pressed letter keys (only while the viewer is active). */
  getState() {
    if (!this.active) return { keys: [] };
    return { keys: Array.from(this.keys) };
  }
}
