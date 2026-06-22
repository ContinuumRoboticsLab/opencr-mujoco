"""Gradual "return to home" helper for teleop controllers.

Hold-to-home: while the home key is held, each control step nudges the commanded
target one step toward home; releasing the key simply stops the motion (like any
other teleop key). ``step_toward`` keeps the move synced -- the element with the
farthest to go moves at the rate cap and the rest move proportionally, so they
converge together.
"""

import numpy as np


def step_toward(current, goal, max_step):
    """Return ``current`` advanced one synced step toward ``goal``.

    The element with the largest remaining delta moves by at most ``max_step``;
    the others move proportionally so they all reach ``goal`` on the same step.
    Returns ``goal`` once everything is within ``max_step``.

    Set ``max_step`` to the teleop per-step increment so homing moves at teleop
    speed. Call once per control step while the home key is held.
    """
    current = np.asarray(current, dtype=float)
    goal = np.asarray(goal, dtype=float)
    delta = goal - current
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs <= max_step or max_abs == 0.0:
        return goal.copy()
    return current + delta * (max_step / max_abs)
