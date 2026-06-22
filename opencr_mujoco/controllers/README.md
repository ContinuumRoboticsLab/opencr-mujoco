# Controllers — TDCR control architecture

Four layers, wired together by `teleop.py`:

1. **Input layer** — `KeyboardInputDevice` / `DualSenseInputDevice` read raw
   input state
2. **Mapping layer** — input mappers translate keys/sticks into robot commands
3. **Control layer** — controllers compute actuator targets:
   - `JointController` / `IKController` — Franka joint / Cartesian control
     (incremental Jacobian pseudo-inverse IK)
   - `TDCRJointController` — TDCR via **Clark coordinates** (2 values per
     segment describing bend direction/magnitude), converted to per-tendon
     length changes by `opencr_mujoco/tdcr_kinematics/`; supports coupled, independent,
     and tension modes with automatic segment/tendon detection from
     `seg_X_ten_Y` actuator names
   - `TDCRIKController` — task-space tip control via damped least squares
     over a numerically-differenced Clark-coordinate Jacobian
   - `CombinedController` — simultaneous Franka IK + TDCR control
   - `TDCRMultiPoint*Controller` — task-space control of selectable points
     along the backbone (segment ends), with insertion/extraction
4. **Simulation layer** — MuJoCo steps the physics; all motion is relative to
   the pretension keyframe baked into generated models

For programmatic (non-teleop) use of the controllers, see
[`examples/trace_tip_demo.py`](../../examples/trace_tip_demo.py) — closed-loop
tip tracing, station-keeping, and the kick demo are all driven through these
classes.
