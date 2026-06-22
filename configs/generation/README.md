# TDCR Generation Configs

Each JSON file here describes one Tendon-Driven Continuum Robot (TDCR). Run
`python generate.py --config <name>` to turn a config into a MuJoCo model.

## What's here

| File | What it is | Start here if… |
|------|------------|----------------|
| **`example_three_segment_franka.json`** | A 3-segment, cable-driven TDCR on a Franka arm. Fully annotated. | …you're new — this is the canonical starting point. |
| **`example_modular.json`** | A *heterogeneous* TDCR built from a small library of modules (stiff base → soft tip). Fully annotated. | …you want different materials/stiffness along the robot. |
| **`example_modular_tension.json`** | The tension-mode counterpart of `example_modular`: tendons driven by **motor (force)** actuators instead of position actuators. | …you use tendon tension controller rather than position control. |
| `ftdcr_v4_sysid.json` | The v4 research robot used in our paper (we plan to open source that soon). Used by the calibrated teleop scenes and as the base model the sysid pipeline (`configs/sysid/ftdcr_v4_pipeline.json`) starts from. | …you're running the sysid pipeline or the calibrated configs. |
| `ftdcr_v4_sysid_0326.json` | The committed v4 calibration — the paper's sysid figure/RMSE regenerates its model from this config. | …you're reproducing the paper's results. |
| `ftdcr_v6_sysid.json` | The v6 research robot, used by `configs/sysid/ftdcr_v6_pipeline.json`. | |
| `examples/…` | Demo/prop robots (e.g. `example_contact.json` for the README's kick clip). Resolved by name like any other config, but **exempt from the golden physics baseline** — top-level configs are physics-locked by `tests/unit/test_generator_golden.py`; demo configs here can be added or tweaked freely. | …you're adding a demo. |

> **Tip — the examples are self-documenting.** Open
> `example_three_segment_franka.json` in an editor: every option is preceded by a
> `"_comment_…"` field explaining what it does and what values are sensible.
> (JSON has no real comments, so these are ordinary string fields — the generator
> ignores any key it doesn't recognize.)

## Quickstart

```bash
python generate.py --list-configs                              # see available configs
python generate.py --config example_three_segment_franka --franka   # TDCR on a Franka arm
python generate.py --config example_three_segment_franka --world     # TDCR mounted in the world
python generate.py --config example_modular --franka                 # the modular example
```

The generated scene is written under `assets/`; open it with
`python viewer.py` to look at the robot.

## Option reference

### Backbone geometry
- **`num_segments`** — number of segments (1–5).
- **`links_per_segment`** — `{"1": 10, ...}`: rigid links approximating each segment. More links = smoother bending, slower simulation (8–15 typical).
- **`segment_lengths`** — `{"1": 0.1, ...}`: length of each segment in **meters**. Segments may differ.
- **`radius`** — *visual* link radius (m). Cosmetic; structural stiffness comes from `material_properties.outer_radius`.
- **`mass`** — total backbone mass (kg). Only used by `joint_config_mode: "direct"`.

### Joints / stiffness
- **`joints_per_link`** — `2` (bending about two axes, typical) or `3` (also adds axial twist).
- **`joint_config_mode`** — `"material"` (derive joint stiffness from beam theory) or `"direct"` (give stiffness/damping explicitly).
- **`joint_range` / `vert_joint_range`** — per-joint limits in **radians**
  (the models compile with `angle="radian"`). Only applied when
  `joint_limited` is true (default: unlimited).
- **`material_properties`** (when mode is `"material"`):
  - `youngs_modulus` (Pa) — stiffer material = stiffer robot.
  - `outer_radius` / `inner_radius` (m) — the *structural* backbone cross-section. Small `outer_radius` ⇒ flexible.
  - `density` (kg/m³) — with the cross-section, sets per-link mass.
  - `poisson_ratio` (≈0.3), `damping` (per-joint).
- Top-level **`stiffness`** / **`damping`** (when mode is `"direct"`): explicit per-joint values.

### Tendon actuation
- **`actuation_mode`** — `"parallel_tendons"` (cable-driven; the standard
  TDCR), `"direct_torque"` (motors on every joint), or `"none"` (passive rod —
  what the SoroSim evaluations use). Position vs force control of the tendons
  is chosen by `actuator_properties.tendon_actuator_type`, not here.
- **`actuation_details.segments[i]`** — per segment: `number_of_tendons` (3 = triangular, 4 = cross), `distance_to_backbone` (m; bigger ⇒ more bending torque per cable pull).
- **`actuator_properties`**:
  - `tendon_actuator_type` — `"position"` (length control, common) or `"motor"` (force).
  - `tendon_kp` (position-type servo stiffness; `tendon_kp_array` gives
    per-tendon values, as the sysid calibrations do), `tendon_kp_scaling` +
    `tendon_kp_scale_factor` (interpolates a per-segment kp multiplier from
    `scale_factor` at the base segment to 1.0 at the tip; 1.0 = uniform).
  - `tendon_pretension` — baseline tension (N) holding the robot taut at rest.
  - `tendon_ctrlrange`, `tendon_forcelimited`, `tendon_forcerange` (cables only pull ⇒ force ≤ 0).
- **`seg_offsets`** — rotate each segment's tendon pattern about the backbone (radians); staggering lets stacked segments bend independently.

### World / mounting
- **`gravity`**, **`plane`**, **`plane_style`**, **`disable_self_collision`**.
- **`mounting`** — `franka_pos`/`franka_euler` (base pose in the Franka EE frame, used with `--franka`) and `world_pos`/`world_euler` (used with `--world`).

## Modular (heterogeneous) configs

Set `"modular": true` and, instead of one global `material_properties`, provide:
- **`module_library`** — reusable blocks, each with its own `length`, `radius`, `num_links`, `color`, and `material_properties`.
- **`segments[i].modules`** — a list of module names making up that segment, plus its `actuation`.

Module and segment boundaries are **rigid** joins, so you can grade material along
the robot (e.g. a stiff base into a soft tip). See `example_modular.json`.

> For per-segment material on a *non-modular* config, add a
> `material_properties_per_segment` list (one block per segment); it is internally
> realized as a rigid one-module-per-segment robot.

## Make your own

1. Copy `example_three_segment_franka.json` to `my_robot.json`.
2. Edit the values (the `_comment_…` fields tell you what each does).
3. `python generate.py --config my_robot --franka`.

For the generator internals and the full physics, see
[`opencr_mujoco/generators/README.md`](../../opencr_mujoco/generators/README.md).
