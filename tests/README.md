# Test Suite

Unit tests live in `tests/unit/`, end-to-end script tests in
`tests/test_main_scripts.py`. The suite has two layers:

- **Smoke layer** — things construct, load, and run (`test_generate`,
  `test_viewer`, `test_teleop`, `test_evaluate`, `test_evaluation`,
  `test_pretension`, `test_scene_creation`, `test_main_scripts`)
- **Correctness layer** (the nightly suite) — things produce the *right
  numbers*:
  - `test_generator_physics.py` — the discretization's physical invariants:
    total mass = ρ·A·L, tip exactly at `total_length` in every chain mode,
    isotropic bending stiffness + the GJ torsion ratio, exact joint count
    (clamped base half-link), collision-exclusion coverage, pretension
    keyframe force, and a passive-rod tip deflection matched against the
    Euler–Bernoulli cantilever solution `F·L³/3EI` to 2%
  - `test_teleop_correctness.py` — closed-loop control direction (a Clark
    command bends the simulated robot in the calibrated direction and
    returns home on reset), Clark↔tendon round trips and per-segment
    distance coupling, keyboard-mapper key→command tables and LSHIFT gating,
    config-precedence semantics (CLI > config > defaults, booleans survive
    unset flags), and an inventory check that every shipped teleop config is
    runnable as documented (scene ships in git or comes from a shipped
    generation config)
  - `test_evaluation_correctness.py` — the SoroSim loader conventions
    (non-uniform Gauss–Lobatto arc stations, 500-shape bank, dynamics
    moments used as stored), wrench-at-site moment arms about the body COM,
    backbone sample arc fractions, and a regression proving arc-length
    interpolation collapses shape_error to ~0 on a perfectly matching shape
  - `test_generator_golden.py` — golden snapshot locking the generator's
    outputs (masses, stiffnesses, tendon lengths, actuator gains, pretension
    forces, tip positions) for every top-level + synthetic config; regenerate
    intentionally with `python tests/unit/test_generator_golden.py` after a
    deliberate behavior change. Scope: `configs/generation/*.json` only —
    demo configs under `configs/generation/examples/` are exempt, so adding
    a demo never requires a baseline regen

## Running

```bash
python run_tests.py                     # full unit suite
python run_tests.py --quick             # smoke subset
python run_tests.py -t generator_physics   # one module (choices auto-discovered)

python -m pytest tests/unit/ -v                  # pytest directly
python -m pytest tests/unit/ --cov=src --cov-report=term-missing
python -m pytest tests/test_main_scripts.py -v   # end-to-end script tests
```

### Nightly invocation

```bash
python -m pytest tests/unit/ tests/test_main_scripts.py -q
```

On GitHub this runs automatically every night via
[`.github/workflows/nightly-tests.yml`](../.github/workflows/nightly-tests.yml)
(07:00 UTC, Python 3.10 + 3.13, manual runs via the Actions tab's
"Run workflow" button). The workflow wraps pytest in `xvfb-run` so the
pynput-based keyboard-mapper tests get an X display instead of skipping.

Everything is headless (no GUI needed). The keyboard-mapper tests skip
themselves on machines without an input backend (e.g. display-less CI), and
all MuJoCo-dependent tests skip if `mujoco` is not importable.

## Writing new tests

1. Put smoke tests next to the feature's existing module; put *numerical*
   claims (signs, units, conventions, physical formulas) in the matching
   `*_correctness.py` / `*_physics.py` module so the nightly layer guards
   them.
2. Generate models into `tmp_path` — never into `assets/`.
3. If you intentionally change generator behavior, regenerate the golden
   baseline (`python tests/unit/test_generator_golden.py`) in the same
   commit and say so in the commit message.

## Tooling

```bash
python run_tests.py                 # full unit suite
python run_tests.py --quick         # smoke subset
python run_tests.py -t generate     # one module (see -h for the list)
python -m pytest tests/unit/ --cov=src --cov-report=term-missing
black opencr_mujoco/                          # formatting
flake8 opencr_mujoco/                         # linting (.flake8 / pyproject configure them)
```
