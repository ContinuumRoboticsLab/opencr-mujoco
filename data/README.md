# Data Directory Structure

This directory contains reference data and evaluation results for the TDCR system.

## Directory Layout

```
data/
├── reference/              # Reference datasets (tracked in git)
│   └── sorosim/           # SoroSim reference data for TDCR validation
│       ├── README.md       # data formats + dynamics provenance table
│       ├── sorosim_statics/    # static-equilibrium CSVs (2 materials)
│       │   ├── SpringSteelRodMuJoCo_dataStatics.csv
│       │   └── TPURodMuJoCo_dataStatics.csv
│       └── sorosim_dynamics/   # 13-column tip-release txt (10 tests/material)
│           ├── SpringSteelRodMuJoCo_1.txt .. _10.txt
│           └── TPURodMuJoCo_1.txt .. _10.txt
└── sysid/                  # hardware mocap recordings for system identification
    ├── ftdcr_v4/<session>_analysis/   # train/val/tendon-pull CSVs + plots
    └── ftdcr_v6/<session>_analysis/
```

Generated evaluation results (git-ignored) are written under
`paper_results/evaluation_results/<config_name>/` (`*.csv` numerical results,
`*.pkl` link positions).

## Reference Data Format

The SoroSim files in `reference/sorosim/` are produced by the SoroSim
Cosserat-rod solver and provide high-fidelity reference shapes/trajectories.
There are two formats (see `data/reference/sorosim/README.md` for the full
spec and the dynamics provenance table):

- **Statics** (`sorosim_statics/*.csv`): 2 header rows (labels + per-segment
  arc lengths), then 6 rows per static shape (EulX/EulY/EulZ, Px/Py/Pz). Each
  shape carries its own gravity 3-vector and mid/tip 6-DoF wrenches, plus
  positions/orientations at 14 (non-uniform, Gauss-Lobatto) arc-length
  stations; 500 shapes per material.

- **Dynamics** (`sorosim_dynamics/*.txt`, 13 columns, tab-separated): line 1 =
  per-test damping ratio + gravity 3-vector, line 2 = mid/tip holding
  wrenches, lines 3+ = time + mid pose(6) + tip pose(6) sampled at 200 Hz.
  The bank is 10 tests per material — the cases that stay numerically stable
  across the full 200/500/1000 Hz sweep (lowest-damping cases that diverge at the
  coarse 200 Hz step are dropped), renumbered `1..10` — see
  `reference/sorosim/README.md` for the per-id provenance.

All 3-vectors are stored in the SoroSim file frame; the evaluator applies each
eval config's `frame_conversion.file_to_mujoco` matrix to compare reference and
simulation in the same frame.

## Usage

Reference data is automatically loaded by the evaluation system. To load it
directly (SoroSim configs require a `frame_conversion` matrix):

```python
from opencr_mujoco.evaluation import ReferenceDataLoader

loader = ReferenceDataLoader(
    "data/reference/sorosim",
    frame_conversion=[[0, 0, 1], [0, -1, 0], [1, 0, 0]],
)
# data_dict keys are (mid_wrench, tip_wrench, gravity) tuples;
# arc_lengths are the normalized arc positions of the sample columns
data_dict, num_samples, arc_lengths = loader.load_sorosim_statics_csv(
    "SpringSteelRodMuJoCo"
)
```

## Notes

- Reference data should not be modified
- Generated results under `paper_results/evaluation_results/` are git-ignored
- The evaluation system interpolates the reference shapes by arc length onto
  the simulation's sample points (the statics stations are non-uniform)