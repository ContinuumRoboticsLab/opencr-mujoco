# Paper results — SoroSim validation & figures

All evaluation and figure scripts live here. Everything runs from a clean
clone (the SoroSim reference data is bundled under `data/reference/sorosim/`).

```bash
# Full reproduction: every statics + dynamics evaluation, then all figures
python paper_results/reproduce_paper_results.py
python paper_results/reproduce_paper_results.py --figures-only   # rebuild figures only

# Individual pieces
python paper_results/run_statics.py                  # both statics evaluations
python paper_results/run_dynamics.py --all-materials --sim-hz 200 500 1000 # dynamics bank (10 tests/material)
python paper_results/visualize_statics.py            # statics figures
python paper_results/visualize_dynamics.py           # dynamics figures
python paper_results/visualize_sysid.py              # sysid figure (committed calibration)

# Single evaluation with overrides (shared engine)
python paper_results/evaluate.py --config spring_steel_statics \
    --n-values 25 50 100 --early-stop 20
python paper_results/evaluate.py --help              # all options
```

## What the evaluations are

- **Statics** (`spring_steel_statics`, `tpu_statics`): 500 equilibrium shapes
  per material under gravity + mid/tip wrenches, measured at 14 arc-length
  stations per shape; errors are computed after interpolating the reference
  onto the simulation's arc positions
- **Dynamics** (`sorosim_dynamics_steel`, `sorosim_dynamics_tpu`): 10
  tip-release tests per material with mid+tip pose time series at 200 Hz
- **Frame conversion**: each evaluation config carries an explicit
  `frame_conversion.file_to_mujoco` rotation applied to all reference
  3-vectors at load time, so reference and simulation are compared in the
  same frame

Results land in `paper_results/evaluation_results/` and figures in
`paper_results/paper_figures/` (both git-ignored). Reference-data formats are
documented in [data/reference/sorosim/README.md](../data/reference/sorosim/README.md).
