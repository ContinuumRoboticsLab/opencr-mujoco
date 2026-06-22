# SoroSim reference data

Reference datasets used by the paper-results pipeline (`paper_results/`) to
validate the MuJoCo TDCR simulation against SoroSim. All 3-vectors are stored in
the SoroSim file frame; the evaluator applies each eval config's
`frame_conversion.file_to_mujoco` matrix (`[[0,0,1],[0,-1,0],[1,0,0]]`) to compare
reference and simulation in the same frame.

## Layout

```
sorosim/
├── sorosim_statics/    # static-equilibrium CSVs (2 materials)
│   ├── SpringSteelRodMuJoCo_dataStatics.csv
│   └── TPURodMuJoCo_dataStatics.csv
└── sorosim_dynamics/   # 13-column tip-release time series (10 tests / material)
    ├── SpringSteelRodMuJoCo_1.txt .. _10.txt
    └── TPURodMuJoCo_1.txt .. _10.txt
```

- **Statics CSV**: 2 header rows (labels + arc lengths), then 6 rows per shape
  (EulX/EulY/EulZ, Px/Py/Pz) across gravity + mid/tip wrench + per-segment arc
  lengths. Loaded by `ReferenceDataLoader.load_sorosim_statics_csv`.
- **Dynamics txt** (13 columns, tab-separated): line 1 = damping + gravity, line 2 =
  mid/tip holding wrenches, lines 3+ = time + mid pose(6) + tip pose(6). Loaded by
  `ReferenceDataLoader.load_tip_release_data`.

## Dynamics test provenance

The dynamics bank is 10 tests per material, renumbered `1..10` in ascending
original-SoroSim order. These are the cases that stay numerically stable across
the full 200/500/1000 Hz sweep under the reference protocol (Euler integrator,
5 s force ramp, 2 s hold): the lowest-damping cases (d ≲ 0.003) diverge at the
coarse 200 Hz / 5 ms explicit step, so they were dropped from the bank rather
than excluded post-hoc from the aggregate. The table maps each shipped id back to
its original SoroSim index and damping ratio, so aggregate results and the
per-test trajectory figure (`visualize_dynamics.py --steel-test 4` = original
steel test 5) are reproducible.

### SpringSteelRodMuJoCo

| new id | original SoroSim index | damping |
|:------:|:----------------------:|:-------:|
| 1 | 2 | 0.0041 |
| 2 | 3 | 0.0050 |
| 3 | 4 | 0.0046 |
| 4 | 5 | 0.0043 |
| 5 | 7 | 0.0037 |
| 6 | 9 | 0.0040 |
| 7 | 10 | 0.0027 |
| 8 | 13 | 0.0033 |
| 9 | 14 | 0.0039 |
| 10 | 15 | 0.0029 |

### TPURodMuJoCo

| new id | original SoroSim index | damping |
|:------:|:----------------------:|:-------:|
| 1 | 1 | 0.0030 |
| 2 | 2 | 0.0036 |
| 3 | 3 | 0.0048 |
| 4 | 5 | 0.0050 |
| 5 | 7 | 0.0032 |
| 6 | 9 | 0.0041 |
| 7 | 11 | 0.0031 |
| 8 | 12 | 0.0033 |
| 9 | 25 | 0.0033 |
| 10 | 29 | 0.0043 |
