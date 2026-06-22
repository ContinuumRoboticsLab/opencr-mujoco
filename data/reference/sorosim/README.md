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
