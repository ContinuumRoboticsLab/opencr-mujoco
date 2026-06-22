# TDCR System Identification

Calibrate the simulation's generation-config parameters against hardware
motion-capture recordings, so the generated MuJoCo model reproduces the real
robot's tip trajectories.

The mocap recordings for two hardware robots are bundled under `data/sysid/`
(`ftdcr_v4`, `ftdcr_v6`), so everything below runs from a clean clone.

## Quick start

```bash
# Discover bundled pipeline configs (configs/sysid/)
python sysid_pipeline.py --list-configs

# Validate a pipeline config + its data files without running anything
python sysid_pipeline.py --config ftdcr_v4_pipeline --dry-run

# Full 3-step pipeline (writes sysid_results/<name>_<timestamp>/)
python sysid_pipeline.py --config ftdcr_v4_pipeline

# Re-run a single step, skip the comparison videos, watch live
python sysid_pipeline.py --config ftdcr_v4_pipeline --step 2 --no-videos
python sysid_pipeline.py --config ftdcr_v4_pipeline --viewer

# Reduce the optimizer budget for a quick smoke run
python sysid_pipeline.py --config ftdcr_v4_pipeline --iterations 30
```

`--config` takes a name from `configs/sysid/` (the same convention as
`generate.py` and `teleop.py`) or an explicit path to a JSON file.
`--iterations` overrides the optimizer's per-start function-evaluation budget
(`optimization.maxfev`).

During optimization the pipeline shows a compact live status line (objective
evaluations done, best-so-far on the strided search metric, elapsed time) plus
one summary line per completed Powell start. Pass `--debug` to instead stream
the raw per-iteration logs from every worker process (the parallel multistart
runs ~11 workers at once, so expect heavy interleaving).

## The 3-step pipeline

Driven by `pipeline_orchestrator.py` from a single JSON config
(`configs/sysid/*.json`):

1. **Geometric identification** — from the `tendon_pull_file` (one tendon
   pulled at a time): identifies the hardware servo → `seg_X_ten_Y` mapping,
   the marker position bias (so the neutral tip sits at the expected height),
   and per-segment tendon angle offsets. Results are written into the working
   generation config.
2. **Tendon parameter optimization** — fits the enabled parameters (typically
   per-segment `pretension` in Newtons; per-segment `tendon_kp` can be fitted
   or pinned in the generation config) on the `train_file` trajectory.
3. **Refinement + validation** — re-fits around step 2's optimum with
   tightened bounds, then evaluates on the held-out `val_file` and renders
   comparison videos (requires `ffmpeg`; skipped gracefully without it).

Each iteration regenerates the model from the working generation config
(`create_tdcr_from_config`), replays the recorded tendon commands
quasi-statically (`trajectory_simulator.py`, per-step settling from
`simulation.settling_time`), and compares simulated vs recorded tip positions.

## Optimizer

`parallel_multistart` (the only implemented algorithm): N Sobol-seeded Powell
starts run in parallel worker processes, each on a stride-subsampled
trajectory; the winner is re-evaluated on the full trajectory. The kp-pinned
fitting problem is effectively unimodal, so the multistart converges reliably.

Config keys under `optimization`:

```json
{
  "algorithm": "parallel_multistart",
  "n_starts": 11,
  "workers": 11,
  "search_stride": 8,
  "maxfev": 120
}
```

(`n_starts` Powell starts across `workers` processes; `search_stride`
subsamples the trajectory during the search; `maxfev` caps Powell evaluations
per start.)

## Data format

`data.data_dir` + `tendon_pull_file` / `train_file` / `val_file` name CSVs
with columns:

- `timestamp`, `pattern_label`
- `servo_<id>_mm` — per-servo tendon displacement (hardware ordering; the
  step-1 servo mapping reorders them to `seg_X_ten_Y`)
- `tip_x`, `tip_y`, `tip_z` — mocap tip position (mm); a legacy
  `marker_x/y/z` naming is auto-detected too

## Outputs

```
sysid_results/<name>_<timestamp>/
├── pipeline_config.json
├── step1_geometric/            # servo mapping, bias, geometry results
├── step2_tendon_optimization/  # best_model.xml, results.json,
│   ├── parameter_history.csv   #   error_history.csv, trajectory plots
│   └── ...
├── step3_refinement/           # refined best_model.xml + validation
├── final_model.xml             # the calibrated model
└── summary.json                # per-step errors
```

### Metric convention

There is exactly ONE error number in this pipeline: the **standard
(Euclidean, pointwise) RMSE** of the tip position, mean-offset-aligned. The
optimizer objective (`train_error_mm` / `best_error_mm`), the printed
train/val numbers, the `trajectory_*.png` annotations,
`evaluate_sysid_model.py`, and the paper figure all report this same
quantity, so any two numbers you see are directly comparable.

Two historical traps to be aware of when reading OLD runs/logs:

- Before 2026-06-11 the objective was the **per-coordinate** RMSE, which is
  exactly tip-RMSE / sqrt(3) — old `train_error_mm` values are 1.73x smaller
  than today's for the same fit.
- Old validation numbers (`val_error_mm`) used an unaligned closest-point
  convention and are not comparable to current ones.

Closest-point (timing-robust) error was evaluated as an alternative — both
as a training objective and as the reported metric — and removed: an A/B on
the ftdcr_v4 data showed the data/robot timing misalignment it guards
against inflates the RMSE by a roughly parameter-independent floor (~1.8 mm)
without changing which parameters win, so the extra metric bought nothing
but confusion.

## Auxiliary tools

```bash
# Evaluate a calibrated model on any recorded dataset
python -m opencr_mujoco.sysid.evaluate_sysid_model --model assets/tdcr/ftdcr_v4_sysid.xml \
    --gen-config configs/generation/ftdcr_v4_sysid.json \
    --data-dir data/sysid/ftdcr_v4/tdcr_sysid_20260511-183002_analysis \
    --output eval_results/v4_eval

# Parameter sensitivity / identifiability analysis of a finished run
python -m opencr_mujoco.sysid.sensitivity_analysis --run-dir sysid_results/<robot>/<run>

# Excitation-trajectory optimization (informative command patterns)
python -m opencr_mujoco.sysid.optimize_excitation --run-dir sysid_results/<robot>/<run>
```

`paper_results/visualize_sysid.py` builds the paper's sysid figure and RMSE
by regenerating the model from the committed calibrated config
(`configs/generation/ftdcr_v4_sysid_0326.json`) and simulating it over the
committed train recording — it runs from a clean clone, no pipeline run
needed.

## Module map

- `pipeline_orchestrator.py` — the 3-step driver
- `pipeline_data_loader.py` — raw-CSV preprocessing (servo mapping, bias)
- `geometric_identifier.py` — step 1
- `sysid_optimizer.py` — objective function + result bookkeeping
- `parallel_optimizer.py` — the multistart Powell search
- `trajectory_simulator.py` — quasi-static rollout of recorded commands
- `data_loader.py` — standard-CSV trajectory loading
- `error_metrics.py` — the standard RMSE metric
- `visualization.py` — optimization/trajectory plots
- `evaluate_sysid_model.py`, `sensitivity_analysis.py`,
  `optimize_excitation.py` — auxiliary tools above
- `parameters/` — the parameter registry; each module defines bounds and how
  a parameter writes into the generation config (`pretension_params.py`,
  `tendon_params.py`, `geometry_params.py`, `material_params.py`,
  `friction_params.py`, `tendon_friction_params.py`, `tendon_slack_params.py`,
  `tendon_distance_params.py`, `tendon_constraint_params.py`,
  `joint_deadband_params.py`)

## Adding a new parameter type

```python
from opencr_mujoco.sysid.parameters import register_parameter
from opencr_mujoco.sysid.base_parameter import BaseParameter

@register_parameter('my_param')
class MyParameter(BaseParameter):
    def get_bounds(self): ...
    def get_dimension_names(self): ...
    def apply_to_config(self, values, generation_config): ...
```

Enable it in a pipeline config under the step's `parameters` block with
`"enabled": true` and `bounds`.
