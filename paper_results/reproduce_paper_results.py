#!/usr/bin/env python3
"""One-line reproduction of the paper's evaluation results and figures.

By default this re-runs every SoroSim evaluation (statics + dynamics across the
simulation rates the figures need) and then regenerates all figures into
``paper_results/paper_figures/``. A full re-run is long (hours): statics is
2 materials x 500 shapes x 5 N, and dynamics is 2 materials x 10 tests x 5 N x
3 sim-rates. Use --figures-only to rebuild just the figures from the existing
``paper_results/evaluation_results/`` in seconds.

    python paper_results/reproduce_paper_results.py                 # full re-run + figures
    python paper_results/reproduce_paper_results.py --dry-run       # smoke test (10 shapes, 1 dyn test) + stamped figures
    python paper_results/reproduce_paper_results.py --figures-only  # figures from existing results
    python paper_results/reproduce_paper_results.py --n-values 25 50 --sim-hz 500   # quick subset
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # paper_results/
PROJECT_ROOT = SCRIPT_DIR.parent  # repo root

# Sim-rates the dynamics figures consume (trajectory + error at 500 Hz, the
# realtime-factor panel at 200/500/1000 Hz). The statics figures are rate-free.
FIGURE_SIM_HZ = [200, 500, 1000]

# Dry-run smoke test: a tiny subset that still exercises the whole pipeline.
DRY_STATICS_SHAPES = 10  # shapes per material (vs 500)
DRY_DYNAMICS_TESTS = [1]  # dynamics test indices (vs all 10)
DRY_DYNAMICS_N = [
    25,
    50,
]  # N sweep for dynamics (vs 25..100; skips the slow high-N sims)
DRY_NOTE = "DRY RUN — example cases only, NOT paper results"


def run(cmd, label):
    print(
        f"\n{'=' * 72}\n# {label}\n$ {' '.join(str(c) for c in cmd)}\n{'=' * 72}",
        flush=True,
    )
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    print(
        f"[{label}] {'OK' if rc == 0 else f'FAIL({rc})'} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return rc


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip the simulations; rebuild figures from existing paper_results/evaluation_results/.",
    )
    p.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=None,
        help="Override the N sweep for the evaluations (default: per-config).",
    )
    p.add_argument(
        "--sim-hz",
        nargs="+",
        type=int,
        default=FIGURE_SIM_HZ,
        help=f"Dynamics sim-rates to sweep (default: {FIGURE_SIM_HZ}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Smoke test: only {DRY_STATICS_SHAPES} statics shapes/material and "
        f"dynamics test(s) {DRY_DYNAMICS_TESTS}, but still build the figures "
        "(stamped as a dry run). Verifies the pipeline quickly; NOT paper results.",
    )
    args = p.parse_args()

    if args.dry_run:
        print("\n" + "!" * 72)
        print(
            f"!! DRY RUN — {DRY_STATICS_SHAPES} statics shapes/material, "
            f"dynamics test(s) {DRY_DYNAMICS_TESTS}."
        )
        print("!! Example cases only, to verify the pipeline — NOT paper results.")
        print("!" * 72)

    py = sys.executable
    results = []

    if not args.figures_only:
        statics_cmd = [py, str(SCRIPT_DIR / "run_statics.py")]
        dynamics_cmd = [
            py,
            str(SCRIPT_DIR / "run_dynamics.py"),
            "--all-materials",
            "--sim-hz",
            *map(str, args.sim_hz),
        ]
        if args.n_values is not None:
            statics_cmd += ["--n-values", *map(str, args.n_values)]
            dynamics_cmd += ["--n-values", *map(str, args.n_values)]
        if args.dry_run:
            statics_cmd += ["--early-stop", str(DRY_STATICS_SHAPES)]
            dynamics_cmd += ["--tests", *map(str, DRY_DYNAMICS_TESTS)]
            if args.n_values is None:
                dynamics_cmd += ["--n-values", *map(str, DRY_DYNAMICS_N)]
        results.append(("statics evals", run(statics_cmd, "Run statics evaluations")))
        results.append(
            ("dynamics evals", run(dynamics_cmd, "Run dynamics evaluations"))
        )

    note_extra = ["--note", DRY_NOTE] if args.dry_run else []
    # The dry run only ran dynamics test(s) DRY_DYNAMICS_TESTS, so point the
    # trajectory panel at one of them (the default --steel-test 4 would have no data).
    dyn_extra = note_extra + (
        ["--steel-test", str(DRY_DYNAMICS_TESTS[0])] if args.dry_run else []
    )
    results.append(
        (
            "statics figures",
            run(
                [py, str(SCRIPT_DIR / "visualize_statics.py"), *note_extra],
                "Build statics figures",
            ),
        )
    )
    results.append(
        (
            "dynamics figures",
            run(
                [py, str(SCRIPT_DIR / "visualize_dynamics.py"), *dyn_extra],
                "Build dynamics figures",
            ),
        )
    )
    # sysid figure: self-skips (rc 0, with a clear message) if its recorded
    # session / identified model aren't present locally.
    results.append(
        (
            "sysid figure",
            run([py, str(SCRIPT_DIR / "visualize_sysid.py")], "Build sysid figure"),
        )
    )

    print(f"\n{'=' * 72}\nSummary\n{'=' * 72}")
    for name, rc in results:
        print(f"  {name:<24} {'OK' if rc == 0 else f'FAIL({rc})'}")
    print(f"\nFigures written to {SCRIPT_DIR / 'paper_figures'}/")
    if args.dry_run:
        print(
            "DRY RUN — example cases only; figures are stamped accordingly. "
            "Run without --dry-run for paper results."
        )
    return 0 if all(rc == 0 for _, rc in results) else 1


if __name__ == "__main__":
    sys.exit(main())
