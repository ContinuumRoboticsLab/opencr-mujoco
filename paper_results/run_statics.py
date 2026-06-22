#!/usr/bin/env python3
"""Run the SoroSim static-equilibrium evaluations (both materials).

Runs ``evaluate.py`` for each statics config in turn (spring steel, then TPU),
sequentially so realtime-factor numbers are not skewed by core contention. Each
material caps at 500 shapes via the per-config ``early_stop``.

    python paper_results/run_statics.py                 # both materials, default N
    python paper_results/run_statics.py --n-values 25 50 100
    python paper_results/run_statics.py --skip tpu_statics
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # paper_results/
PROJECT_ROOT = SCRIPT_DIR.parent  # repo root (eval working dir)

STATICS_CONFIGS = ["spring_steel_statics", "tpu_statics"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=None,
        help="N values to sweep (forwarded to evaluate.py). Default: per-config.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Config names to skip (e.g. --skip tpu_statics).",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=None,
        help="Cap the number of shapes per material (forwarded to evaluate.py). "
        "Default: the per-config cap (500).",
    )
    args = parser.parse_args()

    configs = [c for c in STATICS_CONFIGS if c not in args.skip]
    if not configs:
        print("Nothing to run.", file=sys.stderr)
        return 1

    results = []
    for cfg in configs:
        cmd = [sys.executable, str(SCRIPT_DIR / "evaluate.py"), "--config", cfg]
        if args.n_values is not None:
            cmd += ["--n-values", *map(str, args.n_values)]
        if args.early_stop is not None:
            cmd += ["--early-stop", str(args.early_stop)]
        print(f"\n$ {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        results.append((cfg, rc, time.time() - t0))
        if rc != 0:
            print(f"\n[!] {cfg} exited with code {rc} — continuing.", flush=True)

    print("\n" + "=" * 60 + "\nStatics summary\n" + "=" * 60)
    for cfg, rc, dt in results:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  {cfg:<28} {status:<10} {dt:>8.1f}s")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
