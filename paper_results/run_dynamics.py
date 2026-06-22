#!/usr/bin/env python3
"""
Sweep dynamics evaluations across N (links) and sim_hz (simulation rate).

For each material x sim_hz x test, run `evaluate.py` and aggregate per-test
results into a (sim_hz x N) summary table per material:
  - cells = mean tip position error across tests (OK, per-N rows only)
  - rows annotated with the per-sim_hz count of UNSTABLE and FAILED tests

The test bank is the set of reference files present for each material
(SpringSteelRodMuJoCo_1..N / TPURodMuJoCo_1..N, discovered by globbing the
reference dir). Default sim_hz is 1000 Hz (pass --sim-hz to sweep multiple
rates); default N sweep is 25 / 35 / 50 / 70 / 100.

Usage:
    # Single material, custom sim_hz sweep
    python paper_results/run_dynamics.py --config sorosim_dynamics_steel \
        --n-values 25 50 100 --sim-hz 1000 5000 10000

    # Both materials, default sweep
    python paper_results/run_dynamics.py --all-materials

    # Quick smoke (one test, two rates)
    python paper_results/run_dynamics.py --config sorosim_dynamics_steel \
        --tests 1 --n-values 25 50 --sim-hz 1000 5000
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent  # paper_results/
PROJECT_ROOT = SCRIPT_DIR.parent  # repo root (eval working dir)

# Material name → tip-release reference base name
MATERIALS = {
    "tpu": "TPURodMuJoCo",
    "steel": "SpringSteelRodMuJoCo",
}

# Material → eval config name
MATERIAL_CONFIG = {
    "tpu": "sorosim_dynamics_tpu",
    "steel": "sorosim_dynamics_steel",
}

DEFAULT_SIM_HZ = [1000]


@lru_cache(maxsize=None)
def load_test_metadata(reference_dir, test_type):
    """Load damping and wrench info from a reference file.

    Only the first two rows carry metadata (damping + gravity, then the
    wrenches), so we read just those instead of parsing the full ~2000-row
    time series. Cached per (reference_dir, test_type) since the metadata is
    sim_hz-independent but the function is called once per damping-selection
    pass and once per sim_hz in the sweep loop.
    """
    filepath = Path(reference_dir) / f"{test_type}.txt"
    if not filepath.exists():
        return None, None, None
    try:
        data = np.loadtxt(filepath, delimiter="\t", max_rows=2)
        damping = data[0, 0]
        mid_wrench = data[1, 1:7]
        tip_wrench = data[1, 7:13]
        return damping, mid_wrench, tip_wrench
    except Exception:
        return None, None, None


def check_for_instability(output):
    """Return (has_instability, time_or_None) by scanning subprocess output."""
    instability_patterns = [
        r"WARNING.*Nan.*Inf.*huge value.*QACC",
        r"simulation is unstable",
        r"The simulation is unstable",
    ]
    for pattern in instability_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            time_match = re.search(r"Time\s*=\s*([\d]+\.?[\d]*)", output)
            if time_match:
                try:
                    return True, float(time_match.group(1))
                except ValueError:
                    return True, None
            return True, None
    return False, None


_CSV_LINE = re.compile(r"^\s*-\s*CSV:\s*(.+)$", re.MULTILINE)


def parse_csv_path(stdout: str) -> Path | None:
    """Pull the per-test results CSV path out of evaluate.py's stdout."""
    matches = _CSV_LINE.findall(stdout)
    if not matches:
        return None
    return Path(matches[-1].strip())


def run_one(config_name, test_type, n_values, sim_hz, force_ramp, hold, dry_run):
    """Run a single evaluate.py invocation. Returns (status, csv_path, elapsed_s).

    force_ramp / hold are forwarded only when explicitly set; when None,
    evaluate.py falls back to the per-config values (so the driver does not
    silently override the config's settle times).
    """
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate.py"),
        "--config",
        config_name,
        "--test-type",
        test_type,
        "--n-values",
        *map(str, n_values),
        "--sim-hz",
        str(sim_hz),
        "--no-visualize",
    ]
    if force_ramp is not None:
        cmd += ["--force-ramp-time", str(force_ramp)]
    if hold is not None:
        cmd += ["--hold-time", str(hold)]
    if dry_run:
        print("  DRY: " + " ".join(cmd))
        return "DRY", None, 0.0

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        elapsed = time.time() - t0
        unstable, t_unstable = check_for_instability(result.stdout + result.stderr)
        if unstable:
            status = (
                f"UNSTABLE@{t_unstable:.1f}s" if t_unstable is not None else "UNSTABLE"
            )
        else:
            status = "OK"
        csv_path = parse_csv_path(result.stdout)
        return status, csv_path, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        unstable, t_unstable = check_for_instability(
            (e.stdout or "") + (e.stderr or "")
        )
        status = (
            f"UNSTABLE@{t_unstable:.1f}s"
            if unstable and t_unstable is not None
            else ("UNSTABLE" if unstable else f"FAILED({e.returncode})")
        )
        csv_path = parse_csv_path(e.stdout or "")
        return status, csv_path, elapsed


def collect_csv_rows(csv_path, test_num, sim_hz, run_status, n_values):
    """Read evaluate.py's per-test CSV and emit one long-form row per N.

    A single subprocess runs all N values and yields ONE run-level status
    (from scanning stdout), but only the N(s) that actually diverged should
    count as unstable. So each N's status is derived from its own row — a
    finite tip error is OK; a NaN/Inf error (MuJoCo divergence) or a row that
    never got written (crashed before reaching that N) is not OK and is tagged
    with the run-level reason (UNSTABLE@.../FAILED(rc)) when it carries one,
    else "UNSTABLE". This stops one unstable N from tainting the stable N rows
    of the same test in the aggregate.
    """
    by_n = {}
    if csv_path is not None and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                n = int(r["num_links"])
                mean_err = float(r["mean_tip_position_error"])
                by_n[n] = {
                    "num_links": n,
                    "sim_hz": float(r["sim_hz"]),
                    "mean_tip_position_error": mean_err,
                    "max_tip_position_error": float(r["max_tip_position_error"]),
                    "wall_time": (
                        float(r["wall_time"]) if "wall_time" in r else np.nan
                    ),
                    "realtime_ratio": (
                        float(r["realtime_ratio"]) if "realtime_ratio" in r else np.nan
                    ),
                    # Finite error => this N completed; NaN/Inf => it diverged.
                    "status": "OK" if np.isfinite(mean_err) else "UNSTABLE",
                }
        except Exception as exc:
            print(f"  [warn] could not parse {csv_path}: {exc}")

    # Status for an N that produced no finite row: prefer the run-level reason.
    missing_status = (
        run_status if run_status.startswith(("UNSTABLE", "FAILED")) else "UNSTABLE"
    )
    rows = []
    for n in n_values:
        if n in by_n:
            rows.append({"test_num": test_num, **by_n[n]})
        else:
            rows.append(
                {
                    "test_num": test_num,
                    "num_links": n,
                    "sim_hz": float(sim_hz),
                    "mean_tip_position_error": np.nan,
                    "max_tip_position_error": np.nan,
                    "wall_time": np.nan,
                    "realtime_ratio": np.nan,
                    "status": missing_status,
                }
            )
    return rows


def fmt_err_mm(x):
    if pd.isna(x):
        return "  --  "
    return f"{x*1000:6.2f}"


def print_aggregate_table(material, df_long):
    """Pivot the long-form dataframe into a (sim_hz x N) error/unstable-count table."""
    print(f"\n{'='*100}")
    print(f"AGGREGATE — {material.upper()}")
    print(f"{'='*100}")

    if df_long.empty:
        print("(no results)")
        return None, None, None

    ok = df_long[df_long["status"] == "OK"]
    unstable = df_long[df_long["status"].str.startswith("UNSTABLE", na=False)]
    # FAILED(rc) = subprocess crash without an instability signature. Counted
    # separately so crashes don't silently vanish from the table.
    failed = df_long[df_long["status"].str.startswith("FAILED", na=False)]

    # Mean across tests for OK rows
    if not ok.empty:
        err_pivot = ok.pivot_table(
            index="sim_hz",
            columns="num_links",
            values="mean_tip_position_error",
            aggfunc="mean",
        ).sort_index()
    else:
        err_pivot = pd.DataFrame()

    # Unstable / failed counts per sim_hz — count unique tests, not rows.
    def _count_unique_tests(sub):
        if sub.empty:
            return pd.Series(dtype=int)
        return sub.drop_duplicates(["test_num", "sim_hz"]).groupby("sim_hz").size()

    u_pivot = _count_unique_tests(unstable)
    f_pivot = _count_unique_tests(failed)

    if err_pivot.empty:
        # No OK rows: just print unstable counts
        print("All cells UNSTABLE / FAILED.")
        ns = []
    else:
        ns = sorted(c for c in err_pivot.columns)

    # Header
    rates = sorted(df_long["sim_hz"].unique())
    header = f"{'sim_hz':>10} " + " ".join(f"N={int(n):>4}" for n in ns)
    if ns:
        header += "    unstable"
    else:
        header += "  unstable"
    print(header)
    print("-" * len(header))
    for hz in rates:
        cells = []
        for n in ns:
            v = (
                err_pivot.at[hz, n]
                if (hz in err_pivot.index and n in err_pivot.columns)
                else np.nan
            )
            cells.append(f"{fmt_err_mm(v)} mm")
        u_cnt = int(u_pivot.get(hz, 0))
        f_cnt = int(f_pivot.get(hz, 0))
        notes = []
        if u_cnt:
            notes.append(f"{u_cnt} unstable")
        if f_cnt:
            notes.append(f"{f_cnt} failed")
        line = f"{int(hz):>10} " + " ".join(cells)
        line += f"   [{', '.join(notes)}]" if notes else "   "
        print(line)

    return err_pivot, u_pivot, f_pivot


def write_sweep_csvs(material, df_long, err_pivot, u_pivot, f_pivot, timestamp):
    """Persist long-form + aggregate CSVs alongside the per-config results dir."""
    out_dir = SCRIPT_DIR / "evaluation_results" / MATERIAL_CONFIG[material]
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = out_dir / f"sweep_long_{timestamp}.csv"
    df_long.to_csv(long_path, index=False)

    if err_pivot is not None and not err_pivot.empty:
        agg = err_pivot.copy()
        agg.columns = [f"N={int(c)}_mean_pos_err_m" for c in agg.columns]
        agg.insert(0, "failed_count", [int(f_pivot.get(hz, 0)) for hz in agg.index])
        agg.insert(0, "unstable_count", [int(u_pivot.get(hz, 0)) for hz in agg.index])
        agg_path = out_dir / f"sweep_aggregate_{timestamp}.csv"
        agg.to_csv(agg_path, index_label="sim_hz")
    else:
        agg_path = None

    print(f"  long-form CSV : {long_path}")
    if agg_path is not None:
        print(f"  aggregate CSV : {agg_path}")


def discover_tests(reference_dir, base_name):
    """Return the sorted list of test indices present for a material.

    The test bank is whatever reference files exist (e.g. SpringSteelRodMuJoCo_1
    .. _20); the trimmed dataset itself defines the bank, so there is no
    selection step.
    """
    ref = Path(reference_dir)
    nums = []
    for f in ref.glob(f"{base_name}_*.txt"):
        suffix = f.stem[len(base_name) + 1 :]
        if suffix.isdigit():
            nums.append(int(suffix))
    return sorted(nums)


def run_material(material, args, timestamp):
    config_name = MATERIAL_CONFIG[material]
    base_name = MATERIALS[material]

    # Explicit --tests list takes precedence; otherwise run every test present
    # in the reference dir (the trimmed dataset defines the bank).
    if args.tests:
        kept_tests = sorted(set(args.tests))
        bank_desc = f"explicit list ({len(kept_tests)} tests): {kept_tests}"
    else:
        kept_tests = discover_tests(args.reference_dir, base_name)
        bank_desc = f"all {len(kept_tests)} tests present: {kept_tests}"

    print(f"\n{'#'*100}")
    print(
        f"Material: {material.upper()}   {bank_desc}   "
        f"N={args.n_values}   sim_hz={args.sim_hz}"
    )
    print(f"{'#'*100}\n")

    long_rows = []
    test_summary = []  # (sim_hz, test_num, status, damping, mid_wrench, tip_wrench)
    interrupted = False

    for sim_hz in args.sim_hz:
        if interrupted:
            break
        print(f"\n{'='*80}")
        print(f"  Sweep block: {material.upper()} @ sim_hz={sim_hz}")
        print(f"{'='*80}\n")

        for test_num in kept_tests:
            test_type = f"{base_name}_{test_num}"
            damping, mid_w, tip_w = load_test_metadata(args.reference_dir, test_type)
            print(
                f"  [{material} sim_hz={sim_hz}] test {test_num:2d} ({test_type})"
                + (f"  damping={damping:.4f}" if damping is not None else "")
            )
            try:
                status, csv_path, elapsed = run_one(
                    config_name,
                    test_type,
                    args.n_values,
                    sim_hz,
                    args.force_ramp_time,
                    args.hold_time,
                    args.dry_run,
                )
            except KeyboardInterrupt:
                # Stop the whole sweep (all sim_hz blocks), not just this one
                print("\nInterrupted — saving partial results.")
                interrupted = True
                break

            if not args.dry_run:
                print(f"    → {status}   ({elapsed:.1f}s)")
                long_rows.extend(
                    collect_csv_rows(csv_path, test_num, sim_hz, status, args.n_values)
                )

            test_summary.append(
                {
                    "sim_hz": sim_hz,
                    "test_num": test_num,
                    "status": status,
                    "damping": damping,
                    "mid_wrench": mid_w,
                    "tip_wrench": tip_w,
                }
            )

    if args.dry_run:
        return True

    df_long = pd.DataFrame(long_rows)
    err_pivot, u_pivot, f_pivot = print_aggregate_table(material, df_long)
    write_sweep_csvs(material, df_long, err_pivot, u_pivot, f_pivot, timestamp)

    # Per-test status table at the end (rows = test, columns = sim_hz)
    if test_summary:
        ts_df = pd.DataFrame(test_summary)
        status_pivot = ts_df.pivot_table(
            index="test_num", columns="sim_hz", values="status", aggfunc="first"
        )
        print(f"\nPer-test status grid — {material.upper()}")
        print(status_pivot.to_string())

    return all(str(entry["status"]).startswith("OK") for entry in test_summary)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep dynamics evaluations across N and sim_hz."
    )
    parser.add_argument(
        "--config",
        choices=list(MATERIAL_CONFIG.values()),
        help="Run a single material (e.g. sorosim_dynamics_steel).",
    )
    parser.add_argument(
        "--all-materials",
        action="store_true",
        help="Run both steel and TPU. Overrides --config when set.",
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[25, 35, 50, 70, 100],
        help="N values to evaluate. Default: [25, 35, 50, 70, 100].",
    )
    parser.add_argument(
        "--sim-hz",
        nargs="+",
        type=int,
        default=DEFAULT_SIM_HZ,
        help=f"Simulation rates to sweep. Default: {DEFAULT_SIM_HZ}.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Explicit list of test indices to run (e.g. --tests 1 2 5). "
            "Default: every test present in the reference dir."
        ),
    )
    parser.add_argument(
        "--force-ramp-time",
        type=float,
        default=None,
        help="Override force/gravity ramp time (s). Default: use the config value.",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=None,
        help="Override hold time (s). Default: use the config value.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="data/reference/sorosim/sorosim_dynamics",
        help="Reference data directory (test bank + damping/wrench metadata).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full subprocess command list without executing.",
    )

    args = parser.parse_args()

    # Resolve a relative reference dir against the repo root so test discovery
    # and metadata reads work regardless of the working directory.
    ref_dir = Path(args.reference_dir)
    if not ref_dir.is_absolute():
        ref_dir = PROJECT_ROOT / ref_dir
    args.reference_dir = str(ref_dir)

    if args.all_materials:
        materials = ["steel", "tpu"]
    elif args.config:
        materials = ["tpu" if "tpu" in args.config else "steel"]
    else:
        parser.error("specify --config <name> or --all-materials")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_t0 = time.time()
    all_ok = True
    for material in materials:
        all_ok = run_material(material, args, timestamp) and all_ok

    print(f"\nTotal sweep wall time: {time.time() - overall_t0:.1f}s")
    if not all_ok:
        print("ERROR: one or more dynamics tests FAILED (see status grids above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
