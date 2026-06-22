#!/usr/bin/env python3
"""Generate the paper figure for the dynamics-evaluation results.

Single combined figure (PNG + PDF) at ``paper_results/paper_figures/dynamics_overview``,
laid out 1 row × 3 columns:

    (a) Trajectory column (left) — two stacked subplots for one
        representative test case at one (N, sim_hz). Top: tip position
        (x/y/z) vs time. Bottom: tip orientation (roll/pitch/yaw) vs time.
        Each axis is RGB-coloured. SoroSim reference is solid in the
        bolder/darker shade, MuJoCo simulation is dotted in the lighter
        shade — so the two are distinguishable even when the curves
        overlap.

    (b) Error column (middle) — mean / 95th / max tip-position error
        (% of rod length) vs N for both materials at one sim_hz (200 Hz
        by default, --error-sim-hz — the reference sampling rate and the
        practical operating regime; it is also the conservative choice,
        reporting the largest errors of the swept rates).

    (c) Realtime-factor column (right) — mean RTF vs N, pooled across
        steel and TPU at each sim_hz (single curve per rate).

Reads ``sweep_long_<timestamp>.csv`` per material under
``paper_results/evaluation_results/sorosim_dynamics_<material>/``. Per-test
pickles for the trajectory panel come from the per-test result directories.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from visualize_common import (
    STEEL_COLOR,
    TPU_COLOR,
    format_log_n_axis,
    format_log_y_decimal,
    save_figure,
)

SCRIPT_DIR = Path(__file__).resolve().parent  # paper_results/
PROJECT_ROOT = SCRIPT_DIR.parent  # repo root

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "figure.dpi": 300,
}

# Axis-component colours for the trajectory subplots. Reference uses the
# darker / more saturated shade; simulation uses a lighter shade so that
# overlapping curves remain individually visible alongside the line-style
# (solid vs dotted) cue.
REF_AXIS_COLORS = {
    "x": "#b22222",  # firebrick
    "y": "#2e7d32",  # darker green
    "z": "#1f3b8a",  # darker blue
}
SIM_AXIS_COLORS = {
    "x": "#ff8a80",  # light coral / salmon
    "y": "#7fc07f",  # light sage green
    "z": "#7fa7e6",  # light cornflower
}

# Distinct line styles for the four sim_hz values in panel (c).
SIM_HZ_LINESTYLES = {
    200: "-",
    500: "--",
    1000: "-.",
    5000: (0, (1, 1)),  # dotted-tight
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def latest_sweep_long(material_dir: Path) -> Path:
    candidates = sorted(
        material_dir.glob("sweep_long_*.csv"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"No sweep_long_*.csv found in {material_dir}")
    return candidates[-1]


def find_run_dir(
    material_dir: Path, test_num: int, base_name: str, sim_hz: int | None = None
) -> Path:
    pattern = f"*_{base_name}_{test_num}"
    candidates = sorted(
        material_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No run dir matching {pattern!r} in {material_dir}")
    if sim_hz is None:
        return candidates[0]
    for d in candidates:
        try:
            results = load_pickle(d)
            if any(int(r.get("sim_hz", -1)) == sim_hz for r in results):
                return d
        except Exception:
            continue
    return candidates[0]


def load_pickle(run_dir: Path) -> list:
    pkls = sorted((run_dir / "data").glob("*_full_results.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No *_full_results.pkl under {run_dir / 'data'}")
    with open(pkls[0], "rb") as f:
        return pickle.load(f)


def attach_realtime_ratio(
    long_df: pd.DataFrame, material_dir: Path, base_name: str
) -> pd.DataFrame:
    """Backfill realtime_ratio into sweep_long from per-test results CSVs
    when the driver-collected version was written before the column was
    added to the per-test schema."""
    if "realtime_ratio" in long_df.columns and long_df["realtime_ratio"].notna().any():
        return long_df

    rows = []
    for run_dir in sorted(material_dir.glob(f"*_{base_name}_*")):
        try:
            test_num = int(run_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        csv_files = list((run_dir / "data").glob("*_results.csv"))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[0])
        if "realtime_ratio" not in df.columns:
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "test_num": test_num,
                    "num_links": int(r["num_links"]),
                    "sim_hz": float(r["sim_hz"]),
                    "realtime_ratio": float(r["realtime_ratio"]),
                    "wall_time": float(r.get("wall_time", np.nan)),
                }
            )

    if not rows:
        return long_df

    rt = pd.DataFrame(rows)
    rt = rt.drop_duplicates(["test_num", "num_links", "sim_hz"], keep="last")
    # The long-form sweep CSV always carries realtime_ratio/wall_time columns
    # (collect_csv_rows writes them, here all-NaN). Drop them before the merge
    # so pandas doesn't suffix into realtime_ratio_x/_y and leave no plain
    # 'realtime_ratio' column for the RTF panel to read.
    long_df = long_df.drop(
        columns=[c for c in ("realtime_ratio", "wall_time") if c in long_df.columns]
    )
    return long_df.merge(rt, on=["test_num", "num_links", "sim_hz"], how="left")


def stats_per_N(df: pd.DataFrame, value_col: str, scale: float = 1.0):
    ok = df[df["status"] == "OK"]
    g = ok.groupby("num_links")[value_col]
    ns = np.array(sorted(ok["num_links"].unique()))
    if len(ns) == 0:
        return ns, np.array([]), np.array([]), np.array([])
    mean = np.array([g.get_group(n).mean() for n in ns]) / scale
    p95 = np.array([np.percentile(g.get_group(n), 95) for n in ns]) / scale
    mx = np.array([g.get_group(n).max() for n in ns]) / scale
    return ns, mean, p95, mx


# --------------------------------------------------------------------------- #
# Panel renderers (operate on supplied axes; outer figure handles layout)
# --------------------------------------------------------------------------- #


def render_trajectory(
    ax_pos,
    ax_ori,
    material_dir: Path,
    base_name: str,
    test_num: int,
    target_N: int,
    target_sim_hz: int,
):
    try:
        run_dir = find_run_dir(material_dir, test_num, base_name, sim_hz=target_sim_hz)
        results = load_pickle(run_dir)
    except (FileNotFoundError, OSError) as exc:
        for ax in (ax_pos, ax_ori):
            ax.text(
                0.5,
                0.5,
                "trajectory data\nunavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="0.4",
            )
        print(
            f"  [warn] no trajectory data for test {test_num} "
            f"@ {target_sim_hz} Hz: {exc}"
        )
        return
    chosen = min(
        results,
        key=lambda r: abs(r["num_links"] - target_N)
        + 1e6 * abs(r.get("sim_hz", 0) - target_sim_hz),
    )

    times = chosen["ref_times"]
    ref_pos = chosen["ref_tip_poses"][:, :3]
    ref_ori = chosen["ref_tip_poses"][:, 3:]
    sim_pos = chosen["sim_tip_poses"][:, :3]
    sim_ori = chosen["sim_tip_poses"][:, 3:]

    pos_keys = ["x", "y", "z"]
    ori_keys = ["x", "y", "z"]  # share colour palette with positional axes

    # Position subplot (top of trajectory column).
    for i, key in enumerate(pos_keys):
        ax_pos.plot(
            times,
            ref_pos[:, i],
            color=REF_AXIS_COLORS[key],
            linestyle="-",
            linewidth=1.5,
        )
        ax_pos.plot(
            times,
            sim_pos[:, i],
            color=SIM_AXIS_COLORS[key],
            linestyle=":",
            linewidth=1.7,
        )
    ax_pos.set_ylabel("position (m)")
    ax_pos.tick_params(axis="x", labelbottom=False)
    ax_pos.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # Orientation subplot (bottom of trajectory column).
    for i, key in enumerate(ori_keys):
        ax_ori.plot(
            times,
            ref_ori[:, i],
            color=REF_AXIS_COLORS[key],
            linestyle="-",
            linewidth=1.5,
        )
        ax_ori.plot(
            times,
            sim_ori[:, i],
            color=SIM_AXIS_COLORS[key],
            linestyle=":",
            linewidth=1.7,
        )
    ax_ori.set_xlabel("time (s)")
    ax_ori.set_ylabel("orientation (rad)")
    ax_ori.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # Compact 5-entry legend on the position subplot.
    legend_handles = [
        Line2D([], [], color=REF_AXIS_COLORS["x"], linewidth=1.6, label="x / roll"),
        Line2D([], [], color=REF_AXIS_COLORS["y"], linewidth=1.6, label="y / pitch"),
        Line2D([], [], color=REF_AXIS_COLORS["z"], linewidth=1.6, label="z / yaw"),
        Line2D([], [], color="none", linestyle="None", label=" "),
        Line2D([], [], color="0.3", linestyle="-", linewidth=1.6, label="Sorosim Ref."),
        Line2D([], [], color="0.6", linestyle=":", linewidth=1.8, label="MuJoCo"),
    ]
    ax_pos.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.94),
        frameon=False,
        fontsize=12,
        ncol=1,
        labelspacing=0.2,
        handlelength=1.4,
        borderpad=0.2,
    )

    short_material = "Steel" if "SpringSteel" in base_name else "TPU"
    ax_ori.text(
        0.98,
        0.04,
        f"{short_material} test {test_num}, N={chosen['num_links']}, {int(chosen['sim_hz'])} Hz",
        transform=ax_ori.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )


def render_error(
    ax,
    steel_long: pd.DataFrame,
    tpu_long: pd.DataFrame,
    steel_len: float,
    tpu_len: float,
    sim_hz: int,
):
    steel_at = steel_long[steel_long["sim_hz"] == sim_hz]
    tpu_at = tpu_long[tpu_long["sim_hz"] == sim_hz]

    if steel_at.empty and tpu_at.empty:
        available = sorted(
            set(steel_long["sim_hz"].unique()) | set(tpu_long["sim_hz"].unique())
        )
        print(
            f"WARNING: no sweep rows at sim_hz={sim_hz} for the error panel; "
            f"available rates: {available}. Re-run with --error-sim-hz set to "
            "one of those (or sweep run_dynamics.py with --sim-hz including "
            f"{sim_hz})."
        )
        ax.text(
            0.5,
            0.5,
            f"no data at {sim_hz} Hz",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    n_s, mean_s, p95_s, max_s = stats_per_N(
        steel_at, "mean_tip_position_error", steel_len / 100.0
    )
    n_t, mean_t, p95_t, max_t = stats_per_N(
        tpu_at, "mean_tip_position_error", tpu_len / 100.0
    )

    if len(n_s):
        ax.plot(
            n_s,
            mean_s,
            color=STEEL_COLOR,
            linestyle="-",
            linewidth=2.0,
            marker="o",
            label="Steel mean",
        )
        ax.plot(
            n_s,
            p95_s,
            color=STEEL_COLOR,
            linestyle="--",
            linewidth=1.2,
            label="Steel 95%",
        )
        ax.plot(
            n_s,
            max_s,
            color=STEEL_COLOR,
            linestyle=":",
            linewidth=1.2,
            label="Steel max",
        )
    if len(n_t):
        ax.plot(
            n_t,
            mean_t,
            color=TPU_COLOR,
            linestyle="-",
            linewidth=2.0,
            marker="s",
            label="TPU mean",
        )
        ax.plot(
            n_t, p95_t, color=TPU_COLOR, linestyle="--", linewidth=1.2, label="TPU 95%"
        )
        ax.plot(
            n_t, max_t, color=TPU_COLOR, linestyle=":", linewidth=1.2, label="TPU max"
        )

    ax.text(
        0.85,
        0.02,
        f"{sim_hz} Hz",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.35",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (links)")
    ax.set_ylabel("Tip error (% of rod length)")
    n_ref = n_s if len(n_s) else n_t
    if len(n_ref):
        format_log_n_axis(ax, n_ref)
    format_log_y_decimal(ax)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(
        frameon=False,
        loc="center right",
        bbox_to_anchor=(1.0, 0.63),
        fontsize=12,
        ncol=2,
        columnspacing=0.9,
        labelspacing=0.22,
        borderpad=0.3,
        handlelength=1.6,
        handletextpad=0.5,
    )


def render_rtf_pooled(
    ax, steel_long: pd.DataFrame, tpu_long: pd.DataFrame, sim_hz_list: list[int]
):
    """Pool steel + TPU and plot one curve per sim_hz."""
    if (
        "realtime_ratio" not in steel_long.columns
        and "realtime_ratio" not in tpu_long.columns
    ):
        ax.text(
            0.5,
            0.5,
            "realtime_ratio missing",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    pooled = pd.concat([steel_long, tpu_long], ignore_index=True)
    pooled = pooled[pooled["status"] == "OK"]

    n_ref = None
    cmap = plt.get_cmap("BuPu")
    rate_colors = {
        hz: cmap(0.45 + 0.45 * i / max(len(sim_hz_list) - 1, 1))
        for i, hz in enumerate(sim_hz_list)
    }
    sim_hz_500_curve = None
    for hz in sim_hz_list:
        sub = pooled[pooled["sim_hz"] == hz]
        if sub.empty or sub["realtime_ratio"].isna().all():
            continue
        ns, mean_rtf, _, _ = stats_per_N(sub, "realtime_ratio")
        if not len(ns):
            continue
        ax.plot(
            ns,
            mean_rtf,
            color=rate_colors[hz],
            linestyle=SIM_HZ_LINESTYLES.get(hz, "-"),
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=f"{hz} Hz",
        )
        if hz == 500:
            sim_hz_500_curve = (
                np.asarray(ns, dtype=float),
                np.asarray(mean_rtf, dtype=float),
            )
        if n_ref is None or len(ns) > len(n_ref):
            n_ref = ns

    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (links)")
    ax.set_ylabel("Realtime factor (sim / wall)")
    if n_ref is not None and len(n_ref):
        format_log_n_axis(ax, n_ref)
    format_log_y_decimal(ax)

    # Practical operating regime: N in [30, 50], sim_hz <= 500 Hz
    # (i.e., the area above the 500 Hz curve in this RTF plot).
    if sim_hz_500_curve is not None:
        ns_500, rtf_500 = sim_hz_500_curve
        log_ns = np.log10(ns_500)
        log_rtf = np.log10(rtf_500)
        intermediate = ns_500[(ns_500 > 30) & (ns_500 < 50)]
        n_grid = np.unique(np.concatenate(([30.0], intermediate, [50.0])))
        rtf_grid = 10 ** np.interp(np.log10(n_grid), log_ns, log_rtf)
        ymin, ymax = ax.get_ylim()
        ax.fill_between(
            n_grid,
            rtf_grid,
            np.full_like(n_grid, ymax),
            facecolor="#b6e2b6",
            alpha=0.55,
            edgecolor="none",
            label="Practical operating regime",
            zorder=0,
        )
        ax.set_ylim(ymin, ymax)

    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.17),
        fontsize=12,
        labelspacing=0.22,
        borderpad=0.3,
        handlelength=1.6,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steel-test",
        type=int,
        default=4,
        help="Test number used for the trajectory panel (steel default; "
        "renumbered test 4 = original SoroSim steel test 5).",
    )
    parser.add_argument(
        "--tpu-test",
        type=int,
        default=None,
        help="If --use-tpu-traj, test number for trajectory panel.",
    )
    parser.add_argument(
        "--use-tpu-traj",
        action="store_true",
        help="Use TPU instead of steel for the trajectory panel.",
    )
    parser.add_argument(
        "--target-N",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--target-sim-hz",
        type=int,
        default=200,
        help="sim_hz for the trajectory panel (a). Default 200 — the "
        "reference sampling rate and the practical operating regime "
        "highlighted in panel (c).",
    )
    parser.add_argument(
        "--error-sim-hz",
        type=int,
        default=200,
        help=(
            "sim_hz for the error panel (b). Default 200 (matches the "
            "reference sampling rate and panel (c)'s practical regime); "
            "this is the conservative choice — Euler truncation makes "
            "200 Hz report the LARGEST errors of the swept rates."
        ),
    )
    parser.add_argument(
        "--rtf-sim-hz",
        nargs="+",
        type=int,
        default=[200, 500, 1000],
    )
    parser.add_argument("--steel-length", type=float, default=0.6)
    parser.add_argument("--tpu-length", type=float, default=0.4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "paper_figures",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Stamp this note on the figure (e.g. a dry-run marker).",
    )
    args = parser.parse_args()

    steel_dir = SCRIPT_DIR / "evaluation_results/sorosim_dynamics_steel"
    tpu_dir = SCRIPT_DIR / "evaluation_results/sorosim_dynamics_tpu"

    steel_long_path = latest_sweep_long(steel_dir)
    tpu_long_path = latest_sweep_long(tpu_dir)
    print(f"Steel sweep CSV: {steel_long_path.name}")
    print(f"TPU   sweep CSV: {tpu_long_path.name}")
    steel_long = pd.read_csv(steel_long_path)
    tpu_long = pd.read_csv(tpu_long_path)
    steel_long = attach_realtime_ratio(steel_long, steel_dir, "SpringSteelRodMuJoCo")
    tpu_long = attach_realtime_ratio(tpu_long, tpu_dir, "TPURodMuJoCo")

    plt.rcParams.update(PAPER_STYLE)

    fig = plt.figure(figsize=(13.5, 4.2))
    PANEL_LABEL_Y = 0.04
    AXES_BOTTOM = 0.22
    AXES_TOP = 0.95
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        left=0.065,
        right=0.985,
        top=AXES_TOP,
        bottom=AXES_BOTTOM,
        wspace=0.22,
    )
    # Trajectory column: position on top, orientation on bottom.
    traj_gs = outer[0, 0].subgridspec(2, 1, hspace=0.15)
    ax_pos = fig.add_subplot(traj_gs[0])
    ax_ori = fig.add_subplot(traj_gs[1])
    ax_err = fig.add_subplot(outer[0, 1])
    ax_rtf = fig.add_subplot(outer[0, 2])

    if args.use_tpu_traj:
        traj_dir = tpu_dir
        traj_base = "TPURodMuJoCo"
        traj_test = args.tpu_test or 5
    else:
        traj_dir = steel_dir
        traj_base = "SpringSteelRodMuJoCo"
        traj_test = args.steel_test

    render_trajectory(
        ax_pos,
        ax_ori,
        traj_dir,
        traj_base,
        traj_test,
        args.target_N,
        args.target_sim_hz,
    )
    render_error(
        ax_err,
        steel_long,
        tpu_long,
        args.steel_length,
        args.tpu_length,
        args.error_sim_hz,
    )
    render_rtf_pooled(ax_rtf, steel_long, tpu_long, args.rtf_sim_hz)

    # Panel labels at a common fig-y, centred on each column.
    def _x_center_axes(*axes):
        boxes = [a.get_position() for a in axes]
        x0 = min(b.x0 for b in boxes)
        x1 = max(b.x1 for b in boxes)
        return 0.5 * (x0 + x1)

    fig.text(
        _x_center_axes(ax_pos, ax_ori),
        PANEL_LABEL_Y,
        "(a) Tip trajectory — position and orientation",
        ha="center",
        va="bottom",
        fontsize=16,
    )
    fig.text(
        _x_center_axes(ax_err),
        PANEL_LABEL_Y,
        "(b) Tip-position error",
        ha="center",
        va="bottom",
        fontsize=16,
    )
    fig.text(
        _x_center_axes(ax_rtf),
        PANEL_LABEL_Y,
        "(c) Realtime factor",
        ha="center",
        va="bottom",
        fontsize=16,
    )

    save_figure(fig, args.output_dir / "dynamics_overview", note=args.note)


if __name__ == "__main__":
    sys.exit(main() or 0)
