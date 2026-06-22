#!/usr/bin/env python3
"""Generate the paper figures for the static-evaluation results.

Two standalone figures (each saved as PNG + PDF):

    (a) 3D overlay of sample reference shapes (grey solid) vs MuJoCo
        solutions (blue dotted) at N=50 for spring steel (default 10
        shapes, --num-shapes), with mid- and tip-force vectors drawn as
        black arrows. Requires N=50 rows in the sweep results.
    (b) Tip error (% of rod length) vs N for both materials, log-log,
        with mean / 95th-percentile / max lines per material.

Reads the latest completed runs under ``paper_results/evaluation_results/sorosim_statics/``;
writes the statics figures into ``paper_results/paper_figures/`` by default.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Headless-friendly default
import matplotlib

matplotlib.use("Agg")

from visualize_common import (
    STEEL_COLOR,
    TPU_COLOR,
    format_log_n_axis,
    format_log_y_decimal,
    save_figure,
)

SCRIPT_DIR = Path(__file__).resolve().parent  # paper_results/
PROJECT_ROOT = SCRIPT_DIR.parent  # repo root
# Allow `from opencr_mujoco...` imports when run as paper_results/visualize_statics.py.
sys.path.insert(0, str(PROJECT_ROOT))

# Paper-style rcParams (mirrors EvaluationVisualizer.PAPER_STYLE).
PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
}

REF_COLOR = "0.45"
SIM_COLOR = "tab:blue"  # MuJoCo rod in panel (a)
ARROW_COLOR = "tab:red"  # applied wrench arrows


def find_latest_run(material_glob: str, root: Path) -> Path:
    """Pick the newest result dir (by mtime) that has a results CSV.

    Skipping runs without a ``*_results.csv`` avoids selecting an incomplete or
    in-progress run (e.g. a dir created but not yet populated), which otherwise
    fails downstream with a confusing StopIteration.
    """
    candidates = sorted(
        (p for p in root.glob(material_glob) if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if any((p / "data").glob("*_results.csv")):
            return p
    raise FileNotFoundError(
        f"No completed statics run (with *_results.csv) matching "
        f"{material_glob!r} under {root}"
    )


def load_run(run_dir: Path) -> dict:
    """Load CSV, link-positions pickle, and run_meta for one statics run."""
    data_dir = run_dir / "data"
    csv_path = next(data_dir.glob("*_results.csv"))
    pkl_path = next(data_dir.glob("*_link_positions.pkl"))
    meta_path = next(data_dir.glob("*_run_meta.json"))

    df = pd.read_csv(csv_path)
    with open(pkl_path, "rb") as f:
        positions = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    R = meta.get("frame_conversion_file_to_mujoco")
    if R is not None:
        R = np.asarray(R, dtype=float)
    return {
        "run_dir": run_dir,
        "csv_path": csv_path,
        "df": df,
        "positions": positions,
        "frame_conversion": R,
    }


def parse_wrench(s: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in s.split(","))


def panel_a_sample_shapes(
    ax,
    steel_run: dict,
    num_shapes: int,
    seed: int,
    rod_length: float,
):
    """Panel (a): ref + sim shape overlay with force arrows."""
    from opencr_mujoco.evaluation.reference_data_loader import ReferenceDataLoader

    df = steel_run["df"]
    df_n50 = df[df["N"] == 50].reset_index(drop=True)
    if len(df_n50) == 0:
        raise RuntimeError("No N=50 rows found in steel CSV")

    rng = np.random.default_rng(seed)
    take = min(num_shapes, len(df_n50))

    # Farthest-point sampling on tip positions to spread the chosen shapes
    # across the reachable workspace (instead of a uniform random draw, which
    # tends to clump in dense regions of the test bank).
    tip_pts = np.stack(
        [np.array(parse_wrench(s)) for s in df_n50["ref_tip_position_str"]]
    )
    chosen_idx = [int(rng.integers(0, len(tip_pts)))]
    min_dist = np.linalg.norm(tip_pts - tip_pts[chosen_idx[0]], axis=1)
    while len(chosen_idx) < take:
        nxt = int(np.argmax(min_dist))
        chosen_idx.append(nxt)
        new_dist = np.linalg.norm(tip_pts - tip_pts[nxt], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    chosen_idx = np.array(chosen_idx)

    # Reference loader uses the run's saved frame_conversion so reference
    # 3-vectors come back in MuJoCo frame, matching the sim positions.
    ref_loader = ReferenceDataLoader(
        PROJECT_ROOT / "data/reference/sorosim",
        frame_conversion=steel_run["frame_conversion"],
    )
    ref_data, _, _ = ref_loader.load_sorosim_statics_csv("SpringSteelRodMuJoCo")

    # Resting (un-loaded) shape: straight rod along world Z from origin to
    # (0, 0, rod_length). Drawn first so it sits behind the deformed shapes.
    rest_z = np.linspace(0.0, rod_length, 50)
    ax.plot(
        np.zeros_like(rest_z),
        np.zeros_like(rest_z),
        rest_z,
        color="0.15",
        linestyle=(0, (4, 2)),
        linewidth=1.6,
        zorder=1,
        alpha=0.7,
    )

    # Pre-scan to set a single force-arrow scale across the panel.
    all_force_mags = []
    for i in chosen_idx:
        mid = parse_wrench(df_n50.loc[i, "mid_wrench_str"])
        tip = parse_wrench(df_n50.loc[i, "tip_wrench_str"])
        all_force_mags.append(np.linalg.norm(mid[:3]))
        all_force_mags.append(np.linalg.norm(tip[:3]))
    max_force = max(all_force_mags) if all_force_mags else 1.0
    arrow_scale = (0.12 * rod_length) / max_force  # m per N

    # Use the global df row index (CSV is in the same order as the pickle).
    for i in chosen_idx:
        global_idx = df.index[df["N"] == 50][i]
        sim_pts = np.asarray(steel_run["positions"][global_idx], dtype=float)
        mid = parse_wrench(df_n50.loc[i, "mid_wrench_str"])
        tip = parse_wrench(df_n50.loc[i, "tip_wrench_str"])

        # Look up reference shape by (mid, tip) tuple match.
        ref_pts = None
        for key, pts in ref_data.items():
            if len(key) == 3:
                key_mid, key_tip, _ = key
                if tuple(mid) == tuple(key_mid) and tuple(tip) == tuple(key_tip):
                    ref_pts = np.asarray(pts, dtype=float)
                    break
        if ref_pts is None:
            print(f"  [warn] no ref match for sim row {global_idx}; skipping")
            continue

        ax.plot(
            ref_pts[:, 0],
            ref_pts[:, 1],
            ref_pts[:, 2],
            color=REF_COLOR,
            linestyle="-",
            linewidth=2.0,
            zorder=2,
        )
        ax.plot(
            sim_pts[:, 0],
            sim_pts[:, 1],
            sim_pts[:, 2],
            color=SIM_COLOR,
            linestyle=":",
            linewidth=2.0,
            zorder=3,
        )

        # Force arrows: mid at chain midpoint, tip at last sim point.
        n_pts = len(sim_pts)
        mid_root = sim_pts[n_pts // 2]
        tip_root = sim_pts[-1]
        mid_force = np.asarray(mid[:3]) * arrow_scale
        tip_force = np.asarray(tip[:3]) * arrow_scale
        for root, vec in [(mid_root, mid_force), (tip_root, tip_force)]:
            ax.quiver(
                root[0],
                root[1],
                root[2],
                vec[0],
                vec[1],
                vec[2],
                color=ARROW_COLOR,
                linewidth=1.4,
                arrow_length_ratio=0.25,
                zorder=4,
            )

    # X / Y labels close to ticks (small positive pad); Z keeps a bit more so
    # it doesn't collide with the rod geometry on the right side.
    ax.set_xlabel("X (m)", labelpad=0)
    ax.set_ylabel("Y (m)", labelpad=0)
    ax.set_zlabel("Z (m)", labelpad=0)
    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", pad=-2)
    ax.tick_params(axis="z", pad=2)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-60)

    # Trim the data limits so the cube only spans the workspace we actually
    # care about (no negative-Z padding, Y capped at 0.3 m).
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0.0, rod_length)

    # Ticks pulled in from the cube corners (limits stay at ±0.3) so the
    # X-positive and Y-negative labels don't collide at the front corner.
    from matplotlib import ticker as mticker

    ax.set_xticks([-0.2, 0.0, 0.2])
    ax.set_yticks([-0.2, 0.0, 0.2])
    ax.zaxis.set_major_locator(mticker.MaxNLocator(4))

    # White / transparent panel background instead of the default light grey.
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor((0.7, 0.7, 0.7, 1.0))

    from matplotlib.legend_handler import HandlerTuple

    ref_handle = Line2D([], [], color=REF_COLOR, linewidth=2.0)
    sim_handle = Line2D([], [], color=SIM_COLOR, linestyle=":", linewidth=2.0)
    # "Applied wrench": short dash followed by a right-pointing arrow.
    arrow_dash = Line2D([], [], color=ARROW_COLOR, linestyle="-", linewidth=1.4)
    arrow_marker = Line2D(
        [],
        [],
        color=ARROW_COLOR,
        linestyle="None",
        marker=5,
        markersize=7,
        markerfacecolor=ARROW_COLOR,
        markeredgewidth=1.0,
    )

    ax.legend(
        [ref_handle, sim_handle, (arrow_dash, arrow_marker)],
        ["Sorosim Ref.", "MuJoCo", "Applied wrench"],
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0)},
        loc="lower left",
        bbox_to_anchor=(-0.02, 0.15),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        fontsize=14,
        handlelength=1.6,
        borderpad=0.3,
        labelspacing=0.3,
    )


def panel_b_tip_error(ax, steel_run, tpu_run, steel_len, tpu_len):
    """Panel (b): tip error (% of rod length) vs N, log-log. Three lines
    per material — mean (solid + marker), 95th percentile (dashed), and
    max (dotted) — to make worst-case behaviour explicit."""

    def stats(df, rod_len):
        grouped = df.groupby("N")["tip_error"]
        ns = np.array(sorted(df["N"].unique()))
        scale = 100.0 / rod_len  # fraction → % of rod length
        means = np.array([grouped.get_group(n).mean() for n in ns]) * scale
        p95 = np.array([np.percentile(grouped.get_group(n), 95) for n in ns]) * scale
        maxv = np.array([grouped.get_group(n).max() for n in ns]) * scale
        return ns, means, p95, maxv

    n_s, mean_s, p95_s, max_s = stats(steel_run["df"], steel_len)
    n_t, mean_t, p95_t, max_t = stats(tpu_run["df"], tpu_len)

    # Steel — three lines, same colour, three line styles.
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
        n_s, p95_s, color=STEEL_COLOR, linestyle="--", linewidth=1.2, label="Steel 95%"
    )
    ax.plot(
        n_s, max_s, color=STEEL_COLOR, linestyle=":", linewidth=1.2, label="Steel max"
    )
    # TPU — three lines, same convention.
    ax.plot(
        n_t,
        mean_t,
        color=TPU_COLOR,
        linestyle="-",
        linewidth=2.0,
        marker="s",
        label="TPU mean",
    )
    ax.plot(n_t, p95_t, color=TPU_COLOR, linestyle="--", linewidth=1.2, label="TPU 95%")
    ax.plot(n_t, max_t, color=TPU_COLOR, linestyle=":", linewidth=1.2, label="TPU max")

    # Compact rod-length annotation since the legend no longer carries it.
    ax.text(
        0.02,
        0.02,
        f"Steel L={steel_len:.2f} m, TPU L={tpu_len:.2f} m",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="0.35",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (links)")
    ax.set_ylabel("Tip error (% of rod length)")
    format_log_n_axis(ax, n_s)
    format_log_y_decimal(ax)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=14,
        ncol=2,
        columnspacing=1.0,
        labelspacing=0.25,
        borderpad=0.3,
        handlelength=1.8,
        handletextpad=0.5,
    )


def make_shapes_figure(
    steel_run: dict,
    num_shapes: int,
    seed: int,
    rod_length: float,
    out_prefix: Path,
    note=None,
):
    """Standalone Figure A — sample shapes overlay (3D)."""
    fig = plt.figure(figsize=(6.4, 4.6))
    # Leave generous room on the right for the Z(m) axis label (matplotlib
    # renders 3D axis labels just outside the projected cube and they will
    # fall off-figure if the cube fills the rect).
    ax = fig.add_axes([0.0, 0.0, 0.78, 1.0], projection="3d")
    panel_a_sample_shapes(
        ax, steel_run, num_shapes=num_shapes, seed=seed, rod_length=rod_length
    )
    save_figure(fig, out_prefix, note=note)


def make_error_figure(
    steel_run: dict,
    tpu_run: dict,
    steel_len: float,
    tpu_len: float,
    out_prefix: Path,
    note=None,
):
    """Standalone Figure B — tip error (% of rod length) vs N."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    panel_b_tip_error(ax, steel_run, tpu_run, steel_len, tpu_len)
    save_figure(fig, out_prefix, note=note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steel-dir",
        type=Path,
        default=None,
        help="Override the steel-statics result directory.",
    )
    parser.add_argument(
        "--tpu-dir",
        type=Path,
        default=None,
        help="Override the TPU-statics result directory.",
    )
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=10,
        help="Number of shape pairs to overlay in the shapes figure. Default 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the shapes-figure sample selection. Default 0.",
    )
    parser.add_argument(
        "--steel-length",
        type=float,
        default=0.6,
        help="Spring-steel rod length in metres. Default 0.6.",
    )
    parser.add_argument(
        "--tpu-length",
        type=float,
        default=0.4,
        help="TPU rod length in metres. Default 0.4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "paper_figures",
        help="Directory to write statics_shapes.{png,pdf} and statics_error.{png,pdf}.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Stamp this note on every figure (e.g. a dry-run marker).",
    )
    args = parser.parse_args()

    sorosim_dir = SCRIPT_DIR / "evaluation_results/sorosim_statics"
    steel_dir = args.steel_dir or find_latest_run(
        "*_spring_steel_statics_*", sorosim_dir
    )
    tpu_dir = args.tpu_dir or find_latest_run("*_tpu_statics_*", sorosim_dir)
    print(f"Steel run: {steel_dir.name}")
    print(f"TPU   run: {tpu_dir.name}")

    steel_run = load_run(steel_dir)
    tpu_run = load_run(tpu_dir)

    plt.rcParams.update(PAPER_STYLE)

    make_shapes_figure(
        steel_run,
        num_shapes=args.num_shapes,
        seed=args.seed,
        rod_length=args.steel_length,
        out_prefix=args.output_dir / "statics_shapes",
        note=args.note,
    )
    make_error_figure(
        steel_run,
        tpu_run,
        args.steel_length,
        args.tpu_length,
        out_prefix=args.output_dir / "statics_error",
        note=args.note,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
