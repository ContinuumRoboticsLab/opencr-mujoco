#!/usr/bin/env python3
"""Render a compact paper figure (9 x 4.5 cm, Arial 6pt) comparing a recorded
sysid session (real tip) against the calibrated model (simulated tip), and
report the model's RMSE on that recording.

The model is generated on the fly from the COMMITTED calibrated generation
config (configs/generation/ftdcr_v4_sysid_0326.json) and simulated over the
committed recording, so this runs from a clean clone — no local
sysid_pipeline.py run required.

Left  : 3D trajectory (real vs sim), near top-down view, equal axis scale.
Top R : 9 servo commands vs time.
Bot R : tip-error histogram (standard RMSE convention) with mean labelled.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # paper_results/
PROJECT_ROOT = SCRIPT_DIR.parent  # repo root
sys.path.insert(
    0, str(PROJECT_ROOT)
)  # so `from opencr_mujoco.sysid...` resolves regardless of cwd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from opencr_mujoco.sysid.evaluate_sysid_model import (
    DEFAULT_CONFIG,
    compute_servo_mapping,
    compute_bias_from_first_point,
)
from opencr_mujoco.sysid.pipeline_data_loader import PipelineDataLoader
from opencr_mujoco.sysid.data_loader import TrajectoryDataLoader
from opencr_mujoco.sysid.trajectory_simulator import TrajectorySimulator

# Inputs (both committed): the recorded train session + the calibrated
# generation config. The model XML is regenerated from the config on demand.
RAW_CSV = (
    PROJECT_ROOT
    / "data/sysid/ftdcr_v4/tdcr_sysid_20260511-183002_analysis/tdcr_sysid_20260511-183002_train.csv"
)  # sysid was performed on 20250326 hence the suffix, this is a separate dataset for validating sysid quality.
DATA_DIR = RAW_CSV.parent
GEN_CONFIG = PROJECT_ROOT / "configs/generation/ftdcr_v4_sysid_0326.json"
OUT_DIR = SCRIPT_DIR / "paper_figures" / "sysid"


def _load_gen_config():
    """Load the calibrated generation config, deriving the totals the
    generator expects (same completion generate.py applies)."""
    import json

    with open(GEN_CONFIG) as f:
        cfg = json.load(f)
    if "total_links" not in cfg and "links_per_segment" in cfg:
        cfg["total_links"] = sum(int(v) for v in cfg["links_per_segment"].values())
    if "total_length" not in cfg and "segment_lengths" in cfg:
        cfg["total_length"] = sum(float(v) for v in cfg["segment_lengths"].values())
    return cfg


def run_pipeline():
    gen_cfg = _load_gen_config()
    config = dict(DEFAULT_CONFIG)
    # Exact neutral tip height for the bias removal (instead of the default
    # uniform-segment guess) — the calibrated config knows its true length.
    config["kinematics"] = dict(
        config.get("kinematics", {}), total_length_m=gen_cfg["total_length"]
    )
    preprocessed = OUT_DIR / "preprocessed.csv"
    sim_cache = OUT_DIR / "paper_figure_sim_cache.npz"

    if not preprocessed.exists():
        servo_mapping = compute_servo_mapping(DATA_DIR, config)
        bias = compute_bias_from_first_point(RAW_CSV, config)
        PipelineDataLoader(str(RAW_CSV), config).preprocess_to_standard_csv(
            str(preprocessed), servo_mapping, bias
        )

    dl = TrajectoryDataLoader(str(preprocessed), config)
    traj = dl.get_full_trajectory()

    # The cache is keyed to the calibrated config it was computed from, so
    # editing the config re-simulates instead of silently reusing a stale
    # rollout.
    model_key = f"{GEN_CONFIG}:{GEN_CONFIG.stat().st_mtime_ns}"
    cached = None
    if sim_cache.exists():
        data = np.load(sim_cache, allow_pickle=True)
        if str(data.get("model_key", "")) == model_key:
            cached = data["sim_marker"]
            print(f"  loaded sim cache: {sim_cache}")
        else:
            print("  sim cache is stale (different config); re-simulating")
    if cached is not None:
        sim_marker = cached
    else:
        from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config

        model_xml = OUT_DIR / f"{GEN_CONFIG.stem}.xml"
        create_tdcr_from_config(gen_cfg, str(model_xml))
        print(f"  generated model from {GEN_CONFIG.relative_to(PROJECT_ROOT)}")
        sim = TrajectorySimulator(config, enable_viewer=False)
        sim.load_model(str(model_xml))
        _, sim_marker = sim.simulate_trajectory(
            traj["actuator_commands"], traj["timestamps"]
        )
        sim.cleanup()
        np.savez(sim_cache, sim_marker=sim_marker, model_key=model_key)
        print(f"  saved sim cache: {sim_cache}")

    real_mm = traj["marker_positions"] * 1000
    sim_mm = sim_marker * 1000

    # Mean-offset alignment — the pipeline's reported-RMSE convention — then
    # a common shift to the first real point for plotting (RMSE-invariant).
    sim_mm = sim_mm - np.mean(sim_mm - real_mm, axis=0)
    origin = real_mm[0].copy()
    real_mm = real_mm - origin
    sim_mm = sim_mm - origin

    errors = np.linalg.norm(real_mm - sim_mm, axis=1)
    df = pd.read_csv(preprocessed)
    t = df["timestamp"].values - df["timestamp"].values[0]
    servos = df[[f"servo_{i}_mm" for i in range(1, 10)]].values

    return real_mm, sim_mm, errors, t, servos


def make_figure(real_mm, sim_mm, errors, t, servos):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "legend.fontsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "axes.linewidth": 0.4,
            "lines.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    CM = 1 / 2.54
    fig = plt.figure(figsize=(9 * CM, 4.5 * CM))
    gs = GridSpec(
        2,
        2,
        width_ratios=[0.95, 1.0],
        height_ratios=[1, 1],
        wspace=0.45,
        hspace=0.85,
        left=0.005,
        right=0.985,
        top=0.94,
        bottom=0.20,
    )

    # --- 3D trajectory (left, spans both rows) ---
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.plot(
        real_mm[:, 0],
        real_mm[:, 1],
        real_mm[:, 2],
        color="tab:green",
        alpha=0.8,
        linewidth=0.5,
        label="Real",
    )
    ax3d.plot(
        sim_mm[:, 0],
        sim_mm[:, 1],
        sim_mm[:, 2],
        color="black",
        alpha=0.8,
        linewidth=0.5,
        label="Sim",
    )

    all_pts = np.vstack([real_mm, sim_mm])
    mids = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    # Single half-range so X, Y, Z share an identical data scale.
    half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.02
    ax3d.set_xlim(mids[0] - half, mids[0] + half)
    ax3d.set_ylim(mids[1] - half, mids[1] + half)
    ax3d.set_zlim(mids[2] - half, mids[2] + half)
    ax3d.set_box_aspect((1, 1, 1), zoom=1.32)
    ax3d.view_init(elev=48, azim=-60)

    ax3d.set_xlabel("X (mm)", labelpad=-12)
    ax3d.set_ylabel("Y (mm)", labelpad=-12)
    ax3d.set_zlabel("Z (mm)", labelpad=-12)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_tick_params(pad=-4)
        # transparent panes
        axis.set_pane_color((1, 1, 1, 0))
        axis.pane.set_edgecolor((0, 0, 0, 0))
        # lighter / thinner grid
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0.15)
        axis._axinfo["grid"]["linewidth"] = 0.3
    ax3d.tick_params(axis="both", which="major", labelsize=5)
    ax3d.legend(
        loc="upper right",
        bbox_to_anchor=(1.05, 1.02),
        frameon=False,
        labelspacing=0.15,
        handlelength=1.2,
        borderpad=0.1,
        fontsize=5,
    )

    # --- Servo commands vs time (top right) ---
    ax_act = fig.add_subplot(gs[0, 1])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(9):
        ax_act.plot(t, servos[:, i], color=colors[i], linewidth=0.4, label=f"t{i+1}")
    ax_act.set_xlabel("Time (s)", labelpad=1)
    ax_act.set_ylabel("Tendon (mm)", labelpad=1)
    ax_act.tick_params(axis="both", which="major", pad=1)
    # 9 entries in a single horizontal row below the axes
    ax_act.legend(
        ncol=9,
        frameon=False,
        fontsize=4,
        columnspacing=0.4,
        handlelength=0.8,
        handletextpad=0.2,
        labelspacing=0.1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        borderpad=0.1,
    )
    ax_act.margins(x=0.01)

    # --- Tip-error histogram (bottom right) ---
    ax_hist = fig.add_subplot(gs[1, 1])
    mean_err = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ax_hist.hist(errors, bins=40, color="0.55", edgecolor="black", linewidth=0.25)
    ax_hist.axvline(mean_err, color="red", linewidth=0.6)
    ax_hist.text(
        mean_err,
        0.97,
        f" mean={mean_err:.1f} mm",
        transform=ax_hist.get_xaxis_transform(),
        ha="left",
        va="top",
        color="red",
        fontsize=5,
    )
    ax_hist.set_xlabel("Tip position error (mm)", labelpad=1)
    ax_hist.set_ylabel("Count", labelpad=1)
    ax_hist.tick_params(axis="both", which="major", pad=1)

    # Shift right-column panels 2 mm lower (figure is 4.5 cm tall)
    dy_fig = -0.2 / 4.5
    for ax in (ax_act, ax_hist):
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 + dy_fig, p.width, p.height])

    print(f"  RMSE: {rmse:.2f} mm   mean: {mean_err:.2f} mm   n={len(errors)}")

    pdf_path = OUT_DIR / "paper_figure.pdf"
    png_path = OUT_DIR / "paper_figure.png"
    fig.savefig(pdf_path, bbox_inches=None, pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches=None, pad_inches=0.02)
    plt.close(fig)
    print(f"  saved: {pdf_path}")
    print(f"  saved: {png_path}")


def main():
    if not RAW_CSV.exists() or not GEN_CONFIG.exists():
        print("⊘ Skipping sysid figure — required input not found:")
        print(
            f"    data session: {RAW_CSV}  [{'ok' if RAW_CSV.exists() else 'MISSING'}]"
        )
        print(
            f"    calibrated config: {GEN_CONFIG}  "
            f"[{'ok' if GEN_CONFIG.exists() else 'MISSING'}]"
        )
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real_mm, sim_mm, errors, t, servos = run_pipeline()
    make_figure(real_mm, sim_mm, errors, t, servos)
    print(f"  sysid figure written to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
