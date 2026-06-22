#!/usr/bin/env python3
"""Evaluate a sysid model on one or more datasets.

Preprocesses data, simulates, and generates comparison plots + videos.

Usage:
    # Generate a model first, e.g.:
    #   python generate.py --config ftdcr_v4_sysid
    python -m opencr_mujoco.sysid.evaluate_sysid_model --model assets/tdcr/ftdcr_v4_sysid.xml \
        --gen-config configs/generation/ftdcr_v4_sysid.json \
        --data-dir data/sysid/ftdcr_v4/tdcr_sysid_20260511-183002_analysis \
        --output eval_results/v4_eval
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

from .pipeline_data_loader import PipelineDataLoader
from .geometric_identifier import GeometricIdentifier
from .trajectory_simulator import TrajectorySimulator
from .data_loader import TrajectoryDataLoader

DEFAULT_CONFIG = {
    "marker_units": "mm",
    "marker_transform": {
        "translation": [0.0, 0.0, 0.008],
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    },
    "kinematics": {
        "num_segments": 3,
        "tendons_per_segment": 3,
        "tendon_distance_mm": 4.5,
    },
}


def compute_servo_mapping(data_dir: Path, config: dict):
    """Get servo mapping from tendon_pull data."""
    tp_csvs = sorted(data_dir.glob("*_tendon_pull.csv"))
    if not tp_csvs:
        raise FileNotFoundError(f"No tendon_pull CSV in {data_dir}")

    kin = config.get("kinematics", {})
    identifier = GeometricIdentifier(kin)
    df = pd.read_csv(tp_csvs[0])
    return identifier.detect_servo_mapping_from_labels(df)


def compute_bias_from_first_point(csv_path: Path, config: dict):
    """Compute position bias from the first data point of a CSV.

    Subtracts the first tip position so that the trajectory starts at
    the expected neutral height (sum of segment lengths + marker offset).
    This aligns datasets with different marker placements.

    Returns:
        np.ndarray of shape (3,) — bias to subtract from tip positions (in mm).
    """
    loader = PipelineDataLoader(str(csv_path), config)
    df = loader.load_raw_dataframe()
    tip_cols = loader.detect_tip_columns(df)

    first_point = np.array(
        [df[tip_cols[0]].iloc[0], df[tip_cols[1]].iloc[0], df[tip_cols[2]].iloc[0]]
    )

    # Expected neutral: (0, 0, total_segment_length + marker_z) in mm
    kin = config.get("kinematics", {})
    marker_z = config.get("marker_transform", {}).get("translation", [0, 0, 0])[2]
    if "total_length_m" in kin:
        total_length_m = kin["total_length_m"]
    else:
        # Fall back to uniform segments of segment_length_m each
        n_seg = kin.get("num_segments", 3)
        total_length_m = n_seg * kin.get("segment_length_m", 0.064)
    expected_z = total_length_m * 1000 + marker_z * 1000  # mm

    bias = first_point.copy()
    bias[2] = first_point[2] - expected_z

    return bias


def evaluate_on_csv(
    model_path, csv_path, output_dir, config, servo_mapping, bias, label, no_video=False
):
    """Evaluate model on a single CSV and generate outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess
    preprocessed = output_dir / "preprocessed.csv"
    loader = PipelineDataLoader(str(csv_path), config)
    loader.preprocess_to_standard_csv(str(preprocessed), servo_mapping, bias)

    # Load data
    dl = TrajectoryDataLoader(str(preprocessed), config)
    traj = dl.get_full_trajectory()
    raw_df = pd.read_csv(str(preprocessed))
    pattern_labels = (
        raw_df["pattern_label"].values if "pattern_label" in raw_df.columns else None
    )

    # Simulate
    sim = TrajectorySimulator(config, enable_viewer=False)
    sim.load_model(str(model_path))
    _, sim_marker = sim.simulate_trajectory(
        traj["actuator_commands"], traj["timestamps"]
    )

    real_mm = traj["marker_positions"] * 1000
    sim_mm = sim_marker * 1000

    # Align both trajectories by subtracting their respective first points
    # so both start at the same reference (removes constant offsets)
    real_offset = real_mm[0].copy()
    sim_offset = sim_mm[0].copy()
    real_mm = real_mm - real_offset
    sim_mm = sim_mm - sim_offset

    # Standard (Euclidean, pointwise) RMSE — the one reported convention
    errors = np.linalg.norm(real_mm - sim_mm, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))

    print(f"\n  {label}:")
    print(f"    Samples: {len(real_mm)}")
    print(f"    RMSE: {rmse:.2f} mm")

    # Static plot
    fig = plt.figure(figsize=(14, 12))

    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot(
        real_mm[:, 0],
        real_mm[:, 1],
        real_mm[:, 2],
        "g.-",
        alpha=0.7,
        markersize=3,
        label="Real",
    )
    ax1.plot(
        sim_mm[:, 0],
        sim_mm[:, 1],
        sim_mm[:, 2],
        "k.-",
        alpha=0.7,
        markersize=3,
        label="Sim",
    )
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.legend()
    ax1.set_title("3D Trajectory")

    ax2 = fig.add_subplot(222)
    ax2.plot(real_mm[:, 0], real_mm[:, 1], "g.-", alpha=0.7, markersize=3, label="Real")
    ax2.plot(sim_mm[:, 0], sim_mm[:, 1], "k.-", alpha=0.7, markersize=3, label="Sim")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.legend()
    ax2.set_title("XY Projection")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(223)
    ax3.plot(real_mm[:, 0], real_mm[:, 2], "g.-", alpha=0.7, markersize=3, label="Real")
    ax3.plot(sim_mm[:, 0], sim_mm[:, 2], "k.-", alpha=0.7, markersize=3, label="Sim")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.legend()
    ax3.set_title("XZ Projection")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(224)
    ax4.plot(errors, "b-", alpha=0.7, label=f"RMSE={rmse:.1f}mm")
    ax4.set_xlabel("Sample")
    ax4.set_ylabel("Error (mm)")
    ax4.set_title("Position Error")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(label, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = output_dir / "trajectory_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {plot_path}")

    if no_video:
        sim.cleanup()
        return {"rmse_mm": rmse, "n_samples": len(real_mm)}

    # Video
    try:
        n = len(real_mm)
        fig_v, (ax_xy, ax_err) = plt.subplots(1, 2, figsize=(14, 6))
        all_xy = np.vstack([real_mm[:, :2], sim_mm[:, :2]])
        pad = 10
        ax_xy.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
        ax_xy.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
        ax_xy.set_xlabel("X (mm)")
        ax_xy.set_ylabel("Y (mm)")
        ax_xy.set_aspect("equal")
        ax_xy.grid(True, alpha=0.3)
        (rl,) = ax_xy.plot([], [], "g-", alpha=0.4, lw=1)
        (sl,) = ax_xy.plot([], [], "k-", alpha=0.4, lw=1)
        (rd,) = ax_xy.plot([], [], "go", ms=10, label="Real")
        (sd,) = ax_xy.plot([], [], "ks", ms=8, label="Sim")
        ax_xy.legend(loc="upper right")
        tt = ax_xy.set_title(label)

        ax_err.set_xlim(0, n)
        ax_err.set_ylim(0, max(np.max(errors) * 1.1, 1))
        ax_err.set_xlabel("Sample")
        ax_err.set_ylabel("Error (mm)")
        ax_err.set_title("Position Error")
        ax_err.grid(True, alpha=0.3)
        (el,) = ax_err.plot([], [], "r-", lw=1)
        mt = ax_err.text(0.02, 0.95, "", transform=ax_err.transAxes, va="top")

        def update(i):
            rl.set_data(real_mm[: i + 1, 0], real_mm[: i + 1, 1])
            sl.set_data(sim_mm[: i + 1, 0], sim_mm[: i + 1, 1])
            rd.set_data([real_mm[i, 0]], [real_mm[i, 1]])
            sd.set_data([sim_mm[i, 0]], [sim_mm[i, 1]])
            el.set_data(np.arange(i + 1), errors[: i + 1])
            mt.set_text(f"Mean: {np.mean(errors[:i+1]):.1f}mm")
            pl = pattern_labels[i] if pattern_labels is not None else ""
            tt.set_text(f"{label}  [{i+1}/{n}]  {pl}")
            return rl, sl, rd, sd, el, mt, tt

        anim = animation.FuncAnimation(fig_v, update, frames=n, interval=50, blit=True)
        vid_path = output_dir / "comparison.mp4"
        anim.save(str(vid_path), writer="ffmpeg", fps=20, dpi=100)
        plt.close(fig_v)
        print(f"    Saved: {vid_path}")
    except Exception as e:
        print(f"    Video failed: {e}")

    sim.cleanup()

    return {
        "rmse_mm": rmse,
        "n_samples": len(real_mm),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate sysid model on datasets")
    parser.add_argument("--model", required=True, help="Path to MJCF model XML")
    parser.add_argument(
        "--data-dir", nargs="+", required=True, help="Data directories to evaluate on"
    )
    parser.add_argument("--output", default="eval_results", help="Output directory")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Data splits to evaluate (default: train val)",
    )
    parser.add_argument(
        "--config", help="Optional JSON config file for marker_transform etc."
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=0.064,
        help="Uniform segment length in meters used for the "
        "expected neutral height (default: 0.064)",
    )
    parser.add_argument(
        "--gen-config",
        help="Generation config JSON; its segment_lengths " "override --segment-length",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip comparison-video rendering (much faster for multi-dataset sweeps)",
    )
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            user_cfg = json.load(f)
        config.update(user_cfg.get("data", user_cfg))

    # Expected neutral height: prefer explicit segment lengths from a
    # generation config (via --gen-config), else uniform --segment-length.
    config.setdefault("kinematics", {})
    if args.gen_config:
        with open(args.gen_config) as f:
            gen_cfg = json.load(f)
        seg_lengths = gen_cfg.get("segment_lengths", {})
        if seg_lengths:
            config["kinematics"]["total_length_m"] = sum(
                float(v) for v in seg_lengths.values()
            )
    config["kinematics"].setdefault("segment_length_m", args.segment_length)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Splits: {args.splits}")

    all_results = {}

    for data_dir_str in args.data_dir:
        data_dir = Path(data_dir_str)
        dataset_name = data_dir.name
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        # Get servo mapping from this dataset's tendon_pull
        servo_mapping = compute_servo_mapping(data_dir, config)

        for split in args.splits:
            csvs = sorted(data_dir.glob(f"*_{split}.csv"))
            if not csvs:
                print(f"  No {split} CSV found, skipping")
                continue

            # Compute per-CSV bias from first data point
            bias = compute_bias_from_first_point(csvs[0], config)

            out_dir = Path(args.output) / dataset_name / split
            label = f"{dataset_name} / {split}"
            results = evaluate_on_csv(
                model_path,
                csvs[0],
                out_dir,
                config,
                servo_mapping,
                bias,
                label,
                no_video=args.no_video,
            )
            all_results[f"{dataset_name}/{split}"] = results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset/Split':<50} {'RMSE':>10}")
    print("-" * 62)
    for key, r in all_results.items():
        print(f"{key:<50} {r['rmse_mm']:>8.2f}mm")

    # Save summary
    out_path = Path(args.output) / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
