"""
Visualization utilities for evaluation results.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reference_data_loader import ReferenceDataLoader, validate_frame_conversion


class EvaluationVisualizer:
    """Create publication-quality plots for evaluation results."""

    # Default plot style for academic papers
    PAPER_STYLE = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 16,
        "figure.figsize": (6, 4),
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "figure.dpi": 300,
    }

    def __init__(
        self,
        results_dir: Union[str, Path] = "./evaluation_results",
        plots_dir: Union[str, Path] = "./plots",
        frame_conversion: Optional[np.ndarray] = None,
    ):
        """
        Initialize the visualizer.

        Args:
            results_dir: Directory containing evaluation results
            plots_dir: Directory to save plots
            frame_conversion: Optional 3x3 file→MuJoCo matrix. Forwarded to
                any ReferenceDataLoader created during plotting so reference
                shapes are loaded into the same frame as the simulation
                results stored in CSV/pickle.
        """
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.frame_conversion = validate_frame_conversion(frame_conversion)

    def set_paper_style(self, custom_style: Optional[dict] = None):
        """Set matplotlib style for paper-quality plots."""
        style = self.PAPER_STYLE.copy()
        if custom_style:
            style.update(custom_style)
        plt.rcParams.update(style)

    def plot_error_vs_n(
        self,
        csv_path: Union[str, Path],
        config_name: str,
        n_values_to_exclude: Optional[List[int]] = None,
        error_unit: str = "mm",
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Create log-log plot of error vs number of links.

        Args:
            csv_path: Path to results CSV
            config_name: Configuration name for saving
            n_values_to_exclude: N values to exclude from plot
            error_unit: Unit for error ('m' or 'mm')
            save: Whether to save the plot
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        self.set_paper_style()

        # Load data
        df = pd.read_csv(csv_path)

        # Exclude specified N values
        if n_values_to_exclude:
            df = df[~df["N"].isin(n_values_to_exclude)]

        # Group by N and compute statistics
        error_stats = df.groupby("N")["tip_error"].agg(["mean", "std"]).reset_index()

        # Convert units if needed
        scale = 1000 if error_unit == "mm" else 1
        error_stats["mean"] *= scale
        error_stats["std"] *= scale

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

        # Plot mean with confidence interval
        ax.plot(
            error_stats["N"],
            error_stats["mean"],
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=6,
            label="Mean",
        )
        ax.fill_between(
            error_stats["N"],
            error_stats["mean"] - 1.96 * error_stats["std"],
            error_stats["mean"] + 1.96 * error_stats["std"],
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )

        # Set log scales
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Labels
        ax.set_xlabel("Number of Links (N)", labelpad=2)
        ax.set_ylabel(f"Error ({error_unit})", labelpad=2)

        # More dense y-axis ticks
        from matplotlib.ticker import LogLocator, FuncFormatter

        # Set major and minor locators for y-axis
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(
            LogLocator(
                base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10
            )
        )

        # Format y-axis labels to use normal format (no scientific notation)
        def format_func(value, tick_number):
            if value >= 1:
                return f"{value:.0f}"
            elif value >= 0.1:
                return f"{value:.1f}"
            elif value >= 0.01:
                return f"{value:.2f}"
            elif value >= 0.001:
                return f"{value:.3f}"
            elif value >= 0.0001:
                return f"{value:.4f}"
            else:
                return f"{value:.5f}"

        ax.yaxis.set_major_formatter(FuncFormatter(format_func))

        # Disable offset text completely
        ax.xaxis.offsetText.set_visible(False)
        ax.yaxis.offsetText.set_visible(False)

        # Custom x-axis ticks with normal formatting - ONLY show the N values used
        unique_n = sorted(error_stats["N"].unique())
        ax.set_xticks(unique_n)
        ax.set_xticklabels([f"{n:d}" for n in unique_n])

        # Remove any minor ticks on x-axis to prevent scientific notation
        ax.xaxis.set_minor_locator(plt.NullLocator())

        # Grid
        ax.grid(True, linestyle="--", alpha=0.5, which="major")
        ax.grid(True, linestyle=":", alpha=0.3, which="minor", axis="y")

        # Legend
        ax.legend(frameon=False, handlelength=2, handletextpad=1, loc="best")

        # Save/show
        if save:
            save_path = self.plots_dir / f"{config_name}_error_vs_n.png"
            plt.savefig(save_path, dpi=300, transparent=True, bbox_inches="tight")
            print(f"Saved error vs N plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_shape_comparison_3d(
        self,
        csv_path: Union[str, Path],
        positions_path: Union[str, Path],
        reference_data_dir: Union[str, Path],
        config_name: str,
        n_value: int,
        num_samples: int = 20,
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Create 3D visualization comparing simulated and reference shapes.

        Args:
            csv_path: Path to results CSV
            positions_path: Path to pickled positions
            reference_data_dir: Directory with reference data
            config_name: Configuration name
            n_value: Number of links to plot
            num_samples: Number of random samples to show
            save: Whether to save the plot
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        self.set_paper_style({"figure.figsize": (8, 6)})

        # Load data
        df = pd.read_csv(csv_path)
        with open(positions_path, "rb") as f:
            all_positions = pickle.load(f)
        df["link_positions"] = all_positions

        # Filter for specific N
        n_df = df[df["N"] == n_value]

        # Sample random cases
        if len(n_df) > num_samples:
            sampled_df = n_df.sample(num_samples)
        else:
            sampled_df = n_df

        # Create 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Color map for different samples
        colors = plt.cm.tab10(np.linspace(0, 1, len(sampled_df)))

        # Load reference data — propagate frame_conversion so reference
        # 3-vectors come back in the same frame the simulation results were
        # saved in, otherwise wrench-key lookups silently miss.
        ref_loader = ReferenceDataLoader(
            reference_data_dir, frame_conversion=self.frame_conversion
        )

        # Track the z-extent across all plotted shapes so the axis limit is
        # derived from the data rather than pinned to one specific rod length.
        z_min = np.inf
        z_max = -np.inf

        for idx, (_, row) in enumerate(sampled_df.iterrows()):
            color = colors[idx]

            # Plot simulated shape
            sim_positions = row["link_positions"]
            sim_x = [pos[0] for pos in sim_positions]
            sim_y = [pos[1] for pos in sim_positions]
            sim_z = [pos[2] for pos in sim_positions]
            if sim_z:
                z_min = min(z_min, min(sim_z))
                z_max = max(z_max, max(sim_z))

            ax.plot(sim_x, sim_y, sim_z, color=color, linewidth=1.5, alpha=0.7)
            ax.scatter(sim_x[-1], sim_y[-1], sim_z[-1], color=color, s=10, alpha=0.7)

            # Get reference shape
            mid_wrench = ref_loader.parse_wrench_string(row["mid_wrench_str"])
            tip_wrench = ref_loader.parse_wrench_string(row["tip_wrench_str"])

            # Load SoroSim reference data (CSV); keys are (mid, tip, gravity).
            test_type = row["test_type"]
            ref_data, _, _ = ref_loader.load_sorosim_statics_csv(test_type)

            # Find matching shape by (mid, tip) wrench, ignoring gravity.
            ref_positions = None
            for key, positions in ref_data.items():
                if len(key) == 3:
                    key_mid, key_tip, _ = key
                    if tuple(mid_wrench) == tuple(key_mid) and tuple(
                        tip_wrench
                    ) == tuple(key_tip):
                        ref_positions = positions
                        break

            if ref_positions is not None:
                ref_x = [pos[0] for pos in ref_positions]
                ref_y = [pos[1] for pos in ref_positions]
                ref_z = [pos[2] for pos in ref_positions]
                if ref_z:
                    z_min = min(z_min, min(ref_z))
                    z_max = max(z_max, max(ref_z))

                ax.plot(
                    ref_x,
                    ref_y,
                    ref_z,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax.scatter(
                    ref_x[-1], ref_y[-1], ref_z[-1], color=color, s=10, alpha=0.7
                )

        # Add legend
        ax.plot([], [], "gray", linewidth=1.5, label="Simulated")
        ax.plot([], [], "gray", linestyle="--", linewidth=1.5, label="Reference")
        ax.legend(loc="upper right", frameon=False)

        # Labels and formatting
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        # Derive z-limit from the plotted data with a small margin instead of
        # hardcoding a single rod length.
        if np.isfinite(z_min) and np.isfinite(z_max):
            z_margin = max(0.05 * (z_max - z_min), 1e-3)
            ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.set_title(f"Shape Comparison (N={n_value})")

        # Save/show
        if save:
            save_path = self.plots_dir / f"{config_name}_shapes_N{n_value}.png"
            plt.savefig(save_path, dpi=300, transparent=True, bbox_inches="tight")
            print(f"Saved shape comparison to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_runtime_performance(
        self,
        csv_path: Union[str, Path],
        config_name: str,
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot real-time performance (RTF) vs number of links.

        Args:
            csv_path: Path to results CSV
            config_name: Configuration name
            save: Whether to save the plot
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        self.set_paper_style()

        # Load data
        df = pd.read_csv(csv_path)

        # Group by N
        rtf_stats = df.groupby("N")["realtime_ratio"].agg(["mean", "std"]).reset_index()

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

        # Plot
        ax.plot(
            rtf_stats["N"],
            rtf_stats["mean"],
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=6,
            label="Mean",
        )
        ax.fill_between(
            rtf_stats["N"],
            rtf_stats["mean"] - 1.96 * rtf_stats["std"],
            rtf_stats["mean"] + 1.96 * rtf_stats["std"],
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )

        # Add horizontal line at RTF=1
        ax.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Real-time")

        # Log scales
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Labels
        ax.set_xlabel("Number of Links (N)")
        ax.set_ylabel("Real-time Factor (RTF)")

        # Custom x-axis ticks with normal formatting - ONLY show the N values used
        unique_n = sorted(rtf_stats["N"].unique())
        ax.set_xticks(unique_n)
        ax.set_xticklabels([f"{n:d}" for n in unique_n])

        # Remove any minor ticks on x-axis to prevent scientific notation
        ax.xaxis.set_minor_locator(plt.NullLocator())

        # Format y-axis to use normal numbers
        from matplotlib.ticker import FuncFormatter

        def format_func(value, tick_number):
            if 0.001 <= value <= 10000:
                if value >= 1:
                    return f"{value:.0f}"
                elif value >= 0.1:
                    return f"{value:.1f}"
                elif value >= 0.01:
                    return f"{value:.2f}"
                else:
                    return f"{value:.3f}"
            else:
                return f"{value:.1e}"

        ax.yaxis.set_major_formatter(FuncFormatter(format_func))

        # Disable offset text completely
        ax.xaxis.offsetText.set_visible(False)
        ax.yaxis.offsetText.set_visible(False)

        # Grid and legend
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=False)

        # Save/show
        if save:
            save_path = self.plots_dir / f"{config_name}_rtf_vs_n.png"
            plt.savefig(save_path, dpi=300, transparent=True, bbox_inches="tight")
            print(f"Saved RTF plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def create_summary_report(
        self,
        csv_path: Union[str, Path],
        config_name: str,
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Create a text summary report of evaluation results.

        Args:
            csv_path: Path to results CSV
            config_name: Configuration name
            save_path: Optional path to save report

        Returns:
            Report text
        """
        df = pd.read_csv(csv_path)

        report = []
        report.append(f"Evaluation Summary Report: {config_name}")
        report.append("=" * 60)
        report.append(f"Total evaluations: {len(df)}")
        report.append(f"Test types: {', '.join(df['test_type'].unique())}")
        report.append(f"N values: {sorted(df['N'].unique())}")
        report.append("")

        # Summary by N
        report.append("Results by Number of Links:")
        report.append("-" * 40)

        for n in sorted(df["N"].unique()):
            n_df = df[df["N"] == n]
            tip_error_mm = n_df["tip_error"] * 1000

            report.append(f"\nN = {n}:")
            report.append(f"  Samples: {len(n_df)}")
            report.append(f"  Tip Error (mm):")
            report.append(f"    Mean: {tip_error_mm.mean():.2f}")
            report.append(f"    Std:  {tip_error_mm.std():.2f}")
            report.append(f"    Min:  {tip_error_mm.min():.2f}")
            report.append(f"    Max:  {tip_error_mm.max():.2f}")
            report.append(f"  Real-time Factor:")
            report.append(f"    Mean: {n_df['realtime_ratio'].mean():.2f}")
            report.append(f"    Std:  {n_df['realtime_ratio'].std():.2f}")

        report_text = "\n".join(report)

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report_text)
            print(f"Saved report to {save_path}")

        return report_text
