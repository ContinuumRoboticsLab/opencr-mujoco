"""
Enhanced visualization utilities for paper-quality plots.

This module extends the basic visualization with additional plot types
used in the OpenTDCR paper visualization script.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .visualization import EvaluationVisualizer
from .reference_data_loader import ReferenceDataLoader


class PaperVisualizer(EvaluationVisualizer):
    """Extended visualizer for paper-quality plots with all OpenTDCR features."""

    def plot_random_shapes_comparison(
        self,
        csv_path: Union[str, Path],
        positions_path: Union[str, Path],
        reference_data_dir: Union[str, Path],
        config_name: str,
        n_values: List[int],
        n_exclude: List[int] = [5, 10],
        num_samples: int = 20,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
    ):
        """
        Create 3D shape comparisons using consistent random samples across N values.

        This matches the plot_random_shapes function from OpenTDCR.

        Args:
            csv_path: Path to results CSV
            positions_path: Path to pickled positions
            reference_data_dir: Directory with reference data
            config_name: Configuration name
            n_values: List of N values to plot
            n_exclude: N values to exclude from plots
            num_samples: Number of random samples to show
            save_dir: Optional custom save directory
            show: Whether to show plots
        """
        self.set_paper_style({"figure.figsize": (6, 4)})

        # Filter N values
        filtered_n_values = [n for n in n_values if n not in n_exclude]

        # Create save directory
        if save_dir is None:
            save_dir = self.plots_dir / config_name / "shape_plots"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        df = pd.read_csv(csv_path)
        with open(positions_path, "rb") as f:
            all_positions = pickle.load(f)
        df["link_positions"] = all_positions

        # Select random samples from highest N
        highest_n = max(filtered_n_values)
        highest_n_df = df[df["N"] == highest_n]

        sample_size = min(num_samples, len(highest_n_df))
        selected_samples = highest_n_df.sample(sample_size)
        selected_wrench_strs = selected_samples["tip_wrench_str"].tolist()

        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, sample_size))

        # Load reference data loader
        ref_loader = ReferenceDataLoader(
            reference_data_dir, frame_conversion=self.frame_conversion
        )

        # Plot for each N value
        for n in filtered_n_values:
            n_df = df[df["N"] == n]

            fig = plt.figure(figsize=(6, 4), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")

            # Track the z-extent across all plotted shapes so the axis limit is
            # derived from the data rather than pinned to one rod length.
            z_min = np.inf
            z_max = -np.inf

            # Plot each sample
            for i, wrench_str in enumerate(selected_wrench_strs):
                wrench_rows = n_df[n_df["tip_wrench_str"] == wrench_str]

                if len(wrench_rows) == 0:
                    print(f"Warning: Wrench {wrench_str} not found for N={n}")
                    continue

                row = wrench_rows.iloc[0]

                # Plot simulated shape
                mujoco_positions = row["link_positions"]
                mujoco_x = [pos[0] for pos in mujoco_positions]
                mujoco_y = [pos[1] for pos in mujoco_positions]
                mujoco_z = [pos[2] for pos in mujoco_positions]

                ax.plot(
                    mujoco_x,
                    mujoco_y,
                    mujoco_z,
                    color=colors[i],
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax.scatter(
                    mujoco_x[-1],
                    mujoco_y[-1],
                    mujoco_z[-1],
                    color=colors[i],
                    s=10,
                    alpha=0.7,
                )

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
                        color=colors[i],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    ax.scatter(
                        ref_x[-1],
                        ref_y[-1],
                        ref_z[-1],
                        color=colors[i],
                        s=10,
                        alpha=0.7,
                    )

            # Add legend for first plot
            if n == filtered_n_values[0]:
                ax.plot([], [], "gray", linewidth=1.5, label="MuJoCo")
                ax.plot(
                    [], [], "gray", linestyle="--", linewidth=1.5, label="Reference"
                )
                ax.legend(
                    loc="upper right", frameon=False, handlelength=2, handletextpad=1
                )

            # Formatting
            # Derive z-limit from the plotted data with a small margin instead
            # of hardcoding a single rod length.
            if np.isfinite(z_min) and np.isfinite(z_max):
                z_margin = max(0.05 * (z_max - z_min), 1e-3)
                ax.set_zlim(z_min - z_margin, z_max + z_margin)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.zaxis.set_major_locator(plt.MaxNLocator(5))
            ax.tick_params(pad=2)

            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

            # Save
            if save_dir:
                save_path = save_dir / f"shapes_N{n}.png"
                plt.savefig(
                    save_path,
                    dpi=300,
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                print(f"Created shape plot for N={n}")
            if show:
                plt.show()
            else:
                plt.close()

    def plot_highest_error_shapes(
        self,
        csv_path: Union[str, Path],
        positions_path: Union[str, Path],
        reference_data_dir: Union[str, Path],
        config_name: str,
        n_values: List[int],
        n_exclude: List[int] = [5, 10],
        num_cases: int = 10,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
    ):
        """
        Plot shapes with highest errors for debugging.

        Args:
            csv_path: Path to results CSV
            positions_path: Path to pickled positions
            reference_data_dir: Directory with reference data
            config_name: Configuration name
            n_values: List of N values
            n_exclude: N values to exclude
            num_cases: Number of highest error cases to plot
            save_dir: Optional custom save directory
            show: Whether to show plots
        """
        self.set_paper_style({"figure.figsize": (6, 4)})

        # Filter N values
        filtered_n_values = [n for n in n_values if n not in n_exclude]

        # Create save directory
        if save_dir is None:
            save_dir = self.plots_dir / config_name / "debug_plots"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        df = pd.read_csv(csv_path)
        with open(positions_path, "rb") as f:
            all_positions = pickle.load(f)
        df["link_positions"] = all_positions

        # Load reference data
        ref_loader = ReferenceDataLoader(
            reference_data_dir, frame_conversion=self.frame_conversion
        )

        # Plot for each N
        for n in filtered_n_values:
            n_df = df[df["N"] == n]
            top_error_df = n_df.nlargest(num_cases, "tip_error")

            # Larger figure to accommodate legend
            fig = plt.figure(figsize=(10, 6), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")

            # Track the z-extent across all plotted shapes so the axis limit is
            # derived from the data rather than pinned to one rod length.
            z_min = np.inf
            z_max = -np.inf

            colors = plt.cm.tab10(np.linspace(0, 1, num_cases))

            # Load SoroSim reference data first to get gravity vectors.
            test_type = top_error_df.iloc[0]["test_type"]
            ref_data, _, _ = ref_loader.load_sorosim_statics_csv(test_type)

            # Plot each high-error case
            wrench_labels = []  # Store wrench info for legend
            for i, (idx, row) in enumerate(top_error_df.iterrows()):
                # Get wrenches
                mid_wrench = ref_loader.parse_wrench_string(row["mid_wrench_str"])
                tip_wrench = ref_loader.parse_wrench_string(row["tip_wrench_str"])

                # Extract gravity from the matching reference key.
                gravity_vec = None
                for key in ref_data.keys():
                    if len(key) == 3:  # (mid_wrench, tip_wrench, gravity)
                        key_mid, key_tip, key_grav = key
                        if tuple(mid_wrench) == tuple(key_mid) and tuple(
                            tip_wrench
                        ) == tuple(key_tip):
                            gravity_vec = key_grav
                            break

                if gravity_vec is None or len(gravity_vec) != 3:
                    gravity_vec = (0.0, 0.0, -9.81)  # Default gravity

                gx, gy, gz = gravity_vec
                gravity_label = f"[{gx:.1f}, {gy:.1f}, {gz:.1f}]"

                # Create wrench label
                tip_err_mm = row["tip_error"] * 1000
                wrench_label = f"Case {i+1} (err={tip_err_mm:.1f}mm)\nG:{gravity_label}\nTip:{tip_wrench}\nMid:{mid_wrench}"
                wrench_labels.append(wrench_label)

                # Plot simulated
                mujoco_positions = row["link_positions"]
                mujoco_x = [pos[0] for pos in mujoco_positions]
                mujoco_y = [pos[1] for pos in mujoco_positions]
                mujoco_z = [pos[2] for pos in mujoco_positions]
                if mujoco_z:
                    z_min = min(z_min, min(mujoco_z))
                    z_max = max(z_max, max(mujoco_z))

                ax.plot(
                    mujoco_x,
                    mujoco_y,
                    mujoco_z,
                    color=colors[i],
                    linewidth=1.5,
                    alpha=0.7,
                    label=wrench_label,
                )
                ax.scatter(
                    mujoco_x[-1],
                    mujoco_y[-1],
                    mujoco_z[-1],
                    color=colors[i],
                    s=10,
                    alpha=0.7,
                )

                # Plot gravity vector for this case (same color as shape)
                if gravity_vec is not None and len(gravity_vec) == 3:
                    gravity_scale = 0.09  # Scale factor for visualization (3x larger)
                    # Normalize by 9.81 and scale
                    ax.quiver(
                        0,
                        0,
                        0.05,  # Start slightly above origin for visibility
                        gx * gravity_scale / 9.81,
                        gy * gravity_scale / 9.81,
                        gz * gravity_scale / 9.81,
                        color=colors[i],
                        arrow_length_ratio=0.3,
                        linewidth=2.5,
                        alpha=0.7,
                    )

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
                        color=colors[i],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    ax.scatter(
                        ref_x[-1],
                        ref_y[-1],
                        ref_z[-1],
                        color=colors[i],
                        s=10,
                        alpha=0.7,
                    )

            # Add legend with wrench information and style indicators
            # Add dummy lines for MuJoCo vs Reference distinction
            ax.plot([], [], "gray", linewidth=1.5, label="MuJoCo (solid)")
            ax.plot(
                [],
                [],
                "gray",
                linestyle="--",
                linewidth=1.5,
                label="Reference (dashed)",
            )
            ax.legend(
                loc="upper left", frameon=True, fontsize=7, bbox_to_anchor=(1.0, 1.0)
            )

            # Add title with error info
            max_error = top_error_df["tip_error"].max() * 1000
            mean_error = top_error_df["tip_error"].mean() * 1000
            ax.set_title(
                f"N={n}: Top {num_cases} Errors - Max={max_error:.1f}mm, Mean={mean_error:.1f}mm",
                fontsize=12,
                pad=10,
            )

            # Formatting
            # Derive z-limit from the plotted data with a small margin instead
            # of hardcoding a single rod length.
            if np.isfinite(z_min) and np.isfinite(z_max):
                z_margin = max(0.05 * (z_max - z_min), 1e-3)
                ax.set_zlim(z_min - z_margin, z_max + z_margin)
            ax.tick_params(pad=2)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
            # Save
            if save_dir:
                save_path = save_dir / f"highest_error_shapes_N{n}.png"
                plt.savefig(
                    save_path,
                    dpi=300,
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                print(
                    f"Created highest error shape plot for N={n} (max error: {max_error:.1f}mm)"
                )
            if show:
                plt.show()
            else:
                plt.close()

    def plot_reference_shape_distribution(
        self,
        reference_data_dir: Union[str, Path],
        test_type: str,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
    ):
        """
        Plot all reference shapes to visualize the shape distribution used in evaluation.

        Args:
            reference_data_dir: Directory with reference data
            test_type: Test type name (e.g., "SpringSteelRodMuJoCo")
            save_dir: Optional custom save directory
            show: Whether to show plot
        """
        self.set_paper_style({"figure.figsize": (8, 6)})

        # Create save directory
        if save_dir is None:
            save_dir = self.plots_dir / "shape_distribution"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load SoroSim reference data (CSV).
        ref_loader = ReferenceDataLoader(
            reference_data_dir, frame_conversion=self.frame_conversion
        )
        ref_data, num_links, _ = ref_loader.load_sorosim_statics_csv(test_type)

        # Create 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Plot all reference shapes with thin lines
        num_shapes = len(ref_data)
        print(f"Plotting {num_shapes} reference shapes...")

        for i, (key, positions) in enumerate(ref_data.items()):
            # Extract positions
            ref_x = [pos[0] for pos in positions]
            ref_y = [pos[1] for pos in positions]
            ref_z = [pos[2] for pos in positions]

            # Plot with thin gray lines for all shapes
            ax.plot(
                ref_x,
                ref_y,
                ref_z,
                color="gray",
                linewidth=0.5,
                alpha=0.3,
            )

        # Set labels and title
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_zlabel("Z (m)", fontsize=10)
        ax.set_title(
            f"Reference Shape Distribution: {test_type}\n({num_shapes} shapes, {num_links} links)",
            fontsize=12,
            pad=15,
        )

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Save
        if save_dir:
            save_path = save_dir / f"shape_distribution_{test_type}.png"
            plt.savefig(
                save_path,
                dpi=300,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            print(f"Created shape distribution plot: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_convergence_history(
        self,
        model_path: str,
        mid_wrench: tuple,
        tip_wrench: tuple,
        config_name: str,
        n_value: int,
        sim_time: float = 5.0,
        sim_hz: float = 500.0,
        force_ramp_time: float = 1.0,
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot time history of tip position to check convergence.

        Args:
            model_path: Path to MuJoCo model
            mid_wrench: Wrench applied at mid-point
            tip_wrench: Wrench applied at tip
            config_name: Configuration name
            n_value: Number of links
            sim_time: Total simulation time
            sim_hz: Simulation frequency
            force_ramp_time: Time to ramp up forces (seconds)
            save: Whether to save plot
            show: Whether to show plot

        Returns:
            Matplotlib figure
        """
        import mujoco

        self.set_paper_style()

        # Load model
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # Set timestep
        sim_dt = 1.0 / sim_hz
        model.opt.timestep = sim_dt
        total_steps = int(sim_time / sim_dt)

        # Find bodies for force application
        try:
            mid_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"link_{n_value//2}"
            )
            tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
        except Exception:
            print(f"Could not find force application bodies for N={n_value}")
            return None

        # Storage for time history
        times = []
        tip_positions = []

        # Run simulation with force ramping
        ramp_time = force_ramp_time  # Use parameter instead of hardcoded value

        for step in range(total_steps):
            # Clear forces
            data.xfrc_applied[:] = 0

            # Calculate ramp factor
            elapsed_time = step * sim_dt
            if elapsed_time < ramp_time:
                ramp_factor = elapsed_time / ramp_time
            else:
                ramp_factor = 1.0

            # Apply ramped forces
            data.xfrc_applied[mid_body_id] = np.array(mid_wrench) * ramp_factor
            data.xfrc_applied[tip_body_id] = np.array(tip_wrench) * ramp_factor

            # Step simulation
            mujoco.mj_step(model, data)

            # Record data
            times.append(elapsed_time)
            tip_pos = data.body(tip_body_id).xpos.copy()
            tip_positions.append(tip_pos)

        # Convert to arrays
        times = np.array(times)
        tip_positions = np.array(tip_positions)

        # Create figure with subplots for X, Y, Z positions and convergence rate
        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

        labels = ["X", "Y", "Z"]
        colors = ["tab:red", "tab:green", "tab:blue"]

        # Plot X, Y, Z positions
        for i, (ax, label, color) in enumerate(zip(axes[:3], labels, colors)):
            ax.plot(times, tip_positions[:, i], color=color, linewidth=1.5)
            ax.set_ylabel(f"{label} Position (m)", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")

            # Mark ramp period
            ax.axvline(
                ramp_time,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label=f"Ramp End ({ramp_time:.1f}s)",
            )

            # Calculate and show settling value (last 10% of data)
            settle_start = int(0.9 * len(times))
            settled_value = np.mean(tip_positions[settle_start:, i])
            ax.axhline(
                settled_value,
                color=color,
                linestyle=":",
                alpha=0.5,
                label=f"Settled: {settled_value:.4f}m",
            )

            if i == 0:
                ax.legend(loc="upper right", fontsize=8, frameon=False)

        # Calculate and plot convergence rate (position change per step)
        position_changes = np.zeros(len(tip_positions) - 1)
        for i in range(len(tip_positions) - 1):
            position_changes[i] = np.linalg.norm(
                tip_positions[i + 1] - tip_positions[i]
            )

        # Add small value to avoid log(0)
        position_changes = np.maximum(position_changes, 1e-12)

        # Plot convergence rate on log scale
        ax_conv = axes[3]
        ax_conv.semilogy(times[1:], position_changes, "k-", linewidth=1.5, alpha=0.7)
        ax_conv.set_ylabel("Position Change\nper Step (m)", fontsize=10)
        ax_conv.set_xlabel("Time (s)", fontsize=10)
        ax_conv.grid(True, alpha=0.3, linestyle="--", which="both")
        ax_conv.set_ylim(bottom=1e-10)  # Set reasonable lower limit

        # Mark ramp period
        ax_conv.axvline(
            ramp_time,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Ramp End ({ramp_time:.1f}s)",
        )

        # Add convergence threshold line
        convergence_threshold = 1e-8
        ax_conv.axhline(
            convergence_threshold,
            color="green",
            linestyle=":",
            alpha=0.5,
            label=f"Threshold: {convergence_threshold:.0e}m",
        )

        # Calculate when convergence is achieved (if it is)
        converged_indices = np.where(position_changes < convergence_threshold)[0]
        if len(converged_indices) > 0:
            first_converged_time = times[converged_indices[0] + 1]
            ax_conv.axvline(
                first_converged_time,
                color="green",
                linestyle="-",
                alpha=0.3,
                label=f"Converged: {first_converged_time:.1f}s",
            )

        ax_conv.legend(loc="upper right", fontsize=8, frameon=False)

        # Calculate final error if we have a settled position
        final_pos = tip_positions[-1]
        final_change_rate = position_changes[-1] if len(position_changes) > 0 else 0

        title = f"Tip Position Convergence - N={n_value}"
        if "worst" in config_name.lower():
            title = f"Worst Case Convergence - N={n_value} (Final: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]m, Rate: {final_change_rate:.2e}m/step)"
        fig.suptitle(title, fontsize=11)

        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{config_name}_convergence_N{n_value}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            # Print convergence info
            if len(converged_indices) > 0:
                print(
                    f"Saved convergence plot to {save_path} (converged at {first_converged_time:.1f}s, final rate: {final_change_rate:.2e}m/step)"
                )
            else:
                print(
                    f"Saved convergence plot to {save_path} (NOT converged, final rate: {final_change_rate:.2e}m/step)"
                )

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_error_histograms(
        self,
        csv_path: Union[str, Path],
        config_name: str,
        n_values: List[int],
        n_exclude: List[int] = [5, 10],
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Create error histograms for each N value.

        Args:
            csv_path: Path to results CSV
            config_name: Configuration name
            n_values: List of N values
            n_exclude: N values to exclude
            save: Whether to save plot
            show: Whether to show plot

        Returns:
            Matplotlib figure
        """
        self.set_paper_style()

        # Load data
        df = pd.read_csv(csv_path)
        df = df[~df["N"].isin(n_exclude)]
        df["tip_error"] *= 1000  # Convert to mm

        # Filter n_values
        filtered_n_values = [n for n in n_values if n not in n_exclude]

        # Create subplots
        fig, axes = plt.subplots(
            len(filtered_n_values), 1, figsize=(6, 4 * len(filtered_n_values))
        )
        if len(filtered_n_values) == 1:
            axes = [axes]

        # Plot histogram for each N
        for i, n in enumerate(filtered_n_values):
            n_data = df[df["N"] == n]["tip_error"]

            mean_error = n_data.mean()
            std_error = n_data.std()

            # Create histogram
            axes[i].hist(n_data, bins=30, alpha=0.7, color="tab:blue")

            # Add statistics
            axes[i].axvline(
                mean_error,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_error:.2f} mm",
            )
            axes[i].axvline(
                mean_error - 1.96 * std_error,
                color="green",
                linestyle=":",
                label="95% CI",
            )
            axes[i].axvline(mean_error + 1.96 * std_error, color="green", linestyle=":")

            # Labels
            axes[i].set_xlabel("Error (mm)")
            axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"N = {n}")
            axes[i].legend()
            axes[i].grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()

        # Save
        if save:
            save_path = self.plots_dir / f"{config_name}_error_histograms.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Created error histograms")

        if show:
            plt.show()
        else:
            plt.close()

        return fig
