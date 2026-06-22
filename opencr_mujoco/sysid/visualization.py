"""Visualization tools for system identification."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class SysIDVisualizer:
    """Visualization tools for system identification results."""

    def __init__(self, output_dir: Optional[str] = None, show_plots: bool = True):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            show_plots: Whether to show plots interactively
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.show_plots = show_plots
        self.iteration_history = []
        self.error_history = []
        self.best_error = float("inf")
        self.best_params = None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enable interactive mode if showing plots
        if self.show_plots:
            plt.ion()  # Turn on interactive mode
        else:
            # Use non-interactive backend for headless operation
            matplotlib.use("Agg")

    def plot_trajectory_comparison(
        self,
        real_positions: np.ndarray,
        simulated_positions: np.ndarray,
        title: str = "Trajectory Comparison",
        save_name: Optional[str] = None,
    ):
        """Plot 3D comparison of real vs simulated trajectories.

        Args:
            real_positions: Real marker positions (N x 3)
            simulated_positions: Simulated marker positions (N x 3)
            title: Plot title
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(12, 5))

        # 3D trajectory plot
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot(
            real_positions[:, 0],
            real_positions[:, 1],
            real_positions[:, 2],
            "b-",
            label="Real",
            linewidth=2,
            alpha=0.7,
        )
        ax1.plot(
            simulated_positions[:, 0],
            simulated_positions[:, 1],
            simulated_positions[:, 2],
            "r--",
            label="Simulated",
            linewidth=2,
            alpha=0.7,
        )

        # Start and end markers
        ax1.scatter(*real_positions[0], c="green", s=100, marker="o", label="Start")
        ax1.scatter(*real_positions[-1], c="red", s=100, marker="s", label="End")

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("3D Trajectories")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2D projections
        ax2 = fig.add_subplot(122)

        # XY projection
        ax2.plot(
            real_positions[:, 0],
            real_positions[:, 1],
            "b-",
            label="Real",
            linewidth=2,
            alpha=0.7,
        )
        ax2.plot(
            simulated_positions[:, 0],
            simulated_positions[:, 1],
            "r--",
            label="Simulated",
            linewidth=2,
            alpha=0.7,
        )

        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title("XY Projection")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis("equal")

        fig.suptitle(title)
        plt.tight_layout()

        # Save the plot if requested
        if save_name and self.output_dir:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {save_path}")

        # Show interactively if requested
        if self.show_plots:
            plt.draw()
            plt.pause(0.1)  # Brief pause to allow display update

        # Always close the figure to free memory
        plt.close(fig)

    def plot_error_over_time(
        self,
        real_positions: np.ndarray,
        simulated_positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        save_name: Optional[str] = None,
    ):
        """Plot error over time.

        Args:
            real_positions: Real marker positions (N x 3)
            simulated_positions: Simulated marker positions (N x 3)
            timestamps: Optional timestamps
            save_name: Filename to save plot
        """
        # Compute errors
        errors = np.linalg.norm(real_positions - simulated_positions, axis=1)

        if timestamps is None:
            timestamps = np.arange(len(errors))
            x_label = "Sample"
        else:
            x_label = "Time (s)"

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Error over time
        ax1.plot(timestamps, errors * 1000, "b-", linewidth=2)  # Convert to mm
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Position Error (mm)")
        ax1.set_title("Position Error Over Time")
        ax1.grid(True, alpha=0.3)

        # Add statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        ax1.axhline(
            mean_error * 1000,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Mean: {mean_error*1000:.2f} mm",
        )
        ax1.axhline(
            max_error * 1000,
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Max: {max_error*1000:.2f} mm",
        )
        ax1.legend()

        # Error histogram
        ax2.hist(errors * 1000, bins=30, edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Position Error (mm)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Error Distribution")
        ax2.axvline(
            mean_error * 1000,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_error*1000:.2f} mm",
        )
        ax2.legend()

        plt.tight_layout()

        if save_name and self.output_dir:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")

        if self.show_plots:
            plt.draw()
            plt.pause(0.1)  # Brief pause to allow display update
        else:
            plt.close()

    def plot_optimization_progress(self, save_name: Optional[str] = None):
        """Plot optimization progress.

        Args:
            save_name: Filename to save plot
        """
        if not self.error_history:
            return

        iterations = np.arange(len(self.error_history))
        errors = np.array(self.error_history)

        # Compute best so far
        best_so_far = np.minimum.accumulate(errors)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot errors
        ax.plot(iterations, errors * 1000, "b.", alpha=0.5, label="Current Error")
        ax.plot(iterations, best_so_far * 1000, "r-", linewidth=2, label="Best So Far")

        # Mark best point
        best_idx = np.argmin(errors)
        ax.scatter(
            best_idx,
            errors[best_idx] * 1000,
            c="green",
            s=100,
            marker="*",
            zorder=5,
            label=f"Best: {errors[best_idx]*1000:.3f} mm",
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE (mm)")
        ax.set_title("Optimization Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add convergence info
        if len(errors) > 10:
            recent_std = np.std(errors[-10:])
            ax.text(
                0.02,
                0.98,
                f"Recent std: {recent_std*1000:.3f} mm",
                transform=ax.transAxes,
                va="top",
            )

        plt.tight_layout()

        if save_name and self.output_dir:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")

        if self.show_plots:
            plt.draw()
            plt.pause(0.1)  # Brief pause to allow display update
        else:
            plt.close()

    def update_optimization_history(
        self, iteration: int, error: float, params: Optional[np.ndarray] = None
    ):
        """Update optimization history.

        Args:
            iteration: Iteration number
            error: Current error value
            params: Current parameters
        """
        self.iteration_history.append(iteration)
        self.error_history.append(error)

        if error < self.best_error:
            self.best_error = error
            self.best_params = params.copy() if params is not None else None

    def plot_parameter_evolution(
        self,
        param_history: List[np.ndarray],
        param_names: List[str],
        save_name: Optional[str] = None,
    ):
        """Plot parameter evolution during optimization.

        Args:
            param_history: List of parameter arrays
            param_names: Names of parameters
            save_name: Filename to save plot
        """
        if not param_history:
            return

        param_array = np.array(param_history)
        num_params = param_array.shape[1]

        # Create subplots
        cols = min(3, num_params)
        rows = (num_params + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        # Handle single subplot case
        if num_params == 1:
            axes = np.array([[axes]])
        elif rows == 1 and cols > 1:
            axes = axes.reshape(1, -1)
        elif cols == 1 and rows > 1:
            axes = axes.reshape(-1, 1)
        elif rows > 1 and cols > 1:
            # Already 2D, no reshape needed
            pass

        iterations = np.arange(len(param_history))

        for i, name in enumerate(param_names[:num_params]):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            ax.plot(iterations, param_array[:, i], "b-", alpha=0.7)
            ax.scatter(iterations, param_array[:, i], c="b", s=10, alpha=0.5)

            ax.set_xlabel("Iteration")
            ax.set_ylabel(name)
            ax.set_title(f"{name} Evolution")
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(num_params, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis("off")

        plt.suptitle("Parameter Evolution During Optimization")
        plt.tight_layout()

        if save_name and self.output_dir:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")

        if self.show_plots:
            plt.draw()
            plt.pause(0.1)  # Brief pause to allow display update
        else:
            plt.close()

    def create_summary_plot(
        self,
        real_positions: np.ndarray,
        best_simulated: np.ndarray,
        initial_simulated: Optional[np.ndarray] = None,
        save_name: str = "summary.png",
    ):
        """Create comprehensive summary plot.

        Args:
            real_positions: Real marker positions
            best_simulated: Best simulated positions
            initial_simulated: Initial simulated positions (optional)
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(15, 10))

        # 3D trajectory comparison
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax1.plot(
            real_positions[:, 0],
            real_positions[:, 1],
            real_positions[:, 2],
            "b-",
            label="Real",
            linewidth=2,
        )
        ax1.plot(
            best_simulated[:, 0],
            best_simulated[:, 1],
            best_simulated[:, 2],
            "g-",
            label="Optimized",
            linewidth=2,
        )
        if initial_simulated is not None:
            ax1.plot(
                initial_simulated[:, 0],
                initial_simulated[:, 1],
                initial_simulated[:, 2],
                "r--",
                label="Initial",
                alpha=0.5,
            )

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("3D Trajectories")
        ax1.legend()

        # XY projection
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(
            real_positions[:, 0], real_positions[:, 1], "b-", label="Real", linewidth=2
        )
        ax2.plot(
            best_simulated[:, 0],
            best_simulated[:, 1],
            "g-",
            label="Optimized",
            linewidth=2,
        )
        if initial_simulated is not None:
            ax2.plot(
                initial_simulated[:, 0],
                initial_simulated[:, 1],
                "r--",
                label="Initial",
                alpha=0.5,
            )
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title("XY Projection")
        ax2.legend()
        ax2.axis("equal")

        # XZ projection
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(
            real_positions[:, 0], real_positions[:, 2], "b-", label="Real", linewidth=2
        )
        ax3.plot(
            best_simulated[:, 0],
            best_simulated[:, 2],
            "g-",
            label="Optimized",
            linewidth=2,
        )
        if initial_simulated is not None:
            ax3.plot(
                initial_simulated[:, 0],
                initial_simulated[:, 2],
                "r--",
                label="Initial",
                alpha=0.5,
            )
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Z (m)")
        ax3.set_title("XZ Projection")
        ax3.legend()
        ax3.axis("equal")

        # Error over samples
        ax4 = fig.add_subplot(2, 3, 4)
        best_errors = np.linalg.norm(real_positions - best_simulated, axis=1) * 1000
        ax4.plot(best_errors, "g-", label="Optimized", linewidth=2)
        if initial_simulated is not None:
            initial_errors = (
                np.linalg.norm(real_positions - initial_simulated, axis=1) * 1000
            )
            ax4.plot(initial_errors, "r--", label="Initial", alpha=0.5)
        ax4.set_xlabel("Sample")
        ax4.set_ylabel("Error (mm)")
        ax4.set_title("Position Error")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Optimization progress
        if self.error_history:
            ax5 = fig.add_subplot(2, 3, 5)
            iterations = np.arange(len(self.error_history))
            errors = np.array(self.error_history)
            best_so_far = np.minimum.accumulate(errors)
            ax5.plot(iterations, errors * 1000, "b.", alpha=0.3)
            ax5.plot(iterations, best_so_far * 1000, "r-", linewidth=2)
            ax5.set_xlabel("Iteration")
            ax5.set_ylabel("RMSE (mm)")
            ax5.set_title("Optimization Progress")
            ax5.grid(True, alpha=0.3)

        # Statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis("off")

        stats_text = []
        if self.error_history:
            stats_text.append(f"Best RMSE: {min(self.error_history)*1000:.3f} mm")

        stats_text.append(f"Max Error: {np.max(best_errors):.3f} mm")
        stats_text.append(f"Mean Error: {np.mean(best_errors):.3f} mm")
        stats_text.append(f"Std Error: {np.std(best_errors):.3f} mm")

        if self.error_history:
            stats_text.append(f"Iterations: {len(self.error_history)}")

        ax6.text(
            0.1,
            0.9,
            "\n".join(stats_text),
            transform=ax6.transAxes,
            fontsize=12,
            va="top",
            family="monospace",
        )

        plt.suptitle("System Identification Results", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_name and self.output_dir:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")

        if self.show_plots:
            plt.draw()
            plt.pause(0.1)  # Brief pause to allow display update
        else:
            plt.close()

    def plot_actuation_comparison(
        self,
        servo_commands: np.ndarray,
        clark_commands: np.ndarray,
        mixed_commands: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Actuation Commands Comparison",
        save_name: Optional[str] = None,
    ):
        """Plot comparison of servo, clark-derived, and mixed actuation commands.

        Args:
            servo_commands: Raw servo commands in meters (N x 9)
            clark_commands: Clark-derived tendon commands in meters (N x 9)
            mixed_commands: Mixed actuation commands in meters (N x 9)
            timestamps: Optional timestamps for x-axis
            title: Plot title
            save_name: Filename to save plot
        """
        num_samples = len(servo_commands)
        num_actuators = servo_commands.shape[1]

        if timestamps is None:
            timestamps = np.arange(num_samples)
            x_label = "Sample"
        else:
            x_label = "Time (s)"

        # Create figure with subplots for each actuator
        cols = 3  # 3 columns for 3 segments
        rows = 3  # 3 rows for 3 tendons
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

        # Flatten axes for easier indexing
        axes_flat = axes.flatten()

        # Plot each actuator
        for i in range(num_actuators):
            ax = axes_flat[i]

            # Convert from meters to mm for plotting
            ax.plot(
                timestamps,
                servo_commands[:, i] * 1000,
                "b-",
                label="Servo",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.plot(
                timestamps,
                clark_commands[:, i] * 1000,
                "r--",
                label="Clark",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.plot(
                timestamps,
                mixed_commands[:, i] * 1000,
                "g:",
                label="Mixed",
                linewidth=2,
                alpha=0.8,
            )

            # Determine segment and tendon indices
            seg_idx = i // 3
            ten_idx = i % 3

            ax.set_title(f"Seg {seg_idx}, Ten {ten_idx}", fontsize=10)
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel("Command (mm)", fontsize=8)
            ax.grid(True, alpha=0.3)

            # Only show legend on first subplot
            if i == 0:
                ax.legend(fontsize=8)

            # Compute statistics
            servo_mean = np.mean(servo_commands[:, i]) * 1000
            clark_mean = np.mean(clark_commands[:, i]) * 1000
            diff = abs(servo_mean - clark_mean)

            # Add text with difference
            ax.text(
                0.02,
                0.98,
                f"Δ: {diff:.2f}mm",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
            )

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save the plot if requested
        if save_name and self.output_dir:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved actuation comparison plot: {save_path}")

        # Show interactively if requested
        if self.show_plots:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.close()
