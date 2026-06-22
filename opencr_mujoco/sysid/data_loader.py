"""Data loader for real robot trajectory data."""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class TrajectoryDataLoader:
    """Loads and preprocesses real robot trajectory data from CSV files."""

    def __init__(
        self,
        data_file: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        """Initialize data loader.

        Args:
            data_file: Path to CSV file with trajectory data
            config: Optional configuration for data processing
            verbose: If True, print a summary of the loaded data (gated off by
                default — the parallel optimizer loads this in every worker)
        """
        self.data_file = Path(data_file)
        self.config = config or {}
        self.verbose = verbose
        self.data = None
        self.timestamps = None
        self.actuator_commands = None
        self.marker_positions = None
        self.clark_coords = None

        # Load data
        self._load_data()

    def _load_data(self):
        """Load and parse CSV data."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # Read CSV
        self.data = pd.read_csv(self.data_file)

        # Extract timestamps
        if "timestamp" in self.data.columns:
            self.timestamps = self.data["timestamp"].values
        else:
            # Generate timestamps if not present
            num_samples = len(self.data)
            dt = self.config.get("sample_dt", 0.01)  # 100Hz default
            self.timestamps = np.arange(num_samples) * dt

        # Extract actuator commands (servo positions in mm)
        servo_cols = [col for col in self.data.columns if col.startswith("servo_")]
        if servo_cols:
            # Sort by servo number
            servo_cols_sorted = sorted(servo_cols, key=lambda x: int(x.split("_")[1]))
            self.actuator_commands = self.data[servo_cols_sorted].values
            # Convert from mm to m
            self.actuator_commands = self.actuator_commands / 1000.0
        else:
            raise ValueError("No servo columns found in data")

        # Extract marker positions
        if all(
            col in self.data.columns for col in ["marker_x", "marker_y", "marker_z"]
        ):
            self.marker_positions = self.data[
                ["marker_x", "marker_y", "marker_z"]
            ].values
            # Convert from mm to m if needed
            if self.config.get("marker_units", "mm") == "mm":
                self.marker_positions = self.marker_positions / 1000.0
        else:
            raise ValueError("Marker position columns not found in data")

        # Extract Clark coordinates if available
        clark_cols = [
            col for col in self.data.columns if col.startswith("clark_coord_")
        ]
        if clark_cols:
            clark_cols_sorted = sorted(clark_cols, key=lambda x: int(x.split("_")[-1]))
            self.clark_coords = self.data[clark_cols_sorted].values

        if self.verbose:
            print("Loaded trajectory data:")
            print(f"  - {len(self.data)} samples")
            print(f"  - {len(servo_cols)} actuators")
            # Span, not the raw last stamp (recordings use absolute unix time)
            duration = self.timestamps[-1] - self.timestamps[0]
            print(f"  - Duration: {duration:.2f} seconds")
            print(f"  - Sample rate: {1.0/np.mean(np.diff(self.timestamps)):.1f} Hz")

    def get_trajectory_segment(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """Get a segment of the trajectory.

        Args:
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end)

        Returns:
            Dict with timestamps, actuator_commands, and marker_positions
        """
        # Find time indices
        if start_time is None:
            start_idx = 0
        else:
            start_idx = np.searchsorted(self.timestamps, start_time)

        if end_time is None:
            end_idx = len(self.timestamps)
        else:
            end_idx = np.searchsorted(self.timestamps, end_time)

        segment = {
            "timestamps": self.timestamps[start_idx:end_idx],
            "actuator_commands": self.actuator_commands[start_idx:end_idx],
            "marker_positions": self.marker_positions[start_idx:end_idx],
        }

        if self.clark_coords is not None:
            segment["clark_coords"] = self.clark_coords[start_idx:end_idx]

        return segment

    def get_full_trajectory(self) -> Dict[str, np.ndarray]:
        """Get the full trajectory data.

        Returns:
            Dict with all trajectory data
        """
        return self.get_trajectory_segment()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the trajectory data.

        Returns:
            Dict with trajectory statistics
        """
        stats = {
            "num_samples": len(self.timestamps),
            "duration": self.timestamps[-1] - self.timestamps[0],
            "sample_rate": 1.0 / np.mean(np.diff(self.timestamps)),
            "num_actuators": self.actuator_commands.shape[1],
            "marker_range": {
                "x": (
                    self.marker_positions[:, 0].min(),
                    self.marker_positions[:, 0].max(),
                ),
                "y": (
                    self.marker_positions[:, 1].min(),
                    self.marker_positions[:, 1].max(),
                ),
                "z": (
                    self.marker_positions[:, 2].min(),
                    self.marker_positions[:, 2].max(),
                ),
            },
            "actuator_range": {
                f"actuator_{i}": (
                    self.actuator_commands[:, i].min(),
                    self.actuator_commands[:, i].max(),
                )
                for i in range(self.actuator_commands.shape[1])
            },
        }

        if self.clark_coords is not None:
            stats["clark_range"] = {
                f"clark_{i}": (
                    self.clark_coords[:, i].min(),
                    self.clark_coords[:, i].max(),
                )
                for i in range(self.clark_coords.shape[1])
            }

        return stats
