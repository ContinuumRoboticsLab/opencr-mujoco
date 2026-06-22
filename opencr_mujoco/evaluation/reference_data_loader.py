"""
Reference data loader for SoroSim reference datasets.

Loads the two live reference formats:
- SoroSim static-equilibrium CSV files (``sorosim_statics/``)
- SoroSim 13-column tip-release dynamics text files (``sorosim_dynamics/``)
plus the saved simulation pickles / CSVs produced by the evaluator.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def validate_frame_conversion(
    frame_conversion: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Coerce a frame_conversion to a validated 3x3 float ndarray (or None).

    Single source of truth for the 3x3 check shared by the evaluator, the
    reference loader, the visualizer, and evaluate.py. Returns None when
    ``frame_conversion`` is None; otherwise returns it as a float ndarray,
    raising ValueError if it is not 3x3.
    """
    if frame_conversion is None:
        return None
    R = np.asarray(frame_conversion, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"frame_conversion must be 3x3, got shape {R.shape}")
    return R


class ReferenceDataLoader:
    """Load and manage reference data for evaluation."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        frame_conversion: Optional[np.ndarray] = None,
    ):
        """
        Initialize the reference data loader.

        Args:
            data_dir: Directory containing reference data files
            frame_conversion: Optional 3x3 matrix R such that
                ``mujoco_vec = R @ file_vec`` for any 3-vector
                (positions, forces, moments, gravity, Euler triples).
                If None, no frame conversion is applied — reference data is
                returned in its original file frame.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        self.frame_conversion = validate_frame_conversion(frame_conversion)

    def _to_mujoco(self, vec3: np.ndarray) -> np.ndarray:
        """Apply frame_conversion (R @ vec) if configured, else identity.

        Accepts an array whose last axis is 3 and returns the same shape.
        """
        if self.frame_conversion is None:
            return np.asarray(vec3, dtype=float)
        v = np.asarray(vec3, dtype=float)
        return v @ self.frame_conversion.T

    def load_tip_release_data(self, test_type: str) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        float,
        np.ndarray,
        float,
        np.ndarray,
    ]:
        """
        Load SoroSim 13-column tip-release dynamics data from a text file.

        File layout (tab-separated, 13 columns):
        - Line 1: damping, then zeros, then the gravity vector
                  (damping, 0, 0, 0, 0, 0, 0, 0, 0, 0, gx, gy, gz)
        - Line 2: holding wrenches at mid and tip
                  (0, mx_mid, my_mid, mz_mid, fx_mid, fy_mid, fz_mid,
                      mx_tip, my_tip, mz_tip, fx_tip, fy_tip, fz_tip)
          The stored values are the applied moments AS-IS (no sign flip).
          The "-0" entries that appear in the steel files are cosmetic
          SoroSim-exporter artifacts, not a negation convention — verified
          against the t=0 held equilibrium of the TPU tests, which have
          nonzero moments (~5 mm tip error as-is vs ~560 mm if negated).
        - Lines 3+: time, mid Euler(3), mid pos(3), tip Euler(3), tip pos(3)

        All 3-vectors are converted from the file frame to the MuJoCo frame
        via ``self._to_mujoco`` when a frame_conversion is configured.

        Args:
            test_type: Reference file base name (e.g. "SpringSteelRodMuJoCo_1").

        Returns:
            wrenches: tuple (mid_wrench, tip_wrench), each (6,) as [f(3), m(3)]
            poses: tuple (mid_poses, tip_poses), each (N, 6) as [pos(3), euler(3)]
            dt: time step inferred from the timestamps
            timestamps: (N,) time series
            damping_ratio: damping coefficient from line 1
            gravity: (3,) gravity vector in MuJoCo frame
        """
        # Use test_type directly as filename base
        filename = f"{test_type}.txt"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Tip release data file not found: {filepath}")

        # Load all data
        data = np.loadtxt(filepath, delimiter="\t")

        # Detect format based on number of columns
        num_cols = data.shape[1]

        if num_cols == 13:
            # sorosim_dynamics format with mid and tip tracking.
            # All 3-vectors below are converted from file frame to MuJoCo
            # frame via self._to_mujoco when a frame_conversion is configured.
            # Line 1: damping, zeros..., gx, gy, gz (last 3 = gravity in file coords)
            damping_ratio = data[0, 0]
            gravity = self._to_mujoco(data[0, -3:])

            # Line 2: 0, mx_mid, my_mid, mz_mid, fx_mid, fy_mid, fz_mid,
            #            mx_tip, my_tip, mz_tip, fx_tip, fy_tip, fz_tip
            # (moments used as stored; see docstring)
            mid_m_file = data[1, 1:4]
            mid_f_file = data[1, 4:7]
            tip_m_file = data[1, 7:10]
            tip_f_file = data[1, 10:13]

            mid_wrench = np.concatenate(
                [self._to_mujoco(mid_f_file), self._to_mujoco(mid_m_file)]
            )
            tip_wrench = np.concatenate(
                [self._to_mujoco(tip_f_file), self._to_mujoco(tip_m_file)]
            )

            # Remaining lines: time, mid_euler(3), mid_pos(3), tip_euler(3), tip_pos(3)
            timestamps = data[2:, 0]

            mid_eul_file = data[2:, 1:4]
            mid_pos_file = data[2:, 4:7]
            tip_eul_file = data[2:, 7:10]
            tip_pos_file = data[2:, 10:13]

            mid_poses = np.column_stack(
                [self._to_mujoco(mid_pos_file), self._to_mujoco(mid_eul_file)]
            )
            tip_poses = np.column_stack(
                [self._to_mujoco(tip_pos_file), self._to_mujoco(tip_eul_file)]
            )

            # Calculate dt from timestamps
            if len(timestamps) > 1:
                dt = timestamps[1] - timestamps[0]
            else:
                dt = 0.005  # Default to 200Hz based on observed data

            return (
                (mid_wrench, tip_wrench),
                (mid_poses, tip_poses),
                dt,
                timestamps,
                damping_ratio,
                gravity,
            )

        raise ValueError(
            f"Unexpected number of columns in {filepath}: {num_cols}. "
            f"Expected 13 (SoroSim dynamics format)."
        )

    def load_csv_results(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load simulation results from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with simulation results
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        return pd.read_csv(filepath)

    def load_link_positions(self, filepath: Union[str, Path]) -> List[List[np.ndarray]]:
        """
        Load link positions from pickle file.

        Args:
            filepath: Path to pickle file

        Returns:
            List of link position arrays
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Pickle file not found: {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def save_link_positions(
        self, positions: List[List[np.ndarray]], filepath: Union[str, Path]
    ):
        """
        Save link positions to pickle file.

        Args:
            positions: List of link position arrays
            filepath: Path to save pickle file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(positions, f)

    def parse_wrench_string(self, wrench_str: str) -> List[float]:
        """
        Parse wrench string from CSV format.

        Args:
            wrench_str: Comma-separated string of wrench components

        Returns:
            List of float values [fx, fy, fz, mx, my, mz]
        """
        return [float(x) for x in wrench_str.split(",")]

    def format_wrench_for_csv(
        self, wrench: Union[List[float], Tuple[float, ...]]
    ) -> str:
        """
        Format wrench data for CSV storage.

        Args:
            wrench: Wrench components [fx, fy, fz, mx, my, mz]

        Returns:
            Comma-separated string
        """
        return ",".join(map(str, wrench))

    def load_sorosim_statics_csv(self, test_type: str) -> Tuple[Dict, int, np.ndarray]:
        """
        Load SoroSim static equilibrium data from CSV file.

        CSV Format (from sorosim_statics directory):
        - Row 1: Column labels (rowLabel, gravity, mid_wrench, tip_wrench, seg1_s01, ...)
        - Row 2: Arc length markers
        - Rows 3+: Data in groups of 6 rows per shape:
            * EulX_sXXX, EulY_sXXX, EulZ_sXXX (Euler angles)
            * Px_sXXX, Py_sXXX, Pz_sXXX (positions)

        Each row contains:
        - Gravity component (single value)
        - Mid-point wrench component (single value)
        - Tip wrench component (single value)
        - Position/angle values at arc lengths for each segment

        Args:
            test_type: Test type name (e.g., "SpringSteelRodMuJoCo", "TPURodMuJoCo")

        Returns:
            Tuple of (data_dict, num_samples, arc_lengths) where:
                data_dict: Maps (mid_wrench, tip_wrench, gravity) to link positions
                num_samples: Number of arc-length sample columns per shape
                arc_lengths: (num_samples,) normalized global arc positions in
                    [0, 1] of each sample column (the SoroSim measurement
                    stations are NOT uniformly spaced)
        """
        # Try different possible filenames
        possible_files = [
            self.data_dir / "sorosim_statics" / f"{test_type}_dataStatics.csv",
            self.data_dir / "sorosim_statics" / f"{test_type}.csv",
        ]

        filepath = None
        for path in possible_files:
            if path.exists():
                filepath = path
                break

        if filepath is None:
            raise FileNotFoundError(
                f"SoroSim static CSV file not found. Tried: {possible_files}"
            )

        # Load CSV
        df = pd.read_csv(filepath)

        # Parse header to get arc lengths and segment structure
        arc_lengths = []
        segment_columns = []
        segment_indices = {}  # Maps segment number to (start_idx, end_idx)

        # Second row contains arc lengths
        arc_row = df.iloc[0]

        # Parse columns to identify segments
        current_segment = None
        segment_start_idx = 0

        for col in df.columns:
            if col.startswith("seg"):
                segment_columns.append(col)
                # Extract segment number from column name (e.g., "seg1_s01" -> 1)
                seg_num = int(col.split("_")[0].replace("seg", ""))

                # Track segment boundaries
                if current_segment is None:
                    current_segment = seg_num
                elif seg_num != current_segment:
                    # New segment started
                    segment_indices[current_segment] = (
                        segment_start_idx,
                        len(segment_columns) - 1,
                    )
                    current_segment = seg_num
                    segment_start_idx = len(segment_columns) - 1

        # Add final segment
        if current_segment is not None:
            segment_indices[current_segment] = (segment_start_idx, len(segment_columns))

        num_segments = len(segment_indices)
        num_links = len(segment_columns)

        # Convert per-segment normalized arc lengths to global arc lengths
        for i, col in enumerate(segment_columns):
            seg_num = int(col.split("_")[0].replace("seg", ""))
            local_arc_length = float(arc_row[col])

            # Global arc length = (segment_index + local_arc_length) / num_segments
            global_arc_length = (seg_num - 1 + local_arc_length) / num_segments
            arc_lengths.append(global_arc_length)

        # Parse shapes (skip header row, then process in groups of 6)
        data_dict = {}
        num_shapes = (len(df) - 1) // 6

        for shape_idx in range(num_shapes):
            start_row = 1 + shape_idx * 6

            # Get the 6 rows for this shape
            euler_x = df.iloc[start_row + 0]
            euler_y = df.iloc[start_row + 1]
            euler_z = df.iloc[start_row + 2]
            pos_x = df.iloc[start_row + 3]
            pos_y = df.iloc[start_row + 4]
            pos_z = df.iloc[start_row + 5]

            # Extract gravity, wrenches, link positions in file frame, then
            # convert each 3-vector to MuJoCo frame via self._to_mujoco.
            gravity_file = np.array(
                [pos_x["gravity"], pos_y["gravity"], pos_z["gravity"]]
            )
            gravity_vec = tuple(self._to_mujoco(gravity_file).tolist())

            mid_f_file = np.array(
                [pos_x["mid_wrench"], pos_y["mid_wrench"], pos_z["mid_wrench"]]
            )
            mid_m_file = np.array(
                [euler_x["mid_wrench"], euler_y["mid_wrench"], euler_z["mid_wrench"]]
            )
            tip_f_file = np.array(
                [pos_x["tip_wrench"], pos_y["tip_wrench"], pos_z["tip_wrench"]]
            )
            tip_m_file = np.array(
                [euler_x["tip_wrench"], euler_y["tip_wrench"], euler_z["tip_wrench"]]
            )

            mid_f = self._to_mujoco(mid_f_file)
            mid_m = self._to_mujoco(mid_m_file)
            tip_f = self._to_mujoco(tip_f_file)
            tip_m = self._to_mujoco(tip_m_file)

            mid_wrench = tuple(np.concatenate([mid_f, mid_m]).tolist())
            tip_wrench = tuple(np.concatenate([tip_f, tip_m]).tolist())

            # Extract link positions and convert frame component-wise
            link_pos_file = np.array(
                [[pos_x[col], pos_y[col], pos_z[col]] for col in segment_columns]
            )
            link_positions = [tuple(p) for p in self._to_mujoco(link_pos_file).tolist()]

            data_dict[(mid_wrench, tip_wrench, gravity_vec)] = link_positions

        return data_dict, num_links, np.asarray(arc_lengths)
