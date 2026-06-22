"""Data loader and CSV preprocessor for the sysid pipeline.

Handles the new data format with tip_x/y/z columns and non-sequential
servo naming (servo_4_mm, servo_3_mm, etc.), and preprocesses CSVs into
the standard format expected by TrajectoryDataLoader.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class PipelineDataLoader:
    """Loads raw sysid CSV data and preprocesses it for the optimizer.

    Supports both old format (marker_x/y/z) and new format (tip_x/y/z).
    Handles servo column reordering from hardware IDs to simulation actuator order.
    """

    def __init__(self, data_file: str, config: Optional[Dict] = None):
        self.data_file = Path(data_file)
        self.config = config or {}
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

    def load_raw_dataframe(self) -> pd.DataFrame:
        """Load CSV into a DataFrame without any transformations."""
        return pd.read_csv(self.data_file)

    def detect_tip_columns(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Detect whether data uses tip_x/y/z or marker_x/y/z columns.

        Returns:
            List of column names for tip/marker positions, e.g. ['tip_x', 'tip_y', 'tip_z']
        """
        if df is None:
            df = self.load_raw_dataframe()

        if all(c in df.columns for c in ["tip_x", "tip_y", "tip_z"]):
            return ["tip_x", "tip_y", "tip_z"]
        elif all(c in df.columns for c in ["marker_x", "marker_y", "marker_z"]):
            return ["marker_x", "marker_y", "marker_z"]
        else:
            raise ValueError(
                "CSV must contain either tip_x/y/z or marker_x/y/z columns. "
                f"Found columns: {list(df.columns)}"
            )

    def get_servo_columns(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Get servo column names sorted by servo number.

        Returns:
            List of servo column names sorted by hardware ID number.
        """
        if df is None:
            df = self.load_raw_dataframe()

        servo_cols = [
            c for c in df.columns if c.startswith("servo_") and c.endswith("_mm")
        ]
        if not servo_cols:
            raise ValueError("No servo_*_mm columns found in data")

        return sorted(servo_cols, key=lambda x: int(x.split("_")[1]))

    def preprocess_to_standard_csv(
        self,
        output_path: str,
        servo_mapping: Dict[int, int],
        position_bias: Optional[np.ndarray] = None,
    ) -> Path:
        """Preprocess raw CSV into standard format for TrajectoryDataLoader.

        Performs:
        1. Detect and rename tip columns to marker_x/y/z
        2. Subtract position bias from tip positions
        3. Reorder servo columns to simulation actuator order
        4. Rename servo columns to sequential servo_1_mm..servo_N_mm

        Args:
            output_path: Path to write preprocessed CSV
            servo_mapping: Dict mapping tendon_index -> hardware_servo_id.
                Used to reorder columns so column order matches MJCF actuator order
                (base segment → middle → tip).
            position_bias: Optional [bx, by, bz] to subtract from tip positions (in mm).

        Returns:
            Path to the preprocessed CSV file.
        """
        df = self.load_raw_dataframe()
        tip_cols = self.detect_tip_columns(df)

        # Subtract position bias
        if position_bias is not None:
            for i, col in enumerate(tip_cols):
                df[col] = df[col] - position_bias[i]

        # Rename tip columns to marker_x/y/z
        if tip_cols[0] != "marker_x":
            rename_map = {
                tip_cols[0]: "marker_x",
                tip_cols[1]: "marker_y",
                tip_cols[2]: "marker_z",
            }
            df = df.rename(columns=rename_map)

        # Reorder servo columns to simulation actuator order
        # servo_mapping: {tendon_idx: hardware_servo_id}
        # tendon_idx 0..N-1 corresponds to MJCF actuator order (base→tip)
        num_tendons = len(servo_mapping)
        new_servo_cols = []
        for tendon_idx in range(num_tendons):
            hw_id = servo_mapping[tendon_idx]
            col_name = f"servo_{hw_id}_mm"
            if col_name not in df.columns:
                raise ValueError(
                    f"Servo column {col_name} not found for tendon {tendon_idx}. "
                    f"Available: {self.get_servo_columns(df)}"
                )
            new_servo_cols.append(col_name)

        # Build output DataFrame with standard column names
        out_cols = ["timestamp"]

        # Add pattern_label if present
        has_pattern = "pattern_label" in df.columns
        if has_pattern:
            out_cols.append("pattern_label")

        # Rename servo columns to sequential servo_1_mm, servo_2_mm, ...
        servo_rename = {}
        for i, old_col in enumerate(new_servo_cols):
            new_col = f"servo_{i + 1}_mm"
            servo_rename[old_col] = new_col
            out_cols.append(new_col)

        out_cols.extend(["marker_x", "marker_y", "marker_z"])

        df = df.rename(columns=servo_rename)
        df = df[out_cols]

        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        print(
            f"Preprocessed CSV written to {output_path} ({len(df)} rows, {len(out_cols)} columns)"
        )
        return output_path
