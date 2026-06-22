"""Geometric parameter identification from tendon pull data.

Identifies seg_offsets and tendon_angle_deltas by analyzing the deflection
direction of each individual tendon pull. Refactored from identify_tendon_offsets.py
for programmatic use in the sysid pipeline.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class GeometricIdentifier:
    """Identifies geometric tendon parameters from individual tendon pull data.

    Analyzes the deflection direction of each tendon pull to determine the
    actual angular position of each tendon, then computes per-tendon deltas
    relative to expected uniform spacing.
    """

    def __init__(self, config: Dict):
        """Initialize geometric identifier.

        Args:
            config: Dict with keys:
                - num_segments: Number of TDCR segments (default 3)
                - tendons_per_segment: Tendons per segment (default 3)
        """
        self.num_segments = config.get("num_segments", 3)
        self.tendons_per_segment = config.get("tendons_per_segment", 3)
        self.total_tendons = self.num_segments * self.tendons_per_segment

    def identify(self, df: pd.DataFrame) -> Dict:
        """Run full geometric identification on tendon pull data.

        Args:
            df: DataFrame with tendon pull data. Must contain:
                - pattern_label column with 'tendonX_idY_pull' patterns
                - tip_x/y/z or marker_x/y/z columns
                - servo_*_mm columns

        Returns:
            Dict with:
                - servo_mapping: {tendon_idx: hardware_servo_id}
                - position_bias: [bx, by, bz] neutral tip position in mm
                - deflection_angles: {tendon_idx: (angle_rad, magnitude_mm)}
                - tendon_angle_deltas: [[s0t0, s0t1, s0t2], ...] (centered, radians)
                - seg_offsets: [off0, off1, off2] (mean rotation per segment, radians)
        """
        # Detect tip column names
        xy_cols = self._get_xy_columns(df)

        # Parse servo mapping from pattern labels
        servo_mapping = self.detect_servo_mapping_from_labels(df)
        print(f"\nServo mapping from labels:")
        for tidx in sorted(servo_mapping):
            seg = tidx // self.tendons_per_segment
            ten = tidx % self.tendons_per_segment
            print(f"  tendon{tidx} → servo_id{servo_mapping[tidx]} → seg{seg}_ten{ten}")

        # Compute position bias (neutral tip position)
        position_bias = self.compute_position_bias(df, xy_cols)
        print(
            f"\nPosition bias (neutral): ({position_bias[0]:.2f}, {position_bias[1]:.2f}, {position_bias[2]:.2f}) mm"
        )

        # Compute deflection angles for each tendon pull
        deflection_angles = self.compute_deflection_angles(
            df, position_bias[:2], xy_cols
        )

        # Build sim_mapping: {tendon_idx: (segment, tendon_in_segment)}
        sim_mapping = {}
        for tidx in servo_mapping:
            seg = tidx // self.tendons_per_segment
            ten = tidx % self.tendons_per_segment
            sim_mapping[tidx] = (seg, ten)

        # Compute per-tendon deltas and seg_offsets
        tendon_angle_deltas, seg_offsets = self.compute_per_tendon_deltas(
            deflection_angles, sim_mapping
        )

        # Print results
        theta_deg = 360.0 / self.tendons_per_segment
        print(
            f"\n--- Per-Tendon Angular Deltas (deviation from {theta_deg:.0f}° uniform spacing) ---"
        )
        for s, seg_deltas in enumerate(tendon_angle_deltas):
            deltas_str = ", ".join(f"{np.degrees(d):+.2f}°" for d in seg_deltas)
            print(f"  Seg {s}: [{deltas_str}]")

        print(f"\n--- Segment Offsets ---")
        for s, off in enumerate(seg_offsets):
            print(f"  Seg {s}: {np.degrees(off):.2f}° ({off:.4f} rad)")

        return {
            "servo_mapping": servo_mapping,
            "position_bias": position_bias.tolist(),
            "deflection_angles": {
                str(k): (float(v[0]), float(v[1])) for k, v in deflection_angles.items()
            },
            "tendon_angle_deltas": [
                [round(d, 4) for d in seg] for seg in tendon_angle_deltas
            ],
            "seg_offsets": [round(off, 4) for off in seg_offsets],
        }

    def _get_xy_columns(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """Detect tip/marker column names.

        Returns:
            Tuple of (x_col, y_col, z_col) column names.
        """
        if "tip_x" in df.columns:
            return ("tip_x", "tip_y", "tip_z")
        elif "marker_x" in df.columns:
            return ("marker_x", "marker_y", "marker_z")
        else:
            raise ValueError("No tip_x/y/z or marker_x/y/z columns found")

    def detect_servo_mapping_from_labels(self, df: pd.DataFrame) -> Dict[int, int]:
        """Parse servo mapping from pattern labels.

        Pattern format: 'rep{N}_tendon{X}_id{Y}_pull[_hold]'
        Extracts tendon index X and hardware servo ID Y.

        Args:
            df: DataFrame with pattern_label column

        Returns:
            Dict mapping tendon_index -> hardware_servo_id
        """
        if "pattern_label" not in df.columns:
            raise ValueError("DataFrame must have 'pattern_label' column")

        mapping = {}
        pattern = re.compile(r"rep\d+_tendon(\d+)_id(\d+)_pull")

        for label in df["pattern_label"].unique():
            match = pattern.match(str(label))
            if match:
                tendon_idx = int(match.group(1))
                servo_id = int(match.group(2))
                if tendon_idx not in mapping:
                    mapping[tendon_idx] = servo_id

        if len(mapping) != self.total_tendons:
            found = sorted(mapping.keys())
            expected = list(range(self.total_tendons))
            print(f"  Warning: found mapping for tendons {found}, expected {expected}")

        return mapping

    def compute_position_bias(
        self, df: pd.DataFrame, xy_cols: Optional[Tuple[str, str, str]] = None
    ) -> np.ndarray:
        """Compute neutral tip position from first sample of each tendon pull.

        Args:
            df: DataFrame with tendon pull data
            xy_cols: Tuple of (x_col, y_col, z_col) column names

        Returns:
            np.ndarray of shape (3,) with [bias_x, bias_y, bias_z] in mm
        """
        if xy_cols is None:
            xy_cols = self._get_xy_columns(df)
        x_col, y_col, z_col = xy_cols

        # Get unique tendon pull labels (exclude _hold patterns)
        pull_labels = [
            label
            for label in df["pattern_label"].unique()
            if re.match(r"rep\d+_tendon\d+_id\d+_pull$", str(label))
        ]

        if not pull_labels:
            raise ValueError("No tendon pull patterns found in data")

        neutrals = []
        for label in pull_labels:
            group = df[df["pattern_label"] == label]
            first = group.iloc[0]
            neutrals.append([first[x_col], first[y_col], first[z_col]])

        return np.mean(neutrals, axis=0)

    def compute_deflection_angles(
        self,
        df: pd.DataFrame,
        neutral_xy: np.ndarray,
        xy_cols: Optional[Tuple[str, str, str]] = None,
    ) -> Dict[int, Tuple[float, float]]:
        """Compute deflection angle for each tendon pull using return-edge tangent.

        For each tendon pull, finds the peak deflection, then computes the
        tangent direction of the return path (from last returning point back
        toward peak). This measures the elastic restoring direction, which
        gives a cleaner angular signal than the peak position alone.

        Args:
            df: DataFrame with tendon pull data
            neutral_xy: [neutral_x, neutral_y] reference position in mm
            xy_cols: Tuple of (x_col, y_col, z_col)

        Returns:
            Dict mapping tendon_index -> (angle_rad, magnitude_mm)
        """
        if xy_cols is None:
            xy_cols = self._get_xy_columns(df)
        x_col, y_col, _ = xy_cols

        # Get unique tendon pull labels (exclude _hold)
        pull_labels = sorted(
            [
                label
                for label in df["pattern_label"].unique()
                if re.match(r"rep\d+_tendon\d+_id\d+_pull$", str(label))
            ]
        )

        # Group by tendon index, average across repetitions
        tendon_angles = {}  # tendon_idx -> list of (angle, magnitude)

        for label in pull_labels:
            match = re.match(r"rep\d+_tendon(\d+)_id\d+_pull$", label)
            if not match:
                continue
            tendon_idx = int(match.group(1))

            group = df[df["pattern_label"] == label]
            positions_xy = group[[x_col, y_col]].values

            # Peak = max displacement from first sample
            displacements = np.linalg.norm(positions_xy - positions_xy[0], axis=1)
            peak_idx = np.argmax(displacements)
            magnitude = displacements[peak_idx]

            # Return-edge tangent: direction from last returning point back toward peak
            # This measures the elastic restoring direction
            ret = positions_xy[peak_idx:]
            if len(ret) >= 2:
                dx = ret[0, 0] - ret[-1, 0]
                dy = ret[0, 1] - ret[-1, 1]
            else:
                # Fallback: use peak position relative to neutral
                dx = positions_xy[peak_idx, 0] - neutral_xy[0]
                dy = positions_xy[peak_idx, 1] - neutral_xy[1]
            angle = np.arctan2(dy, dx)

            if tendon_idx not in tendon_angles:
                tendon_angles[tendon_idx] = []
            tendon_angles[tendon_idx].append((angle, magnitude))

        # Average across repetitions using circular mean for angles
        results = {}
        for tendon_idx, measurements in tendon_angles.items():
            if len(measurements) == 1:
                results[tendon_idx] = measurements[0]
            else:
                # Circular mean for angles
                sin_sum = sum(np.sin(m[0]) for m in measurements)
                cos_sum = sum(np.cos(m[0]) for m in measurements)
                avg_angle = np.arctan2(sin_sum, cos_sum)
                avg_magnitude = np.mean([m[1] for m in measurements])
                results[tendon_idx] = (avg_angle, avg_magnitude)

        return results

    def compute_per_tendon_deltas(
        self,
        deflection_angles: Dict[int, Tuple[float, float]],
        sim_mapping: Dict[int, Tuple[int, int]],
    ) -> Tuple[List[List[float]], List[float]]:
        """Compute per-tendon angular deltas and segment offsets.

        Adapted from identify_tendon_offsets.py compute_per_tendon_deltas().

        Args:
            deflection_angles: Dict mapping tendon_index -> (angle_rad, magnitude_mm)
            sim_mapping: Dict mapping tendon_index -> (segment, tendon_in_segment)

        Returns:
            Tuple of:
                - tendon_angle_deltas: Nested list [[s0t0, s0t1, s0t2], ...] (centered)
                - seg_offsets: List of per-segment mean rotation offsets
        """
        theta = 2 * np.pi / self.tendons_per_segment

        # Group deflection angles by segment and tendon
        sim_angles = [
            [None] * self.tendons_per_segment for _ in range(self.num_segments)
        ]
        for tendon_idx, (seg, ten) in sim_mapping.items():
            if tendon_idx in deflection_angles:
                sim_angles[seg][ten] = deflection_angles[tendon_idx][0]

        all_deltas = []
        seg_offsets = []

        for s in range(self.num_segments):
            angles = sim_angles[s]
            if any(a is None for a in angles):
                print(f"  Warning: missing tendon data for segment {s}")
                all_deltas.append([0.0] * self.tendons_per_segment)
                seg_offsets.append(0.0)
                continue

            # Compute observed spacing relative to first tendon (ten0)
            ref_angle = angles[0]
            observed_relative = []
            for t in range(self.tendons_per_segment):
                rel = (angles[t] - ref_angle) % (2 * np.pi)
                observed_relative.append(rel)

            # Expected relative angles: [0, theta, 2*theta, ...]
            expected_relative = [t * theta for t in range(self.tendons_per_segment)]

            # Deviations from expected uniform spacing
            raw_deltas = []
            for t in range(self.tendons_per_segment):
                delta = observed_relative[t] - expected_relative[t]
                delta = (delta + np.pi) % (2 * np.pi) - np.pi
                raw_deltas.append(delta)

            # Mean delta becomes the segment offset (overall rotation of this segment's tendons)
            # The return-edge tangent points in the deflection direction, which is the
            # tendon position (pulling a tendon bends toward it, tip deflects that way).
            mean_delta = np.mean(raw_deltas)
            seg_offsets.append(float(ref_angle + mean_delta))

            # Center the deltas (mean=0) so seg_offsets handles the mean rotation
            centered_deltas = [d - mean_delta for d in raw_deltas]
            all_deltas.append(centered_deltas)

        return all_deltas, seg_offsets

    def apply_to_generation_config(self, results: Dict, config: Dict) -> Dict:
        """Apply geometric identification results to a generation config.

        Sets seg_offsets and tendon_angle_deltas in the config.

        Args:
            results: Output from identify()
            config: Generation config dict (modified in place and returned)

        Returns:
            Updated generation config
        """
        config["seg_offsets"] = results["seg_offsets"]
        config["tendon_angle_deltas"] = results["tendon_angle_deltas"]
        return config
