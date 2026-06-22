import numpy as np
from typing import List, Union
from .n_tendon_segment_kinematics import NTendonSegmentKinematics


class MultiSegmentTDCRTensionKinematics:
    """
    Multi-segment TDCR kinematics for tension control mode.

    Key difference from MultiSegmentTDCRKinematics:
    - In position control: upper segment tendons accumulate contributions from lower segments
    - In tension control: each segment is controlled independently with no cross-segment coupling

    This allows independent force control of each segment without compensating for
    tendon routing through other segments.
    """

    def __init__(
        self,
        n_tendons_per_segment: Union[List[int], np.ndarray],
        tendon_distances_mm: Union[List[float], np.ndarray, float],
        angle_offsets_rad_ccw: Union[List[float], np.ndarray] = None,
        max_bending_angles_rad: Union[List[float], np.ndarray, float] = 10000.0,
    ):
        """
        Initialize multi-segment TDCR tension kinematics.

        Args:
            n_tendons_per_segment: List of number of tendons for each segment
            tendon_distances_mm: Distance from tendons to backbone for each segment
                                Can be a single value (same for all) or list
            angle_offsets_rad_ccw: Angular offset for each segment's tendons
                                   If None, defaults to zeros
            max_bending_angles_rad: Maximum bending angle for each segment
                                    Can be a single value or list
        """
        self.n_segments = len(n_tendons_per_segment)
        self.n_tendons_per_segment = np.array(n_tendons_per_segment)
        self.total_tendons = np.sum(self.n_tendons_per_segment)

        # Handle scalar inputs by converting to arrays
        if np.isscalar(tendon_distances_mm):
            tendon_distances_mm = np.ones(self.n_segments) * tendon_distances_mm
        else:
            tendon_distances_mm = np.array(tendon_distances_mm)

        if angle_offsets_rad_ccw is None:
            angle_offsets_rad_ccw = np.zeros(self.n_segments)
        else:
            angle_offsets_rad_ccw = np.array(angle_offsets_rad_ccw)

        if np.isscalar(max_bending_angles_rad):
            max_bending_angles_rad = np.ones(self.n_segments) * max_bending_angles_rad
        else:
            max_bending_angles_rad = np.array(max_bending_angles_rad)

        # Create segment kinematics objects
        self.segments = []
        for i in range(self.n_segments):
            segment = NTendonSegmentKinematics(
                n=self.n_tendons_per_segment[i],
                tendon_distance_mm=tendon_distances_mm[i],
                angle_offset_rad_ccw=angle_offsets_rad_ccw[i],
                max_bending_angle_rad=max_bending_angles_rad[i],
            )
            self.segments.append(segment)

        # Store max clark coords magnitude for each segment
        self.max_clark_coords_magnitudes = np.array(
            [seg.max_clark_coords_magnitude for seg in self.segments]
        )

        # Initialize goal clark coordinates (2 coords per segment)
        self.goal_clark_coords = np.zeros(2 * self.n_segments)

        # Calculate tendon indices for each segment
        self._calculate_tendon_indices()

    def _calculate_tendon_indices(self):
        """Calculate start and end indices for each segment's tendons."""
        self.tendon_indices = []
        start_idx = 0
        for n_tendons in self.n_tendons_per_segment:
            end_idx = start_idx + n_tendons
            self.tendon_indices.append((start_idx, end_idx))
            start_idx = end_idx

    def clark_to_tendons_mm(self, clark_coords):
        """
        Convert Clark coordinates to tendon tensions (in tension control mode).

        Tension control logic:
        - Each segment's Clark coordinates are converted to tensions independently
        - No cross-segment compensation is applied

        This allows independent control of each segment without coupling effects.

        Args:
            clark_coords: Array of Clark coordinates [seg1_x, seg1_y, seg2_x, seg2_y, ...]

        Returns:
            Array of tendon tensions in mm (to be scaled by tension_scale)
        """
        tendon_tensions = np.zeros(self.total_tendons)

        # Convert each segment's Clark coords to tensions independently
        for seg_idx in range(self.n_segments):
            start_idx, end_idx = self.tendon_indices[seg_idx]
            clark_start = seg_idx * 2
            clark_end = clark_start + 2

            segment_clark = clark_coords[clark_start:clark_end]
            segment_tensions = self.segments[seg_idx].clark_to_tendons_mm(segment_clark)
            tendon_tensions[start_idx:end_idx] = segment_tensions

        return tendon_tensions

    def tendons_mm_to_clark(self, tendon_tensions_mm):
        """
        Convert tendon tensions to Clark coordinates (inverse transformation).

        Each segment is processed independently with no coupling compensation.

        Args:
            tendon_tensions_mm: Array of tendon tensions in mm

        Returns:
            Array of Clark coordinates
        """
        clark_coords = np.zeros(2 * self.n_segments)
        tendon_tensions_mm = np.array(tendon_tensions_mm, dtype=np.float64)

        # Process each segment independently
        for seg_idx in range(self.n_segments):
            start_idx, end_idx = self.tendon_indices[seg_idx]
            segment_tensions = tendon_tensions_mm[start_idx:end_idx].copy()

            # Convert to Clark coordinates
            clark_start = seg_idx * 2
            clark_end = clark_start + 2
            clark_coords[clark_start:clark_end] = self.segments[
                seg_idx
            ].tendons_mm_to_clark(segment_tensions)

        return clark_coords

    def set_goal_clark_coords(self, goal_clark_coords):
        """Set goal Clark coordinates and return corresponding tendon tensions."""
        self.goal_clark_coords = np.array(goal_clark_coords)
        return self.clark_to_tendons_mm(self.goal_clark_coords)

    def set_goal_clark_coords_to_current(self, current_tensions_mm):
        """Set goal Clark coordinates to match current tendon tensions."""
        self.goal_clark_coords = self.tendons_mm_to_clark(current_tensions_mm)
        return True

    def set_goal_clark_coords_to_home(self):
        """Set goal Clark coordinates to home position (all zeros)."""
        self.goal_clark_coords = np.zeros(2 * self.n_segments)
        return self.clark_to_tendons_mm(self.goal_clark_coords)

    def clark_coords_increment_to_tendon(self, goal_clark_coords_increment):
        """
        Increment goal Clark coordinates and apply constraints.

        Args:
            goal_clark_coords_increment: Incremental change in Clark coordinates

        Returns:
            Corresponding tendon tensions in mm
        """
        self.goal_clark_coords = self.goal_clark_coords + goal_clark_coords_increment

        # Apply constraints for each segment
        for seg_idx in range(self.n_segments):
            clark_start = seg_idx * 2
            clark_end = clark_start + 2
            segment_clark = self.goal_clark_coords[clark_start:clark_end]

            # Check magnitude constraint
            clark_norm = np.linalg.norm(segment_clark)
            max_magnitude = self.max_clark_coords_magnitudes[seg_idx]

            if clark_norm > max_magnitude:
                # Scale down to maximum allowed magnitude
                self.goal_clark_coords[clark_start:clark_end] = (
                    max_magnitude * segment_clark / clark_norm
                )

        return self.clark_to_tendons_mm(self.goal_clark_coords)

    def get_segment_clark_coords(self, clark_coords, segment_idx):
        """Get Clark coordinates for a specific segment."""
        clark_start = segment_idx * 2
        clark_end = clark_start + 2
        return clark_coords[clark_start:clark_end]

    def get_segment_tendon_tensions(self, tendon_tensions_mm, segment_idx):
        """Get tendon tensions for a specific segment."""
        start_idx, end_idx = self.tendon_indices[segment_idx]
        return tendon_tensions_mm[start_idx:end_idx]
