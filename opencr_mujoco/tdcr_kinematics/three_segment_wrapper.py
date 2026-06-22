"""Backward compatibility wrapper for ThreeTendonThreeSegmentTDCRKinematics."""

import numpy as np
from typing import Optional
from .multi_segment_tdcr_kinematics import MultiSegmentTDCRKinematics


class ThreeTendonThreeSegmentTDCRKinematics(MultiSegmentTDCRKinematics):
    """Wrapper class for backward compatibility with the old API.

    This class maintains the same interface as the old ThreeTendonThreeSegmentTDCRKinematics
    while using the new generalized MultiSegmentTDCRKinematics internally.
    """

    def __init__(
        self,
        tendon_distance_mm: float = 3.0,
        angle_offset_rad_ccw: Optional[np.ndarray] = None,
        max_bending_angle_rad: float = np.pi / 2,
    ):
        """Initialize 3-segment TDCR kinematics with backward compatible API.

        Args:
            tendon_distance_mm: Distance from backbone to tendons
            angle_offset_rad_ccw: Angular offsets for each segment
            max_bending_angle_rad: Maximum bending angle per segment
        """
        if angle_offset_rad_ccw is None:
            angle_offset_rad_ccw = np.array([0, np.pi / 6, np.pi / 3])

        # Call parent with the new API
        super().__init__(
            n_tendons_per_segment=[3, 3, 3],
            tendon_distances_mm=tendon_distance_mm,
            angle_offsets_rad_ccw=angle_offset_rad_ccw,
            max_bending_angles_rad=max_bending_angle_rad,
        )

        # Store for compatibility
        self.tendon_distance_mm = tendon_distance_mm
        self.angle_offset_rad_ccw = angle_offset_rad_ccw
        self.max_clark_coords_magnitude = max_bending_angle_rad * tendon_distance_mm

        # Create segment references for backward compatibility
        self.seg1 = self.segments[0]
        self.seg2 = self.segments[1]
        self.seg3 = self.segments[2]

    def get_goal_segment_clark_coords(self, segment_idx: int) -> np.ndarray:
        """Get the goal Clark coordinates for a specific segment.

        (Named distinctly from the parent's two-argument
        ``get_segment_clark_coords(clark_coords, segment_idx)``.)

        Args:
            segment_idx: Segment index (0, 1, or 2)

        Returns:
            2D Clark coordinates for the segment
        """
        return self.goal_clark_coords[segment_idx * 2 : (segment_idx + 1) * 2]

    def set_segment_clark_coords(
        self, segment_idx: int, clark_coords: np.ndarray
    ) -> np.ndarray:
        """Set Clark coordinates for a specific segment.

        Args:
            segment_idx: Segment index (0, 1, or 2)
            clark_coords: 2D Clark coordinates

        Returns:
            New 9D tendon lengths
        """
        self.goal_clark_coords[segment_idx * 2 : (segment_idx + 1) * 2] = clark_coords
        return self.clark_to_tendons_mm(self.goal_clark_coords)
