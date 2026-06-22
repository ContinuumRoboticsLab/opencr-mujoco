"""Base kinematics class for N-tendon single segment TDCR.

This module implements the Clark transformation for converting between
tendon lengths and Clark coordinates for a single segment with N tendons.
"""

import numpy as np


class NTendonSegmentKinematics:
    """Kinematics for a single TDCR segment with N tendons.

    The Clark transformation converts between tendon lengths and a 2D
    representation of the segment's bending state (Clark coordinates).

    Attributes:
        tendon_distance_mm: Distance from backbone to tendons in mm
        clark_transform_mat: 2xN transformation matrix
        clark_transform_inv_mat: Nx2 inverse transformation matrix
        max_clark_coords_magnitude: Maximum allowed bending (rad * mm)
        goal_clark_coords: Current goal in Clark coordinates
    """

    def __init__(
        self,
        n: int,
        tendon_distance_mm: float,
        angle_offset_rad_ccw: float = 0.0,
        max_bending_angle_rad: float = np.pi / 2,
    ):
        """Initialize N-tendon segment kinematics.

        Args:
            n: Number of tendons
            tendon_distance_mm: Distance from backbone to tendons
            angle_offset_rad_ccw: Angular offset of first tendon (rad)
            max_bending_angle_rad: Maximum bending angle (rad)
        """
        self.n = n
        self.tendon_distance_mm = tendon_distance_mm
        self.angle_offset_rad_ccw = angle_offset_rad_ccw
        self.max_bending_angle_rad = max_bending_angle_rad

        # Build Clark transformation matrices
        self.clark_transform_mat = np.zeros([2, n])
        for i in range(n):
            angle = angle_offset_rad_ccw + 2 * np.pi / n * i
            self.clark_transform_mat[0, i] = np.cos(angle)
            self.clark_transform_mat[1, i] = np.sin(angle)
        self.clark_transform_mat = 2 / n * self.clark_transform_mat
        self.clark_transform_inv_mat = n / 2 * self.clark_transform_mat.T

        self.max_clark_coords_magnitude = max_bending_angle_rad * tendon_distance_mm
        self.goal_clark_coords = np.zeros((2,))

    def tendons_mm_to_clark(self, tendon_lengths_mm: np.ndarray) -> np.ndarray:
        """Convert tendon lengths to Clark coordinates.

        Args:
            tendon_lengths_mm: Array of N tendon lengths in mm

        Returns:
            2D Clark coordinates [cx, cy]
        """
        return self.clark_transform_mat @ tendon_lengths_mm

    def clark_to_tendons_mm(self, clark_coords: np.ndarray) -> np.ndarray:
        """Convert Clark coordinates to tendon lengths.

        Args:
            clark_coords: 2D Clark coordinates [cx, cy]

        Returns:
            Array of N tendon lengths in mm
        """
        return self.clark_transform_inv_mat @ clark_coords

    def set_goal_clark_coords(self, goal_clark_coords: np.ndarray) -> np.ndarray:
        """Set goal Clark coordinates.

        Args:
            goal_clark_coords: Desired Clark coordinates [cx, cy]

        Returns:
            Corresponding tendon lengths in mm
        """
        self.goal_clark_coords = goal_clark_coords
        return self.clark_to_tendons_mm(self.goal_clark_coords)

    def set_goal_clark_coords_to_current(self, current_tendons_mm: np.ndarray) -> bool:
        """Set goal to match current tendon lengths.

        Args:
            current_tendons_mm: Current tendon lengths in mm

        Returns:
            True on success
        """
        self.goal_clark_coords = self.tendons_mm_to_clark(current_tendons_mm)
        return True

    def set_goal_clark_coords_to_home(self) -> np.ndarray:
        """Set goal to home position (straight).

        Returns:
            Tendon lengths for home position
        """
        self.goal_clark_coords = np.array([0.0, 0.0])
        return self.clark_to_tendons_mm(self.goal_clark_coords)

    def clark_coords_increment_to_tendon(
        self, goal_clark_coords_increment: np.ndarray
    ) -> np.ndarray:
        """Increment goal Clark coordinates with saturation.

        Args:
            goal_clark_coords_increment: Desired increment [dcx, dcy]

        Returns:
            New tendon lengths in mm
        """
        self.goal_clark_coords = self.goal_clark_coords + goal_clark_coords_increment
        goal_clark_coords_norm = np.linalg.norm(self.goal_clark_coords)

        # Saturate if exceeding maximum bending
        if goal_clark_coords_norm > self.max_clark_coords_magnitude:
            self.goal_clark_coords = (
                self.max_clark_coords_magnitude
                * self.goal_clark_coords
                / goal_clark_coords_norm
            )

        return self.clark_to_tendons_mm(self.goal_clark_coords)
