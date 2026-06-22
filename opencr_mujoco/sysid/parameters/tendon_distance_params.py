"""Tendon distance parameter optimization."""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class TendonDistanceParameter(BaseParameter):
    """Parameter for optimizing tendon distance from backbone.

    The tendon distance (in mm) controls how far tendons are from the central backbone.
    This affects the torque arm and bending behavior of the TDCR.

    This is a global parameter (single value for entire TDCR).
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize tendon distance parameter.

        Args:
            name: Parameter name
            config: Configuration with bounds
        """
        super().__init__(name, config)

        # Get bounds (default: 2.0 to 10.0 mm)
        self.bounds = config.get("bounds", [2.0, 10.0])

        if len(self.bounds) != 2:
            raise ValueError("Bounds must be [min, max]")

        if self.bounds[0] <= 0.0:
            raise ValueError("Tendon distance must be positive")

        self.num_dims = 1

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds.

        Returns:
            List with single (min, max) tuple
        """
        return [(self.bounds[0], self.bounds[1])]

    def get_dimension_names(self) -> List[str]:
        """Get dimension names.

        Returns:
            List with single dimension name
        """
        return ["tendon_distance_mm"]

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply tendon distance to generation config.

        Args:
            values: Optimized tendon distance value (single element, in mm)
            generation_config: TDCR generation configuration

        Returns:
            Modified generation config
        """
        # The generator reads per-segment distance_to_backbone in METERS from
        # actuation_details.segments (a 'tendon_config' section is read by
        # nothing — writing there was a silent no-op).
        distance_m = float(values[0]) * 1e-3
        segments = generation_config.get("actuation_details", {}).get("segments")
        if not segments:
            raise ValueError(
                "tendon_distance parameter requires the generation config to "
                "define actuation_details.segments"
            )
        for segment in segments:
            segment["distance_to_backbone"] = distance_m

        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        """Format values for display.

        Args:
            values: Parameter values

        Returns:
            Formatted string
        """
        return f"Tendon Distance: {values[0]:.3f} mm"
