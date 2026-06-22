"""Pretension parameter optimization for TDCR tendons."""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class PretensionParameter(BaseParameter):
    """Parameter for optimizing tendon pretension values.

    Supports different modes:
    - 'global': Single pretension value for all tendons
    - 'per_segment': One value per segment (3 values)
    - 'per_tendon': Individual value for each tendon (9 values)
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize pretension parameter.

        Args:
            name: Parameter name
            config: Configuration with mode and bounds
        """
        super().__init__(name, config)
        self.num_segments = config.get("num_segments", 3)
        self.tendons_per_segment = config.get("tendons_per_segment", 3)
        self.total_tendons = self.num_segments * self.tendons_per_segment

        # Determine number of optimization dimensions based on mode
        if self.mode == "global":
            self.num_dims = 1
        elif self.mode == "per_segment":
            self.num_dims = self.num_segments
        elif self.mode == "per_tendon":
            self.num_dims = self.total_tendons
        else:
            raise ValueError(f"Unknown pretension mode: {self.mode}")

        # Get bounds (pretension is a FORCE in Newtons; the generator turns it
        # into a ctrl offset via rest_length - pretension/kp)
        bounds = config.get("bounds", [0.0, 8.0])
        if isinstance(bounds[0], (list, tuple)):
            # Per-dimension bounds
            self.bounds = bounds
        else:
            # Same bounds for all dimensions
            self.bounds = [bounds] * self.num_dims

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds for pretension values.

        Returns:
            List of (min, max) tuples
        """
        return [(b[0], b[1]) for b in self.bounds]

    def get_dimension_names(self) -> List[str]:
        """Get descriptive names for each dimension.

        Returns:
            List of dimension names
        """
        if self.mode == "global":
            return ["pretension_all"]
        elif self.mode == "per_segment":
            return [f"pretension_seg{i+1}" for i in range(self.num_segments)]
        else:  # per_tendon
            names = []
            for seg in range(self.num_segments):
                for ten in range(self.tendons_per_segment):
                    names.append(f"pretension_s{seg+1}_t{ten+1}")
            return names

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply pretension values to generation config.

        Args:
            values: Optimized pretension values
            generation_config: TDCR generation configuration

        Returns:
            Modified generation config
        """
        # Expand values based on mode
        if self.mode == "global":
            pretension_values = np.full(self.total_tendons, values[0])
        elif self.mode == "per_segment":
            pretension_values = np.repeat(values, self.tendons_per_segment)
        else:  # per_tendon
            pretension_values = values

        # Apply to actuator properties
        if "actuator_properties" not in generation_config:
            generation_config["actuator_properties"] = {}

        # Store as per-tendon pretension array
        generation_config["actuator_properties"][
            "tendon_pretension"
        ] = pretension_values.tolist()

        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        """Format pretension values for display.

        Args:
            values: Parameter values

        Returns:
            Formatted string
        """
        if self.mode == "global":
            return f"Pretension: {values[0]:.4f} N (all)"
        elif self.mode == "per_segment":
            parts = [f"S{i+1}:{v:.4f}N" for i, v in enumerate(values)]
            return f"Pretension: {' '.join(parts)}"
        else:
            # Compact display for per-tendon
            return (
                f"Pretension: {values.min():.4f} to {values.max():.4f} N (per-tendon)"
            )
