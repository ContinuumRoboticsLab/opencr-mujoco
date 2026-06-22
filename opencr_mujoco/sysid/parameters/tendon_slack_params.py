"""Tendon slack scaling parameter for TDCR system identification."""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class TendonSlackParameter(BaseParameter):
    """Parameter for optimizing tendon slack scaling factor.

    This scales the effective displacement: actual_displacement = slack_factor * commanded_displacement
    Used to account for slack, compliance, or mechanical losses in the tendon transmission.

    Supports:
    - 'global': Single scaling factor for all tendons
    - 'per_segment': One scaling factor per segment
    - 'per_tendon': Individual scaling factor for each tendon
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize tendon slack parameter.

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
            raise ValueError(f"Unknown tendon slack mode: {self.mode}")

        # Get bounds (default 0.6 to 1.0)
        bounds = config.get("bounds", [0.6, 1.0])
        if isinstance(bounds[0], (list, tuple)):
            # Per-dimension bounds
            self.bounds = bounds
        else:
            # Same bounds for all dimensions
            self.bounds = [bounds] * self.num_dims

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds.

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
            return ["tendon_slack_all"]
        elif self.mode == "per_segment":
            return [f"tendon_slack_seg{i+1}" for i in range(self.num_segments)]
        else:  # per_tendon
            names = []
            for seg in range(self.num_segments):
                for ten in range(self.tendons_per_segment):
                    names.append(f"tendon_slack_s{seg+1}_t{ten+1}")
            return names

    def get_initial_values(self) -> np.ndarray:
        """Get initial values for optimization.

        Returns:
            Array of initial values (starts at 1.0 = no scaling)
        """
        return np.ones(self.num_dims)

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply tendon slack scaling to generation config.

        Note: This parameter is applied during simulation, not during model generation.
        We store it in a custom field that the simulator will read.

        Args:
            values: Optimized slack scaling factor(s)
            generation_config: TDCR generation configuration

        Returns:
            Modified generation config
        """
        # Expand values based on mode
        if self.mode == "global":
            slack_values = np.full(self.total_tendons, values[0])
        elif self.mode == "per_segment":
            slack_values = np.repeat(values, self.tendons_per_segment)
        else:  # per_tendon
            slack_values = values

        # Store in a custom field for the simulator to use
        if "sysid_params" not in generation_config:
            generation_config["sysid_params"] = {}

        generation_config["sysid_params"][
            "tendon_slack_scaling"
        ] = slack_values.tolist()

        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        """Format slack scaling values for display.

        Args:
            values: Parameter values

        Returns:
            Formatted string
        """
        if self.mode == "global":
            return f"Tendon slack: {values[0]:.4f} (all)"
        elif self.mode == "per_segment":
            parts = [f"S{i+1}:{v:.4f}" for i, v in enumerate(values)]
            return f"Tendon slack: {' '.join(parts)}"
        else:
            return (
                f"Tendon slack: {values.min():.4f} to {values.max():.4f} (per-tendon)"
            )
