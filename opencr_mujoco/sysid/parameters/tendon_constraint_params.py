"""Tendon constraint factor parameter optimization."""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class TendonConstraintParameter(BaseParameter):
    """Parameter for optimizing tendon constraint factor.

    The tendon constraint factor controls tendon routing sites (0-1).
    - 0.0: No constraint sites
    - 1.0: Full constraint sites

    This is a global parameter (single value for entire TDCR).
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize tendon constraint parameter.

        Args:
            name: Parameter name
            config: Configuration with bounds
        """
        super().__init__(name, config)

        # Get bounds (default: 0.0 to 1.0)
        self.bounds = config.get("bounds", [0.0, 1.0])

        if len(self.bounds) != 2:
            raise ValueError("Bounds must be [min, max]")

        if self.bounds[0] < 0.0 or self.bounds[1] > 1.0:
            raise ValueError("Tendon constraint factor must be in range [0.0, 1.0]")

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
        return ["tendon_constraint_factor"]

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply tendon constraint factor to generation config.

        Args:
            values: Optimized tendon constraint factor value (single element)
            generation_config: TDCR generation configuration

        Returns:
            Modified generation config
        """
        # The generator reads this from actuator_properties (a 'tendon_config'
        # section is read by nothing — writing there was a silent no-op).
        if "actuator_properties" not in generation_config:
            generation_config["actuator_properties"] = {}

        # Apply the tendon constraint factor
        generation_config["actuator_properties"]["tendon_constraint_factor"] = float(
            values[0]
        )

        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        """Format values for display.

        Args:
            values: Parameter values

        Returns:
            Formatted string
        """
        return f"Tendon Constraint Factor: {values[0]:.4f}"
