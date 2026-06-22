"""Tendon stiffness parameter optimization."""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class TendonStiffnessParameter(BaseParameter):
    """Parameter for optimizing tendon actuator stiffness (kp values).

    Supports different modes:
    - 'global': Single kp value for all tendons
    - 'per_segment': One kp per segment (all tendons in segment share same kp)
    - 'per_tendon': One kp per individual tendon
    - 'scale_factor': Multiplicative factor on base kp
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize tendon stiffness parameter.

        Args:
            name: Parameter name
            config: Configuration with mode and bounds
        """
        super().__init__(name, config)
        self.num_segments = config.get("num_segments", 3)
        self.tendons_per_segment = config.get("tendons_per_segment", 3)
        self.total_tendons = self.num_segments * self.tendons_per_segment

        # Determine optimization dimensions
        if self.mode == "global":
            self.num_dims = 1
        elif self.mode == "per_segment":
            self.num_dims = self.num_segments
        elif self.mode == "per_tendon":
            self.num_dims = self.total_tendons
        elif self.mode == "scale_factor":
            self.num_dims = 1  # Single scaling factor
        else:
            raise ValueError(f"Unknown tendon stiffness mode: {self.mode}")

        # Get bounds
        if self.mode == "scale_factor":
            # Scale factor bounds (multiplicative)
            bounds = config.get("bounds", [0.5, 2.0])
        else:
            # Direct kp value bounds
            bounds = config.get("bounds", [5000, 20000])

        if isinstance(bounds[0], (list, tuple)):
            self.bounds = bounds
        else:
            self.bounds = [bounds] * self.num_dims

        # Base kp value for scale_factor mode
        self.base_kp = config.get("base_kp", 10000)

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds.

        Returns:
            List of (min, max) tuples
        """
        return [(b[0], b[1]) for b in self.bounds]

    def get_dimension_names(self) -> List[str]:
        """Get dimension names.

        Returns:
            List of dimension names
        """
        if self.mode == "global":
            return ["tendon_kp"]
        elif self.mode == "per_segment":
            return [f"tendon_kp_seg{i+1}" for i in range(self.num_segments)]
        elif self.mode == "per_tendon":
            names = []
            for seg in range(self.num_segments):
                for ten in range(self.tendons_per_segment):
                    names.append(f"tendon_kp_s{seg+1}t{ten+1}")
            return names
        else:  # scale_factor
            return ["tendon_kp_scale"]

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply stiffness values to generation config.

        Args:
            values: Optimized stiffness values or scale factors
            generation_config: TDCR generation configuration

        Returns:
            Modified generation config
        """
        if "actuator_properties" not in generation_config:
            generation_config["actuator_properties"] = {}

        if self.mode == "global":
            # Single kp value for all
            generation_config["actuator_properties"]["tendon_kp"] = float(values[0])

        elif self.mode == "per_segment":
            # Per-segment kp values (repeated for each tendon in segment)
            kp_values = []
            for seg_idx, kp in enumerate(values):
                for _ in range(self.tendons_per_segment):
                    kp_values.append(float(kp))
            generation_config["actuator_properties"]["tendon_kp_array"] = kp_values

        elif self.mode == "per_tendon":
            # Per-tendon kp values (one value per individual tendon)
            kp_values = [float(kp) for kp in values]
            generation_config["actuator_properties"]["tendon_kp_array"] = kp_values

        else:  # scale_factor
            # Apply scale factor to base kp
            scaled_kp = self.base_kp * values[0]
            generation_config["actuator_properties"]["tendon_kp"] = float(scaled_kp)
            generation_config["actuator_properties"]["tendon_kp_scale_factor"] = float(
                values[0]
            )

        # Enable inverse length scaling if configured
        if "kp_scaling" in self.config:
            generation_config["actuator_properties"]["tendon_kp_scaling"] = self.config[
                "kp_scaling"
            ]

        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        """Format values for display.

        Args:
            values: Parameter values

        Returns:
            Formatted string
        """
        if self.mode == "scale_factor":
            return f"Tendon kp scale: {values[0]:.3f}x"
        elif self.mode == "global":
            return f"Tendon kp: {values[0]:.0f} N/m"
        elif self.mode == "per_segment":
            parts = [f"S{i+1}:{v:.0f}" for i, v in enumerate(values)]
            return f"Tendon kp: {' '.join(parts)} N/m"
        else:  # per_tendon
            parts = []
            for seg in range(self.num_segments):
                seg_values = values[
                    seg
                    * self.tendons_per_segment : (seg + 1)
                    * self.tendons_per_segment
                ]
                seg_str = (
                    f"S{seg+1}:[" + ",".join([f"{v:.0f}" for v in seg_values]) + "]"
                )
                parts.append(seg_str)
            return f"Tendon kp: {' '.join(parts)} N/m"
