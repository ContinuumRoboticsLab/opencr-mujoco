"""Material property parameter optimization."""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class MaterialParameter(BaseParameter):
    """Parameter for optimizing material properties.

    Can optimize:
    - Young's modulus
    - Damping ratio
    - Density
    - Poisson ratio
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize material parameter.

        Args:
            name: Parameter name
            config: Configuration with properties to optimize
        """
        super().__init__(name, config)

        # Determine which properties to optimize
        self.optimize_properties = []
        self.property_bounds = []

        if config.get("optimize_youngs_modulus", True):
            self.optimize_properties.append("youngs_modulus")
            bounds = config.get("youngs_modulus_bounds", [100e9, 300e9])
            self.property_bounds.append(bounds)

        if config.get("optimize_damping_ratio", True):
            self.optimize_properties.append("damping_ratio")
            bounds = config.get("damping_ratio_bounds", [0.0001, 0.01])
            self.property_bounds.append(bounds)

        if config.get("optimize_density", False):
            self.optimize_properties.append("density")
            bounds = config.get("density_bounds", [1000, 10000])
            self.property_bounds.append(bounds)

        if config.get("optimize_poisson_ratio", False):
            self.optimize_properties.append("poisson_ratio")
            bounds = config.get("poisson_ratio_bounds", [0.2, 0.49])
            self.property_bounds.append(bounds)

        self.num_dims = len(self.optimize_properties)

        if self.num_dims == 0:
            raise ValueError("No material properties selected for optimization")

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds.

        Returns:
            List of (min, max) tuples
        """
        return [(b[0], b[1]) for b in self.property_bounds]

    def get_dimension_names(self) -> List[str]:
        """Get dimension names.

        Returns:
            List of dimension names
        """
        names = []
        for prop in self.optimize_properties:
            if prop == "youngs_modulus":
                names.append("E_modulus")
            elif prop == "damping_ratio":
                names.append("damping_ratio")
            elif prop == "density":
                names.append("density")
            elif prop == "poisson_ratio":
                names.append("poisson_ratio")
        return names

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply material properties to generation config.

        Args:
            values: Optimized material property values
            generation_config: TDCR generation configuration

        Returns:
            Modified generation config
        """
        # Ensure material properties section exists
        if "material_properties" not in generation_config:
            generation_config["material_properties"] = {}

        # Apply each optimized property
        for prop, val in zip(self.optimize_properties, values):
            if prop == "youngs_modulus":
                generation_config["material_properties"]["youngs_modulus"] = float(val)
            elif prop == "damping_ratio":
                generation_config["material_properties"]["damping_ratio"] = float(val)
                # Also update damping if it exists
                if "damping" in generation_config["material_properties"]:
                    generation_config["material_properties"]["damping"] = float(val)
            elif prop == "density":
                generation_config["material_properties"]["density"] = float(val)
            elif prop == "poisson_ratio":
                generation_config["material_properties"]["poisson_ratio"] = float(val)

        # Ensure material mode is used
        generation_config["joint_config_mode"] = "material"

        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        """Format values for display.

        Args:
            values: Parameter values

        Returns:
            Formatted string
        """
        parts = []
        for prop, val in zip(self.optimize_properties, values):
            if prop == "youngs_modulus":
                parts.append(f"E={val/1e9:.1f}GPa")
            elif prop == "damping_ratio":
                parts.append(f"ζ={val:.5f}")
            elif prop == "density":
                parts.append(f"ρ={val:.0f}kg/m³")
            elif prop == "poisson_ratio":
                parts.append(f"ν={val:.3f}")
        return f"Material: {', '.join(parts)}"
