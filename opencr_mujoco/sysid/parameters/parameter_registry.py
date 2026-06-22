"""Parameter registry for managing all optimizable parameters."""

from typing import Dict, Type, Any, List, Optional, Tuple
import numpy as np
from ..base_parameter import BaseParameter
from .pretension_params import PretensionParameter
from .tendon_params import TendonStiffnessParameter
from .material_params import MaterialParameter
from .geometry_params import TendonOffsetParameter
from .tendon_slack_params import TendonSlackParameter
from .tendon_constraint_params import TendonConstraintParameter
from .tendon_distance_params import TendonDistanceParameter
from .friction_params import FrictionParameter
from .joint_deadband_params import JointDeadbandParameter
from .tendon_friction_params import TendonFrictionParameter

# Global parameter type registry
PARAMETER_TYPES: Dict[str, Type[BaseParameter]] = {
    "pretension": PretensionParameter,
    "tendon_kp": TendonStiffnessParameter,
    "tendon_stiffness": TendonStiffnessParameter,  # Alias
    "material": MaterialParameter,
    "tendon_offsets": TendonOffsetParameter,
    "geometry": TendonOffsetParameter,  # Alias
    "tendon_slack": TendonSlackParameter,
    "tendon_constraint_factor": TendonConstraintParameter,
    "tendon_distance_mm": TendonDistanceParameter,
    "friction": FrictionParameter,
    "joint_deadband": JointDeadbandParameter,
    "tendon_friction": TendonFrictionParameter,
}


def register_parameter(name: str, parameter_class: Type[BaseParameter]):
    """Register a new parameter type.

    Args:
        name: Name to register the parameter under
        parameter_class: Parameter class (must inherit from BaseParameter)
    """
    if not issubclass(parameter_class, BaseParameter):
        raise ValueError(f"Parameter class must inherit from BaseParameter")
    PARAMETER_TYPES[name] = parameter_class


class ParameterRegistry:
    """Manages all parameters for system identification."""

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """Initialize parameter registry.

        Args:
            config: Parameters configuration dict with enabled parameters
            verbose: If True, print each loaded parameter (gated off by
                default — the parallel optimizer builds a registry per worker)
        """
        self.config = config
        self.verbose = verbose
        self.parameters: List[BaseParameter] = []
        self.parameter_map: Dict[str, BaseParameter] = {}
        self._total_dims = 0
        self._dim_to_param: List[BaseParameter] = []
        self._dim_to_local_idx: List[int] = []

        # Load enabled parameters
        self._load_parameters()

    def _load_parameters(self):
        """Load and initialize enabled parameters from config."""
        params_config = self.config.get("parameters", {})

        for param_name, param_config in params_config.items():
            if not param_config.get("enabled", False):
                continue

            # Find parameter type
            param_type = None
            for type_name, type_class in PARAMETER_TYPES.items():
                if param_name.startswith(type_name) or param_name == type_name:
                    param_type = type_class
                    break

            if param_type is None:
                print(f"Warning: Unknown parameter type '{param_name}', skipping")
                continue

            # Create parameter instance
            param = param_type(param_name, param_config)
            self.add_parameter(param)

            if self.verbose:
                print(f"Loaded parameter: {param_name} ({param.num_dims} dimensions)")

    def add_parameter(self, parameter: BaseParameter):
        """Add a parameter to the registry.

        Args:
            parameter: Parameter instance to add
        """
        if not parameter.enabled:
            return

        self.parameters.append(parameter)
        self.parameter_map[parameter.name] = parameter

        # Track dimension mapping
        param_dims = len(parameter.get_bounds())
        for local_idx in range(param_dims):
            self._dim_to_param.append(parameter)
            self._dim_to_local_idx.append(local_idx)

        self._total_dims += param_dims

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get combined bounds for all parameters.

        Returns:
            List of (min, max) tuples for all dimensions
        """
        bounds = []
        for param in self.parameters:
            bounds.extend(param.get_bounds())
        return bounds

    def get_dimension_names(self) -> List[str]:
        """Get names for all optimization dimensions.

        Returns:
            List of dimension names
        """
        names = []
        for param in self.parameters:
            names.extend(param.get_dimension_names())
        return names

    def get_initial_values(self) -> Optional[np.ndarray]:
        """Get initial values for all parameters.

        Returns:
            Combined initial values or None
        """
        has_initial = False
        values = []

        for param in self.parameters:
            param_initial = param.get_initial_values()
            if param_initial is not None:
                has_initial = True
                values.extend(param_initial.tolist())
            else:
                # Use middle of bounds as default
                param_bounds = param.get_bounds()
                for min_val, max_val in param_bounds:
                    values.append((min_val + max_val) / 2)

        return np.array(values) if has_initial else None

    def split_values(self, values: np.ndarray) -> Dict[str, np.ndarray]:
        """Split combined values into per-parameter arrays.

        Args:
            values: Combined parameter values

        Returns:
            Dict mapping parameter names to their values
        """
        result = {}
        idx = 0

        for param in self.parameters:
            param_dims = len(param.get_bounds())
            param_values = values[idx : idx + param_dims]
            result[param.name] = param_values
            param.set_current_value(param_values)
            idx += param_dims

        return result

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply all parameter values to generation config.

        Args:
            values: Combined parameter values
            generation_config: Base TDCR generation configuration

        Returns:
            Modified generation config
        """
        # Split values by parameter
        param_values = self.split_values(values)

        # Apply each parameter's modifications
        for param in self.parameters:
            generation_config = param.apply_to_config(
                param_values[param.name], generation_config
            )

        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        """Format all parameter values for display.

        Args:
            values: Combined parameter values

        Returns:
            Formatted string representation
        """
        param_values = self.split_values(values)
        lines = []

        for param in self.parameters:
            lines.append(param.format_values(param_values[param.name]))

        return "\n".join(lines)

    def get_total_dimensions(self) -> int:
        """Get total number of optimization dimensions.

        Returns:
            Total dimension count
        """
        return self._total_dims

    def get_parameter_names(self) -> List[str]:
        """Get list of enabled parameter names.

        Returns:
            List of parameter names
        """
        return [p.name for p in self.parameters]
