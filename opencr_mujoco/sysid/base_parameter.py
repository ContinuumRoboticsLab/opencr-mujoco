"""Base parameter interface for system identification.

Provides abstract base class for all optimizable parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class BaseParameter(ABC):
    """Abstract base class for optimizable parameters.

    All parameter types must inherit from this class and implement
    the required methods for bounds definition and config application.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize parameter.

        Args:
            name: Parameter name for identification
            config: Parameter-specific configuration
        """
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
        self.mode = config.get("mode", "global")
        self._current_value = None

    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds for this parameter.

        Returns:
            List of (min, max) tuples for each optimization dimension
        """
        pass

    @abstractmethod
    def get_dimension_names(self) -> List[str]:
        """Get names for each optimization dimension.

        Returns:
            List of descriptive names for each dimension
        """
        pass

    @abstractmethod
    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply optimized values to generation config.

        Args:
            values: Optimized parameter values
            generation_config: TDCR generation configuration dict

        Returns:
            Modified generation config
        """
        pass

    def get_initial_values(self) -> Optional[np.ndarray]:
        """Get initial values for optimization.

        Returns:
            Initial parameter values or None to use optimizer defaults
        """
        if "initial_values" in self.config:
            return np.array(self.config["initial_values"])
        return None

    def validate_values(self, values: np.ndarray) -> bool:
        """Validate parameter values.

        Args:
            values: Parameter values to validate

        Returns:
            True if values are valid
        """
        bounds = self.get_bounds()
        if len(values) != len(bounds):
            return False

        for val, (min_val, max_val) in zip(values, bounds):
            if val < min_val or val > max_val:
                return False

        return True

    def set_current_value(self, values: np.ndarray):
        """Store current parameter values.

        Args:
            values: Current parameter values
        """
        self._current_value = values.copy()

    def format_values(self, values: np.ndarray) -> str:
        """Format values for display.

        Args:
            values: Parameter values

        Returns:
            Formatted string representation
        """
        names = self.get_dimension_names()
        parts = []
        for name, val in zip(names, values):
            parts.append(f"{name}={val:.6f}")
        return f"{self.name}: {', '.join(parts)}"
