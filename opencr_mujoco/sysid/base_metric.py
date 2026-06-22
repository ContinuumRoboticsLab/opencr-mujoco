"""Base metric interface for error computation.

Provides abstract base class for all error metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseMetric(ABC):
    """Abstract base class for error metrics.

    All metric types must inherit from this class and implement
    the required computation methods.
    """

    def __init__(
        self, name: str, weight: float = 1.0, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize metric.

        Args:
            name: Metric name for identification
            weight: Weight for combining multiple metrics
            config: Optional metric-specific configuration
        """
        self.name = name
        self.weight = weight
        self.config = config or {}

    @abstractmethod
    def compute(
        self, real_positions: np.ndarray, simulated_positions: np.ndarray
    ) -> float:
        """Compute error metric between real and simulated positions.

        Args:
            real_positions: Real marker positions (N x 3)
            simulated_positions: Simulated marker positions (N x 3)

        Returns:
            Error value (lower is better)
        """
        pass

    def get_weight(self) -> float:
        """Get metric weight for combining multiple metrics.

        Returns:
            Metric weight
        """
        return self.weight

    def set_weight(self, weight: float):
        """Set metric weight.

        Args:
            weight: New weight value
        """
        self.weight = weight
