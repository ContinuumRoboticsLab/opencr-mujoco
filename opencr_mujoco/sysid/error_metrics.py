"""Error metric for system identification (standard pointwise RMSE)."""

from typing import Dict, Any, List, Optional
import numpy as np

from .base_metric import BaseMetric


class RMSEMetric(BaseMetric):
    """Standard (Euclidean, pointwise) RMSE for position comparison.

    RMS of the per-sample tip-position error NORMS — the one error number
    reported everywhere (printouts, plots, results.json), so the optimizer
    objective and the reported accuracy are literally the same quantity.

    Historical note: this used to be the per-coordinate RMSE, which is
    exactly tip-RMSE / sqrt(3) — same argmin, but a second confusingly
    smaller number. Objective values from runs before 2026-06-11 are 1.73x
    smaller than today's.
    """

    def __init__(
        self, name: str, weight: float = 1.0, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize RMSE metric.

        Args:
            name: Metric name
            weight: Metric weight
            config: Config with 'ignore_z' flag (default False)
        """
        super().__init__(name, weight, config)
        self.ignore_z = config.get("ignore_z", False) if config else False

    def compute(
        self, real_positions: np.ndarray, simulated_positions: np.ndarray
    ) -> float:
        """Compute pointwise RMSE between real and simulated positions (meters)."""
        assert (
            real_positions.shape == simulated_positions.shape
        ), f"Shape mismatch: {real_positions.shape} vs {simulated_positions.shape}"

        diff = real_positions - simulated_positions
        if self.ignore_z:
            diff = diff[:, :2]

        rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        return rmse

    def compute_per_axis(
        self, real_positions: np.ndarray, simulated_positions: np.ndarray
    ) -> Dict[str, float]:
        """Compute RMSE per axis (x, y, z)."""
        errors = {}
        for i, axis in enumerate(["x", "y", "z"]):
            diff = real_positions[:, i] - simulated_positions[:, i]
            errors[axis] = np.sqrt(np.mean(diff**2))
        return errors


class MetricCombiner:
    """Combines weighted metrics into a single objective value."""

    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics
        self.last_errors = {}
        self.last_combined = None

    def compute(
        self, real_positions: np.ndarray, simulated_positions: np.ndarray
    ) -> float:
        """Weighted, weight-normalized combination of the enabled metrics."""
        total_error = 0.0
        total_weight = 0.0

        for metric in self.metrics:
            if metric.weight > 0:
                error = metric.compute(real_positions, simulated_positions)
                total_error += error * metric.weight
                total_weight += metric.weight
                self.last_errors[metric.name] = error

        combined_error = total_error / total_weight if total_weight > 0 else 0.0
        self.last_combined = combined_error
        return combined_error

    def get_individual_errors(self) -> Dict[str, float]:
        return self.last_errors.copy()

    def format_errors(self) -> str:
        lines = [f"  {name}: {error:.6f} m" for name, error in self.last_errors.items()]
        if self.last_combined is not None:
            lines.append(f"  Combined: {self.last_combined:.6f} m")
        return "\n".join(lines)


def create_metrics_from_config(config: Dict[str, Any]) -> MetricCombiner:
    """Create a MetricCombiner from config. Only RMSE is supported.

    Args:
        config: metrics configuration dict, e.g. {"rmse": {"weight": 1.0}}

    Returns:
        MetricCombiner with an RMSE metric (defaults to weight 1.0 if unspecified).
    """
    metrics = []
    if "rmse" in config and config["rmse"].get("weight", 0.0) > 0:
        metrics.append(RMSEMetric("RMSE", config["rmse"]["weight"], config["rmse"]))
    if not metrics:
        metrics.append(RMSEMetric("RMSE", 1.0))
    return MetricCombiner(metrics)
