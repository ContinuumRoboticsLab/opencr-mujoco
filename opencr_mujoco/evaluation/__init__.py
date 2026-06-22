"""
Evaluation module for comparing simulation results against SoroSim reference data.

This module provides tools for:
- Loading SoroSim reference data (static CSV + 13-column dynamics)
- Running systematic evaluations across parameter (N, sim_hz) sweeps
- Computing error metrics between simulations and references
- Visualizing results with publication-quality plots
"""

from .trajectory_evaluator import TrajectoryEvaluator
from .metrics import compute_tip_error, compute_shape_error, compute_trajectory_error
from .reference_data_loader import ReferenceDataLoader, validate_frame_conversion


def __getattr__(name):
    """Import plotting helpers only when callers actually request them."""
    if name == "EvaluationVisualizer":
        from .visualization import EvaluationVisualizer

        return EvaluationVisualizer
    if name == "PaperVisualizer":
        from .paper_visualization import PaperVisualizer

        return PaperVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrajectoryEvaluator",
    "ReferenceDataLoader",
    "validate_frame_conversion",
    "EvaluationVisualizer",
    "PaperVisualizer",
    "compute_tip_error",
    "compute_shape_error",
    "compute_trajectory_error",
]
