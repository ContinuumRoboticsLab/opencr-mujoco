"""
Evaluation metrics for comparing simulations against reference data.
"""

import numpy as np
from typing import List, Optional, Tuple, Union


def compute_tip_error(
    sim_tip_pos: np.ndarray, ref_tip_pos: Union[np.ndarray, Tuple[float, float, float]]
) -> float:
    """
    Compute Euclidean distance between simulated and reference tip positions.

    Args:
        sim_tip_pos: Simulated tip position [x, y, z]
        ref_tip_pos: Reference tip position [x, y, z]

    Returns:
        Euclidean distance in meters
    """
    return np.linalg.norm(np.array(sim_tip_pos) - np.array(ref_tip_pos))


def compute_shape_error(
    sim_positions: List[np.ndarray],
    ref_positions: List[Tuple[float, float, float]],
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute weighted shape error between simulated and reference link positions.

    Args:
        sim_positions: List of simulated link positions
        ref_positions: List of reference link positions
        weights: Optional weights for each link (default: uniform)

    Returns:
        Weighted mean shape error in meters
    """
    if len(sim_positions) != len(ref_positions):
        raise ValueError(
            f"Position lists must have same length: "
            f"sim={len(sim_positions)}, ref={len(ref_positions)}"
        )

    if weights is None:
        weights = np.ones(len(sim_positions)) / len(sim_positions)
    else:
        weights = np.array(weights) / np.sum(weights)

    errors = []
    for sim_pos, ref_pos in zip(sim_positions, ref_positions):
        error = np.linalg.norm(np.array(sim_pos) - np.array(ref_pos))
        errors.append(error)

    return np.sum(weights * np.array(errors))


def compute_trajectory_error(
    sim_trajectory: np.ndarray, ref_trajectory: np.ndarray, metric: str = "mse"
) -> float:
    """
    Compute error between simulated and reference trajectories.

    Args:
        sim_trajectory: Simulated trajectory (time x dims)
        ref_trajectory: Reference trajectory (time x dims)
        metric: Error metric ('mse', 'rmse', 'mae', 'max')

    Returns:
        Error value
    """
    if sim_trajectory.shape != ref_trajectory.shape:
        raise ValueError(
            f"Trajectory shapes must match: "
            f"sim={sim_trajectory.shape}, ref={ref_trajectory.shape}"
        )

    diff = sim_trajectory - ref_trajectory

    if metric == "mse":
        # Calculate squared Euclidean distance per row (time step), then average
        row_errors = np.sum(diff**2, axis=1)
        return np.mean(row_errors)
    elif metric == "rmse":
        # Calculate squared Euclidean distance per row, average, then sqrt
        row_errors = np.sum(diff**2, axis=1)
        return np.sqrt(np.mean(row_errors))
    elif metric == "mae":
        # Calculate sum of absolute errors per row, then average
        row_errors = np.sum(np.abs(diff), axis=1)
        return np.mean(row_errors)
    elif metric == "max":
        # Maximum error across all dimensions and time
        return np.max(np.abs(diff))
    else:
        raise ValueError(f"Unknown metric: {metric}")
