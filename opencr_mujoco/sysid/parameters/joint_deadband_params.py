"""Joint deadband (springlength) parameter for TDCR system identification.

Models hysteresis near the origin by adding a springlength deadband to backbone joints.
Within the deadband range, the joint spring restoring force is zero, so the robot
doesn't fully return to straight after bending.

MuJoCo: springlength="-db db" means spring force = 0 when joint pos in [-db, db].
Outside the deadband, normal spring restoring force applies.

Supports per-segment deadband values (one per segment).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class JointDeadbandParameter(BaseParameter):
    """Parameter for optimizing joint springlength deadband.

    Per-segment: one deadband half-width (radians) per segment.
    Bounds should be positive, e.g., [0.0, 0.05] (0 to ~3 degrees).
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.num_segments = config.get("num_segments", 3)
        self.num_dims = self.num_segments

        bounds = config.get("bounds", [0.0, 0.05])
        self.bounds = [bounds] * self.num_segments

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(b[0], b[1]) for b in self.bounds]

    def get_dimension_names(self) -> List[str]:
        return [f"joint_deadband_seg{i+1}" for i in range(self.num_segments)]

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        generation_config["joint_deadband"] = [float(v) for v in values]
        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        parts = []
        for i, v in enumerate(values):
            deg = np.degrees(v)
            parts.append(f"S{i+1}:{v:.4f}rad ({deg:.2f}deg)")
        return f"Joint deadband: {' '.join(parts)}"
