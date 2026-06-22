"""Tendon friction (hysteresis) parameter for system identification.

Models direction-dependent friction in the tendon transmission. When a tendon
changes direction (pulling → releasing or vice versa), a friction offset resists
the change, causing residual tension after release.

Friction model: friction_offset = max(const, linear_factor * |tension|)
- const: minimum friction always present (even at zero tension)
- linear_factor: friction proportional to current tendon command magnitude

Per-segment: 2 parameters per segment (const + linear_factor).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class TendonFrictionParameter(BaseParameter):
    """Optimizable tendon friction (hysteresis) parameters.

    Per-segment with 2 params each: constant term and linear factor.
    Total dimensions = 2 * num_segments.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.num_segments = config.get("num_segments", 3)
        self.bounds_const = config.get("bounds_const", [0.0, 0.002])
        self.bounds_linear = config.get("bounds_linear", [0.0, 0.5])
        self.num_dims = 2 * self.num_segments

    def get_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        for _ in range(self.num_segments):
            bounds.append((self.bounds_const[0], self.bounds_const[1]))
            bounds.append((self.bounds_linear[0], self.bounds_linear[1]))
        return bounds

    def get_dimension_names(self) -> List[str]:
        names = []
        for i in range(self.num_segments):
            names.append(f"friction_const_seg{i+1}")
            names.append(f"friction_linear_seg{i+1}")
        return names

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        consts = [float(values[2 * i]) for i in range(self.num_segments)]
        linears = [float(values[2 * i + 1]) for i in range(self.num_segments)]
        generation_config.setdefault("sysid_params", {})
        generation_config["sysid_params"]["tendon_friction_const"] = consts
        generation_config["sysid_params"]["tendon_friction_linear"] = linears
        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        parts = []
        for i in range(self.num_segments):
            c = values[2 * i] * 1000  # to mm
            linear = values[2 * i + 1]
            parts.append(f"S{i+1}:{c:.3f}mm+{linear:.3f}x")
        return f"Tendon friction: {' '.join(parts)}"
