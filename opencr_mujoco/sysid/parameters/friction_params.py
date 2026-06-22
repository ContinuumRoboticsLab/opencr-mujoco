"""Friction/hysteresis parameter optimization for TDCR."""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class FrictionParameter(BaseParameter):
    """Parameter for optimizing joint and tendon frictionloss.

    MuJoCo's frictionloss adds Coulomb (dry) friction - a constant opposing
    torque/force that creates hysteresis. Two components:
    - joint_frictionloss: friction in backbone joints (N·m)
    - tendon_frictionloss: friction in tendon routing (N)

    Modes:
    - 'joint_only': Optimize only joint frictionloss (1 dim)
    - 'tendon_only': Optimize only tendon frictionloss (1 dim)
    - 'both': Optimize both jointly (2 dims)
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        self.friction_mode = config.get("friction_mode", "both")

        # Get bounds
        joint_bounds = config.get("joint_frictionloss_bounds", [0.0, 0.01])
        tendon_bounds = config.get("tendon_frictionloss_bounds", [0.0, 1.0])

        if self.friction_mode == "joint_only":
            self.num_dims = 1
            self.bounds = [joint_bounds]
            self._components = ["joint"]
        elif self.friction_mode == "tendon_only":
            self.num_dims = 1
            self.bounds = [tendon_bounds]
            self._components = ["tendon"]
        elif self.friction_mode == "both":
            self.num_dims = 2
            self.bounds = [joint_bounds, tendon_bounds]
            self._components = ["joint", "tendon"]
        else:
            raise ValueError(f"Unknown friction mode: {self.friction_mode}")

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(b[0], b[1]) for b in self.bounds]

    def get_dimension_names(self) -> List[str]:
        names = []
        for comp in self._components:
            names.append(f"{comp}_frictionloss")
        return names

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        idx = 0
        for comp in self._components:
            if comp == "joint":
                generation_config["joint_frictionloss"] = float(values[idx])
            elif comp == "tendon":
                generation_config["tendon_frictionloss"] = float(values[idx])
            idx += 1
        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        parts = []
        idx = 0
        for comp in self._components:
            if comp == "joint":
                parts.append(f"joint_friction={values[idx]:.6f} N·m")
            elif comp == "tendon":
                parts.append(f"tendon_friction={values[idx]:.4f} N")
            idx += 1
        return f"Friction: {', '.join(parts)}"
