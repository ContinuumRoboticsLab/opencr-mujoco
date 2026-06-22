"""Geometric parameter optimization for TDCR."""

from typing import Dict, Any, List, Tuple
import numpy as np
from ..base_parameter import BaseParameter


class TendonOffsetParameter(BaseParameter):
    """Parameter for optimizing tendon angular offsets.

    Optimizes the angular placement of tendons around each segment.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize tendon offset parameter.

        Args:
            name: Parameter name
            config: Configuration with mode and bounds
        """
        super().__init__(name, config)
        self.num_segments = config.get("num_segments", 3)
        self.tendons_per_segment = config.get("tendons_per_segment", 3)

        # Mode determines optimization strategy
        if self.mode == "global":
            # Single offset pattern for all segments
            self.num_dims = 1
        elif self.mode == "per_segment":
            # Different offset for each segment
            self.num_dims = self.num_segments
        elif self.mode == "per_segment_relative":
            # Base offset + relative offsets for each segment
            self.num_dims = self.num_segments
        elif self.mode == "per_tendon":
            # Individual offset delta for every tendon
            self.num_dims = self.num_segments * self.tendons_per_segment
        else:
            raise ValueError(f"Unknown offset mode: {self.mode}")

        # Get bounds (in radians)
        bounds = config.get("bounds", [-0.2, 0.2])
        if isinstance(bounds[0], (list, tuple)):
            self.bounds = bounds
        else:
            self.bounds = [bounds] * self.num_dims

        # Base offsets (default 60-degree increments: 0, 60, 120 deg)
        self.base_offsets = config.get("base_offsets", [0, np.pi / 3, 2 * np.pi / 3])

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds.

        Returns:
            List of (min, max) tuples
        """
        return [(b[0], b[1]) for b in self.bounds]

    def get_dimension_names(self) -> List[str]:
        """Get dimension names.

        Returns:
            List of dimension names
        """
        if self.mode == "global":
            return ["tendon_offset_delta"]
        elif self.mode == "per_segment":
            return [f"offset_seg{i+1}" for i in range(self.num_segments)]
        elif self.mode == "per_tendon":
            return [
                f"offset_s{s+1}_t{t+1}"
                for s in range(self.num_segments)
                for t in range(self.tendons_per_segment)
            ]
        else:  # per_segment_relative
            return [f"offset_delta_seg{i+1}" for i in range(self.num_segments)]

    def apply_to_config(
        self, values: np.ndarray, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply offset values to generation config.

        Args:
            values: Optimized offset values (radians)
            generation_config: TDCR generation configuration

        Returns:
            Modified generation config
        """
        # Calculate final offsets based on mode
        if self.mode == "per_tendon":
            # Per-tendon deltas stored as nested list
            tendon_deltas = []
            idx = 0
            for s in range(self.num_segments):
                seg_deltas = []
                for t in range(self.tendons_per_segment):
                    seg_deltas.append(float(values[idx]))
                    idx += 1
                tendon_deltas.append(seg_deltas)
            generation_config["tendon_angle_deltas"] = tendon_deltas
            return generation_config

        if self.mode == "global":
            # Apply same delta to all segments
            seg_offsets = [
                self.base_offsets[i % 3] + values[0] for i in range(self.num_segments)
            ]

        elif self.mode == "per_segment":
            # Direct offsets for each segment
            seg_offsets = []
            for seg_idx, delta in enumerate(values):
                base = self.base_offsets[seg_idx % 3]
                seg_offsets.append(base + delta)

        else:  # per_segment_relative
            # Each segment gets its own delta
            seg_offsets = []
            for seg_idx, delta in enumerate(values):
                base = self.base_offsets[seg_idx % 3]
                seg_offsets.append(base + delta)

        # Apply to generation config
        generation_config["seg_offsets"] = [float(off) for off in seg_offsets]

        # Also update actuation details if present
        if "actuation_details" in generation_config:
            if "segments" in generation_config["actuation_details"]:
                # Update angular offsets in actuation details
                for seg_idx, segment in enumerate(
                    generation_config["actuation_details"]["segments"]
                ):
                    if seg_idx < len(seg_offsets):
                        segment["angular_offset"] = float(seg_offsets[seg_idx])

        return generation_config

    def format_values(self, values: np.ndarray) -> str:
        """Format values for display.

        Args:
            values: Parameter values

        Returns:
            Formatted string
        """
        if self.mode == "global":
            return f"Tendon offset: {np.degrees(values[0]):.1f}° (all)"
        elif self.mode == "per_tendon":
            idx = 0
            parts = []
            for s in range(self.num_segments):
                t_parts = [
                    f"t{t+1}:{np.degrees(values[idx + t]):.1f}°"
                    for t in range(self.tendons_per_segment)
                ]
                parts.append(f"S{s+1}[{' '.join(t_parts)}]")
                idx += self.tendons_per_segment
            return f"Tendon offsets: {' '.join(parts)}"
        else:
            parts = [f"S{i+1}:{np.degrees(v):.1f}°" for i, v in enumerate(values)]
            return f"Tendon offsets: {' '.join(parts)}"
