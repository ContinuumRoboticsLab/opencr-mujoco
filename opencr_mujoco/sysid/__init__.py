"""System Identification module for TDCR parameter optimization.

3-step pipeline workflow:
  Step 1: Geometric identification (seg_offsets, tendon_angle_deltas) from tendon pulls
  Step 2: Tendon parameter optimization (pretension, kp, constraint) on tendon pulls
  Step 3: Refinement on train data, validation on val data
"""

from .sysid_optimizer import SystemIdentificationOptimizer
from .data_loader import TrajectoryDataLoader
from .trajectory_simulator import TrajectorySimulator
from .error_metrics import RMSEMetric, MetricCombiner
from .visualization import SysIDVisualizer
from .pipeline_orchestrator import PipelineOrchestrator
from .geometric_identifier import GeometricIdentifier
from .pipeline_data_loader import PipelineDataLoader

__all__ = [
    "SystemIdentificationOptimizer",
    "TrajectoryDataLoader",
    "TrajectorySimulator",
    "RMSEMetric",
    "MetricCombiner",
    "SysIDVisualizer",
    "PipelineOrchestrator",
    "GeometricIdentifier",
    "PipelineDataLoader",
]
