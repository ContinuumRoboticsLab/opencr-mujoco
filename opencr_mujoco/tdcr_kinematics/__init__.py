"""TDCR Kinematics module for tendon-driven continuum robots."""

from .n_tendon_segment_kinematics import NTendonSegmentKinematics
from .multi_segment_tdcr_kinematics import MultiSegmentTDCRKinematics
from .multi_segment_tdcr_independent import MultiSegmentTDCRIndependentKinematics
from .multi_segment_tdcr_tension import MultiSegmentTDCRTensionKinematics
from .three_segment_wrapper import ThreeTendonThreeSegmentTDCRKinematics

__all__ = [
    "NTendonSegmentKinematics",
    "MultiSegmentTDCRKinematics",
    "MultiSegmentTDCRIndependentKinematics",
    "MultiSegmentTDCRTensionKinematics",
    "ThreeTendonThreeSegmentTDCRKinematics",  # Keep for backward compatibility
]
