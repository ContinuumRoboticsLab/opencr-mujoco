"""
Scene generation modules for creating MuJoCo XML files.

This module provides generators for:
- TDCR (Tendon-Driven Continuum Robots) with unified configuration
- Franka robots
- Hybrid systems
- Obstacles and environments
"""

from .unified_tdcr_generator import UnifiedTDCRConfig, create_tdcr_from_config

__all__ = [
    "UnifiedTDCRConfig",
    "create_tdcr_from_config",
]
