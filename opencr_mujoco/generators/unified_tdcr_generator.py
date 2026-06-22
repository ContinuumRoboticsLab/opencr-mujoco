"""Unified TDCR (Tendon-Driven Continuum Robot) MJCF XML generator.

This module provides a flexible system for generating MuJoCo XML models of TDCRs
with various actuation modes, material properties, and structural configurations.

Key Features:
    - Material-based stiffness calculation with optional backbone radius
    - Multiple actuation modes (``parallel_tendons``, ``direct_torque``, ``none``);
      tendon actuators can be ``motor`` (force) or ``position`` via
      ``actuator_properties.tendon_actuator_type``
    - Multi-segment robots with per-segment customization
    - Automatic tendon routing and actuator generation
    - Collision exclusion support
    - Pre-tension initialization via keyframes

Example:
    from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config

    config = {
        "num_segments": 1,
        "links_per_segment": {"1": 30},
        "segment_lengths": {"1": 0.3},
        "radius": 0.01,
        "joint_config_mode": "material",
        "material_properties": {
            "youngs_modulus": 200e9,
            "density": 8000,
            "poisson_ratio": 0.3,
            "outer_radius": 0.001,
        },
        "actuation_mode": "none",
    }

    create_tdcr_from_config(config, "my_tdcr.xml")
"""

import xml.etree.ElementTree as ET
import math
import os
import glob
import json
import argparse
from typing import Optional, List, Tuple, Dict, Any, NamedTuple
import numpy as np


class LinkProps(NamedTuple):
    """Per-link physical properties of a uniform rod section."""

    link_length: float
    link_mass: float
    bend_stiffness: float
    torsion_stiffness: float
    bend_damping: Optional[float]
    torsion_damping: Optional[float]


def compute_link_properties(
    *,
    length: float,
    num_links: int,
    material_properties: Dict[str, Any],
    fallback_radius: float,
    damping_fallback: str = "module",
) -> LinkProps:
    """Per-link properties for a uniform section, from beam theory.

    Single source of truth shared by the modular path (per module) and the
    standard path (per segment / global). Formulas:
        link_length      = L / n
        link_mass        = density * A * L / n,   A = pi (ro^2 - ri^2)
        bend_stiffness   = n * E * I_x / L,        I_x = pi/4 (ro^4 - ri^4)
        torsion_stiffness= n * G * I_z / L,        I_z = 2 I_x, G = E / 2(1+nu)

    Damping: ``damping_ratio * K`` if a ratio is given, else the absolute
    ``damping``/``torsion_damping``. When neither is given the ``damping_fallback``
    decides: ``"module"`` -> ``0.01 * K`` (modular behavior); ``"config_default"``
    -> ``None`` (caller keeps its own configured damping default).
    """
    props = material_properties
    n = num_links
    outer_radius = props.get("outer_radius", fallback_radius)
    inner_radius = props.get("inner_radius", 0)
    density = props["density"]
    youngs_modulus = props["youngs_modulus"]
    poisson_ratio = props.get("poisson_ratio", 0.3)

    area = math.pi * (outer_radius**2 - inner_radius**2)
    i_x = (math.pi / 4) * (outer_radius**4 - inner_radius**4)
    i_z = 2 * i_x
    shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))

    link_length = length / n
    link_mass = (density * area * length) / n
    bend_stiffness = n * (youngs_modulus * i_x) / length
    torsion_stiffness = n * (shear_modulus * i_z) / length

    if "damping_ratio" in props:
        ratio = props["damping_ratio"]
        bend_damping = ratio * bend_stiffness
        torsion_damping = ratio * torsion_stiffness
    elif "damping" in props:
        bend_damping = props["damping"]
        torsion_damping = props.get("torsion_damping", props["damping"])
    elif damping_fallback == "module":
        bend_damping = 0.01 * bend_stiffness
        torsion_damping = 0.01 * torsion_stiffness
    else:  # "config_default" -> caller keeps its own default
        bend_damping = None
        torsion_damping = None

    return LinkProps(
        link_length,
        link_mass,
        bend_stiffness,
        torsion_stiffness,
        bend_damping,
        torsion_damping,
    )


class ModuleDefinition:
    """Definition of a single TDCR module with independent physical properties.

    A module represents a physically distinct section of the robot with uniform
    material properties. Modules can be composed to create heterogeneous TDCRs.

    Attributes:
        module_id: Unique identifier for this module
        length: Physical length of the module (meters)
        radius: Visual/collision radius (meters)
        num_links: Number of MuJoCo links used to simulate this module
        color: RGBA color string (e.g., "0.9 0.3 0.3 1.0")
        material_properties: Dict with density, youngs_modulus, poisson_ratio, etc.
        link_length: Computed length per link (m)
        link_mass: Computed mass per link (kg)
        joint_stiffness: Computed bending stiffness for joints (Nm/rad)
        torsion_stiffness: Computed torsional stiffness (Nm/rad)
        joint_damping: Computed damping (Nm·s/rad)
    """

    def __init__(self, module_id: str, module_dict: Dict[str, Any]):
        """Initialize module from config dictionary.

        Args:
            module_id: Unique identifier for this module
            module_dict: Dictionary with module properties (length, radius, num_links, etc.)
        """
        self.module_id = module_id
        self.length = module_dict["length"]
        self.radius = module_dict.get("radius", 0.006)
        self.num_links = module_dict.get("num_links", 5)
        self.color = module_dict.get("color", None)  # None means use default gradient
        self.material_properties = module_dict.get("material_properties", {})

        # Validate required fields
        if "material_properties" not in module_dict:
            raise ValueError(f"Module '{module_id}' missing 'material_properties'")

        # Compute derived physical properties
        self._compute_physical_properties()

    def _compute_physical_properties(self):
        """Calculate link-level physical properties from module parameters.

        Delegates to the shared ``compute_link_properties`` (beam theory); modules
        fall back to ``0.01 * K`` damping when no damping is configured.
        """
        lp = compute_link_properties(
            length=self.length,
            num_links=self.num_links,
            material_properties=self.material_properties,
            fallback_radius=self.radius,
            damping_fallback="module",
        )
        self.link_length = lp.link_length
        self.link_mass = lp.link_mass
        self.joint_stiffness = lp.bend_stiffness
        self.torsion_stiffness = lp.torsion_stiffness
        self.joint_damping = lp.bend_damping
        self.torsion_damping = lp.torsion_damping

    def __repr__(self):
        return (
            f"ModuleDefinition(id='{self.module_id}', length={self.length:.4f}m, "
            f"num_links={self.num_links}, K={self.joint_stiffness:.2f}Nm/rad)"
        )


class UnifiedTDCRConfig:
    """Configuration class for TDCR generation with comprehensive options.

    This class handles all configuration parameters for generating TDCRs,
    including structural properties, material parameters, actuation modes,
    and visual settings.

    Attributes:
        num_segments: Number of robot segments
        total_links: Total number of links across all segments
        radius: Visual/collision radius of the robot (m)
        joint_config_mode: 'direct' or 'material' for stiffness calculation
        material_properties: Dict with density, youngs_modulus, etc.
        actuation_mode: 'direct_torque', 'parallel_tendons', or 'none'
        disable_self_collision: Whether to add collision exclusions
    """

    def __init__(self, config_dict: Dict[str, Any]):
        # Basic robot parameters
        self.num_segments = config_dict.get("num_segments", 1)
        self.links_per_segment = config_dict.get("links_per_segment", {"1": 10})
        self.segment_lengths = config_dict.get("segment_lengths", {"1": 0.6})

        # Modular configs derive links_per_segment / segment_lengths from their
        # module library further down in __init__. Until then the dicts above are
        # just placeholders, so the totals cannot be computed here (doing so would
        # KeyError on a segment that the placeholder dict doesn't contain). Read
        # the modular flag now and defer the totals to the modular block.
        self.modular = config_dict.get("modular", False)

        # Calculate total_links and total_length from per-segment values if not provided
        if self.modular:
            self.total_links = 0  # populated from the module library below
            self.total_length = 0.0
        elif "total_links" not in config_dict:
            self.total_links = sum(
                self.links_per_segment[str(i + 1)] for i in range(self.num_segments)
            )
        else:
            self.total_links = config_dict["total_links"]

        if not self.modular:
            if "total_length" not in config_dict:
                self.total_length = sum(
                    self.segment_lengths[str(i + 1)] for i in range(self.num_segments)
                )
            else:
                self.total_length = config_dict["total_length"]

        self.radius = config_dict.get("radius", 0.035)
        self.mass = config_dict.get("mass", 1.0)

        # Joint configuration
        self.joints_per_link = config_dict.get("joints_per_link", 2)  # 2 or 3
        self.joint_config_mode = config_dict.get(
            "joint_config_mode", "direct"
        )  # 'direct' or 'material'

        # Direct joint properties (used when joint_config_mode == 'direct')
        self.stiffness = config_dict.get("stiffness", 100.0)
        self.damping = config_dict.get("damping", 1.0)
        self.torsion_stiffness = config_dict.get("torsion_stiffness", None)  # Optional

        # Material properties (used when joint_config_mode == 'material')
        self.material_properties = config_dict.get(
            "material_properties",
            {
                "density": 37708,
                "youngs_modulus": 200e9,
                "poisson_ratio": 0.3,
                "inner_radius": 0,
            },
        )

        # Optional per-segment material override. When present (one block per
        # segment), each segment derives its own link length / mass / stiffness
        # from its own length and material — independent parameters per segment,
        # like the modular path. Absent => the single global block above applies
        # to all segments (current behavior). Presence also enables per-segment
        # geometry; `per_segment_geometry: true` enables it with shared material.
        self.material_properties_per_segment = config_dict.get(
            "material_properties_per_segment", None
        )
        if (
            self.material_properties_per_segment is not None
            and len(self.material_properties_per_segment) != self.num_segments
        ):
            raise ValueError(
                "material_properties_per_segment must have one entry per segment "
                f"(num_segments={self.num_segments}); got "
                f"{len(self.material_properties_per_segment)}"
            )
        self.per_segment_geometry = bool(
            self.material_properties_per_segment is not None
            or config_dict.get("per_segment_geometry", False)
        )

        # Joint ranges
        self.joint_limited = config_dict.get(
            "joint_limited", False
        )  # Default to unlimited
        self.joint_range = config_dict.get("joint_range", "-500 500")
        self.vert_joint_range = config_dict.get("vert_joint_range", "-500 500")

        # Actuation
        self.actuation_mode = config_dict.get(
            "actuation_mode", "direct_torque"
        )  # 'direct_torque', 'parallel_tendons', 'none'
        self.actuation_details = config_dict.get("actuation_details", {})

        # Validate actuation details match segment count
        if self.actuation_mode == "parallel_tendons" and not self.modular:
            if "segments" not in self.actuation_details:
                raise ValueError(
                    "actuation_mode 'parallel_tendons' requires "
                    "actuation_details.segments (one entry per segment with "
                    "number_of_tendons and distance_to_backbone)."
                )
            if len(self.actuation_details["segments"]) < self.num_segments:
                raise ValueError(
                    f"Actuation details has {len(self.actuation_details['segments'])} segments "
                    f"but num_segments is {self.num_segments}. Please provide actuation details for all segments."
                )

        # Actuator properties
        self.actuator_type = config_dict.get(
            "actuator_type", "motor"
        )  # 'motor', 'position', 'velocity'
        self.actuator_properties = config_dict.get("actuator_properties", {})

        # Visual
        self.colour_scheme = config_dict.get("colour_scheme", "Clean")
        self.link_shape = config_dict.get("link_shape", "cylinder")

        # Environment
        self.gravity = config_dict.get("gravity", "0 0 -9.81")
        self.plane = config_dict.get("plane", True)
        self.plane_style = config_dict.get(
            "plane_style", "checkered"
        )  # 'checkered' or 'white'
        self.axis_label = config_dict.get("axis_label", False)
        self.disable_self_collision = config_dict.get("disable_self_collision", False)
        self.disable_contact = config_dict.get(
            "disable_contact", False
        )  # Disable all contact/collision

        # Collision settings for TDCR interaction with external objects
        self.enable_tdcr_collision = config_dict.get("enable_tdcr_collision", True)
        self.tdcr_contype = config_dict.get("tdcr_contype", 4)
        self.tdcr_conaffinity = config_dict.get("tdcr_conaffinity", 7)

        # Mounting (Franka mounting is handled by generate.py's scene merge)
        self.enable_movement = config_dict.get("enable_movement", True)

        # Obstacles
        self.taskspace_file = config_dict.get("taskspace_file", None)
        self.custom_obstacles = ([], [])

        # Segment offsets for tendon routing
        if "seg_offsets" in config_dict:
            self.seg_offsets = np.array(config_dict["seg_offsets"])
            if len(self.seg_offsets) < self.num_segments:
                raise ValueError(
                    f"seg_offsets has {len(self.seg_offsets)} entries but "
                    f"num_segments is {self.num_segments}."
                )
        else:
            self.seg_offsets = np.zeros(self.num_segments)

        # Independent segments mode (tendons only route through their own segment)
        self.independent_segments = config_dict.get("independent_segments", False)

        # Per-tendon angular deltas: list of lists [[d1,d2,d3], [d4,d5,d6], ...]
        self.tendon_angle_deltas = config_dict.get("tendon_angle_deltas", None)

        # Friction / hysteresis
        self.joint_frictionloss = config_dict.get("joint_frictionloss", 0.0)
        self.tendon_frictionloss = config_dict.get("tendon_frictionloss", 0.0)

        # Joint deadband (springlength deadband in radians)
        # When set, joint spring force is zero within [-deadband, +deadband]
        # This prevents the backbone from fully returning to straight after bending
        # Can be scalar (all segments) or list (per-segment)
        db_raw = config_dict.get("joint_deadband", 0.0)
        if isinstance(db_raw, list):
            self.joint_deadband_per_segment = db_raw
            self.joint_deadband = max(db_raw)  # nonzero if any segment has deadband
        else:
            self.joint_deadband_per_segment = None
            self.joint_deadband = db_raw

        # Modular TDCR support (self.modular was read near the top of __init__)
        self.module_library = {}
        self.modular_segments = []

        if self.modular:
            # Parse module library
            module_lib_dict = config_dict.get("module_library", {})
            for module_id, module_dict in module_lib_dict.items():
                self.module_library[module_id] = ModuleDefinition(
                    module_id, module_dict
                )

            # Parse segment definitions
            segments_list = config_dict.get("segments", [])
            for seg_idx, seg_dict in enumerate(segments_list):
                # Get module IDs for this segment
                module_ids = seg_dict.get("modules", [])

                # Validate that all referenced modules exist
                for mod_id in module_ids:
                    if mod_id not in self.module_library:
                        raise ValueError(
                            f"Segment {seg_idx} references undefined module '{mod_id}'. "
                            f"Available modules: {list(self.module_library.keys())}"
                        )

                # Store segment definition
                self.modular_segments.append(
                    {
                        "module_ids": module_ids,
                        "actuation": seg_dict.get("actuation", {}),
                    }
                )

            # Compute total links and lengths from modules
            total_links = 0
            segment_links = {}
            segment_lengths = {}

            for seg_idx, seg_def in enumerate(self.modular_segments):
                seg_num = seg_idx + 1
                seg_links = 0
                seg_length = 0.0

                for mod_id in seg_def["module_ids"]:
                    module = self.module_library[mod_id]
                    seg_links += module.num_links
                    seg_length += module.length

                segment_links[str(seg_num)] = seg_links
                segment_lengths[str(seg_num)] = seg_length
                total_links += seg_links

            # Override non-modular values with computed ones
            self.total_links = total_links
            self.links_per_segment = segment_links
            self.segment_lengths = segment_lengths
            self.total_length = sum(segment_lengths.values())

            # Populate actuation_details from modular segment definitions
            self.actuation_details["segments"] = [
                seg_def["actuation"] for seg_def in self.modular_segments
            ]

        elif self.per_segment_geometry:
            # A per-segment-material standard config IS a modular robot with one
            # module per segment: each segment has its own material, forms a
            # half-link-bounded chain, and joins rigidly to the next. Synthesize
            # that modular representation and reuse the modular build path, so
            # multi-material and modular share a single per-section builder.
            # total_links / links_per_segment / segment_lengths and the
            # per-segment actuation_details are already correct here.
            for s in range(self.num_segments):
                material = (
                    self.material_properties_per_segment[s]
                    if self.material_properties_per_segment is not None
                    else self.material_properties
                )
                module_id = f"segment_{s + 1}"
                self.module_library[module_id] = ModuleDefinition(
                    module_id,
                    {
                        "length": float(self.segment_lengths[str(s + 1)]),
                        "radius": self.radius,
                        "num_links": self.links_per_segment[str(s + 1)],
                        "color": None,
                        "material_properties": material,
                    },
                )
                segs = self.actuation_details.get("segments", [])
                self.modular_segments.append(
                    {
                        "module_ids": [module_id],
                        "actuation": segs[s] if s < len(segs) else {},
                    }
                )
            self.modular = True

        # --- Feature-combination validation -------------------------------
        # joint_deadband moves stiffness from joints onto fixed tendons with a
        # springlength deadband; that compensation is only implemented for the
        # standard parallel_tendons chain. Reject combinations that would
        # silently produce a zero-stiffness robot (or silently ignore the key).
        if self.joint_deadband > 0 and (
            self.actuation_mode != "parallel_tendons" or self.modular
        ):
            raise ValueError(
                "joint_deadband is only supported for standard (non-modular) "
                "parallel_tendons configs; it would zero the joint stiffness "
                "without adding the compensating deadband tendons here."
            )
        if self.modular:
            # The modular chain writes explicit per-joint attributes and its
            # own coupled tendon routing, so these standard-path options do
            # not apply. Warn instead of silently ignoring them.
            if self.joint_frictionloss > 0:
                print(
                    "Warning: joint_frictionloss is not implemented for "
                    "modular configs and will be ignored."
                )
            if self.independent_segments:
                print(
                    "Warning: independent_segments is not implemented for "
                    "modular configs; tendon routing stays coupled."
                )

    def get_segment_from_link(self, link_num: int) -> int:
        """Get segment number from link index."""
        link_num = link_num + 1
        accumulated_links = 0

        for segment in range(1, self.num_segments + 1):
            accumulated_links += self.links_per_segment[str(segment)]
            if link_num <= accumulated_links:
                return segment

        return self.num_segments


class UnifiedTDCRGenerator:
    """Unified TDCR XML generator supporting all configuration options."""

    def __init__(self, config: UnifiedTDCRConfig, output_path: Optional[str] = None):
        self.config = config
        self.output_path = output_path or "tdcr_generated.xml"
        self.abs_xml_filepath = None

        # Colors
        self.default_colors = ["0.274 0.274 0.274 1.000", "0.574 0.574 0.574 1.000"]
        self.segment_colours = self._generate_gradient_grey(config.num_segments)
        self.tendon_colours = [
            "0.4 0.6 0.8 1",  # Soft Blue
            "0.5 0.7 0.5 1",  # Muted Green
            "0.9 0.6 0.3 1",  # Warm Orange
            "0.5 0.3 0.7 1",  # Deep Purple
            "0.8 0.4 0.4 1",  # Soft Red
            "0.3 0.7 0.7 1",  # Teal
        ]

        self.joint_names = []
        # Chain body names in base-to-tip order, filled by the chain builders;
        # used for neighbor-pair collision exclusions.
        self.chain_body_names = []

        # Calculate joint stiffness if using material properties
        if config.joint_config_mode == "material":
            self._calculate_from_material_properties()
        else:
            self.link_length = config.total_length / config.total_links
            self.link_mass = config.mass / config.total_links
            self.stiffness = config.stiffness
            self.torsion_stiffness = config.torsion_stiffness or config.stiffness

    def _calculate_from_material_properties(self):
        """Calculate joint stiffness and mass from material properties.

        This method computes physical parameters based on material science principles:
        - Uses outer_radius (if specified) for backbone stiffness calculation
        - Falls back to robot radius if outer_radius not provided
        - Calculates bending and torsional stiffness from Young's modulus
        - Supports hollow structures via inner_radius
        - Optionally uses damping from material properties

        The stiffness calculation follows beam theory:
        - Bending stiffness: K = n * E * I / L
        - Torsional stiffness: K_t = n * G * I_z / L

        Where:
        - n: number of links
        - E: Young's modulus
        - I: second moment of area
        - L: total length
        - G: shear modulus (calculated from E and Poisson's ratio)
        """
        lp = compute_link_properties(
            length=self.config.total_length,
            num_links=self.config.total_links,
            material_properties=self.config.material_properties,
            fallback_radius=self.config.radius,
            damping_fallback="config_default",
        )
        self.link_length = lp.link_length
        self.link_mass = lp.link_mass
        self.stiffness = lp.bend_stiffness
        self.torsion_stiffness = lp.torsion_stiffness

        # Only override the config damping when the material block actually
        # specifies damping (damping_ratio or damping); otherwise keep the
        # configured default (historical behavior of this path).
        if lp.bend_damping is not None:
            self.config.damping = lp.bend_damping
            self.config.torsion_damping = lp.torsion_damping

    def _emit_link_geom(
        self,
        body: ET.Element,
        *,
        length: float,
        mass: float,
        color: str,
        name: str,
        gtype: Optional[str] = None,
        radius: Optional[float] = None,
    ):
        """Emit one link capsule/cylinder geom (shared by every body-chain mode).

        Centralizes the geom attributes + collision-class handling that were
        duplicated across the three builders.
        """
        attrib = {
            "fromto": f"0 0 0 0 0 {length}",
            "size": str(radius if radius is not None else self.config.radius),
            "type": gtype or self.config.link_shape,
            "rgba": color,
            "mass": str(mass),
            "name": name,
        }
        if self.config.enable_tdcr_collision:
            attrib["class"] = "tdcr_collision"
        else:
            attrib["contype"] = "0"
            attrib["conaffinity"] = "0"
        ET.SubElement(body, "geom", attrib=attrib)

    def _generate_gradient_grey(self, num_segments: int) -> List[str]:
        """Generate gradient grey colors for segments."""
        dark_grey = [0.274, 0.274, 0.274, 1]
        light_grey = [0.91, 0.91, 0.91, 1]

        segment_colours = []
        for i in range(num_segments):
            t = i / (num_segments - 1) if num_segments > 1 else 0
            color = [
                dark_grey[j] + t * (light_grey[j] - dark_grey[j]) for j in range(4)
            ]
            color_str = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} {color[3]:.3f}"
            segment_colours.append(color_str)

        return segment_colours

    def _add_actuator(
        self, actuator: ET.Element, joint: str, name: str, ctrlrange: str
    ):
        """Add an actuator with configurable properties.

        Args:
            actuator: Actuator XML element
            joint: Joint name
            name: Actuator name
            ctrlrange: Control range
        """
        actuator_attrib = {"joint": joint, "name": name, "ctrlrange": ctrlrange}

        # Add default properties based on actuator type
        if self.config.actuator_type == "position":
            # Default position control properties
            actuator_attrib.update(
                {
                    "kp": str(self.config.actuator_properties.get("kp", 1000)),
                    "kv": str(self.config.actuator_properties.get("kv", 10)),
                }
            )
        elif self.config.actuator_type == "velocity":
            # Default velocity control properties
            actuator_attrib.update(
                {"kv": str(self.config.actuator_properties.get("kv", 100))}
            )

        # Add any additional custom properties
        for key, value in self.config.actuator_properties.items():
            if key not in ["kp", "kv"]:  # Skip already handled properties
                actuator_attrib[key] = str(value)

        # Create actuator element
        ET.SubElement(actuator, self.config.actuator_type, attrib=actuator_attrib)

    def generate(self) -> str:
        """Generate the MJCF XML file."""
        mujoco = self._create_xml_structure()

        # Save file
        tree = ET.ElementTree(mujoco)
        ET.indent(tree, space="    ", level=0)
        tree.write(self.output_path)
        self.abs_xml_filepath = os.path.abspath(self.output_path)

        return self.abs_xml_filepath

    def _create_xml_structure(self) -> ET.Element:
        """Create the main XML structure."""
        # Root element
        mujoco = ET.Element("mujoco")

        # Compiler settings
        ET.SubElement(
            mujoco, "compiler", attrib={"angle": "radian", "autolimits": "true"}
        )

        # Option - simplified for stable simulation
        option_elem = ET.SubElement(
            mujoco,
            "option",
            attrib={
                "gravity": self.config.gravity,
                "noslip_iterations": "1",  # Further reduced for speed
            },
        )

        # Add flag element with contact="disable" if requested (useful for evaluations)
        if self.config.disable_contact:
            ET.SubElement(option_elem, "flag", attrib={"contact": "disable"})

        # Asset
        asset = ET.SubElement(mujoco, "asset")
        self._create_assets(asset)

        # Default
        default = ET.SubElement(mujoco, "default")
        self._create_joint_defaults(default)

        # Worldbody
        worldbody = ET.SubElement(mujoco, "worldbody")

        # Add plane if enabled
        if self.config.plane:
            self._create_plane(worldbody)

        # Add obstacles
        self._add_obstacles(mujoco, worldbody)

        # Create robot
        actuator = ET.SubElement(mujoco, "actuator")
        self._create_robot(worldbody, mujoco, actuator)

        # Add axis labels if enabled (requires arrow STL meshes on disk)
        if self.config.axis_label:
            if os.path.isdir("MJCFS/STLS_dir/AxisArrowSTLS"):
                self._add_axis_labels(worldbody, asset)
            else:
                print(
                    "Warning: axis_label requires arrow STL meshes under "
                    "MJCFS/STLS_dir/AxisArrowSTLS/ (not shipped); skipping."
                )

        # Add keyframe for tendons (always needed to set initial control values)
        if self.config.actuation_mode == "parallel_tendons":
            self._add_pretension_keyframe(mujoco)

        # Add collision exclusions if specified
        if (
            hasattr(self.config, "disable_self_collision")
            and self.config.disable_self_collision
        ):
            self._add_collision_exclusions(mujoco)

        return mujoco

    def _create_assets(self, asset: ET.Element):
        """Create asset elements."""
        # Sky texture
        ET.SubElement(
            asset,
            "texture",
            attrib={
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "1 1 1",
                "rgb2": ".6 .8 1",
                "width": "256",
                "height": "256",
            },
        )

        # Plane textures if using checkered style
        if self.config.plane and self.config.plane_style == "checkered":
            ET.SubElement(
                asset,
                "texture",
                attrib={
                    "name": "texplane",
                    "type": "2d",
                    "builtin": "checker",
                    "rgb1": ".2 .3 .4",
                    "rgb2": ".1 .15 .2",
                    "width": "512",
                    "height": "512",
                },
            )
            ET.SubElement(
                asset,
                "material",
                attrib={
                    "name": "matplane",
                    "texture": "texplane",
                    "texrepeat": "5 5",
                    "reflectance": "0.5",
                    "texuniform": "true",
                },
            )

    def _create_joint_defaults(self, default: ET.Element):
        """Create default joint classes."""

        # Add default geom settings to match reference
        ET.SubElement(
            default,
            "geom",
            attrib={
                "contype": "0",
                "conaffinity": "0",
                "condim": "1",  # Changed to 1 to match reference
                "friction": "1.0 0.005 0.0001",
            },
        )

        # Create geometry defaults for TDCR collision
        if self.config.enable_tdcr_collision:
            tdcr_geom_class = ET.SubElement(
                default, "default", attrib={"class": "tdcr_collision"}
            )
            ET.SubElement(
                tdcr_geom_class,
                "geom",
                attrib={
                    "contype": "1",
                    "conaffinity": "1",  # Enable collision with objects
                    "condim": "3",  # 3D contact for speed
                    "friction": "1.0 0.005 0.0001",  # High static, very low sliding and torsional friction
                    "solimp": "0.8 0.95 0.001",  # Updated solimp to match tuned values
                },
            )

        # When joint_deadband > 0, move stiffness from joints to fixed tendons
        # with springlength deadband. Joints keep damping but have zero stiffness.
        has_deadband = self.config.joint_deadband > 0
        bending_stiffness = "0" if has_deadband else str(self.stiffness)
        torsion_stiffness = "0" if has_deadband else str(self.torsion_stiffness)

        if self.config.joints_per_link == 3:
            # 3-joint configuration (like tdcr_generator)
            # X-axis joint (torsion)
            x_joint_class = ET.SubElement(
                default, "default", attrib={"class": "x_joint"}
            )
            x_joint_attrib = {
                "type": "hinge",
                "axis": "0 0 1",
                "stiffness": torsion_stiffness,
                "damping": str(
                    getattr(self.config, "torsion_damping", self.config.damping)
                ),
                "limited": "false",  # Default to unlimited
            }
            if self.config.joint_frictionloss > 0:
                x_joint_attrib["frictionloss"] = str(self.config.joint_frictionloss)
            # Only add range if joints are limited
            if self.config.joint_limited:
                x_joint_attrib["limited"] = "true"
                x_joint_attrib["range"] = self.config.joint_range
            ET.SubElement(x_joint_class, "joint", attrib=x_joint_attrib)

            # Y-axis joint (bending)
            y_joint_class = ET.SubElement(
                default, "default", attrib={"class": "y_joint"}
            )
            y_joint_attrib = {
                "type": "hinge",
                "axis": "0 1 0",
                "stiffness": bending_stiffness,
                "damping": str(self.config.damping),
                "limited": "false",  # Default to unlimited
            }
            if self.config.joint_frictionloss > 0:
                y_joint_attrib["frictionloss"] = str(self.config.joint_frictionloss)
            # Only add range if joints are limited
            if self.config.joint_limited:
                y_joint_attrib["limited"] = "true"
                y_joint_attrib["range"] = self.config.joint_range
            ET.SubElement(y_joint_class, "joint", attrib=y_joint_attrib)

            # Z-axis joint (bending)
            z_joint_class = ET.SubElement(
                default, "default", attrib={"class": "z_joint"}
            )
            z_joint_attrib = {
                "type": "hinge",
                "axis": "1 0 0",
                "stiffness": bending_stiffness,
                "damping": str(self.config.damping),
                "limited": "false",  # Default to unlimited
            }
            if self.config.joint_frictionloss > 0:
                z_joint_attrib["frictionloss"] = str(self.config.joint_frictionloss)
            # Only add range if joints are limited
            if self.config.joint_limited:
                z_joint_attrib["limited"] = "true"
                z_joint_attrib["range"] = self.config.joint_range
            ET.SubElement(z_joint_class, "joint", attrib=z_joint_attrib)
        else:
            # 2-joint configuration (like create_XML)
            # Planar joint
            planar_class = ET.SubElement(
                default, "default", attrib={"class": "planar_link"}
            )
            planar_attrib = {
                "stiffness": bending_stiffness,
                "damping": str(self.config.damping),
                "limited": "false",  # Default to unlimited
                "axis": "1 0 0",
                "pos": "0 0 0",
                "type": "hinge",
            }
            if self.config.joint_frictionloss > 0:
                planar_attrib["frictionloss"] = str(self.config.joint_frictionloss)
            # Only add range if joints are limited
            if self.config.joint_limited:
                planar_attrib["limited"] = "true"
                planar_attrib["range"] = self.config.joint_range
            ET.SubElement(planar_class, "joint", attrib=planar_attrib)

            # Vertical joint
            vert_class = ET.SubElement(
                default, "default", attrib={"class": "vertical_link"}
            )
            vert_attrib = {
                "stiffness": bending_stiffness,
                "damping": str(self.config.damping),
                "limited": "false",  # Default to unlimited
                "axis": "0 -1 0",
                "pos": "0 0 0",
                "type": "hinge",
            }
            if self.config.joint_frictionloss > 0:
                vert_attrib["frictionloss"] = str(self.config.joint_frictionloss)
            # Only add range if joints are limited
            if self.config.joint_limited:
                vert_attrib["limited"] = "true"
                vert_attrib["range"] = self.config.vert_joint_range
            ET.SubElement(vert_class, "joint", attrib=vert_attrib)

    def _get_deadband_for_link(self, link_idx: int) -> float:
        """Get deadband value for a given link index.

        Args:
            link_idx: Link index (0-based)

        Returns:
            Deadband value in radians
        """
        if self.config.joint_deadband_per_segment is not None:
            seg = self.config.get_segment_from_link(link_idx)  # 1-indexed
            seg_idx = seg - 1
            if seg_idx < len(self.config.joint_deadband_per_segment):
                return self.config.joint_deadband_per_segment[seg_idx]
            return 0.0
        return self.config.joint_deadband

    def _add_joint_deadband_tendons(self, tendon_elem: ET.Element):
        """Add fixed tendons with springlength deadband to replace joint stiffness.

        When joint_deadband > 0, joint stiffness is set to 0 in defaults.
        This method creates a fixed tendon per joint with the original stiffness
        and a springlength deadband, so the restoring force is zero within
        [-deadband, +deadband] radians.

        Supports per-segment deadband values when joint_deadband is a list.
        """
        if self.config.joint_deadband <= 0:
            return

        # Joints exist on links 1..total_links (link_0 is the rigid clamp).
        num_links = self.config.total_links + 1  # includes half-links

        if self.config.joints_per_link == 3:
            axes = ["x", "y", "z"]
            for i in range(1, num_links):
                db = self._get_deadband_for_link(i)
                if db <= 0:
                    continue
                for axis in axes:
                    joint_name = f"joint_{i}_{axis}"
                    stiff = self.torsion_stiffness if axis == "x" else self.stiffness
                    fixed_tendon = ET.SubElement(
                        tendon_elem,
                        "fixed",
                        attrib={
                            "name": f"spring_{joint_name}",
                            "stiffness": str(stiff),
                            "springlength": f"{-db} {db}",
                        },
                    )
                    ET.SubElement(
                        fixed_tendon,
                        "joint",
                        attrib={"joint": joint_name, "coef": "1"},
                    )
        else:
            for i in range(1, num_links):
                db = self._get_deadband_for_link(i)
                if db <= 0:
                    continue
                joint_name = f"joint_{i}"
                # Planar (x) joint
                fixed_x = ET.SubElement(
                    tendon_elem,
                    "fixed",
                    attrib={
                        "name": f"spring_{joint_name}_x",
                        "stiffness": str(self.stiffness),
                        "springlength": f"{-db} {db}",
                    },
                )
                ET.SubElement(
                    fixed_x,
                    "joint",
                    attrib={"joint": f"{joint_name}_x", "coef": "1"},
                )
                # Vertical (z) joint
                fixed_z = ET.SubElement(
                    tendon_elem,
                    "fixed",
                    attrib={
                        "name": f"spring_{joint_name}_z",
                        "stiffness": str(self.stiffness),
                        "springlength": f"{-db} {db}",
                    },
                )
                ET.SubElement(
                    fixed_z,
                    "joint",
                    attrib={"joint": f"{joint_name}_z", "coef": "1"},
                )

    def _create_plane(self, worldbody: ET.Element):
        """Create ground plane."""
        # Light
        ET.SubElement(worldbody, "light", attrib={"pos": "0 0 10", "dir": "0 0 -1"})

        # Plane geom
        plane_attrib = {
            "type": "plane",
            "size": "100 100 0.1",
            "pos": f"0 0 -{self.config.radius * 1.2}",
        }

        if self.config.plane_style == "checkered":
            plane_attrib["material"] = "matplane"
        else:
            plane_attrib["rgba"] = "1 1 1 1"  # White plane

        ET.SubElement(worldbody, "geom", attrib=plane_attrib)

    def _add_obstacles(self, mujoco: ET.Element, worldbody: ET.Element):
        """Add custom obstacles if specified."""
        if hasattr(self.config, "custom_obstacles"):
            bodies, stls = self.config.custom_obstacles

            # Add STL obstacles
            for stl_asset, stl_body in stls:
                mujoco.insert(2, stl_asset)
                worldbody.insert(2, stl_body)

            # Add basic shape obstacles
            for body in bodies:
                worldbody.insert(2, body)

    def _create_robot(
        self, worldbody: ET.Element, mujoco: ET.Element, actuator: ET.Element
    ):
        """Create the robot structure."""
        # Create base/mounting (Franka mounting is handled by generate.py's
        # scene-merge path, not here)
        base_body = self._create_mocap_base(worldbody)

        # Create robot links via the single body-chain builder.
        if self.config.actuation_mode == "parallel_tendons":
            mode = "modular" if self.config.modular else "tendon"
        else:
            mode = "direct"
        self._build_body_chain(base_body, mujoco, actuator, mode)

    def _create_mocap_base(self, worldbody: ET.Element) -> ET.Element:
        """Create mocap base for robot mounting.

        The backbone is built along the world +Z axis, so the mocap base
        carries no rotation. (Franka-mounted scenes are assembled by
        generate.py, which re-parents the chain under the flange.)
        """
        mocap_body = ET.SubElement(
            worldbody,
            "body",
            attrib={
                "name": "mocap_base",
                "pos": "0 0 0",
                "mocap": "true",
                "euler": "0 0 0",  # backbone built along +Z; no base rotation needed
            },
        )

        # No base visualization geom - removed for cleaner visualization
        # Users can add a site or geom in the config if needed

        # Return mocap_body directly - link_0 will be the first link
        return mocap_body

    def _chain_direct(self, base_body: ET.Element, actuator: ET.Element):
        """Body chain for direct-torque actuation (mode='direct').

        Convention: total_links bodies + a separate `link_end` half-link + an
        `EE_pos` tip body; joints actuated directly (no tendons); 2-joint names
        joint_{i+1}/joint_{i+1}_vert, 3-joint joint_{i}_{axis}.
        """
        curr_body = base_body

        # Calculate positions for force application sites
        if self.config.joints_per_link == 3:
            # For 3-joint config: total length includes start, middle links, and end
            # start: link_length/2, middle: (N-1)*link_length, end: link_length/2
            total_robot_length = self.link_length * self.config.total_links
            midpoint_position = total_robot_length / 2

            # Track cumulative position along the robot
            cumulative_pos = self.link_length / 2  # Start link extends link_length/2
        else:
            # For 2-joint config: calculate based on segment lengths
            total_robot_length = sum(
                float(self.config.segment_lengths[str(i + 1)])
                for i in range(self.config.num_segments)
            )
            midpoint_position = total_robot_length / 2
            cumulative_pos = 0

        # Flags to track if sites have been added
        midpoint_site_added = False

        # Both joint configurations: the loop creates N-1 links and link_end is
        # appended below, so the tip (EE_pos) lands exactly at total_length
        # (link_0 spans a half link, link_end spans the final half link).
        num_links_to_create = self.config.total_links - 1
        for i in range(num_links_to_create):
            # Uniform link spacing (first link is a half-link), matching the
            # capsule geom length (fromto below) and the tendon-actuated path.
            # Previously the 2-joint branch used per-segment spacing
            # (prev_seg_length / prev_num_links) which (a) disagreed with the
            # global geom length, overlapping/gapping links on non-uniform
            # configs, and (b) selected the wrong segment at boundaries via
            # get_segment_from_link(i - 1).
            pos = f"0 0 {self.link_length / 2}" if i == 0 else f"0 0 {self.link_length}"
            link_name = f"link_{i}"

            # Create body
            curr_body = ET.SubElement(
                curr_body, "body", attrib={"name": link_name, "pos": pos}
            )
            self.chain_body_names.append(link_name)

            # Update cumulative position and check if we should add midpoint site (same for both)
            cumulative_pos += self.link_length
            if not midpoint_site_added and cumulative_pos >= midpoint_position:
                # Calculate exact position within this link
                offset_in_link = midpoint_position - (cumulative_pos - self.link_length)
                ET.SubElement(
                    curr_body,
                    "site",
                    attrib={
                        "name": "force_site_mid",
                        "pos": f"0 0 {offset_in_link}",
                        "size": "0.003",
                        "rgba": "0 1 0 0.8",
                        "type": "sphere",
                    },
                )
                midpoint_site_added = True

            # Determine color based on segment - use alternating colors
            if self.config.joints_per_link == 3:
                color = self.default_colors[i % 2]
            else:
                # Use alternating dark/light grey for any segment configuration
                segment_num = (
                    self.config.get_segment_from_link(i) - 1
                )  # Convert to 0-based index
                # Alternate between dark grey (index 0) and light grey (last index)
                if segment_num % 2 == 0:
                    color = self.segment_colours[0]  # Dark grey
                else:
                    color = self.segment_colours[-1]  # Light grey

            # Capsule link, uniform length; name differs by joint config.
            self._emit_link_geom(
                curr_body,
                length=self.link_length,
                mass=self.link_mass,
                color=color,
                gtype="capsule",
                name=(
                    f"geom_link_{i+1}"
                    if self.config.joints_per_link == 2
                    else f"geom_{i}"
                ),
            )

            # Add joints
            if self.config.joints_per_link == 3:
                # 3-joint configuration
                for axis in ["x", "y", "z"]:
                    joint_name = f"joint_{i}_{axis}"
                    ET.SubElement(
                        curr_body,
                        "joint",
                        attrib={"class": f"{axis}_joint", "name": joint_name},
                    )
                    # Add actuator if not passive
                    if self.config.actuation_mode != "none":
                        self._add_actuator(
                            actuator,
                            joint_name,
                            f"act_{i}_{axis}",
                            self.config.joint_range,
                        )
            else:
                # 2-joint configuration
                joint_name = f"joint_{i+1}"
                self.joint_names.append(joint_name)

                # Planar joint
                ET.SubElement(
                    curr_body,
                    "joint",
                    attrib={"class": "planar_link", "name": joint_name},
                )
                if self.config.actuation_mode != "none":
                    self._add_actuator(
                        actuator, joint_name, f"act_{i+1}", self.config.joint_range
                    )

                # Vertical joint
                vert_joint_name = f"{joint_name}_vert"
                ET.SubElement(
                    curr_body,
                    "joint",
                    attrib={"class": "vertical_link", "name": vert_joint_name},
                )
                if self.config.actuation_mode != "none":
                    self._add_actuator(
                        actuator,
                        vert_joint_name,
                        f"act_{i+1}_vert",
                        self.config.vert_joint_range,
                    )

        # Add end link (same for both configurations)
        end_body = ET.SubElement(
            curr_body,
            "body",
            attrib={"name": "link_end", "pos": f"0 0 {self.link_length}"},
        )
        self.chain_body_names.append("link_end")
        end_link_length = self.link_length / 2

        # End link color should match last segment pattern
        if self.config.joints_per_link == 3:
            end_color = self.default_colors[1]  # Alternate pattern for 3-joint
        else:
            # Use same alternating pattern based on last segment
            last_segment = self.config.num_segments - 1
            if last_segment % 2 == 0:
                end_color = self.segment_colours[0]  # Dark grey
            else:
                end_color = self.segment_colours[-1]  # Light grey
        self._emit_link_geom(
            end_body,
            length=end_link_length,
            mass=self.link_mass / 2,
            color=end_color,
            gtype="cylinder",
            name="geom_end",
        )

        # Add joints for end link
        if self.config.joints_per_link == 3:
            for axis in ["x", "y", "z"]:
                joint_name = f"joint_end_{axis}"
                ET.SubElement(
                    end_body,
                    "joint",
                    attrib={"class": f"{axis}_joint", "name": joint_name},
                )
        else:
            # 2-joint config end link joints with 3-joint style naming
            ET.SubElement(
                end_body,
                "joint",
                attrib={"class": "planar_link", "name": "joint_end_x"},
            )
            ET.SubElement(
                end_body,
                "joint",
                attrib={"class": "vertical_link", "name": "joint_end_z"},
            )

        # End effector marker (same for both configurations)
        ee_pos = f"0 0 {end_link_length}"

        ee_body = ET.SubElement(
            end_body, "body", attrib={"name": "EE_pos", "pos": ee_pos}
        )

        # Add force application site at tip
        ET.SubElement(
            ee_body,
            "site",
            attrib={
                "name": "force_site_tip",
                "pos": "0 0 0",
                "size": "0.003",
                "rgba": "1 0 0 0.8",
                "type": "sphere",
            },
        )

    def _build_body_chain(
        self,
        base_body: ET.Element,
        mujoco: ET.Element,
        actuator: ET.Element,
        mode: str,
    ):
        """Single entry point for building the robot body chain.

        ``mode`` is one of ``'direct'``, ``'modular'``, ``'tendon'``. Material
        math is shared via ``compute_link_properties``; the per-mode body/joint/
        site conventions (half-link placement, joint axes and naming, tendon
        routing) genuinely differ and are reproduced by the mode-specific
        routines below, locked by the golden snapshot tests.

        A standard config with ``material_properties_per_segment`` is rewritten
        in __init__ as a one-module-per-segment modular robot, so multi-material
        and modular share the ``modular`` path (rigid per-section chains).
        """
        if mode == "modular":
            self._chain_modular(base_body, mujoco, actuator)
        elif mode == "direct":
            self._chain_direct(base_body, actuator)
        else:
            self._chain_tendon(base_body, mujoco, actuator)

    def _chain_modular(
        self, base_body: ET.Element, mujoco: ET.Element, actuator: ET.Element
    ):
        """Body chain for modular tendon robots (mode='modular').

        Convention: one half-link at every module boundary, per-module geometry
        and explicit per-joint stiffness/damping, rigid joins (no joint) at
        module/segment boundaries, tendon sites at module boundaries, EE_pos tip.
        """
        # Setup tendons
        tendon_elem = ET.SubElement(mujoco, "tendon")
        all_tendon_locations, spatial = self._setup_tendons_and_sites(
            tendon_elem, actuator
        )

        # Store tendon info for use in end link creation
        self._tendon_info = (spatial, all_tendon_locations)

        # Create robot with modular structure
        curr_body = base_body
        global_link_idx = 0  # Global link counter across all modules
        cumulative_pos = 0  # Track position for force sites

        # Calculate midpoint for force site
        total_robot_length = self.config.total_length
        midpoint_position = total_robot_length / 2
        midpoint_site_added = False

        # Iterate through segments and modules
        prev_link_length = 0  # Track previous link length for positioning
        for seg_idx, seg_def in enumerate(self.config.modular_segments):
            for module_idx, mod_id in enumerate(seg_def["module_ids"]):
                module = self.config.module_library[mod_id]

                # Create links for this module
                # First link of each module is half-length, last is half-length
                for link_in_module in range(module.num_links + 1):
                    # Determine position
                    if global_link_idx == 0:
                        # Very first link at origin
                        pos = "0 0 0"
                        prev_link_length = module.link_length / 2
                    else:
                        # All subsequent links positioned relative to previous
                        pos = f"0 0 {prev_link_length}"
                        # Update prev_link_length for next iteration
                        if link_in_module == 0:
                            prev_link_length = (
                                module.link_length / 2
                            )  # Half-length start
                        elif link_in_module == module.num_links:
                            prev_link_length = module.link_length / 2  # Half-length end
                        else:
                            prev_link_length = module.link_length  # Full-length middle

                    link_name = f"link_{global_link_idx}"

                    curr_body = ET.SubElement(
                        curr_body, "body", attrib={"name": link_name, "pos": pos}
                    )
                    self.chain_body_names.append(link_name)

                    # Update cumulative position for force sites
                    if link_in_module == 0:
                        cumulative_pos += module.link_length / 2
                    else:
                        cumulative_pos += prev_link_length

                    # Check if midpoint site should be added
                    if not midpoint_site_added and cumulative_pos >= midpoint_position:
                        offset_in_link = midpoint_position - (
                            cumulative_pos - prev_link_length
                        )
                        ET.SubElement(
                            curr_body,
                            "site",
                            attrib={
                                "name": "force_site_mid",
                                "pos": f"0 0 {offset_in_link}",
                                "size": "0.003",
                                "rgba": "0 1 0 0.8",
                                "type": "sphere",
                            },
                        )
                        midpoint_site_added = True

                    # Determine color - use module color if specified
                    if module.color:
                        color = module.color
                    else:
                        # Fall back to gradient based on segment
                        if seg_idx % 2 == 0:
                            color = self.segment_colours[0]
                        else:
                            color = self.segment_colours[-1]

                    # Determine link length (first and last are half)
                    if link_in_module == 0 or link_in_module == module.num_links:
                        geom_len = module.link_length / 2
                        link_mass = module.link_mass / 2
                    else:
                        geom_len = module.link_length
                        link_mass = module.link_mass
                    self._emit_link_geom(
                        curr_body,
                        length=geom_len,
                        mass=link_mass,
                        color=color,
                        name=f"geom_{global_link_idx}",
                        radius=module.radius,
                    )

                    # Add tendon routing sites at module boundaries
                    # For base and tip links (half-length), add sites at both boundary and midpoint
                    is_module_start = link_in_module == 0
                    is_module_end = link_in_module == module.num_links

                    if is_module_start or is_module_end:
                        # For this module's segment (seg_idx), add sites to all segment tendons
                        # that include this segment (seg_idx and all later segments)
                        for s in range(seg_idx, self.config.num_segments):
                            # This module is part of segment s's tendon path
                            for tendon_idx, tendon_obj in enumerate(spatial[s]):
                                site_loc = all_tendon_locations[s][tendon_idx]

                                if is_module_start:
                                    # Base link of module - add sites at boundary (0) and midpoint
                                    # Site 1: At the boundary (origin)
                                    site_name_boundary = f"mod_{seg_idx}_{module_idx}_base_boundary_seg_{s}_ten_{tendon_idx}"
                                    site_pos_boundary = (
                                        f"{site_loc[0]} {site_loc[1]} 0.0"
                                    )
                                    ET.SubElement(
                                        curr_body,
                                        "site",
                                        attrib={
                                            "name": site_name_boundary,
                                            "pos": site_pos_boundary,
                                            "rgba": "0 0 0 0",
                                        },
                                    )
                                    ET.SubElement(
                                        tendon_obj,
                                        "site",
                                        attrib={"site": site_name_boundary},
                                    )

                                    # Site 2: At the midpoint of the half-length link
                                    site_name_mid = f"mod_{seg_idx}_{module_idx}_base_mid_seg_{s}_ten_{tendon_idx}"
                                    site_pos_mid = f"{site_loc[0]} {site_loc[1]} {module.link_length / 4}"
                                    ET.SubElement(
                                        curr_body,
                                        "site",
                                        attrib={
                                            "name": site_name_mid,
                                            "pos": site_pos_mid,
                                            "rgba": "0 0 0 0",
                                        },
                                    )
                                    ET.SubElement(
                                        tendon_obj,
                                        "site",
                                        attrib={"site": site_name_mid},
                                    )

                                else:  # is_module_end
                                    # Tip link of module - add sites at midpoint and boundary
                                    # Site 1: At the midpoint of the half-length link
                                    site_name_mid = f"mod_{seg_idx}_{module_idx}_tip_mid_seg_{s}_ten_{tendon_idx}"
                                    site_pos_mid = f"{site_loc[0]} {site_loc[1]} {module.link_length / 4}"
                                    ET.SubElement(
                                        curr_body,
                                        "site",
                                        attrib={
                                            "name": site_name_mid,
                                            "pos": site_pos_mid,
                                            "rgba": "0 0 0 0",
                                        },
                                    )
                                    ET.SubElement(
                                        tendon_obj,
                                        "site",
                                        attrib={"site": site_name_mid},
                                    )

                                    # Site 2: At the boundary (end of half-length link)
                                    site_name_boundary = f"mod_{seg_idx}_{module_idx}_tip_boundary_seg_{s}_ten_{tendon_idx}"
                                    site_pos_boundary = f"{site_loc[0]} {site_loc[1]} {module.link_length / 2}"
                                    ET.SubElement(
                                        curr_body,
                                        "site",
                                        attrib={
                                            "name": site_name_boundary,
                                            "pos": site_pos_boundary,
                                            "rgba": "0 0 0 0",
                                        },
                                    )
                                    ET.SubElement(
                                        tendon_obj,
                                        "site",
                                        attrib={"site": site_name_boundary},
                                    )

                    # Add joints (using module-specific stiffness and damping)
                    # Skip joints for module boundary links to create rigid connections between modules
                    # This includes boundaries within a segment AND boundaries between segments
                    is_first_link_of_non_first_module = (
                        module_idx > 0 and link_in_module == 0
                    )
                    is_last_link_of_non_last_module = (
                        module_idx < len(seg_def["module_ids"]) - 1
                        and link_in_module == module.num_links
                    )

                    # Also check for segment boundaries
                    is_first_link_of_non_first_segment = (
                        seg_idx > 0 and module_idx == 0 and link_in_module == 0
                    )
                    is_last_link_of_non_last_segment = (
                        seg_idx < self.config.num_segments - 1
                        and module_idx == len(seg_def["module_ids"]) - 1
                        and link_in_module == module.num_links
                    )

                    # Don't add joints at any module/segment boundaries
                    if (
                        is_first_link_of_non_first_module
                        or is_last_link_of_non_last_module
                        or is_first_link_of_non_first_segment
                        or is_last_link_of_non_last_segment
                    ):
                        # Skip adding joints - this creates a rigid connection between modules/segments
                        pass
                    elif self.config.joints_per_link == 2:
                        # Planar joint
                        joint_name = f"joint_{global_link_idx}_x"
                        ET.SubElement(
                            curr_body,
                            "joint",
                            attrib={
                                "name": joint_name,
                                "type": "hinge",
                                "axis": "0 1 0",
                                "limited": str(self.config.joint_limited).lower(),
                                "range": self.config.joint_range,
                                "stiffness": str(module.joint_stiffness),
                                "damping": str(module.joint_damping),
                            },
                        )

                        # Vertical joint (second bending axis — EI-based, like the
                        # standard 2-joint path; torsion is unmodeled in 2-joint mode)
                        vert_joint_name = f"joint_{global_link_idx}_z"
                        ET.SubElement(
                            curr_body,
                            "joint",
                            attrib={
                                "name": vert_joint_name,
                                "type": "hinge",
                                "axis": "1 0 0",
                                "limited": str(self.config.joint_limited).lower(),
                                "range": self.config.vert_joint_range,
                                "stiffness": str(module.joint_stiffness),
                                "damping": str(module.joint_damping),
                            },
                        )
                    else:
                        # 3-joint configuration. Matches the standard-path classes:
                        # joint_*_x = axis "0 0 1" (backbone twist, GJ-based torsion
                        # stiffness); joint_*_y/_z = bending axes (EI-based).
                        for axis_name, axis_vec in [
                            ("x", "0 0 1"),
                            ("y", "0 1 0"),
                            ("z", "1 0 0"),
                        ]:
                            joint_name = f"joint_{global_link_idx}_{axis_name}"
                            is_torsion = axis_name == "x"
                            stiff = (
                                module.torsion_stiffness
                                if is_torsion
                                else module.joint_stiffness
                            )
                            damp = (
                                module.torsion_damping
                                if is_torsion
                                else module.joint_damping
                            )
                            ET.SubElement(
                                curr_body,
                                "joint",
                                attrib={
                                    "name": joint_name,
                                    "type": "hinge",
                                    "axis": axis_vec,
                                    "limited": str(self.config.joint_limited).lower(),
                                    "range": self.config.joint_range,
                                    "stiffness": str(stiff),
                                    "damping": str(damp),
                                },
                            )

                    global_link_idx += 1

        # Add end effector marker
        ee_pos = f"0 0 {prev_link_length}"
        ee_body = ET.SubElement(
            curr_body, "body", attrib={"name": "EE_pos", "pos": ee_pos}
        )
        ET.SubElement(
            ee_body,
            "site",
            attrib={
                "name": "force_site_tip",
                "pos": "0 0 0",
                "size": "0.003",
                "rgba": "1 0 0 0.8",
                "type": "sphere",
            },
        )

    def _global_tendon_index(self, s: int, n: int) -> int:
        """Flat index of tendon ``n`` of segment ``s`` across all segments.

        Handles heterogeneous per-segment tendon counts (e.g. [3, 4, 3]); a flat
        ``s * num_tendons + n`` mis-maps whenever the counts differ.
        """
        prior = sum(
            self.config.actuation_details["segments"][k]["number_of_tendons"]
            for k in range(s)
        )
        return prior + n

    def _tendon_actual_kp(self, s: int, n: int) -> float:
        """Position-actuator stiffness (kp) for tendon ``n`` of segment ``s``.

        A per-tendon ``tendon_kp_array`` (sysid output) takes precedence, indexed
        by the flat tendon index; otherwise ``tendon_kp`` with optional
        ``inverse_length`` scaling (lower-segment / shorter tendons stiffer).
        Used by both the actuator and the pretension keyframe so they agree.
        """
        props = self.config.actuator_properties
        base_kp = props.get("tendon_kp", 1000)
        kp_array = props.get("tendon_kp_array", None)
        if isinstance(kp_array, (list, tuple)) and kp_array:
            idx = self._global_tendon_index(s, n)
            return kp_array[idx] if idx < len(kp_array) else kp_array[-1]
        if props.get("tendon_kp_scaling", "none") == "inverse_length":
            scale_factor = props.get("tendon_kp_scale_factor", 1.0)
            if self.config.num_segments > 1:
                kp_multiplier = 1.0 + (scale_factor - 1.0) * (
                    self.config.num_segments - 1 - s
                ) / (self.config.num_segments - 1)
            else:
                kp_multiplier = 1.0
            return base_kp * kp_multiplier
        return base_kp

    def _remap_axial_to_arc(self, u: float) -> float:
        """Map a uniform-layout axial coordinate onto the per-segment arc-length.

        Tendon sites are first positioned in the uniform body layout (every link
        is ``self.link_length`` long). ``u`` is such an axial coordinate. This
        maps it piecewise-linearly so that each segment's uniform span
        ``[cum_links*link_length]`` is stretched/compressed onto that segment's
        configured ``segment_lengths``. Segment boundaries therefore land at the
        cumulative segment lengths, so each tendon spans exactly its segment
        length(s) no matter how many links each segment has. The map is
        monotonic, and the identity when segment lengths are proportional to the
        per-segment link counts.
        """
        ll = self.link_length
        u_start = 0.0  # uniform axial coord at the start of the current segment
        arc_start = 0.0  # configured arc-length at the start of the current segment
        cum_links = 0
        for s in range(self.config.num_segments):
            cum_links += self.config.links_per_segment[str(s + 1)]
            u_end = cum_links * ll
            arc_end = arc_start + float(self.config.segment_lengths[str(s + 1)])
            # Use this segment's mapping for coords up to its end (or the last
            # segment, which absorbs any tip remainder).
            if u <= u_end or s == self.config.num_segments - 1:
                span = u_end - u_start
                frac = 0.0 if span <= 0 else (u - u_start) / span
                return arc_start + frac * (arc_end - arc_start)
            u_start, arc_start = u_end, arc_end
        return arc_start

    def _chain_tendon(
        self, base_body: ET.Element, mujoco: ET.Element, actuator: ET.Element
    ):
        """Body chain for standard (single-material) tendon robots (mode='tendon').

        Convention: total_links+1 bodies with half-links only at the two global
        ends, uniform link spacing and stiffness, continuous bending across
        segment boundaries, tendon sites arc-length-remapped, class-ref joints
        (joint_{i}_x/_z). Per-segment material instead goes through the modular
        path (see __init__'s material_properties_per_segment synthesis).
        """
        # Setup tendons
        tendon_elem = ET.SubElement(mujoco, "tendon")
        all_tendon_locations, spatial = self._setup_tendons_and_sites(
            tendon_elem, actuator
        )

        # Store tendon info for use in end link creation
        self._tendon_info = (spatial, all_tendon_locations)

        # Create robot with tendon routing
        curr_body = base_body

        # Calculate positions for force application sites
        if self.config.joints_per_link == 3:
            # For 3-joint config: total length includes start, middle links, and end
            total_robot_length = self.link_length * self.config.total_links
            midpoint_position = total_robot_length / 2
            cumulative_pos = self.link_length / 2  # Start link extends link_length/2
        else:
            # For 2-joint config: calculate based on segment lengths
            total_robot_length = sum(
                float(self.config.segment_lengths[str(i + 1)])
                for i in range(self.config.num_segments)
            )
            midpoint_position = total_robot_length / 2
            cumulative_pos = 0

        # Flag to track if midpoint site has been added
        midpoint_site_added = False

        # Create all links including first half-length and last half-length
        # The total number of physical links is total_links + 1 to account for half-links at boundaries
        num_links_to_create = self.config.total_links + 1
        for i in range(num_links_to_create):
            # First link (link_0) is at origin, others follow
            if i == 0:
                pos = "0 0 0"
            else:
                # Previous link length determines position
                # Link 0 is half-length, last link will be half-length, others are full
                if i == 1:
                    # After first half-length link
                    prev_link_length = self.link_length / 2
                else:
                    # After full-length links
                    prev_link_length = self.link_length
                pos = f"0 0 {prev_link_length}"

            link_name = f"link_{i}"

            curr_body = ET.SubElement(
                curr_body, "body", attrib={"name": link_name, "pos": pos}
            )
            self.chain_body_names.append(link_name)

            # Determine color based on segment - use alternating colors
            if self.config.joints_per_link == 3:
                color = self.default_colors[i % 2]
            else:
                # Use alternating dark/light grey for any segment configuration
                segment_num = (
                    self.config.get_segment_from_link(i) - 1
                )  # Convert to 0-based index
                # Alternate between dark grey (index 0) and light grey (last index)
                if segment_num % 2 == 0:
                    color = self.segment_colours[0]  # Dark grey
                else:
                    color = self.segment_colours[-1]  # Light grey

            # First and last links are half-length (and half-mass), others are full
            is_half_link = i == 0 or i == num_links_to_create - 1
            geom_len = self.link_length / 2 if is_half_link else self.link_length
            link_mass = self.link_mass / 2 if is_half_link else self.link_mass
            self._emit_link_geom(
                curr_body,
                length=geom_len,
                mass=link_mass,
                color=color,
                name=f"geom_{i}",
            )

            # Add joints based on configuration (use 3-joint naming style for
            # consistency). link_0 is the clamped base half-link and gets no
            # joints: with joints on links 1..N the chain has exactly
            # total_links joint stations of stiffness n*EI/L (n = total_links),
            # so the series compliance is exactly L/EI — matching the direct
            # chain, whose first joint also pivots at z = link_length/2.
            if i == 0:
                pass
            elif self.config.joints_per_link == 3:
                # 3-joint configuration
                for axis in ["x", "y", "z"]:
                    joint_name = f"joint_{i}_{axis}"
                    ET.SubElement(
                        curr_body,
                        "joint",
                        attrib={"class": f"{axis}_joint", "name": joint_name},
                    )
            else:
                # 2-joint configuration with 3-joint style naming
                joint_name = f"joint_{i}"
                self.joint_names.append(joint_name)
                ET.SubElement(
                    curr_body,
                    "joint",
                    attrib={"class": "planar_link", "name": f"{joint_name}_x"},
                )
                ET.SubElement(
                    curr_body,
                    "joint",
                    attrib={"class": "vertical_link", "name": f"{joint_name}_z"},
                )

            # Update cumulative position and check if we should add midpoint site (same for both)
            cumulative_pos += self.link_length
            if not midpoint_site_added and cumulative_pos >= midpoint_position:
                # Calculate exact position within this link
                offset_in_link = midpoint_position - (cumulative_pos - self.link_length)
                ET.SubElement(
                    curr_body,
                    "site",
                    attrib={
                        "name": "force_site_mid",
                        "pos": f"0 0 {offset_in_link}",
                        "size": "0.003",
                        "rgba": "0 1 0 0.8",
                        "type": "sphere",
                    },
                )
                midpoint_site_added = True

            # Add tendon sites
            # Determine which tendons pass through this link
            # Standard mode: A tendon for segment S passes through all links from 0 to the last link of segment S
            # Independent mode: A tendon for segment S only passes through segment S's own links
            cumulative_links = 0
            segment_start_links = []  # Track where each segment starts
            for s in range(self.config.num_segments):
                segment_start_link = cumulative_links
                segment_start_links.append(segment_start_link)
                cumulative_links += self.config.links_per_segment[str(s + 1)]
                # The physical link index where segment s ends
                # For example with 10 links per segment:
                # Segment 0: links 0-10, Segment 1: links 11-20, Segment 2: links 21-30
                segment_end_link = cumulative_links

                # Check if this link should have tendon sites for segment s
                if self.config.independent_segments:
                    # Independent mode: only add sites if link is within this segment's range
                    link_in_segment = segment_start_link <= i <= segment_end_link
                else:
                    # Standard coupled mode: add sites if link is before or in this segment
                    link_in_segment = i <= segment_end_link

                if link_in_segment:
                    # This link is part of segment s's tendon path.
                    #
                    # Sites are placed in the uniform body layout (constant
                    # link_length) and then remapped onto the per-segment
                    # arc-length via _remap_axial_to_arc, so each segment's tendon
                    # spans exactly its configured segment_lengths and the segment
                    # boundaries land at the cumulative segment lengths -
                    # regardless of how many links each segment has. The global
                    # axial coord of link i's body origin is
                    #   link 0 -> 0,   link i>=1 -> link_length * (i - 0.5)
                    # (link 0 is a half-link, the rest are full links).
                    body_origin_x = 0.0 if i == 0 else self.link_length * (i - 0.5)

                    for tendon_idx, tendon_obj in enumerate(spatial[s]):
                        site_loc = all_tendon_locations[s][tendon_idx]

                        # Get tendon constraint factor (default to 1.0 if not specified)
                        constraint_factor = self.config.actuator_properties.get(
                            "tendon_constraint_factor", 1.0
                        )

                        # Determine link length based on position
                        if i == 0 or i == num_links_to_create - 1:
                            # Half-length links at boundaries
                            link_length = self.link_length / 2
                        else:
                            # Full-length links in the middle
                            link_length = self.link_length

                        if constraint_factor == 0:
                            # Original behavior: single site at midpoint
                            if i == 0:
                                # First link - add site at base (origin) only
                                base_site_name = (
                                    f"l_{i}_base_seg_{s}_tendon_{tendon_idx}"
                                )
                                base_site_pos = f"{site_loc[0]} {site_loc[1]} 0"
                                ET.SubElement(
                                    curr_body,
                                    "site",
                                    attrib={
                                        "name": base_site_name,
                                        "pos": base_site_pos,
                                        "rgba": "0 0 0 0",
                                    },
                                )
                                ET.SubElement(
                                    tendon_obj, "site", attrib={"site": base_site_name}
                                )
                            else:
                                # Midpoint of the link (the last link of the last
                                # segment uses its end so the tendon reaches the
                                # tip), then remapped onto the per-segment
                                # arc-length so segment boundaries land exactly at
                                # the cumulative segment lengths.
                                if (
                                    i == num_links_to_create - 1
                                    and s == self.config.num_segments - 1
                                    and i == segment_end_link
                                ):
                                    site_x_pos = link_length
                                else:
                                    site_x_pos = link_length / 2
                                site_x_pos = (
                                    self._remap_axial_to_arc(body_origin_x + site_x_pos)
                                    - body_origin_x
                                )

                                site_name = f"l_{i}_seg_{s}_tendon_{tendon_idx}"
                                site_pos = f"{site_loc[0]} {site_loc[1]} {site_x_pos}"
                                ET.SubElement(
                                    curr_body,
                                    "site",
                                    attrib={
                                        "name": site_name,
                                        "pos": site_pos,
                                        "rgba": "0 0 0 0",
                                    },
                                )
                                ET.SubElement(
                                    tendon_obj, "site", attrib={"site": site_name}
                                )
                        else:
                            # New behavior: two sites per link based on constraint factor
                            # Calculate site positions based on constraint factor
                            # Factor of 0: both at midpoint (handled above)
                            # Factor of 1: at base and tip
                            # Factor between 0 and 1: interpolate symmetrically about midpoint

                            mid_point = link_length / 2
                            offset = (link_length / 2) * constraint_factor

                            if i == 0:
                                # First half-length link: one site at base, one offset from base
                                site1_x = 0
                                site2_x = min(
                                    offset, link_length
                                )  # Don't exceed link length
                            elif i == num_links_to_create - 1:
                                # Last half-length link: one offset from start, one at end
                                site1_x = max(
                                    0, link_length - offset
                                )  # Don't go negative
                                site2_x = link_length
                            elif i == segment_end_link and i < num_links_to_create - 1:
                                # Last link for this segment (not the last overall):
                                # keep the second site at the midpoint, which is the
                                # segment boundary after remapping.
                                site1_x = mid_point - offset
                                site2_x = mid_point  # Keep at midpoint, not extended
                            else:
                                # Full-length links: two sites symmetric about midpoint
                                site1_x = mid_point - offset
                                site2_x = mid_point + offset

                            # Remap both sites onto the configured per-segment arc-length.
                            site1_x = (
                                self._remap_axial_to_arc(body_origin_x + site1_x)
                                - body_origin_x
                            )
                            site2_x = (
                                self._remap_axial_to_arc(body_origin_x + site2_x)
                                - body_origin_x
                            )

                            # Add first site
                            site1_name = f"l_{i}_s1_seg_{s}_tendon_{tendon_idx}"
                            site1_pos = f"{site_loc[0]} {site_loc[1]} {site1_x}"
                            ET.SubElement(
                                curr_body,
                                "site",
                                attrib={
                                    "name": site1_name,
                                    "pos": site1_pos,
                                    "rgba": "0 0 0 0",
                                },
                            )
                            ET.SubElement(
                                tendon_obj, "site", attrib={"site": site1_name}
                            )

                            # Add second site
                            site2_name = f"l_{i}_s2_seg_{s}_tendon_{tendon_idx}"
                            site2_pos = f"{site_loc[0]} {site_loc[1]} {site2_x}"
                            ET.SubElement(
                                curr_body,
                                "site",
                                attrib={
                                    "name": site2_name,
                                    "pos": site2_pos,
                                    "rgba": "0 0 0 0",
                                },
                            )
                            ET.SubElement(
                                tendon_obj, "site", attrib={"site": site2_name}
                            )

        # End-effector marker at the tip of the last half-link, matching the
        # direct and modular chains so downstream code (evaluation, IK
        # controllers, sysid) can rely on EE_pos / force_site_tip existing.
        ee_body = ET.SubElement(
            curr_body,
            "body",
            attrib={"name": "EE_pos", "pos": f"0 0 {self.link_length / 2}"},
        )
        ET.SubElement(
            ee_body,
            "site",
            attrib={
                "name": "force_site_tip",
                "pos": "0 0 0",
                "size": "0.003",
                "rgba": "1 0 0 0.8",
                "type": "sphere",
            },
        )

        # Add fixed tendons with springlength deadband for joint deadband
        self._add_joint_deadband_tendons(tendon_elem)

    def _setup_tendons_and_sites(
        self, tendon: ET.Element, actuator: ET.Element
    ) -> Tuple[List, List]:
        """Setup tendons and their routing sites."""
        all_tendon_locations = []
        spatial = []

        for s in range(self.config.num_segments):
            curr_seg_tendons = []
            num_tendons = self.config.actuation_details["segments"][s][
                "number_of_tendons"
            ]

            # Calculate tendon positions
            theta = 2 * math.pi / num_tendons
            distance_to_backbone = self.config.actuation_details["segments"][s][
                "distance_to_backbone"
            ]

            # Compute tendon angles: base spacing + segment offset + per-tendon delta
            off = self.config.seg_offsets[s]
            deltas = (
                self.config.tendon_angle_deltas[s]
                if self.config.tendon_angle_deltas
                and s < len(self.config.tendon_angle_deltas)
                else [0.0] * num_tendons
            )
            tendon_locations = [
                (
                    distance_to_backbone * math.cos(off + i * theta + deltas[i]),
                    distance_to_backbone * math.sin(off + i * theta + deltas[i]),
                )
                for i in range(num_tendons)
            ]
            all_tendon_locations.append(tendon_locations)

            # Create spatial elements and actuators
            for n in range(num_tendons):
                tcolor = self.tendon_colours[s % len(self.tendon_colours)]
                tendon_spatial_attrib = {
                    "name": f"seg_{s}_tendon_{n}",
                    "rgba": tcolor,
                    "width": "0.001",
                }
                if self.config.tendon_frictionloss > 0:
                    tendon_spatial_attrib["frictionloss"] = str(
                        self.config.tendon_frictionloss
                    )
                spatial_elem = ET.SubElement(
                    tendon,
                    "spatial",
                    attrib=tendon_spatial_attrib,
                )
                curr_seg_tendons.append(spatial_elem)

                # Add tendon actuator if not passive
                if self.config.actuation_mode != "none":
                    # Calculate tendon rest length
                    # In independent mode: only this segment's length
                    # In coupled mode: cumulative length from segment 0 to s
                    if self.config.independent_segments:
                        rest_length = self.config.segment_lengths[str(s + 1)]
                    else:
                        # The tendon runs parallel to the robot backbone
                        # Its length is simply the sum of segment lengths it spans
                        rest_length = 0.0
                        for seg in range(s + 1):
                            rest_length += self.config.segment_lengths[str(seg + 1)]

                    # Get the control range span from config
                    ctrl_range_str = self.config.actuator_properties.get(
                        "tendon_ctrlrange", "-0.05 0.05"
                    )
                    ctrl_min, ctrl_max = map(float, ctrl_range_str.split())

                    # Adaptive control range based on the tendon's actual kp
                    # (per-tendon kp_array if present, else base_kp with optional
                    # inverse_length scaling). Stiffer tendons get a finer range.
                    base_kp = self.config.actuator_properties.get("tendon_kp", 1000)
                    actual_kp = self._tendon_actual_kp(s, n)

                    range_scale = base_kp / actual_kp if actual_kp > 0 else 1.0
                    scaled_min = ctrl_min * range_scale
                    scaled_max = ctrl_max * range_scale

                    # Set control range centered at rest length
                    actual_min = rest_length + scaled_min
                    actual_max = rest_length + scaled_max

                    # Handle tendon actuator type
                    tendon_actuator_type = self.config.actuator_properties.get(
                        "tendon_actuator_type", "motor"
                    )

                    # Build base attributes (common to all types)
                    tendon_attrib = {
                        "name": f"seg_{s}_ten_{n}",
                        "tendon": f"seg_{s}_tendon_{n}",
                    }

                    if tendon_actuator_type == "motor":
                        # Motor actuators use force control, not position control
                        # They don't need ctrlrange for length - they apply forces
                        # Add custom motor properties from config
                        for key, value in self.config.actuator_properties.items():
                            if key.startswith("tendon_") and key not in [
                                "tendon_actuator_type",
                                "tendon_kp",
                                "tendon_kp_array",
                                "tendon_kp_scaling",
                                "tendon_kp_scale_factor",
                                "tendon_pretension",
                                "tendon_constraint_factor",
                                "tendon_ctrlrange",  # Not used for motor
                            ]:
                                actual_key = key.replace("tendon_", "")
                                tendon_attrib[actual_key] = str(value)

                    elif tendon_actuator_type == "position":
                        # Position actuators use length control with ctrlrange.
                        # actual_kp (computed above via _tendon_actual_kp) already
                        # respects the per-tendon kp_array / inverse_length scaling,
                        # so the ctrlrange and the kp are derived from the same kp.
                        tendon_attrib["ctrlrange"] = f"{actual_min} {actual_max}"
                        tendon_attrib.update({"kp": str(actual_kp)})

                        # Add custom position actuator properties
                        for key, value in self.config.actuator_properties.items():
                            if key.startswith("tendon_") and key not in [
                                "tendon_ctrlrange",
                                "tendon_kp",
                                "tendon_kp_array",
                                "tendon_actuator_type",
                                "tendon_kp_scaling",
                                "tendon_kp_scale_factor",
                                "tendon_pretension",
                                "tendon_constraint_factor",
                            ]:
                                actual_key = key.replace("tendon_", "")
                                tendon_attrib[actual_key] = str(value)

                    else:
                        # For other actuator types (velocity, etc.), add ctrlrange and custom properties
                        tendon_attrib["ctrlrange"] = f"{actual_min} {actual_max}"

                        for key, value in self.config.actuator_properties.items():
                            if key.startswith("tendon_") and key not in [
                                "tendon_ctrlrange",
                                "tendon_kp",
                                "tendon_kp_array",
                                "tendon_actuator_type",
                                "tendon_kp_scaling",
                                "tendon_kp_scale_factor",
                                "tendon_pretension",
                                "tendon_constraint_factor",
                            ]:
                                actual_key = key.replace("tendon_", "")
                                tendon_attrib[actual_key] = str(value)

                    ET.SubElement(actuator, tendon_actuator_type, attrib=tendon_attrib)

            spatial.append(curr_seg_tendons)

        return all_tendon_locations, spatial

    def _add_pretension_keyframe(self, mujoco: ET.Element):
        """Add keyframe for initial tendon control values and optional pre-tension."""
        keyframe_elem = ET.SubElement(mujoco, "keyframe")

        # Check actuator type
        tendon_actuator_type = self.config.actuator_properties.get(
            "tendon_actuator_type", "motor"
        )

        # Motor (force-control) actuators: the keyframe ctrl is a FORCE, not a
        # length. With no pretension configured the baseline is zero force (the
        # tension controller commands forces at runtime). If a pretension IS
        # explicitly set, apply it as a constant baseline tension force so it is
        # not silently dropped. Sign matches position mode: tension is a negative
        # tendon-actuator force.
        if tendon_actuator_type == "motor":
            pretension = self.config.actuator_properties.get("tendon_pretension", None)
            pre_array = pretension if isinstance(pretension, (list, tuple)) else None
            ctrl_values = []
            tendon_idx = 0
            for s in range(self.config.num_segments):
                num_tendons = self.config.actuation_details["segments"][s][
                    "number_of_tendons"
                ]
                for n in range(num_tendons):
                    if pretension is None:
                        force = 0.0
                    elif pre_array is not None:
                        force = (
                            pre_array[tendon_idx]
                            if tendon_idx < len(pre_array)
                            else pre_array[-1]
                        )
                    else:
                        force = pretension
                    ctrl_values.append(str(-force))  # negative force = tension
                    tendon_idx += 1

            ET.SubElement(
                keyframe_elem,
                "key",
                attrib={"name": "pretension", "ctrl": " ".join(ctrl_values)},
            )
            return

        # For position control: calculate control values for pre-tension
        pretension = self.config.actuator_properties.get("tendon_pretension", 1.0)

        # Support both scalar and array pretension values
        if isinstance(pretension, (list, tuple)):
            pretension_array = pretension
        else:
            pretension_array = None
            pretension_scalar = pretension

        ctrl_values = []
        tendon_idx = 0  # Global tendon index for array lookup

        for s in range(self.config.num_segments):
            num_tendons = self.config.actuation_details["segments"][s][
                "number_of_tendons"
            ]

            # Calculate rest length for this segment
            # In independent mode: only this segment's length
            # In coupled mode: cumulative length from segment 0 to s
            if self.config.independent_segments:
                rest_length = self.config.segment_lengths[str(s + 1)]
            else:
                rest_length = 0.0
                for seg in range(s + 1):
                    rest_length += self.config.segment_lengths[str(seg + 1)]

            # Get the control range for this segment
            ctrl_range_str = self.config.actuator_properties.get(
                "tendon_ctrlrange", "-0.05 0.05"
            )
            ctrl_min, ctrl_max = map(float, ctrl_range_str.split())

            # For each tendon in this segment
            for n in range(num_tendons):
                # Get pretension value for this specific tendon
                if pretension_array is not None:
                    if tendon_idx < len(pretension_array):
                        tendon_pretension = pretension_array[tendon_idx]
                    else:
                        tendon_pretension = pretension_array[
                            -1
                        ]  # Use last value as fallback
                else:
                    tendon_pretension = pretension_scalar

                # Actual kp for this tendon (per-tendon kp_array / inverse_length
                # scaling), matching the actuator via the shared helper.
                base_kp = self.config.actuator_properties.get("tendon_kp", 1000)
                actual_kp = self._tendon_actual_kp(s, n)

                # Use the SAME adaptive control-range scaling as the actuator
                # (base_kp / actual_kp) so the pretension ctrl stays inside the
                # actuator's ctrlrange instead of being silently clamped by MuJoCo.
                range_scale = base_kp / actual_kp if actual_kp > 0 else 1.0
                scaled_min = ctrl_min * range_scale
                scaled_max = ctrl_max * range_scale

                # The actual control range
                actual_min = rest_length + scaled_min
                actual_max = rest_length + scaled_max

                # Calculate length change for desired pretension
                # F = -kp * (length - ctrl)
                # For pretension force, we want F = -pretension (negative for tension)
                # So: -pretension = -kp * (length - ctrl)
                # pretension = kp * (length - ctrl)
                # ctrl = length - pretension/kp
                # Since rest position is at rest_length, we start there and reduce by pretension/kp
                if actual_kp > 0:
                    delta_length = tendon_pretension / actual_kp
                    ctrl_value = rest_length - delta_length
                else:
                    # kp=0 means no force, so pretension has no effect
                    ctrl_value = rest_length

                # Make sure we don't exceed the control range
                ctrl_value = max(actual_min, min(actual_max, ctrl_value))
                ctrl_values.append(str(ctrl_value))

                tendon_idx += 1

        # Create keyframe with all control values
        ET.SubElement(
            keyframe_elem,
            "key",
            attrib={"name": "pretension", "ctrl": " ".join(ctrl_values)},
        )

    def _add_collision_exclusions(self, mujoco: ET.Element):
        """Exclude collisions between each chain body and its 3 nearest neighbors.

        Adjacent capsules always overlap slightly at the joints; excluding the
        3-neighborhood prevents those permanent contacts from slowing down the
        simulation while still allowing genuine self-collision under sharp
        bending. Uses the body names recorded during chain construction, so it
        covers all chain modes (direct/tendon/modular) including the end link.
        """
        contact_elem = ET.SubElement(mujoco, "contact")

        names = self.chain_body_names
        for i in range(len(names)):
            for j in range(i + 1, min(len(names), i + 4)):
                ET.SubElement(
                    contact_elem,
                    "exclude",
                    attrib={"body1": names[i], "body2": names[j]},
                )

    def _add_axis_labels(self, worldbody: ET.Element, asset: ET.Element):
        """Add axis label visualization."""
        axis_body = ET.SubElement(
            worldbody, "body", attrib={"name": "axis_body", "pos": "-0.4 -0.1 -0.4"}
        )

        scale = "0.003"

        for axis_name, color, euler in [
            ["X", "1 0 0 1", "0 1.5708 -1.5708"],
            ["Y", "0 1 0 1", "-1.5708 0 0"],
            ["Z", "0 0 1 1", "0 0 -1.5708"],
        ]:
            ET.SubElement(
                asset,
                "mesh",
                attrib={
                    "name": f"{axis_name}axis_STL",
                    "content_type": "model/stl",
                    "file": f"MJCFS/STLS_dir/AxisArrowSTLS/AxisArrow{axis_name}.stl",
                    "scale": f"{scale} {scale} {scale}",
                },
            )

            arrow_body = ET.SubElement(
                axis_body,
                "body",
                attrib={
                    "name": f"{axis_name}_arrow_body",
                    "pos": "0 0 0.5",
                    "euler": euler,
                },
            )

            ET.SubElement(
                arrow_body,
                "geom",
                attrib={
                    "contype": "2",
                    "conaffinity": "2",
                    "type": "mesh",
                    "mesh": f"{axis_name}axis_STL",
                    "rgba": color,
                },
            )


def load_obstacles_from_file(
    filename: str,
) -> Tuple[List[ET.Element], List[Tuple[ET.Element, ET.Element]]]:
    """Load obstacles from taskspace configuration file."""

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    def process_single_stl(
        shape, ind, scale, x, y, z, alpha, beta, gamma, color, static, STLS
    ):
        name = f"{ind}_{os.path.basename(shape)}"
        asset_tag = ET.Element("asset")
        ET.SubElement(
            asset_tag,
            "mesh",
            attrib={
                "name": name,
                "content_type": "model/stl",
                "file": shape,
                "scale": f"{scale} {scale} {scale}",
            },
        )

        stl_bod = ET.Element(
            "body",
            attrib={
                "pos": f"{x} {y} {z}",
                "euler": f"{math.radians(alpha)} {math.radians(beta)} {math.radians(gamma)}",
            },
        )
        ET.SubElement(
            stl_bod, "geom", attrib={"type": "mesh", "mesh": name, "rgba": color}
        )

        if static == "False":
            ET.SubElement(
                stl_bod,
                "joint",
                attrib={
                    "name": f"obstacle_custom_{ind}_joint",
                    "type": "free",
                    "damping": "0.001",
                },
            )

        STLS.append((asset_tag, stl_bod))

    basic_shapes = ["Circle", "Square"]
    bodies = []
    STLS = []

    with open(filename) as f:
        for ind, line in enumerate(f.readlines()):
            parts = line.strip().split(" ")
            if len(parts) < 10:
                continue

            shape, scale, y, x, z, static, alpha, beta, gamma, color = parts[:10]

            # Convert color
            color_rgb = hex_to_rgb(color)
            color_str = f"{color_rgb[0]} {color_rgb[1]} {color_rgb[2]} 1"

            if shape in basic_shapes:
                body = ET.Element(
                    "body",
                    attrib={
                        "name": f"obstacle_custom_{ind}",
                        "pos": f"{x} {y} {z}",
                        "euler": f"{math.radians(float(alpha))} {math.radians(float(beta))} {math.radians(float(gamma))}",
                    },
                )

                if static == "False":
                    ET.SubElement(
                        body,
                        "joint",
                        attrib={
                            "name": f"obstacle_custom_{ind}_joint",
                            "type": "free",
                            "damping": "0.001",
                        },
                    )

                ET.SubElement(
                    body,
                    "geom",
                    attrib={
                        "name": f"obstacle_custom_{ind}_geom",
                        "size": f"{scale}",
                        "rgba": color_str,
                        "fromto": "0 0 0 0 0 0.2",
                        "type": "cylinder",
                        "friction": "2 1.5 1.5",
                        "mass": "0.001",
                    },
                )
                bodies.append(body)
            else:
                # Handle STL files
                if os.path.isfile(shape) and shape.lower().endswith(".stl"):
                    process_single_stl(
                        shape,
                        ind,
                        scale,
                        x,
                        y,
                        z,
                        alpha,
                        beta,
                        gamma,
                        color_str,
                        static,
                        STLS,
                    )
                elif os.path.isdir(shape):
                    stl_files = glob.glob(os.path.join(shape.rstrip("/"), "*.stl"))
                    for stl_file in stl_files:
                        process_single_stl(
                            stl_file,
                            ind,
                            scale,
                            x,
                            y,
                            z,
                            alpha,
                            beta,
                            gamma,
                            color_str,
                            static,
                            STLS,
                        )

    return bodies, STLS


def create_tdcr_from_config(
    config_dict: Dict[str, Any], output_path: Optional[str] = None
) -> str:
    """Create TDCR XML from configuration dictionary.

    Args:
        config_dict: Configuration dictionary with robot parameters
        output_path: Optional output file path

    Returns:
        Path to generated XML file
    """
    config = UnifiedTDCRConfig(config_dict)

    # Load obstacles if taskspace file is specified
    if config.taskspace_file and os.path.exists(config.taskspace_file):
        bodies, stls = load_obstacles_from_file(config.taskspace_file)
        config.custom_obstacles = (bodies, stls)

    generator = UnifiedTDCRGenerator(config, output_path)
    return generator.generate()


def main():
    """Command line interface for unified TDCR XML generation."""
    parser = argparse.ArgumentParser(description="Generate unified TDCR MJCF XML files")
    parser.add_argument("config", help="Path to JSON configuration file")
    parser.add_argument(
        "-o", "--output", help="Output XML file path", default="tdcr_generated.xml"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config_dict = json.load(f)

    # Generate XML
    output_path = create_tdcr_from_config(config_dict, args.output)
    print(f"Generated unified TDCR XML: {output_path}")


if __name__ == "__main__":
    main()
