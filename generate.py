#!/usr/bin/env python3
"""Generate MuJoCo models and scenes for tendon-driven continuum robots.

This is the main entry point for building TDCR MJCF files from JSON
configuration. It supports standalone continuum robots, robots mounted on a
Franka Panda arm, and fixed world-mounted scenes. Runtime tools call the same
generation helpers automatically when an ignored generated scene is missing, so
most users do not need to run this script before using `viewer.py` or
`teleop.py`.

Usage:
    # See bundled generation configs.
    python generate.py --list-configs

    # Build common examples.
    python generate.py --config example_three_segment_franka --franka
    python generate.py --config example_modular --franka
    python generate.py --config example_modular_tension --franka

    # Mount with --franka, --world, or neither for a standalone TDCR.
    python generate.py --config example_three_segment_franka
    python generate.py --config example_three_segment_franka --world

    # Use a custom config file, or rebuild every bundled config.
    python generate.py --config-file path/to/myconfig.json --franka
    python generate.py --all --franka

Configuration reference: configs/generation/README.md
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import tempfile
from typing import Optional, List, Tuple

from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config
from opencr_mujoco.controllers.ik_controller import FRANKA_HOME_QPOS

# Get the project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).parent.resolve()


def load_generation_configs(config_path: Path) -> dict:
    """Load generation configurations from JSON file or directory.

    Args:
        config_path: Path to a single JSON config file or directory containing multiple configs

    Returns:
        Dictionary mapping config names to config data
    """
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if it's a single config (has expected fields) or multi-config
        if isinstance(data, dict) and any(
            key in data for key in ["num_segments", "total_links", "radius"]
        ):
            # Single config file - use filename (without extension) as key
            config_name = config_path.stem
            # Auto-calculate total_links and total_length if missing
            if "total_links" not in data and "links_per_segment" in data:
                total = sum(int(v) for v in data["links_per_segment"].values())
                data["total_links"] = total
            if "total_length" not in data and "segment_lengths" in data:
                total = sum(float(v) for v in data["segment_lengths"].values())
                data["total_length"] = total
            return {config_name: data}
        else:
            # Multi-config file
            return data

    elif config_path.is_dir():
        # Load all JSON files in directory
        configs = {}
        for json_file in sorted(config_path.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                # Auto-calculate totals if missing
                if (
                    "total_links" not in config_data
                    and "links_per_segment" in config_data
                ):
                    total = sum(
                        int(v) for v in config_data["links_per_segment"].values()
                    )
                    config_data["total_links"] = total
                if (
                    "total_length" not in config_data
                    and "segment_lengths" in config_data
                ):
                    total = sum(
                        float(v) for v in config_data["segment_lengths"].values()
                    )
                    config_data["total_length"] = total
                configs[json_file.stem] = config_data
        return configs
    else:
        raise ValueError(f"Config path not found: {config_path}")


def list_available_configs() -> List[Tuple[str, Path]]:
    """List all available TDCR configurations."""
    config_dir = PROJECT_ROOT / "configs/generation"
    configs = []

    # Check main directory
    for json_file in config_dir.glob("*.json"):
        if not json_file.name.startswith("unified"):
            configs.append((json_file.stem, json_file))

    # Check examples directory
    examples_dir = config_dir / "examples"
    if examples_dir.exists():
        for json_file in examples_dir.glob("*.json"):
            configs.append((json_file.stem, json_file))

    return configs


def resolve_scene_source(scene_path) -> Optional[Tuple[str, Optional[str], Path]]:
    """Reverse-map a generated scene path to its generation config.

    Generated scenes follow generate_scene()'s deterministic naming:
        assets/tdcr/<config>.xml             -> standalone TDCR
        assets/<config>_franka_scene.xml     -> mounted on the Franka
        assets/<config>_world_scene.xml      -> mounted in the world

    Returns (config_name, mount_type, config_json_path), or None when the
    path doesn't follow the convention or no matching generation config
    ships in configs/generation/.
    """
    path = Path(scene_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    name = mount = None
    if rel.parts[:2] == ("assets", "tdcr") and len(rel.parts) == 3:
        name, mount = path.stem, None
    elif rel.parts[:1] == ("assets",) and len(rel.parts) == 2:
        for m in ("franka", "world"):
            suffix = f"_{m}_scene"
            if path.stem.endswith(suffix):
                name, mount = path.stem[: -len(suffix)], m
                break
    if name is None:
        return None

    for config_path in (
        PROJECT_ROOT / "configs/generation" / f"{name}.json",
        PROJECT_ROOT / "configs/generation/examples" / f"{name}.json",
    ):
        if config_path.exists():
            return name, mount, config_path
    return None


def tdcr_geometry_from_scene(scene_path) -> dict:
    """Recover a TDCR's controller geometry from the config that built its scene.

    ``angle_offset_rad_ccw`` (the per-segment tendon phase, sysid-calibrated)
    and ``tendon_distance_mm`` are geometric facts of the robot, not teleop
    preferences — they must match the loaded asset for the Clark-coordinate
    tendon coordination to be correct. They are NOT recoverable from the scene
    XML (the offset is baked into the link body orientations and obscured by
    the mount transform), so read them from the authoritative source: the
    generation config the scene was built from.

    Returns a dict with any of ``angle_offset_rad_ccw`` / ``tendon_distance_mm``
    that the gen config defines, or ``{}`` for a hand-authored scene with no
    generation config. Callers merge this UNDER any explicit teleop value, so
    an intentional override still wins.
    """
    source = resolve_scene_source(scene_path)
    if source is None:
        return {}
    name, _mount, config_path = source
    cfg = load_generation_configs(config_path)[name]

    geometry: dict = {}
    seg_offsets = cfg.get("seg_offsets")
    if seg_offsets is not None:
        geometry["angle_offset_rad_ccw"] = list(seg_offsets)

    segments = cfg.get("actuation_details", {}).get("segments", [])
    dists_mm = [
        s["distance_to_backbone"] * 1000.0
        for s in segments
        if s.get("distance_to_backbone") is not None
    ]
    if dists_mm:
        # scalar when uniform across segments, else per-segment (mm)
        geometry["tendon_distance_mm"] = (
            dists_mm[0] if len(set(dists_mm)) == 1 else dists_mm
        )
    return geometry


def ensure_scene(scene_path) -> Path:
    """Return the scene path, generating it from its generation config if missing.

    Generated scene XMLs are not tracked in git; shipped teleop/viewer configs
    reference them by their deterministic generate_scene() names, so on a
    fresh clone this builds the scene on first use (no manual generate.py
    step needed).

    Raises FileNotFoundError when the scene is missing and cannot be derived
    from a shipped generation config.
    """
    path = Path(scene_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        return path

    source = resolve_scene_source(path)
    if source is None:
        raise FileNotFoundError(
            f"Scene file not found: {path}\n"
            "It does not follow a generate.py naming convention "
            "(assets/tdcr/<config>.xml or assets/<config>_franka_scene.xml), "
            "so it cannot be regenerated automatically."
        )
    name, mount, config_path = source

    print(
        f"Scene not found — generating it from "
        f"{config_path.relative_to(PROJECT_ROOT)} ..."
    )
    config_data = load_generation_configs(config_path)[name]
    output_path = generate_scene(name, config_data, mount_type=mount)
    return output_path


def _prepend_franka_home_to_key(key: ET.Element) -> ET.Element:
    """Copy a TDCR keyframe <key>, prepending the Franka home ctrl values.

    The Franka home comes from FRANKA_HOME_QPOS (single source of truth,
    shared with the controllers).
    """
    new_key = ET.Element("key")
    for attr, value in key.attrib.items():
        if attr == "ctrl":
            franka_joints = ["0" if v == 0 else str(float(v)) for v in FRANKA_HOME_QPOS]
            new_key.set("ctrl", f"{' '.join(franka_joints)} {value}")
        else:
            new_key.set(attr, value)
    return new_key


def _set_franka_home_qpos_in_keyframe(scene_root: ET.Element, scene_path: Path) -> None:
    """Pin the 'pretension' keyframe's qpos so the Franka arm starts at home.

    ``_prepend_franka_home_to_key`` writes only the keyframe *ctrl* (the position
    servos' target = the Franka home pose). With no matching *qpos*, MuJoCo
    defaults key_qpos to qpos0 (Franka arm joints = 0, i.e. fully extended), so
    on load the servos swing the arm from extended to home over a few seconds.
    We compile the just-written scene to read qpos0 (correct for any TDCR joint
    layout, including ball joints) and overwrite only the 7 Franka arm joints
    with FRANKA_HOME_QPOS, so the keyframe starts settled at home.
    """
    import mujoco

    keyframe = scene_root.find("keyframe")
    if keyframe is None:
        return
    key = next(
        (k for k in keyframe.findall("key") if k.get("name") == "pretension"), None
    )
    if key is None or key.get("qpos") is not None:
        return

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    except Exception as exc:  # pragma: no cover - a broken scene is a louder failure
        print(f"  Warning: could not pin Franka home qpos in keyframe: {exc}")
        return

    qpos = model.qpos0.copy()
    found_franka = False
    for i in range(1, 8):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{i}")
        if jid >= 0:
            qpos[model.jnt_qposadr[jid]] = FRANKA_HOME_QPOS[i - 1]
            found_franka = True
    if not found_franka:
        return  # no Franka arm in this scene; nothing to settle

    key.set("qpos", " ".join("0" if v == 0 else repr(float(v)) for v in qpos))
    # Only an attribute was added to the already-indented tree, so re-writing
    # preserves formatting without re-indenting.
    ET.ElementTree(scene_root).write(
        str(scene_path), encoding="unicode", xml_declaration=True
    )


def load_franka_base():
    """Load the base Franka robot structure without gripper."""
    franka_xml = """<mujoco>
    <compiler angle="radian" autolimits="true"/>
    <option gravity="0 0 -9.81"/>

    <default>
        <joint axis="0 0 1" limited="true"/>
        <geom contype="0" conaffinity="0" condim="1" friction="1.0 0.005 0.0001"/>
    </default>

    <asset>
        <material name="panda_white" rgba="1 1 1 1" shininess="0.4" specular="0.4"/>
        <material name="panda" rgba="0.8 0.8 0.8 1" shininess="0.4" specular="0.4"/>
        <material name="mount_adapter" rgba="0.3 0.3 0.3 1" shininess="0.2" specular="0.2"/>
        <material name="mount_connector" rgba="0.5 0.5 0.5 1" shininess="0.3" specular="0.3"/>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1" width="256" height="256"/>

        <!-- Ground plane assets from basic_scene.xml -->
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512"/>
        <material name="MatGnd" reflectance="0.5" texture="texplane" texrepeat="1 1" texuniform="true"/>
        
        <!-- Franka meshes from local assets -->
        <mesh name="link0" file="franka_assets/meshes/visual/link0.stl"/>
        <mesh name="link1" file="franka_assets/meshes/visual/link1.stl"/>
        <mesh name="link2" file="franka_assets/meshes/visual/link2.stl"/>
        <mesh name="link3" file="franka_assets/meshes/visual/link3.stl"/>
        <mesh name="link4" file="franka_assets/meshes/visual/link4.stl"/>
        <mesh name="link5" file="franka_assets/meshes/visual/link5.stl"/>
        <mesh name="link6" file="franka_assets/meshes/visual/link6.stl"/>
        <mesh name="link7" file="franka_assets/meshes/visual/link7.stl"/>
    </asset>
    
    <worldbody>
        <!-- Lights from basic_scene.xml -->
        <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="1  1 3" dir="-1 -1 -3"/>
        <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="1 -1 3" dir="-1 1 -3"/>
        <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="-1 0 3" dir="1 0 -3" />
        
        <!-- Ground plane from basic_scene.xml -->
        <geom name="ground" pos="0 0 0" size="5 5 10" material="MatGnd" type="plane" contype="1" conaffinity="1"/>
        
        <!-- Franka robot base -->
        <body name="panda_link0">
            <geom type="mesh" material="panda" mesh="link0"/>
            <inertial pos="-4.1018e-02 -1.4e-04 4.9974e-02" mass="6.29769e-01" 
                      fullinertia="3.15e-03 3.88e-03 4.285e-03 8.2904e-07 1.5e-04 8.2299e-06"/>
            <body name="panda_link1" pos="0 0 0.333">
                <inertial pos="3.875e-03 2.081e-03 -4.762e-02" mass="4.970684" 
                          fullinertia="7.0337e-01 7.0661e-01 9.1170e-03 -1.3900e-04 6.7720e-03 1.9169e-02"/>
                <joint name="panda_joint1" pos="0 0 0" axis="0 0 1" limited="true" 
                       range="-2.8973 2.8973" damping="100"/>
                <geom type="mesh" material="panda_white" mesh="link1"/>
                <body name="panda_link2" pos="0 0 0" quat="0.707107 -0.707107 0 0">
                    <inertial pos="-3.141e-03 -2.872e-02 3.495e-03" mass="0.646926" 
                              fullinertia="7.9620e-03 2.8110e-02 2.5995e-02 -3.9250e-03 1.0254e-02 7.0400e-04"/>
                    <joint name="panda_joint2" pos="0 0 0" axis="0 0 1" limited="true" 
                           range="-1.7628 1.7628" damping="100"/>
                    <geom type="mesh" material="panda_white" mesh="link2"/>
                    <body name="panda_link3" pos="0 -0.316 0" quat="0.707107 0.707107 0 0">
                        <inertial pos="2.7518e-02 3.9252e-02 -6.6502e-02" mass="3.228604" 
                                  fullinertia="3.7242e-02 3.6155e-02 1.0830e-02 -4.7610e-03 -1.1396e-02 -1.2805e-02"/>
                        <joint name="panda_joint3" pos="0 0 0" axis="0 0 1" limited="true" 
                               range="-2.8973 2.8973" damping="100"/>
                        <geom type="mesh" material="panda" mesh="link3"/>
                        <body name="panda_link4" pos="0.0825 0 0" quat="0.707107 0.707107 0 0">
                            <inertial pos="-5.317e-02 1.04419e-01 2.7454e-02" mass="3.587895" 
                                      fullinertia="2.5853e-02 1.9552e-02 2.8323e-02 7.7960e-03 -1.3320e-03 8.6410e-03"/>
                            <joint name="panda_joint4" pos="0 0 0" axis="0 0 1" limited="true" 
                                   range="-3.0718 -0.0698" damping="100"/>
                            <geom type="mesh" material="panda" mesh="link4"/>
                            <body name="panda_link5" pos="-0.0825 0.384 0" quat="0.707107 -0.707107 0 0">
                                <inertial pos="-1.1953e-02 4.1065e-02 -3.8437e-02" mass="1.225946" 
                                          fullinertia="3.5549e-02 2.9474e-02 8.6270e-03 -2.1170e-03 -4.0370e-03 2.2900e-04"/>
                                <joint name="panda_joint5" pos="0 0 0" axis="0 0 1" limited="true" 
                                       range="-2.8973 2.8973" damping="10"/>
                                <geom type="mesh" material="panda" mesh="link5"/>
                                <body name="panda_link6" pos="0 0 0" quat="0.707107 0.707107 0 0">
                                    <inertial pos="6.0149e-02 -1.4117e-02 -1.0517e-02" mass="1.666555" 
                                              fullinertia="1.9640e-03 4.3540e-03 5.4330e-03 1.0900e-04 -1.1580e-03 3.4100e-04"/>
                                    <joint name="panda_joint6" pos="0 0 0" axis="0 0 1" limited="true" 
                                           range="-0.0873 3.8223" damping="10"/>
                                    <geom type="mesh" material="panda" mesh="link6"/>
                                    <body name="panda_link7" pos="0.088 0 0" quat="0.707107 0.707107 0 0">
                                        <inertial pos="1.0517e-02 -4.252e-03 6.1597e-02" mass="7.35522e-01" 
                                                  fullinertia="1.2516e-02 1.0027e-02 4.8150e-03 -4.2800e-04 -1.1960e-03 -7.4100e-04"/>
                                        <joint name="panda_joint7" pos="0 0 0" axis="0 0 1" limited="true" 
                                               range="-2.8973 2.8973" damping="10"/>
                                        <geom type="mesh" material="panda" mesh="link7"/>
                                        <body name="panda_link8" pos="0 0 0.107">
                                            <body name="end_effector">
                                                <inertial pos="-1e-02 0 3e-02" mass="7.3e-01" diaginertia="1e-03 2.5e-03 1.7e-03"/>
                                                <!-- Mount point for TDCR -->
                                                <body name="mount" euler="0 -1.57079632679 0">
                                                    <body name="tdcr_mount">
                                                        <!-- TDCR will be inserted here -->
                                                    </body>
                                                </body>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <position name="panda_joint1" joint="panda_joint1" kp="870" forcerange="-87 87" ctrlrange="-2.8973 2.8973"/>
        <position name="panda_joint2" joint="panda_joint2" kp="870" forcerange="-87 87" ctrlrange="-1.7628 1.7628"/>
        <position name="panda_joint3" joint="panda_joint3" kp="870" forcerange="-87 87" ctrlrange="-2.8973 2.8973"/>
        <position name="panda_joint4" joint="panda_joint4" kp="870" forcerange="-87 87" ctrlrange="-3.0718 -0.4"/>
        <position name="panda_joint5" joint="panda_joint5" kp="120" forcerange="-12 12" ctrlrange="-2.8973 2.8973"/>
        <position name="panda_joint6" joint="panda_joint6" kp="120" forcerange="-12 12" ctrlrange="-0.0873 3.8223"/>
        <position name="panda_joint7" joint="panda_joint7" kp="120" forcerange="-12 12" ctrlrange="-2.9671 2.9671"/>
    </actuator>
</mujoco>"""
    return ET.fromstring(franka_xml)


def merge_tdcr_into_scene(
    scene_root,
    tdcr_root,
    mount_type="franka",
    mount_pos=None,
    mount_euler=None,
    mount_angle=0.0,
):
    """Merge TDCR model into a scene (Franka or world).

    Args:
        mount_angle: Rotation angle (radians) around the mount frame's X axis.
            Rotates the TDCR mount position and adjusts orientations accordingly.
    """
    import numpy as np

    # Find the mount point
    if mount_type == "franka":
        mount_point = None
        for elem in scene_root.iter("body"):
            if elem.get("name") == "tdcr_mount":
                mount_point = elem
                break

        # Find the mount parent (should be inside end_effector/mount)
        mount_parent = None
        for elem in scene_root.iter("body"):
            if elem.get("name") == "mount":
                mount_parent = elem
                # Remove the existing tdcr_mount from mount
                for child in list(mount_parent):
                    if child.get("name") == "tdcr_mount":
                        mount_parent.remove(child)
                break
    else:
        # World mounting - create mount point
        worldbody = scene_root.find("worldbody")
        mount_parent = worldbody
        mount_point = None

    # Add mounting adapter visualization if mounting on Franka
    if mount_type == "franka" and mount_parent is not None:
        # Parse mount position and euler
        pos_values = [0.09, 0, -0.115]  # Default values
        if mount_pos:
            pos_values = [float(x) for x in mount_pos.split()]

        euler_values = [0, 1.5708, 0]  # Default euler for TDCR
        if mount_euler:
            euler_values = [float(x) for x in mount_euler.split()]

        # Apply mount angle rotation around X axis
        if mount_angle != 0.0:
            cos_a = np.cos(mount_angle)
            sin_a = np.sin(mount_angle)
            y_orig = pos_values[1]
            z_orig = pos_values[2]
            # Rotate YZ by -mount_angle around X
            pos_values[1] = y_orig * cos_a + z_orig * sin_a
            pos_values[2] = -y_orig * sin_a + z_orig * cos_a
            # Adjust tdcr euler Z to compensate
            euler_values[2] -= mount_angle

        # In the mount frame (which has euler="0 -1.57079632679 0"):
        # - Local X points forward (robot's original Z)
        # - Local Y stays the same
        # - Local Z points down (robot's original -X)

        # Calculate adapter dimensions based on mount position
        # Create mount_adapter body positioned halfway to the mount point
        adapter_body = ET.Element("body")
        adapter_body.set("name", "mount_adapter")

        # Round values for clean XML output
        def fmt(v):
            return round(v, 7)

        # Position halfway to the mount point
        adapter_body.set("pos", f"0 0 {fmt(pos_values[2]/2)}")
        if mount_angle != 0.0:
            adapter_body.set("euler", f"{fmt(mount_angle)} 0.0 0.0")

        # Add box geometry
        # The box should be offset forward and up from its body center
        adapter_geom = ET.Element("geom")
        adapter_geom.set("name", "mount_adapter_box")
        adapter_geom.set("type", "box")

        # Calculate box position and size based on mount distance
        # Box should extend from near the flange to near the TDCR mount
        box_forward_offset = pos_values[0] * 0.8  # 80% of forward distance
        box_vertical_offset = abs(pos_values[2]) * 0.3  # 30% up from center
        box_length = pos_values[0] * 0.8  # Length in forward direction
        box_height = abs(pos_values[2]) / 2  # Half the vertical distance

        adapter_geom.set(
            "pos", f"{fmt(box_forward_offset)} 0 {fmt(box_vertical_offset)}"
        )
        adapter_geom.set(
            "size", f"{fmt(box_length)} 0.05 {fmt(box_height)}"
        )  # Wider in Y (0.05m = 10cm width)
        adapter_geom.set("material", "mount_adapter")
        adapter_geom.set("contype", "0")
        adapter_geom.set("conaffinity", "0")
        adapter_body.append(adapter_geom)

        # Add adapter body to mount parent
        mount_parent.append(adapter_body)

        # Create new tdcr_mount with proper position and orientation
        mount_point = ET.Element("body")
        mount_point.set("name", "tdcr_mount")
        mount_point.set(
            "pos", f"{fmt(pos_values[0])} {fmt(pos_values[1])} {fmt(pos_values[2])}"
        )

        # The TDCR mount euler should counteract the mount rotation to point forward
        # Since mount has euler="0 -1.5708 0", we need euler="0 1.5708 0" to point forward
        # But we also need to apply any user-specified rotation
        tdcr_euler = [euler_values[0], euler_values[1], euler_values[2]]
        mount_point.set(
            "euler", f"{fmt(tdcr_euler[0])} {fmt(tdcr_euler[1])} {fmt(tdcr_euler[2])}"
        )

        # Add cylinder at TDCR mount point
        # The TDCR extends along the X axis after the tdcr_mount rotation
        # Cylinder default orientation is along Z axis
        # We need to rotate it 90 degrees around Y to point along positive X
        cylinder_geom = ET.Element("geom")
        cylinder_geom.set("name", "mount_connector_cylinder")
        cylinder_geom.set("type", "cylinder")
        cylinder_geom.set(
            "pos", "-0.03 0 0"
        )  # Move back 3cm along negative X to avoid overlap
        cylinder_geom.set("size", "0.025 0.03")  # radius 2.5cm, height 6cm
        cylinder_geom.set(
            "euler", "0 1.5708 0"
        )  # Rotate 90 degrees around Y to align with positive X
        cylinder_geom.set("material", "mount_connector")
        cylinder_geom.set("contype", "0")
        cylinder_geom.set("conaffinity", "0")
        mount_point.append(cylinder_geom)

        # Add tdcr_mount to mount parent
        mount_parent.append(mount_point)

    elif mount_type == "world":
        # World mounting - create mount point with transforms
        mount_point = ET.Element("body")
        mount_point.set("name", "tdcr_mount")
        if mount_pos:
            mount_point.set("pos", mount_pos)
        if mount_euler:
            mount_point.set("euler", mount_euler)
        mount_parent.append(mount_point)

    if mount_point is None:
        raise ValueError("Could not create tdcr_mount point in scene")

    # Extract TDCR worldbody content
    tdcr_worldbody = tdcr_root.find("worldbody")
    if tdcr_worldbody is None:
        raise ValueError("TDCR file has no worldbody")

    # Find the base link of TDCR
    tdcr_base = None
    for body in tdcr_worldbody:
        if body.tag == "body" and body.get("mocap") == "true":
            # TDCR is usually inside the mocap body
            for child in body:
                if child.tag == "body" and child.get("name") in [
                    "link_start",
                    "link_0",
                    "base_link",
                    "link_1",
                ]:
                    tdcr_base = child
                    break
            if tdcr_base is not None:
                body.remove(tdcr_base)
                break

    # If not found in mocap, look for direct children
    if tdcr_base is None:
        for body in tdcr_worldbody:
            if body.tag == "body" and body.get("mocap") != "true":
                body_name = body.get("name", "")
                if body_name in ["base_link", "link_start", "link_0"]:
                    tdcr_base = body
                    break

    if tdcr_base is None:
        raise ValueError("Could not find base body in TDCR file")

    # TDCR backbone extends along +Z. When mounting on Franka, rotate link_0
    # by 90° about Y so the backbone points along the mount frame's +X (forward).
    if mount_type == "franka" and not tdcr_base.get("euler"):
        tdcr_base.set("euler", "0 1.5708 3.1416")

    # Append TDCR base to mount point
    mount_point.append(tdcr_base)

    # Merge defaults (handle duplicate geom elements)
    tdcr_default = tdcr_root.find("default")
    if tdcr_default is not None:
        scene_default = scene_root.find("default")
        if scene_default is None:
            scene_default = ET.SubElement(scene_root, "default")

        # Check if there's already a geom element
        scene_geom = scene_default.find("geom")

        for child in tdcr_default:
            # Skip duplicate geom defaults
            if child.tag == "geom" and child.get("class") is None:
                if scene_geom is not None:
                    continue
            scene_default.append(child)

    # Merge other elements
    for element_name in ["asset", "actuator", "tendon", "contact", "keyframe"]:
        tdcr_element = tdcr_root.find(element_name)
        if tdcr_element is not None:
            scene_element = scene_root.find(element_name)
            if scene_element is None:
                # Special handling for keyframes with Franka even when creating new element
                if element_name == "keyframe" and mount_type == "franka":
                    new_keyframe = ET.Element("keyframe")
                    for key in tdcr_element:
                        new_keyframe.append(_prepend_franka_home_to_key(key))
                    scene_root.append(new_keyframe)
                else:
                    scene_root.append(tdcr_element)
            else:
                # Special handling for assets to avoid duplicates
                if element_name == "asset":
                    existing_materials = {
                        mat.get("name") for mat in scene_element.findall("material")
                    }
                    existing_textures = {
                        tex.get("name")
                        for tex in scene_element.findall("texture")
                        if tex.get("name")
                    }

                    for child in tdcr_element:
                        if (
                            child.tag == "material"
                            and child.get("name") in existing_materials
                        ):
                            continue
                        if (
                            child.tag == "texture"
                            and child.get("name")
                            and child.get("name") in existing_textures
                        ):
                            continue
                        if child.tag == "texture" and child.get("type") == "skybox":
                            continue
                        scene_element.append(child)
                # Special handling for keyframes with Franka
                elif element_name == "keyframe" and mount_type == "franka":
                    for key in tdcr_element:
                        scene_element.append(_prepend_franka_home_to_key(key))
                else:
                    for child in tdcr_element:
                        scene_element.append(child)

    # Update options from TDCR
    tdcr_option = tdcr_root.find("option")
    if tdcr_option is not None:
        scene_option = scene_root.find("option")
        if scene_option is None:
            scene_option = ET.SubElement(scene_root, "option")
        for attr, value in tdcr_option.attrib.items():
            scene_option.set(attr, value)


def indent_xml(elem, level=0):
    """Add proper indentation to XML elements."""
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def generate_scene(
    config_name: str,
    config_data: dict,
    mount_type: Optional[str] = None,
    mount_pos: Optional[str] = None,
    mount_euler: Optional[str] = None,
    output_dir: Optional[Path] = None,
    mount_angle_override: Optional[float] = None,
) -> Path:
    """Generate a complete TDCR scene with an optional Franka mount.

    Args:
        config_name: Name to use for output file
        config_data: TDCR configuration dictionary
        mount_type: 'franka' or 'world' to mount TDCR, None for standalone
        mount_pos: Mount position override (space-separated x y z)
        mount_euler: Mount euler angles override (space-separated roll pitch yaw in radians)
        output_dir: Output directory (relative to project root if not absolute)
        mount_angle_override: Mount angle override in radians (overrides config value)

    Returns:
        Path to generated XML file
    """

    # Auto-calculate total_links and total_length if missing
    if "total_links" not in config_data and "links_per_segment" in config_data:
        total = sum(int(v) for v in config_data["links_per_segment"].values())
        config_data["total_links"] = total
    if "total_length" not in config_data and "segment_lengths" in config_data:
        total = sum(float(v) for v in config_data["segment_lengths"].values())
        config_data["total_length"] = total

    # Determine output path (relative to project root)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "assets"
        if mount_type is None:
            output_dir = output_dir / "tdcr"
    else:
        # If user provided output_dir, make it relative to project root if it's not absolute
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename based on options
    if mount_type is None:
        # Standalone TDCR
        output_filename = f"{config_name}.xml"
    else:
        # Scene with mounting
        output_filename = f"{config_name}_{mount_type}_scene"
        output_filename += ".xml"

    output_path = output_dir / output_filename

    # Get mounting defaults from config if available
    mount_angle = 0.0
    # Standard Franka mount (matches configs/generation/ftdcr_v4_sysid.json) used
    # as the default whenever a config doesn't override it, so every Franka-mounted
    # robot shares the same bracket pose.
    FRANKA_MOUNT_POS = "0.075 0 -0.13055"
    FRANKA_MOUNT_EULER = "0 1.5708 0"
    FRANKA_MOUNT_ANGLE = 0.7854
    if mount_type and "mounting" in config_data:
        mount_config = config_data["mounting"]
        default_angle = FRANKA_MOUNT_ANGLE if mount_type == "franka" else 0.0
        mount_angle = mount_config.get("mount_angle", default_angle)
        if mount_pos is None:
            if mount_type == "franka":
                mount_pos = mount_config.get("franka_pos", FRANKA_MOUNT_POS)
            else:
                mount_pos = mount_config.get("world_pos", "0 0 0.3")
        if mount_euler is None:
            if mount_type == "franka":
                mount_euler = mount_config.get("franka_euler", FRANKA_MOUNT_EULER)
            else:
                mount_euler = mount_config.get("world_euler", "0 0 0")
    else:
        # Use hardcoded defaults (Franka standardized to the ftdcr_v4_sysid pose)
        if mount_type == "franka":
            mount_angle = FRANKA_MOUNT_ANGLE
        if mount_pos is None:
            mount_pos = FRANKA_MOUNT_POS if mount_type == "franka" else "0 0 0.3"
        if mount_euler is None:
            mount_euler = FRANKA_MOUNT_EULER if mount_type == "franka" else "0 0 0"
    if mount_angle_override is not None:
        mount_angle = mount_angle_override

    # Generate TDCR model to temp file
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tdcr_path = Path(tmp.name)

    try:
        # Generate TDCR model
        print(f"\nGenerating TDCR model '{config_name}'...")
        create_tdcr_from_config(config_data, str(tdcr_path))

        if mount_type is None:
            # Standalone TDCR - just copy to output
            import shutil

            shutil.move(str(tdcr_path), str(output_path))
        else:
            # Create scene with mounting
            if mount_type == "franka":
                print("  Mounting on Franka robot...")
                scene_root = load_franka_base()
            else:
                # World mounting
                print("  Creating world-mounted scene...")
                world_xml = """<mujoco>
    <compiler angle="radian" autolimits="true"/>
    <option gravity="0 0 -9.81"/>
    
    <default>
        <joint axis="0 0 1" limited="true"/>
        <geom contype="0" conaffinity="0" condim="1" friction="1.0 0.005 0.0001"/>
    </default>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1" width="256" height="256"/>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512"/>
        <material name="MatGnd" reflectance="0.5" texture="texplane" texrepeat="1 1" texuniform="true"/>
    </asset>
    
    <worldbody>
        <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="1  1 3" dir="-1 -1 -3"/>
        <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="1 -1 3" dir="-1 1 -3"/>
        <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="-1 0 3" dir="1 0 -3"/>
        <geom name="ground" pos="0 0 0" size="5 5 10" material="MatGnd" type="plane" contype="1" conaffinity="1"/>
    </worldbody>
</mujoco>"""
                scene_root = ET.fromstring(world_xml)

            # Load and merge TDCR
            tdcr_tree = ET.parse(tdcr_path)
            tdcr_root = tdcr_tree.getroot()
            merge_tdcr_into_scene(
                scene_root, tdcr_root, mount_type, mount_pos, mount_euler, mount_angle
            )

            # Save scene
            indent_xml(scene_root)
            tree = ET.ElementTree(scene_root)
            tree.write(str(output_path), encoding="unicode", xml_declaration=True)

            # Franka-mounted scenes: pin the keyframe qpos to the Franka home so
            # the arm starts settled instead of swinging there from qpos0=0.
            if mount_type == "franka":
                _set_franka_home_qpos_in_keyframe(scene_root, output_path)

        print(f"✓ Successfully created: {output_path}")

        if mount_type:
            print(f"  Mount type: {mount_type}")
            print(f"  Mount position: {mount_pos}")
            print(f"  Mount orientation: {mount_euler}")

        print(f"\nView with: python viewer.py --scene {output_path}")

        return output_path

    finally:
        # Clean up temp file
        tdcr_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Unified TDCR scene generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --list-configs              # List TDCR configurations

  python generate.py --config example_three_segment_franka                    # Generate standalone TDCR
  python generate.py --config example_three_segment_franka --franka           # TDCR mounted on Franka
  python generate.py --config example_three_segment_franka --world            # TDCR mounted in world

  python generate.py --config example_three_segment_franka --franka --mount-pos "0.1 0 -0.1"  # Custom mount

  python generate.py --config-file path/to/myconfig.json --franka  # Custom config file

  python generate.py --all                       # Generate all standalone TDCRs
  python generate.py --all --franka              # Generate all with Franka
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Name of configuration from configs/generation/ (e.g., example_three_segment_franka)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to custom configuration JSON file (alternative to --config)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory (default: assets/tdcr for standalone, assets/ for scenes, relative to project root)",
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Generate all configurations"
    )
    parser.add_argument(
        "--list-configs", action="store_true", help="List available TDCR configurations"
    )

    # Scene options
    scene_group = parser.add_argument_group("scene options")
    mount_group = scene_group.add_mutually_exclusive_group()
    mount_group.add_argument(
        "--franka", "-f", action="store_true", help="Mount TDCR on Franka robot"
    )
    mount_group.add_argument(
        "--world", "-w", action="store_true", help="Mount TDCR in world coordinates"
    )

    scene_group.add_argument(
        "--mount-pos", type=str, help="Mount position (default depends on mount type)"
    )
    scene_group.add_argument(
        "--mount-euler", type=str, help="Mount euler angles in radians"
    )
    scene_group.add_argument(
        "--mount-angle",
        type=float,
        default=None,
        help="Mount angle in radians (rotation around mount X axis)",
    )

    args = parser.parse_args()

    # List configurations
    if args.list_configs:
        configs = list_available_configs()
        if not configs:
            print("No TDCR configurations found")
            sys.exit(1)

        print("\nAvailable TDCR configurations:")
        print("-" * 70)
        for name, path in sorted(configs):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    desc = config.get("description", "No description")
                    mount_info = (
                        " [has mounting config]" if "mounting" in config else ""
                    )
                    print(f"  {name:<25} {desc}{mount_info}")
            except (OSError, json.JSONDecodeError) as e:
                print(f"  {name:<25} (error loading: {e})")
        print("-" * 70)
        print("\nUsage: python generate.py --config <config_name>")
        return

    # Check config argument
    if not args.config and not args.config_file and not args.all:
        print("Error: Configuration required (use --config, --config-file, or --all)")
        print("Use --list-configs to see available configurations")
        sys.exit(1)

    # Validate mutual exclusivity of --config and --config-file
    if args.config and args.config_file:
        print("Error: Cannot use both --config and --config-file")
        sys.exit(1)

    # --all is standalone: it builds every bundled config and ignores any
    # single-config selection, so reject the ambiguous combination.
    if args.all and (args.config or args.config_file):
        print("Error: --all cannot be combined with --config or --config-file")
        sys.exit(1)

    # Determine mount type
    mount_type = None
    if args.franka:
        mount_type = "franka"
    elif args.world:
        mount_type = "world"

    # Load configurations
    config_path = None
    if args.config_file:
        # Custom config file path (can be absolute or relative to CWD, will be resolved)
        config_path = Path(args.config_file)
        if not config_path.is_absolute():
            config_path = config_path.resolve()
    elif args.config:
        # Config name from configs/generation/
        base_dir = PROJECT_ROOT / "configs/generation"
        # Try direct file first
        single_config = base_dir / f"{args.config}.json"
        if single_config.exists():
            config_path = single_config
        else:
            # Try in examples
            config_path = base_dir / "examples" / f"{args.config}.json"
            if not config_path.exists():
                print(f"Error: Configuration '{args.config}' not found")
                print("Use --list-configs to see available configurations")
                sys.exit(1)
    # Load configurations
    try:
        if args.all:
            # Enumerate every bundled config exactly as --list-configs does:
            # top-level configs/generation/*.json (excluding 'unified*') plus
            # configs/generation/examples/*.json. (Previously --all read only the
            # examples/ subdir and silently built just example_contact.)
            configs = {}
            for _name, cfg_file in list_available_configs():
                configs.update(load_generation_configs(cfg_file))
        else:
            configs = load_generation_configs(config_path)
    except Exception as e:
        print(f"Error loading configurations: {e}")
        sys.exit(1)

    # Generate all mode
    if args.all:
        print(f"\nGenerating all {len(configs)} configurations...")
        success_count = 0
        for name, config in configs.items():
            try:
                generate_scene(
                    name,
                    config,
                    mount_type,
                    args.mount_pos,
                    args.mount_euler,
                    args.output_dir,
                    args.mount_angle,
                )
                success_count += 1
            except Exception as e:
                print(f"  ✗ Error generating {name}: {e}")

        print(
            f"\nSummary: {success_count}/{len(configs)} models generated successfully"
        )
        return

    # Generate specific config
    # For single config, the name is either the --config value or the stem of --config-file
    config_name = args.config if args.config else Path(args.config_file).stem
    if config_name not in configs:
        print(f"Error: Configuration '{config_name}' not found")
        sys.exit(1)

    try:
        generate_scene(
            config_name,
            configs[config_name],
            mount_type,
            args.mount_pos,
            args.mount_euler,
            args.output_dir,
            args.mount_angle,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
