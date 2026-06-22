# TDCR Generation Guide

This document provides comprehensive information about generating Tendon-Driven Continuum Robot (TDCR) models using the unified TDCR generator.

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration Structure](#configuration-structure)
4. [Material Properties](#material-properties)
5. [Actuation Modes](#actuation-modes)
6. [Advanced Features](#advanced-features)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Overview

The TDCR generation system creates MuJoCo XML models for tendon-driven continuum robots based on JSON configuration files. The system supports various actuation modes, material-based stiffness calculations, and advanced features like collision exclusion and pre-tension.

### Key Features
- Multiple actuation modes (direct torque, parallel tendons, position/velocity control)
- Material-based stiffness calculation with custom backbone radius
- Automatic tendon routing and actuator generation
- Configurable segments with varying properties
- Support for collision exclusion and pre-tension
- Inverse length-based stiffness scaling for tendons
- Obstacle integration from taskspace configuration files
- Franka Panda mounting options
- Customizable visuals with different color schemes and link shapes

## Quick Start

### Basic Usage

```bash
# List available configurations
python generate.py

# Generate a specific configuration
python generate.py --config example_three_segment_franka

# Generate from a custom config file
python generate.py --config-file path/to/custom.json

# Generate all example configurations
python generate.py --all

# Use a specific config file
python generate.py --config myrobot.json
```

### Python API

```python
from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config

config = {
    "num_segments": 3,
    "total_links": 30,
    "total_length": 0.6,
    "radius": 0.01,
    "mass": 1.0,
    "joints_per_link": 3,
    "joint_config_mode": "material",
    "material_properties": {
        "density": 37708,
        "youngs_modulus": 200e9,
        "poisson_ratio": 0.3
    },
    "actuation_mode": "direct_torque"
}

xml_path = create_tdcr_from_config(config, "my_robot.xml")
```

### Configuration Locations

- **Example configs**: `configs/generation/examples/`
- **Custom configs**: `configs/generation/*.json`
- **Output directory**: `assets/tdcr/`

## Configuration Structure

### Basic Robot Parameters

```json
{
  "description": "My TDCR robot",
  "num_segments": 3,
  "total_links": 30,
  "total_length": 0.6,
  "radius": 0.01,
  "mass": 1.0,
  "joints_per_link": 3
}
```

### Multi-Segment Configuration

For robots with varying segment properties:

```json
{
  "num_segments": 3,
  "links_per_segment": {"1": 10, "2": 15, "3": 5},
  "segment_lengths": {"1": 0.2, "2": 0.3, "3": 0.1}
}
```

### Joint Configuration Modes

#### 1. Direct Stiffness Specification

```json
{
  "joint_config_mode": "direct",
  "stiffness": 100.0,
  "damping": 1.0,
  "torsion_stiffness": 80.0,  // Optional, defaults to stiffness
  "joint_range": "-45 45"
}
```

#### 2. Material-Based Calculation

```json
{
  "joint_config_mode": "material",
  "material_properties": {
    "density": 37708,           // kg/m³
    "youngs_modulus": 200e9,    // Pa
    "poisson_ratio": 0.3,
    "inner_radius": 0,          // m (for hollow structures)
    "outer_radius": 0.0005,     // m (backbone radius, optional)
    "damping": 0.8              // Optional damping override
  }
}
```

## Material Properties

### Outer Radius Feature

The `outer_radius` parameter allows specifying a backbone radius different from the robot's visual radius:

- **Purpose**: Accurate stiffness calculation based on actual backbone dimensions
- **Default**: Uses robot's `radius` parameter if not specified
- **Example**: A robot with 6mm visual radius but 0.5mm backbone

```json
{
  "radius": 0.006,  // Visual/collision radius
  "material_properties": {
    "outer_radius": 0.0005  // Actual backbone radius for stiffness
  }
}
```

### Stiffness Calculation

For material-based configurations, the generator calculates:
- Cross-sectional area: `A = π(r_outer² - r_inner²)`
- Second moment of area: `I = (π/4)(r_outer⁴ - r_inner⁴)`
- Bending stiffness: `K = n × E × I / L`
- Torsional stiffness: `K_t = n × G × I_z / L`

## Actuation Modes

### 1. Direct Torque Control

```json
{
  "actuation_mode": "direct_torque",
  "actuator_type": "motor"  // or "position", "velocity"
}
```

### 2. Parallel Tendons

```json
{
  "actuation_mode": "parallel_tendons",
  "actuation_details": {
    "segments": [
      {
        "number_of_tendons": 3,
        "distance_to_backbone": 0.004
      }
    ]
  },
  "seg_offsets": [0, 0.5236, 1.0472]  // Radians
}
```

### 3. Custom Actuator Properties

```json
{
  "actuator_properties": {
    "tendon_ctrlrange": "-0.05 0.05",
    "tendon_constraint_factor": 1.0,  // Controls tendon routing sites (0-1)
    "tendon_actuator_type": "position",
    "tendon_kp": 100000,
    "tendon_kp_scaling": "inverse_length",
    "tendon_kp_scale_factor": 1.0,
    "tendon_pretension": 1.0,
    "tendon_forcelimited": "true",
    "tendon_forcerange": "-100 0"
  }
}
```

## Advanced Features

### Collision Exclusion

Disable self-collision between robot links:

```json
{
  "disable_self_collision": true
}
```

This creates collision exclusions between adjacent links in the kinematic chain.

### Pre-tension

Set initial tendon tensions:

```json
{
  "actuator_properties": {
    "tendon_pretension": 1.0  // Newton
  }
}
```

### Tendon Constraint Factor

Control how tendons are routed through links:

```json
{
  "actuator_properties": {
    "tendon_constraint_factor": 1.0  // Range: 0.0 to 1.0, default: 1.0
  }
}
```

- **0.0**: Single site at midpoint of each link (minimal constraint)
- **0.5**: Two sites per link, halfway between midpoint and edges
- **1.0**: Two sites per link at base and tip (maximum constraint)

**Important**: The implementation ensures tendon lengths always match segment lengths exactly, regardless of the constraint factor value. For links where a segment terminates (but isn't the final robot link), the second site is kept at the midpoint to maintain correct segment length.

This parameter affects tendon routing behavior and can influence robot stiffness characteristics without changing the fundamental geometry.

### Inverse Length Stiffness Scaling

Automatically scale actuator stiffness based on tendon length:

```json
{
  "actuator_properties": {
    "tendon_kp_scaling": "inverse_length",
    "tendon_kp_scale_factor": 1.0
  }
}
```

Shorter tendons receive proportionally higher stiffness values.

### Visual Customization

```json
{
  "plane": true,
  "plane_style": "checkered",  // or "white" (any other value renders white)
  "axis_label": true,           // Show coordinate axes
  "link_shape": "cylinder"      // or "box", "capsule", "ellipsoid"
}
```

### Mounting Options

Franka mounting is controlled by `generate.py` CLI flags, not config keys: use
`--franka` to mount the TDCR on the Franka Panda, `--world` to mount it in world
coordinates, or neither for a standalone TDCR (optionally `--mount-pos "x y z"`).

### Taskspace Files

Obstacles can be loaded from taskspace files with format:
```
Shape scale y x z static_flag alpha beta gamma color
Circle 0.05 0 0 0 True 0 0 0 #FF0000
path/to/model.stl 0.05 0 0 0 False 0 0 0 #00FF00
```

## Modular TDCR Generation

The modular TDCR system allows creation of heterogeneous robots composed of multiple modules with different material properties, enabling realistic multi-material simulations.

### Overview

Instead of defining a single set of material properties for the entire robot, you define a library of reusable modules, each with its own:
- Material properties (Young's modulus, density, etc.)
- Physical dimensions (length, radius)
- Simulation fidelity (number of links)
- Visual appearance (color)

Segments are then composed by specifying sequences of module IDs.

### Key Features

- **Per-Module Physics**: Each module independently calculates stiffness, damping, and mass from its material properties
- **Heterogeneous Composition**: Mix soft and stiff materials in one robot
- **Module Library**: Reusable module definitions for easy experimentation
- **Smart Tendon Routing**: Tendons attach only at module boundaries (base/tip) with midpoint sites
- **Visual Debugging**: Per-module colors for easy identification
- **Backward Compatible**: Standard monolithic configs still work

### Configuration Structure

```json
{
  "modular": true,
  "num_segments": 3,

  "module_library": {
    "module_id": {
      "length": 0.05,                    // Module length (m)
      "radius": 0.0125,                  // Visual/collision radius (m)
      "num_links": 5,                    // Simulation fidelity
      "color": "0.9 0.3 0.3 1.0",       // RGBA color (optional)
      "material_properties": {
        "density": 1000,                 // kg/m³
        "youngs_modulus": 100e9,         // Pa
        "poisson_ratio": 0.3,
        "inner_radius": 0,               // m (for hollow)
        "outer_radius": 0.0125,          // m (backbone radius)
        "damping_ratio": 0.05            // Proportional damping
      }
    }
  },

  "segments": [
    {
      "modules": ["module_id_1", "module_id_2", "module_id_3"],
      "actuation": {
        "number_of_tendons": 4,
        "distance_to_backbone": 0.01
      }
    }
  ],

  "actuation_mode": "parallel_tendons",
  "actuator_properties": { ... }
}
```

### Tendon Routing

For modular TDCRs, tendons attach only at module boundaries with 4 sites per module:
1. **Base boundary** (x=0)
2. **Base midpoint** (x=link_length/4)
3. **Tip midpoint** (x=link_length/4)
4. **Tip boundary** (x=link_length/2)

This reduces simulation complexity while maintaining realistic tendon paths.

### Example: Soft Gripper with Stiff Base

```json
{
  "description": "Modular TDCR - stiff base, soft tip",
  "modular": true,
  "num_segments": 2,

  "module_library": {
    "stiff_steel": {
      "length": 0.03,
      "radius": 0.0125,
      "num_links": 2,
      "color": "0.3 0.3 0.9 1.0",
      "material_properties": {
        "density": 7800,
        "youngs_modulus": 200e9,
        "poisson_ratio": 0.3,
        "outer_radius": 0.002,
        "damping_ratio": 0.05
      }
    },
    "soft_silicone": {
      "length": 0.05,
      "radius": 0.0125,
      "num_links": 5,
      "color": "0.9 0.3 0.3 1.0",
      "material_properties": {
        "density": 1200,
        "youngs_modulus": 1e6,
        "poisson_ratio": 0.49,
        "outer_radius": 0.0005,
        "damping_ratio": 0.1
      }
    }
  },

  "segments": [
    {
      "modules": ["stiff_steel", "stiff_steel"],
      "actuation": {
        "number_of_tendons": 4,
        "distance_to_backbone": 0.01
      }
    },
    {
      "modules": ["soft_silicone", "soft_silicone", "soft_silicone"],
      "actuation": {
        "number_of_tendons": 4,
        "distance_to_backbone": 0.01
      }
    }
  ],

  "joints_per_link": 2,
  "joint_range": "-100 100",
  "vert_joint_range": "-500 500",
  "actuation_mode": "parallel_tendons",
  "actuator_type": "motor",
  "actuator_properties": {
    "tendon_ctrlrange": "-0.15 0.15",
    "tendon_actuator_type": "position",
    "tendon_kp": 10000,
    "tendon_kp_scaling": "inverse_length",
    "tendon_pretension": 5.0
  },
  "seg_offsets": [0, 1.5708],
  "disable_self_collision": true
}
```

### Usage

```bash
# Generate modular TDCR
python generate.py --config example_modular

# With Franka mounting
python generate.py --config example_modular --franka

# View the result
python viewer.py --scene assets/tdcr/example_modular.xml
```

### Benefits

1. **Realistic Multi-Material Simulation**: Model robots with varying stiffness (e.g., compliant grippers)
2. **Easy Experimentation**: Swap modules without regenerating entire robot
3. **Visual Debugging**: Different colors help identify module boundaries
4. **Performance**: Fewer tendon sites (only at boundaries) improves simulation speed
5. **Modularity**: Build library of validated modules, combine as needed

### Design Guidelines

- **Stiffness Contrast**: For soft/stiff combinations, use Young's moduli differing by 2-3 orders of magnitude
- **Module Length**: Shorter modules allow finer-grained heterogeneity
- **Link Count**: More links = smoother bending but slower simulation
- **Color Coding**: Use contrasting colors to easily identify module types
- **Tendon Distance**: Keep consistent across modules in same segment for stable tendon routing

## Examples

### Example 1: Simple Material-Based Robot

```json
{
  "description": "Simple 3-joint TDCR",
  "num_segments": 1,
  "total_links": 20,
  "total_length": 0.5,
  "radius": 0.01,
  "mass": 0.5,
  "joints_per_link": 3,
  "joint_config_mode": "material",
  "material_properties": {
    "density": 8000,
    "youngs_modulus": 100e9,
    "poisson_ratio": 0.35
  },
  "actuation_mode": "direct_torque",
  "gravity": "0 0 -9.81",
  "plane": true
}
```

### Example 2: Multi-Segment Tendon Robot

```json
{
  "description": "3-segment tendon-driven TDCR",
  "num_segments": 3,
  "links_per_segment": {"1": 10, "2": 10, "3": 10},
  "segment_lengths": {"1": 0.1, "2": 0.1, "3": 0.1},
  "total_links": 30,
  "total_length": 0.3,
  "radius": 0.006,
  "mass": 0.1,
  "joints_per_link": 3,
  "joint_config_mode": "material",
  "material_properties": {
    "density": 37708,
    "youngs_modulus": 200e9,
    "poisson_ratio": 0.3,
    "outer_radius": 0.0005,
    "damping": 0.8
  },
  "actuation_mode": "parallel_tendons",
  "actuation_details": {
    "segments": [
      {"number_of_tendons": 3, "distance_to_backbone": 0.004},
      {"number_of_tendons": 3, "distance_to_backbone": 0.004},
      {"number_of_tendons": 3, "distance_to_backbone": 0.004}
    ]
  },
  "seg_offsets": [0, 0.5236, 1.0472],
  "actuator_properties": {
    "tendon_ctrlrange": "-0.05 0.05",
    "tendon_actuator_type": "position",
    "tendon_kp": 100000,
    "tendon_kp_scaling": "inverse_length",
    "tendon_pretension": 1.0,
    "tendon_constraint_factor": 1.0
  },
  "disable_self_collision": true,
  "gravity": "0 0 -9.81",
  "plane": true
}
```

## Generated XML Structure

The generator creates a hierarchical robot structure:
- Base link (optionally mounted on Franka)
- Segments with configurable properties
- Links with alternating colors
- Joints with specified stiffness/damping
- Tendons with attachment sites (if enabled)
- End effector marker

## Troubleshooting

### Common Issues

1. **Stiffness too high/low**: Adjust material properties or use direct stiffness mode
2. **Tendons not generating force**: Check actuator properties and control ranges
3. **Self-collision issues**: Enable `disable_self_collision`
4. **Incorrect stiffness calculation**: Verify `outer_radius` matches actual backbone

### Debugging Tips

1. Start with example configurations and modify incrementally

2. For material-based configs, verify physical properties are realistic

3. Check generated XML for actuator definitions and tendon routing

### Parameter Guidelines

- **Link density**: 1000-40000 kg/m³ (plastics to metals)
- **Young's modulus**: 1e9-200e9 Pa (rubber to steel)
- **Poisson's ratio**: 0.2-0.5 (most materials)
- **Backbone radius**: Typically much smaller than visual radius
- **Tendon distance**: Should be less than robot radius
- **Control ranges**: Start small and increase as needed

## See Also

- [Main Project Documentation](../../README.md)
- Example configurations in `configs/generation/examples/`
