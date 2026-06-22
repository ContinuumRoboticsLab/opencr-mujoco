"""Test pretension calculation and rest length for TDCR generation."""

import unittest
import tempfile
import os
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from generate import generate_scene


class TestPretension(unittest.TestCase):
    """Test pretension calculation and rest length."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_config(self, pretension=1.0, gravity="0 0 0"):
        """Create a minimal test configuration with specified pretension."""
        config = {
            "description": "Test TDCR for pretension",
            "num_segments": 1,
            "links_per_segment": {"1": 5},
            "segment_lengths": {"1": 0.1},
            "radius": 0.005,
            "mass": 0.01,
            "joints_per_link": 2,
            "joint_config_mode": "material",
            "material_properties": {
                "density": 1000,
                "youngs_modulus": 1e6,
                "poisson_ratio": 0.3,
                "outer_radius": 0.001,
                "damping": 0.1,
            },
            "joint_range": "-45 45",
            "vert_joint_range": "-45 45",
            "actuation_mode": "parallel_tendons",
            "actuator_type": "motor",
            "actuation_details": {
                "segments": [{"number_of_tendons": 3, "distance_to_backbone": 0.003}]
            },
            "actuator_properties": {
                "tendon_ctrlrange": "-0.02 0.02",
                "tendon_actuator_type": "position",
                "tendon_kp": 1000,
                "tendon_pretension": pretension,
            },
            "gravity": gravity,
            "plane": False,
            "disable_self_collision": True,
        }
        return config

    def test_rest_length_calculation(self):
        """Test that rest length is calculated from actual tendon geometry."""
        config = self.create_test_config(pretension=0.0)  # No pretension

        # Save config to file
        config_path = os.path.join(self.test_dir, "test_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Generate the model
        output_path = generate_scene(
            "test_config", config, output_dir=Path(self.test_dir)
        )

        # Parse the generated XML
        tree = ET.parse(output_path)
        root = tree.getroot()

        # Find tendons
        tendons = root.findall(".//spatial[@name]")
        self.assertEqual(len(tendons), 3, "Should have 3 tendons for 1 segment")

        # Load model in MuJoCo
        model = mujoco.MjModel.from_xml_path(str(output_path))
        data = mujoco.MjData(model)

        # Forward dynamics to get initial state
        mujoco.mj_forward(model, data)

        # Check tendon lengths
        for i in range(3):
            tendon_length = data.ten_length[i]
            print(f"Tendon {i} initial length: {tendon_length:.6f}")

            # The rest length should be close to the segment length (0.1)
            # but slightly different due to routing geometry
            self.assertGreater(
                tendon_length,
                0.095,
                f"Tendon {i} length {tendon_length} should be > 0.095",
            )
            self.assertLess(
                tendon_length,
                0.105,
                f"Tendon {i} length {tendon_length} should be < 0.105",
            )

    def test_pretension_with_no_gravity(self):
        """Test pretension in zero gravity to isolate the effect."""
        # Test with different pretension values
        for pretension_value in [0.0, 1.0, 5.0]:
            with self.subTest(pretension=pretension_value):
                config = self.create_test_config(
                    pretension=pretension_value, gravity="0 0 0"  # Zero gravity
                )

                # Save config
                config_path = os.path.join(
                    self.test_dir, f"test_config_{pretension_value}.json"
                )
                with open(config_path, "w") as f:
                    json.dump(config, f)

                # Generate model
                output_path = generate_scene(
                    f"test_config_{pretension_value}",
                    config,
                    output_dir=Path(self.test_dir),
                )

                # Load in MuJoCo
                model = mujoco.MjModel.from_xml_path(str(output_path))
                data = mujoco.MjData(model)

                # Apply pretension keyframe (should always exist for tendon robots)
                try:
                    key_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_KEY, "pretension"
                    )
                    mujoco.mj_resetDataKeyframe(model, data, key_id)
                except Exception:
                    pass  # No keyframe

                # Forward dynamics
                mujoco.mj_forward(model, data)

                # Get tendon lengths and forces
                tendon_lengths = data.ten_length[:3].copy()
                tendon_forces = data.actuator_force[:3].copy()

                print(f"\nPretension: {pretension_value}")
                print(f"Tendon lengths: {tendon_lengths}")
                print(f"Tendon forces: {tendon_forces}")

                if pretension_value > 0:
                    # With pretension, tendons should be shortened
                    # and forces should be non-zero even in equilibrium
                    # Force should be approximately equal to pretension value
                    self.assertTrue(
                        np.any(np.abs(tendon_forces) > pretension_value * 0.5),
                        f"Tendon forces should be non-zero with pretension={pretension_value}",
                    )
                else:
                    # Without pretension in zero gravity, forces should be near zero
                    self.assertTrue(
                        np.all(np.abs(tendon_forces) < 0.1),
                        "Tendon forces should be near zero without pretension",
                    )


if __name__ == "__main__":
    unittest.main()
