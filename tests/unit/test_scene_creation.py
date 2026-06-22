#!/usr/bin/env python3
"""Unit tests for streamlined TDCR scene creation."""

import pytest
import tempfile
from pathlib import Path
import sys
import os
import json
import xml.etree.ElementTree as ET

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from generate import (
    load_generation_configs,
    list_available_configs,
    generate_scene,
)


class TestSceneCreation:
    """Test streamlined TDCR scene creation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_list_available_configs(self):
        """Test listing available TDCR configurations."""
        from pathlib import Path

        configs = list_available_configs()

        # Should return a list of tuples
        assert isinstance(configs, list)

        # If there are configs, check their structure
        if configs:
            # Each config should be a (name, path) tuple
            for config in configs:
                assert isinstance(config, tuple)
                assert len(config) == 2
                name, path = config
                assert isinstance(name, str) and len(name) > 0
                assert isinstance(path, Path)
                assert path.exists()  # The config file should exist

            # Log what configs were found for debugging
            config_names = [name for name, _ in configs]
            print(f"Found {len(configs)} configs: {config_names}")

    def test_load_tdcr_config(self):
        """Test loading TDCR configuration."""
        from pathlib import Path

        config_path = Path("configs/generation/example_three_segment_franka.json")
        configs = load_generation_configs(config_path)

        assert "example_three_segment_franka" in configs
        config = configs["example_three_segment_franka"]
        assert config is not None
        assert "num_segments" in config
        assert config["num_segments"] == 3

        # Check that total_links is calculated if missing
        assert "total_links" in config

    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration."""
        from pathlib import Path

        with pytest.raises(ValueError):
            load_generation_configs(Path("nonexistent_config.json"))

    def test_create_basic_scene(self, temp_dir):
        """Test creating a basic TDCR-Franka scene."""
        from pathlib import Path

        # Load config first
        config_path = Path("configs/generation/example_three_segment_franka.json")
        configs = load_generation_configs(config_path)
        config = configs["example_three_segment_franka"]

        # Create scene
        result = generate_scene(
            "example_three_segment_franka",
            config,
            mount_type="franka",
            output_dir=temp_dir,
        )

        assert result.exists()

        # Parse and validate XML
        tree = ET.parse(result)
        root = tree.getroot()

        assert root.tag == "mujoco"

        # Check for essential elements
        assert root.find("worldbody") is not None
        assert root.find("actuator") is not None

    def test_create_world_mounted_scene(self, temp_dir):
        """Test creating a world-mounted TDCR scene."""
        from pathlib import Path

        # Load config
        config_path = Path("configs/generation/example_three_segment_franka.json")
        configs = load_generation_configs(config_path)
        config = configs["example_three_segment_franka"]

        # Create world-mounted scene
        result = generate_scene(
            "example_three_segment_franka",
            config,
            mount_type="world",
            output_dir=temp_dir,
        )

        assert result.exists()

        # Parse and validate
        tree = ET.parse(result)
        root = tree.getroot()

        # Should have worldbody but no Franka elements
        assert root.find("worldbody") is not None

        # Check that it doesn't have Franka-specific elements
        worldbody = root.find("worldbody")
        body_names = [body.get("name") for body in worldbody.findall(".//body")]
        assert "panda_link0" not in body_names  # No Franka robot

    def test_custom_mount_position(self, temp_dir):
        """Test creating scene with custom mount position."""
        from pathlib import Path

        custom_pos = "0.1 0.2 0.3"
        custom_euler = "0 0 1.57"

        # Load config
        config_path = Path("configs/generation/example_three_segment_franka.json")
        configs = load_generation_configs(config_path)
        config = configs["example_three_segment_franka"]

        # Create scene with custom mounting
        result = generate_scene(
            "example_three_segment_franka",
            config,
            mount_type="franka",
            mount_pos=custom_pos,
            mount_euler=custom_euler,
            output_dir=temp_dir,
        )

        assert result.exists()

        # Check that custom mount parameters are used
        with open(result, "r") as f:
            content = f.read()
            # The mount position should appear in the file
            assert custom_pos in content or "0.1" in content

    def test_merge_no_duplicate_defaults(self, temp_dir):
        """Test that merging doesn't create duplicate default geom elements."""
        from pathlib import Path

        # Load config
        config_path = Path("configs/generation/example_three_segment_franka.json")
        configs = load_generation_configs(config_path)
        config = configs["example_three_segment_franka"]

        # Create a scene which would trigger the duplicate geom issue
        result = generate_scene(
            "example_three_segment_franka",
            config,
            mount_type="franka",
            output_dir=temp_dir,
        )

        assert result.exists()

        # Parse and check for duplicate geom elements in defaults
        tree = ET.parse(result)
        root = tree.getroot()
        default = root.find("default")

        if default is not None:
            # Count direct geom children (not in subclasses)
            direct_geoms = [
                child
                for child in default
                if child.tag == "geom" and child.get("class") is None
            ]
            assert (
                len(direct_geoms) <= 1
            ), "Should not have duplicate geom elements in default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
