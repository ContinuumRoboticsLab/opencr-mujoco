#!/usr/bin/env python3
"""Unit tests for viewer.py MuJoCo scene visualization."""

import pytest
import tempfile
from pathlib import Path
import sys
import os
import json
import subprocess
import time

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mujoco


class TestViewer:
    """Test MuJoCo viewer functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def simple_scene_xml(self, temp_dir):
        """Create a simple MuJoCo scene for testing."""
        scene_content = """
        <mujoco>
            <option timestep="0.002" gravity="0 0 -9.81"/>
            <worldbody>
                <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
                <body name="box" pos="0 0 0.5">
                    <joint type="free"/>
                    <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        scene_path = temp_dir / "simple_scene.xml"
        scene_path.write_text(scene_content)
        return scene_path

    @pytest.fixture
    def tdcr_scene_xml(self, temp_dir):
        """Create a TDCR scene with pretension keyframe."""
        scene_content = """
        <mujoco>
            <option timestep="0.002" gravity="0 0 -9.81"/>
            <worldbody>
                <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
                <body name="tdcr_base" pos="0 0 0.1">
                    <joint name="joint_0" type="hinge" axis="1 0 0"/>
                    <geom type="cylinder" size="0.01 0.05" rgba="0 1 0 1"/>
                </body>
            </worldbody>
            <actuator>
                <motor joint="joint_0" gear="1"/>
            </actuator>
            <keyframe>
                <key name="pretension" qpos="0.1"/>
            </keyframe>
        </mujoco>
        """
        scene_path = temp_dir / "tdcr_scene.xml"
        scene_path.write_text(scene_content)
        return scene_path

    @pytest.fixture
    def viewer_config(self, temp_dir, simple_scene_xml):
        """Create a viewer configuration file."""
        config = {
            "scene": str(simple_scene_xml),
            "description": "Test viewer configuration",
        }
        config_path = temp_dir / "viewer_config.json"
        config_path.write_text(json.dumps(config))
        return config_path

    def test_load_simple_scene(self, simple_scene_xml):
        """Test loading a simple MuJoCo scene."""
        # Load the model
        model = mujoco.MjModel.from_xml_path(str(simple_scene_xml))
        data = mujoco.MjData(model)

        # Check model properties
        assert model.nbody > 0
        assert model.ngeom > 0
        assert model.opt.timestep == 0.002
        assert model.opt.gravity[2] == -9.81

    def test_load_tdcr_scene_with_pretension(self, tdcr_scene_xml):
        """Test loading TDCR scene and applying pretension keyframe."""
        # Load the model
        model = mujoco.MjModel.from_xml_path(str(tdcr_scene_xml))
        data = mujoco.MjData(model)

        # Check for pretension keyframe
        pretension_found = False
        for i in range(model.nkey):
            key_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name == "pretension":
                # Apply the keyframe
                mujoco.mj_resetDataKeyframe(model, data, i)
                pretension_found = True
                break

        assert pretension_found
        # After applying pretension, joint position should be set
        assert data.qpos[0] == pytest.approx(0.1)

    def test_viewer_headless_mode(self, simple_scene_xml):
        """Test viewer in headless mode."""
        # Run viewer in headless mode
        cmd = [
            sys.executable,
            "viewer.py",
            "--scene",
            str(simple_scene_xml),
            "--headless",
            "--duration",
            "0.1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        # Check that it ran successfully
        assert result.returncode == 0
        assert "Scene loaded successfully" in result.stdout

    def test_viewer_with_config(self, viewer_config):
        """Test viewer with configuration file."""
        # Create a config in the expected location
        config_dir = Path("configs/viewer")
        config_dir.mkdir(parents=True, exist_ok=True)

        # Copy config to expected location
        import shutil

        shutil.copy(viewer_config, config_dir / f"{viewer_config.stem}.json")

        # Run viewer with config in headless mode
        cmd = [
            sys.executable,
            "viewer.py",
            "--config",
            viewer_config.stem,
            "--headless",
            "--duration",
            "0.1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        # Clean up
        (config_dir / f"{viewer_config.stem}.json").unlink(missing_ok=True)

        # Check that config was loaded
        assert result.returncode == 0
        assert "Loading scene" in result.stdout

    def test_scene_info_extraction(self, simple_scene_xml):
        """Test extraction of scene information."""
        model = mujoco.MjModel.from_xml_path(str(simple_scene_xml))

        # Extract scene info
        info = {
            "num_bodies": model.nbody,
            "num_joints": model.njnt,
            "num_geoms": model.ngeom,
            "num_actuators": model.nu,
            "timestep": model.opt.timestep,
        }

        assert info["num_bodies"] >= 1  # At least the box body
        assert info["num_geoms"] >= 2  # Plane and box
        assert info["timestep"] == 0.002

    def test_multiple_scene_formats(self, temp_dir):
        """Test loading different scene formats."""
        scenes = [
            (
                "minimal",
                """
            <mujoco>
                <worldbody>
                    <geom type="plane" size="1 1 0.1"/>
                </worldbody>
            </mujoco>
            """,
            ),
            (
                "with_light",
                """
            <mujoco>
                <worldbody>
                    <light pos="0 0 2"/>
                    <geom type="plane" size="1 1 0.1"/>
                </worldbody>
            </mujoco>
            """,
            ),
            (
                "with_camera",
                """
            <mujoco>
                <worldbody>
                    <camera name="cam" pos="2 2 2" xyaxes="-1 1 0 -1 -1 2"/>
                    <geom type="plane" size="1 1 0.1"/>
                </worldbody>
            </mujoco>
            """,
            ),
        ]

        for name, content in scenes:
            scene_path = temp_dir / f"{name}.xml"
            scene_path.write_text(content)

            # Try to load each scene
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)

            assert model is not None
            assert data is not None

    def test_simulation_stepping(self, simple_scene_xml):
        """Test simulation stepping in headless mode."""
        model = mujoco.MjModel.from_xml_path(str(simple_scene_xml))
        data = mujoco.MjData(model)

        initial_time = data.time

        # Step simulation
        for _ in range(100):
            mujoco.mj_step(model, data)

        # Check that time has advanced
        assert data.time > initial_time
        assert data.time == pytest.approx(100 * model.opt.timestep)

    def test_invalid_scene_handling(self, temp_dir):
        """Test handling of invalid scene files."""
        # Create an invalid XML file
        invalid_xml = temp_dir / "invalid.xml"
        invalid_xml.write_text("This is not valid XML")

        # Try to load it
        with pytest.raises(Exception):
            model = mujoco.MjModel.from_xml_path(str(invalid_xml))

    def test_nonexistent_scene_handling(self):
        """Test handling of non-existent scene files."""
        # Run viewer with non-existent scene
        cmd = [
            sys.executable,
            "viewer.py",
            "--scene",
            "nonexistent_file.xml",
            "--headless",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        # Should fail gracefully
        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
