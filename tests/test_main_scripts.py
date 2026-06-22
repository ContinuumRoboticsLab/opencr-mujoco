#!/usr/bin/env python3
"""
Automated tests for main scripts: viewer.py, teleop.py, evaluate.py
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
import pytest


class TestMainScripts:
    """Test suite for main scripts"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.repo_root = Path(__file__).parent.parent
        cls.test_output_dir = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        """Clean up test outputs"""
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)

    def run_script(self, script_name, args=None, timeout=30):
        """Helper to run a script and capture output"""
        cmd = [sys.executable, str(self.repo_root / script_name)]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.repo_root),
            )
            return result
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            timeout_msg = f"Timed out after {timeout}s: {' '.join(cmd)}"
            return subprocess.CompletedProcess(
                cmd,
                124,
                stdout=stdout,
                stderr=(stderr + "\n" + timeout_msg).strip(),
            )

    def test_viewer_help(self):
        """Test viewer.py --help"""
        result = self.run_script("viewer.py", ["--help"])
        assert result.returncode == 0
        assert "Basic MuJoCo viewer for opencr-mujoco scenes" in result.stdout

    def test_viewer_list_configs(self):
        """Test viewer.py --list-configs"""
        result = self.run_script("viewer.py", ["--list-configs"])
        assert result.returncode == 0
        assert "Available viewer configs:" in result.stdout
        assert "franka" in result.stdout

    def test_viewer_show_config(self):
        """Test viewer.py --show-config"""
        result = self.run_script("viewer.py", ["--show-config"])
        assert result.returncode == 0
        assert "Configuration (viewer):" in result.stdout
        assert "scene" in result.stdout

    def test_viewer_headless(self):
        """Test viewer.py in headless mode"""
        result = self.run_script(
            "viewer.py", ["--headless", "--duration", "1"], timeout=5
        )
        assert result is not None
        assert result.returncode == 0
        assert "Scene loaded successfully" in result.stdout

    def test_teleop_help(self):
        """Test teleop.py --help"""
        result = self.run_script("teleop.py", ["--help"])
        assert result.returncode == 0
        assert "Teleoperation interface for opencr-mujoco" in result.stdout

    def test_teleop_list_configs(self):
        """Test teleop.py --list-configs"""
        result = self.run_script("teleop.py", ["--list-configs"])
        assert result.returncode == 0
        assert "Available teleop configs:" in result.stdout
        assert "franka_keyboard_ik" in result.stdout

    def test_teleop_show_config(self):
        """Test teleop.py --show-config"""
        result = self.run_script("teleop.py", ["--show-config"])
        assert result.returncode == 0
        assert "Configuration (teleop):" in result.stdout
        assert "input_device" in result.stdout

    def test_teleop_keyboard_headless(self):
        """Test teleop.py with keyboard in headless mode"""
        result = self.run_script(
            "teleop.py",
            ["--config", "franka_keyboard_ik", "--headless", "--duration", "1"],
            timeout=5,
        )
        assert result is not None
        assert result.returncode == 0
        assert "Initializing keyboard input device" in result.stdout

    def test_evaluate_help(self):
        """Test evaluate.py --help"""
        result = self.run_script("paper_results/evaluate.py", ["--help"])
        assert result.returncode == 0
        assert "Evaluation interface for TDCR system" in result.stdout

    def test_evaluate_list_configs(self):
        """Test evaluate.py --list-configs"""
        result = self.run_script("paper_results/evaluate.py", ["--list-configs"])
        assert result.returncode == 0
        assert "Available evaluation configs:" in result.stdout
        assert "spring_steel_statics" in result.stdout

    def test_evaluate_show_config(self):
        """Test evaluate.py --show-config"""
        result = self.run_script("paper_results/evaluate.py", ["--show-config"])
        assert result.returncode == 0
        assert "Configuration (evaluation):" in result.stdout
        assert "n_values" in result.stdout

    def test_evaluate_minimal(self):
        """Test paper_results/evaluate.py with a minimal SoroSim statics run"""
        # spring_steel_statics, capped to 2 shapes at N=25 for a quick smoke run
        result = self.run_script(
            "paper_results/evaluate.py",
            [
                "--config",
                "spring_steel_statics",
                "--n-values",
                "25",
                "--early-stop",
                "2",
                "--no-visualize",
                "--output-dir",
                os.path.join(self.test_output_dir, "eval_test"),
            ],
            timeout=60,
        )

        # Check if it ran successfully
        assert result is not None
        assert result.returncode == 0
        assert "Running parameter sweep" in result.stdout
        assert "Results saved" in result.stdout

    def test_config_saving(self):
        """Test config saving functionality"""
        # Test viewer config saving
        import uuid

        config_name = f"test_viewer_config_{uuid.uuid4().hex[:8]}"
        result = self.run_script(
            "viewer.py",
            ["--save-config", config_name, "--scene", "assets/franka_scene.xml"],
            timeout=5,
        )
        assert result.returncode == 0
        assert f"Configuration saved as '{config_name}'" in result.stdout

        # Verify config was saved
        config_path = self.repo_root / "configs" / "viewer" / f"{config_name}.json"
        assert config_path.exists()

        # Clean up
        config_path.unlink()

    def test_invalid_config(self):
        """Test handling of invalid configurations"""
        result = self.run_script("viewer.py", ["--config", "nonexistent_config"])
        assert result.returncode != 0
        assert "not found" in result.stderr or "not found" in result.stdout

    def test_parameter_override(self):
        """Test parameter override functionality"""
        result = self.run_script(
            "paper_results/evaluate.py",
            [
                "--config",
                "spring_steel_statics",
                "--n-values",
                "25",
                "50",
                "--show-config",
            ],
            timeout=15,
        )
        assert result.returncode == 0
        assert "[25, 50]" in result.stdout

    @pytest.mark.parametrize("demo", ["tdcr", "kick"])
    def test_trace_tip_demo_records_gif(self, tmp_path, demo):
        """The README-GIF demos run and write a GIF (incl. scene injection)."""
        out = tmp_path / "demo.gif"
        result = self.run_script(
            "examples/trace_tip_demo.py",
            [
                "--demo",
                demo,
                "--record",
                str(out),
                "--seconds",
                "1.2",
                "--size",
                "160x120",
                "--fps",
                "5",
            ],
            timeout=120,
        )
        if result.returncode != 0 and (
            "gl" in result.stderr.lower() or "render" in result.stderr.lower()
        ):
            pytest.skip(f"no offscreen GL context: {result.stderr[-200:]}")
        assert result.returncode == 0, result.stderr
        assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
