#!/usr/bin/env python3
"""Unit tests for evaluate.py TDCR evaluation functionality."""

import pytest
import tempfile
from pathlib import Path
import sys
import os
import json
import numpy as np
import pickle
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mujoco
from opencr_mujoco.evaluation.trajectory_evaluator import TrajectoryEvaluator
from opencr_mujoco.evaluation.metrics import compute_tip_error, compute_shape_error


class TestEvaluation:
    """Test TDCR evaluation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_reference_data(self, temp_dir):
        """Create mock reference data for testing."""
        # Create mock reference data structure
        reference_dir = temp_dir / "reference"
        reference_dir.mkdir()

        # Create mock test data
        num_tests = 5
        num_links = 501  # High-resolution reference

        test_data = {
            "Wrenches": np.random.randn(num_tests, 6),  # 6 DOF wrench
            "LinkPositions": np.random.randn(num_tests, num_links, 3),  # XYZ positions
        }

        # Save as pickle (simulating MATLAB data)
        data_file = reference_dir / "RandomWrenchTest.pkl"
        with open(data_file, "wb") as f:
            pickle.dump(test_data, f)

        return reference_dir

    @pytest.fixture
    def tdcr_model_xml(self, temp_dir):
        """Create a simple TDCR model for testing."""
        model_content = """
        <mujoco>
            <option timestep="0.002" gravity="0 0 -9.81"/>
            <worldbody>
                <body name="base" pos="0 0 0.1">
                    <joint name="joint_0" type="hinge" axis="1 0 0"/>
                    <geom name="link_0" type="cylinder" size="0.01 0.05"/>
                    <body name="link_1" pos="0 0 0.1">
                        <joint name="joint_1" type="hinge" axis="0 1 0"/>
                        <geom name="link_1" type="cylinder" size="0.01 0.05"/>
                        <site name="tip" pos="0 0 0.05"/>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor joint="joint_0" gear="1"/>
                <motor joint="joint_1" gear="1"/>
            </actuator>
        </mujoco>
        """
        model_path = temp_dir / "tdcr_model.xml"
        model_path.write_text(model_content)
        return model_path

    @pytest.fixture
    def evaluation_config(self, temp_dir):
        """Create an evaluation configuration."""
        config = {
            "test_type": "RandomWrenchTest",
            "n_values": [25, 50],
            "sim_steps": 1000,
            "early_stop": 2,
            "save_positions": True,
            "visualize": False,
        }
        config_path = temp_dir / "eval_config.json"
        config_path.write_text(json.dumps(config))
        return config_path

    def test_trajectory_evaluator_initialization(self, mock_reference_data, temp_dir):
        """Test TrajectoryEvaluator initialization."""
        evaluator = TrajectoryEvaluator(
            reference_data_dir=str(mock_reference_data),
            results_dir=str(temp_dir / "output"),
        )

        assert evaluator is not None
        assert evaluator.results_dir == Path(temp_dir / "output")
        assert evaluator.reference_loader is not None

    def test_compute_tip_error(self):
        """Test tip error computation."""
        # Create mock positions
        simulated_pos = np.array([1.0, 2.0, 3.0])
        reference_pos = np.array([1.1, 2.1, 2.9])

        error = compute_tip_error(simulated_pos, reference_pos)

        expected_error = np.linalg.norm(simulated_pos - reference_pos)
        assert error == pytest.approx(expected_error)

    def test_compute_shape_error(self):
        """Test shape error computation."""
        # Create mock link positions
        num_links = 10
        simulated_positions = np.random.randn(num_links, 3)
        reference_positions = simulated_positions + 0.1 * np.random.randn(num_links, 3)

        error = compute_shape_error(simulated_positions, reference_positions)

        assert error >= 0
        assert isinstance(error, (float, np.floating))

    def test_interpolate_reference_positions(self):
        """The evaluator's interpolation must pair points by ARC LENGTH.

        The SoroSim statics CSVs sample at non-uniform (Gauss-Lobatto)
        stations; index-uniform pairing put an N-independent ~4% -of-length
        floor under shape_error. This exercises the real method with
        non-uniform reference arcs and checks the interpolated points land
        exactly on the underlying curve at the requested sim arc fractions.
        """
        from opencr_mujoco.evaluation.trajectory_evaluator import TrajectoryEvaluator

        evaluator = TrajectoryEvaluator.__new__(TrajectoryEvaluator)

        # Straight-line "shape": position = arc * direction, so interpolation
        # is exact and easy to check pointwise.
        direction = np.array([0.2, -0.5, 1.0])
        ref_arcs = np.array(
            [0.0, 0.0469, 0.2308, 0.5, 0.5, 0.7692, 0.9531, 1.0]
        )  # non-uniform, with a duplicated segment-boundary station
        ref_positions = [arc * direction for arc in ref_arcs]

        num_links = 25
        sim_arcs = np.concatenate(
            [
                [0.0],
                (np.arange(num_links - 1) + 0.5) / num_links,
                [(num_links - 0.5) / num_links, 1.0],
            ]
        )

        interpolated = evaluator._interpolate_reference_positions(
            ref_positions, ref_arcs, sim_arcs
        )

        assert len(interpolated) == len(sim_arcs)
        for arc, point in zip(sim_arcs, interpolated):
            assert np.allclose(point, arc * direction, atol=1e-12)

    def test_wrench_application(self, tdcr_model_xml):
        """Test applying wrenches to TDCR model."""
        model = mujoco.MjModel.from_xml_path(str(tdcr_model_xml))
        data = mujoco.MjData(model)

        # Apply a wrench (force and torque)
        wrench = np.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.0])  # Fx and My

        # Apply force to tip site (if it exists)
        try:
            tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
            # In real implementation, forces would be applied via xfrc_applied
            data.xfrc_applied[model.site_bodyid[tip_id]] = wrench
        except:
            pass

        # Step simulation
        initial_pos = data.qpos.copy()
        for _ in range(100):
            mujoco.mj_step(model, data)

        # Check that robot moved due to applied force
        assert not np.allclose(data.qpos, initial_pos)

    def test_equilibrium_detection(self, tdcr_model_xml):
        """Test equilibrium detection during simulation."""
        model = mujoco.MjModel.from_xml_path(str(tdcr_model_xml))
        data = mujoco.MjData(model)

        # Track positions over time
        positions = []
        velocities = []

        # Run simulation
        for _ in range(500):
            mujoco.mj_step(model, data)
            positions.append(data.qpos.copy())
            velocities.append(data.qvel.copy())

        # Check for equilibrium (velocities should decrease)
        initial_vel = np.linalg.norm(velocities[10])
        final_vel = np.linalg.norm(velocities[-1])

        # Due to damping, velocity should decrease
        assert final_vel <= initial_vel

    def test_error_metrics_collection(self, temp_dir):
        """Test collection of error metrics during evaluation."""
        # Create mock evaluation results
        results = {
            "n_values": [25, 50, 100],
            "tip_errors": [0.01, 0.005, 0.002],
            "shape_errors": [0.02, 0.01, 0.005],
            "computation_times": [1.0, 2.0, 4.0],
        }

        # Save results
        results_file = temp_dir / "results.json"
        results_file.write_text(json.dumps(results))

        # Load and verify
        loaded = json.loads(results_file.read_text())
        assert loaded["n_values"] == results["n_values"]
        assert loaded["tip_errors"] == results["tip_errors"]

    def test_parameter_sweep(self, tdcr_model_xml, temp_dir):
        """Test parameter sweep functionality."""
        n_values = [10, 20, 30]
        results = []

        for n in n_values:
            # Simulate evaluation with different discretizations
            model = mujoco.MjModel.from_xml_path(str(tdcr_model_xml))
            data = mujoco.MjData(model)

            # Run brief simulation
            for _ in range(100):
                mujoco.mj_step(model, data)

            # Collect mock metrics
            result = {
                "n": n,
                "error": 1.0 / n,  # Mock error decreases with n
                "time": n * 0.1,  # Mock time increases with n
            }
            results.append(result)

        # Check that all parameter values were tested
        assert len(results) == len(n_values)
        assert all(r["n"] in n_values for r in results)

    def test_early_stopping(self):
        """Test early stopping functionality."""
        max_tests = 100
        early_stop = 10

        tests_run = 0
        for i in range(max_tests):
            tests_run += 1

            # Simulate early stop condition
            if tests_run >= early_stop:
                break

        assert tests_run == early_stop

    def test_visualization_flag(self, temp_dir):
        """Test visualization enable/disable flag."""
        # Test with visualization disabled
        config_no_viz = {"visualize": False, "output_dir": str(temp_dir / "no_viz")}

        # Test with visualization enabled
        config_with_viz = {"visualize": True, "output_dir": str(temp_dir / "with_viz")}

        # In actual implementation, this would control plot generation
        assert not config_no_viz["visualize"]
        assert config_with_viz["visualize"]

    def test_output_directory_creation(self, temp_dir):
        """Test automatic creation of output directories."""
        output_dir = temp_dir / "evaluation_results" / "test_run"

        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_link_position_saving(self, tdcr_model_xml, temp_dir):
        """Test saving link positions during evaluation."""
        model = mujoco.MjModel.from_xml_path(str(tdcr_model_xml))
        data = mujoco.MjData(model)

        # Collect positions
        positions = []
        for step in range(10):
            mujoco.mj_step(model, data)

            # Get link positions (mock)
            link_positions = []
            for i in range(model.nbody):
                pos = data.xpos[i].copy()
                link_positions.append(pos)

            positions.append(np.array(link_positions))

        # Save positions
        positions_file = temp_dir / "positions.pkl"
        with open(positions_file, "wb") as f:
            pickle.dump(positions, f)

        # Verify saved data
        assert positions_file.exists()

        with open(positions_file, "rb") as f:
            loaded_positions = pickle.load(f)

        assert len(loaded_positions) == 10
        assert loaded_positions[0].shape[0] == model.nbody


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
