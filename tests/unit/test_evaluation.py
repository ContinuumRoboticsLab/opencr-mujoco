#!/usr/bin/env python3
"""
Unit tests for the evaluation module.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from opencr_mujoco.evaluation import (
    ReferenceDataLoader,
    TrajectoryEvaluator,
    compute_tip_error,
    compute_shape_error,
    compute_trajectory_error,
)
from opencr_mujoco.evaluation.visualization import EvaluationVisualizer


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def test_tip_error(self):
        """Test tip error computation."""
        sim_pos = np.array([0.1, 0.2, 0.3])
        ref_pos = (0.11, 0.19, 0.31)

        error = compute_tip_error(sim_pos, ref_pos)
        expected = np.sqrt(0.01**2 + 0.01**2 + 0.01**2)
        self.assertAlmostEqual(error, expected, places=6)

    def test_shape_error(self):
        """Test shape error computation."""
        sim_positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
        ]
        ref_positions = [(0.0, 0.0, 0.0), (0.1, 0.01, 0.0), (0.2, 0.02, 0.0)]

        error = compute_shape_error(sim_positions, ref_positions)
        # Expected: mean of [0, 0.01, 0.02]
        expected = (0 + 0.01 + 0.02) / 3
        self.assertAlmostEqual(error, expected, places=6)

    def test_trajectory_error_mse(self):
        """Test trajectory error with MSE metric."""
        sim_traj = np.array([[0, 0], [1, 1], [2, 2]])
        ref_traj = np.array([[0, 0], [1, 0], [2, 1]])

        error = compute_trajectory_error(sim_traj, ref_traj, metric="mse")
        # Expected: mean of squared differences
        expected = (0 + 1 + 1) / 3
        self.assertAlmostEqual(error, expected, places=6)


class TestReferenceDataLoader(unittest.TestCase):
    """Test reference data loading."""

    def setUp(self):
        """Create temporary directory with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_wrench_string_parsing(self):
        """Test wrench string parsing and formatting."""
        loader = ReferenceDataLoader(self.temp_dir)

        wrench = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        wrench_str = loader.format_wrench_for_csv(wrench)
        parsed = loader.parse_wrench_string(wrench_str)

        self.assertEqual(wrench_str, "1.0,2.0,3.0,0.1,0.2,0.3")
        self.assertEqual(parsed, wrench)


class TestEvaluationVisualizerSmoke(unittest.TestCase):
    """Smoke tests for visualization (no actual plotting)."""

    def setUp(self):
        """Create temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = Path(self.temp_dir) / "results"
        self.plots_dir = Path(self.temp_dir) / "plots"

        # Create sample CSV data
        data = {
            "N": [25, 25, 50, 50],
            "test_type": ["Test"] * 4,
            "tip_error": [0.01, 0.02, 0.005, 0.008],
            "realtime_ratio": [10.0, 12.0, 5.0, 6.0],
            "mid_wrench_str": ["1,0,0,0,0,0"] * 4,
            "tip_wrench_str": ["0,1,0,0,0,0"] * 4,
        }

        self.csv_path = self.results_dir / "test_results.csv"
        self.results_dir.mkdir(parents=True)
        pd.DataFrame(data).to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_visualizer_creation(self):
        """Test visualizer can be created."""
        viz = EvaluationVisualizer(self.results_dir, self.plots_dir)
        self.assertTrue(self.plots_dir.exists())

    def test_summary_report(self):
        """Test summary report generation."""
        viz = EvaluationVisualizer(self.results_dir, self.plots_dir)
        report = viz.create_summary_report(self.csv_path, "test_config")

        self.assertIn("Total evaluations: 4", report)
        self.assertIn("N = 25:", report)
        self.assertIn("N = 50:", report)


if __name__ == "__main__":
    unittest.main()
