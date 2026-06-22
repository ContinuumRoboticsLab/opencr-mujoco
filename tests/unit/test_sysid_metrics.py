#!/usr/bin/env python3
"""Correctness tests for the sysid error metric.

There is ONE error convention in the sysid pipeline: the standard (Euclidean,
pointwise) RMSE. The optimizer objective, the printed train/val numbers, and
the plot annotations are all this same quantity, so these tests pin its exact
definition — in particular that it is the RMS of per-sample error NORMS, not
the per-coordinate RMS it used to be (which is exactly tip-RMSE/sqrt(3) and
was a long-standing source of confusion when quoted as accuracy).
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from opencr_mujoco.sysid.error_metrics import (  # noqa: E402
    RMSEMetric,
    create_metrics_from_config,
)


class TestRMSEMetric:
    def test_rmse_is_euclidean_not_per_coordinate(self):
        """A single sample with error vector (3, 4, 0) mm has tip error 5 mm.

        The Euclidean RMSE must report exactly 5; the old per-coordinate
        definition would report sqrt(25/3) = 2.887 — pinning this prevents a
        silent regression to the confusing 1.73x-smaller convention.
        """
        real = np.array([[0.0, 0.0, 0.0]])
        sim = np.array([[0.003, 0.004, 0.0]])

        rmse = RMSEMetric("rmse", 1.0).compute(real, sim)
        assert np.isclose(rmse, 0.005), f"expected 5 mm, got {rmse*1000:.3f} mm"

    def test_rmse_is_rms_over_samples(self):
        """Two samples with tip errors 3 mm and 4 mm: RMS = sqrt((9+16)/2)."""
        real = np.zeros((2, 3))
        sim = np.array([[0.003, 0.0, 0.0], [0.0, 0.004, 0.0]])

        rmse = RMSEMetric("rmse", 1.0).compute(real, sim)
        assert np.isclose(rmse, np.sqrt((0.003**2 + 0.004**2) / 2))

    def test_ignore_z(self):
        """With ignore_z, only the XY components count."""
        real = np.array([[0.0, 0.0, 0.0]])
        sim = np.array([[0.003, 0.004, 1.0]])  # huge z error, ignored

        rmse = RMSEMetric("rmse", 1.0, {"ignore_z": True}).compute(real, sim)
        assert np.isclose(rmse, 0.005)

    def test_zero_for_identical_trajectories(self):
        traj = np.random.default_rng(0).normal(size=(50, 3))
        assert RMSEMetric("rmse", 1.0).compute(traj, traj.copy()) == 0.0


class TestMetricCombiner:
    def test_default_is_rmse(self):
        combiner = create_metrics_from_config({})
        assert [m.name for m in combiner.metrics] == ["RMSE"]

    def test_combined_value_matches_single_metric(self):
        real = np.zeros((4, 3))
        sim = np.full((4, 3), 0.001)
        combiner = create_metrics_from_config({"rmse": {"weight": 2.0}})
        expected = RMSEMetric("rmse", 1.0).compute(real, sim)
        assert np.isclose(combiner.compute(real, sim), expected)
