#!/usr/bin/env python3
"""Correctness tests for the SoroSim evaluation pipeline.

Lock the numerical conventions the published results depend on:

- the statics loader exposes the (non-uniform) arc-length stations and the
  full (mid, tip, gravity)-keyed shape bank
- the dynamics loader uses the stored moments AS-IS (a docstring once claimed
  they were negated; "fixing" the code to match it would corrupt every TPU
  dynamics result by ~100x)
- wrenches applied at a site reduce to the correct force + moment about the
  carrying body's center of mass (xfrc_applied acts at the COM, not the
  body frame origin)
- the simulation's backbone sample points carry the documented arc fractions

Run nightly alongside test_generator_physics.py.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

mujoco = pytest.importorskip("mujoco")

from opencr_mujoco.evaluation.reference_data_loader import ReferenceDataLoader  # noqa: E402
from opencr_mujoco.evaluation.trajectory_evaluator import TrajectoryEvaluator  # noqa: E402

SOROSIM_DIR = PROJECT_ROOT / "data" / "reference" / "sorosim"
FRAME = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]


@pytest.fixture(scope="module")
def loaded_statics():
    loader = ReferenceDataLoader(SOROSIM_DIR, frame_conversion=FRAME)
    return loader.load_sorosim_statics_csv("SpringSteelRodMuJoCo")


@pytest.mark.skipif(not SOROSIM_DIR.exists(), reason="reference data not present")
class TestStaticsLoader:
    def test_arc_lengths_are_normalized_and_monotone(self, loaded_statics):
        _, num_samples, arc_lengths = loaded_statics
        assert num_samples == len(arc_lengths)
        assert arc_lengths[0] == 0.0 and np.isclose(arc_lengths[-1], 1.0)
        assert np.all(np.diff(arc_lengths) >= -1e-12)  # non-decreasing
        # The SoroSim stations are Gauss-Lobatto, i.e. NOT uniformly spaced —
        # the property the arc-length interpolation fix exists for.
        assert np.std(np.diff(arc_lengths)) > 1e-3

    def test_full_shape_bank_loads(self, loaded_statics):
        data_dict, num_samples, _ = loaded_statics
        assert len(data_dict) == 500  # 500 distinct shapes per material
        key = next(iter(data_dict))
        mid_wrench, tip_wrench, gravity = key
        assert len(mid_wrench) == 6 and len(tip_wrench) == 6 and len(gravity) == 3
        assert all(len(p) == 3 for p in data_dict[key])
        assert len(data_dict[key]) == num_samples


class TestDynamicsLoaderMomentConvention:
    def test_moments_are_used_as_stored(self, tmp_path):
        """Line 2 stores the applied moments AS-IS (no sign flip).

        Verified against the held t=0 equilibrium of the TPU tests (~5 mm
        as-coded vs ~560 mm if negated). This test pins the convention with
        a synthetic file so a future "cleanup" cannot silently flip it.
        """
        mid_m, mid_f = [0.011, 0.012, 0.013], [0.21, 0.22, 0.23]
        tip_m, tip_f = [0.031, 0.032, 0.033], [0.41, 0.42, 0.43]
        line1 = [0.005] + [0.0] * 9 + [0.0, 0.0, -9.81]
        line2 = [0.0] + mid_m + mid_f + tip_m + tip_f
        line3 = [0.0] + [0.0] * 12
        line4 = [0.005] + [0.0] * 12
        content = "\n".join(
            "\t".join(str(v) for v in row) for row in [line1, line2, line3, line4]
        )
        (tmp_path / "SyntheticTest_1.txt").write_text(content + "\n")

        loader = ReferenceDataLoader(tmp_path)  # identity frame
        (mid_wrench, tip_wrench), _, dt, _, damping, gravity = (
            loader.load_tip_release_data("SyntheticTest_1")
        )

        assert np.allclose(mid_wrench, mid_f + mid_m)  # [f(3), m(3)], as stored
        assert np.allclose(tip_wrench, tip_f + tip_m)
        assert np.isclose(damping, 0.005)
        assert np.allclose(gravity, [0.0, 0.0, -9.81])
        assert np.isclose(dt, 0.005)


class TestWrenchAtSite:
    def test_moment_arm_is_about_body_com(self):
        """xfrc_applied acts at the COM (xipos), so a force applied at a site
        must add the moment (site - COM) x F — not (site - frame origin) x F."""
        xml = """
        <mujoco>
          <worldbody>
            <body name="rod" pos="0 0 0">
              <joint type="free"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.01"/>
              <site name="tip_site" pos="0 0 0.2" size="0.001"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        evaluator = TrajectoryEvaluator.__new__(TrajectoryEvaluator)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
        body_id = model.site_bodyid[site_id]

        force = np.array([1.0, 0.0, 0.0])
        wrench = np.concatenate([force, np.zeros(3)])
        data.xfrc_applied[:] = 0.0
        evaluator._apply_wrench_at_site(model, data, site_id, wrench)

        # The capsule spans z in [0, 0.2] with COM at z=0.1; the site sits at
        # the tip, so the moment about the COM is (0,0,0.1) x (1,0,0) = (0,0.1,0)
        r = data.site_xpos[site_id] - data.xipos[body_id]
        expected_moment = np.cross(r, force)
        assert np.allclose(data.xfrc_applied[body_id, :3], force)
        assert np.allclose(data.xfrc_applied[body_id, 3:], expected_moment)
        assert np.allclose(expected_moment, [0.0, 0.1, 0.0], atol=1e-12)


@pytest.mark.skipif(not SOROSIM_DIR.exists(), reason="reference data not present")
class TestDynamicsEndToEnd:
    def test_tip_release_runs_and_matches_reference_scale(self, tmp_path):
        """One real tip-release evaluation, shortened ramp/hold.

        Regression for two failure modes: (1) the dynamics code path crashing
        while statics stays green (a function-local `import mujoco.viewer`
        once shadowed the module-level `mujoco` for the whole function), and
        (2) gross physics drift — the mean tip error against the SoroSim
        reference must stay at paper scale (well under 1% of rod length).
        """
        import json

        from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config

        # Use the SHIPPED evaluation config's generator parameters so the test
        # tracks exactly what the paper pipeline simulates.
        eval_cfg = json.load(
            open(PROJECT_ROOT / "configs/evaluation/sorosim_dynamics_steel.json")
        )
        cfg = dict(eval_cfg["generator_config"])
        n = 25
        cfg.update(
            {
                "num_segments": 1,
                "links_per_segment": {"1": n},
                "segment_lengths": {"1": cfg["total_length"]},
                "total_links": n,
            }
        )
        model_path = str(tmp_path / "rod25.xml")
        create_tdcr_from_config(cfg, model_path)

        evaluator = TrajectoryEvaluator(
            SOROSIM_DIR / "sorosim_dynamics",
            str(tmp_path / "results"),
            frame_conversion=FRAME,
        )
        result = evaluator.evaluate_tip_release(
            model_path,
            num_links=25,
            test_type="SpringSteelRodMuJoCo_1",
            sim_hz=500.0,
            show_progress=False,
            force_ramp_time=5.0,
            hold_time=2.0,
        )

        mean_err = result["mean_position_error"]
        assert np.isfinite(mean_err)
        # 0.6 m rod: paper-scale accuracy is ~1 mm; 6 mm (1% of L) is the
        # gross-regression tripwire.
        assert (
            mean_err < 0.006
        ), f"mean tip error {mean_err*1000:.2f} mm exceeds 1% of rod length"
        # t=0 must be the held equilibrium: the first captured shape frame is
        # the pre-release sample at exactly t=0 (no one-step lead).
        assert result["sim_link_shape_times"][0] == 0.0


class TestSimSamplePoints:
    def test_collected_arc_fractions_match_geometry(self):
        """The backbone sample arcs must equal the points' true arc positions."""
        from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config

        n, length = 20, 0.4
        cfg = {
            "num_segments": 1,
            "links_per_segment": {"1": n},
            "segment_lengths": {"1": length},
            "total_links": n,
            "total_length": length,
            "radius": 0.005,
            "joints_per_link": 3,
            "joint_config_mode": "material",
            "material_properties": {
                "density": 7800,
                "youngs_modulus": 200e9,
                "poisson_ratio": 0.3,
                "outer_radius": 0.0005,
                "damping_ratio": 0.05,
            },
            "actuation_mode": "none",
            "gravity": "0 0 0",
            "plane": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            xml = str(Path(tmp) / "rod.xml")
            create_tdcr_from_config(cfg, xml)
            model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        evaluator = TrajectoryEvaluator.__new__(TrajectoryEvaluator)
        positions, arcs = evaluator._collect_link_positions(model, data, n)

        assert len(positions) == len(arcs) == n + 2
        assert arcs[0] == 0.0 and arcs[-1] == 1.0
        # In the straight rest pose, each sample's z must equal arc * L
        for arc, pos in zip(arcs, positions):
            assert np.isclose(
                pos[2], arc * length, atol=1e-9
            ), f"sample at arc {arc} sits at z={pos[2]}, expected {arc * length}"

    def test_perfect_match_gives_near_zero_shape_error(self):
        """If the reference IS the simulated shape (re-sampled at the real
        Gauss-Lobatto stations), shape_error must collapse to the tiny
        chord-linearization residual — the regression the arc-length
        interpolation fix addressed (index pairing had a large N-independent
        floor)."""
        from opencr_mujoco.evaluation.metrics import compute_shape_error

        evaluator = TrajectoryEvaluator.__new__(TrajectoryEvaluator)
        length = 0.4
        # A bent backbone: quarter circle of radius R (arc length = L)
        radius = 2 * length / np.pi
        # The real statics layout: 2 segments x 7 Gauss-Lobatto stations
        local = np.array([0.0, 0.0469, 0.2308, 0.5, 0.7692, 0.9531, 1.0])
        ref_arcs = np.concatenate([local / 2.0, 0.5 + local / 2.0])

        def curve(arc):
            phi = arc * length / radius
            return np.array([radius * (1 - np.cos(phi)), 0.0, radius * np.sin(phi)])

        ref_positions = [curve(a) for a in ref_arcs]

        n = 50
        sim_arcs = np.concatenate(
            [[0.0], (np.arange(n - 1) + 0.5) / n, [(n - 0.5) / n, 1.0]]
        )
        sim_positions = [curve(a) for a in sim_arcs]

        interp = evaluator._interpolate_reference_positions(
            ref_positions, ref_arcs, sim_arcs
        )
        error = compute_shape_error(sim_positions, interp)

        # What the old index-uniform pairing would have produced on the same
        # data: pair sim sample i with the reference linearly re-indexed.
        ref_array = np.asarray(ref_positions)
        idx = np.linspace(0, len(ref_arcs) - 1, len(sim_arcs))
        old_interp = np.stack(
            [
                np.interp(idx, np.arange(len(ref_arcs)), ref_array[:, dim])
                for dim in range(3)
            ],
            axis=1,
        )
        old_error = compute_shape_error(sim_positions, list(old_interp))

        assert error < 1e-3, f"shape_error {error*1000:.3f} mm should be ~0"
        assert error < old_error / 5, (
            f"arc-length pairing ({error*1000:.3f} mm) should beat index "
            f"pairing ({old_error*1000:.3f} mm) by a wide margin"
        )
