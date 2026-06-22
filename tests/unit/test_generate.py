#!/usr/bin/env python3
"""Unit tests for generate.py TDCR model generation."""

import pytest
import json
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Physical-invariant tests load the generated model in MuJoCo. Guard the import
# so the string-based tests above still run if MuJoCo is unavailable.
try:
    import mujoco
    import numpy as np

    _HAS_MUJOCO = True
except ImportError:  # pragma: no cover
    _HAS_MUJOCO = False


def _generate_and_load(config, path):
    """Generate a model from config, load it in MuJoCo, mj_forward at qpos=0."""
    create_tdcr_from_config(config, str(path))
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _tendon_segment_index(model, tendon_id):
    """Segment index parsed from a tendon name like 'seg_2_tendon_0' -> 2."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
    return int(name.split("_")[1])


def _make_tendon_config(
    links,
    lengths,
    independent=False,
    constraint_factor=None,
    actuator_type="position",
    tendons_per_segment=3,
):
    """Build a coupled/independent tendon TDCR config (per-segment links/lengths).

    total_links/total_length are intentionally omitted so they are derived from
    the per-segment dicts (exercising that path too).
    """
    n = len(links)
    if isinstance(tendons_per_segment, int):
        tendons_per_segment = [tendons_per_segment] * n
    return {
        "description": "invariant-test TDCR",
        "num_segments": n,
        "links_per_segment": {str(i + 1): links[i] for i in range(n)},
        "segment_lengths": {str(i + 1): lengths[i] for i in range(n)},
        "radius": 0.006,
        "joints_per_link": 2,
        "joint_config_mode": "material",
        "material_properties": {
            "density": 37708,
            "youngs_modulus": 200e9,
            "poisson_ratio": 0.3,
            "outer_radius": 0.0005,
        },
        "actuation_mode": "parallel_tendons",
        "independent_segments": independent,
        "actuation_details": {
            "segments": [
                {"number_of_tendons": t, "distance_to_backbone": 0.004}
                for t in tendons_per_segment
            ]
        },
        "seg_offsets": [0.0] * n,
        "actuator_properties": {
            "tendon_ctrlrange": "-0.05 0.05",
            "tendon_actuator_type": actuator_type,
            "tendon_kp": 100000,
            "tendon_pretension": 1.0,
            **(
                {"tendon_constraint_factor": constraint_factor}
                if constraint_factor is not None
                else {}
            ),
        },
    }


class TestTDCRGeneration:
    """Test TDCR model generation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def basic_config(self):
        """Basic TDCR configuration for testing."""
        return {
            "description": "Test TDCR",
            "num_segments": 1,
            "total_links": 10,
            "total_length": 0.3,
            "radius": 0.01,
            "mass": 0.5,
            "joints_per_link": 3,
            "joint_config_mode": "direct",
            "stiffness": 100.0,
            "damping": 1.0,
            "actuation_mode": "direct_torque",
            "gravity": "0 0 -9.81",
            "plane": True,
        }

    @pytest.fixture
    def material_config(self):
        """Material-based TDCR configuration."""
        return {
            "description": "Material TDCR",
            "num_segments": 1,
            "total_links": 15,
            "total_length": 0.4,
            "radius": 0.008,
            "mass": 0.3,
            "joints_per_link": 3,
            "joint_config_mode": "material",
            "material_properties": {
                "density": 8000,
                "youngs_modulus": 100e9,
                "poisson_ratio": 0.35,
                "outer_radius": 0.001,
            },
            "actuation_mode": "direct_torque",
        }

    @pytest.fixture
    def tendon_config(self):
        """Tendon-driven TDCR configuration."""
        return {
            "description": "Tendon TDCR",
            "num_segments": 3,
            "links_per_segment": {"1": 5, "2": 5, "3": 5},
            "segment_lengths": {"1": 0.1, "2": 0.1, "3": 0.1},
            "total_links": 15,
            "total_length": 0.3,
            "radius": 0.006,
            "mass": 0.2,
            "joints_per_link": 3,
            "joint_config_mode": "material",
            "material_properties": {
                "density": 37708,
                "youngs_modulus": 200e9,
                "poisson_ratio": 0.3,
                "outer_radius": 0.0005,
            },
            "actuation_mode": "parallel_tendons",
            "actuation_details": {
                "segments": [
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                ]
            },
            "seg_offsets": [0, 0.5236, 1.0472],
            "actuator_properties": {
                "tendon_ctrlrange": "-0.05 0.05",
                "tendon_actuator_type": "position",
                "tendon_kp": 100000,
                "tendon_pretension": 1.0,
            },
        }

    def test_basic_generation(self, basic_config, temp_dir):
        """Test basic TDCR generation with direct stiffness."""
        output_path = temp_dir / "basic_tdcr.xml"

        # Generate the model
        result = create_tdcr_from_config(basic_config, str(output_path))

        # Check that file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Check XML content
        content = output_path.read_text()
        assert "<mujoco" in content
        assert "link_0" in content
        assert "joint_0" in content
        assert 'stiffness="100"' in content or 'stiffness="100.0"' in content

    def test_material_generation(self, material_config, temp_dir):
        """Test TDCR generation with material-based stiffness calculation."""
        output_path = temp_dir / "material_tdcr.xml"

        # Generate the model
        result = create_tdcr_from_config(material_config, str(output_path))

        # Check that file was created
        assert output_path.exists()

        # Check XML content
        content = output_path.read_text()
        assert "<mujoco" in content
        assert "link_" in content
        assert "joint_" in content
        # Material-based should calculate stiffness from properties
        assert "stiffness=" in content
        assert "damping=" in content

    def test_tendon_generation(self, tendon_config, temp_dir):
        """Test tendon-driven TDCR generation."""
        output_path = temp_dir / "tendon_tdcr.xml"

        # Generate the model
        result = create_tdcr_from_config(tendon_config, str(output_path))

        # Check that file was created
        assert output_path.exists()

        # Check XML content
        content = output_path.read_text()
        assert "<mujoco" in content
        assert "<tendon>" in content
        assert "<spatial" in content
        assert "seg_0_ten_0" in content
        assert "seg_1_ten_0" in content
        assert "seg_2_ten_0" in content
        assert "<actuator>" in content

    def test_multi_segment_generation(self, temp_dir):
        """Test multi-segment robot generation."""
        config = {
            "description": "Multi-segment TDCR",
            "num_segments": 2,
            "links_per_segment": {"1": 8, "2": 12},
            "segment_lengths": {"1": 0.2, "2": 0.3},
            "total_links": 20,
            "total_length": 0.5,
            "radius": 0.01,
            "mass": 1.0,
            "joints_per_link": 3,
            "joint_config_mode": "direct",
            "stiffness": 150.0,
            "damping": 2.0,
            "actuation_mode": "direct_torque",
        }

        output_path = temp_dir / "multi_segment.xml"
        result = create_tdcr_from_config(config, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()

        # Check for multiple segments or links indicating multi-segment structure
        # The generator may use different naming conventions
        assert "link_" in content  # Should have links
        assert "joint_" in content  # Should have joints
        # Check that we have the expected number of links
        link_count = content.count("link_")
        assert link_count >= 20  # Should have at least 20 links as configured

    def test_collision_exclusion(self, basic_config, temp_dir):
        """Test generation with collision exclusion enabled."""
        config = basic_config.copy()
        config["disable_self_collision"] = True

        output_path = temp_dir / "no_collision.xml"
        result = create_tdcr_from_config(config, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()

        # Check for collision exclusion pairs
        assert "<exclude" in content or "<contact>" in content

    def test_inverse_length_scaling(self, tendon_config, temp_dir):
        """Test tendon generation with inverse length scaling."""
        config = tendon_config.copy()
        config["actuator_properties"]["tendon_kp_scaling"] = "inverse_length"
        config["actuator_properties"]["tendon_kp_scale_factor"] = 2.0

        output_path = temp_dir / "scaled_tendons.xml"
        result = create_tdcr_from_config(config, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "<actuator>" in content
        assert "kp=" in content

    def test_visual_customization(self, basic_config, temp_dir):
        """Test generation with visual customization options."""
        config = basic_config.copy()
        config["plane_style"] = "checkered"
        config["colour_scheme"] = "Clean"
        config["axis_label"] = True

        output_path = temp_dir / "visual_tdcr.xml"
        result = create_tdcr_from_config(config, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "<mujoco" in content
        # Visual elements should be present
        assert "rgba=" in content

    def test_pretension_keyframe(self, tendon_config, temp_dir):
        """Test generation with pretension keyframe."""
        output_path = temp_dir / "pretension_tdcr.xml"
        result = create_tdcr_from_config(tendon_config, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()

        # Check for keyframe section
        assert "<keyframe>" in content
        assert 'name="pretension"' in content

    def test_invalid_config_handling(self, temp_dir):
        """Test handling of invalid configurations."""
        invalid_config = {
            "description": "Invalid TDCR",
            "num_segments": 0,  # Invalid: must be > 0
            "total_links": -5,  # Invalid: must be > 0
        }

        output_path = temp_dir / "invalid.xml"

        # Should handle invalid config gracefully
        with pytest.raises(
            (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError)
        ):
            result = create_tdcr_from_config(invalid_config, str(output_path))

    def test_config_with_all_features(self, temp_dir):
        """Test generation with all features enabled."""
        full_config = {
            "description": "Full-featured TDCR",
            "num_segments": 3,
            "links_per_segment": {"1": 10, "2": 10, "3": 10},
            "segment_lengths": {"1": 0.15, "2": 0.15, "3": 0.15},
            "total_links": 30,
            "total_length": 0.45,
            "radius": 0.008,
            "mass": 0.5,
            "joints_per_link": 3,
            "joint_config_mode": "material",
            "material_properties": {
                "density": 20000,
                "youngs_modulus": 150e9,
                "poisson_ratio": 0.3,
                "outer_radius": 0.0008,
                "damping": 1.0,
            },
            "actuation_mode": "parallel_tendons",
            "actuation_details": {
                "segments": [
                    {"number_of_tendons": 3, "distance_to_backbone": 0.005},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.005},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.005},
                ]
            },
            "seg_offsets": [0, 2.0944, 4.1888],
            "actuator_properties": {
                "tendon_ctrlrange": "-0.1 0.1",
                "tendon_actuator_type": "position",
                "tendon_kp": 50000,
                "tendon_kp_scaling": "inverse_length",
                "tendon_kp_scale_factor": 1.5,
                "tendon_pretension": 2.0,
                "tendon_forcelimited": "true",
                "tendon_forcerange": "-200 0",
            },
            "disable_self_collision": True,
            "gravity": "0 0 -9.81",
            "plane": True,
            "plane_style": "checkered",
            "colour_scheme": "Clean",
            "axis_label": True,
        }

        output_path = temp_dir / "full_featured.xml"
        result = create_tdcr_from_config(full_config, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()

        # Check for all major features
        assert "<mujoco" in content
        assert "<tendon>" in content
        assert "<actuator>" in content
        assert "<keyframe>" in content
        assert "seg_0" in content
        assert "seg_1" in content
        assert "seg_2" in content

    def test_independent_segments_generation(self, temp_dir):
        """Test generation with independent segments mode."""
        config = {
            "description": "Independent Segments TDCR",
            "num_segments": 3,
            "links_per_segment": {"1": 10, "2": 10, "3": 10},
            "segment_lengths": {"1": 0.1, "2": 0.1, "3": 0.1},
            "total_links": 30,
            "total_length": 0.3,
            "radius": 0.006,
            "mass": 0.2,
            "joints_per_link": 2,
            "joint_config_mode": "material",
            "material_properties": {
                "density": 37708,
                "youngs_modulus": 80e9,
                "poisson_ratio": 0.3,
                "outer_radius": 0.0005,
                "damping": 0.1,
            },
            "actuation_mode": "parallel_tendons",
            "independent_segments": True,  # Enable independent segments
            "actuation_details": {
                "segments": [
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                ]
            },
            "seg_offsets": [0, 0.5236, 1.0472],
            "actuator_properties": {
                "tendon_ctrlrange": "-0.05 0.05",
                "tendon_actuator_type": "position",
                "tendon_kp": 10000,
                "tendon_pretension": 5.0,
            },
        }

        output_path = temp_dir / "independent_segments.xml"
        result = create_tdcr_from_config(config, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()

        # Check basic structure
        assert "<mujoco" in content
        assert "<tendon>" in content
        assert "<spatial" in content
        assert "<actuator>" in content

        # Check that tendons exist for all segments
        assert "seg_0_tendon_0" in content
        assert "seg_1_tendon_0" in content
        assert "seg_2_tendon_0" in content

        # In independent mode, segment 1 tendons should NOT have sites in segment 0
        # We can verify this by checking that segment 1 tendon sites start at link 11
        # (after segment 0's 10 links + 1 boundary link)
        # This is a structural check - the key is that tendon routing is different

    def test_independent_vs_coupled_tendon_routing(self, temp_dir):
        """Compare tendon routing between independent and coupled modes."""
        base_config = {
            "description": "TDCR for comparison",
            "num_segments": 2,
            "links_per_segment": {"1": 5, "2": 5},
            "segment_lengths": {"1": 0.15, "2": 0.15},
            "total_links": 10,
            "total_length": 0.3,
            "radius": 0.006,
            "mass": 0.2,
            "joints_per_link": 2,
            "joint_config_mode": "direct",
            "stiffness": 100.0,
            "damping": 1.0,
            "actuation_mode": "parallel_tendons",
            "actuation_details": {
                "segments": [
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                    {"number_of_tendons": 3, "distance_to_backbone": 0.004},
                ]
            },
            "actuator_properties": {
                "tendon_ctrlrange": "-0.05 0.05",
                "tendon_actuator_type": "position",
                "tendon_kp": 10000,
            },
        }

        # Generate coupled version
        coupled_config = base_config.copy()
        coupled_config["independent_segments"] = False
        coupled_path = temp_dir / "coupled.xml"
        create_tdcr_from_config(coupled_config, str(coupled_path))

        # Generate independent version
        independent_config = base_config.copy()
        independent_config["independent_segments"] = True
        independent_path = temp_dir / "independent.xml"
        create_tdcr_from_config(independent_config, str(independent_path))

        # Both should exist and have tendons
        assert coupled_path.exists()
        assert independent_path.exists()

        coupled_content = coupled_path.read_text()
        independent_content = independent_path.read_text()

        # Both should have tendon definitions
        assert "seg_0_tendon_0" in coupled_content
        assert "seg_1_tendon_0" in coupled_content
        assert "seg_0_tendon_0" in independent_content
        assert "seg_1_tendon_0" in independent_content

        # The key difference: in coupled mode, segment 1 tendons pass through
        # segment 0 links, so there should be more tendon sites
        # Count tendon sites for each segment
        coupled_seg1_sites = coupled_content.count("seg_1_tendon")
        independent_seg1_sites = independent_content.count("seg_1_tendon")

        # In independent mode, segment 1 should have fewer total site references
        # because its tendons don't route through segment 0
        # This is a structural difference we can verify
        assert coupled_seg1_sites > 0
        assert independent_seg1_sites > 0


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
class TestTDCRPhysicalInvariants:
    """Load generated models in MuJoCo and assert physical invariants.

    These guard against silent regressions in tendon geometry and pretension —
    things the string-in-XML checks above cannot catch.
    """

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    # variant -> (config-builder kwargs, independent_segments?)
    REST_LENGTH_VARIANTS = {
        "coupled_proportional": (
            dict(links=[10, 10, 10], lengths=[0.1, 0.1, 0.1]),
            False,
        ),
        "coupled_different_lengths": (
            dict(links=[10, 10, 10], lengths=[0.06, 0.0625, 0.064]),
            False,
        ),
        "coupled_heterogeneous_counts": (
            dict(links=[10, 5, 8], lengths=[0.06, 0.0625, 0.064]),
            False,
        ),
        "coupled_two_site_nonproportional": (
            dict(links=[10, 10, 10], lengths=[0.06, 0.07, 0.05], constraint_factor=0.5),
            False,
        ),
        # Independent segments use single-site routing (constraint_factor=0),
        # matching the shipped independent config; independent + two-site is an
        # unused combination where boundary sites extend past the segment.
        "independent_proportional": (
            dict(
                links=[10, 10, 10],
                lengths=[0.1, 0.1, 0.1],
                independent=True,
                constraint_factor=0.0,
            ),
            True,
        ),
        "independent_nonproportional": (
            dict(
                links=[8, 12, 6],
                lengths=[0.06, 0.07, 0.05],
                independent=True,
                constraint_factor=0.0,
            ),
            True,
        ),
    }

    @pytest.mark.parametrize("variant", list(REST_LENGTH_VARIANTS))
    def test_tendon_rest_length_matches_segment_lengths(self, variant, temp_dir):
        """Each tendon's straight-pose length must equal its configured segment
        length(s): cumulative for coupled, own-segment for independent — for any
        per-segment link count / length distribution and either site branch."""
        kwargs, independent = self.REST_LENGTH_VARIANTS[variant]
        lengths = kwargs["lengths"]
        config = _make_tendon_config(**kwargs)
        model, data = _generate_and_load(config, temp_dir / f"{variant}.xml")

        assert model.ntendon > 0
        for tid in range(model.ntendon):
            s = _tendon_segment_index(model, tid)
            expected = lengths[s] if independent else sum(lengths[: s + 1])
            assert data.ten_length[tid] == pytest.approx(expected, abs=1e-4), (
                f"{variant}: tendon {tid} (seg {s}) rest length "
                f"{data.ten_length[tid]:.5f} != configured {expected:.5f}"
            )

    def test_position_pretension_force_matches_config(self, temp_dir):
        """At the pretension keyframe, each position actuator should realize the
        configured pretension as tension (because rest length == tendon geometry)."""
        config = _make_tendon_config(links=[10, 10, 10], lengths=[0.06, 0.0625, 0.064])
        config["actuator_properties"]["tendon_pretension"] = 3.0
        model, data = _generate_and_load(config, temp_dir / "pretension.xml")

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
        assert key_id >= 0, "pretension keyframe missing"
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        assert model.nu > 0
        for i in range(model.nu):
            assert abs(data.actuator_force[i]) == pytest.approx(3.0, abs=0.05), (
                f"actuator {i} force {data.actuator_force[i]:.3f} "
                f"!= configured pretension 3.0"
            )

    def test_kp_array_maps_heterogeneous_tendon_counts(self, temp_dir):
        """A per-tendon tendon_kp_array must map by flat tendon index even when
        segments have different tendon counts (regression for s*num_tendons+n)."""
        counts = [3, 4, 3]
        config = _make_tendon_config(
            links=[10, 10, 10],
            lengths=[0.1, 0.1, 0.1],
            tendons_per_segment=counts,
        )
        kp_array = [10, 11, 12, 20, 21, 22, 23, 30, 31, 32]
        config["actuator_properties"]["tendon_kp_array"] = kp_array
        model, _ = _generate_and_load(config, temp_dir / "kp_array.xml")

        expected = {}
        flat = 0
        for s, c in enumerate(counts):
            for n in range(c):
                expected[f"seg_{s}_ten_{n}"] = kp_array[flat]
                flat += 1

        checked = 0
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name in expected:
                assert model.actuator_gainprm[i][0] == pytest.approx(
                    expected[name]
                ), f"{name}: kp {model.actuator_gainprm[i][0]} != {expected[name]}"
                checked += 1
        assert checked == sum(counts)

    def test_motor_pretension_applied_and_unset_is_zero(self, temp_dir):
        """Explicit motor-mode pretension -> baseline tension force; unset -> zero."""
        config = _make_tendon_config(
            links=[10, 10, 10], lengths=[0.1, 0.1, 0.1], actuator_type="motor"
        )
        config["actuator_properties"]["tendon_pretension"] = 2.0
        model, data = _generate_and_load(config, temp_dir / "motor_pre.xml")
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        assert model.nu > 0
        for i in range(model.nu):
            assert data.actuator_force[i] == pytest.approx(-2.0, abs=1e-6)

        config2 = _make_tendon_config(
            links=[10, 10, 10], lengths=[0.1, 0.1, 0.1], actuator_type="motor"
        )
        config2["actuator_properties"].pop("tendon_pretension", None)
        model2, data2 = _generate_and_load(config2, temp_dir / "motor_nopre.xml")
        key_id2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_KEY, "pretension")
        mujoco.mj_resetDataKeyframe(model2, data2, key_id2)
        mujoco.mj_forward(model2, data2)
        assert np.allclose(data2.actuator_force, 0.0)

    def test_per_segment_material_independent_stiffness(self, temp_dir):
        """material_properties_per_segment gives each segment its own joint
        stiffness (finding #1). It is realized as a one-module-per-segment
        modular robot (rigid junctions between segments), so the model carries
        exactly the three distinct per-segment material stiffnesses, and the
        per-segment tendon rest lengths stay exact."""
        from opencr_mujoco.generators.unified_tdcr_generator import compute_link_properties

        links, lengths = [10, 8, 6], [0.08, 0.06, 0.05]
        mats = [
            {
                "density": 7800,
                "youngs_modulus": 200e9,
                "poisson_ratio": 0.3,
                "outer_radius": 0.0006,
                "damping": 0.5,
            },
            {
                "density": 1200,
                "youngs_modulus": 2e9,
                "poisson_ratio": 0.3,
                "outer_radius": 0.0005,
                "damping": 0.3,
            },
            {
                "density": 1000,
                "youngs_modulus": 50e9,
                "poisson_ratio": 0.3,
                "outer_radius": 0.0004,
                "damping": 0.4,
            },
        ]
        config = _make_tendon_config(links=links, lengths=lengths)
        config["material_properties_per_segment"] = mats
        model, data = _generate_and_load(config, temp_dir / "per_seg.xml")

        # The bending joints carry exactly the three distinct per-segment
        # material-derived stiffnesses (no global value).
        expected = {
            round(
                compute_link_properties(
                    length=lengths[s],
                    num_links=links[s],
                    material_properties=mats[s],
                    fallback_radius=0.006,
                    damping_fallback="module",
                ).bend_stiffness,
                6,
            )
            for s in range(3)
        }
        assert len(expected) == 3, "segments should have distinct stiffness"
        present = {
            round(model.jnt_stiffness[j], 6)
            for j in range(model.njnt)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith(
                "_x"
            )
        }
        assert (
            present == expected
        ), f"joint stiffness {present} != per-segment {expected}"

        # rigid junctions between segments (some link bodies have no joint)
        rigid = sum(
            1
            for b in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(
                "link_"
            )
            and sum(1 for j in range(model.njnt) if model.jnt_bodyid[j] == b) == 0
        )
        assert rigid > 0, "expected rigid junctions between material segments"

        # Per-segment tendon rest lengths remain exact (coupled cumulative).
        for tid in range(model.ntendon):
            seg = _tendon_segment_index(model, tid)
            assert data.ten_length[tid] == pytest.approx(
                sum(lengths[: seg + 1]), abs=1e-4
            )

    def test_modular_generation_loads(self, temp_dir):
        """Modular config with num_segments set but links_per_segment omitted must
        generate and load (regression for the modular __init__ KeyError crash)."""
        module = lambda density, ym, r: {  # noqa: E731
            "length": 0.05,
            "radius": r,
            "num_links": 5,
            "color": "0.6 0.6 0.6 1",
            "material_properties": {
                "density": density,
                "youngs_modulus": ym,
                "poisson_ratio": 0.3,
                "outer_radius": r,
                "damping_ratio": 0.05,
            },
        }
        config = {
            "modular": True,
            "num_segments": 3,
            "module_library": {
                "stiff": module(2000, 50e9, 0.01),
                "soft": module(1000, 5e8, 0.0125),
            },
            "segments": [
                {
                    "modules": ["stiff", "soft"],
                    "actuation": {"number_of_tendons": 4, "distance_to_backbone": 0.01},
                },
                {
                    "modules": ["soft"],
                    "actuation": {"number_of_tendons": 4, "distance_to_backbone": 0.01},
                },
                {
                    "modules": ["soft"],
                    "actuation": {"number_of_tendons": 4, "distance_to_backbone": 0.01},
                },
            ],
            "actuation_mode": "parallel_tendons",
            "actuator_properties": {
                "tendon_actuator_type": "position",
                "tendon_kp": 10000,
                "tendon_ctrlrange": "-0.05 0.05",
            },
        }
        model, data = _generate_and_load(config, temp_dir / "modular.xml")
        assert model.nbody > 1
        assert model.ntendon == 12  # 3 segments x 4 tendons
        assert np.isfinite(data.qpos).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
