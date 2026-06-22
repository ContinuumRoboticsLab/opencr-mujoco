#!/usr/bin/env python3
"""Physics-invariant tests for the TDCR generator.

The golden-snapshot tests (test_generator_golden.py) lock the generator's
outputs against *drift*; the tests here check the outputs are *physically
correct* in the first place:

- total mass matches rho*A*L (with the documented clamped-base convention)
- the tip lands exactly at total_length in every chain mode
- both bending axes carry the same EI-based stiffness (isotropic bending),
  and the 3-joint twist axis carries the GJ-based torsion stiffness
- the chain has the exact number of elastic joints (the base half-link is a
  rigid clamp)
- a passive rod's simulated tip deflection under a tip force matches the
  Euler-Bernoulli cantilever solution F*L^3/(3*E*I)
- the tip interface (EE_pos body + force_site_tip site) exists in all modes
- collision exclusions cover each 3-neighborhood exactly once
- the pretension keyframe realizes the configured tendon tension

These are the invariants the discretization method promises; run nightly.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

mujoco = pytest.importorskip("mujoco")

from opencr_mujoco.generators.unified_tdcr_generator import (  # noqa: E402
    create_tdcr_from_config,
    compute_link_properties,
)

GEN_CONFIG_DIR = PROJECT_ROOT / "configs" / "generation"

MATERIAL = {
    "density": 7800.0,
    "youngs_modulus": 200e9,
    "poisson_ratio": 0.3,
    "outer_radius": 0.0005,
    "damping_ratio": 0.05,
}


def build(config):
    """Generate a config and return (model, data) after mj_forward."""
    with tempfile.TemporaryDirectory() as tmp:
        xml = os.path.join(tmp, "model.xml")
        create_tdcr_from_config(config, xml)
        model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def tendon_config(n_links=30, n_segments=3, length=0.3, joints_per_link=2):
    per_seg = n_links // n_segments
    seg_len = length / n_segments
    return {
        "num_segments": n_segments,
        "links_per_segment": {str(i + 1): per_seg for i in range(n_segments)},
        "segment_lengths": {str(i + 1): seg_len for i in range(n_segments)},
        "total_links": n_links,
        "total_length": length,
        "radius": 0.005,
        "joints_per_link": joints_per_link,
        "joint_config_mode": "material",
        "material_properties": dict(MATERIAL),
        "actuation_mode": "parallel_tendons",
        "actuation_details": {
            "segments": [{"number_of_tendons": 3, "distance_to_backbone": 0.004}]
            * n_segments
        },
        "seg_offsets": [0.0, math.pi / 6, math.pi / 3][:n_segments],
        "actuator_properties": {
            "tendon_ctrlrange": "-0.05 0.05",
            "tendon_actuator_type": "position",
            "tendon_kp": 10000,
            "tendon_pretension": 1.0,
        },
        "gravity": "0 0 0",
        "plane": False,
    }


def direct_config(n_links=30, length=0.3, joints_per_link=3):
    return {
        "num_segments": 1,
        "links_per_segment": {"1": n_links},
        "segment_lengths": {"1": length},
        "total_links": n_links,
        "total_length": length,
        "radius": 0.005,
        "joints_per_link": joints_per_link,
        "joint_config_mode": "material",
        "material_properties": dict(MATERIAL),
        "actuation_mode": "none",
        "gravity": "0 0 0",
        "plane": False,
    }


def modular_config():
    return {
        "modular": True,
        "num_segments": 2,
        "module_library": {
            "stiff": {
                "length": 0.06,
                "radius": 0.006,
                "num_links": 6,
                "color": "0.2 0.2 0.8 1.0",
                "material_properties": dict(MATERIAL),
            },
            "soft": {
                "length": 0.05,
                "radius": 0.006,
                "num_links": 5,
                "color": "0.8 0.2 0.2 1.0",
                "material_properties": {**MATERIAL, "youngs_modulus": 2e9},
            },
        },
        "segments": [
            {
                "modules": ["stiff", "soft"],
                "actuation": {"number_of_tendons": 3, "distance_to_backbone": 0.004},
            },
            {
                "modules": ["soft"],
                "actuation": {"number_of_tendons": 3, "distance_to_backbone": 0.004},
            },
        ],
        "joints_per_link": 2,
        "actuation_mode": "parallel_tendons",
        "seg_offsets": [0.0, math.pi / 6],
        "actuator_properties": {
            "tendon_ctrlrange": "-0.05 0.05",
            "tendon_actuator_type": "position",
            "tendon_kp": 5000,
            "tendon_pretension": 0.5,
        },
        "gravity": "0 0 0",
        "plane": False,
    }


def rod_mass(length):
    area = math.pi * MATERIAL["outer_radius"] ** 2
    return MATERIAL["density"] * area * length


def tip_z(model, data):
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
    assert ee >= 0, "EE_pos body missing"
    return float(data.xpos[ee][2])


class TestMassBudget:
    """Total mass must follow rho*A*L with the documented base convention."""

    def test_tendon_chain_mass_is_rho_A_L(self):
        cfg = tendon_config(n_links=30, length=0.3)
        model, _ = build(cfg)
        # Half-length end links carry half mass -> the full rod mass is modeled
        assert np.isclose(np.sum(model.body_mass), rod_mass(0.3), rtol=1e-9)

    @pytest.mark.parametrize("joints_per_link", [2, 3])
    def test_direct_chain_mass_convention(self, joints_per_link):
        n = 30
        cfg = direct_config(n_links=n, length=0.3, joints_per_link=joints_per_link)
        model, _ = build(cfg)
        # The direct chain leaves the clamped base half-link unmodeled
        # (it is rigid and attached to the world), so its mass budget is
        # (N - 0.5)/N of the full rod.
        expected = rod_mass(0.3) * (n - 0.5) / n
        assert np.isclose(np.sum(model.body_mass), expected, rtol=1e-9)

    def test_modular_chain_mass_is_sum_of_module_masses(self):
        model, _ = build(modular_config())
        area = math.pi * MATERIAL["outer_radius"] ** 2
        expected = (
            MATERIAL["density"] * area * 0.06  # stiff module
            + MATERIAL["density"] * area * 0.05 * 2  # two soft modules
        )
        assert np.isclose(np.sum(model.body_mass), expected, rtol=1e-9)


class TestTipPosition:
    """The tip (EE_pos) must land exactly at total_length in every mode."""

    def test_tendon_chain(self):
        model, data = build(tendon_config(n_links=30, length=0.3))
        assert np.isclose(tip_z(model, data), 0.3, atol=1e-12)

    @pytest.mark.parametrize("joints_per_link", [2, 3])
    def test_direct_chain(self, joints_per_link):
        model, data = build(
            direct_config(n_links=24, length=0.36, joints_per_link=joints_per_link)
        )
        assert np.isclose(tip_z(model, data), 0.36, atol=1e-12)

    def test_modular_chain(self):
        model, data = build(modular_config())
        assert np.isclose(tip_z(model, data), 0.06 + 0.05 + 0.05, atol=1e-12)


class TestJointStiffness:
    """Bending must be isotropic; twist must carry the GJ-based stiffness."""

    @staticmethod
    def _stiffness_by_suffix(model):
        by_suffix = {}
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            suffix = name.rsplit("_", 1)[-1]
            by_suffix.setdefault(suffix, []).append(float(model.jnt_stiffness[j]))
        return by_suffix

    def test_tendon_2joint_bending_isotropy(self):
        cfg = tendon_config(n_links=30, length=0.3, joints_per_link=2)
        model, _ = build(cfg)
        props = compute_link_properties(
            length=0.3,
            num_links=30,
            material_properties=MATERIAL,
            fallback_radius=0.005,
        )
        by_suffix = self._stiffness_by_suffix(model)
        assert np.allclose(by_suffix["x"], props.bend_stiffness, rtol=1e-9)
        assert np.allclose(by_suffix["z"], props.bend_stiffness, rtol=1e-9)

    def test_direct_3joint_torsion_ratio(self):
        model, _ = build(direct_config(n_links=30, length=0.3, joints_per_link=3))
        by_suffix = self._stiffness_by_suffix(model)
        # y/z are the bending axes (EI-based), x is twist (GJ-based, J = 2I)
        bend = np.unique(np.round(by_suffix["y"] + by_suffix["z"], 12))
        twist = np.unique(np.round(by_suffix["x"], 12))
        assert len(bend) == 1 and len(twist) == 1
        expected_ratio = 1.0 / (1.0 + MATERIAL["poisson_ratio"])  # 2G/E
        assert np.isclose(twist[0] / bend[0], expected_ratio, rtol=1e-9)

    def test_modular_bending_isotropy_per_module(self):
        model, _ = build(modular_config())
        by_suffix = self._stiffness_by_suffix(model)
        # Pair up joint_<i>_x / joint_<i>_z by index and require equality
        assert len(by_suffix["x"]) == len(by_suffix["z"]) > 0
        assert np.allclose(by_suffix["x"], by_suffix["z"], rtol=1e-9)


class TestChainTopology:
    def test_tendon_chain_has_exactly_N_joint_stations(self):
        n = 30
        model, _ = build(tendon_config(n_links=n, joints_per_link=2))
        # link_0 is the rigid clamp: joints sit on links 1..N only, so the
        # series compliance of N joints at stiffness N*EI/L is exactly L/EI.
        assert model.njnt == 2 * n
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_0_x") == -1

    @pytest.mark.parametrize("make_cfg", [tendon_config, direct_config, modular_config])
    def test_tip_interface_exists(self, make_cfg):
        model, _ = build(make_cfg())
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "force_site_tip") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "force_site_mid") >= 0

    def test_collision_exclusions_cover_neighborhood_once(self):
        cfg = tendon_config(n_links=30)
        cfg["disable_self_collision"] = True
        model, _ = build(cfg)
        pairs = set()
        for i in range(model.nexclude):
            sig = int(model.exclude_signature[i])
            pairs.add((sig >> 16, sig & 0xFFFF))
        # Chain bodies: link_0..link_30 (EE_pos carries no geom). Each pair
        # within distance 3 appears exactly once.
        chain_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}")
            for i in range(31)
        ]
        expected = set()
        for a in range(len(chain_ids)):
            for b in range(a + 1, min(len(chain_ids), a + 4)):
                lo, hi = sorted((chain_ids[a], chain_ids[b]))
                expected.add((lo, hi))
        normalized = {tuple(sorted(p)) for p in pairs}
        assert normalized == expected
        assert model.nexclude == len(expected)  # no duplicates


class TestPretension:
    def test_keyframe_realizes_configured_tension(self):
        cfg = tendon_config()
        cfg["actuator_properties"]["tendon_pretension"] = 1.5
        model, data = build(cfg)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
        assert key >= 0
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        for i in range(model.nu):
            # Position servos: force = kp * (ctrl - length) = -pretension
            assert np.isclose(
                data.actuator_force[i], -1.5, rtol=1e-6
            ), f"actuator {i} force {data.actuator_force[i]} != -1.5"


class TestCantileverDeflection:
    """End-to-end correctness of the discretization against beam theory."""

    def test_tip_deflection_matches_euler_bernoulli(self):
        length, n = 0.3, 24
        cfg = direct_config(n_links=n, length=length, joints_per_link=3)
        # Stronger damping so the quasi-static settle is fast
        cfg["material_properties"]["damping_ratio"] = 0.2
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "rod.xml")
            create_tdcr_from_config(cfg, xml)
            model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)
        model.opt.gravity[:] = 0.0

        e_mod, r = MATERIAL["youngs_modulus"], MATERIAL["outer_radius"]
        inertia = math.pi / 4 * r**4
        # Pick F for a ~1% -of-length deflection: small enough to stay in the
        # linear (Euler-Bernoulli) regime, large enough to dominate noise.
        deflection_target = 0.01 * length
        force = 3 * e_mod * inertia * deflection_target / length**3

        ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
        tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "force_site_tip")
        body_id = model.site_bodyid[tip_site]

        for _ in range(60000):
            data.xfrc_applied[:] = 0.0
            site_pos = data.site_xpos[tip_site]
            r_arm = site_pos - data.xipos[body_id]
            data.xfrc_applied[body_id, :3] = [force, 0.0, 0.0]
            data.xfrc_applied[body_id, 3:] = np.cross(r_arm, [force, 0.0, 0.0])
            mujoco.mj_step(model, data)
            if data.time > 0.5 and np.max(np.abs(data.qvel)) < 1e-7:
                break

        deflection = float(data.xpos[ee][0])
        expected = force * length**3 / (3 * e_mod * inertia)
        assert np.isclose(deflection, expected, rtol=0.02), (
            f"tip deflection {deflection*1000:.3f} mm vs Euler-Bernoulli "
            f"{expected*1000:.3f} mm"
        )


class TestShippedConfigsStillBuild:
    """Every shipped generation config must compile and be stable."""

    @pytest.mark.parametrize(
        "config_path", sorted(GEN_CONFIG_DIR.glob("*.json")), ids=lambda p: p.stem
    )
    def test_config_builds_and_steps(self, config_path):
        cfg = json.load(open(config_path))
        if "total_links" not in cfg and "links_per_segment" in cfg:
            cfg["total_links"] = sum(cfg["links_per_segment"].values())
        if "total_length" not in cfg and "segment_lengths" in cfg:
            cfg["total_length"] = sum(cfg["segment_lengths"].values())
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "model.xml")
            create_tdcr_from_config(cfg, xml)
            model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key)
        for _ in range(500):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
