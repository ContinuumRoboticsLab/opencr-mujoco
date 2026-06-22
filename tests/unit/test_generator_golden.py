#!/usr/bin/env python3
"""Golden-snapshot regression tests for the TDCR generator.

For every shipped config in ``configs/generation/*.json`` (loaded the same way
``generate.py`` loads a standalone model) plus a set of synthetic gap-filler
configs that exercise the direct-torque / 3-joint / deadband paths the shipped
set under-covers, this generates the model, loads it in MuJoCo, and compares a
tuple of physical invariants against a committed baseline
(``generator_golden_baseline.json``).

This locks the *behavior* of the generator (body topology, masses, joint
stiffness/damping, tendon rest lengths, actuator kp, pretension forces, names,
tip position) so the generator refactor cannot silently change any existing
config. Physics is read back via MuJoCo, so it is invariant to XML-structure
changes (e.g. moving stiffness from a <default> class onto each <joint>).

Regenerate the baseline intentionally with:

    python tests/unit/test_generator_golden.py        # rewrites the baseline
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from opencr_mujoco.generators.unified_tdcr_generator import create_tdcr_from_config

try:
    import mujoco
    import numpy as np

    _HAS_MUJOCO = True
except ImportError:  # pragma: no cover
    _HAS_MUJOCO = False

BASELINE_PATH = Path(__file__).resolve().parent / "generator_golden_baseline.json"
GEN_CONFIG_DIR = PROJECT_ROOT / "configs" / "generation"

# Float comparison tolerance. The refactor preserves the SAME formulas, so
# results should be identical to ~machine precision; a real behavior change
# (different geometry/mass/stiffness) is orders of magnitude larger.
_REL_TOL = 1e-6
_ABS_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Config loading (mirrors generate.py's standalone path)
# --------------------------------------------------------------------------- #
def _autocalc_totals(d):
    if "total_links" not in d and "links_per_segment" in d:
        d["total_links"] = sum(d["links_per_segment"].values())
    if "total_length" not in d and "segment_lengths" in d:
        d["total_length"] = sum(d["segment_lengths"].values())
    return d


def _synthetic_configs():
    """Configs exercising paths the shipped set under-covers (direct-torque,
    joints_per_link=3, joint_deadband)."""
    material = {
        "density": 37708,
        "youngs_modulus": 200e9,
        "poisson_ratio": 0.3,
        "outer_radius": 0.0005,
        "damping": 0.5,
    }
    tendon_actuation = {
        "segments": [
            {"number_of_tendons": 3, "distance_to_backbone": 0.004},
            {"number_of_tendons": 3, "distance_to_backbone": 0.004},
        ]
    }
    tendon_props = {
        "tendon_ctrlrange": "-0.05 0.05",
        "tendon_actuator_type": "position",
        "tendon_kp": 10000,
        "tendon_pretension": 1.0,
    }
    return {
        "synth_direct_torque_2joint": {
            "num_segments": 2,
            "links_per_segment": {"1": 6, "2": 6},
            "segment_lengths": {"1": 0.15, "2": 0.15},
            "radius": 0.01,
            "mass": 0.5,
            "joints_per_link": 2,
            "joint_config_mode": "direct",
            "stiffness": 100.0,
            "damping": 1.0,
            "actuation_mode": "direct_torque",
        },
        "synth_direct_torque_3joint": {
            "num_segments": 1,
            "links_per_segment": {"1": 8},
            "segment_lengths": {"1": 0.3},
            "radius": 0.01,
            "mass": 0.5,
            "joints_per_link": 3,
            "joint_config_mode": "direct",
            "stiffness": 120.0,
            "torsion_stiffness": 60.0,
            "damping": 1.5,
            "actuation_mode": "direct_torque",
        },
        "synth_tendon_3joint": {
            "num_segments": 2,
            "links_per_segment": {"1": 6, "2": 6},
            "segment_lengths": {"1": 0.12, "2": 0.12},
            "radius": 0.006,
            "joints_per_link": 3,
            "joint_config_mode": "material",
            "material_properties": material,
            "actuation_mode": "parallel_tendons",
            "actuation_details": tendon_actuation,
            "seg_offsets": [0.0, 0.5236],
            "actuator_properties": tendon_props,
        },
        "synth_deadband": {
            "num_segments": 2,
            "links_per_segment": {"1": 8, "2": 8},
            "segment_lengths": {"1": 0.1, "2": 0.1},
            "radius": 0.006,
            "joints_per_link": 2,
            "joint_config_mode": "material",
            "material_properties": material,
            "actuation_mode": "parallel_tendons",
            "actuation_details": tendon_actuation,
            "seg_offsets": [0.0, 0.5236],
            "actuator_properties": tendon_props,
            "joint_deadband": 0.02,
        },
        # Finding #1 feature: independent per-segment material (stiff base /
        # soft mid / stiff tip), proportional and non-proportional lengths.
        "synth_per_segment_material": {
            "num_segments": 3,
            "links_per_segment": {"1": 10, "2": 10, "3": 10},
            "segment_lengths": {"1": 0.06, "2": 0.06, "3": 0.06},
            "radius": 0.006,
            "joints_per_link": 2,
            "joint_config_mode": "material",
            "material_properties": material,
            "actuation_mode": "parallel_tendons",
            "material_properties_per_segment": [
                {
                    "density": 7800,
                    "youngs_modulus": 200e9,
                    "poisson_ratio": 0.3,
                    "outer_radius": 0.0005,
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
                    "outer_radius": 0.0005,
                    "damping": 0.4,
                },
            ],
            "actuation_details": {
                "segments": [{"number_of_tendons": 3, "distance_to_backbone": 0.004}]
                * 3
            },
            "seg_offsets": [0.0, 0.5236, 1.0472],
            "actuator_properties": tendon_props,
        },
        "synth_per_segment_nonproportional": {
            "num_segments": 3,
            "links_per_segment": {"1": 10, "2": 8, "3": 6},
            "segment_lengths": {"1": 0.08, "2": 0.06, "3": 0.05},
            "radius": 0.006,
            "joints_per_link": 2,
            "joint_config_mode": "material",
            "material_properties": material,
            "actuation_mode": "parallel_tendons",
            "material_properties_per_segment": [
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
            ],
            "actuation_details": {
                "segments": [{"number_of_tendons": 3, "distance_to_backbone": 0.004}]
                * 3
            },
            "seg_offsets": [0.0, 0.5236, 1.0472],
            "actuator_properties": tendon_props,
        },
    }


def _all_configs():
    """All configs to snapshot, keyed by name. Shipped first, then synthetic.

    Scope is deliberately the TOP LEVEL of configs/generation/ only: that is
    the physics-locked set. Demo/prop configs (README clips and the like) go
    in configs/generation/examples/ — generate.py and ensure_scene resolve
    them all the same, but they are exempt from the golden baseline, so
    adding or tweaking a demo never requires a regen.
    """
    configs = {}
    for path in sorted(GEN_CONFIG_DIR.glob("*.json")):
        configs[path.stem] = _autocalc_totals(json.load(open(path)))
    for name, cfg in _synthetic_configs().items():
        configs[name] = _autocalc_totals(cfg)
    return configs


# --------------------------------------------------------------------------- #
# Invariant extraction
# --------------------------------------------------------------------------- #
def _name(model, objtype, i):
    return mujoco.mj_id2name(model, objtype, i) or f"<unnamed_{objtype}_{i}>"


def compute_invariants(config):
    """Generate + MuJoCo-load a config and return JSON-serializable invariants."""
    with tempfile.TemporaryDirectory() as tmp:
        xml = os.path.join(tmp, "model.xml")
        create_tdcr_from_config(config, xml)
        m = mujoco.MjModel.from_xml_path(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

        B, J, T, A, S, K = (
            mujoco.mjtObj.mjOBJ_BODY,
            mujoco.mjtObj.mjOBJ_JOINT,
            mujoco.mjtObj.mjOBJ_TENDON,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            mujoco.mjtObj.mjOBJ_SITE,
            mujoco.mjtObj.mjOBJ_KEY,
        )
        body_names = [_name(m, B, i) for i in range(m.nbody)]
        act_names = [_name(m, A, i) for i in range(m.nu)]

        inv = {
            "nbody": int(m.nbody),
            "body_names": body_names,
            "total_mass": float(np.sum(m.body_mass)),
            "body_mass": {body_names[i]: float(m.body_mass[i]) for i in range(m.nbody)},
            "jnt_stiffness": {
                _name(m, J, i): float(m.jnt_stiffness[i]) for i in range(m.njnt)
            },
            "jnt_damping": {
                _name(m, J, i): float(m.dof_damping[m.jnt_dofadr[i]])
                for i in range(m.njnt)
            },
            "tendon_names": sorted(_name(m, T, i) for i in range(m.ntendon)),
            "ten_length": {
                _name(m, T, i): float(d.ten_length[i]) for i in range(m.ntendon)
            },
            "actuator_names": sorted(act_names),
            "actuator_kp": {
                act_names[i]: float(m.actuator_gainprm[i][0]) for i in range(m.nu)
            },
            "site_names": sorted(_name(m, S, i) for i in range(m.nsite)),
        }

        ee_id = mujoco.mj_name2id(m, B, "EE_pos")
        inv["ee_pos"] = [float(x) for x in d.xpos[ee_id]] if ee_id >= 0 else None
        # Tip of the body chain (deepest body) — catches cumulative geometry
        # drift uniformly, including configs that have no EE_pos body.
        inv["tip_xpos"] = [float(x) for x in d.xpos[m.nbody - 1]]

        key_id = mujoco.mj_name2id(m, K, "pretension")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(m, d, key_id)
            mujoco.mj_forward(m, d)
            inv["pretension_force"] = {
                act_names[i]: float(d.actuator_force[i]) for i in range(m.nu)
            }
        else:
            inv["pretension_force"] = None
        return inv


def _diff(baseline, current, path=""):
    """Return a list of human-readable difference strings (empty == match)."""
    diffs = []
    if isinstance(baseline, dict):
        if not isinstance(current, dict) or set(baseline) != set(current):
            return [
                f"{path}: dict keys differ "
                f"(only baseline={set(baseline) - set(current or {})}, "
                f"only current={set(current or {}) - set(baseline)})"
            ]
        for k in baseline:
            diffs += _diff(baseline[k], current[k], f"{path}.{k}")
    elif isinstance(baseline, list):
        if not isinstance(current, list) or len(baseline) != len(current):
            return [
                f"{path}: list length {len(baseline)} != "
                f"{len(current) if isinstance(current, list) else 'n/a'}"
            ]
        for i, (a, b) in enumerate(zip(baseline, current)):
            diffs += _diff(a, b, f"{path}[{i}]")
    elif isinstance(baseline, bool) or isinstance(current, bool):
        if baseline != current:
            diffs.append(f"{path}: {baseline!r} != {current!r}")
    elif isinstance(baseline, (int, float)) and isinstance(current, (int, float)):
        if not math.isclose(baseline, current, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
            diffs.append(f"{path}: {baseline} != {current}")
    else:
        if baseline != current:
            diffs.append(f"{path}: {baseline!r} != {current!r}")
    return diffs


def _write_baseline():
    baseline = {name: compute_invariants(cfg) for name, cfg in _all_configs().items()}
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=1, sort_keys=True)
    print(f"Wrote golden baseline for {len(baseline)} configs -> {BASELINE_PATH}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
@pytest.mark.skipif(
    not BASELINE_PATH.exists(),
    reason="baseline missing; run `python tests/unit/test_generator_golden.py`",
)
@pytest.mark.parametrize("config_name", sorted(_all_configs()) if _HAS_MUJOCO else [])
def test_generator_matches_golden(config_name):
    baseline = json.load(open(BASELINE_PATH))
    assert (
        config_name in baseline
    ), f"{config_name} missing from baseline; regenerate it"
    current = compute_invariants(_all_configs()[config_name])
    diffs = _diff(baseline[config_name], current)
    assert not diffs, f"{config_name} drifted from golden baseline:\n" + "\n".join(
        diffs
    )


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="baseline missing")
def test_baseline_covers_all_configs():
    baseline = json.load(open(BASELINE_PATH))
    assert set(baseline) == set(
        _all_configs()
    ), "baseline config set differs from current; regenerate the baseline"


if __name__ == "__main__":
    if not _HAS_MUJOCO:
        sys.exit("mujoco not installed")
    _write_baseline()
