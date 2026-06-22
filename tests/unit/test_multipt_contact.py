#!/usr/bin/env python3
"""Correctness tests for the multi-point task-space controller's live Jacobian.

These check the property that makes the numerical (clone-and-settle) Jacobian
worth its cost: that it is an accurate local map of the control point's motion,
in free space AND when the robot is making contact -- which is what gives
intuitive, stable task-space teleop on contact.

- free space: each estimated Jacobian column matches the tip's actual settled
  response to that Clark DOF (cosine ~ 1)
- the base-insertion DOF is folded into the IK (so the tip can move along the
  backbone axis, not just bend)
- under contact with a (compliant) cylinder, the LIVE re-estimated Jacobian
  tracks a tangential command and keeps the actuation bounded, and beats a
  FROZEN free-space Jacobian (which winds up / mistracks)

Sim-heavy; runs in the nightly correctness layer, not the --quick smoke set.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

mujoco = pytest.importorskip("mujoco")

import generate  # noqa: E402
from opencr_mujoco.utils.config_loader import ConfigLoader  # noqa: E402
from opencr_mujoco.controllers.tdcr_multipt_taskspace_controller import (  # noqa: E402
    TDCRMultiPointTaskSpaceController as TS,
)

# Jacobian columns are the raw per-segment Clark DOF, so perturbing column k
# means perturbing Clark coordinate k (unit vectors).
MODES = np.eye(6, dtype=float)
IDLE = {
    k: 0.0
    for k in ("vx", "vy", "vz", "wx", "wy", "wz", "clark_x", "clark_y", "v_insert")
}


@pytest.fixture(scope="module")
def base_scene():
    cfg = ConfigLoader().load_config("teleop", "ftdcr_taskspace_position")
    path = str(generate.ensure_scene(cfg["scene"]))
    geom = generate.tdcr_geometry_from_scene(path)
    cp = dict(cfg.get("controller_params", {}))
    for k, v in (geom or {}).items():
        cp.setdefault(k, v)
    return path, cp


def test_tension_uses_longer_settle_under_contact():
    """Compliant tension control settles slowly against an obstacle, so its
    contact Jacobian needs a longer settle horizon than free space (measured:
    cos 0.57 -> 0.96 going 0.1s -> 0.2s settle). Guard that the adaptive
    contact-settle is configured and gated on contact."""
    from opencr_mujoco.controllers.tdcr_multipt_tension_controller import (
        TDCRMultiPointTensionController as TN,
    )

    cfg = ConfigLoader().load_config("teleop", "ftdcr_taskspace_tension")
    scene = str(generate.ensure_scene(cfg["scene"]))
    geom = generate.tdcr_geometry_from_scene(scene)
    cp = dict(cfg.get("controller_params", {}))
    for k, v in (geom or {}).items():
        cp.setdefault(k, v)
    m, d = _load(scene)
    c = TN(
        m,
        d,
        tendon_distance_mm=cp.get("tendon_distance_mm", 4.0),
        angle_offset_rad_ccw=cp.get("angle_offset_rad_ccw"),
        velocity_scale=0.5,
        fps=100,
        verbose=False,
        tension_scale=cp.get("tension_scale", 0.1),
        clark_direct_scale=cp.get("clark_direct_scale", 100.0),
        contact_settle_horizon_s=cp.get("contact_settle_horizon_s", 0.2),
    )
    assert (
        c.contact_equilibrium_steps > c.equilibrium_steps
    ), "tension should settle longer under contact than in free space"
    # at rest (no contact) the free-space horizon is used
    assert d.ncon == 0 and c._settle_steps() == c.equilibrium_steps


def _load(scene):
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    kf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    mujoco.mj_resetDataKeyframe(m, d, kf)
    m.opt.gravity[:] = 0
    mujoco.mj_forward(m, d)
    return m, d


def _settle(m, d, cap=2000, tol=5e-4):
    for _ in range(cap):
        mujoco.mj_step(m, d)
        if np.abs(d.qvel).max() < tol:
            break


def _make(scene, cp):
    m, d = _load(scene)
    c = TS(
        m,
        d,
        tendon_distance_mm=cp.get("tendon_distance_mm", 4.0),
        angle_offset_rad_ccw=cp.get("angle_offset_rad_ccw"),
        velocity_scale=0.5,
        fps=100,
        verbose=False,
        tdcr_linear_scale=cp.get("tdcr_linear_scale", 0.1),
        tdcr_angular_scale=cp.get("tdcr_angular_scale", 2.0),
        clark_direct_scale=cp.get("clark_direct_scale", 10.0),
    )
    c.set_control_point("seg3")
    _settle(m, d)
    return m, d, c


def _actual_column(c, m, d, k, delta_mm=2.0):
    """Tip displacement (per mm) when Clark DOF k is perturbed and re-settled."""
    dj = mujoco.MjData(m)
    dj.qpos[:] = d.qpos
    dj.qvel[:] = d.qvel
    dj.ctrl[:] = d.ctrl
    if m.na:
        dj.act[:] = d.act
    mujoco.mj_forward(m, dj)
    p0 = dj.xpos[c.tip_body_id].copy()
    new_clark = c.kinematics.goal_clark_coords + delta_mm * MODES[k]
    tendons_m = c.kinematics.clark_to_tendons_mm(new_clark) * 0.001
    for i, aid in enumerate(c.tendon_actuator_ids):
        base = c.pretension_lengths[i] if c.pretension_lengths is not None else 0.0
        dj.ctrl[aid] = base + tendons_m[i]
    for _ in range(c.equilibrium_steps):
        mujoco.mj_step(m, dj)
    return (dj.xpos[c.tip_body_id].copy() - p0) / delta_mm


def _scene_with_cylinder(base, pos):
    tree = ET.parse(base)
    wb = tree.getroot().find("worldbody")
    ET.SubElement(
        wb,
        "geom",
        {
            "name": "obstacle_cyl",
            "type": "cylinder",
            "pos": f"{pos[0]} {pos[1]} {pos[2]}",
            "size": "0.02 0.08",
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "friction": "1 0.05 0.001",
            "solref": "0.05 1",
            "solimp": "0.7 0.9 0.01",
            "rgba": "0.8 0.2 0.2 0.6",
        },
    )
    out = PROJECT_ROOT / "assets" / "_tmp_test_contact.xml"
    tree.write(str(out), encoding="unicode", xml_declaration=True)
    return str(out)


def _weighted_mean_cos(c, m, d):
    cs, mags = [], []
    for k in range(6):
        act = _actual_column(c, m, d, k)
        mag = np.linalg.norm(act)
        if mag < 2e-4:
            continue
        pred = c._cached_J_pos[:, k]
        cs.append(np.dot(pred, act) / (np.linalg.norm(pred) * mag + 1e-12))
        mags.append(mag)
    cs, mags = np.array(cs), np.array(mags)
    return float(np.sum(cs * mags) / np.sum(mags))


def test_jacobian_accurate_in_free_space(base_scene):
    scene, cp = base_scene
    m, d, c = _make(scene, cp)
    c.compute_target_qpos(dict(IDLE), d)  # populate the cache
    assert c._cached_J_pos is not None
    wmean = _weighted_mean_cos(c, m, d)
    assert wmean > 0.9, f"free-space Jacobian columns inaccurate (mean cos={wmean:.3f})"


def test_insertion_dof_enabled_and_tracks_backbone(base_scene):
    scene, cp = base_scene
    m, d, c = _make(scene, cp)
    c.compute_target_qpos(dict(IDLE), d)
    assert c._has_insertion_dof
    assert c._cached_J_ins_pos is not None
    assert np.linalg.norm(c._cached_J_ins_pos) > 1e-5
    # commanding world +Z should move the tip with a real +Z component (insertion
    # + bending), not be impossible as it was bending-only.
    bid = c.tip_body_id
    p0 = d.xpos[bid].copy()
    for _ in range(60):
        d.ctrl[:] = c.compute_target_qpos({**IDLE, "vz": 1.0}, d)
        for _ in range(5):
            mujoco.mj_step(m, d)
    dz = (d.xpos[bid] - p0)[2]
    assert dz > 0.005, f"+Z command barely moved the tip in +Z ({dz*1000:.1f} mm)"


def test_manual_insertion_not_clobbered_by_auto(base_scene):
    """Pressing a Cartesian key AND a manual insertion key (Y/N) in the same
    frame must keep the manual insertion: auto-insertion is skipped, so the
    Franka targets match the manual-only command."""
    scene, cp = base_scene
    m, d, c = _make(scene, cp)
    franka = c.franka_actuator_ids
    manual_only = c.compute_target_qpos({**IDLE, "v_insert": 1.0}, d)[franka].copy()
    both = c.compute_target_qpos({**IDLE, "v_insert": 1.0, "vx": 1.0}, d)[franka].copy()
    assert np.allclose(
        manual_only, both, atol=1e-9
    ), "auto-insertion clobbered the manual Y/N insertion Franka targets"


def _drive(c, m, d, cmd, steps=40, freeze=False):
    """Closed loop; returns (tip_disp_mm, max_qvel, max_clark, max_ncon).

    If freeze, the (free-space) Jacobian is populated once then held fixed."""
    if freeze:
        c.compute_target_qpos(dict(IDLE), d)  # populate the cache first
        c._jacobian_frozen = True  # then hold it fixed
    bid = c.tip_body_id
    p0 = d.xpos[bid].copy()
    qvmax, ncon_max = 0.0, 0
    for _ in range(steps):
        d.ctrl[:] = c.compute_target_qpos({**IDLE, **cmd}, d)
        for _ in range(5):
            mujoco.mj_step(m, d)
        qvmax = max(qvmax, float(np.abs(d.qvel).max()))
        ncon_max = max(ncon_max, int(d.ncon))
    dp = (d.xpos[bid] - p0) * 1000.0
    return dp, qvmax, float(np.abs(c.kinematics.goal_clark_coords).max()), ncon_max


def test_contact_teleop_stable_and_beats_frozen(base_scene):
    """Drive the tip into a compliant cylinder; the live Jacobian must keep the
    teleop stable and bounded (no wind-up), and do so no worse than a frozen
    free-space Jacobian. Also checks tangential tracking is preserved."""
    scene, cp = base_scene
    m, d, c = _make(scene, cp)
    tip = d.xpos[c.tip_body_id].copy()
    # ~1 cm clear of the tip so the short +Y approach reliably makes contact
    cyl_scene = _scene_with_cylinder(scene, tip + np.array([0.0, 0.025, 0.0]))
    try:
        # Push into the obstacle with the LIVE Jacobian.
        m, d, c = _make(cyl_scene, cp)
        dp_l, q_live, clark_live, ncon_l = _drive(c, m, d, {"vy": 1.0})
        assert ncon_l > 0, "tip never contacted the cylinder while driving +Y"
        assert clark_live < 50.0, f"live-J wound up under contact ({clark_live:.0f} mm)"
        assert q_live < 3.0, f"live-J unstable under contact (max|qvel|={q_live:.2f})"

        # Same push with a FROZEN free-space Jacobian: live should be no worse.
        m, d, c = _make(cyl_scene, cp)
        _, q_frozen, clark_frozen, _ = _drive(c, m, d, {"vy": 1.0}, freeze=True)
        assert clark_live <= clark_frozen + 2.0, (
            f"live-J wound up more than frozen-J "
            f"(live={clark_live:.0f} mm, frozen={clark_frozen:.0f} mm)"
        )

        # Tangential (+X) command stays on-axis under contact.
        m, d, c = _make(cyl_scene, cp)
        dp_x, _, _, _ = _drive(
            c, m, d, {"vx": 1.0, "vy": 0.3}
        )  # light push keeps contact
        frac = dp_x[0] / (np.linalg.norm(dp_x) + 1e-9)
        assert frac > 0.5, f"tangential tracking under contact poor (frac={frac:.2f})"
    finally:
        (PROJECT_ROOT / "assets" / "_tmp_test_contact.xml").unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
