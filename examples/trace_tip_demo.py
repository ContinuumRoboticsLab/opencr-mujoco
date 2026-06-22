#!/usr/bin/env python3
"""Closed-loop tip demos (and the README GIFs/videos).

Three presets, all driven by the repo's own controllers/kinematics — no
motion planner, no external data:

  tdcr         A standalone 3-segment TDCR traces a repeating figure-8 with
               its tip, closed-loop via TDCRIKController (damped least
               squares over a numerically-differenced Clark-coordinate
               Jacobian).
  franka_tdcr  Station-keeping: the Franka sweeps a figure-8 with its wrist
               while the TDCR bends against it, closed loop, to pin the tip
               to a fixed point in space.
  kick         Elastic whip kick: a hanging TDCR bends hard away from a
               teed ball, then releases its tendons so the springy backbone
               swings back and strikes the ball into the goal — the swing,
               strike, roll, and net catch are all MuJoCo's native dynamics
               + contact solver (contact points shown).

Usage:
    python examples/trace_tip_demo.py --demo tdcr                  # live viewer
    python examples/trace_tip_demo.py --demo tdcr --record docs/media/tdcr_tip_trace
    python examples/trace_tip_demo.py --demo franka_tdcr --record docs/media/franka_tdcr_trace
    python examples/trace_tip_demo.py --demo kick --record docs/media/kick_goal

--record renders offscreen at high resolution (default 1280x840 @ 30 fps)
and writes BOTH <name>.mp4 (H.264, the high-quality artifact) and
<name>.gif (palette-optimized 720px, what the README embeds) via ffmpeg;
without ffmpeg it falls back to a PIL-encoded GIF.

Scene XMLs are generated automatically on first use (generate.ensure_scene).
Recording needs an offscreen GL context (works on macOS/Linux; under CI use
xvfb).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generate import ensure_scene  # noqa: E402

# --------------------------------------------------------------------------- #
# Demo presets
# --------------------------------------------------------------------------- #

DEMOS = {
    "tdcr": {
        "scene": "assets/tdcr/example_three_segment_franka.xml",
        "description": "standalone TDCR tip tracing a figure-8 (task-space IK)",
        "seconds": 20.0,
        "record_from": 10.0,  # skip the ramp lap -> the GIF loops seamlessly
        "camera": dict(
            distance=0.45, azimuth=140, elevation=-15, lookat=(0.0, 0.0, 0.18)
        ),
    },
    "kick": {
        "scene": "assets/example_contact_world_scene.xml",
        "description": "elastic whip kick: the TDCR bends hard away from the "
        "ball, releases its tendons to spring back and strike "
        "it into the goal (native MuJoCo contacts shown)",
        "seconds": 4.2,
        "record_from": 0.3,
        "contact_viz": True,
        "camera": dict(
            distance=0.95, azimuth=90, elevation=-12, lookat=(0.10, 0.0, 0.15)
        ),
    },
    "franka_tdcr": {
        "scene": "assets/ftdcr_v4_sysid_franka_scene.xml",
        "description": "station-keeping: Franka sweeps a figure-8, TDCR pins "
        "the tip to a fixed point (combined closed loop)",
        "seconds": 26.0,
        "record_from": 13.0,  # skip the ramp lap -> the GIF loops seamlessly
        "camera": dict(
            distance=0.9, azimuth=155, elevation=-16, lookat=(0.5, 0.0, 0.5)
        ),
    },
}

CONTROL_FPS = 50  # control ticks per second


def load_scene(demo):
    scene_path = ensure_scene(DEMOS[demo]["scene"])
    if demo == "kick":
        scene_path = _inject_kick_props(scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return model, data


def settle(model, data, seconds=1.0):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def tip_pos(model, data, body="EE_pos"):
    return data.body(body).xpos.copy()


def figure8(
    center, t, period, span_a, span_b, axes=((1, 0, 0), (0, 1, 0)), ramp_periods=1.0
):
    """A Gerono lemniscate around `center`, ramped in over the first lap."""
    ramp = min(t / (period * ramp_periods), 1.0)
    phi = 2 * np.pi * t / period
    a = np.asarray(axes[0], dtype=float)
    b = np.asarray(axes[1], dtype=float)
    return center + ramp * (
        span_a * np.sin(phi) * a + span_b * np.sin(phi) * np.cos(phi) * b
    )


# --------------------------------------------------------------------------- #
# Overlay primitives (scene decoration geoms)
# --------------------------------------------------------------------------- #


def _add_marker(scene, pos, size, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0, 0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten().astype(np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_segment(scene, p1, p2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten().astype(np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )
    scene.ngeom += 1


def _draw_trail(scene, trail, radius, color):
    n = max(len(trail) - 1, 1)
    for i in range(len(trail) - 1):
        age = i / n  # 0 oldest .. 1 newest
        rgba = (*color, 0.15 + 0.85 * age)
        _add_segment(scene, trail[i], trail[i + 1], radius, rgba)


class Trail:
    def __init__(self, seconds):
        self.points = []
        self.max_len = int(seconds * CONTROL_FPS)

    def append(self, p):
        # np.array (not asarray): p may be a live view into MuJoCo's data
        # buffer, and an un-copied view would make every stored point alias
        # the CURRENT position
        self.points.append(np.array(p, dtype=float, copy=True))
        if len(self.points) > self.max_len:
            self.points.pop(0)


# --------------------------------------------------------------------------- #
# The two demos. Each returns (step, draw, report):
#   step(t)      advance controllers (writes data.ctrl)
#   draw(scene)  add demo-specific overlays
#   report()     print a closing accuracy line
# --------------------------------------------------------------------------- #


def make_tdcr_demo(model, data):
    """Standalone TDCR: tip tracks a moving figure-8 target, closed loop."""
    from opencr_mujoco.controllers.tdcr_ik_controller import TDCRIKController

    model.opt.gravity[:] = 0  # task-space tracking runs gravity-free
    controller = TDCRIKController(
        model,
        data,
        tendon_distance_mm=4.0,  # example_three_segment_franka geometry
        velocity_scale=0.5,
        damping_factor=0.01,
        fps=CONTROL_FPS,
        verbose=False,
    )
    settle(model, data, 1.0)

    center = tip_pos(model, data) + np.array([0.0, 0.0, 0.012])
    period = 10.0
    trail = Trail(seconds=period)  # exactly one lap stays on screen
    errors = []
    state = {"target": center}

    def target_at(t):
        # figure-8 in the horizontal plane through the tip's home position
        return figure8(
            center, t, period, span_a=0.07, span_b=0.075, axes=((1, 0, 0), (0, 1, 0))
        )

    def step(t):
        target = target_at(t)
        err = target - tip_pos(model, data)
        cmd = np.clip(40.0 * err, -1.0, 1.0)  # saturating P-law on tip error
        data.ctrl[:] = controller.compute_target_qpos(
            {"vx": cmd[0], "vy": cmd[1], "vz": cmd[2]}
        )
        state["target"] = target
        if t > 1.5:
            trail.append(tip_pos(model, data))
        if t > period:
            errors.append(np.linalg.norm(err))

    def draw(scene):
        _draw_trail(scene, trail.points, 0.0016, (0.1, 0.6, 1.0))
        _add_marker(scene, state["target"], 0.005, (1.0, 0.3, 0.2, 0.9))

    def report():
        if errors:
            e = np.asarray(errors) * 1000
            print(
                f"  tip tracking error (post-ramp): mean {e.mean():.1f} mm, "
                f"max {e.max():.1f} mm"
            )

    return step, draw, report


def make_franka_tdcr_demo(model, data):
    """Station-keeping: the arm sweeps a figure-8; the TDCR pins the tip.

    The Franka runs through CombinedController's Jacobian IK toward a moving
    wrist target; a TDCRIKController runs the opposite loop, commanding tip
    velocities toward a FIXED world point so the continuum robot cancels the
    arm's motion.
    """
    from opencr_mujoco.controllers.combined_controller import CombinedController
    from opencr_mujoco.controllers.tdcr_ik_controller import TDCRIKController

    combined = CombinedController(
        model,
        data,
        {
            "tendon_distance_mm": 4.5,  # ftdcr_v4 geometry
            "angle_offset_rad_ccw": [0, -0.5236, -1.0472],
            "fps": CONTROL_FPS,
        },
    )
    tdcr_ik = TDCRIKController(
        model,
        data,
        tendon_distance_mm=4.5,
        angle_offset_rad_ccw=[0, -0.5236, -1.0472],
        velocity_scale=0.65,
        damping_factor=0.01,
        fps=CONTROL_FPS,
        verbose=False,
    )
    tendon_ids = list(tdcr_ik.tendon_actuator_ids)

    # The pretension keyframe holds franka HOME in ctrl but not in qpos; snap
    # the arm joints onto their targets so we start quiescent at home.
    for act_id in combined.franka_actuator_ids:
        joint_id = model.actuator_trnid[act_id, 0]
        data.qpos[model.jnt_qposadr[joint_id]] = data.ctrl[act_id]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    settle(model, data, 2.0)

    ee_body = combined.franka_controller.body_id
    ee_home = data.body(ee_body).xpos.copy()
    tip_hold = tip_pos(model, data)  # the point the tip must not leave
    period = 13.0

    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_0")
    base_trail = Trail(seconds=period)  # the TDCR base draws the figure-8
    tip_dev = []

    def ee_target_at(t):
        # vertical figure-8 for the wrist (kept inside the TDCR's
        # bend-correction envelope so the tip CAN stay pinned)
        return figure8(
            ee_home, t, period, span_a=0.06, span_b=0.09, axes=((0, 1, 0), (0, 0, 1))
        )

    def step(t):
        # Franka: chase the moving wrist target
        f_err = ee_target_at(t) - data.body(ee_body).xpos
        # cap below the full per-tick IK step so the position servos keep
        # up (saturating for long winds ctrl far beyond qpos -> instability)
        f_cmd = np.clip(60.0 * f_err, -0.7, 0.7)
        target = combined.compute_target_qpos(
            {"dx": f_cmd[0], "dy": f_cmd[1], "dz": f_cmd[2]},
            {"x": 0.0, "y": 0.0},
        )
        # TDCR: chase the FIXED tip point (cancels the arm's motion)
        t_err = tip_hold - tip_pos(model, data)
        t_cmd = np.clip(50.0 * t_err, -1.0, 1.0)
        tdcr_target = tdcr_ik.compute_target_qpos(
            {"vx": t_cmd[0], "vy": t_cmd[1], "vz": t_cmd[2]}
        )
        for act_id in tendon_ids:
            target[act_id] = tdcr_target[act_id]
        data.ctrl[:] = target

        if t > 1.5:
            base_trail.append(data.body(base_body).xpos)
        if t > period:
            tip_dev.append(np.linalg.norm(t_err))

    def draw(scene):
        _draw_trail(scene, base_trail.points, 0.0022, (1.0, 0.6, 0.1))
        _add_marker(scene, tip_hold, 0.007, (1.0, 0.2, 0.2, 0.95))

    def report():
        if tip_dev:
            d = np.asarray(tip_dev) * 1000
            print(
                f"  tip hold deviation (post-ramp): mean {d.mean():.1f} mm, "
                f"max {d.max():.1f} mm"
            )

    return step, draw, report


BALL_RADIUS = 0.05
BALL_START = (0.13, 0.0)  # teed +X, clear of the rest tip and wind-up
WIND_CURL = (5.0, 1.0, 0.0)  # proximal pendulum bend that stores the swing
WIND_AZIMUTH = np.pi  # bend AWAY from the ball (toward -X)
RELEASE_SLACK = 0.04  # tendons paid fully limp -> a free, natural swing
GOAL_CENTER = (0.42, 0.0)  # +X, where the struck ball reliably rolls
GOAL_WIDTH = 0.26
GOAL_HEIGHT = 0.14
GOAL_DEPTH = 0.12


def _inject_kick_props(scene_path):
    """Add the ball and the goal to the generated hanging-TDCR scene.

    The goal faces -X (toward the robot): two posts split along Y, a crossbar,
    and a shallow dead net behind so the struck ball settles inside.
    Contact parameters match the tuned settings used elsewhere in the project.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(scene_path)
    worldbody = tree.getroot().find("worldbody")

    ball = ET.SubElement(
        worldbody,
        "body",
        name="ball",
        pos=f"{BALL_START[0]} {BALL_START[1]} {BALL_RADIUS}",
    )
    ET.SubElement(ball, "freejoint", name="ball_free")
    # beach-ball density: the whole robot weighs ~9 g, so the ball must be
    # light for the strike to transfer real momentum.
    # condim=6 makes rolling friction PHYSICAL (condim=3 ignores the third
    # coefficient and the roll decelerates only through solver artifacts,
    # which made the stopping distance irreproducible); the dead net absorbs
    # overshoot, so the ball coasts in with margin.
    ET.SubElement(
        ball,
        "geom",
        name="ball_geom",
        type="sphere",
        size=f"{BALL_RADIUS}",
        density="40",
        contype="1",
        conaffinity="1",
        condim="6",
        priority="1",
        friction="1.0 0.005 0.001",
        solimp="0.8 0.9 0.001",
        rgba="0.95 0.95 0.95 1.0",
    )

    gx, gy = GOAL_CENTER
    half_w, h, r, d = GOAL_WIDTH / 2, GOAL_HEIGHT, 0.007, GOAL_DEPTH
    goal = ET.SubElement(worldbody, "body", name="goal", pos=f"{gx} {gy} 0")
    # uprights split along Y, crossbar across Y (goal mouth faces -X)
    for sign, name in ((-1, "right"), (1, "left")):
        ET.SubElement(
            goal,
            "geom",
            name=f"goal_post_{name}",
            type="capsule",
            fromto=f"0 {sign * half_w} 0 0 {sign * half_w} {h}",
            size=f"{r}",
            contype="1",
            conaffinity="1",
            rgba="0.98 0.98 0.98 1.0",
        )
    ET.SubElement(
        goal,
        "geom",
        name="goal_crossbar",
        type="capsule",
        fromto=f"0 {-half_w} {h} 0 {half_w} {h}",
        size=f"{r}",
        contype="1",
        conaffinity="1",
        rgba="0.98 0.98 0.98 1.0",
    )
    # shallow net (back + side panels), overdamped + high priority so it
    # absorbs the shot rather than returning it
    net_rgba = "0.85 0.88 0.92 0.35"
    ET.SubElement(
        goal,
        "geom",
        name="goal_net_back",
        type="box",
        pos=f"{d} 0 {h / 2}",
        size=f"0.005 {half_w} {h / 2}",
        contype="1",
        conaffinity="1",
        priority="2",
        solref="0.02 6",
        friction="2.0 0.5 0.01",
        rgba=net_rgba,
    )
    for sign, name in ((-1, "right"), (1, "left")):
        ET.SubElement(
            goal,
            "geom",
            name=f"goal_net_{name}",
            type="box",
            pos=f"{d / 2} {sign * half_w} {h / 2}",
            size=f"{d / 2} 0.005 {h / 2}",
            contype="1",
            conaffinity="1",
            priority="2",
            solref="0.02 6",
            friction="2.0 0.5 0.01",
            rgba=net_rgba,
        )

    out = scene_path.parent / "tmp_kick_scene.xml"  # gitignored
    tree.write(out)
    return out


def make_kick_demo(model, data):
    """Elastic whip kick: the hanging tentacle bends hard AWAY from the ball,
    then releases its tendons so the springy backbone swings back through the
    bottom, strikes the teed ball, and sends it rolling into the goal — while
    the robot oscillates and settles.

    Entirely open loop: the wind-up is the only commanded motion; the strike,
    the roll, the net catch, and the robot's post-release ringdown are all
    the native dynamics + contact solver (contact points rendered).
    """
    from opencr_mujoco.tdcr_kinematics.multi_segment_tdcr_kinematics import (
        MultiSegmentTDCRKinematics,
    )

    # per-segment angle offsets match the generator default (30deg steps)
    kin = MultiSegmentTDCRKinematics([3, 3, 3], 4.0, [0.0, 0.5236, 1.0472])
    tendon_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"seg_{s}_ten_{t}")
        for s in range(3)
        for t in range(3)
    ]
    pretension = data.ctrl[np.array(tendon_ids)].copy()

    def apply_clark(mags, azimuth):
        clark = np.zeros(6)
        clark[0::2] = np.asarray(mags) * np.cos(azimuth)
        clark[1::2] = np.asarray(mags) * np.sin(azimuth)
        tendons_mm = kin.clark_to_tendons_mm(clark)
        for i, act_id in enumerate(tendon_ids):
            data.ctrl[act_id] = pretension[i] + tendons_mm[i] * 0.001

    def release():
        # pay every tendon out well past its OWN neutral length so all of
        # them go fully limp; with nothing holding it, the springy backbone
        # (low damping) swings back entirely on its own — a free, natural
        # whip that sweeps the tip low through the bottom, strikes the ball,
        # and rings down. (Per-tendon neutral matters: offsetting from a
        # single tendon's neutral makes the slack asymmetric and the tip
        # flails upward instead of striking low.)
        for i, act_id in enumerate(tendon_ids):
            data.ctrl[act_id] = pretension[i] + RELEASE_SLACK

    settle(model, data, 0.5)

    wind_start, wind_end, release_t = 0.6, 1.6, 2.0  # seconds

    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_trail = Trail(seconds=8.0)
    stats = {"speed": 0.0, "released": False}

    def step(t):
        if t < release_t:
            u = np.clip((t - wind_start) / (wind_end - wind_start), 0.0, 1.0)
            u = 0.5 - 0.5 * np.cos(np.pi * u)  # cosine ease the wind-up
            apply_clark(u * np.asarray(WIND_CURL), WIND_AZIMUTH)
        elif not stats["released"]:
            release()
            stats["released"] = True

        p = data.body(ball_body).xpos
        if t > release_t and p[2] < 0.2:
            ball_trail.append(p)
        if stats["released"]:
            v = data.body(ball_body).cvel[3:]
            stats["speed"] = max(stats["speed"], float(np.linalg.norm(v[:2])))

    def draw(scene):
        _draw_trail(scene, ball_trail.points, 0.003, (0.1, 0.6, 1.0))

    def report():
        p = data.body(ball_body).xpos
        gx, gy = GOAL_CENTER
        scored = (
            p[0] > gx - 0.02
            and p[0] < gx + GOAL_DEPTH + 0.05
            and abs(p[1] - gy) < GOAL_WIDTH / 2
        )
        verdict = "GOAL!" if scored else "no goal"
        print(
            f"  whip kick {stats['speed']:.2f} m/s -> ball at "
            f"({p[0]:.3f}, {p[1]:.3f}): {verdict}"
        )

    return step, draw, report


MAKERS = {
    "tdcr": make_tdcr_demo,
    "franka_tdcr": make_franka_tdcr_demo,
    "kick": make_kick_demo,
}


# --------------------------------------------------------------------------- #
# Recording (ffmpeg mp4 + palette GIF; PIL GIF fallback)
# --------------------------------------------------------------------------- #


def write_outputs(frames, record, size, fps, gif_width=720, gif_fps=15):
    record = Path(record)
    base = record.with_suffix("")  # tolerate name, name.gif, or name.mp4
    base.parent.mkdir(parents=True, exist_ok=True)
    mp4, gif = base.with_suffix(".mp4"), base.with_suffix(".gif")

    if shutil.which("ffmpeg"):
        encode = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{size[0]}x{size[1]}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-crf",
                "22",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4),
            ],
            input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
        )
        if encode.returncode == 0:
            print(
                f"  wrote {mp4} ({len(frames)} frames, "
                f"{mp4.stat().st_size / 1e6:.1f} MB)"
            )
            palette = (
                f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,"
                "split[s0][s1];[s0]palettegen=max_colors=160[p];"
                "[s1][p]paletteuse=dither=bayer:bayer_scale=4"
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(mp4),
                    "-filter_complex",
                    palette,
                    str(gif),
                ],
                check=True,
            )
            print(f"  wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
            return

    print("  ffmpeg not found (or failed) — writing PIL GIF only")
    from PIL import Image

    imgs = [
        Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128)
        for f in frames[:: max(1, fps // 15)]
    ]
    imgs[0].save(
        gif,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / min(fps, 15)),
        loop=0,
        optimize=True,
    )
    print(f"  wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- #
# Run modes
# --------------------------------------------------------------------------- #


def run(
    demo, record=None, seconds=None, fps=30, size=(1280, 840), gif_width=720, gif_fps=15
):
    spec = DEMOS[demo]
    model, data = load_scene(demo)
    step, draw, report = MAKERS[demo](model, data)

    seconds = seconds if seconds is not None else spec["seconds"]
    record_from = spec.get("record_from", 0.0)
    if record_from >= seconds:  # e.g. a shortened smoke run
        record_from = 0.0
    substeps = max(1, int(round(1.0 / (CONTROL_FPS * model.opt.timestep))))
    n_ticks = int(seconds * CONTROL_FPS)

    if spec.get("contact_viz"):
        # contact points rendered as the viewer's "C" toggle does — short
        # normal-aligned cylinders at every active contact (ball-ground,
        # backbone-ball at the strike, ball-net at the catch). The scales
        # multiply model.stat.meansize, which the thin links make tiny
        # (~15 mm), hence the large-looking numbers.
        model.vis.scale.contactwidth = 1.6
        model.vis.scale.contactheight = 0.45

    frames = []
    renderer = camera = viewer = None
    if record:
        # the offscreen framebuffer defaults to 640x480; widen it first
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, size[0])
        model.vis.global_.offheight = max(model.vis.global_.offheight, size[1])
        renderer = mujoco.Renderer(model, height=size[1], width=size[0])
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        vis_option = mujoco.MjvOption()
        if spec.get("contact_viz"):
            vis_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        for k, v in spec["camera"].items():
            if k == "lookat":
                camera.lookat[:] = v
            else:
                setattr(camera, k, v)
    else:
        import mujoco.viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)
        if spec.get("contact_viz"):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    ticks_per_frame = max(1, round(CONTROL_FPS / fps))
    import time as _time

    t_wall = _time.monotonic()

    for tick in range(n_ticks):
        t = tick / CONTROL_FPS
        step(t)
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        if record and t >= record_from and tick % ticks_per_frame == 0:
            renderer.update_scene(data, camera, vis_option)
            draw(renderer.scene)
            frames.append(renderer.render().copy())
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.user_scn.ngeom = 0
            draw(viewer.user_scn)
            viewer.sync()
            # real-time pacing
            t_wall += 1.0 / CONTROL_FPS
            _time.sleep(max(0.0, t_wall - _time.monotonic()))

    if viewer is not None:
        viewer.close()
    report()
    if record:
        write_outputs(frames, record, size, fps, gif_width, gif_fps)
        renderer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop tip demos (README GIFs/videos)"
    )
    parser.add_argument(
        "--demo", choices=sorted(DEMOS), default="tdcr", help="which demo to run"
    )
    parser.add_argument(
        "--record",
        metavar="OUT",
        help="render offscreen; writes OUT.mp4 + OUT.gif " "(extension optional)",
    )
    parser.add_argument("--seconds", type=float, help="override demo duration")
    parser.add_argument(
        "--size", default="1280x840", help="recorded frame size WxH (default 1280x840)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="recorded video frame rate (default 30)"
    )
    parser.add_argument(
        "--gif-width", type=int, default=720, help="GIF width in px (default 720)"
    )
    parser.add_argument(
        "--gif-fps", type=int, default=15, help="GIF frame rate (default 15)"
    )
    args = parser.parse_args()

    w, h = (int(v) for v in args.size.split("x"))
    print(f"Demo '{args.demo}': {DEMOS[args.demo]['description']}")
    run(
        args.demo,
        record=args.record,
        seconds=args.seconds,
        fps=args.fps,
        size=(w, h),
        gif_width=args.gif_width,
        gif_fps=args.gif_fps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
