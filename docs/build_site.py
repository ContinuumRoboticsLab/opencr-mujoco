#!/usr/bin/env python3
"""Build the static GitHub-Pages demo site under ``docs/``.

This script is run manually (no CI) from an environment where ``mujoco`` is
importable, e.g.::

    conda activate genericMujoco
    python docs/build_site.py

It does three things, all emitting static files under ``docs/``:

1. Copies the real ``opencr_mujoco/`` and ``configs/`` trees into ``docs/pysrc/`` so the
   in-browser Pyodide runtime can import and run the *actual* controllers.
2. For every ``configs/teleop/*.json`` it resolves the referenced scene, mirrors
   the scene XML + every ``<include>`` + every mesh/texture asset into
   ``docs/scenes/`` (preserving the relative paths MuJoCo needs so the browser
   ``mujoco_wasm`` engine can resolve includes/meshdir natively).
3. Writes ``docs/manifest.json`` describing every teleop config (controller,
   params, scene file list, actuator/body names in id order, and the
   ``pretension`` keyframe) so the JS loader + Python ``mujoco`` shim never have
   to re-parse the XML in the browser.

Scenes that don't exist yet (e.g. generated ``assets/tdcr/*.xml``) are recorded
with ``available: false`` -- the dropdown still lists them and the viewer shows a
friendly message rather than crashing.

Convert mp4 to gif easily ->
ffmpeg  -i ./tdcr_keyboard_modular.mp4  -vf "fps=30,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 tdcr_keyboard_modular.gif
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from generate import tdcr_geometry_from_scene  # noqa: E402

DOCS = REPO / "docs"
TELEOP_DIR = REPO / "configs" / "teleop"
ASSETS = REPO / "assets"

# Teleop configs the published site exposes. Keep this small and curated —
# each one needs a recorded preview GIF under docs/assets/teleopPreviews/.
# Set to None to publish every configs/teleop/*.json instead.
SITE_CONFIGS = {"franka_tdcr_combined", "tdcr_keyboard"}

PYSRC_OUT = DOCS / "pysrc"  # human-inspectable copy of opencr_mujoco/ + configs/
PYEXTRA = DOCS / "pyextra"  # authored browser-only python (shim, runtime, pynput stub)
SCENES_OUT = DOCS / "scenes"
MANIFEST_OUT = DOCS / "manifest.json"
PYBUNDLE_OUT = DOCS / "pybundle.tar.gz"  # what Pyodide unpacks at runtime

# Actuator name -> kind classification.
TENDON_RE = re.compile(r"^seg_(\d+)_ten_(\d+)$")
FRANKA_RE = re.compile(r"(panda|gripper|finger)", re.IGNORECASE)

# ----------------------------------------------------------------------------
# 1. Copy Python sources for Pyodide
# ----------------------------------------------------------------------------

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store")


def copy_pysrc() -> None:
    PYSRC_OUT.mkdir(parents=True, exist_ok=True)
    for name in ("opencr_mujoco", "configs"):
        src = REPO / name
        dst = PYSRC_OUT / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=_IGNORE)
        print(f"  copied {name}/ -> {dst.relative_to(DOCS)}")


def build_python_bundle() -> None:
    """Pack the Python tree Pyodide imports at runtime into one tarball.

    Contents (all at archive root so they're importable directly):
      opencr_mujoco/, configs/   -- the real repo code (copied above), plus
      mujoco/, pynput/, web_runtime.py -- authored browser shims from pyextra/.
    """
    if PYBUNDLE_OUT.exists():
        PYBUNDLE_OUT.unlink()
    with tarfile.open(PYBUNDLE_OUT, "w:gz") as tar:
        tar.add(
            PYSRC_OUT / "opencr_mujoco", arcname="opencr_mujoco", filter=_tar_filter
        )
        tar.add(PYSRC_OUT / "configs", arcname="configs", filter=_tar_filter)
        for item in sorted(PYEXTRA.iterdir()):
            if item.name in ("__pycache__", ".DS_Store"):
                continue
            tar.add(item, arcname=item.name, filter=_tar_filter)
    size_kb = PYBUNDLE_OUT.stat().st_size / 1024
    print(f"  wrote {PYBUNDLE_OUT.relative_to(DOCS)} ({size_kb:.0f} KB)")


def _tar_filter(info: tarfile.TarInfo):
    base = info.name.rsplit("/", 1)[-1]
    if (
        "__pycache__" in info.name
        or base.endswith((".pyc", ".pyo"))
        or base == ".DS_Store"
    ):
        return None
    return info


# ----------------------------------------------------------------------------
# 2. Resolve scene -> set of files (scene + includes + meshes/textures)
# ----------------------------------------------------------------------------


def _compiler_dirs(root: ET.Element) -> tuple[str, str]:
    """Return (meshdir, texturedir) declared by any <compiler> in this fragment."""
    meshdir, texturedir = "", ""
    for comp in root.iter("compiler"):
        meshdir = comp.get("meshdir", meshdir)
        texturedir = comp.get("texturedir", texturedir)
    return meshdir, texturedir


def collect_scene_files(scene_path: Path) -> list[Path]:
    """Recursively collect every file MuJoCo needs to load ``scene_path``.

    Returns absolute paths (scene + transitive includes + mesh/texture assets).
    ``meshdir``/``texturedir`` are resolved relative to the *top* model dir, and
    ``<include>`` relative to the including file's dir, matching MuJoCo.
    """
    top_dir = scene_path.parent
    files: list[Path] = []
    seen: set[Path] = set()

    # First pass over the whole include tree to discover compiler dirs.
    meshdir = texturedir = ""

    def walk_includes(path: Path) -> None:
        nonlocal meshdir, texturedir
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        files.append(path)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:  # pragma: no cover - surfaced to user
            raise RuntimeError(f"Failed to parse {path}: {exc}") from exc
        md, td = _compiler_dirs(root)
        meshdir = md or meshdir
        texturedir = td or texturedir
        for inc in root.iter("include"):
            f = inc.get("file")
            if f:
                walk_includes((path.parent / f))

    walk_includes(scene_path)

    # Second pass: gather mesh/texture/hfield/skin file references.
    asset_files: set[Path] = set()
    for path in list(files):
        root = ET.parse(path).getroot()
        for tag, subdir in (
            ("mesh", meshdir),
            ("texture", texturedir),
            ("hfield", meshdir),
            ("skin", meshdir),
        ):
            for el in root.iter(tag):
                f = el.get("file")
                if not f:
                    continue
                resolved = (top_dir / subdir / f).resolve()
                asset_files.add(resolved)

    for f in sorted(asset_files):
        if f not in seen:
            seen.add(f)
            files.append(f)

    return files


# Configs that get interactive "play" props (a pedestal of knock-around balls
# and cylinders that the web runtime auto-respawns when they fall off).
PLAY_PROPS_FOR = {"franka_tdcr_combined"}

# Pedestal centre + table-top height, under the settled home tip (~0.62, 0,
# 0.52) so the props sit right in the TDCR's sweep when the arm dips down.
_PLAY_CX, _PLAY_CY, _PLAY_TOP = 0.58, 0.0, 0.40
# (dx, dy, kind, "r g b") relative to the pedestal centre. A small set reads
# cleaner than a crowded table: one ball up front, two cylinders behind it.
_PLAY_SLOTS = [
    (0.00, -0.05, "ball", "0.92 0.30 0.25"),
    (-0.055, 0.045, "cyl", "0.40 0.80 0.45"),
    (0.055, 0.045, "cyl", "0.70 0.45 0.90"),
]
_PLAY_BALL_R = 0.028
_PLAY_CYL_HALF = 0.035


def inject_play_props(scene_xml: Path) -> None:
    """Add a pedestal + free-floating balls/cylinders to a mirrored scene.

    Props are named ``prop_<i>`` with joints ``prop_<i>_free`` so the web
    runtime can discover and respawn them. The scene's pretension keyframe now
    carries an explicit ``qpos`` (the Franka home pose), and MuJoCo requires
    keyframe ``qpos`` to match ``nq`` — so we extend it with each prop's initial
    free-joint qpos (pedestal pose + identity quat). That keeps the keyframe
    valid AND resets the props onto the pedestal. Contact params match the
    tuned settings used elsewhere in the project.

    Also forces gravity on: this scene is the sysid scene, which disables
    gravity (``gravity="0 0 0"``) for calibration, so without this the props
    would just drift off when bumped instead of falling and resting. Web-only;
    the source generation config stays gravity-free for sysid/golden.
    """
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("gravity", "0 0 -9.81")
    # The sysid scene runs the noslip friction refinement (noslip_iterations=1),
    # which pumps energy into the light props — the cylinders buzz and slowly
    # creep instead of settling. The demo needs no slip accuracy, so turn it off.
    opt.set("noslip_iterations", "1")
    wb = root.find("worldbody")

    ped = ET.SubElement(
        wb,
        "body",
        name="play_pedestal",
        pos=f"{_PLAY_CX} {_PLAY_CY} {_PLAY_TOP / 2:.4f}",
    )
    ET.SubElement(
        ped,
        "geom",
        name="play_pedestal_geom",
        type="box",
        size=f"0.11 0.11 {_PLAY_TOP / 2:.4f}",
        contype="1",
        conaffinity="1",
        rgba="0.32 0.34 0.40 1",
    )

    prop_qpos = []  # free-joint init (pos + identity quat), in body order
    for i, (dx, dy, kind, rgb) in enumerate(_PLAY_SLOTS):
        z = _PLAY_TOP + (_PLAY_BALL_R if kind == "ball" else _PLAY_CYL_HALF)
        px, py = _PLAY_CX + dx, _PLAY_CY + dy
        body = ET.SubElement(
            wb, "body", name=f"prop_{i}", pos=f"{px:.4f} {py:.4f} {z:.4f}"
        )
        ET.SubElement(body, "freejoint", name=f"prop_{i}_free")
        prop_qpos += [px, py, z, 1.0, 0.0, 0.0, 0.0]
        common = dict(
            name=f"prop_{i}_geom",
            density="50",
            contype="1",
            conaffinity="1",
            condim="3",
            friction="1.0 0.02 0.001",
            solimp="0.9 0.95 0.001",
            rgba=f"{rgb} 1",
        )
        if kind == "ball":
            ET.SubElement(body, "geom", type="sphere", size=f"{_PLAY_BALL_R}", **common)
        else:
            # condim=4 adds torsional friction so a flat-faced cylinder resting
            # on the box doesn't micro-spin/buzz (still slides/topples when hit).
            ET.SubElement(
                body,
                "geom",
                type="cylinder",
                size=f"{_PLAY_BALL_R} {_PLAY_CYL_HALF}",
                **{**common, "condim": "4"},
            )

    # Keep the pretension keyframe valid: the props' free joints add qpos at the
    # end of the state vector, so extend the keyframe qpos to match (and qvel if
    # present). Without this, MuJoCo's qpos count no longer matches nq and the
    # props reset off the pedestal. ctrl is unchanged (props have no actuators).
    key = root.find("keyframe/key[@name='pretension']")
    if key is not None and key.get("qpos"):
        extra = " ".join("0" if v == 0 else repr(float(v)) for v in prop_qpos)
        key.set("qpos", f"{key.get('qpos')} {extra}")
        if key.get("qvel"):
            key.set("qvel", f"{key.get('qvel')} " + " ".join(["0"] * len(prop_qpos)))

    tree.write(scene_xml)


def mirror_scene(scene_path: Path) -> dict:
    """Copy a scene + its dependencies into docs/scenes/, mirroring paths under
    ``assets/``. Returns dict with ``scene`` relpath and ``files`` relpath list."""
    files = collect_scene_files(scene_path)
    rels: list[str] = []
    for f in files:
        try:
            rel = f.relative_to(ASSETS)
        except ValueError:
            print(f"  ! {f} is outside assets/; skipping (not mirrored)")
            continue
        dst = SCENES_OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        rels.append(str(rel).replace("\\", "/"))
    scene_rel = str(scene_path.relative_to(ASSETS)).replace("\\", "/")
    return {"scene": scene_rel, "files": rels}


# ----------------------------------------------------------------------------
# 3. Extract model metadata (actuators, bodies, pretension keyframe)
# ----------------------------------------------------------------------------


def scene_metadata(scene_path: Path) -> dict:
    """Load the scene with MuJoCo and extract actuator/body/keyframe metadata."""
    import mujoco  # imported lazily so the copy step works without mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))

    def name(obj, i):
        return mujoco.mj_id2name(model, obj, i)

    act_names = [name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    body_names = [name(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]

    tendon_actuators = []  # {name, id, seg, ten}
    franka_actuators = []  # {name, id}
    for i, nm in enumerate(act_names):
        if nm is None:
            continue
        m = TENDON_RE.match(nm)
        if m:
            tendon_actuators.append(
                {"name": nm, "id": i, "seg": int(m.group(1)), "ten": int(m.group(2))}
            )
        elif FRANKA_RE.search(nm):
            franka_actuators.append({"name": nm, "id": i})

    # pretension keyframe (home pose)
    pretension = None
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    if key_id >= 0:
        pretension = {
            "qpos": model.key_qpos[key_id].tolist(),
            "ctrl": model.key_ctrl[key_id].tolist(),
        }

    return {
        "nu": int(model.nu),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nkey": int(model.nkey),
        "timestep": float(model.opt.timestep),
        "actuator_names": act_names,
        "body_names": body_names,
        "actuator_ctrlrange": model.actuator_ctrlrange.tolist(),
        "tendon_actuators": tendon_actuators,
        "franka_actuators": franka_actuators,
        "pretension": pretension,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def build_manifest() -> list[dict]:
    entries: list[dict] = []
    for cfg_path in sorted(TELEOP_DIR.glob("*.json")):
        cfg_id = cfg_path.stem
        if SITE_CONFIGS is not None and cfg_id not in SITE_CONFIGS:
            continue  # not in the published set
        with open(cfg_path) as f:
            cfg = json.load(f)

        scene_rel = cfg.get("scene")

        # Fill missing geometric TDCR params (angle offsets + tendon distance)
        # from the scene's generation config, exactly like teleop.py does, so
        # the in-browser controllers coordinate the tendons correctly even
        # when the teleop config doesn't (and can't drift) carry them.
        controller_params = json.loads(json.dumps(cfg.get("controller_params", {})))
        if scene_rel:
            geom = tdcr_geometry_from_scene(scene_rel)
            target = (
                controller_params.setdefault("tdcr", {})
                if cfg.get("controller") == "combined"
                else controller_params
            )
            for key, value in geom.items():
                if target.get(key) is None:  # explicit config value wins
                    target[key] = value

        entry = {
            "id": cfg_id,
            "label": cfg_id,
            "description": cfg.get("description", ""),
            "controller": cfg.get("controller"),
            "input_device": cfg.get("input_device"),
            "controller_params": controller_params,
            "fps": cfg.get(
                "fps", cfg.get("simulation_params", {}).get("render_fps", 100)
            ),
            "scene_ref": scene_rel,
            "available": False,
        }

        scene_path = (REPO / scene_rel).resolve() if scene_rel else None
        if not scene_path or not scene_path.exists():
            print(f"  - {cfg_id}: scene '{scene_rel}' missing -> available:false")
            entries.append(entry)
            continue

        try:
            mirrored = mirror_scene(scene_path)
            if cfg_id in PLAY_PROPS_FOR:
                # inject into the mirrored copy only; the original asset and
                # the actuator/keyframe metadata (read below) are untouched
                inject_play_props(SCENES_OUT / mirrored["scene"])
                print(f"    + play props injected into {mirrored['scene']}")
            meta = scene_metadata(scene_path)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  ! {cfg_id}: failed ({exc}) -> available:false")
            entry["error"] = str(exc)
            entries.append(entry)
            continue

        entry.update(mirrored)
        entry["model"] = meta
        entry["available"] = True
        n_seg = len({a["seg"] for a in meta["tendon_actuators"]})
        print(
            f"  + {cfg_id}: {mirrored['scene']} "
            f"(nu={meta['nu']}, {len(meta['tendon_actuators'])} tendons/{n_seg} seg, "
            f"{len(meta['franka_actuators'])} franka, {len(mirrored['files'])} files)"
        )
        entries.append(entry)
    return entries


def main() -> int:
    print(f"Building docs site from {REPO}")
    if SCENES_OUT.exists():
        shutil.rmtree(SCENES_OUT)
    SCENES_OUT.mkdir(parents=True, exist_ok=True)

    print("Copying Python sources for Pyodide...")
    copy_pysrc()

    print("Packing Python bundle for Pyodide...")
    build_python_bundle()

    print("Mirroring teleop scenes + building manifest...")
    entries = build_manifest()

    manifest = {"configs": entries}
    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2)

    n_ok = sum(1 for e in entries if e["available"])
    print(
        f"\nWrote {MANIFEST_OUT.relative_to(DOCS)}: "
        f"{len(entries)} configs ({n_ok} available, {len(entries) - n_ok} missing)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
