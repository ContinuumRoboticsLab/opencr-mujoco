// MuJoCo (mujoco-js WASM) loader + three.js renderer.
//
// Adapted from zalo/mujoco_wasm's src/mujocoUtils.js (loadSceneFromURL) -- the
// canonical renderer -- so the scene looks like MuJoCo's: real materials and
// textures from the model (mat_rgba / tex_data), and MuJoCo's Z-up coordinates
// swizzled into three.js's Y-up convention at load AND every frame:
//   position  (x, y, z) -> (x,  z, -y)
//   quaternion(w, x, y, z) wxyz -> three (x,y,z,w) = (-x, -z, y, -w)
//
// API surface used (mujoco-js@0.0.7): mujoco.FS.*, MjModel.loadFromXML,
// new MjData(model), mj_forward/step/resetDataKeyframe, model.geom_*/mesh_*/
// mat_*/tex_*, data.xpos/xquat (typed arrays).

import * as THREE from "three";
import { MUJOCO_JS_URL, SCENES_BASE } from "./config.js";

let _mujoco = null;

export async function initMujoco() {
  if (_mujoco) return _mujoco;
  const { default: load_mujoco } = await import(/* @vite-ignore */ MUJOCO_JS_URL);
  _mujoco = await load_mujoco();
  try {
    _mujoco.FS.mkdir("/working");
    _mujoco.FS.mount(_mujoco.MEMFS, { root: "." }, "/working");
  } catch (e) {
    /* already mounted */
  }
  return _mujoco;
}

const _textExt = new Set(["xml", "mjcf", "urdf", "txt", "json"]);

async function writeSceneFiles(mujoco, entry) {
  const files = entry.files || [entry.scene];
  await Promise.all(
    files.map(async (rel) => {
      // `no-cache` = always revalidate with the server (conditional GET) before
      // reusing a cached copy. Without it, `python -m http.server` sends no
      // Cache-Control, so browsers heuristically cache scene XML/meshes and keep
      // serving a stale pre-rebuild scene (e.g. missing freshly-added props)
      // until the heuristic window expires. 304s keep unchanged meshes cheap.
      const resp = await fetch(SCENES_BASE + rel, { cache: "no-cache" });
      if (!resp.ok) throw new Error(`fetch ${rel} -> ${resp.status}`);
      const dir = ("/working/" + rel).split("/").slice(0, -1).join("/");
      mkdirp(mujoco, dir);
      const ext = rel.split(".").pop().toLowerCase();
      if (_textExt.has(ext)) mujoco.FS.writeFile("/working/" + rel, await resp.text());
      else mujoco.FS.writeFile("/working/" + rel, new Uint8Array(await resp.arrayBuffer()));
    })
  );
}

function mkdirp(mujoco, dir) {
  let cur = "";
  for (const p of dir.split("/").filter(Boolean)) {
    cur += "/" + p;
    try {
      mujoco.FS.mkdir(cur);
    } catch (e) {
      /* exists */
    }
  }
}

// --- Z-up (MuJoCo) -> Y-up (three.js) swizzles (match zalo) ---
function setPos(target, buf, i) {
  target.set(buf[i * 3 + 0], buf[i * 3 + 2], -buf[i * 3 + 1]);
}
function setQuat(target, buf, i) {
  target.set(-buf[i * 4 + 1], -buf[i * 4 + 3], buf[i * 4 + 2], -buf[i * 4 + 0]);
}

function buildMeshGeometry(model, meshId) {
  const vadr = model.mesh_vertadr[meshId];
  const vnum = model.mesh_vertnum[meshId];
  const fadr = model.mesh_faceadr[meshId];
  const fnum = model.mesh_facenum[meshId];

  const src = model.mesh_vert.subarray(vadr * 3, (vadr + vnum) * 3);
  const verts = new Float32Array(src.length);
  for (let v = 0; v < src.length; v += 3) {
    verts[v + 0] = src[v + 0];
    verts[v + 1] = src[v + 2];
    verts[v + 2] = -src[v + 1];
  }
  const faces = model.mesh_face.subarray(fadr * 3, (fadr + fnum) * 3);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geometry.setIndex(Array.from(faces));
  geometry.computeVertexNormals();
  return geometry;
}

// Build a three.js texture from the model's tex_data for material `matId`, or
// null if the material has no RGB texture.
function buildTexture(model, matId) {
  if (model.geom_matid == null || model.mat_texid == null || model.tex_data == null) return null;
  const mjNTEXROLE = 10;
  const mjTEXROLE_RGB = 1;
  const texId = model.mat_texid[matId * mjNTEXROLE + mjTEXROLE_RGB];
  if (texId === undefined || texId === -1) return null;

  const width = model.tex_width[texId];
  const height = model.tex_height[texId];
  const offset = model.tex_adr[texId];
  const channels = model.tex_nchannel ? model.tex_nchannel[texId] : 3;
  const data = model.tex_data;
  const rgba = new Uint8Array(width * height * 4);
  for (let p = 0; p < width * height; p++) {
    rgba[p * 4 + 0] = data[offset + p * channels + 0];
    rgba[p * 4 + 1] = channels > 1 ? data[offset + p * channels + 1] : rgba[p * 4 + 0];
    rgba[p * 4 + 2] = channels > 2 ? data[offset + p * channels + 2] : rgba[p * 4 + 0];
    rgba[p * 4 + 3] = channels > 3 ? data[offset + p * channels + 3] : 255;
  }
  const tex = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  const rep = model.mat_texrepeat;
  tex.repeat.set(rep ? rep[matId * 2] : 1, rep ? rep[matId * 2 + 1] : 1);
  tex.needsUpdate = true;
  return tex;
}

function buildMaterial(model, g) {
  const matId = model.geom_matid ? model.geom_matid[g] : -1;
  let color;
  if (matId !== -1 && model.mat_rgba) {
    color = [model.mat_rgba[matId * 4], model.mat_rgba[matId * 4 + 1], model.mat_rgba[matId * 4 + 2], model.mat_rgba[matId * 4 + 3]];
  } else {
    color = [model.geom_rgba[g * 4], model.geom_rgba[g * 4 + 1], model.geom_rgba[g * 4 + 2], model.geom_rgba[g * 4 + 3]];
  }
  const tex = matId !== -1 ? buildTexture(model, matId) : null;
  const opacity = color[3];
  const params = {
    color: new THREE.Color(color[0], color[1], color[2]),
    transparent: opacity < 1.0,
    opacity,
    metalness: matId !== -1 ? 0.1 : 0.0,
    roughness:
      matId !== -1 && model.mat_shininess ? 1.0 - model.mat_shininess[matId] : 0.7,
  };
  if (tex) params.map = tex;
  return new THREE.MeshPhysicalMaterial(params);
}

export class SceneHandle {
  constructor(mujoco, model, data, root) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.root = root;
    this.bodyGroups = [];
  }

  applyKeyframe(keyIndex) {
    if (keyIndex >= 0 && keyIndex < this.model.nkey) {
      this.mujoco.mj_resetDataKeyframe(this.model, this.data, keyIndex);
    }
    this.mujoco.mj_forward(this.model, this.data);
  }

  sync() {
    const xpos = this.data.xpos;
    const xquat = this.data.xquat;
    for (let b = 0; b < this.bodyGroups.length; b++) {
      const grp = this.bodyGroups[b];
      if (!grp) continue;
      setPos(grp.position, xpos, b);
      setQuat(grp.quaternion, xquat, b);
    }
  }

  dispose() {
    this.root.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    if (this.data) this.data.delete();
    if (this.model) this.model.delete();
  }
}

export async function loadScene(mujoco, entry, parent) {
  await writeSceneFiles(mujoco, entry);
  const model = mujoco.MjModel.loadFromXML("/working/" + entry.scene);
  const data = new mujoco.MjData(model);

  const root = new THREE.Group();
  const handle = new SceneHandle(mujoco, model, data, root);

  for (let b = 0; b < model.nbody; b++) {
    handle.bodyGroups[b] = new THREE.Group();
    root.add(handle.bodyGroups[b]);
  }

  const G = mujoco.mjtGeom;
  const meshCache = {};
  for (let g = 0; g < model.ngeom; g++) {
    const group = model.geom_group ? model.geom_group[g] : 0;
    if (group >= 3) continue; // collision-only; MuJoCo hides these by default
    const alpha = model.geom_rgba[g * 4 + 3];

    const type = model.geom_type[g];
    const sx = model.geom_size[g * 3 + 0];
    const sy = model.geom_size[g * 3 + 1];
    const sz = model.geom_size[g * 3 + 2];
    const b = model.geom_bodyid[g];

    let geometry = null;
    let isPlane = false;
    if (type === G.mjGEOM_PLANE.value) {
      isPlane = true;
      geometry = new THREE.PlaneGeometry(100, 100);
    } else if (type === G.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(sx, 24, 16);
    } else if (type === G.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(sx, sy * 2, 8, 16);
    } else if (type === G.mjGEOM_ELLIPSOID.value) {
      geometry = new THREE.SphereGeometry(1, 24, 16);
    } else if (type === G.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(sx, sx, sy * 2, 24);
    } else if (type === G.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(sx * 2, sz * 2, sy * 2);
    } else if (type === G.mjGEOM_MESH.value) {
      const meshId = model.geom_dataid[g];
      if (meshId < 0) continue;
      if (!(meshId in meshCache)) meshCache[meshId] = buildMeshGeometry(model, meshId);
      geometry = meshCache[meshId];
    } else {
      continue;
    }
    if (alpha === 0 && !isPlane) continue; // invisible

    const material = buildMaterial(model, g);
    const mesh = new THREE.Mesh(geometry, material);
    if (type === G.mjGEOM_ELLIPSOID.value) mesh.scale.set(sx, sz, sy);
    mesh.castShadow = !isPlane;
    mesh.receiveShadow = true;
    mesh.userData.isPlane = isPlane;

    // three.js geoms are authored Y-up; CapsuleGeometry/CylinderGeometry run
    // along Y, matching MuJoCo's local Z after the global swizzle, so no extra
    // per-geom rotation is needed beyond the geom quaternion.
    setPos(mesh.position, model.geom_pos, g);
    if (isPlane) {
      mesh.rotateX(-Math.PI / 2); // plane normal +Z (MuJoCo) -> +Y (three)
      if (material.map) material.map.repeat.set(50, 50); // tile the ground checker
    } else {
      setQuat(mesh.quaternion, model.geom_quat, g);
    }
    handle.bodyGroups[b].add(mesh);
  }

  handle.applyKeyframe(findKeyframe(mujoco, model, "pretension"));
  handle.sync();
  parent.add(root);
  return handle;
}

export function findKeyframe(mujoco, model, name) {
  if (model.nkey <= 0) return -1;
  const keyObj = mujoco.mjtObj && mujoco.mjtObj.mjOBJ_KEY;
  if (keyObj) {
    const objVal = keyObj.value !== undefined ? keyObj.value : keyObj;
    for (let i = 0; i < model.nkey; i++) {
      try {
        if (mujoco.mj_id2name(model, objVal, i) === name) return i;
      } catch (e) {
        break;
      }
    }
  }
  return 0;
}
