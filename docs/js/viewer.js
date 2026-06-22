// Demo orchestration: dropdown -> load scene (mujoco-js) -> run the real Python
// controller (Pyodide) -> render with three.js. Entry point for the page.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { MANIFEST_URL } from "./config.js";
import { initMujoco, loadScene } from "./mujoco-loader.js";
import { initPyodide, publishContext, makeRuntime } from "./pyodide-bridge.js";
import { KeyboardCapture } from "./input.js";

const el = (id) => document.getElementById(id);

// Control legends shown in the on-viewer overlay (HTML rows). No modifier keys.
const KEY_LEGENDS = {
  combined: [
    ["Franka move", "W A S D"],
    ["Franka up / down", "Q E"],
    // ["Franka rotate", "I J K L"],
    // ["Franka yaw", "U O"],
    ["TDCR bend", "T F G H"],
    ["TDCR segment", "Z X C"],
    ["Reset to home", "R"],
  ],
  tdcr_joint: [
    ["Bend segment", "T F G H"],
    ["Select segment", "Z X C"],
    ["Reset to home", "R"],
  ],
  ik: [
    ["Move", "W A S D"],
    ["Up / down", "Q E"],
    // ["Roll / pitch", "I J K L"],
    // ["Yaw", "U O"],
    ["Gripper", "N M"],
    ["Reset to home", "H"],
  ],
  joint: [
    ["Nudge joints", "W A S D Q E"],
    ["More joints", "I J K L U O"],
    ["Reset to home", "H"],
  ],
  tdcr_ik: [
    ["Move tip", "W A S D"],
    ["Up / down", "Q E"],
    ["Reset to home", "R"],
  ],
  tdcr_multipt: [["Drive control", "W A S D Q E"], ["Reset to home", "R"]],
  tdcr_multipt_tension: [["Drive control", "W A S D Q E"], ["Reset to home", "R"]],
};

function legendHTML(controller) {
  const rows = KEY_LEGENDS[controller];
  if (!rows) return "";
  const body = rows
    .map(
      ([label, keys]) =>
        `<tr><td>${label}</td><td>${keys
          .split(" ")
          .map((k) => `<kbd>${k}</kbd>`)
          .join("")}</td></tr>`
    )
    .join("");
  return `<div class="overlay-title">Controls</div><table>${body}</table>`;
}

class Demo {
  constructor() {
    this.canvasWrap = el("viewer-canvas");
    this.statusEl = el("viewer-status");
    this.rail = el("config-rail");

    this.mujoco = null;
    this.pyReady = false;
    this.handle = null;
    this.runtime = null;
    this.loadToken = 0;
    this.selectedId = null;
    this.cards = {};

    this._initThree();
    this.input = new KeyboardCapture(this.canvasWrap);
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  status(msg) {
    if (this.statusEl) this.statusEl.textContent = msg;
  }

  _initThree() {
    const wrap = this.canvasWrap;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    wrap.appendChild(this.renderer.domElement);

    // On-viewer control overlay (bottom-right).
    this.overlay = document.createElement("div");
    this.overlay.className = "viewer-overlay";
    this.overlay.style.display = "none";
    wrap.appendChild(this.overlay);

    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xeef0f3);

    // The loader swizzles MuJoCo Z-up into three.js Y-up, so keep the default
    // Y-up camera.
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.position.set(0.9, 0.7, 0.9);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0.3, 0);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(3, 6, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 30;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(-3, 2, -2);
    this.scene.add(fill);

    this.sceneRoot = new THREE.Group();
    this.scene.add(this.sceneRoot);

    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _resize() {
    const w = this.canvasWrap.clientWidth || 800;
    const h = this.canvasWrap.clientHeight || 520;
    this.renderer.setSize(w, h); // updateStyle: keep CSS size = container, buffer = size*dpr
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  async init() {
    // Revalidate so a rebuilt manifest (new/changed demos) is picked up on a
    // normal reload instead of a heuristically-cached stale copy. See the
    // matching note in mujoco-loader.js writeSceneFiles.
    const manifest = await (await fetch(MANIFEST_URL, { cache: "no-cache" })).json();
    this.configs = manifest.configs;
    this._buildRail();

    // Default to franka_tdcr_combined (else first available) and auto-load it.
    const def =
      this.configs.find((c) => c.id === "franka_tdcr_combined" && c.available) ||
      this.configs.find((c) => c.available);
    if (def) this.selectConfig(def.id);
    else this.status("No demos available.");
  }

  _buildRail() {
    this.rail.innerHTML = "";
    this.cards = {};
    for (const c of this.configs) {
      const card = document.createElement("div");
      card.className = "card config-card" + (c.available ? "" : " config-card--unavailable");
      card.dataset.id = c.id;
      card.setAttribute("role", "option");
      card.tabIndex = 0;

      const preview = document.createElement("div");
      preview.className = "card-preview";
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = c.id;
      img.src = `assets/teleopPreviews/${c.id}.gif`;
      img.onerror = () => {
        preview.classList.add("card-preview--empty");
        preview.innerHTML = '<span class="card-preview__msg">no preview yet</span>';
      };
      preview.appendChild(img);

      const body = document.createElement("div");
      body.className = "card-body";
      const name = document.createElement("div");
      name.className = "card-name";
      name.textContent = c.id;
      name.title = c.id;
      body.appendChild(name);

      card.append(preview, body);
      if (!c.available) {
        const badge = document.createElement("span");
        badge.className = "badge text-bg-secondary card-badge";
        badge.textContent = "soon";
        card.appendChild(badge);
      }
      const choose = () => this.selectConfig(c.id);
      card.addEventListener("click", choose);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          choose();
        }
      });
      this.rail.appendChild(card);
      this.cards[c.id] = card;
    }
  }

  selectConfig(id) {
    if (id === this.selectedId) return;
    this.selectedId = id;
    for (const [cid, card] of Object.entries(this.cards)) {
      card.classList.toggle("config-card--selected", cid === id);
    }
    this.cards[id]?.scrollIntoView({ inline: "nearest", block: "nearest", behavior: "smooth" });
    this.loadConfig(id);
  }

  async loadConfig(id) {
    const entry = this.configs.find((c) => c.id === id);
    if (!entry) return;
    const token = ++this.loadToken;
    this.overlay.style.display = "none";

    // Tear down previous runtime/scene.
    this.runtime = null;
    if (this.handle) {
      this.sceneRoot.remove(this.handle.root);
      this.handle.dispose();
      this.handle = null;
    }

    if (!entry.available) {
      this.status(
        `“${entry.id}” references a scene that isn't generated yet (${entry.scene_ref}). ` +
          `Generate it and re-run build_site.py.`
      );
      return;
    }

    try {
      this.status("Loading MuJoCo engine…");
      if (!this.mujoco) this.mujoco = await initMujoco();
      if (token !== this.loadToken) return;

      this.status("Loading scene…");
      this.handle = await loadScene(this.mujoco, entry, this.sceneRoot);
      if (token !== this.loadToken) return;
      this._frameCamera(this.handle.root, entry);

      if (!this.pyReady) {
        await initPyodide((m) => this.status(m));
        this.pyReady = true;
      }
      if (token !== this.loadToken) return;

      this.status("Starting controller…");
      publishContext(this.mujoco, this.handle.model, this.handle.data);
      this.runtime = makeRuntime(entry);
      if (this.runtime.unsupported) {
        this.status(`Loaded — ${this.runtime.unsupported} Drag to orbit the view.`);
        this.runtime = null;
      } else {
        // Focus so keyboard control works immediately, but don't scroll the
        // page back to the demo -- loading is slow and the reader may have
        // scrolled away by the time it finishes.
        this.canvasWrap.focus({ preventScroll: true });
        this.status("Ready — click the view, then use the keys shown.");
        const html = legendHTML(entry.controller);
        if (html) {
          this.overlay.innerHTML = html;
          this.overlay.style.display = "block";
        }
      }
    } catch (err) {
      this.status("");
    }
  }

  _frameCamera(root, entry) {
    this.scene.updateMatrixWorld(true);
    // Bound the robot only -- exclude the (huge) ground plane so it doesn't
    // dominate the framing.
    const box = new THREE.Box3();
    root.traverse((o) => {
      if (o.isMesh && !o.userData.isPlane) box.expandByObject(o);
    });
    if (box.isEmpty()) box.setFromObject(root);
    if (box.isEmpty()) return;

    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const center = sphere.center;
    const radius = sphere.radius || 0.5;

    // Distance so the bounding sphere fits the (vertical) FOV, with margin.
    const vfov = THREE.MathUtils.degToRad(this.camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.camera.aspect);
    const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.15;

    this.controls.target.copy(center);
    // Y-up: 3/4 view, slightly above the robot's center. Both demos use the
    // same side now. The standalone TDCR used to be spun 180deg about vertical
    // so its TFGH keys lined up; that alignment now comes from the controller's
    // command_frame_offset_rad (pi) instead, so the camera stays on the
    // original side (rotated 180 about vertical from the old spun view).
    const dir = new THREE.Vector3(0.6, 0.35, 1).normalize();
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.camera.near = Math.max(dist - radius * 2, 0.001);
    this.camera.far = dist + radius * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _animate(now) {
    requestAnimationFrame(this._animate);
    this.controls.update();

    if (this.handle) {
      const mujoco = this.mujoco;
      const model = this.handle.model;
      const data = this.handle.data;
      const dt = model.opt.timestep || 0.002;

      // Control tick: drives data.ctrl via the real Python controller.
      if (this.runtime) {
        const st = this.input.getState();
        try {
          this.runtime.setKeys(st.keys);
          this.runtime.step();
        } catch (e) {
          // Ignore a transient frame error; the controller keeps running.
        }
      }

      // Step physics toward real time (bounded to avoid spirals).
      this._acc = (this._acc || 0) + Math.min((now - (this._last || now)) / 1000, 0.05);
      this._last = now;
      let steps = 0;
      while (this._acc >= dt && steps < 40) {
        mujoco.mj_step(model, data);
        this._acc -= dt;
        steps++;
      }
      this.handle.sync();
    }

    this.renderer.render(this.scene, this.camera);
  }
}

const demo = new Demo();
demo.init();
