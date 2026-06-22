// Pinned CDN versions for the static demo. Bump these to upgrade.
// Everything is loaded from a CDN as ES modules / static assets -- no npm build.

export const THREE_VERSION = "0.181.0"; // matches zalo/mujoco_wasm peer dep
export const MUJOCO_JS_VERSION = "0.0.7"; // compiled MuJoCo WASM engine
export const PYODIDE_VERSION = "0.27.2"; // ships numpy + scipy as loadable pkgs

export const THREE_URL = `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/build/three.module.js`;
export const THREE_ADDONS = `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/examples/jsm/`;
export const MUJOCO_JS_URL = `https://cdn.jsdelivr.net/npm/mujoco-js@${MUJOCO_JS_VERSION}/dist/mujoco_wasm.js`;
export const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Where the build script mirrors scenes (relative to docs/ root).
export const SCENES_BASE = "scenes/";
export const MANIFEST_URL = "manifest.json";
