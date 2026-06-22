// Pyodide bridge: runs the REAL Python controllers in the browser.
//
// Boots Pyodide, loads numpy + scipy, unpacks the project's Python bundle
// (opencr_mujoco/ + configs/ + the mujoco shim + pynput stub + web_runtime), and exposes
// a per-scene Runtime that drives data.ctrl exactly like teleop.py. The Python
// `mujoco` shim reaches MuJoCo through `globalThis.MJ_CTX`, set here.

import { PYODIDE_URL } from "./config.js";

let _pyodide = null;

export async function initPyodide(onStatus = () => {}) {
  if (_pyodide) return _pyodide;
  onStatus("Loading Python runtime…");
  const { loadPyodide } = await import(/* @vite-ignore */ PYODIDE_URL + "pyodide.mjs");
  _pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  // Keep the browser console clean: the reused controllers print status text and
  // Python may emit warnings -- swallow all of it.
  const silent = { batched: () => {} };
  _pyodide.setStdout(silent);
  _pyodide.setStderr(silent);

  onStatus("Loading numpy + scipy…");
  await _pyodide.loadPackage(["numpy", "scipy"], {
    messageCallback: () => {},
    errorCallback: () => {},
  });

  onStatus("Unpacking controllers…");
  const buf = new Uint8Array(await (await fetch("pybundle.tar.gz")).arrayBuffer());
  _pyodide.FS.mkdir("/session");
  _pyodide.FS.writeFile("/session/pybundle.tar.gz", buf);
  await _pyodide.runPythonAsync(`
import sys, tarfile, os
os.makedirs("/session/repo", exist_ok=True)
with tarfile.open("/session/pybundle.tar.gz") as t:
    t.extractall("/session/repo")
if "/session/repo" not in sys.path:
    sys.path.insert(0, "/session/repo")
import web_runtime  # pulls in the mujoco shim + pynput stub + controllers
`);
  return _pyodide;
}

/**
 * Publish the live MuJoCo (mujoco-js) context for the Python shim, including
 * precomputed id<->name tables. Tables are read straight from the model's
 * `names` buffer via the `name_*adr` offset arrays -- no dependence on
 * mj_name2id (unbound) or the mjtObj enum.
 */
export function publishContext(mujoco, model, data) {
  const buf = asBytes(model.names);
  const names = {
    actuator: nameTable(buf, model.name_actuatoradr, model.nu),
    body: nameTable(buf, model.name_bodyadr, model.nbody),
    site: nameTable(buf, model.name_siteadr, model.nsite),
    joint: nameTable(buf, model.name_jntadr, model.njnt),
    key: nameTable(buf, model.name_keyadr, model.nkey),
  };
  globalThis.MJ_CTX = { mujoco, model, data, names };
}

const _decoder = new TextDecoder();

function asBytes(names) {
  if (names instanceof Uint8Array) return names;
  if (ArrayBuffer.isView(names)) return new Uint8Array(names.buffer, names.byteOffset, names.byteLength);
  return null;
}

function nameTable(buf, adr, n) {
  const out = [];
  if (!buf || !adr) {
    for (let i = 0; i < n; i++) out.push("");
    return out;
  }
  for (let i = 0; i < n; i++) {
    let s = adr[i];
    let e = s;
    while (e < buf.length && buf[e] !== 0) e++;
    out.push(_decoder.decode(buf.subarray(s, e)));
  }
  return out;
}

/**
 * Build a Python Runtime for the selected config. Returns a JS handle:
 *   { setKeys(keysArray, shift), step(), unsupported }
 * or throws on Python error.
 */
export function makeRuntime(entry) {
  const py = _pyodide;
  const params = py.toPy(entry.controller_params || {});
  const make = py.globals.get("web_runtime").make_runtime;
  const rt = make(entry.controller, entry.input_device, params, entry.fps || 100);
  params.destroy?.();

  const unsupported = rt.unsupported ? rt.unsupported.toString() : null;
  return {
    unsupported,
    setKeys(keysArray) {
      // Pass the JS array straight through; the Python side iterates it (Pyodide
      // exposes it as an iterable JsProxy).
      rt.set_keys(keysArray);
    },
    step() {
      rt.step();
    },
    destroy() {
      rt.destroy?.();
    },
  };
}
