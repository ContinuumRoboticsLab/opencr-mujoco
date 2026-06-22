"""In-browser ``mujoco`` shim bridging the real Python controllers to mujoco-js.

The compiled ``mujoco`` Python package can't run under Pyodide, so this module
shadows it and forwards model/data access + the handful of ``mj_*`` calls the
controllers use to the JavaScript ``mujoco-js`` engine. The JS side publishes a
context object on ``globalThis.MJ_CTX`` (see js/pyodide-bridge.js):

    globalThis.MJ_CTX = {
        mujoco,            // the mujoco-js module (mj_step, mj_forward, MjData, ...)
        model,             // live JS MjModel
        data,              // live JS MjData
        names: {           // precomputed id<->name tables per object type
            actuator: [...], body: [...], site: [...], joint: [...], key: [...]
        }
    }

Only the API surface exercised by opencr_mujoco/controllers + opencr_mujoco/tdcr_kinematics is
implemented. mj_jacBody / mj_name2id / mju_quat2Mat are emulated (mujoco-js does
not export them); everything else delegates to the engine.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from js import globalThis


# ---------------------------------------------------------------------------
# Bridge access helpers
# ---------------------------------------------------------------------------


def _ctx():
    return globalThis.MJ_CTX


def context():
    """Public accessor for the JS bridge context (avoids dunder name-mangling)."""
    return globalThis.MJ_CTX


def _engine():
    return globalThis.MJ_CTX.mujoco


def _np_copy(js_arr):
    return np.array(js_arr.to_py(), dtype=np.float64)


class _WTArray(np.ndarray):
    """A numpy array that mirrors a mujoco-js TypedArray and writes changes back.

    Pyodide and mujoco-js are separate WASM modules with separate heaps, so a
    plain ``to_py()`` is a *copy* -- writes wouldn't reach the engine. This holds
    a reference to the source JS typed array and pushes the whole array back via
    ``TypedArray.set`` whenever an element/slice is assigned, so the controllers'
    ``data.ctrl[i] = v`` / ``data.ctrl[:] = arr`` propagate to MuJoCo.
    """

    def __new__(cls, js_arr):
        base = np.array(js_arr.to_py(), dtype=np.float64)
        obj = base.view(cls)
        obj._js = js_arr
        obj._n = base.shape[0]
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._js = getattr(obj, "_js", None)
        self._n = getattr(obj, "_n", None)

    def _push(self):
        js = getattr(self, "_js", None)
        if js is None or self.ndim != 1 or self.shape[0] != self._n:
            return
        vals = self.tolist()
        try:
            js.set(vals)  # TypedArray.set(array) -- one shot
        except Exception:  # noqa: BLE001 - fall back to element-wise
            for i in range(self._n):
                js[i] = float(vals[i])

    def __setitem__(self, idx, val):
        np.ndarray.__setitem__(self, idx, val)
        self._push()


def _np_view(js_arr):
    """Writable, write-through mirror of a JS TypedArray (see _WTArray)."""
    return _WTArray(js_arr)


# ---------------------------------------------------------------------------
# mjtObj enum (only the object types the controllers reference)
# ---------------------------------------------------------------------------


class _MjtObj:
    mjOBJ_BODY = "body"
    mjOBJ_JOINT = "joint"
    mjOBJ_SITE = "site"
    mjOBJ_ACTUATOR = "actuator"
    mjOBJ_KEY = "key"
    mjOBJ_GEOM = "geom"


mjtObj = _MjtObj()


# ---------------------------------------------------------------------------
# Name <-> id (mj_name2id is not bound; use the JS-precomputed tables)
# ---------------------------------------------------------------------------


def _name_table(objtype):
    names = _ctx().names
    table = getattr(names, objtype, None)
    if table is None:
        return []
    return list(table)  # JS array -> python list of strings


def mj_name2id(model, objtype, name):
    table = _name_table(objtype)
    try:
        return table.index(name)
    except ValueError:
        return -1


def mj_id2name(model, objtype, idx):
    table = _name_table(objtype)
    if 0 <= idx < len(table):
        return table[idx]
    return None


# ---------------------------------------------------------------------------
# Model / Data wrappers
# ---------------------------------------------------------------------------


class _Opt:
    def __init__(self, js_model):
        self._js = js_model

    @property
    def timestep(self):
        return float(self._js.opt.timestep)

    @property
    def gravity(self):
        return _np_view(self._js.opt.gravity)


class MjModel:
    """Wraps a JS MjModel, exposing the attributes the controllers read."""

    def __init__(self, js_model):
        self._js = js_model

    # --- counts ---
    @property
    def nu(self):
        return int(self._js.nu)

    @property
    def nq(self):
        return int(self._js.nq)

    @property
    def nv(self):
        return int(self._js.nv)

    @property
    def nbody(self):
        return int(self._js.nbody)

    @property
    def nsite(self):
        return int(self._js.nsite)

    @property
    def njnt(self):
        return int(self._js.njnt)

    @property
    def nkey(self):
        return int(self._js.nkey)

    @property
    def ngeom(self):
        return int(self._js.ngeom)

    @property
    def opt(self):
        return _Opt(self._js)

    # --- keyframe data (2D: nkey x n) ---
    @property
    def key_ctrl(self):
        if self.nkey == 0:
            return np.zeros((0, self.nu))
        return _np_copy(self._js.key_ctrl).reshape(self.nkey, self.nu)

    @property
    def key_qpos(self):
        if self.nkey == 0:
            return np.zeros((0, self.nq))
        return _np_copy(self._js.key_qpos).reshape(self.nkey, self.nq)

    @property
    def actuator_ctrlrange(self):
        return _np_copy(self._js.actuator_ctrlrange).reshape(self.nu, 2)

    # --- addressing used by the finite-difference Jacobian ---
    @property
    def dof_bodyid(self):
        return _np_copy(self._js.dof_bodyid).astype(int)

    @property
    def body_parentid(self):
        return _np_copy(self._js.body_parentid).astype(int)

    @property
    def jnt_dofadr(self):
        return _np_copy(self._js.jnt_dofadr).astype(int)

    @property
    def jnt_qposadr(self):
        return _np_copy(self._js.jnt_qposadr).astype(int)


class _NamedBody:
    def __init__(self, data, body_id):
        self._data = data
        self.id = body_id

    @property
    def xpos(self):
        i = self.id
        return self._data.xpos[i * 3 : i * 3 + 3]

    @property
    def xquat(self):
        i = self.id
        return self._data.xquat[i * 4 : i * 4 + 4]


class MjData:
    """Wraps a JS MjData. Constructing ``MjData(model)`` allocates a fresh
    (scratch) JS state on the model -- used by the IK controllers for
    finite-difference rollouts."""

    def __init__(self, model):
        self._model = model
        self._js = _engine().MjData.new(model._js)

    @classmethod
    def _wrap(cls, model, js_data):
        obj = cls.__new__(cls)
        obj._model = model
        obj._js = js_data
        return obj

    # State arrays are re-fetched per access so views never point at a stale
    # (detached) heap buffer.
    @property
    def ctrl(self):
        return _np_view(self._js.ctrl)

    @property
    def qpos(self):
        return _np_view(self._js.qpos)

    @property
    def qvel(self):
        return _np_view(self._js.qvel)

    @property
    def xpos(self):
        return _np_copy(self._js.xpos)

    @property
    def xquat(self):
        return _np_copy(self._js.xquat)

    @property
    def site_xpos(self):
        return _np_copy(self._js.site_xpos)

    @property
    def site_xmat(self):
        return _np_copy(self._js.site_xmat)

    def body(self, name):
        # Real mujoco's data.body() accepts an int id OR a name string; match
        # that. The Franka IK looks bodies up by id (data.body(self.body_id)),
        # so an id-only path is required — without it that call raised
        # KeyError, panda_ik threw, and the combined demo's Franka (WASD)
        # controls silently died while the TDCR (TFGH) kept working.
        if isinstance(name, (int, np.integer)):
            return _NamedBody(self, int(name))
        bid = mj_name2id(self._model, mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(name)
        return _NamedBody(self, bid)


# ---------------------------------------------------------------------------
# Simulation functions (delegate to the engine)
# ---------------------------------------------------------------------------


def mj_forward(model, data):
    _engine().mj_forward(model._js, data._js)


def mj_step(model, data):
    _engine().mj_step(model._js, data._js)


def mj_kinematics(model, data):
    _engine().mj_kinematics(model._js, data._js)


def mj_resetData(model, data):
    _engine().mj_resetData(model._js, data._js)


def mj_resetDataKeyframe(model, data, key):
    _engine().mj_resetDataKeyframe(model._js, data._js, key)


# ---------------------------------------------------------------------------
# Emulated helpers (not bound by mujoco-js)
# ---------------------------------------------------------------------------


def _quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def mju_quat2Mat(res, quat):
    """Fill ``res`` (len-9) with the rotation matrix for ``quat`` ([w,x,y,z])."""
    R = Rotation.from_quat(_quat_wxyz_to_xyzw(np.asarray(quat))).as_matrix()
    res[:] = R.reshape(9)


def mj_jacBody(model, data, jacp, jacr, body_id):
    """Finite-difference body Jacobian (mujoco-js does not export mj_jacBody).

    Fills ``jacp`` (3xnv translational) and ``jacr`` (3xnv rotational). Only dofs
    whose joint body is an ancestor of ``body_id`` are perturbed (others are
    zero), which keeps this fast and exact for serial chains. Relies on nq==nv
    (all 1-dof joints) so qpos index i corresponds to dof i -- true for every
    scene in this project.
    """
    nv = model.nv
    jacp[:] = 0.0
    if jacr is not None:
        jacr[:] = 0.0

    # Ancestor body set of the target.
    parent = model.body_parentid
    ancestors = set()
    b = int(body_id)
    while b > 0:
        ancestors.add(b)
        b = int(parent[b])
    if body_id == 0:
        return

    dof_body = model.dof_bodyid

    # Central differences give a much cleaner Jacobian than forward differences
    # (no first-order bias / noise), which matters here because the controllers
    # feed it straight into a plain pseudo-inverse with no damping.
    eps = 1e-6
    q = data.qpos  # write-through mirror

    def _pose():
        p = np.array(data.xpos[body_id * 3 : body_id * 3 + 3])
        quat = np.array(data.xquat[body_id * 4 : body_id * 4 + 4])
        return p, Rotation.from_quat(_quat_wxyz_to_xyzw(quat))

    for i in range(nv):
        if int(dof_body[i]) not in ancestors:
            continue
        old = float(q[i])
        q[i] = old + eps
        mj_kinematics(model, data)
        pos_p, rot_p = _pose()
        q[i] = old - eps
        mj_kinematics(model, data)
        pos_m, rot_m = _pose()
        q[i] = old

        jacp[:, i] = (pos_p - pos_m) / (2.0 * eps)
        if jacr is not None:
            dR = rot_p * rot_m.inv()
            jacr[:, i] = dR.as_rotvec() / (2.0 * eps)

    mj_kinematics(model, data)  # restore consistent kinematics for qpos0
