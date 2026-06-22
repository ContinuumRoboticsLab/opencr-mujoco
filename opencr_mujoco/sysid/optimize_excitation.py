#!/usr/bin/env python3
"""Optimal excitation design for quasi-static TDCR system identification.

Quasi-static + friction-free  =>  each settled pose is order-independent, so the
best "trajectory" is the best SET of poses, and Fisher information is additive
over poses:  F = sum_i J_i^T J_i,  J_i = d(observed marker positions)/d(theta).

This tool:
  1. builds a feasible candidate pool of poses (per-segment magnitude+direction,
     tendon cos-pattern, capped at the +/-10 mm hardware limit),
  2. computes each pose's sensitivity to the 6 params (centered differences),
  3. greedily selects a D-optimal subset (max log det F)  ->  information-vs-N,
  4. reports the predicted recovery floor (Cramer-Rao sigma per param) for the
     optimized set vs the current train trajectory, under two observation models:
        tip_only      -> just the end-effector (your current setup)
        per_segment   -> end of each segment (link_20 / link_40 / link_60),
     so you can see how much better SENSING helps vs better EXCITATION,
  5. writes the selected poses as an executable tendon-command CSV + plots.

    ./.venv/bin/python -m opencr_mujoco.sysid.optimize_excitation
"""

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .sensitivity_analysis import build_optimizer, load_baseline_theta
from ..generators.unified_tdcr_generator import create_tdcr_from_config

TENDON_ANGLES = np.deg2rad([0.0, 120.0, 240.0])
NOISE_M = 0.001  # 1 mm i.i.d. marker noise (Cramer-Rao)
# Segment-end link indices for the v4 model (3 segments x 10 links, link_0 =
# clamped base half-link): link_10 / link_20 / link_30(tip).
SEG_END_LINKS = [10, 20, 30]
OBS_MODELS = {"tip_only": [2], "per_segment": [0, 1, 2]}  # indices into SEG_END_LINKS


def pose_to_tendons(pose, mag_max):
    """pose = [m0,phi0, m1,phi1, m2,phi2] (m in [0,1], phi rad) -> 9 tendon cmds (m)."""
    out = np.empty(9)
    for s in range(3):
        out[3 * s : 3 * s + 3] = (
            pose[2 * s] * mag_max * np.cos(TENDON_ANGLES - pose[2 * s + 1])
        )
    return out


def make_pool(mag_max, n_random, seed):
    """Structured single/two-segment families + random multi-segment fill."""
    rng = np.random.default_rng(seed)
    poses, desc = [], []
    dirs = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    mags = [0.35, 0.7, 1.0]
    # single-segment sweeps (isolate each segment's params)
    for s in range(3):
        for m in mags:
            for ph in dirs:
                p = np.zeros(6)
                p[2 * s] = m
                p[2 * s + 1] = ph
                poses.append(p)
                desc.append(f"single_seg{s+1}_m{m}_d{int(np.degrees(ph))}")
    # two-segment coordinated (incl. opposing S-shapes)
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        for m in (0.7, 1.0):
            for ph in dirs[::2]:
                for opp in (0.0, np.pi):
                    p = np.zeros(6)
                    p[2 * a] = m
                    p[2 * a + 1] = ph
                    p[2 * b] = m
                    p[2 * b + 1] = ph + opp
                    poses.append(p)
                    desc.append(
                        f"two_seg{a+1}{b+1}_m{m}_d{int(np.degrees(ph))}_opp{int(opp>0)}"
                    )
    # random three-segment
    for k in range(n_random):
        p = np.empty(6)
        p[0::2] = rng.uniform(0, 1, 3)
        p[1::2] = rng.uniform(0, 2 * np.pi, 3)
        poses.append(p)
        desc.append(f"rand_{k}")
    return np.array(poses), desc


class Forward:
    """Settles a tendon command from the keyframe and reads backbone marker xpos."""

    def __init__(self, opt, settling_time):
        self.opt = opt
        self.sim = opt.simulator
        self.settle_steps = None
        self.settling_time = settling_time
        self.xml = opt.output_dir / "_oed_model.xml"
        self.body_ids = None

    def load_theta(self, theta):
        gen = self.opt.parameter_registry.apply_to_config(
            theta, copy.deepcopy(self.opt.base_generation_config)
        )
        create_tdcr_from_config(gen, str(self.xml))
        self.sim.load_model(str(self.xml))
        m = self.sim.model
        self.settle_steps = int(self.settling_time / m.opt.timestep)
        self.body_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}")
            for i in SEG_END_LINKS
        ]
        missing = [i for i, b in zip(SEG_END_LINKS, self.body_ids) if b < 0]
        if missing:
            raise RuntimeError(
                f"link bodies {missing} not found in {self.xml}; adjust "
                "SEG_END_LINKS for this model"
            )

    def settle(self, cmd_m):
        """Reset, apply absolute ctrl, settle, return marker xpos (len(SEG_END_LINKS) x 3)."""
        m, d, sim = self.sim.model, self.sim.data, self.sim
        mujoco.mj_resetDataKeyframe(m, d, sim.pretension_keyframe_idx)
        mujoco.mj_forward(m, d)
        for j, aid in enumerate(sim.actuator_ids):
            d.ctrl[aid] = sim.pretension_baseline[j] - cmd_m[j]
        for _ in range(self.settle_steps):
            mujoco.mj_step(m, d)
        return np.array([d.xpos[b].copy() for b in self.body_ids])

    def sweep(self, theta, commands):
        """All-marker output for every command at this theta -> (P, 3*n_markers)."""
        self.load_theta(theta)
        return np.array([self.settle(c).reshape(-1) for c in commands])


def jacobian(fwd, theta, commands, frac):
    """Centered finite-diff J = d y / d ln(theta): (P, 3*n_markers, 6)."""
    P, Y = len(commands), 3 * len(SEG_END_LINKS)
    J = np.zeros((P, Y, len(theta)))
    for j in range(len(theta)):
        tp = theta.copy()
        tp[j] = theta[j] * (1 + frac)
        tm = theta.copy()
        tm[j] = theta[j] * (1 - frac)
        J[:, :, j] = (fwd.sweep(tp, commands) - fwd.sweep(tm, commands)) / (2 * frac)
    return J


def per_pose_info(J, marker_idx):
    """6x6 info matrix per pose for an observation model (subset of markers)."""
    cols = np.concatenate([3 * k + np.arange(3) for k in marker_idx])
    Jm = J[:, cols, :]  # (P, 3*len, 6)
    return np.einsum("poa,pob->pab", Jm, Jm)  # (P, 6, 6)


def worst_sigma_pct(F):
    """Cramer-Rao std-dev of the least-identifiable param (% of value, 1mm noise)."""
    s = NOISE_M * np.sqrt(np.clip(np.diag(np.linalg.pinv(F, rcond=1e-12)), 0, None))
    return float(np.max(s) * 100), (s * 100)


def greedy_dopt(M, n_max):
    """Greedily pick poses maximizing log det F; return order, logdet & floor curves."""
    P = M.shape[0]
    ridge = 1e-9 * np.median([np.trace(M[i]) for i in range(P)]) * np.eye(6)
    F = ridge.copy()
    chosen, avail = [], list(range(P))
    logdets, floors = [], []
    for _ in range(min(n_max, P)):
        best_i, best_v = -1, -np.inf
        for i in avail:
            v = np.linalg.slogdet(F + M[i])[1]
            if v > best_v:
                best_v, best_i = v, i
        chosen.append(best_i)
        avail.remove(best_i)
        F = F + M[best_i]
        logdets.append(float(np.linalg.slogdet(F)[1]))
        floors.append(worst_sigma_pct(F)[0])
    return chosen, logdets, floors


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-dir",
        required=True,
        help="pipeline run directory (sysid_results/<robot>/<run>)",
    )
    ap.add_argument("--mag-max-mm", type=float, default=10.0)
    ap.add_argument("--n-random", type=int, default=160)
    ap.add_argument("--n-select", type=int, default=50)
    ap.add_argument("--jac-frac", type=float, default=0.02)
    ap.add_argument("--train-stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = (
        Path(args.out) if args.out else run_dir / "sensitivity_analysis" / "excitation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mag_max = args.mag_max_mm / 1000.0

    print(f"{'='*72}\nOPTIMAL EXCITATION DESIGN (quasi-static, D-optimal)\n{'='*72}")
    opt, settling, work = build_optimizer(
        run_dir, run_dir / "step3_refinement" / "preprocessed_train.csv"
    )
    names = opt.parameter_registry.get_dimension_names()
    theta = load_baseline_theta(run_dir, "step3", names)
    fwd = Forward(opt, settling)

    # candidate pool (tendon commands, metres)
    poses, desc = make_pool(mag_max, args.n_random, args.seed)
    pool_cmds = np.array([pose_to_tendons(p, mag_max) for p in poses])
    # current train trajectory poses for a baseline comparison
    train_cmds = opt.actuator_commands[:: args.train_stride]
    print(
        f"pool: {len(pool_cmds)} candidate poses | train baseline: {len(train_cmds)} poses "
        f"(stride {args.train_stride}) | mag_max {args.mag_max_mm} mm"
    )

    print("computing sensitivities (pool) ...")
    Jp = jacobian(fwd, theta, pool_cmds, args.jac_frac)
    print("computing sensitivities (train baseline) ...")
    Jt = jacobian(fwd, theta, train_cmds, args.jac_frac)

    report = {
        "baseline_theta": dict(zip(names, theta.tolist())),
        "config": vars(args),
        "observation_models": {},
    }
    curves = {}
    for model, mk in OBS_MODELS.items():
        Mp = per_pose_info(Jp, mk)
        F_train = per_pose_info(Jt, mk).sum(0)
        chosen, logdets, floors = greedy_dopt(Mp, args.n_select)
        F_sel = Mp[chosen].sum(0)
        wf_sel, sig_sel = worst_sigma_pct(F_sel)
        wf_train, sig_train = worst_sigma_pct(F_train)
        # smallest N whose floor beats the full train trajectory
        n_match = next((i + 1 for i, f in enumerate(floors) if f <= wf_train), None)
        curves[model] = (logdets, floors, wf_train)
        report["observation_models"][model] = {
            "train_worst_sigma_pct": wf_train,
            "train_logdetF": float(np.linalg.slogdet(F_train)[1]),
            "oed_worst_sigma_pct": wf_sel,
            "oed_logdetF": float(np.linalg.slogdet(F_sel)[1]),
            "n_poses_to_beat_train": n_match,
            "train_sigma_pct": dict(zip(names, sig_train.round(3).tolist())),
            "oed_sigma_pct": dict(zip(names, sig_sel.round(3).tolist())),
            "selected_pose_idx": chosen,
        }
        print(f"\n[{model}]  worst-param recovery floor (1mm noise):")
        print(
            f"   current train traj : {wf_train:6.2f}%   (log det F {np.linalg.slogdet(F_train)[1]:.1f})"
        )
        print(
            f"   OED {args.n_select} poses    : {wf_sel:6.2f}%   (log det F {np.linalg.slogdet(F_sel)[1]:.1f})"
            f"   | {f'{n_match} OED poses already beat the full train traj' if n_match else ''}"
        )
        print(
            "   per-param σ%  "
            + "  ".join(
                f"{n.replace('tendon_','').replace('pretension','pre')}:{sig_sel[i]:.2f}"
                for i, n in enumerate(names)
            )
        )

    # ---- selected poses (per_segment model used for the deliverable set) ----
    best_model = "tip_only"  # deliverable for current hardware
    sel = report["observation_models"][best_model]["selected_pose_idx"]
    sel_cmds_mm = pool_cmds[sel] * 1000.0
    import csv

    csv_path = out_dir / "excitation_poses.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["pattern_label"]
            + [f"seg{j//3+1}_ten{j%3+1}_mm" for j in range(9)]
            + [f"{nm}_desc" for nm in ["pose"]]
        )
        for r, i in enumerate(sel):
            w.writerow([f"oed_{r:03d}"] + sel_cmds_mm[r].round(3).tolist() + [desc[i]])
    print(f"\nExecutable poses ({best_model}, N={len(sel)}): {csv_path}")
    print(
        "  (tendon order seg-major; apply your step1 servo_mapping to map to physical servos)"
    )

    # ---- plots ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for model, (ld, fl, wf_tr) in curves.items():
        a1.plot(range(1, len(ld) + 1), ld, "o-", ms=3, label=model)
        a2.semilogy(range(1, len(fl) + 1), fl, "o-", ms=3, label=f"OED {model}")
        a2.axhline(
            wf_tr,
            ls="--",
            alpha=0.6,
            color=a2.lines[-1].get_color(),
            label=f"train {model}",
        )
    a1.set_xlabel("# poses (N)")
    a1.set_ylabel("log det F")
    a1.set_title("Information vs N")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.set_xlabel("# poses (N)")
    a2.set_ylabel("worst-param σ (%) @1mm")
    a2.set_title("Predicted recovery floor vs N")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "excitation_info_curves.png", dpi=150)
    plt.close(fig)

    # per-param floor: train vs OED, both observation models
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    w = 0.2
    series = [
        ("train tip", "tip_only", "train_sigma_pct"),
        ("OED tip", "tip_only", "oed_sigma_pct"),
        ("train per-seg", "per_segment", "train_sigma_pct"),
        ("OED per-seg", "per_segment", "oed_sigma_pct"),
    ]
    for k, (lab, model, key) in enumerate(series):
        vals = [report["observation_models"][model][key][n] for n in names]
        ax.bar(x + (k - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("pretension_", "pre_").replace("tendon_kp_", "kp_") for n in names],
        rotation=20,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Cramér-Rao σ (% of param) @1mm")
    ax.set_title(
        "Recovery floor: trajectory lever (train→OED) vs sensing lever (tip→per-seg)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "excitation_floor_comparison.png", dpi=150)
    plt.close(fig)

    json.dump(report, open(out_dir / "excitation_report.json", "w"), indent=2)
    print(
        f"\nSaved: {out_dir}/excitation_{{info_curves,floor_comparison}}.png, excitation_report.json"
    )
    opt.cleanup()
    import shutil

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
