#!/usr/bin/env python3
"""Parameter sensitivity & identifiability analysis for a TDCR sysid result.

Given an identified parameter set (theta*) and the trajectory it was identified
on, this answers two coupled questions:

  (1) ROBUSTNESS / model uncertainty  -- if theta* is off by a few percent, how
      much does the predicted tip trajectory move, and how much does the fit to
      the real data degrade?  (one-at-a-time perturbation sweep)

  (2) IDENTIFIABILITY / trajectory informativeness -- which parameters does this
      trajectory actually constrain, and which are confounded?  Built from the
      output sensitivity matrix  S = d y / d theta  along the trajectory:
          Fisher information  F = S^T S   (assuming i.i.d. marker noise)
          parameter covariance ~ F^-1     (Cramer-Rao lower bound)
      Large, well-conditioned S  ->  tight, separable estimates  ->  a "good"
      sysid trajectory.  Near-collinear columns of S  ->  confounded parameters.

Both come from the same forward-rollout machinery used by the sysid pipeline
(apply params -> regenerate MJCF -> TrajectorySimulator.simulate_trajectory).

Usage (run as a module from the repo root):
    ./.venv/bin/python -m opencr_mujoco.sysid.sensitivity_analysis \
        --run-dir sysid_results/ftdcr_v4/ftdcr_v4_pipeline_20260604_004025

    ./.venv/bin/python -m opencr_mujoco.sysid.sensitivity_analysis --quick   # coarse + fast (~3 min)
"""

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sysid_optimizer import SystemIdentificationOptimizer
from ..generators.unified_tdcr_generator import create_tdcr_from_config


# --------------------------------------------------------------------------- #
# Forward rollout helpers
# --------------------------------------------------------------------------- #
class Roller:
    """Wraps the sysid forward model so a parameter vector -> tip trajectory."""

    def __init__(
        self,
        optimizer: SystemIdentificationOptimizer,
        work_dir: Path,
        settling_time: float,
    ):
        self.opt = optimizer
        self.reg = optimizer.parameter_registry
        self.base_config = optimizer.base_generation_config
        self.sim = optimizer.simulator
        self.settling_time = settling_time
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._xml = work_dir / "_perturbed_model.xml"
        self.n_rolls = 0
        self.roll_time = 0.0

    def rollout(
        self, theta: np.ndarray, cmds: np.ndarray, ts: np.ndarray
    ) -> np.ndarray:
        """Apply theta, regenerate model, simulate -> marker positions (N x 3, m)."""
        gen = self.reg.apply_to_config(theta, copy.deepcopy(self.base_config))
        create_tdcr_from_config(gen, str(self._xml))
        self.sim.load_model(str(self._xml))
        t0 = time.time()
        _, marker = self.sim.simulate_trajectory(cmds, ts, self.settling_time)
        self.roll_time += time.time() - t0
        self.n_rolls += 1
        return marker


def aligned(sim: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Remove the constant centroid offset, exactly as objective_function does."""
    return sim - np.mean(sim - real, axis=0)


def fit_rmse_mm(sim: np.ndarray, real: np.ndarray) -> float:
    """Standard (mean-offset-aligned, pointwise) RMSE in mm — the same
    convention as the pipeline's reported numbers, on the strided data."""
    s = aligned(sim, real)
    return float(np.sqrt(np.mean(np.linalg.norm(s - real, axis=1) ** 2)) * 1000.0)


def pred_shift_mm(sim: np.ndarray, base: np.ndarray) -> float:
    """RMS tip displacement vs the baseline prediction, in mm (model-vs-model)."""
    return float(np.sqrt(np.mean(np.linalg.norm(sim - base, axis=1) ** 2)) * 1000.0)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def build_optimizer(run_dir: Path, train_csv: Path):
    """Reconstruct step3's exact optimizer setup (registry, simulator, metrics)."""
    cfg = json.load(open(run_dir / "step3_refinement" / "config.json"))
    # Pin paths to absolutes so the tool works from any CWD.
    cfg["base_generation_config"] = str(
        (run_dir / "step3_refinement" / "input_generation_config.json").resolve()
    )
    work = run_dir / "sensitivity_analysis" / "_work"
    opt = SystemIdentificationOptimizer(cfg, enable_viewer=False, output_dir=work)
    opt.load_data(str(train_csv))
    settling = cfg.get("simulation", {}).get("settling_time", 0.5)
    return opt, settling, work


def load_baseline_theta(run_dir: Path, source: str, dim_names):
    """Load identified theta* from the chosen artifact."""
    if source == "step3":
        res = json.load(open(run_dir / "step3_refinement" / "results.json"))
        theta = np.array(res["best_params"], dtype=float)
        assert res["parameter_names"] == dim_names, "parameter order mismatch"
    elif source == "config":
        # Pull per-segment values straight out of the final generation config.
        gc = json.load(open(run_dir / "final_generation_config.json"))
        ap = gc["actuator_properties"]
        pre = ap["tendon_pretension"]
        kp = ap["tendon_kp_array"]
        theta = np.array([pre[0], pre[3], pre[6], kp[0], kp[3], kp[6]], dtype=float)
    else:
        raise ValueError(source)
    return theta


# --------------------------------------------------------------------------- #
# (1) Robustness sweep
# --------------------------------------------------------------------------- #
def robustness_sweep(roller, theta, names, cmds, ts, real, base_pred, levels, jac_frac):
    """One-at-a-time +/- perturbation of each parameter (relative steps)."""
    base_fit = fit_rmse_mm(base_pred, real)
    fracs = sorted(set(levels) | {jac_frac})
    out = {}
    print(f"\n[robustness] baseline RMSE (strided) = {base_fit:.2f} mm")
    print(
        f"{'param':<16}{'+/-%':>6}{'pred shift (mm)':>18}{'fit (mm)':>12}"
        f"{'d fit (mm)':>12}"
    )
    # cache rollouts keyed by (param index, signed fraction) for reuse by Fisher
    cache = {}
    for j, name in enumerate(names):
        per_level = {}
        for f in fracs:
            shifts, fits = [], []
            for sgn in (+1.0, -1.0):
                th = theta.copy()
                th[j] = theta[j] * (1.0 + sgn * f)
                pred = roller.rollout(th, cmds, ts)
                cache[(j, sgn * f)] = pred
                shifts.append(pred_shift_mm(pred, base_pred))
                fits.append(fit_rmse_mm(pred, real))
            ps = float(np.mean(shifts))  # symmetric average
            ft = float(np.mean(fits))
            per_level[f] = {
                "pred_shift_mm": ps,
                "fit_mm": ft,
                "fit_delta_mm": ft - base_fit,
            }
            if f in levels:
                print(
                    f"{name:<16}{f*100:>5.0f}%{ps:>18.2f}{ft:>12.2f}"
                    f"{ft - base_fit:>+12.2f}"
                )
        # normalized sensitivity: mm tip shift per 1% relative change (from jac step)
        per_level["norm_sens_mm_per_pct"] = per_level[jac_frac]["pred_shift_mm"] / (
            jac_frac * 100.0
        )
        out[name] = per_level
    return out, base_fit, cache


# --------------------------------------------------------------------------- #
# (2) Identifiability via the sensitivity / Fisher matrix
# --------------------------------------------------------------------------- #
def sensitivity_matrix(roller, theta, names, cmds, ts, jac_frac, cache=None):
    """Centered finite-difference Jacobian S = d y / d ln(theta), columns per param.

    Returns raw S and the centroid-aligned S' (per-coordinate temporal mean
    removed -- what the offset-aligned metric actually 'sees').  Units: metres
    per unit fractional (~100%) change in the parameter.
    """
    cache = cache or {}
    n = len(cmds)
    P = len(names)
    S = np.zeros((3 * n, P))
    for j in range(P):
        yp = cache.get((j, +jac_frac))
        ym = cache.get((j, -jac_frac))
        if yp is None:
            th = theta.copy()
            th[j] = theta[j] * (1 + jac_frac)
            yp = roller.rollout(th, cmds, ts)
        if ym is None:
            th = theta.copy()
            th[j] = theta[j] * (1 - jac_frac)
            ym = roller.rollout(th, cmds, ts)
        S[:, j] = ((yp - ym) / (2.0 * jac_frac)).reshape(-1)
    # aligned: remove per-coordinate temporal mean (models centroid alignment)
    S_aligned = S.copy()
    blk = S_aligned.reshape(n, 3, P)
    blk -= blk.mean(axis=0, keepdims=True)
    S_aligned = blk.reshape(3 * n, P)
    return S, S_aligned


def fisher_report(S, S_aligned, names):
    """Eigen-spectrum, conditioning, parameter correlations, identifiability."""
    F = S_aligned.T @ S_aligned
    F_raw = S.T @ S
    eigval, eigvec = np.linalg.eigh(F)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    eigval = np.clip(eigval, 0, None)

    cond = float(eigval[0] / eigval[-1]) if eigval[-1] > 0 else float("inf")
    eig_raw = np.clip(np.linalg.eigvalsh(F_raw), 0, None)
    cond_raw = (
        float(eig_raw.max() / eig_raw.min()) if eig_raw.min() > 0 else float("inf")
    )

    # covariance ~ pinv(F); correlation matrix
    cov = np.linalg.pinv(F, rcond=1e-12)
    d = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    corr = cov / np.outer(d, d)
    np.clip(corr, -1, 1, out=corr)

    # per-param sensitivity (aligned column RMS, mm per 1% change)
    npts = S.shape[0] // 3
    col_rms_aligned = np.sqrt((S_aligned**2).sum(axis=0) / npts) * 1000.0 / 100.0
    col_rms_raw = np.sqrt((S**2).sum(axis=0) / npts) * 1000.0 / 100.0

    least = eigvec[:, -1]  # eigenvector of smallest eigenvalue = sloppiest combo
    with np.errstate(divide="ignore"):
        logdet = float(np.sum(np.log(eigval[eigval > 0])))

    return (
        {
            "eigenvalues": eigval.tolist(),
            "condition_number_aligned": cond,
            "condition_number_raw": cond_raw,
            "log_det_F": logdet,
            "lambda_min": float(eigval[-1]),
            "correlation_matrix": corr.tolist(),
            "per_param_sens_mm_per_pct_aligned": dict(
                zip(names, col_rms_aligned.tolist())
            ),
            "per_param_sens_mm_per_pct_raw": dict(zip(names, col_rms_raw.tolist())),
            "sloppiest_direction": dict(zip(names, least.tolist())),
            "sloppiest_eigenvalue": float(eigval[-1]),
        },
        corr,
        eigval,
    )


def print_fisher(tag, rep, names):
    print(f"\n[identifiability:{tag}]")
    print(
        f"  condition number  aligned={rep['condition_number_aligned']:.1f}"
        f"  (raw={rep['condition_number_raw']:.1f})   "
        f"log det F={rep['log_det_F']:.2f}   lambda_min={rep['lambda_min']:.3e}"
    )
    print("  per-param sensitivity (mm tip shift per 1% change):")
    for nm in names:
        a = rep["per_param_sens_mm_per_pct_aligned"][nm]
        r = rep["per_param_sens_mm_per_pct_raw"][nm]
        print(
            f"    {nm:<16} aligned={a:7.3f}   raw={r:7.3f}"
            f"   ({'OFFSET-DOMINATED' if r > 3 * max(a, 1e-9) else ''})"
        )
    corr = np.array(rep["correlation_matrix"])
    print("  strongly correlated (|r|>0.9) parameter pairs:")
    found = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) > 0.9:
                print(f"    {names[i]} <-> {names[j]}:  r = {corr[i, j]:+.3f}")
                found = True
    if not found:
        print("    (none -- parameters are well separated by this trajectory)")
    sl = rep["sloppiest_direction"]
    combo = "  ".join(f"{v:+.2f}*{n}" for n, v in sl.items() if abs(v) > 0.25)
    print(f"  sloppiest (least identifiable) direction: {combo}")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(
    out_dir,
    names,
    robustness,
    fisher_train,
    corr_train,
    eig_train,
    fisher_val,
    eig_val,
    levels,
):
    short = [
        n.replace("pretension_", "pre_").replace("tendon_kp_", "kp_") for n in names
    ]
    x = np.arange(len(names))

    # 1. robustness: predicted tip shift per param at each level
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.8 / len(levels)
    for k, f in enumerate(sorted(levels)):
        vals = [robustness[n][f]["pred_shift_mm"] for n in names]
        ax.bar(x + k * w, vals, w, label=f"±{f*100:.0f}%")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(short, rotation=20)
    ax.set_ylabel("predicted tip shift (mm, RMS)")
    ax.set_title("Robustness: prediction sensitivity to parameter error")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_sweep.png", dpi=150)
    plt.close(fig)

    # 2. correlation heatmap (train)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr_train, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(x)
    ax.set_yticklabels(short)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{corr_train[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )
    ax.set_title("Parameter correlation (train)\n|r|→1 ⇒ confounded")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "parameter_correlation.png", dpi=150)
    plt.close(fig)

    # 3. eigenvalue spectra train vs val
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(eig_train) + 1), eig_train, "o-", label="train")
    if eig_val is not None:
        ax.semilogy(range(1, len(eig_val) + 1), eig_val, "s--", label="val")
    ax.set_xlabel("mode (stiff → sloppy)")
    ax.set_ylabel("Fisher eigenvalue")
    ax.set_title("Information spectrum (flatter = better conditioned)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fisher_spectrum.png", dpi=150)
    plt.close(fig)

    # 4. aligned vs raw per-param sensitivity (offset-masking diagnostic)
    fig, ax = plt.subplots(figsize=(9, 5))
    a = [fisher_train["per_param_sens_mm_per_pct_aligned"][n] for n in names]
    r = [fisher_train["per_param_sens_mm_per_pct_raw"][n] for n in names]
    ax.bar(x - 0.2, r, 0.4, label="raw output")
    ax.bar(x + 0.2, a, 0.4, label="centroid-aligned (what metric sees)")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=20)
    ax.set_ylabel("sensitivity (mm per 1% change)")
    ax.set_title(
        "Raw vs aligned sensitivity\n(big drop ⇒ effect is mostly a rigid shift,"
        " hidden by offset alignment)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "raw_vs_aligned_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"\nPlots saved to {out_dir}/")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-dir",
        required=True,
        help="pipeline run directory (sysid_results/<robot>/<run>)",
    )
    ap.add_argument("--baseline", choices=["step3", "config"], default="step3")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--jac-frac", type=float, default=0.02)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument(
        "--no-baseline-check",
        action="store_true",
        help="skip the slow full-resolution baseline validation",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="coarse + fast: stride 10, single ±10%% level, no baseline check",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        args.stride, args.levels, args.no_baseline_check = 10, [0.10], True

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "sensitivity_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = run_dir / "step3_refinement" / "preprocessed_train.csv"
    val_csv = run_dir / "step3_refinement" / "preprocessed_val.csv"

    print(f"{'='*68}\nSENSITIVITY & IDENTIFIABILITY ANALYSIS\n{'='*68}")
    print(f"run: {run_dir}")

    opt, settling, work = build_optimizer(run_dir, train_csv)
    names = opt.parameter_registry.get_dimension_names()
    theta = load_baseline_theta(run_dir, args.baseline, names)
    print(
        f"baseline θ ({args.baseline}): "
        f"{dict(zip(names, np.round(theta, 3).tolist()))}"
    )
    roller = Roller(opt, work, settling)

    # --- data (full + strided) ---
    cmds_f, ts_f, real_f = (
        opt.actuator_commands,
        opt.timestamps,
        opt.real_marker_positions,
    )
    idx = np.arange(0, len(cmds_f), args.stride)
    cmds, ts, real = cmds_f[idx], ts_f[idx], real_f[idx]
    print(f"train: {len(cmds_f)} samples → {len(cmds)} after stride {args.stride}")

    # --- baseline validation against step3's reported error (full res, exact metric) ---
    reported = None
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            reported = json.load(f).get("step3", {}).get("train_error_mm")
    baseline = {
        "source": args.baseline,
        "theta": dict(zip(names, theta.tolist())),
        "step3_reported_train_error_mm": reported,
    }
    if not args.no_baseline_check:
        print("\n[baseline check] full-res rollout with the exact step-3 metric ...")
        pred_full = roller.rollout(theta, cmds_f, ts_f)
        err = opt.metrics.compute(real_f, aligned(pred_full, real_f)) * 1000.0
        baseline["train_error_mm_full"] = float(err)
        ref = f"(step3 reported {reported:.2f} mm)" if reported is not None else ""
        print(f"  reproduced train error = {err:.2f} mm {ref}")

    # --- baseline strided prediction ---
    base_pred = roller.rollout(theta, cmds, ts)
    baseline["train_rmse_mm_strided"] = fit_rmse_mm(base_pred, real)
    eta = roller.roll_time / roller.n_rolls
    n_planned = 6 * (2 * len(set(args.levels) | {args.jac_frac})) + (
        0 if args.no_val else 12
    )
    print(
        f"  ~{eta:.1f}s / strided rollout → est. ~{eta * n_planned / 60:.1f} min remaining"
    )

    # --- (1) robustness ---
    robustness, base_fit, cache = robustness_sweep(
        roller, theta, names, cmds, ts, real, base_pred, args.levels, args.jac_frac
    )

    # --- (2) identifiability: train ---
    S, S_al = sensitivity_matrix(roller, theta, names, cmds, ts, args.jac_frac, cache)
    fisher_train, corr_train, eig_train = fisher_report(S, S_al, names)
    print_fisher("train", fisher_train, names)

    # --- identifiability: val (trajectory-informativeness comparison) ---
    fisher_val = corr_val = eig_val = None
    if not args.no_val and val_csv.exists():
        opt.load_data(str(val_csv))  # resets pattern labels for the val split
        cmds_v = opt.actuator_commands[:: args.stride]
        ts_v = opt.timestamps[:: args.stride]
        Sv, Sv_al = sensitivity_matrix(
            roller, theta, names, cmds_v, ts_v, args.jac_frac
        )
        fisher_val, corr_val, eig_val = fisher_report(Sv, Sv_al, names)
        print_fisher("val", fisher_val, names)
        print(
            f"\n[trajectory comparison]  log det F: "
            f"train={fisher_train['log_det_F']:.2f}  val={fisher_val['log_det_F']:.2f}"
            f"   (higher ⇒ more informative)"
        )

    # --- plots + report ---
    make_plots(
        out_dir,
        names,
        robustness,
        fisher_train,
        corr_train,
        eig_train,
        fisher_val,
        eig_val,
        args.levels,
    )
    report = {
        "baseline": baseline,
        "config": {
            "stride": args.stride,
            "jac_frac": args.jac_frac,
            "levels": args.levels,
            "settling_time": settling,
            "n_train_strided": len(cmds),
            "scaling": "relative (d y / d ln theta)",
        },
        "robustness": robustness,
        "identifiability": {"train": fisher_train, "val": fisher_val},
        "total_rollouts": roller.n_rolls,
    }
    with open(out_dir / "sensitivity_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"\nReport: {out_dir/'sensitivity_report.json'}  ({roller.n_rolls} rollouts, "
        f"{roller.roll_time:.0f}s sim)"
    )
    opt.cleanup()
    import shutil

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
