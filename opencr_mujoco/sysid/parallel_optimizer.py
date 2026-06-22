"""Parallel multi-start local optimization for system identification.

The MuJoCo simulator is not picklable, so a parallel optimizer cannot share one
objective across processes. Instead each worker rebuilds its OWN optimizer from
the (picklable) config — the per-process-model pattern. N Powell searches from
diverse Sobol starts run concurrently on separate cores; the best is returned.

This is the parallel/multi-start alternative to sequential Bayesian optimization:
- parallelism: ~one Powell's wall-clock for N searches (saturates all cores)
- robustness: multi-start escapes the flat-valley "landing lottery" that makes a
  single Bayesian run scatter
- speed: searching a strided objective (rmse is stride-robust) keeps each eval cheap

Progress display: by default the workers run quietly and the parent shows a
single live status line (evals done, best-so-far, elapsed) plus one summary
line per completed start. Pass verbose=True (sysid_pipeline.py --debug) for the
raw per-iteration worker logs instead.

Exposed via SystemIdentificationOptimizer.optimize() when the config sets
optimization.algorithm = "parallel_multistart".
"""

import os
import sys
import time
import warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_W = {}  # per-process state populated by the Pool initializer


def _winit(config, search_stride, workdir, verbose=False, progress=None, lock=None):
    """Rebuild a fresh optimizer (own MuJoCo model) for this worker process."""
    from .sysid_optimizer import SystemIdentificationOptimizer

    opt = SystemIdentificationOptimizer(
        config,
        enable_viewer=False,
        output_dir=Path(workdir) / f"pid_{os.getpid()}",
        verbose=verbose,
    )
    opt.load_data(config["data"]["trajectory_file"])

    if search_stride and search_stride > 1:
        sl = slice(None, None, search_stride)
        opt.actuator_commands = opt.actuator_commands[sl]
        opt.timestamps = opt.timestamps[sl]
        opt.real_marker_positions = opt.real_marker_positions[sl]

    _W["opt"] = opt
    _W["bounds"] = opt.parameter_registry.get_bounds()
    _W["progress"] = progress
    _W["lock"] = lock


def _wrun(task):
    """Run one Powell search from a start; return the converged params + error."""
    idx, x0, maxfev = task
    opt, bounds = _W["opt"], _W["bounds"]
    progress, lock = _W.get("progress"), _W.get("lock")
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]

    def objective(theta):
        err = opt.objective_function(np.asarray(theta))
        if progress is not None:  # shared live-progress counters (quiet mode)
            with lock:
                progress["evals"] += 1
                if err < progress["best"]:
                    progress["best"] = err
        return err

    try:
        r = minimize(
            objective,
            np.asarray(x0, float),
            method="Powell",
            bounds=bounds,
            options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4},
        )
        xhat = np.clip(r.x, lo, hi).tolist()
        err = float(objective(np.asarray(xhat)))
    except Exception as e:  # a single bad start must not kill the run
        return {"idx": idx, "theta": list(x0), "err": 1e6, "exc": str(e)}
    return {"idx": idx, "theta": xhat, "err": err, "nfev": int(r.nfev) + 1}


def _fmt_mm(err_m):
    """Format an error in meters as mm; '--' for the not-yet/failed sentinels."""
    if not np.isfinite(err_m) or err_m >= 1e5:
        return "--"
    return f"{err_m * 1000:.3f} mm"


def _fmt_elapsed(seconds):
    """mm:ss, or h:mm:ss once past an hour."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60:02d}:{s % 60:02d}"
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class _StatusLine:
    """Live single-line progress on a TTY; periodic plain lines otherwise.

    update() repaints one carriage-return line in place (so a long search shows
    a ticking counter instead of scrolling); println() emits a permanent line,
    clearing the live line first so the two never collide.
    """

    def __init__(self, stream=None, tty_period=0.5, plain_period=30.0):
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.period = tty_period if self.tty else plain_period
        # None = "never printed -> print immediately". (A 0.0 sentinel only
        # works when time.monotonic() is large; on a freshly booted machine
        # monotonic() starts near zero and would suppress the first line.)
        self._last = None
        self._live = False

    def update(self, text):
        now = time.monotonic()
        if self._last is not None and now - self._last < self.period:
            return
        self._last = now
        if self.tty:
            self.stream.write("\r\x1b[2K" + text)
            self._live = True
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    def println(self, text):
        self.clear()
        self.stream.write(text + "\n")
        self.stream.flush()
        self._last = None  # let the live line repaint promptly

    def clear(self):
        if self._live:
            self.stream.write("\r\x1b[2K")
            self.stream.flush()
            self._live = False


def parallel_multistart(
    config,
    bounds,
    dim_names,
    n_starts=11,
    workers=None,
    search_stride=1,
    maxfev=120,
    seed_x0=None,
    workdir="sysid_results/_ms",
    log=print,
    verbose=False,
):
    """Run N Powell searches from Sobol starts in parallel; return the best.

    Args:
        config: optimizer config (must be picklable; workers rebuild from it)
        bounds: list of (lo, hi) per dimension
        dim_names: dimension names (for logging)
        n_starts: number of starts (>=2). If seed_x0 is given it is start 0.
        workers: process count (default min(n_starts, cpu-1))
        search_stride: subsample the trajectory by this stride for speed
        maxfev: Powell max objective evals per start
        seed_x0: optional warm start (e.g. current model values)
        workdir: scratch dir for per-process model outputs
        verbose: True = raw per-iteration worker logs (debug); False = quiet
            workers + a live status line driven by shared counters
    Returns:
        (best_theta, best_err, all_results)
    """
    from scipy.stats import qmc

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    n_starts = max(2, int(n_starts))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol n != 2^k balance warning
        sob = qmc.scale(
            qmc.Sobol(d=len(bounds), scramble=True, seed=1).random(n_starts - 1), lo, hi
        )

    starts = []
    if seed_x0 is not None:
        starts.append(np.clip(np.asarray(seed_x0, float), lo, hi))
    starts += [sob[i] for i in range(len(sob))]
    starts = starts[:n_starts]
    tasks = [(i, list(map(float, x0)), maxfev) for i, x0 in enumerate(starts)]
    workers = int(workers) if workers else min(len(tasks), max(1, mp.cpu_count() - 1))

    log(
        f"  parallel multi-start: {len(tasks)} Powell starts x {workers} workers "
        f"(stride {search_stride}, maxfev {maxfev}, {len(bounds)} params: "
        f"{', '.join(dim_names)})"
    )

    # Quiet mode: workers report per-eval progress through a Manager dict and
    # the parent paints a live status line. The +1 is each start's final
    # clipped re-evaluation, so the total is an upper bound (<=).
    progress = lock = status = None
    if not verbose:
        mgr = mp.Manager()
        progress = mgr.dict({"evals": 0, "best": float("inf")})
        lock = mgr.Lock()
        status = _StatusLine()
    max_evals = len(tasks) * (maxfev + 1)

    results = []
    t0 = time.monotonic()
    with mp.Pool(
        workers,
        initializer=_winit,
        initargs=(config, search_stride, str(workdir), verbose, progress, lock),
    ) as pool:
        it = pool.imap_unordered(_wrun, tasks)
        try:
            while len(results) < len(tasks):
                if status is None:
                    r = next(it)
                else:
                    try:
                        r = it.next(timeout=1.0)
                    except mp.TimeoutError:
                        snap = dict(progress)
                        status.update(
                            f"    {snap['evals']}/<={max_evals} evals | "
                            f"best {_fmt_mm(snap['best'])} (strided search) | "
                            f"starts {len(results)}/{len(tasks)} | "
                            f"{_fmt_elapsed(time.monotonic() - t0)}"
                        )
                        continue
                results.append(r)
                tag = (
                    "seed"
                    if (r["idx"] == 0 and seed_x0 is not None)
                    else f"start{r['idx']}"
                )
                line = (
                    f"    [{len(results)}/{len(tasks)}] {tag}: {_fmt_mm(r['err'])}"
                    + (f" ({r['nfev']} evals)" if "nfev" in r else "")
                    + (f"  (FAILED: {r['exc']})" if "exc" in r else "")
                )
                if status is not None:
                    status.println(line)
                else:
                    log(line)
        finally:
            if status is not None:
                status.clear()

    results.sort(key=lambda r: r["err"])
    best = results[0]
    log(
        f"  multi-start search done: best {_fmt_mm(best['err'])} (strided) "
        f"in {_fmt_elapsed(time.monotonic() - t0)}"
    )
    return np.array(best["theta"]), best["err"], results
