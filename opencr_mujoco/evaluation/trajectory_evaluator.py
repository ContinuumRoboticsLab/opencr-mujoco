"""
Trajectory evaluator for systematic comparison of simulations against reference data.
"""

import json
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import mujoco
import numpy as np
import pandas as pd
from tqdm import tqdm

from .reference_data_loader import ReferenceDataLoader, validate_frame_conversion
from .metrics import compute_tip_error, compute_shape_error


class TrajectoryEvaluator:
    """Evaluate robot trajectories against reference data."""

    # MuJoCo integrator name → mjtIntegrator enum. Only `implicit` includes
    # joint damping inside the implicit step; `implicitfast` skips it (per
    # MuJoCo docs) and is no more stable than Euler for low-damping stiff
    # systems like the TPU dynamics rod.
    _INTEGRATOR_MAP = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }

    def __init__(
        self,
        reference_data_dir: Union[str, Path],
        results_dir: Union[str, Path] = "./evaluation_results",
        frame_conversion: Optional[np.ndarray] = None,
        integrator: Optional[str] = None,
    ):
        """
        Initialize the trajectory evaluator.

        Args:
            reference_data_dir: Directory containing reference data
            results_dir: Directory to save evaluation results
            frame_conversion: Optional 3x3 matrix R (file → MuJoCo) such that
                ``mujoco_vec = R @ file_vec``. Passed through to the
                ReferenceDataLoader so that all reference 3-vectors
                (positions, wrenches, gravity, Euler triples) come back
                already in MuJoCo frame. If None, no conversion is applied.
            integrator: Optional MuJoCo integrator override (one of
                ``euler``, ``rk4``, ``implicit``, ``implicitfast``). When set,
                applied to the model just after loading. Use ``implicit`` for
                low-damping stiff systems (e.g. TPU dynamics) where Euler /
                implicitfast diverge even at 10 kHz.
        """
        self.frame_conversion = validate_frame_conversion(frame_conversion)

        if integrator is not None:
            key = integrator.lower()
            if key not in self._INTEGRATOR_MAP:
                raise ValueError(
                    f"unknown integrator {integrator!r}; expected one of "
                    f"{sorted(self._INTEGRATOR_MAP)}"
                )
            self.integrator = self._INTEGRATOR_MAP[key]
            self.integrator_name = key
        else:
            self.integrator = None
            self.integrator_name = None

        self.reference_loader = ReferenceDataLoader(
            reference_data_dir, frame_conversion=self.frame_conversion
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for this evaluation session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = None  # Will be set when running evaluation

    def _apply_solver_settings(self, model: "mujoco.MjModel") -> None:
        """Apply runtime overrides (e.g. integrator) to a freshly-loaded model."""
        if self.integrator is not None:
            model.opt.integrator = self.integrator

    def evaluate_static_configurations(
        self,
        model_path: str,
        test_type: str,
        num_links: int,
        sim_steps: Optional[int] = None,
        sim_time: Optional[float] = None,
        sim_timestep: Optional[float] = None,
        early_stop: Optional[int] = None,
        show_progress: bool = True,
        force_ramp_time: float = 1.0,
    ) -> List[Dict]:
        """
        Evaluate static configurations against SoroSim reference data.

        Args:
            model_path: Path to MuJoCo XML model
            test_type: Type of test from reference data
            num_links: Number of links in the robot
            sim_steps: Number of simulation steps (optional, use sim_time instead)
            sim_time: Simulation time in seconds (optional, preferred over sim_steps)
            sim_timestep: Override model timestep (optional)
            early_stop: Stop after this many tests (for debugging)
            show_progress: Show progress bar

        Returns:
            List of evaluation results
        """
        # Validate inputs
        if sim_steps is None and sim_time is None:
            # Default to 5 seconds of simulation time
            sim_time = 5.0
        elif sim_steps is not None and sim_time is not None:
            raise ValueError("Specify either sim_steps or sim_time, not both")
        # Load SoroSim static reference data (CSV format). ref_arc_lengths are
        # the normalized (non-uniform) arc positions of the sample columns.
        ref_data, _, ref_arc_lengths = self.reference_loader.load_sorosim_statics_csv(
            test_type
        )

        results = []
        test_counter = 0

        # Apply early_stop up front so the progress bar shows the true total and
        # finishes cleanly (no mid-loop break that splits the tqdm line).
        items = list(ref_data.items())
        if early_stop is not None:
            items = items[:early_stop]
        iterable = (
            tqdm(items, desc=f"Evaluating {test_type} (N={num_links})")
            if show_progress
            else items
        )

        for wrenches, ref_link_positions in iterable:
            # SoroSim CSV keys are (mid_wrench, tip_wrench, gravity_vec).
            mid_wrench, tip_wrench, gravity_vec = wrenches

            # Run single evaluation
            result = self._evaluate_single_static(
                model_path,
                num_links,
                mid_wrench,
                tip_wrench,
                ref_link_positions,
                ref_arc_lengths,
                sim_steps,
                sim_time,
                sim_timestep,
                force_ramp_time,
                test_type,
                gravity_vec,
            )

            # Add test metadata
            result["test_type"] = test_type
            result["test_id"] = test_counter
            result["N"] = num_links

            results.append(result)
            test_counter += 1

        return results

    def evaluate_tip_release(
        self,
        model_path: str,
        num_links: int,
        test_type: str,
        sim_hz: float = 500.0,
        show_progress: bool = True,
        force_ramp_time: float = 1.0,
        hold_time: float = 30.0,
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate tip release dynamic test against reference data.

        This test gradually ramps up both gravity and an initial wrench to the tip,
        holds them to reach equilibrium, then releases the wrench and records the dynamic response.

        Args:
            model_path: Path to MuJoCo XML model
            num_links: Number of links in the robot
            test_type: Test type for reference data (e.g., "TipReleaseNiTiTube")
            sim_hz: Simulation frequency (Hz)
            show_progress: Show progress bar
            force_ramp_time: Time to ramp up gravity and force together (prevents instability)
            hold_time: Time to hold force at full value for equilibrium (default 30s)
            visualize: Launch MuJoCo viewer for debugging (realtime playback)

        Returns:
            Dictionary with evaluation results
        """
        # Load reference data
        load_result = self.reference_loader.load_tip_release_data(test_type)
        (
            applied_wrench,
            ref_poses,
            ref_dt,
            ref_timestamps,
            damping_ratio,
            gravity_vec,
        ) = load_result

        # Check if this is dual-tracking format (13 columns)
        is_dual_tracking = isinstance(applied_wrench, tuple)

        if is_dual_tracking:
            # Unpack dual tracking data
            mid_wrench, tip_wrench = applied_wrench
            ref_mid_poses, ref_tip_poses = ref_poses
        else:
            # Single tip tracking (old format)
            tip_wrench = applied_wrench
            mid_wrench = None
            ref_tip_poses = ref_poses
            ref_mid_poses = None

        # Load MuJoCo model
        model = mujoco.MjModel.from_xml_path(model_path)
        self._apply_solver_settings(model)
        data = mujoco.MjData(model)

        # Apply per-test-case damping if specified
        if damping_ratio is not None:
            # Apply proportional damping to all joints
            for i in range(model.njnt):
                # Get joint stiffness
                jnt_stiffness = model.jnt_stiffness[i]

                # Determine number of DOFs for this joint
                # Use the difference between consecutive dofadr values
                dof_start = model.jnt_dofadr[i]
                if i < model.njnt - 1:
                    dof_end = model.jnt_dofadr[i + 1]
                else:
                    dof_end = model.nv  # Last joint goes to end of DOF array

                # Set damping = damping_ratio * stiffness for all DOFs of this joint
                model.dof_damping[dof_start:dof_end] = damping_ratio * jnt_stiffness

        # Calculate simulation parameters
        sim_dt = 1.0 / sim_hz
        model.opt.timestep = sim_dt

        # Store original gravity
        original_gravity = model.opt.gravity.copy()

        # Apply per-test-case gravity if specified
        if gravity_vec is not None:
            # Use the full gravity vector from reference data
            # For 13-column format, this is the actual [gx, gy, gz] vector
            # For older formats, this will be None and we use the model's default
            if isinstance(gravity_vec, np.ndarray):
                original_gravity = gravity_vec
            else:
                # Scalar magnitude from older formats - scale the direction
                original_gravity = (
                    original_gravity / np.linalg.norm(original_gravity) * gravity_vec
                )

        # Total simulation time: ramp + hold + release duration from reference data
        ref_duration = (
            ref_timestamps[-1]
            if len(ref_timestamps) > 0
            else len(ref_tip_poses) * ref_dt
        )
        total_sim_time = force_ramp_time + hold_time + ref_duration
        total_steps = int(total_sim_time / sim_dt)

        # Steps for force application and hold
        ramp_steps = int(force_ramp_time / sim_dt)
        hold_steps = int(hold_time / sim_dt)
        release_step = ramp_steps + hold_steps

        # Find tip body (mj_name2id returns -1 when missing — it does not raise)
        tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
        if tip_body_id < 0:
            raise ValueError(f"Model {model_path} has no 'EE_pos' tip body")

        # Wrenches are applied at the generator's force sites when present:
        # force_site_mid sits exactly at the arc midpoint while link_{N//2}'s
        # body origin can be up to half a link off (and xfrc acts at the COM).
        tip_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "force_site_tip"
        )
        mid_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "force_site_mid"
        )

        # Find mid body if dual tracking
        mid_body_id = None
        if is_dual_tracking:
            # Mid point fallback: the body containing the arc midpoint
            mid_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"link_{num_links//2}"
            )
            if mid_site_id < 0 and mid_body_id < 0:
                raise ValueError(
                    f"Model {model_path} has neither a 'force_site_mid' site "
                    f"nor a 'link_{num_links//2}' body for the mid wrench"
                )

        tip_wrench = np.asarray(tip_wrench, dtype=float)
        mid_wrench = np.asarray(mid_wrench, dtype=float) if is_dual_tracking else None

        def apply_wrenches(scale: float):
            """Apply the (scaled) holding wrenches at the sites/bodies."""
            if tip_site_id >= 0:
                self._apply_wrench_at_site(model, data, tip_site_id, tip_wrench * scale)
            else:
                data.xfrc_applied[tip_body_id, :] = tip_wrench * scale
            if is_dual_tracking:
                if mid_site_id >= 0:
                    self._apply_wrench_at_site(
                        model, data, mid_site_id, mid_wrench * scale
                    )
                else:
                    data.xfrc_applied[mid_body_id, :] = mid_wrench * scale

        # Storage for simulation results
        sim_tip_poses = []
        sim_mid_poses = [] if is_dual_tracking else None
        sim_times = []

        # Per-link MuJoCo shape, sampled at reference timestamps (one shape
        # frame per ref_timestamps[i]). next_shape_idx tracks the next
        # reference timestamp we still need to capture; we sample the first
        # post-release step whose elapsed time meets or exceeds it.
        sim_link_shapes = []
        sim_link_shape_times = []
        next_shape_idx = 0

        # Setup visualization if requested
        viewer = None
        if visualize:
            # NOTE: must not be `import mujoco.viewer` — a function-local
            # import of the package would shadow the module-level `mujoco`
            # name for the WHOLE function scope (UnboundLocalError above).
            import mujoco.viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(model, data)
            # Set camera for better view
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            print(f"\nVisualization launched. Close window to stop.")
            print(
                f"Phase timing: Ramp={force_ramp_time}s, Hold={hold_time}s, Release={ref_duration}s"
            )
            start_time = time.time()
            last_render_time = start_time

        # Progress tracking
        if show_progress and not visualize:  # Don't show progress bar when visualizing
            tracking_desc = "mid+tip" if is_dual_tracking else "tip"
            desc = f"Tip release {tracking_desc} (ramp gravity+force:{force_ramp_time:.1f}s, hold:{hold_time:.1f}s, release:{ref_duration:.1f}s)"
            pbar = tqdm(total=total_steps, desc=desc)
        else:
            pbar = None

        # Wall-clock timing for the dynamics simulation loop. Used to
        # compute realtime_ratio (sim_seconds / wall_seconds) for the
        # dynamics overview figures.
        sim_start_data_time = data.time
        wall_start_time = time.time()

        from scipy.spatial.transform import Rotation

        def get_body_pose(body_id, pos_override=None):
            """Pose [pos(3), euler XYZ(3)] of a body (optionally site position).

            The backbone is built along +Z (unified_tdcr_generator), so the
            MuJoCo body frame already matches the reference frame — no offset
            needed. (The old +X build required a +90 deg-about-Y correction
            here, which scrambled the oscillation axis once the build switched
            to +Z.)
            """
            pos = (
                pos_override.copy()
                if pos_override is not None
                else data.body(body_id).xpos.copy()
            )
            # MuJoCo uses [w, x, y, z]; scipy expects [x, y, z, w]
            q = data.body(body_id).xquat
            roll, pitch, yaw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler(
                "xyz", degrees=False
            )
            return np.concatenate([pos, [roll, pitch, yaw]])

        def record_sample(post_release_t):
            """Record tip (and mid) pose plus any due per-link shape frames."""
            nonlocal next_shape_idx
            sim_tip_poses.append(get_body_pose(tip_body_id))
            if is_dual_tracking:
                # Track the mid position at the exact arc midpoint (site)
                # when available; the orientation comes from its parent body.
                if mid_site_id >= 0:
                    body = model.site_bodyid[mid_site_id]
                    sim_mid_poses.append(
                        get_body_pose(body, pos_override=data.site_xpos[mid_site_id])
                    )
                else:
                    sim_mid_poses.append(get_body_pose(mid_body_id))
            sim_times.append(post_release_t)

            # Capture per-link shape at the reference cadence.
            while (
                next_shape_idx < len(ref_timestamps)
                and post_release_t >= ref_timestamps[next_shape_idx]
            ):
                shape, _ = self._collect_link_positions(model, data, num_links)
                sim_link_shapes.append(np.asarray(shape, dtype=float))
                sim_link_shape_times.append(post_release_t)
                next_shape_idx += 1

        # Simulation loop
        for step in range(total_steps):
            # Clear previous forces
            data.xfrc_applied[:] = 0

            if step == release_step:
                # The reference series' t=0 sample IS the held equilibrium
                # immediately before release — record it before zeroing the
                # wrench, so the sim and reference time bases align exactly.
                record_sample(0.0)

            # Apply force and gravity ramping during ramp phase
            if step < ramp_steps:
                # Ramp up both gravity and applied force together
                scale = step / ramp_steps
                model.opt.gravity[:] = original_gravity * scale
                apply_wrenches(scale)
            elif step < release_step:
                # Hold phase - full gravity and force
                model.opt.gravity[:] = original_gravity
                apply_wrenches(1.0)
            else:
                # Release - maintain gravity, applied forces stay zero
                model.opt.gravity[:] = original_gravity

            # Step simulation
            mujoco.mj_step(model, data)

            # Record poses after release (t=0 was recorded pre-release above)
            if step >= release_step:
                record_sample((step - release_step + 1) * sim_dt)

            # Handle visualization
            if visualize and viewer.is_running():
                # Update viewer at realtime rate
                current_time = time.time()
                sim_elapsed = step * sim_dt
                real_elapsed = current_time - start_time

                # Sync to realtime (wait if simulation is ahead)
                if sim_elapsed > real_elapsed:
                    time.sleep(sim_elapsed - real_elapsed)

                # Update viewer at reasonable framerate (60 fps)
                if current_time - last_render_time > 1.0 / 60.0:
                    viewer.sync()
                    last_render_time = current_time
                    viewer.user_scn.ngeom = 0  # Clear any previous overlays

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Capture wall-clock timing immediately after the sim loop, before
        # any post-processing dilutes the measurement.
        wall_time = time.time() - wall_start_time
        actual_sim_seconds = data.time - sim_start_data_time
        realtime_ratio = actual_sim_seconds / wall_time if wall_time > 0 else 0.0

        # Close viewer if it was opened
        if viewer is not None and viewer.is_running():
            viewer.close()

        # Convert to arrays
        sim_tip_poses = np.array(sim_tip_poses)
        sim_times = np.array(sim_times)
        if is_dual_tracking:
            sim_mid_poses = np.array(sim_mid_poses)

        # Catch-up capture: the largest post-release time reached in the loop
        # is ~ref_duration - sim_dt (int() truncation of total_steps), so the
        # final reference timestamp(s) can fall just past the last step. Record
        # the final shape for any remaining timestamps so sim_link_shapes has
        # exactly one frame per reference timestamp (the documented invariant).
        while next_shape_idx < len(ref_timestamps):
            shape, _ = self._collect_link_positions(model, data, num_links)
            sim_link_shapes.append(np.asarray(shape, dtype=float))
            sim_link_shape_times.append(float(ref_timestamps[next_shape_idx]))
            next_shape_idx += 1

        # Stack per-link shape samples into (N_frames, N_links+2, 3) where
        # the last axis is xyz and the link axis covers
        # [base, link_0..link_{n-2}, link_end, EE_pos] in MuJoCo frame.
        if sim_link_shapes:
            sim_link_shapes = np.stack(sim_link_shapes, axis=0)
            sim_link_shape_times = np.asarray(sim_link_shape_times)
        else:
            sim_link_shapes = np.empty((0, num_links + 2, 3))
            sim_link_shape_times = np.empty((0,))

        # Interpolate simulation results to match reference time points
        # Use actual timestamps from reference data
        interp_sim_tip_poses = np.zeros_like(ref_tip_poses)

        for i in range(6):  # x, y, z, roll, pitch, yaw
            interp_sim_tip_poses[:, i] = np.interp(
                ref_timestamps, sim_times, sim_tip_poses[:, i]
            )

        # Compute tip errors
        tip_position_errors = np.linalg.norm(
            interp_sim_tip_poses[:, :3] - ref_tip_poses[:, :3], axis=1
        )
        # NOTE: orientation error is indicative only, not a validated metric.
        # The reference Euler triples are frame-converted component-wise (like
        # positions), which is not a valid SO(3) transform, and this is a raw
        # Euler-difference norm rather than a geodesic angle between rotations.
        # A correct version needs SoroSim's Euler convention; until then rely on
        # the position error. Same caveat applies to mid orientation below.
        tip_orientation_errors = np.linalg.norm(
            interp_sim_tip_poses[:, 3:] - ref_tip_poses[:, 3:], axis=1
        )

        # Compute tip metrics
        mean_tip_pos_error = np.mean(tip_position_errors)
        max_tip_pos_error = np.max(tip_position_errors)
        mean_tip_orient_error = np.mean(tip_orientation_errors)
        max_tip_orient_error = np.max(tip_orientation_errors)

        # Handle mid pose errors if dual tracking
        result_dict = {
            "applied_wrench": (
                tip_wrench if not is_dual_tracking else (mid_wrench, tip_wrench)
            ),
            "ref_tip_poses": ref_tip_poses,
            "ref_times": ref_timestamps,
            "sim_tip_poses": interp_sim_tip_poses,
            "sim_tip_poses_raw": sim_tip_poses,
            "sim_times_raw": sim_times,
            "position_errors": tip_position_errors,
            "orientation_errors": tip_orientation_errors,
            "mean_position_error": mean_tip_pos_error,
            "max_position_error": max_tip_pos_error,
            "mean_orientation_error": mean_tip_orient_error,
            "max_orientation_error": max_tip_orient_error,
            "num_links": num_links,
            "sim_hz": sim_hz,
            "sim_dt": sim_dt,
            "ref_dt": ref_dt,
            "damping_ratio": damping_ratio,
            "gravity_vec": gravity_vec,
            "test_type": test_type,
            "is_dual_tracking": is_dual_tracking,
            "sim_link_shapes": sim_link_shapes,
            "sim_link_shape_times": sim_link_shape_times,
            "wall_time": wall_time,
            "realtime_ratio": realtime_ratio,
            "integrator": self.integrator_name,
            "frame_conversion": (
                self.frame_conversion.tolist()
                if self.frame_conversion is not None
                else None
            ),
        }

        if is_dual_tracking:
            # Interpolate mid poses
            interp_sim_mid_poses = np.zeros_like(ref_mid_poses)
            for i in range(6):
                interp_sim_mid_poses[:, i] = np.interp(
                    ref_timestamps, sim_times, sim_mid_poses[:, i]
                )

            # Compute mid errors
            mid_position_errors = np.linalg.norm(
                interp_sim_mid_poses[:, :3] - ref_mid_poses[:, :3], axis=1
            )
            mid_orientation_errors = np.linalg.norm(
                interp_sim_mid_poses[:, 3:] - ref_mid_poses[:, 3:], axis=1
            )

            # Add mid-specific results
            result_dict.update(
                {
                    "ref_mid_poses": ref_mid_poses,
                    "sim_mid_poses": interp_sim_mid_poses,
                    "sim_mid_poses_raw": sim_mid_poses,
                    "mid_position_errors": mid_position_errors,
                    "mid_orientation_errors": mid_orientation_errors,
                    "mean_mid_position_error": np.mean(mid_position_errors),
                    "max_mid_position_error": np.max(mid_position_errors),
                    "mean_mid_orientation_error": np.mean(mid_orientation_errors),
                    "max_mid_orientation_error": np.max(mid_orientation_errors),
                }
            )

        return result_dict

    def _evaluate_single_static(
        self,
        model_path: str,
        num_links: int,
        mid_wrench: Tuple[float, ...],
        tip_wrench: Tuple[float, ...],
        ref_link_positions: List[Tuple[float, float, float]],
        ref_arc_lengths: np.ndarray,
        sim_steps: Optional[int],
        sim_time: Optional[float],
        sim_timestep: Optional[float],
        force_ramp_time: float = 1.0,
        test_type: str = "",
        gravity_vec: Optional[Tuple[float, float, float]] = None,
    ) -> Dict:
        """Evaluate a single static configuration.

        Args:
            ref_arc_lengths: Normalized arc positions of ref_link_positions.
            test_type: Test type name (used for logging only — frame
                conversion is now controlled by self.frame_conversion).
            gravity_vec: Optional gravity vector (already in MuJoCo frame
                if a frame_conversion was supplied at loader construction).
        """
        # Load model
        model = mujoco.MjModel.from_xml_path(model_path)
        self._apply_solver_settings(model)
        data = mujoco.MjData(model)

        # Apply per-test gravity if provided. The reference loader has
        # already converted gravity_vec into MuJoCo frame when a
        # frame_conversion was configured.
        if gravity_vec is not None:
            model.opt.gravity[:] = np.asarray(gravity_vec, dtype=float)

        # Override timestep if specified
        if sim_timestep is not None:
            model.opt.timestep = sim_timestep

        # Calculate number of steps from sim_time if provided
        if sim_time is not None:
            actual_sim_steps = int(sim_time / model.opt.timestep)
        else:
            actual_sim_steps = sim_steps

        # Get site IDs for force application (preferred). mj_name2id returns
        # -1 when a name is missing (it does NOT raise), so check explicitly.
        mid_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "force_site_mid"
        )
        tip_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "force_site_tip"
        )
        use_sites = mid_site_id >= 0 and tip_site_id >= 0
        if not use_sites:
            # Fallback to body IDs if sites not found
            mid_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"link_{num_links//2}"
            )
            tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
            if mid_body_id < 0 or tip_body_id < 0:
                raise ValueError(
                    f"Model {model_path} has neither force sites "
                    f"(force_site_mid/force_site_tip) nor fallback bodies "
                    f"(link_{num_links//2}/EE_pos) for wrench application"
                )
        # Measure simulation time
        sim_start_time = data.time
        wall_start_time = time.time()

        # Use the provided ramp time
        ramp_time = force_ramp_time

        # Wrenches arrive already in MuJoCo frame from the loader when a
        # frame_conversion was configured; otherwise they are passed through.
        mid_wrench_mujoco = np.asarray(mid_wrench, dtype=float)
        tip_wrench_mujoco = np.asarray(tip_wrench, dtype=float)

        # Apply forces and run simulation
        for step in range(actual_sim_steps):
            # Clear previous forces
            data.xfrc_applied[:] = 0

            # Calculate ramp factor (gradual force application)
            elapsed_time = step * model.opt.timestep
            if ramp_time > 0 and elapsed_time < ramp_time:
                ramp_factor = elapsed_time / ramp_time  # Linear ramp from 0 to 1
            else:
                ramp_factor = 1.0

            # Apply ramped forces (convert to numpy arrays for multiplication)
            ramped_mid_wrench = mid_wrench_mujoco * ramp_factor
            ramped_tip_wrench = tip_wrench_mujoco * ramp_factor

            if use_sites:
                # Apply wrenches at sites for consistent force application
                self._apply_wrench_at_site(model, data, mid_site_id, ramped_mid_wrench)
                self._apply_wrench_at_site(model, data, tip_site_id, ramped_tip_wrench)
            else:
                # Fallback to direct body application
                data.xfrc_applied[mid_body_id] = list(ramped_mid_wrench)
                data.xfrc_applied[tip_body_id] = list(ramped_tip_wrench)

            mujoco.mj_step(model, data)

        wall_time = time.time() - wall_start_time
        actual_sim_time = data.time - sim_start_time
        realtime_ratio = actual_sim_time / wall_time if wall_time > 0 else 0

        # Collect final link positions (and their arc fractions) in MuJoCo frame.
        link_positions, sim_arc_fractions = self._collect_link_positions(
            model, data, num_links
        )

        # Interpolate the reference shape onto the simulation's arc positions.
        interpolated_ref_positions = self._interpolate_reference_positions(
            ref_link_positions, ref_arc_lengths, sim_arc_fractions
        )

        # Get tip position (MuJoCo frame).
        tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
        tip_position = data.body(tip_body_id).xpos.copy()

        # Compute errors using interpolated reference positions.
        # Reference positions were already converted to MuJoCo frame by the
        # loader when a frame_conversion was configured, so both sides are
        # in the same frame here (and paired at identical arc positions).
        tip_error = compute_tip_error(tip_position, interpolated_ref_positions[-1])
        shape_error = compute_shape_error(link_positions, interpolated_ref_positions)

        return {
            "mid_wrench": list(mid_wrench),
            "tip_wrench": list(tip_wrench),
            "link_positions": link_positions,
            "tip_position": tip_position,
            "ref_tip_position": interpolated_ref_positions[-1],
            "ref_link_positions": interpolated_ref_positions,
            "original_ref_link_positions": ref_link_positions,
            "tip_error": tip_error,
            "shape_error": shape_error,
            "sim_time": actual_sim_time,
            "wall_time": wall_time,
            "realtime_ratio": realtime_ratio,
            "sim_steps": actual_sim_steps,
            "timestep": model.opt.timestep,
        }

    def _collect_link_positions(
        self, model: mujoco.MjModel, data: mujoco.MjData, num_links: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Collect backbone sample positions and their arc fractions.

        Returns ``(positions, arc_fractions)`` for: the clamped base (arc 0),
        each link body origin (link_i sits at arc (i + 0.5) / N), link_end at
        (N - 0.5) / N, and the EE_pos tip at arc 1. No frame conversion is
        applied — reference data is converted to MuJoCo frame at load time
        (see ReferenceDataLoader.frame_conversion).
        """
        positions = []
        arc_fractions = []

        # Base (arc 0): the mount the rod is clamped to.
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_base")
        if base_id >= 0:
            positions.append(data.body(base_id).xpos.copy())
            arc_fractions.append(0.0)

        # Link body origins: link_i's frame (= its joint) sits at the start of
        # its full-length capsule, arc (i + 0.5) / N along the rod.
        for i in range(num_links - 1):
            positions.append(data.body(f"link_{i}").xpos.copy())
            arc_fractions.append((i + 0.5) / num_links)

        # End half-link
        positions.append(data.body("link_end").xpos.copy())
        arc_fractions.append((num_links - 0.5) / num_links)

        # Tip
        positions.append(data.body("EE_pos").xpos.copy())
        arc_fractions.append(1.0)

        return positions, np.asarray(arc_fractions)

    def _interpolate_reference_positions(
        self,
        ref_positions: List[Tuple[float, float, float]],
        ref_arc_lengths: np.ndarray,
        sim_arc_fractions: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Interpolate the reference shape onto the simulation's sample points,
        by arc length.

        The SoroSim statics CSVs sample each shape at a small number of
        NON-uniform (Gauss–Lobatto) arc-length stations, and the simulation's
        sample points (link body origins) sit at half-link offsets. Pairing
        them by index — as an older version of this function did — compared
        points up to ~0.09·L apart along the rod and put an N-independent
        floor under shape_error. Interpolating by arc length pairs both sides
        at identical arc positions.

        Args:
            ref_positions: Reference positions at ``ref_arc_lengths``
            ref_arc_lengths: Normalized arc positions in [0, 1] of the
                reference samples (monotonically non-decreasing)
            sim_arc_fractions: Normalized arc positions of the simulation
                sample points to interpolate at

        Returns:
            Interpolated positions, one per entry of ``sim_arc_fractions``
        """
        ref_array = np.asarray(ref_positions, dtype=float)

        interpolated = np.zeros((len(sim_arc_fractions), 3))
        for dim in range(3):
            interpolated[:, dim] = np.interp(
                sim_arc_fractions, ref_arc_lengths, ref_array[:, dim]
            )

        return [interpolated[i] for i in range(len(sim_arc_fractions))]

    def save_results(
        self, results: List[Dict], config_name: str, save_positions: bool = True
    ) -> Tuple[Path, Optional[Path]]:
        """
        Save evaluation results to CSV and optionally pickle files.

        Args:
            results: List of evaluation results
            config_name: Configuration name for file naming
            save_positions: Whether to save link positions separately

        Returns:
            Paths to saved CSV and pickle files
        """
        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert complex types to strings for CSV
        for col in ["mid_wrench", "tip_wrench", "tip_position", "ref_tip_position"]:
            if col in df.columns:
                df[f"{col}_str"] = df[col].apply(
                    lambda x: (
                        ",".join(map(str, x)) if hasattr(x, "__iter__") else str(x)
                    )
                )

        # Use session directory if available, otherwise use default results_dir
        save_dir = self.session_dir / "data" if self.session_dir else self.results_dir

        # Save link positions separately if requested
        pickle_path = None
        if save_positions and "link_positions" in df.columns:
            pickle_path = save_dir / f"{config_name}_link_positions.pkl"
            self.reference_loader.save_link_positions(
                df["link_positions"].tolist(), pickle_path
            )

            # Sibling metadata file so downstream visualization can convert
            # MuJoCo-frame shapes back to file frame for comparison.
            meta_path = save_dir / f"{config_name}_run_meta.json"
            meta = {
                "frame_conversion_file_to_mujoco": (
                    self.frame_conversion.tolist()
                    if self.frame_conversion is not None
                    else None
                ),
                "saved_positions_frame": "mujoco",
                "integrator": self.integrator_name,
            }
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # Drop complex columns for CSV
        csv_df = df.drop(
            columns=[
                c
                for c in df.columns
                if c
                in [
                    "mid_wrench",
                    "tip_wrench",
                    "link_positions",
                    "tip_position",
                    "ref_tip_position",
                    "ref_link_positions",
                    "original_ref_link_positions",
                ]
            ],
            errors="ignore",
        )

        # Save CSV
        csv_path = save_dir / f"{config_name}_results.csv"
        csv_df.to_csv(csv_path, index=False)

        print(f"Results saved to {csv_path}")
        if pickle_path:
            print(f"Link positions saved to {pickle_path}")

        return csv_path, pickle_path

    def _apply_wrench_at_site(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_id: int,
        wrench: Tuple[float, ...],
    ):
        """Apply wrench at a specific site location."""
        force = np.array(wrench[:3])
        torque = np.array(wrench[3:])

        # Get site position and associated body
        site_pos = data.site(site_id).xpos.copy()
        body_id = model.site_bodyid[site_id]
        # xfrc_applied forces act at the body's center of mass (xipos), not
        # its frame origin (xpos), so the moment arm must be measured from
        # the COM. For the half-link capsules, xpos sits half a link below
        # the COM, which previously injected a spurious ~(l/2)*|F| moment.
        body_com = data.xipos[body_id].copy()

        # Compute moment from force applied at site about body COM
        r = site_pos - body_com
        moment_from_force = np.cross(r, force)

        # Total torque = original torque + moment from offset force
        total_torque = torque + moment_from_force

        # Apply to body
        data.xfrc_applied[body_id][:3] = force
        data.xfrc_applied[body_id][3:] = total_torque

    def run_tip_release_sweep(
        self,
        model_generator,
        n_values: List[int],
        test_type: str,
        config_name: str,
        **eval_kwargs,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Run tip release evaluation across multiple N values.

        Args:
            model_generator: Function that generates model XML given num_links
            n_values: List of N values (number of links)
            test_type: Test type for reference data
            config_name: Configuration name for saving
            **eval_kwargs: Additional arguments for evaluate_tip_release

        Returns:
            Paths to saved results
        """
        # Include test type in session name for clarity
        session_name = f"{self.timestamp}_{config_name}_{test_type}"
        self.session_dir = self.results_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.session_dir / "data").mkdir(exist_ok=True)
        (self.session_dir / "plots").mkdir(exist_ok=True)
        (self.session_dir / "models").mkdir(exist_ok=True)

        # Save evaluation configuration
        config_summary = {
            "timestamp": self.timestamp,
            "config_name": config_name,
            "test_type": test_type,
            "n_values": n_values,
            "eval_kwargs": eval_kwargs,
            "session_name": session_name,
        }

        with open(self.session_dir / "evaluation_config.json", "w") as f:
            json.dump(config_summary, f, indent=2, default=str)

        print(f"\nTip Release evaluation session: {session_name}")
        print(f"Output directory: {self.session_dir}")

        all_results = []

        for n in n_values:
            print(f"\nEvaluating tip release for N={n}")

            # Generate model for this N
            model_path = model_generator(n)

            # Move generated model to session directory
            model_copy = self.session_dir / "models" / f"tdcr_n{n}.xml"
            shutil.move(str(model_path), str(model_copy))

            # Run tip release evaluation
            result = self.evaluate_tip_release(
                str(model_copy), n, test_type=test_type, **eval_kwargs
            )

            # Add N value to result
            result["num_links"] = n
            all_results.append(result)

        # Save results
        return self.save_tip_release_results(all_results, config_name)

    def save_tip_release_results(
        self, results: List[Dict], config_name: str
    ) -> Tuple[Path, Path]:
        """
        Save tip release evaluation results.

        Args:
            results: List of evaluation results
            config_name: Configuration name for file naming

        Returns:
            Paths to saved CSV and pickle files
        """

        # Create DataFrame from results
        summary_data = []
        for result in results:
            # Handle gravity - convert vector to magnitude for CSV storage
            gravity_vec = result.get("gravity_vec")
            if gravity_vec is not None and isinstance(gravity_vec, np.ndarray):
                gravity_mag = np.linalg.norm(gravity_vec)
            elif gravity_vec is not None:
                gravity_mag = gravity_vec  # Already a scalar
            else:
                gravity_mag = None

            row_data = {
                "num_links": result["num_links"],
                "test_type": result.get("test_type", "Unknown"),
                "sim_hz": result["sim_hz"],
                "damping_ratio": result.get("damping_ratio"),
                "gravity_mag": gravity_mag,
                "is_dual_tracking": result.get("is_dual_tracking", False),
                "mean_tip_position_error": result["mean_position_error"],
                "max_tip_position_error": result["max_position_error"],
                "mean_tip_orientation_error": result["mean_orientation_error"],
                "max_tip_orientation_error": result["max_orientation_error"],
                "wall_time": result.get("wall_time"),
                "realtime_ratio": result.get("realtime_ratio"),
                "integrator": result.get("integrator"),
            }

            # Add mid pose metrics if available
            if result.get("is_dual_tracking", False):
                row_data.update(
                    {
                        "mean_mid_position_error": result.get(
                            "mean_mid_position_error"
                        ),
                        "max_mid_position_error": result.get("max_mid_position_error"),
                        "mean_mid_orientation_error": result.get(
                            "mean_mid_orientation_error"
                        ),
                        "max_mid_orientation_error": result.get(
                            "max_mid_orientation_error"
                        ),
                    }
                )

            summary_data.append(row_data)

        df = pd.DataFrame(summary_data)

        # Save summary CSV
        save_dir = self.session_dir / "data" if self.session_dir else self.results_dir
        csv_path = save_dir / f"{config_name}_results.csv"
        df.to_csv(csv_path, index=False)

        # Save full results as pickle
        pickle_path = save_dir / f"{config_name}_full_results.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to:")
        print(f"  - Summary CSV: {csv_path}")
        print(f"  - Full results: {pickle_path}")

        # Generate visualization plots
        self.visualize_tip_release_results(results, save_dir.parent / "plots")

        return csv_path, pickle_path

    def visualize_tip_release_results(self, results: List[Dict], plot_dir: Path):
        """
        Create visualization plots for tip release results.

        Args:
            results: List of evaluation results
            plot_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt

        plot_dir.mkdir(exist_ok=True)

        for result in results:
            n = result["num_links"]
            is_dual = result.get("is_dual_tracking", False)
            test_type = result.get("test_type", "Unknown")
            ref_times = result["ref_times"]

            # TIP POSES PLOT
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(
                f'{test_type} - N={n} links, {result["sim_hz"]:.0f}Hz - TIP',
                fontsize=14,
            )

            ref_tip_poses = result["ref_tip_poses"]
            sim_tip_poses = result["sim_tip_poses"]

            # Position components
            for i, label in enumerate(["X", "Y", "Z"]):
                ax = axes[i, 0]
                ax.plot(
                    ref_times,
                    ref_tip_poses[:, i],
                    "b-",
                    label="Reference",
                    linewidth=1.5,
                )
                ax.plot(
                    ref_times,
                    sim_tip_poses[:, i],
                    "r--",
                    label=f'Simulation ({result["sim_hz"]:.0f}Hz)',
                    linewidth=1.5,
                )
                ax.set_ylabel(f"Tip {label} Position (m)")
                ax.set_xlabel("Time (s)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_title(f"Tip {label} Position Trajectory")

            # Orientation components
            for i, label in enumerate(["Roll", "Pitch", "Yaw"]):
                ax = axes[i, 1]
                ax.plot(
                    ref_times,
                    ref_tip_poses[:, i + 3],
                    "b-",
                    label="Reference",
                    linewidth=1.5,
                )
                ax.plot(
                    ref_times,
                    sim_tip_poses[:, i + 3],
                    "r--",
                    label="Simulation",
                    linewidth=1.5,
                )
                ax.set_ylabel(f"Tip {label} (rad)")
                ax.set_xlabel("Time (s)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_title(f"Tip {label} Orientation")

            plt.tight_layout()
            plt.savefig(plot_dir / f"tip_release_n{n}_tip_trajectories.png", dpi=150)
            plt.close()

            # MID POSES PLOT (if dual tracking)
            if is_dual:
                fig, axes = plt.subplots(3, 2, figsize=(15, 12))
                fig.suptitle(
                    f'{test_type} - N={n} links, {result["sim_hz"]:.0f}Hz - MID',
                    fontsize=14,
                )

                ref_mid_poses = result["ref_mid_poses"]
                sim_mid_poses = result["sim_mid_poses"]

                # Position components
                for i, label in enumerate(["X", "Y", "Z"]):
                    ax = axes[i, 0]
                    ax.plot(
                        ref_times,
                        ref_mid_poses[:, i],
                        "b-",
                        label="Reference",
                        linewidth=1.5,
                    )
                    ax.plot(
                        ref_times,
                        sim_mid_poses[:, i],
                        "r--",
                        label=f'Simulation ({result["sim_hz"]:.0f}Hz)',
                        linewidth=1.5,
                    )
                    ax.set_ylabel(f"Mid {label} Position (m)")
                    ax.set_xlabel("Time (s)")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_title(f"Mid {label} Position Trajectory")

                # Orientation components
                for i, label in enumerate(["Roll", "Pitch", "Yaw"]):
                    ax = axes[i, 1]
                    ax.plot(
                        ref_times,
                        ref_mid_poses[:, i + 3],
                        "b-",
                        label="Reference",
                        linewidth=1.5,
                    )
                    ax.plot(
                        ref_times,
                        sim_mid_poses[:, i + 3],
                        "r--",
                        label="Simulation",
                        linewidth=1.5,
                    )
                    ax.set_ylabel(f"Mid {label} (rad)")
                    ax.set_xlabel("Time (s)")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_title(f"Mid {label} Orientation")

                plt.tight_layout()
                plt.savefig(
                    plot_dir / f"tip_release_n{n}_mid_trajectories.png", dpi=150
                )
                plt.close()

            # Error plots - TIP
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f"Tip Release Errors - N={n} links - TIP", fontsize=14)

            # Tip position error over time
            ax1.plot(
                ref_times,
                result["position_errors"],
                "b-",
                linewidth=1.5,
                label="Tip Position Error",
            )
            ax1.set_ylabel("Position Error (m)")
            ax1.set_xlabel("Time (s)")
            ax1.grid(True, alpha=0.3)
            ax1.set_title(
                f'Tip Position Error (Mean: {result["mean_position_error"]:.6f}m, Max: {result["max_position_error"]:.6f}m)'
            )
            ax1.legend()

            # Tip orientation error over time
            ax2.plot(
                ref_times,
                result["orientation_errors"],
                "r-",
                linewidth=1.5,
                label="Tip Orientation Error",
            )
            ax2.set_ylabel("Orientation Error (rad)")
            ax2.set_xlabel("Time (s)")
            ax2.grid(True, alpha=0.3)
            ax2.set_title(
                f'Tip Orientation Error (Mean: {result["mean_orientation_error"]:.6f}rad, Max: {result["max_orientation_error"]:.6f}rad)'
            )
            ax2.legend()

            plt.tight_layout()
            plt.savefig(plot_dir / f"tip_release_n{n}_tip_errors.png", dpi=150)
            plt.close()

            # Error plots - MID (if dual tracking)
            if is_dual:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                fig.suptitle(f"Tip Release Errors - N={n} links - MID", fontsize=14)

                # Mid position error over time
                ax1.plot(
                    ref_times,
                    result["mid_position_errors"],
                    "b-",
                    linewidth=1.5,
                    label="Mid Position Error",
                )
                ax1.set_ylabel("Position Error (m)")
                ax1.set_xlabel("Time (s)")
                ax1.grid(True, alpha=0.3)
                ax1.set_title(
                    f'Mid Position Error (Mean: {result["mean_mid_position_error"]:.6f}m, Max: {result["max_mid_position_error"]:.6f}m)'
                )
                ax1.legend()

                # Mid orientation error over time
                ax2.plot(
                    ref_times,
                    result["mid_orientation_errors"],
                    "r-",
                    linewidth=1.5,
                    label="Mid Orientation Error",
                )
                ax2.set_ylabel("Orientation Error (rad)")
                ax2.set_xlabel("Time (s)")
                ax2.grid(True, alpha=0.3)
                ax2.set_title(
                    f'Mid Orientation Error (Mean: {result["mean_mid_orientation_error"]:.6f}rad, Max: {result["max_mid_orientation_error"]:.6f}rad)'
                )
                ax2.legend()

                plt.tight_layout()
                plt.savefig(plot_dir / f"tip_release_n{n}_mid_errors.png", dpi=150)
                plt.close()

        # Summary plot comparing all N values - TIP
        if len(results) > 1:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Tip Release Error vs Number of Links - TIP", fontsize=14)

            n_values = [r["num_links"] for r in results]
            mean_pos_errors = [r["mean_position_error"] for r in results]
            max_pos_errors = [r["max_position_error"] for r in results]
            mean_orient_errors = [r["mean_orientation_error"] for r in results]
            max_orient_errors = [r["max_orientation_error"] for r in results]

            # Mean position error
            ax1.loglog(
                n_values, mean_pos_errors, "bo-", linewidth=2, markersize=8, label="Tip"
            )
            ax1.set_xlabel("Number of Links")
            ax1.set_ylabel("Mean Position Error (m)")
            ax1.grid(True, alpha=0.3, which="both")
            ax1.set_title("Mean Tip Position Error")
            ax1.legend()

            # Max position error
            ax2.loglog(
                n_values, max_pos_errors, "bs-", linewidth=2, markersize=8, label="Tip"
            )
            ax2.set_xlabel("Number of Links")
            ax2.set_ylabel("Max Position Error (m)")
            ax2.grid(True, alpha=0.3, which="both")
            ax2.set_title("Maximum Tip Position Error")
            ax2.legend()

            # Mean orientation error
            ax3.loglog(
                n_values,
                mean_orient_errors,
                "ro-",
                linewidth=2,
                markersize=8,
                label="Tip",
            )
            ax3.set_xlabel("Number of Links")
            ax3.set_ylabel("Mean Orientation Error (rad)")
            ax3.grid(True, alpha=0.3, which="both")
            ax3.set_title("Mean Tip Orientation Error")
            ax3.legend()

            # Max orientation error
            ax4.loglog(
                n_values,
                max_orient_errors,
                "rs-",
                linewidth=2,
                markersize=8,
                label="Tip",
            )
            ax4.set_xlabel("Number of Links")
            ax4.set_ylabel("Max Orientation Error (rad)")
            ax4.grid(True, alpha=0.3, which="both")
            ax4.set_title("Maximum Tip Orientation Error")
            ax4.legend()

            plt.tight_layout()
            plt.savefig(plot_dir / "tip_release_tip_error_summary.png", dpi=150)
            plt.close()

            # Summary plot for MID (if dual tracking)
            if any(r.get("is_dual_tracking", False) for r in results):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle("Tip Release Error vs Number of Links - MID", fontsize=14)

                mean_mid_pos_errors = [
                    r.get("mean_mid_position_error", 0) for r in results
                ]
                max_mid_pos_errors = [
                    r.get("max_mid_position_error", 0) for r in results
                ]
                mean_mid_orient_errors = [
                    r.get("mean_mid_orientation_error", 0) for r in results
                ]
                max_mid_orient_errors = [
                    r.get("max_mid_orientation_error", 0) for r in results
                ]

                # Mean position error
                ax1.loglog(
                    n_values,
                    mean_mid_pos_errors,
                    "go-",
                    linewidth=2,
                    markersize=8,
                    label="Mid",
                )
                ax1.set_xlabel("Number of Links")
                ax1.set_ylabel("Mean Position Error (m)")
                ax1.grid(True, alpha=0.3, which="both")
                ax1.set_title("Mean Mid Position Error")
                ax1.legend()

                # Max position error
                ax2.loglog(
                    n_values,
                    max_mid_pos_errors,
                    "gs-",
                    linewidth=2,
                    markersize=8,
                    label="Mid",
                )
                ax2.set_xlabel("Number of Links")
                ax2.set_ylabel("Max Position Error (m)")
                ax2.grid(True, alpha=0.3, which="both")
                ax2.set_title("Maximum Mid Position Error")
                ax2.legend()

                # Mean orientation error
                ax3.loglog(
                    n_values,
                    mean_mid_orient_errors,
                    "mo-",
                    linewidth=2,
                    markersize=8,
                    label="Mid",
                )
                ax3.set_xlabel("Number of Links")
                ax3.set_ylabel("Mean Orientation Error (rad)")
                ax3.grid(True, alpha=0.3, which="both")
                ax3.set_title("Mean Mid Orientation Error")
                ax3.legend()

                # Max orientation error
                ax4.loglog(
                    n_values,
                    max_mid_orient_errors,
                    "ms-",
                    linewidth=2,
                    markersize=8,
                    label="Mid",
                )
                ax4.set_xlabel("Number of Links")
                ax4.set_ylabel("Max Orientation Error (rad)")
                ax4.grid(True, alpha=0.3, which="both")
                ax4.set_title("Maximum Mid Orientation Error")
                ax4.legend()

                plt.tight_layout()
                plt.savefig(plot_dir / "tip_release_mid_error_summary.png", dpi=150)
                plt.close()

            print(f"Visualization plots saved to {plot_dir}")

    def run_parameter_sweep(
        self,
        model_generator,
        test_types: List[str],
        n_values: List[int],
        config_name: str,
        **eval_kwargs,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Run evaluation across multiple parameter values.

        Args:
            model_generator: Function that generates model XML given num_links
            test_types: List of test types to evaluate
            n_values: List of N values (number of links)
            config_name: Configuration name for saving
            **eval_kwargs: Additional arguments for evaluate_static_configurations

        Returns:
            Paths to saved results
        """
        # Create session directory with timestamp and test type
        test_type_str = test_types[0] if len(test_types) == 1 else "multi_test"
        session_name = f"{self.timestamp}_{config_name}_{test_type_str}"
        self.session_dir = self.results_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.session_dir / "data").mkdir(exist_ok=True)
        (self.session_dir / "plots").mkdir(exist_ok=True)
        (self.session_dir / "models").mkdir(exist_ok=True)

        # Save evaluation configuration
        config_summary = {
            "timestamp": self.timestamp,
            "config_name": config_name,
            "test_types": test_types,
            "n_values": n_values,
            "eval_kwargs": eval_kwargs,
            "session_name": session_name,
        }

        with open(self.session_dir / "evaluation_config.json", "w") as f:
            json.dump(config_summary, f, indent=2)

        # Create README for session
        readme_content = f"""# Evaluation Session: {session_name}

## Overview
- **Timestamp**: {self.timestamp}
- **Configuration**: {config_name}
- **Test Types**: {', '.join(test_types)}
- **N Values**: {n_values}

## Directory Structure
- `data/`: Evaluation results (CSV and pickle files)
  - `{config_name}_results.csv`: Main results file
  - `{config_name}_link_positions.pkl`: Link position data (if saved)
- `plots/`: Visualization outputs
  - Error plots, shape comparisons, debug plots
- `models/`: Generated TDCR model files
  - One XML file per N value tested
- `evaluation_config.json`: Complete configuration used

## Generated Plots
- Error vs N: Shows how simulation error varies with discretization
- Runtime performance: Computational efficiency metrics
- Error histograms: Distribution of errors for each N
- Shape comparisons: Visual comparison with reference data
- Debug plots: Worst-case scenarios for each N value
"""

        with open(self.session_dir / "README.md", "w") as f:
            f.write(readme_content)

        print(f"\nEvaluation session: {session_name}")
        print(f"Output directory: {self.session_dir}")

        all_results = []

        for test_type in test_types:
            print(f"\nProcessing test type: {test_type}")

            for n in n_values:
                print(f"Evaluating N={n}")

                # Generate model for this N
                model_path = model_generator(n)

                # Move generated model to session directory
                model_copy = self.session_dir / "models" / f"tdcr_n{n}.xml"
                shutil.move(str(model_path), str(model_copy))

                # Run evaluation
                results = self.evaluate_static_configurations(
                    str(model_copy), test_type, n, **eval_kwargs
                )

                all_results.extend(results)

        # Save all results
        csv_path, pickle_path = self.save_results(all_results, config_name)

        # Generate visualization plots
        try:
            from .paper_visualization import PaperVisualizer

            plot_dir = self.session_dir / "plots"
            plot_dir.mkdir(exist_ok=True)

            visualizer = PaperVisualizer(
                results_dir=self.session_dir,
                plots_dir=plot_dir,
                frame_conversion=self.frame_conversion,
            )

            # Generate error vs N plot
            visualizer.plot_error_vs_n(csv_path, config_name, save=True, show=False)

            # Generate runtime performance plot
            visualizer.plot_runtime_performance(
                csv_path, config_name, save=True, show=False
            )

            # Generate error histograms (single figure with subplots for all N values)
            try:
                visualizer.plot_error_histograms(
                    csv_path,
                    config_name,
                    n_values=n_values,
                    n_exclude=[],  # Don't exclude any N values
                    save=True,
                    show=False,
                )
            except Exception as hist_error:
                print(f"Could not generate error histograms: {hist_error}")

            # Generate debug plots with worst-case shapes
            if pickle_path and pickle_path.exists():
                try:
                    reference_dir = self.reference_loader.data_dir
                    visualizer.plot_highest_error_shapes(
                        csv_path,
                        pickle_path,
                        reference_dir,
                        config_name,
                        n_values=n_values,
                        n_exclude=[],
                        num_cases=5,  # Show top 5 worst cases
                        save_dir=plot_dir / "debug",
                        show=False,
                    )
                    print("Generated debug plots for worst-case shapes")
                except Exception as debug_error:
                    print(f"Could not generate debug plots: {debug_error}")

            # Generate reference shape distribution plot
            try:
                # Use the first test type (since run_parameter_sweep gets test_types list)
                plot_test_type = test_types[0] if test_types else "Unknown"
                visualizer.plot_reference_shape_distribution(
                    self.reference_loader.data_dir,
                    plot_test_type,
                    save_dir=plot_dir,
                    show=False,
                )
                print("Generated reference shape distribution plot")
            except Exception as dist_error:
                print(f"Could not generate shape distribution plot: {dist_error}")

            # Generate convergence plots for worst-case shapes in debug folder
            try:
                df = pd.read_csv(csv_path)
                debug_dir = plot_dir / "debug"
                debug_dir.mkdir(exist_ok=True)

                for n in n_values:
                    # Get model path for this N
                    model_path = self.session_dir / "models" / f"tdcr_n{n}.xml"
                    if not model_path.exists():
                        continue

                    # Get the worst-case wrench from the results
                    n_df = df[df["N"] == n]
                    if len(n_df) > 0:
                        # Use the worst error case for convergence plot
                        worst_sample = n_df.loc[n_df["tip_error"].idxmax()]

                        # Parse comma-separated wrench strings safely
                        mid_wrench = [
                            float(v)
                            for v in str(worst_sample["mid_wrench_str"]).split(",")
                        ]
                        tip_wrench = [
                            float(v)
                            for v in str(worst_sample["tip_wrench_str"]).split(",")
                        ]

                        # Create a temporary visualizer with debug directory
                        debug_visualizer = PaperVisualizer(
                            results_dir=self.session_dir,
                            plots_dir=debug_dir,
                            frame_conversion=self.frame_conversion,
                        )

                        # Get actual simulation parameters from eval_kwargs
                        actual_sim_time = eval_kwargs.get("sim_time", 5.0)
                        actual_sim_hz = eval_kwargs.get("sim_hz", 500.0)
                        actual_force_ramp = eval_kwargs.get("force_ramp_time", 1.0)

                        debug_visualizer.plot_convergence_history(
                            str(model_path),
                            tuple(mid_wrench),
                            tuple(tip_wrench),
                            f"{config_name}_worst",
                            n_value=n,
                            sim_time=actual_sim_time,
                            sim_hz=actual_sim_hz,
                            force_ramp_time=actual_force_ramp,
                            save=True,
                            show=False,
                        )
                print("Generated convergence plots for worst cases in debug folder")
            except Exception as conv_error:
                print(f"Could not generate convergence plots: {conv_error}")

            # Generate shape comparison plots for each N value if positions were saved
            if pickle_path and pickle_path.exists():
                reference_dir = self.reference_loader.data_dir
                for n in n_values:
                    try:
                        visualizer.plot_shape_comparison_3d(
                            csv_path,
                            pickle_path,
                            reference_dir,
                            config_name,
                            n_value=n,
                            num_samples=10,
                            save=True,
                            show=False,
                        )
                    except Exception as shape_error:
                        print(
                            f"Could not generate shape comparison for N={n}: {shape_error}"
                        )

            # Generate summary report
            report_path = plot_dir / f"{config_name}_summary.txt"
            report_text = visualizer.create_summary_report(
                csv_path, config_name, save_path=report_path
            )

            print(f"\nVisualization plots saved to {plot_dir}")
            print("\nSummary Statistics:")
            print("-" * 40)
            # Print key statistics from the report. Keep the metric headers
            # ("Tip Error (mm):", "Real-time Factor:") so the two Mean lines are
            # unambiguous.
            keep = ("N =", "Samples:", "Tip Error", "Real-time Factor", "Mean:")
            for line in report_text.split("\n"):
                if any(k in line for k in keep):
                    print(line)

        except Exception as e:
            print(f"Warning: Could not generate visualization plots: {e}")
            import traceback

            traceback.print_exc()

        return csv_path, pickle_path
