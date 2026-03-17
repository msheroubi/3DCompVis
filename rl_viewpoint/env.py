"""
Gymnasium environment for RL-based viewpoint selection.

Three operating modes:
  • **simulated** — generates random scenes with primitive objects and
                    uses analytic DDA ray-casting to simulate visibility
                    with occlusion.  **No arm, no server, 100 % offline.**
  • **cached**    — replays pre-recorded occupancy transitions from a
                    ``transitions.npz`` file for fast offline training.
  • **live**      — sends real camera frames to the GPU server for
                    reconstruction and reads back the occupancy grid.

Observation space
-----------------
  Dict:
    "occupancy"   : Box(0, 2, shape=(1, RES, RES, RES))
                    Down-sampled occupancy grid (channel-first for Conv3d)
    "cam_pos"     : Box(-inf, inf, shape=(3,))
                    Current camera XYZ in world frame
    "cam_dir"     : Box(-1, 1, shape=(3,))
                    Current viewing direction (unit vector)
    "explored"    : Box(0, 1, shape=(1,))
                    Scalar explored_fraction

Action space
------------
  Box(-1, 1, shape=(4,)) — normalised (x, y, z, wrist_roll).
  Mapped back to metres / degrees, then matched to nearest FK bank entry.

Reward
------
  Δ explored_fraction (the increase from one step to the next).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import (
    UNKNOWN, FREE, OCCUPIED,
    build_fk_bank,
    downsample_occupancy,
    match_to_bank,
    action_to_cartesian,
)

logger = logging.getLogger(__name__)

# Default down-sampled grid resolution
OBS_GRID_RES: int = 16


# ======================================================================
# Cached environment (offline training from pre-recorded transitions)
# ======================================================================

class CachedViewpointEnv(gym.Env):
    """Replay pre-recorded occupancy transitions for offline RL training.

    The cached dataset stores, for each transition:
      • ``occ_before``  — 64³ occupancy grid before the action
      • ``occ_after``   — 64³ occupancy grid after the action
      • ``action_xyz``  — (3,) camera position of the chosen viewpoint
      • ``action_roll`` — wrist roll (degrees)
      • ``cam_pos``     — (3,) camera position before acting
      • ``cam_dir``     — (3,) camera direction before acting

    A trajectory is sampled by randomly chaining transitions that are
    consistent (i.e. ``occ_before[k+1] == occ_after[k]``), or by
    simply replaying an episode in order.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str | Path,
        obs_res: int = OBS_GRID_RES,
        max_steps: int = 20,
    ):
        super().__init__()
        self.obs_res = obs_res
        self.max_steps = max_steps

        # Load cached data
        data = np.load(data_path)
        self.occ_before: np.ndarray = data["occ_before"]   # (N, 64, 64, 64)
        self.occ_after: np.ndarray = data["occ_after"]      # (N, 64, 64, 64)
        self.action_xyz: np.ndarray = data["action_xyz"]    # (N, 3)
        self.action_roll: np.ndarray = data["action_roll"]  # (N,)
        self.cam_pos_data: np.ndarray = data["cam_pos"]     # (N, 3)
        self.cam_dir_data: np.ndarray = data["cam_dir"]     # (N, 3)

        # Episode boundaries: data["episode_ids"] (N,) — int episode id
        if "episode_ids" in data:
            self.episode_ids: np.ndarray = data["episode_ids"]
        else:
            # Treat entire dataset as one episode
            self.episode_ids = np.zeros(len(self.occ_before), dtype=np.int32)

        self.n_transitions = len(self.occ_before)
        self._unique_eps = np.unique(self.episode_ids)
        logger.info("Loaded %d transitions across %d episodes from %s",
                     self.n_transitions, len(self._unique_eps), data_path)

        # Position bounds for action normalisation
        self.pos_low = self.action_xyz.min(axis=0).astype(np.float32)
        self.pos_high = self.action_xyz.max(axis=0).astype(np.float32)
        self.roll_low = float(self.action_roll.min())
        self.roll_high = float(self.action_roll.max())
        # Small padding so extremes map to ±1 properly
        pad = 0.02
        self.pos_low -= pad
        self.pos_high += pad
        self.roll_low -= 1.0
        self.roll_high += 1.0

        # Spaces
        self.observation_space = spaces.Dict({
            "occupancy": spaces.Box(
                low=0, high=2,
                shape=(1, obs_res, obs_res, obs_res),
                dtype=np.float32,
            ),
            "cam_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "cam_dir": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "explored": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Episode state
        self._step_count: int = 0
        self._current_occ: np.ndarray = np.zeros((64, 64, 64), dtype=np.uint8)
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._cam_dir = np.array([0, 0, 1], dtype=np.float32)
        self._ep_indices: list[int] = []
        self._ep_cursor: int = 0

    # ── helpers ─────────────────────────────────────────────────────
    def _explored_fraction(self, grid: np.ndarray) -> float:
        total = grid.size
        known = np.count_nonzero(grid != UNKNOWN)
        return known / total

    def _make_obs(self) -> dict[str, np.ndarray]:
        ds = downsample_occupancy(self._current_occ, self.obs_res)
        return {
            "occupancy": ds[np.newaxis].astype(np.float32),  # (1, R, R, R)
            "cam_pos": self._cam_pos.copy(),
            "cam_dir": self._cam_dir.copy(),
            "explored": np.array(
                [self._explored_fraction(self._current_occ)],
                dtype=np.float32,
            ),
        }

    # ── gymnasium API ──────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        # Pick a random episode
        ep_id = self.np_random.choice(self._unique_eps)
        mask = self.episode_ids == ep_id
        self._ep_indices = list(np.where(mask)[0])
        self._ep_cursor = 0
        self._step_count = 0

        # Initial state = occ_before of first transition in episode
        first = self._ep_indices[0]
        self._current_occ = self.occ_before[first].copy()
        self._cam_pos = self.cam_pos_data[first].astype(np.float32).copy()
        self._cam_dir = self.cam_dir_data[first].astype(np.float32).copy()

        return self._make_obs(), {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        explored_before = self._explored_fraction(self._current_occ)

        # ── Apply action ───────────────────────────────────────────
        # In cached mode the action is *ignored* for state transitions;
        # we simply advance to the next recorded transition.  The agent
        # still gets the real Δ explored_fraction as reward, so it learns
        # which states lead to good coverage growth.
        #
        # (In a future on-policy version the action would select which
        # bank candidate to visit.)
        if self._ep_cursor < len(self._ep_indices):
            idx = self._ep_indices[self._ep_cursor]
            self._current_occ = self.occ_after[idx].copy()
            # Update camera pose to the pose *after* this transition
            if self._ep_cursor + 1 < len(self._ep_indices):
                next_idx = self._ep_indices[self._ep_cursor + 1]
                self._cam_pos = self.cam_pos_data[next_idx].astype(np.float32).copy()
                self._cam_dir = self.cam_dir_data[next_idx].astype(np.float32).copy()
            self._ep_cursor += 1

        explored_after = self._explored_fraction(self._current_occ)
        reward = explored_after - explored_before

        # Termination
        terminated = explored_after >= 0.95
        truncated = (
            self._step_count >= self.max_steps
            or self._ep_cursor >= len(self._ep_indices)
        )

        info: dict[str, Any] = {
            "explored_fraction": explored_after,
            "delta_explored": reward,
            "step": self._step_count,
        }
        return self._make_obs(), float(reward), terminated, truncated, info


# ======================================================================
# Live environment (talks to the GPU server + real arm)
# ======================================================================

class LiveViewpointEnv(gym.Env):
    """Gymnasium environment that drives the real SO-101 arm + GPU server.

    Each ``step()`` call:
      1. Maps the action to the nearest bank candidate (via FK matching).
      2. Moves the arm to that candidate.
      3. Captures a frame and sends it to the server.
      4. Triggers reconstruction (merge_all=True).
      5. Fetches the new occupancy grid.
      6. Returns reward = Δ explored_fraction.

    This is **slow** (one step ≈ 10-30 s) and intended for evaluation or
    fine-tuning, not for bulk training.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        server_url: str,
        robot: Any,                # SOFollower instance
        candidates: list[list[float]],
        obs_res: int = OBS_GRID_RES,
        max_steps: int = 15,
        scene_center: np.ndarray | None = None,
        scene_forward: np.ndarray | None = None,
        scene_extent: float = 2.4,
        bounds_type: str = "hemisphere",
    ):
        super().__init__()
        self.server = server_url
        self.robot = robot
        self.candidates = candidates
        self.obs_res = obs_res
        self.max_steps = max_steps
        self.scene_center = scene_center
        self.scene_forward = scene_forward
        self.scene_extent = scene_extent
        self.bounds_type = bounds_type

        # Precompute FK bank
        self.bank_pos, self.bank_dir, self.bank_rolls = build_fk_bank(candidates)

        # Position bounds for normalisation
        self.pos_low = self.bank_pos.min(axis=0) - 0.02
        self.pos_high = self.bank_pos.max(axis=0) + 0.02
        self.roll_low = float(self.bank_rolls.min()) - 1.0
        self.roll_high = float(self.bank_rolls.max()) + 1.0

        # Spaces
        self.observation_space = spaces.Dict({
            "occupancy": spaces.Box(
                low=0, high=2,
                shape=(1, obs_res, obs_res, obs_res),
                dtype=np.float32,
            ),
            "cam_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "cam_dir": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "explored": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # State
        self._step_count = 0
        self._current_occ = np.zeros((64, 64, 64), dtype=np.uint8)
        self._grid_origin = np.zeros(3, dtype=np.float32)
        self._voxel_size = 0.0
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._cam_dir = np.array([0, 0, 1], dtype=np.float32)
        self._session_id: str | None = None
        self._visited: set[int] = set()

    # ── server helpers ─────────────────────────────────────────────
    def _api(self, path: str, method: str = "GET", **kwargs) -> dict:
        import requests
        url = self.server.rstrip("/") + path
        r = getattr(requests, method.lower())(url, timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()

    def _fetch_occupancy(self) -> tuple[np.ndarray, np.ndarray, float]:
        import gzip, io, requests
        r = requests.get(self.server.rstrip("/") + "/occupancy", timeout=30)
        if r.status_code == 404:
            return self._current_occ, self._grid_origin, self._voxel_size
        r.raise_for_status()
        shape = tuple(int(d) for d in r.headers["X-Grid-Shape"].split(","))
        voxel_size = float(r.headers["X-Voxel-Size"])
        origin = np.array([float(v) for v in r.headers["X-Voxel-Origin"].split(",")])
        raw = r.content
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass
        grid = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        return grid, origin, voxel_size

    # ── helpers ─────────────────────────────────────────────────────
    def _explored_fraction(self, grid: np.ndarray) -> float:
        total = grid.size
        known = np.count_nonzero(grid != UNKNOWN)
        return known / total

    def _make_obs(self) -> dict[str, np.ndarray]:
        ds = downsample_occupancy(self._current_occ, self.obs_res)
        return {
            "occupancy": ds[np.newaxis].astype(np.float32),
            "cam_pos": self._cam_pos.copy(),
            "cam_dir": self._cam_dir.copy(),
            "explored": np.array(
                [self._explored_fraction(self._current_occ)],
                dtype=np.float32,
            ),
        }

    # ── gymnasium API ──────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._visited.clear()
        self._current_occ = np.zeros((64, 64, 64), dtype=np.uint8)

        # Reset server state
        try:
            self._api("/reset", "POST")
        except Exception as e:
            logger.warning("Server /reset failed: %s", e)

        # Initialise scene bounds
        if self.scene_center is not None:
            try:
                self._api("/scene/init", "POST", json={
                    "center": self.scene_center.tolist(),
                    "forward": (self.scene_forward.tolist()
                                if self.scene_forward is not None
                                else [0, 0, 1]),
                    "half_extent": self.scene_extent,
                    "bounds_type": self.bounds_type,
                })
            except Exception as e:
                logger.warning("Server /scene/init failed: %s", e)

        # Move arm to reference viewpoint (first candidate)
        from active_capture import move_arm, read_joint_positions, joint_to_pose7
        ref_vp = self.candidates[0]
        move_arm(self.robot, ref_vp)
        time.sleep(0.5)

        # Read initial camera state
        joints = read_joint_positions(self.robot)
        from so101_kinematics import ee_pose
        T_c2w = ee_pose(joints)
        self._cam_pos = T_c2w[:3, 3].astype(np.float32).copy()
        fwd = -T_c2w[:3, 2].copy()
        fwd /= np.linalg.norm(fwd) + 1e-12
        self._cam_dir = fwd.astype(np.float32)

        return self._make_obs(), {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        explored_before = self._explored_fraction(self._current_occ)

        # 1. Map action → Cartesian target → nearest bank candidate
        target_xyz, target_roll = action_to_cartesian(
            action, self.pos_low, self.pos_high, self.roll_low, self.roll_high)
        bank_idx = match_to_bank(
            target_xyz, target_roll,
            self.bank_pos, self.bank_rolls)
        vp_joints = self.candidates[bank_idx]
        self._visited.add(bank_idx)

        # 2. Move arm + capture frame
        import cv2
        from active_capture import move_arm, read_joint_positions, joint_to_pose7
        move_arm(self.robot, vp_joints)
        time.sleep(0.3)

        obs = self.robot.get_observation()
        frame = obs.get("wrist_cam")

        if frame is not None:
            # Send frame to server via a mini-session
            import asyncio, websockets, orjson, struct
            POSE_STRUCT = struct.Struct("<7f")
            joints_now = read_joint_positions(self.robot)
            pose = joint_to_pose7(joints_now)
            jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ok, encoded = cv2.imencode(".jpg", frame, jpeg_params)

            if ok:
                try:
                    resp = self._api("/session/start", "POST",
                                     params={"max_frames": 5})
                    sid = resp["session_id"]
                    ws_uri = (self.server
                              .replace("http://", "ws://")
                              .replace("https://", "wss://")
                              .rstrip("/") + "/ws/vision")

                    async def _send():
                        async with websockets.connect(ws_uri, max_size=2**24) as ws:
                            msg = POSE_STRUCT.pack(*pose) + encoded.tobytes()
                            await ws.send(msg)
                            _ = await ws.recv()

                    asyncio.get_event_loop().run_until_complete(_send())
                    self._api("/session/stop", "POST")

                    # 3. Reconstruct
                    self._api("/reconstruct", "POST",
                              params={"session_id": sid, "merge_all": True})
                    for _ in range(60):
                        time.sleep(2.0)
                        status = self._api("/reconstruct/status")
                        if not status["running"]:
                            break
                except Exception as e:
                    logger.warning("Live step failed: %s", e)

        # 4. Fetch new occupancy
        self._current_occ, self._grid_origin, self._voxel_size = (
            self._fetch_occupancy())

        # Update camera state
        from so101_kinematics import ee_pose as _ee
        T_c2w = _ee(np.array(vp_joints, dtype=float))
        self._cam_pos = T_c2w[:3, 3].astype(np.float32).copy()
        fwd = -T_c2w[:3, 2].copy()
        fwd /= np.linalg.norm(fwd) + 1e-12
        self._cam_dir = fwd.astype(np.float32)

        explored_after = self._explored_fraction(self._current_occ)
        reward = explored_after - explored_before

        terminated = explored_after >= 0.95
        truncated = self._step_count >= self.max_steps

        info = {
            "explored_fraction": explored_after,
            "delta_explored": reward,
            "bank_idx": bank_idx,
            "vp_joints": vp_joints,
            "step": self._step_count,
        }
        return self._make_obs(), float(reward), terminated, truncated, info


# ======================================================================
# Simulated environment (analytic visibility, no arm, no server)
# ======================================================================

class SimulatedViewpointEnv(gym.Env):
    """Fully simulated viewpoint exploration with random primitive scenes.

    Each episode:
      1. Generates a new random scene (boxes, spheres, cylinders).
      2. Builds a ground-truth 64³ occupancy grid.
      3. The agent's observed grid starts all UNKNOWN.
      4. Each step: the agent chooses a viewpoint (action → bank match),
         and the simulator casts rays from that camera pose.  Rays
         traverse the grid using 3-D DDA; intermediate empty voxels are
         revealed as FREE, and the first OCCUPIED voxel hit is revealed
         (with natural occlusion — the ray stops).
      5. Reward = Δ explored_fraction.

    Camera model mirrors the real SO-101 wrist camera:
      • Mounted to the LEFT of the gripper.
      • Angled ~5° rightward so the optical axis converges with the
        gripper's pointing direction.
      • 80° H-FOV × 60° V-FOV, max range 0.6 m.

    This environment is **fast** (pure numpy / optional numba) and
    designed for large-scale PPO training without any hardware.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        candidates: list[list[float]],
        obs_res: int = OBS_GRID_RES,
        grid_res: int = 64,
        max_steps: int = 20,
        half_extent: float = 0.3,
        scene_center: np.ndarray | None = None,
        n_objects: tuple[int, int] = (3, 8),
        object_size_range: tuple[float, float] = (0.02, 0.08),
        hfov_deg: float = 60.0,
        vfov_deg: float = 47.0,
        max_range: float = 0.5,
        ray_w: int = 130,
        ray_h: int = 98,
        use_real_fk: bool = False,
    ):
        """
        Parameters
        ----------
        candidates : list of joint-angle lists
            Viewpoint bank (same format as ``generate_viewpoint_bank``).
        obs_res : int
            Downsampled grid resolution for the observation (default 16).
        grid_res : int
            Full occupancy grid resolution (default 64).
        max_steps : int
            Max viewpoints per episode.
        half_extent : float
            Half side-length of the voxel volume (metres).
        scene_center : (3,) or None
            Centre of the scene bounding volume.
        n_objects : (min, max)
            Range for number of random objects per scene.
        object_size_range : (min, max)
            Range for object half-extents / radii (metres).
        hfov_deg, vfov_deg : float
            Camera field of view.
        max_range : float
            Maximum ray distance (metres).
        ray_w, ray_h : int
            Sparse ray grid resolution.
        use_real_fk : bool
            If True, use ``so101_kinematics.ee_pose`` for FK.
            If False (default), use the approximate ``simulated_fk_bank``.
        """
        super().__init__()
        from .simulator import (
            generate_random_scene,
            VisibilitySimulator,
            simulated_fk_bank,
            _patch_simulator_numba,
        )
        self._generate_scene = generate_random_scene
        self._VisSim = VisibilitySimulator
        self._patch_numba = _patch_simulator_numba

        self.candidates = candidates
        self.obs_res = obs_res
        self.grid_res = grid_res
        self.max_steps = max_steps
        self.half_extent = half_extent
        self.scene_center = (
            np.asarray(scene_center, dtype=np.float64)
            if scene_center is not None
            else np.array([0.15, 0.0, 0.0], dtype=np.float64)
        )
        self.n_objects = n_objects
        self.object_size_range = object_size_range
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.max_range = max_range
        self.ray_w = ray_w
        self.ray_h = ray_h

        # Build FK bank — try real FK first, fall back to simulated
        if use_real_fk:
            try:
                self.bank_pos, self.bank_dir, self.bank_rolls = build_fk_bank(candidates)
                logger.info("Using real FK (so101_kinematics)")
            except ImportError:
                logger.warning("so101_kinematics not found, using simulated FK")
                self.bank_pos, self.bank_dir, self.bank_rolls = simulated_fk_bank(
                    candidates, self.scene_center)
        else:
            self.bank_pos, self.bank_dir, self.bank_rolls = simulated_fk_bank(
                candidates, self.scene_center)

        # Normalisation bounds
        self.pos_low = self.bank_pos.min(axis=0) - 0.02
        self.pos_high = self.bank_pos.max(axis=0) + 0.02
        self.roll_low = float(self.bank_rolls.min()) - 1.0
        self.roll_high = float(self.bank_rolls.max()) + 1.0

        # Spaces
        self.observation_space = spaces.Dict({
            "occupancy": spaces.Box(
                low=0, high=2,
                shape=(1, obs_res, obs_res, obs_res),
                dtype=np.float32,
            ),
            "cam_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "cam_dir": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "explored": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Episode state (set in reset)
        self._sim: VisibilitySimulator | None = None
        self._step_count = 0
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._cam_dir = np.array([0, 0, 1], dtype=np.float32)
        self._visited: set[int] = set()

    # ── helpers ─────────────────────────────────────────────────────
    def _make_obs(self) -> dict[str, np.ndarray]:
        ds = downsample_occupancy(self._sim.observed, self.obs_res)
        return {
            "occupancy": ds[np.newaxis].astype(np.float32),
            "cam_pos": self._cam_pos.copy(),
            "cam_dir": self._cam_dir.copy(),
            "explored": np.array(
                [self._sim.explored_fraction], dtype=np.float32),
        }

    # ── gymnasium API ──────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        # Generate a new random scene
        gt, grid_origin, voxel_size, objects = self._generate_scene(
            rng=rng,
            grid_res=self.grid_res,
            half_extent=self.half_extent,
            scene_center=self.scene_center,
            n_objects=self.n_objects,
            size_range=self.object_size_range,
        )

        # Create visibility simulator
        self._sim = self._VisSim(
            ground_truth=gt,
            grid_origin=grid_origin,
            voxel_size=voxel_size,
            hfov_deg=self.hfov_deg,
            vfov_deg=self.vfov_deg,
            max_range=self.max_range,
            ray_w=self.ray_w,
            ray_h=self.ray_h,
        )
        self._patch_numba(self._sim)

        self._step_count = 0
        self._visited.clear()

        # Initial camera pose = first bank candidate
        self._cam_pos = self.bank_pos[0].copy()
        self._cam_dir = self.bank_dir[0].copy()

        # Take an initial observation from the starting pose
        self._sim.observe_from(self._cam_pos, self._cam_dir)

        info = {
            "n_objects": len(objects),
            "explored_fraction": self._sim.explored_fraction,
        }
        return self._make_obs(), info

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        explored_before = self._sim.explored_fraction

        # 1. Map normalised action to Cartesian → nearest bank candidate
        target_xyz, target_roll = action_to_cartesian(
            action, self.pos_low, self.pos_high,
            self.roll_low, self.roll_high)
        bank_idx = match_to_bank(
            target_xyz, target_roll,
            self.bank_pos, self.bank_rolls)
        self._visited.add(bank_idx)

        # 2. Get camera pose from bank
        self._cam_pos = self.bank_pos[bank_idx].copy()
        self._cam_dir = self.bank_dir[bank_idx].copy()

        # 3. Simulate observation (DDA ray-casting with occlusion)
        delta = self._sim.observe_from(self._cam_pos, self._cam_dir)

        explored_after = self._sim.explored_fraction
        reward = explored_after - explored_before

        terminated = explored_after >= 0.95
        truncated = self._step_count >= self.max_steps

        info = {
            "explored_fraction": explored_after,
            "delta_explored": reward,
            "bank_idx": bank_idx,
            "step": self._step_count,
            "n_visited": len(self._visited),
        }
        return self._make_obs(), float(reward), terminated, truncated, info
