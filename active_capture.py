#!/usr/bin/env python3
"""
active_capture.py — Capture frames from SO-101 + stream to GPU server for
                     3D reconstruction with occupancy-grid exploration.

Runs **on the local machine** (where the arm is plugged in).
Talks to the GPU server over the network.

Workflow
--------
  1. Connect to the SO-101 arm (follower + optional leader for teleop).
  2. Connect to the GPU server's WebSocket.
  3. Stream camera frames + FK-derived poses while you teleoperate.
  4. After collection → trigger reconstruction on the server.
  5. Fetch occupancy grid stats + next-best-view plan.
  6. Display suggested viewpoints; optionally auto-move the arm there.
  7. Repeat from step 3 for iterative coverage.

Usage examples (run on your LOCAL machine):
    # Teleop mode — move the leader arm, press SPACE to capture
    python active_capture.py \\
        --server http://GPU_SERVER_IP:8765 \\
        --port /dev/ttyACM1 \\
        --teleop-port /dev/ttyACM0 \\
        --mode teleop

    # Manual mode — move arm by hand, press SPACE to capture
    python active_capture.py \\
        --server http://GPU_SERVER_IP:8765 \\
        --port /dev/ttyACM1 \\
        --mode manual

    # Auto mode — arm moves through predefined viewpoints
    python active_capture.py \\
        --server http://GPU_SERVER_IP:8765 \\
        --port /dev/ttyACM1 \\
        --mode auto

Requirements (local machine):
    pip install opencv-python numpy websockets requests orjson
    # + lerobot SDK for arm control
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import logging
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Occupancy constants (must match reconstruction.py)
# ---------------------------------------------------------------------------
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

# ---------------------------------------------------------------------------
# Wire protocol (same as client.py)
# ---------------------------------------------------------------------------
POSE_STRUCT = struct.Struct("<7f")  # x y z qx qy qz qw — 28 bytes LE


# ---------------------------------------------------------------------------
# Lerobot imports — gracefully degrade
# ---------------------------------------------------------------------------
_HAS_LEROBOT = False
try:
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    _HAS_LEROBOT = True
except ImportError:
    pass

# FK kinematics — must be on PYTHONPATH or in cwd on the local machine
try:
    from so101_kinematics import camera_extrinsic, ee_pose, pose_to_quaternion_translation
    _HAS_FK = True
except ImportError:
    _HAS_FK = False

# ---------------------------------------------------------------------------
# Predefined auto-capture viewpoints (joint degrees)
# ---------------------------------------------------------------------------
# Rest / home position the arm returns to at the end of a run.
# Adjust these joint angles (degrees) to match your setup.
REST_POSITION: list[float] = [0, -90, 90, 0, 0, 0]

DEFAULT_VIEWPOINTS: list[list[float]] = [
    [   0, -30,  60, -30,   0, 50],
    [  20, -30,  60, -30,   0, 50],
    [ -20, -30,  60, -30,   0, 50],
    [  40, -30,  60, -30,   0, 50],
    [ -40, -30,  60, -30,   0, 50],
    [   0, -45,  75, -30,   0, 50],
    [  20, -45,  75, -30,   0, 50],
    [ -20, -45,  75, -30,   0, 50],
    [   0, -20,  45, -25,   0, 50],
    [  30, -20,  45, -25,   0, 50],
    [ -30, -20,  45, -25,   0, 50],
    [   0, -30,  60, -30,  45, 50],
    [   0, -30,  60, -30, -45, 50],
    [  15, -35,  65, -30,  20, 50],
    [ -15, -35,  65, -30, -20, 50],
    [   0, -55,  85, -30,   0, 50],
]


# ---------------------------------------------------------------------------
# SO-101 joint limits (degrees) — CONSERVATIVE bounds
# ---------------------------------------------------------------------------
# Derived from tested DEFAULT_VIEWPOINTS range plus a safety margin.
# Every auto-generated target is validated against these before commanding.
#   J0: base rotation
#   J1: shoulder (negative = forward/down, -90 = folded back at rest)
#   J2: elbow
#   J3: wrist pitch
#   J4: wrist roll
#   J5: gripper / camera tilt (fixed at 50° during scanning)
SO101_JOINT_LIMITS: list[tuple[float, float]] = [
    (-90.0,  90.0),   # J0: base rotation
    (-90.0,   0.0),   # J1: shoulder
    ( 20.0, 110.0),   # J2: elbow
    (-60.0,  10.0),   # J3: wrist pitch
    (-90.0,  90.0),   # J4: wrist roll
    (  0.0,  50.0),   # J5: gripper / camera tilt
]

# Tighter limits used for auto-generated viewpoint sampling (extra margin)
_SAMPLE_MARGIN_DEG = 5.0
_SCAN_LIMITS: list[tuple[float, float]] = [
    (lo + _SAMPLE_MARGIN_DEG, hi - _SAMPLE_MARGIN_DEG)
    if hi - lo > 2 * _SAMPLE_MARGIN_DEG else (lo, hi)
    for lo, hi in SO101_JOINT_LIMITS
]


def validate_joint_angles(joints: list[float] | np.ndarray) -> bool:
    """Return True if all joints are within SO101_JOINT_LIMITS."""
    for i, (lo, hi) in enumerate(SO101_JOINT_LIMITS):
        if i >= len(joints):
            break
        if joints[i] < lo - 0.1 or joints[i] > hi + 0.1:
            return False
    return True


def clamp_joint_angles(joints: list[float] | np.ndarray) -> list[float]:
    """Clamp joints to SO101_JOINT_LIMITS, warn if any were out of range."""
    result = list(joints)
    for i, (lo, hi) in enumerate(SO101_JOINT_LIMITS):
        if i >= len(result):
            break
        original = result[i]
        result[i] = max(lo, min(hi, result[i]))
        if abs(result[i] - original) > 0.1:
            logger.warning("Joint %d clamped: %.1f -> %.1f (limits [%.1f, %.1f])",
                          i, original, result[i], lo, hi)
    return result


# ---------------------------------------------------------------------------
# Dense viewpoint bank generation
# ---------------------------------------------------------------------------

def _halton_value(index: int, base: int) -> float:
    """Single value from a Halton low-discrepancy sequence."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def generate_viewpoint_bank(
    n: int = 200,
    seed_offset: int = 20,
    include_defaults: bool = True,
) -> list[list[float]]:
    """Generate a dense viewpoint bank via Halton quasi-random sampling.

    Samples joints J0-J4 within ``_SCAN_LIMITS`` and fixes J5 = 50 deg
    (camera tilt for scanning).  Filters out self-collision-prone
    configurations using conservative heuristics.

    Parameters
    ----------
    n : int
        Target number of *sampled* viewpoints (excluding defaults).
    seed_offset : int
        Starting index in the Halton sequence (skip warm-up).
    include_defaults : bool
        If True, prepend the hand-tuned DEFAULT_VIEWPOINTS.
    """
    bases = [2, 3, 5, 7, 11]  # primes for 5-D Halton
    limits = _SCAN_LIMITS[:5]   # only the 5 active joints

    bank: list[list[float]] = []
    if include_defaults:
        bank.extend(DEFAULT_VIEWPOINTS)

    n_defaults = len(bank)
    idx = seed_offset
    attempts = 0
    max_attempts = n * 5  # avoid infinite loop

    while len(bank) < n_defaults + n:
        joints = []
        for d, base in enumerate(bases):
            lo, hi = limits[d]
            val = lo + (hi - lo) * _halton_value(idx, base)
            joints.append(round(val, 1))
        joints.append(50.0)  # J5 fixed for scanning
        idx += 1
        attempts += 1
        if attempts > max_attempts:
            break

        # --- Self-collision / stress heuristics -------------------------
        j0, j1, j2, j3, j4 = joints[:5]

        # Shoulder + elbow too folded → arm folds into base
        if j1 + j2 < -10:
            continue

        # Elbow nearly closed while shoulder is barely tilted → squeeze
        if j2 < 35 and j1 > -25:
            continue

        # Extreme wrist roll with shoulder high → servo stress
        if abs(j4) > 70 and j1 > -20:
            continue

        # Duplicate check (within 3 deg per joint of existing entry)
        duplicate = False
        for existing in bank:
            if all(abs(existing[d] - joints[d]) < 3.0 for d in range(5)):
                duplicate = True
                break
        if duplicate:
            continue

        bank.append(joints)

    logger.info("Generated viewpoint bank: %d viewpoints "
                "(%d defaults + %d sampled)",
                len(bank), n_defaults, len(bank) - n_defaults)
    return bank


# ======================================================================
# Robot helpers
# ======================================================================

def return_to_rest(robot: "SOFollower") -> None:
    """Move the arm back to the resting/home position."""
    print("\n  Returning arm to rest position ...")
    try:
        move_arm(robot, REST_POSITION, check_limits=False)
        print("  Arm is at rest position.")
    except Exception as e:
        logger.warning("Could not move arm to rest position: %s", e)


def read_joint_positions(robot: "SOFollower") -> np.ndarray:
    obs = robot.get_observation()
    motor_names = list(robot.bus.motors.keys())
    return np.array([obs[f"{m}.pos"] for m in motor_names], dtype=np.float64)


def move_arm(robot: "SOFollower", target_deg: list[float],
             steps: int = 40, dt: float = 0.03,
             check_limits: bool = True) -> None:
    if check_limits:
        if not validate_joint_angles(target_deg):
            logger.warning("Target outside joint limits - clamping: %s", target_deg)
            target_deg = clamp_joint_angles(target_deg)
    current = read_joint_positions(robot)
    target = np.array(target_deg, dtype=np.float64)
    motor_names = list(robot.bus.motors.keys())
    for i in range(1, steps + 1):
        interp = current + (target - current) * (i / steps)
        # Safety: clamp every interpolated step to joint limits
        if check_limits:
            for j_idx in range(min(len(interp), len(SO101_JOINT_LIMITS))):
                lo, hi = SO101_JOINT_LIMITS[j_idx]
                interp[j_idx] = max(lo, min(hi, interp[j_idx]))
        action = {f"{m}.pos": float(interp[j]) for j, m in enumerate(motor_names)}
        robot.send_action(action)
        time.sleep(dt)
    time.sleep(0.3)


def joint_to_pose7(joint_deg: np.ndarray) -> tuple[float, ...]:
    """Convert joint angles → (x, y, z, qx, qy, qz, qw) using FK."""
    if _HAS_FK:
        T_w2c = camera_extrinsic(joint_deg)
        T_c2w = ee_pose(joint_deg)
        quat, tvec = pose_to_quaternion_translation(T_w2c)
        # Return camera position in world frame + quaternion
        pos = T_c2w[:3, 3]
        return (float(pos[0]), float(pos[1]), float(pos[2]),
                float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]))
    else:
        # Fallback: identity pose
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


# ======================================================================
# Server communication
# ======================================================================

def api(base: str, path: str, method: str = "GET", **kwargs) -> dict:
    url = base.rstrip("/") + path
    r = getattr(requests, method.lower())(url, timeout=120, **kwargs)
    r.raise_for_status()
    return r.json()


def ws_url(base: str) -> str:
    return base.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws/vision"


def encode_message(jpeg_bytes: bytes, pose: tuple[float, ...]) -> bytes:
    return POSE_STRUCT.pack(*pose) + jpeg_bytes


def print_reconstruction_result(result: dict) -> None:
    """Pretty-print the /reconstruct/result JSON."""
    print()
    print("=" * 60)
    print("  RECONSTRUCTION RESULT")
    print("=" * 60)
    print(f"  Input frames     : {result['n_input_frames']}")
    print(f"  Processing time  : {result['processing_time_s']} s")
    print(f"  Fused points     : {result['n_fused_points']:,}")
    print(f"  Mean confidence  : {result['mean_confidence']:.4f}")

    unc = result.get("uncertainty", {})
    print(f"  Mean uncertainty : {unc.get('mean_uncertainty', 0):.4f}")
    print(f"  High-unc fraction: {unc.get('high_uncertainty_fraction', 0):.2%}")

    occ = result.get("occupancy", {})
    if occ:
        print()
        total = occ.get("n_free", 0) + occ.get("n_occupied", 0) + occ.get("n_unknown", 0)
        print(f"  --- Occupancy Grid ---")
        print(f"  Grid shape       : {occ.get('grid_shape')}")
        print(f"  Voxel size       : {occ.get('voxel_size_m', 0):.4f} m")
        print(f"  FREE             : {occ.get('n_free', 0):,}  ({occ.get('free_fraction', 0):.1%})")
        print(f"  OCCUPIED         : {occ.get('n_occupied', 0):,}  ({occ.get('occupied_fraction', 0):.1%})")
        print(f"  UNKNOWN          : {occ.get('n_unknown', 0):,}")
        print(f"  Explored         : {occ.get('explored_fraction', 0):.1%}")
    print("=" * 60)


def print_plan(views: list[dict]) -> None:
    """Pretty-print next-best-view suggestions."""
    if not views:
        print("\n  Scene looks fully explored — no new viewpoints needed!")
        return
    print(f"\n  Next-Best-View Plan ({len(views)} viewpoints):")
    for v in views:
        p = v["pose"]
        print(f"    [P{v['priority']}]  unc={v['expected_uncertainty']:.3f}"
              f"  pos=({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f})"
              f"  → target=({v['target_world'][0]:.3f},"
              f" {v['target_world'][1]:.3f},"
              f" {v['target_world'][2]:.3f})")


# ======================================================================
# Live viser visualizer (runs in a daemon thread)
# ======================================================================

def _fetch_pointcloud(base: str) -> dict:
    """Fetch fused point cloud + camera poses from /pointcloud."""
    r = requests.get(base.rstrip("/") + "/pointcloud", timeout=60)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    raw = r.content
    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass
    npz = np.load(io.BytesIO(raw))
    return {k: npz[k] for k in npz.files}


def _fetch_occupancy(base: str) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Returns (grid, voxel_origin, voxel_size) or (None, None, 0) on error."""
    r = requests.get(base.rstrip("/") + "/occupancy", timeout=30)
    if r.status_code == 404:
        return None, None, 0.0
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


def _fetch_plan(base: str) -> list[dict]:
    try:
        r = requests.get(base.rstrip("/") + "/plan", timeout=30)
        r.raise_for_status()
        return r.json().get("viewpoints", [])
    except Exception:
        return []


def _compute_frontier(occupancy: np.ndarray) -> np.ndarray:
    is_unknown = occupancy == UNKNOWN
    is_free = occupancy == FREE
    dilated = np.zeros_like(is_free)
    dilated[1:, :, :] |= is_free[:-1, :, :]
    dilated[:-1, :, :] |= is_free[1:, :, :]
    dilated[:, 1:, :] |= is_free[:, :-1, :]
    dilated[:, :-1, :] |= is_free[:, 1:, :]
    dilated[:, :, 1:] |= is_free[:, :, :-1]
    dilated[:, :, :-1] |= is_free[:, :, 1:]
    return is_unknown & dilated


def _invert_extrinsics(ext: np.ndarray) -> np.ndarray:
    """(S, 3, 4) world-to-cam → cam-to-world."""
    c2w = np.zeros_like(ext)
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    c2w[:, :3, :3] = R.transpose(0, 2, 1)
    c2w[:, :3, 3] = -(R.transpose(0, 2, 1) @ t[:, :, None])[:, :, 0]
    return c2w


class LiveVisualizer:
    """Background viser server that refreshes after each reconstruction round.

    Usage::

        vis = LiveVisualizer(port=8080)
        vis.start()          # starts viser in a daemon thread
        # ... after each reconstruction round:
        vis.update(base_url)  # fetches latest data and redraws
    """

    def __init__(self, port: int = 8080):
        self.port = port
        self._server = None
        self._center = np.zeros(3, dtype=np.float32)

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        import viser
        self._server = viser.ViserServer(host="0.0.0.0", port=self.port)
        self._server.gui.configure_theme(
            titlebar_content=None, control_layout="collapsible")
        self._round_label = self._server.gui.add_text(
            "Status", initial_value="Waiting for first reconstruction ...")
        self._gui_show_occ = self._server.gui.add_checkbox(
            "Show occupancy grid", initial_value=True)
        self._gui_show_bounds = self._server.gui.add_checkbox(
            "Show grid bounding box", initial_value=True)
        self._gui_show_unknown = self._server.gui.add_checkbox(
            "Show UNKNOWN voxels", initial_value=False)
        self._gui_conf = self._server.gui.add_slider(
            "Confidence threshold %", min=0, max=100, step=1, initial_value=10)

        # References to scene objects (overwritten on update)
        self._pc_handle = None
        self._occ_handle = None
        self._unk_handle = None
        self._bounds_handle = None
        self._nbv_group = None
        self._cam_handles: list = []
        self._cur_xyz: np.ndarray | None = None
        self._cur_rgb: np.ndarray | None = None
        self._cur_conf: np.ndarray | None = None

        # Wire up toggle callbacks
        @self._gui_show_occ.on_update
        def _(_) -> None:
            if self._occ_handle is not None:
                self._occ_handle.visible = self._gui_show_occ.value

        @self._gui_show_bounds.on_update
        def _(_) -> None:
            if self._bounds_handle is not None:
                self._bounds_handle.visible = self._gui_show_bounds.value

        @self._gui_show_unknown.on_update
        def _(_) -> None:
            if self._unk_handle is not None:
                self._unk_handle.visible = self._gui_show_unknown.value

        @self._gui_conf.on_update
        def _(_) -> None:
            if self._cur_conf is not None and self._pc_handle is not None:
                thr = np.percentile(self._cur_conf, self._gui_conf.value)
                m = self._cur_conf >= thr
                self._pc_handle.points = self._cur_xyz[m] - self._center
                self._pc_handle.colors = self._cur_rgb[m]

        print(f"\n  Live viser viewer at  http://localhost:{self.port}")
        print(f"  Open in a browser to see live updates.\n")

    # ── update (called from the main loop) ────────────────────────────────
    def update(self, base_url: str, *, round_n: int = 0) -> None:
        """Fetch latest reconstruction data from the server and redraw."""
        if self._server is None:
            return
        try:
            self._do_update(base_url, round_n)
        except Exception as e:
            logger.warning("LiveVisualizer update failed: %s", e)

    def _do_update(self, base: str, round_n: int) -> None:
        import viser.transforms as vtf

        s = self._server
        assert s is not None

        # ── Fetch data ────────────────────────────────────────────────
        pc = _fetch_pointcloud(base)
        grid, grid_origin, voxel_size = _fetch_occupancy(base)
        plan = _fetch_plan(base)

        if not pc or "xyz" not in pc:
            self._round_label.value = f"Round {round_n} — no point cloud yet"
            return

        xyz = pc["xyz"].astype(np.float32)
        rgb = pc["rgb"].astype(np.uint8)
        conf = pc["conf"].astype(np.float32)
        self._center = xyz.mean(axis=0) if len(xyz) > 0 else np.zeros(3)
        self._cur_xyz = xyz
        self._cur_rgb = rgb
        self._cur_conf = conf

        def rc(pts: np.ndarray) -> np.ndarray:
            return pts - self._center

        # ── Status label ──────────────────────────────────────────────
        explored = 0.0
        if grid is not None:
            explored = np.count_nonzero(grid != UNKNOWN) / grid.size
        self._round_label.value = (
            f"Round {round_n} — {len(xyz):,} pts — "
            f"{explored:.1%} explored — {len(plan)} NBV"
        )

        # ── Point cloud ──────────────────────────────────────────────
        thr = np.percentile(conf, self._gui_conf.value)
        m = conf >= thr
        self._pc_handle = s.scene.add_point_cloud(
            "pointcloud", points=rc(xyz[m]), colors=rgb[m],
            point_size=0.002, point_shape="circle",
        )

        # ── Camera frustums ──────────────────────────────────────────
        # Remove old cameras
        for h in self._cam_handles:
            h.remove()
        self._cam_handles.clear()

        extr = pc.get("extrinsics")
        intr = pc.get("intrinsics")
        imgs = pc.get("images")
        if extr is not None and intr is not None:
            c2w = _invert_extrinsics(extr)
            for i in range(len(extr)):
                R = c2w[i, :3, :3]
                t = c2w[i, :3, 3]
                wxyz = vtf.SO3.from_matrix(R).wxyz
                frame = s.scene.add_frame(
                    f"cameras/cam_{i}",
                    wxyz=wxyz, position=t - self._center,
                    axes_length=0.03, axes_radius=0.001, origin_radius=0.001,
                )
                fy = intr[i, 1, 1]
                H = int(round(intr[i, 1, 2] * 2))
                W = int(round(intr[i, 0, 2] * 2))
                fov = float(2 * np.arctan2(H / 2, fy)) if fy > 0 else 0.7
                asp = W / H if H > 0 else 1.0
                thumb = imgs[i] if imgs is not None and len(imgs) > i else None
                s.scene.add_camera_frustum(
                    f"cameras/cam_{i}/frustum",
                    fov=fov, aspect=asp, scale=0.04,
                    image=thumb, line_width=1.0,
                )
                self._cam_handles.append(frame)

        # ── Occupancy grid ───────────────────────────────────────────
        if grid is not None and grid_origin is not None:
            G = grid.shape[0]
            occ_ijk = np.argwhere(grid == OCCUPIED)
            front_ijk = np.argwhere(_compute_frontier(grid))
            occ_xyz = grid_origin + (occ_ijk + 0.5) * voxel_size
            front_xyz = grid_origin + (front_ijk + 0.5) * voxel_size
            all_occ_xyz = (np.vstack([occ_xyz, front_xyz])
                           if len(front_xyz) > 0 else occ_xyz)
            colors_occ = np.vstack([
                np.tile([220, 40, 40], (len(occ_xyz), 1)),
                np.tile([160, 160, 160], (len(front_xyz), 1)),
            ]).astype(np.uint8) if (len(occ_xyz) + len(front_xyz)) > 0 else np.zeros((0, 3), np.uint8)

            if len(all_occ_xyz) > 0:
                self._occ_handle = s.scene.add_point_cloud(
                    "occupancy", points=rc(all_occ_xyz), colors=colors_occ,
                    point_size=voxel_size * 0.8, point_shape="square",
                )
                self._occ_handle.visible = self._gui_show_occ.value
            else:
                self._occ_handle = None

            # UNKNOWN voxels (sparse sample)
            unk_ijk = np.argwhere(grid == UNKNOWN)
            if len(unk_ijk) > 0:
                rng = np.random.default_rng(0)
                sample = rng.choice(len(unk_ijk), min(len(unk_ijk), 8000), replace=False)
                unk_xyz = grid_origin + (unk_ijk[sample] + 0.5) * voxel_size
                self._unk_handle = s.scene.add_point_cloud(
                    "unknown", points=rc(unk_xyz),
                    colors=np.tile([80, 80, 120], (len(unk_xyz), 1)).astype(np.uint8),
                    point_size=voxel_size * 0.4, point_shape="square",
                )
                self._unk_handle.visible = self._gui_show_unknown.value
            else:
                self._unk_handle = None

            # Bounding-box wireframe
            lo = grid_origin - self._center
            hi = grid_origin + G * voxel_size - self._center
            x0, y0, z0 = lo
            x1, y1, z1 = hi
            corners = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ])
            edges = np.array([
                [corners[0], corners[1]], [corners[1], corners[2]],
                [corners[2], corners[3]], [corners[3], corners[0]],
                [corners[4], corners[5]], [corners[5], corners[6]],
                [corners[6], corners[7]], [corners[7], corners[4]],
                [corners[0], corners[4]], [corners[1], corners[5]],
                [corners[2], corners[6]], [corners[3], corners[7]],
            ], dtype=np.float32)
            self._bounds_handle = s.scene.add_line_segments(
                "grid_bounds", points=edges,
                colors=np.array([255, 255, 100], dtype=np.uint8),
                line_width=1.5,
            )
            self._bounds_handle.visible = self._gui_show_bounds.value
        else:
            self._occ_handle = None
            self._unk_handle = None
            self._bounds_handle = None

        # ── NBV viewpoints ───────────────────────────────────────────
        # Remove old NBV markers
        try:
            s.scene.remove("nbv")
        except Exception:
            pass

        if plan:
            for i, vp in enumerate(plan[:5]):
                pose = vp["pose"]
                pos = np.array([pose["x"], pose["y"], pose["z"]]) - self._center
                tgt_raw = vp.get("target_world")
                if tgt_raw:
                    tgt = np.array(tgt_raw) - self._center
                    s.scene.add_line_segments(
                        f"nbv/ray_{i}",
                        points=np.array([[pos, tgt]]),
                        colors=np.array([255, 220, 0], dtype=np.uint8),
                        line_width=1.5,
                    )
                s.scene.add_icosphere(
                    f"nbv/point_{i}", radius=0.008,
                    position=pos, color=(255, 220, 0),
                )


# ======================================================================
# Gripper removal
# ======================================================================

def _crop_bottom(frame: np.ndarray) -> np.ndarray:
    """Crop the bottom 1/3 of the frame (removes gripper cleanly)."""
    h = frame.shape[0]
    return frame[: h * 2 // 3, :]


def apply_gripper_mask(frame: np.ndarray) -> np.ndarray:
    """Black-out the trapezoid region where the gripper is visible."""
    h, w = frame.shape[:2]
    pts = np.array([
        [int(0.25 * w), int(2/3 * h)],
        [int(0.65 * w), int(2/3 * h)],
        [int(0.70 * w), h - 1],
        [int(0.20 * w), h - 1],
    ], dtype=np.int32)
    out = frame.copy()
    cv2.fillPoly(out, [pts], color=(0, 0, 0))
    return out


def _process_frame(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Apply gripper removal: crop bottom 1/3 by default, or polygon mask with --mask-gripper."""
    if getattr(args, "mask_gripper", False):
        return apply_gripper_mask(frame)
    return _crop_bottom(frame)


# ======================================================================
# Core capture → reconstruct → plan loop
# ======================================================================

async def capture_and_reconstruct(
    args: argparse.Namespace,
    robot: "SOFollower",
    leader: "SO101Leader | None" = None,
) -> None:
    """Run one iteration of capture→reconstruct→plan.

    Returns the planned viewpoints so the caller can decide to continue.
    """
    import orjson
    import websockets

    base = args.server
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    # ---- 1. Start collection session on server ----
    resp = api(base, "/session/start", "POST",
               params={"max_frames": args.max_frames,
                       "min_baseline_m": args.min_baseline})
    sid = resp["session_id"]
    print(f"\n[session] Started: {sid}")

    # ---- 2. Stream frames ----
    n_accepted = 0
    n_sent = 0
    t_start = time.time()

    if args.mode == "auto":
        # Auto: move through viewpoints, capture each one
        viewpoints = DEFAULT_VIEWPOINTS
        if args.viewpoints_json:
            with open(args.viewpoints_json) as f:
                viewpoints = json.load(f)

        async with websockets.connect(ws_url(base), max_size=2**24) as ws:
            for vp_idx, vp in enumerate(viewpoints):
                print(f"\n  Moving to viewpoint {vp_idx+1}/{len(viewpoints)}: {vp}")
                move_arm(robot, vp)

                obs = robot.get_observation()
                frame = obs.get("wrist_cam")
                if frame is None:
                    continue
                frame = _process_frame(frame, args)

                joint_deg = read_joint_positions(robot)
                pose = joint_to_pose7(joint_deg)

                ok, encoded = cv2.imencode(".jpg", frame, jpeg_params)
                if not ok:
                    continue

                msg = encode_message(encoded.tobytes(), pose)
                await ws.send(msg)
                resp_raw = await ws.recv()
                payload = orjson.loads(resp_raw)
                n_sent += 1

                coll = payload.get("collection", {})
                if coll.get("frame_accepted"):
                    n_accepted += 1
                print(f"    Sent frame {n_sent}, accepted: {coll.get('n_frames', '?')}")

    else:
        # Manual / Teleop: interactive capture with key press
        async with websockets.connect(ws_url(base), max_size=2**24) as ws:
            print("\n  Controls:")
            print("    SPACE / ENTER  = capture frame")
            print("    R              = stop capturing & reconstruct")
            print("    Q / ESC        = quit without reconstructing\n")

            while True:
                # Teleop: relay leader → follower
                if leader is not None:
                    action = leader.get_action()
                    robot.send_action(action)

                obs = robot.get_observation()
                frame = obs.get("wrist_cam")
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Show live preview
                display = frame.copy()
                elapsed = time.time() - t_start
                cv2.putText(display,
                            f"Sent: {n_sent}  Accepted: {n_accepted}  "
                            f"Time: {elapsed:.0f}s  [SPACE=snap R=reconstruct Q=quit]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("Active Capture", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # quit
                    print("\n  Cancelled — no reconstruction.")
                    api(base, "/session/stop", "POST")
                    cv2.destroyAllWindows()
                    return
                if key == ord("r"):  # reconstruct
                    break
                if key not in (ord(" "), 13):  # not SPACE/ENTER → skip capture
                    continue

                # Capture this frame
                masked = _process_frame(frame, args)
                joint_deg = read_joint_positions(robot)
                pose = joint_to_pose7(joint_deg)

                ok, encoded = cv2.imencode(".jpg", masked, jpeg_params)
                if not ok:
                    continue

                msg = encode_message(encoded.tobytes(), pose)
                await ws.send(msg)
                resp_raw = await ws.recv()
                payload = orjson.loads(resp_raw)
                n_sent += 1

                coll = payload.get("collection", {})
                if coll.get("frame_accepted"):
                    n_accepted += 1
                print(f"    Captured #{n_sent}  → server accepted: "
                      f"{coll.get('n_frames', '?')} total")

        cv2.destroyAllWindows()

    # ---- 3. Stop session ----
    resp = api(base, "/session/stop", "POST")
    n_stored = resp.get("n_frames", 0)
    print(f"\n[session] Stopped — {n_stored} frames stored on server")

    if n_stored < 2:
        print("  Not enough frames for reconstruction (need ≥ 2). Try again.")
        return

    # ---- 4. Trigger reconstruction ----
    print("[reconstruct] Starting VGGT reconstruction + occupancy grid ...")
    resp = api(base, "/reconstruct", "POST", params={"session_id": sid})
    print(f"  Processing {resp.get('n_frames', '?')} frames ...")

    # ---- 5. Poll until done ----
    while True:
        await asyncio.sleep(2.0)
        status = api(base, "/reconstruct/status")
        if not status["running"]:
            if status.get("error"):
                print(f"\n  ERROR: {status['error']}")
                return
            break
        print("  ... reconstructing ...", flush=True)

    # ---- 6. Show results ----
    result = api(base, "/reconstruct/result")
    print_reconstruction_result(result)

    plan = api(base, "/plan")
    views = plan.get("viewpoints", [])
    print_plan(views)


# ======================================================================
# NBV-driven automatic exploration
# ======================================================================

def _precompute_fk_positions(candidates: list[list[float]]) -> np.ndarray:
    """Return (N, 3) world-space camera positions using FK for each candidate."""
    if not _HAS_FK:
        return np.zeros((len(candidates), 3))
    positions = []
    for vp in candidates:
        T = ee_pose(np.array(vp, dtype=float))  # 4x4 camera-to-world
        positions.append(T[:3, 3])
    return np.array(positions)


def _precompute_fk_data(
    candidates: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, 3) camera positions and (N, 3) viewing directions via FK.

    The viewing direction is the camera z-axis in world frame (OpenCV
    convention: z = forward/optical axis).
    """
    if not _HAS_FK:
        pos = np.zeros((len(candidates), 3))
        dirs = np.tile([0.0, 0.0, 1.0], (len(candidates), 1))
        return pos, dirs
    positions = []
    directions = []
    for vp in candidates:
        T_c2w = ee_pose(np.array(vp, dtype=float))  # 4x4 camera-to-world
        positions.append(T_c2w[:3, 3])
        directions.append(T_c2w[:3, 2])  # z-axis = viewing direction
    return np.array(positions), np.array(directions)


def _farthest_point_seed(positions: np.ndarray, k: int) -> list[int]:
    """Select *k* indices via farthest-point sampling for max spatial spread."""
    n = len(positions)
    if k >= n:
        return list(range(n))
    selected = [0]
    min_dists = np.full(n, np.inf)
    for _ in range(k - 1):
        last = positions[selected[-1]]
        dists = np.linalg.norm(positions - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))
    return selected


def _score_information_gain(
    cam_positions: np.ndarray,      # (N, 3)
    cam_directions: np.ndarray,     # (N, 3) unit viewing dirs
    occupancy: np.ndarray,          # (G, G, G) uint8
    grid_origin: np.ndarray,        # (3,)
    voxel_size: float,
    unvisited: list[int],
    fov_half_angle: float = 0.52,   # ~30° half-angle ≈ 60° full FOV
    max_range: float = 0.5,         # max viewing distance in metres
    scene_center: np.ndarray | None = None,
) -> np.ndarray:
    """Score each unvisited candidate by UNKNOWN voxels in its view frustum.

    Uses a simplified cone test: a voxel is "visible" if it falls within
    *fov_half_angle* of the camera's viewing direction and is closer than
    *max_range*.

    When *scene_center* is provided, viewpoints closer to the scene centre
    receive a proximity bonus (1 / (1 + dist)).
    """
    unk_ijk = np.argwhere(occupancy == UNKNOWN)
    if len(unk_ijk) == 0:
        return np.zeros(len(unvisited))

    unk_world = grid_origin + (unk_ijk + 0.5) * voxel_size  # (M, 3)

    # Subsample for speed
    if len(unk_world) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(unk_world), 5000, replace=False)
        unk_world = unk_world[idx]

    cos_thresh = np.cos(fov_half_angle)
    scores = np.zeros(len(unvisited), dtype=np.float64)

    for i, vi in enumerate(unvisited):
        pos = cam_positions[vi]
        d = cam_directions[vi]
        to_voxel = unk_world - pos                           # (M, 3)
        dists = np.linalg.norm(to_voxel, axis=1)
        valid = dists > 1e-6
        to_voxel_norm = np.zeros_like(to_voxel)
        to_voxel_norm[valid] = to_voxel[valid] / dists[valid, np.newaxis]
        cos_angles = to_voxel_norm @ d
        in_view = valid & (cos_angles > cos_thresh) & (dists < max_range)
        scores[i] = float(in_view.sum())

    # Proximity bonus: viewpoints closer to scene centre score higher
    if scene_center is not None:
        for i, vi in enumerate(unvisited):
            d_cam = np.linalg.norm(cam_positions[vi] - scene_center)
            scores[i] *= 1.0 / (1.0 + d_cam)

    return scores


def _tsp_nearest_neighbor(
    vp_indices: list[int],
    candidates: list[list[float]],
    start_joints: np.ndarray | None = None,
) -> list[int]:
    """Reorder viewpoint indices for shortest joint-space travel.

    Uses nearest-neighbour greedy heuristic starting from the configuration
    closest to *start_joints* (or the first in the list).
    """
    if len(vp_indices) <= 2:
        return list(vp_indices)

    joint_configs = np.array([candidates[i] for i in vp_indices], dtype=np.float64)
    n = len(vp_indices)

    if start_joints is not None:
        dists_to_start = np.linalg.norm(joint_configs - start_joints, axis=1)
        current = int(np.argmin(dists_to_start))
    else:
        current = 0

    visited_set: set[int] = {current}
    order = [current]

    for _ in range(n - 1):
        dists = np.linalg.norm(joint_configs - joint_configs[current], axis=1)
        dists[list(visited_set)] = np.inf
        nearest = int(np.argmin(dists))
        visited_set.add(nearest)
        order.append(nearest)
        current = nearest

    return [vp_indices[i] for i in order]


def _select_next_viewpoints(
    views: list[dict],
    cam_positions: np.ndarray,
    cam_directions: np.ndarray,
    occupancy: np.ndarray | None,
    grid_origin: np.ndarray | None,
    voxel_size: float,
    unvisited: list[int],
    candidates: list[list[float]],
    current_joints: np.ndarray | None,
    n_select: int = 3,
    scene_center: np.ndarray | None = None,
) -> list[int]:
    """Smart viewpoint selection: information-gain + NBV matching + TSP.

    Strategy
    --------
    1. Score every unvisited candidate by information gain (UNKNOWN voxels
       visible in its view frustum).
    2. Give candidates matched by the server's NBV plan a 50 % score bonus
       (using combined position + orientation distance).
    3. Pick the top *n_select* by score.
    4. Reorder via nearest-neighbour TSP for shortest joint-space travel.
    """
    if not unvisited:
        return []

    # ── 1. Information-gain scores ─────────────────────────────────
    if (occupancy is not None and grid_origin is not None
            and voxel_size > 0 and _HAS_FK):
        ig_scores = _score_information_gain(
            cam_positions, cam_directions,
            occupancy, grid_origin, voxel_size,
            unvisited,
            scene_center=scene_center,
        )
    else:
        ig_scores = np.ones(len(unvisited))  # uniform fallback

    # ── 2. NBV matching bonus (position + orientation) ─────────────
    nbv_matched: set[int] = set()
    if views and _HAS_FK:
        unvisited_pos = cam_positions[unvisited]
        unvisited_dir = cam_directions[unvisited]
        chosen_local: set[int] = set()
        for vp_cmd in views:
            nb = vp_cmd["pose"]
            nbv_pos = np.array([nb["x"], nb["y"], nb["z"]])
            tgt = np.array(vp_cmd.get("target_world", [0.0, 0.0, 0.0]))
            nbv_dir = tgt - nbv_pos
            nbv_dir_norm = np.linalg.norm(nbv_dir)
            if nbv_dir_norm > 1e-6:
                nbv_dir = nbv_dir / nbv_dir_norm
            else:
                nbv_dir = np.array([0.0, 0.0, 1.0])

            # Combined position + orientation distance
            pos_dists = np.linalg.norm(unvisited_pos - nbv_pos, axis=1)
            pos_max = pos_dists.max() + 1e-8
            pos_norm = pos_dists / pos_max
            cos_angles = np.clip(unvisited_dir @ nbv_dir, -1, 1)
            ang_dists = 1.0 - cos_angles
            combined = 0.6 * pos_norm + 0.4 * ang_dists

            for local_idx in np.argsort(combined):
                local_idx = int(local_idx)
                if local_idx not in chosen_local:
                    chosen_local.add(local_idx)
                    nbv_matched.add(unvisited[local_idx])
                    break

    # ── 3. Combine scores and select top-k ─────────────────────────
    scored: list[tuple[int, float]] = []
    for j, vi in enumerate(unvisited):
        score = ig_scores[j]
        if vi in nbv_matched:
            score *= 1.5  # 50 % bonus for NBV-matched candidates
        scored.append((vi, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in scored[:n_select]]

    # ── 4. TSP reorder ─────────────────────────────────────────────
    selected = _tsp_nearest_neighbor(selected, candidates, current_joints)
    logger.info("Selected %d viewpoints (info-gain + NBV + TSP): %s",
                len(selected), selected)
    return selected


async def _capture_batch(
    ws,
    robot: "SOFollower",
    vp_indices: list[int],
    candidates: list[list[float]],
    args: argparse.Namespace,
    visited: set[int],
) -> tuple[int, int]:  # (n_sent, n_accepted)
    """Move arm to each viewpoint in vp_indices, capture one frame each."""
    import orjson
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    n_sent = n_accepted = 0
    for vi in vp_indices:
        vp = candidates[vi]
        print(f"  → VP {vi:2d} ({vp_indices.index(vi)+1}/{len(vp_indices)}): {vp}")
        move_arm(robot, vp)
        visited.add(vi)

        obs = robot.get_observation()
        frame = obs.get("wrist_cam")
        if frame is None:
            logger.warning("    No frame from wrist_cam — skipping.")
            continue

        frame = _process_frame(frame, args)
        joint_deg = read_joint_positions(robot)
        pose = joint_to_pose7(joint_deg)

        ok, encoded = cv2.imencode(".jpg", frame, jpeg_params)
        if not ok:
            continue

        await ws.send(encode_message(encoded.tobytes(), pose))
        payload = orjson.loads(await ws.recv())
        n_sent += 1
        coll = payload.get("collection", {})
        if coll.get("frame_accepted"):
            n_accepted += 1
            print(f"    Accepted  (total in session: {coll.get('n_frames', '?')})")
        else:
            print(f"    Skipped   (too close to previous)")
    return n_sent, n_accepted


async def _reconstruct_and_plan(base: str, sid: str,
                                merge_all: bool = False) -> tuple[dict, list[dict]]:
    """Block until reconstruction finishes; return (result_json, viewpoints).

    When *merge_all* is True the server merges frames from **every**
    historical session so the reconstruction improves cumulatively.
    """
    print("[reconstruct] Running VGGT + occupancy grid "
          f"({'ALL sessions merged' if merge_all else 'current session only'}) ...")
    api(base, "/reconstruct", "POST",
        params={"session_id": sid, "merge_all": merge_all})
    while True:
        await asyncio.sleep(2.0)
        status = api(base, "/reconstruct/status")
        if not status["running"]:
            if status.get("error"):
                raise RuntimeError(status["error"])
            break
        print("  ... reconstructing ...", flush=True)
    result = api(base, "/reconstruct/result")
    plan   = api(base, "/plan")
    return result, plan.get("viewpoints", [])


async def auto_nbv_explore(
    args: argparse.Namespace,
    robot: "SOFollower",
) -> None:
    """Fully automatic NBV-driven exploration loop.

    Round 0 — seed:  capture ``--seed-views`` frames from the start of the
               candidate bank, then reconstruct.
    Round k — NBV:   match the server's next-best-view suggestions to the
               closest unvisited candidates, move there, capture, reconstruct.
    Stops when:
      • scene ``explored_fraction`` ≥ ``--stop-explored``  (default 0.90), or
      • all candidates have been visited, or
      • ``--max-rounds`` rounds have passed.
    """
    import websockets
    base = args.server

    # ── Live viser viewer (optional) ──────────────────────────────────
    live_vis: LiveVisualizer | None = None
    if getattr(args, "live_viser", False):
        live_vis = LiveVisualizer(port=getattr(args, "viser_port", 8080))
        live_vis.start()

    # ── Initialise scene bounds from a forward-looking reference pose ─────
    # We use the first DEFAULT_VIEWPOINT (a known good forward-looking
    # pose) rather than the rest position, which points at the ceiling.
    scene_center: np.ndarray | None = None
    scene_forward: np.ndarray | None = None
    REFERENCE_VIEWPOINT = DEFAULT_VIEWPOINTS[0]  # [0, -30, 60, -30, 0, 50]

    if _HAS_FK:
        # Move arm to the reference viewpoint first
        print("[scene] Moving to reference viewpoint for scene initialisation...")
        move_arm(robot, REFERENCE_VIEWPOINT)
        time.sleep(0.5)

        ref_joints = np.array(REFERENCE_VIEWPOINT, dtype=float)
        T_c2w = ee_pose(ref_joints)  # 4×4 camera-to-world
        cam_pos = T_c2w[:3, 3].copy()
        # The FK z-axis points *away* from the scene (robotics convention).
        # Negate it to get the optical axis pointing *into* the scene.
        cam_fwd = -T_c2w[:3, 2].copy()   # negated z-axis = into scene
        cam_fwd /= np.linalg.norm(cam_fwd) + 1e-12

        # Place the scene centre well in front of the camera, where the
        # object actually is.
        SCENE_OFFSET_M = 0.25   # 25 cm along the viewing direction
        scene_center = cam_pos + cam_fwd * SCENE_OFFSET_M
        scene_forward = cam_fwd.copy()

        half_ext = getattr(args, "scene_extent", 2.4)
        bounds_type = getattr(args, "bounds_type", "hemisphere")
        print(f"[scene] Initialising fixed scene bounds:")
        print(f"  cam_pos = {cam_pos.tolist()}")
        print(f"  center  = {scene_center.tolist()}  (offset {SCENE_OFFSET_M}m forward)")
        print(f"  forward = {scene_forward.tolist()}")
        print(f"  extent  = {half_ext} m  ({bounds_type})")

        try:
            init_url = base.rstrip("/") + "/scene/init"
            r = requests.post(init_url, timeout=30, json={
                "center": scene_center.tolist(),
                "forward": scene_forward.tolist(),
                "half_extent": half_ext,
                "bounds_type": bounds_type,
            })
            r.raise_for_status()
            print(f"  Server response: {r.json()}")
        except Exception as e:
            print(f"  WARNING: /scene/init failed: {e}")
    else:
        print("[scene] FK not available — using dynamic bounds (no fixed scene)")

    # ── Capture reference frame at the forward-looking pose ───────────
    # This first frame anchors the VGGT reconstruction with a good view
    # of the scene (instead of the folded rest position).
    print("[reference] Capturing reference frame at forward-looking position ...")
    try:
        resp = api(base, "/session/start", "POST",
                   params={"max_frames": 5, "min_baseline_m": 0.0})
        ref_sid = resp["session_id"]
        async with websockets.connect(ws_url(base), max_size=2**24) as ws:
            import orjson as _orjson
            obs = robot.get_observation()
            frame = obs.get("wrist_cam")
            if frame is not None:
                frame = _process_frame(frame, args)
                joint_deg = read_joint_positions(robot)
                pose = joint_to_pose7(joint_deg)
                jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
                ok, encoded = cv2.imencode(".jpg", frame, jpeg_params)
                if ok:
                    await ws.send(encode_message(encoded.tobytes(), pose))
                    _payload = _orjson.loads(await ws.recv())
                    coll = _payload.get("collection", {})
                    print(f"  Reference frame: {'accepted' if coll.get('frame_accepted') else 'skipped'}")
        api(base, "/session/stop", "POST")
        print(f"  Reference session {ref_sid} stored")
    except Exception as e:
        print(f"  WARNING: reference frame capture failed: {e}")

    # Build candidate bank — dense sampling for better exploration coverage
    if args.viewpoints_json:
        with open(args.viewpoints_json) as f:
            candidates: list[list[float]] = json.load(f)
    else:
        bank_size = getattr(args, "bank_size", 200)
        candidates = generate_viewpoint_bank(
            n=bank_size, include_defaults=True)
    print(f"Candidate viewpoint bank: {len(candidates)} viewpoints")

    cam_positions, cam_directions = _precompute_fk_data(candidates)

    visited: set[int] = set()
    round_n = 0

    # Seed batch: use the first N DEFAULT_VIEWPOINTS (forward-looking)
    # rather than FPS (which picks extreme positions all over the workspace).
    n_defaults = len(DEFAULT_VIEWPOINTS)
    seed_count = min(args.seed_views, len(candidates))
    if seed_count <= n_defaults:
        # All seed views come from the hand-tuned forward-looking defaults
        next_batch = list(range(seed_count))
    else:
        # Start with all defaults, then FPS fill from the rest
        next_batch = list(range(n_defaults))
        remaining = seed_count - n_defaults
        other_indices = [i for i in range(n_defaults, len(candidates))]
        if _HAS_FK and remaining > 0 and len(other_indices) > remaining:
            extra = _farthest_point_seed(
                cam_positions[other_indices], remaining)
            next_batch.extend([other_indices[e] for e in extra])
        else:
            next_batch.extend(other_indices[:remaining])

    while round_n < args.max_rounds and next_batch:
        round_n += 1
        unvisited_count = len(candidates) - len(visited)
        print(f"\n{'='*60}")
        print(f"  AUTO-EXPLORE  round={round_n}  visited={len(visited)}/{len(candidates)}")
        print(f"  Capturing viewpoints: {next_batch}")
        print(f"{'='*60}")

        # ── Start session ──────────────────────────────────────────────
        resp = api(base, "/session/start", "POST",
                   params={"max_frames": args.max_frames,
                           "min_baseline_m": args.min_baseline})
        sid = resp["session_id"]
        print(f"[session] {sid}")

        async with websockets.connect(ws_url(base), max_size=2**24) as ws:
            n_sent, n_accepted = await _capture_batch(
                ws, robot, next_batch, candidates, args, visited)

        resp = api(base, "/session/stop", "POST")
        n_stored = resp.get("n_frames", 0)
        print(f"[session] stopped — {n_stored} frames stored  "
              f"(sent={n_sent}, accepted={n_accepted})")

        if n_stored < 2:
            print("  Not enough accepted frames — jumping to next unvisited batch.")
            unvisited = [i for i in range(len(candidates)) if i not in visited]
            next_batch = unvisited[: args.seed_views]
            continue

        # ── Reconstruct (merge ALL sessions for cumulative improvement) ──
        try:
            result, views = await _reconstruct_and_plan(base, sid,
                                                        merge_all=True)
        except RuntimeError as e:
            print(f"  Reconstruction error: {e}")
            break

        print_reconstruction_result(result)
        print_plan(views)

        # ── Update live viser viewer ────────────────────────────────────
        if live_vis is not None:
            print("  Updating live viser viewer ...")
            live_vis.update(base, round_n=round_n)

        # ── Check stop condition ────────────────────────────────────────
        explored = result.get("occupancy", {}).get("explored_fraction", 0)
        if explored >= args.stop_explored:
            print(f"\n  Scene {explored:.1%} explored — reached "
                  f"target {args.stop_explored:.1%}. Done.")
            break

        unvisited = [i for i in range(len(candidates)) if i not in visited]
        if not unvisited:
            print("  All candidate viewpoints visited.")
            break

        # ── Select next batch (info-gain + NBV + TSP) ─────────────────
        grid, grid_origin, v_size = _fetch_occupancy(base)
        current_joints = read_joint_positions(robot)
        next_batch = _select_next_viewpoints(
            views=views,
            cam_positions=cam_positions,
            cam_directions=cam_directions,
            occupancy=grid,
            grid_origin=grid_origin,
            voxel_size=v_size,
            unvisited=unvisited,
            candidates=candidates,
            current_joints=current_joints,
            n_select=args.views_per_round,
            scene_center=scene_center,
        )
        print(f"  Selected next batch: {next_batch} "
              f"(info-gain + NBV + TSP from {len(unvisited)} candidates)")

    print(f"\nAuto-explore finished: {len(visited)}/{len(candidates)} viewpoints "
          f"visited in {round_n} round(s).")


# ======================================================================
# Iterative exploration loop  (manual / teleop modes)

async def exploration_loop(args: argparse.Namespace) -> None:
    """Multi-round capture → reconstruct → move-to-NBV loop."""
    if not _HAS_LEROBOT:
        print("ERROR: lerobot is not installed.")
        print("Activate your lerobot conda environment and try again.")
        sys.exit(1)

    # Connect robot
    cam_cfg = OpenCVCameraConfig(
        index_or_path=args.camera_id, fps=30,
        width=args.width, height=args.height,
    )
    robot_cfg = SOFollowerRobotConfig(
        port=args.port, id=args.robot_id,
        cameras={"wrist_cam": cam_cfg}, use_degrees=True,
    )
    robot = SOFollower(robot_cfg)

    leader = None
    if args.mode == "teleop":
        if not args.teleop_port:
            print("ERROR: --teleop-port required for teleop mode.")
            sys.exit(1)
        teleop_cfg = SO101LeaderConfig(
            port=args.teleop_port, id=args.teleop_id, use_degrees=True,
        )
        leader = SO101Leader(teleop_cfg)

    print(f"Connecting follower on {args.port} ...")
    robot.connect(calibrate=True)
    if leader:
        print(f"Connecting leader on {args.teleop_port} ...")
        leader.connect(calibrate=True)
    print("Connected.\n")

    # Check server health
    try:
        h = api(args.server, "/health")
        print(f"Server OK: device={h['device']}\n")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.server}: {e}")
        robot.disconnect()
        if leader:
            leader.disconnect()
        sys.exit(1)

    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"  EXPLORATION ITERATION {iteration}")
            print(f"{'='*60}")

            await capture_and_reconstruct(args, robot, leader)

            # Ask user whether to continue
            print("\n  Continue exploring? [y/n] ", end="", flush=True)
            choice = input().strip().lower()
            if choice not in ("y", "yes", ""):
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        return_to_rest(robot)
        if leader:
            leader.disconnect()
        robot.disconnect()
        cv2.destroyAllWindows()
        print("Disconnected.")


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Active capture: SO-101 arm → GPU server → occupancy grid → NBV exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Server
    p.add_argument("--server", required=True,
                   help="GPU server URL, e.g. http://192.168.1.100:8765")

    # Capture mode
    p.add_argument("--mode", choices=["teleop", "manual", "auto"], default="teleop")

    # Robot
    p.add_argument("--port", default="/dev/ttyACM1",
                   help="Follower arm serial port")
    p.add_argument("--robot-id", default="kowalski_follower")
    p.add_argument("--teleop-port", default="/dev/ttyACM0",
                   help="Leader arm serial port (teleop mode)")
    p.add_argument("--teleop-id", default="shrek_leader")

    # Camera
    p.add_argument("--camera-id", type=int, default=2)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--mask-gripper", action="store_true",
                   help="Use polygon mask on gripper instead of cropping bottom 1/3 (default: crop)")

    # Collection
    p.add_argument("--max-frames", type=int, default=50,
                   help="Max frames per session")
    p.add_argument("--min-baseline", type=float, default=0.02,
                   help="Min translation between accepted frames (m)")

    # Auto-NBV mode
    p.add_argument("--seed-views", type=int, default=3,
                   help="Number of initial viewpoints before first reconstruction (auto mode)")
    p.add_argument("--views-per-round", type=int, default=3,
                   help="NBV candidates to visit per reconstruction round (auto mode)")
    p.add_argument("--max-rounds", type=int, default=8,
                   help="Maximum exploration rounds (auto mode)")
    p.add_argument("--stop-explored", type=float, default=0.90,
                   help="Stop when this fraction of voxels is explored (auto mode, 0–1)")

    # Viewpoints bank
    p.add_argument("--bank-size", type=int, default=200,
                   help="Dense viewpoint bank size for auto exploration (default: 200)")
    p.add_argument("--viewpoints-json", default=None,
                   help="Custom viewpoints file (auto mode, overrides --bank-size)")

    # Scene bounds
    p.add_argument("--scene-extent", type=float, default=2.4,
                   help="Half side-length (cube) or radius (hemisphere) of the fixed "
                        "scene bounding volume in metres (default: 2.4)")
    p.add_argument("--bounds-type", choices=["hemisphere", "cube"],
                   default="hemisphere",
                   help="Type of fixed bounding volume: 'hemisphere' (default, "
                        "forward half only) or 'cube' (full axis-aligned box)")

    # Live viser visualization
    p.add_argument("--live-viser", action="store_true",
                   help="Open a live viser 3-D viewer that updates after each reconstruction round (auto mode)")
    p.add_argument("--viser-port", type=int, default=8080,
                   help="Port for the live viser server (default: 8080)")

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.mode == "auto":
        # Fully automatic NBV-driven exploration
        if not _HAS_LEROBOT:
            print("ERROR: lerobot is not installed.")
            sys.exit(1)
        cam_cfg = OpenCVCameraConfig(
            index_or_path=args.camera_id, fps=30,
            width=args.width, height=args.height,
        )
        robot_cfg = SOFollowerRobotConfig(
            port=args.port, id=args.robot_id,
            cameras={"wrist_cam": cam_cfg}, use_degrees=True,
        )
        robot = SOFollower(robot_cfg)
        print(f"Connecting follower on {args.port} ...")
        robot.connect(calibrate=True)
        try:
            health = api(args.server, "/health")
            print(f"Server OK: device={health['device']}")
        except Exception as e:
            print(f"ERROR: Cannot reach server: {e}")
            robot.disconnect()
            sys.exit(1)
        try:
            asyncio.run(auto_nbv_explore(args, robot))
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            return_to_rest(robot)
            robot.disconnect()
    else:
        asyncio.run(exploration_loop(args))


if __name__ == "__main__":
    main()
