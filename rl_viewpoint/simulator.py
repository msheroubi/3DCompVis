"""
Analytic visibility simulator for RL training — no rendering needed.

Generates random scenes of primitive objects (boxes, spheres, cylinders)
inside a voxelised bounding volume, then simulates camera observations
by ray-marching through the voxel grid.  Occlusion is handled naturally:
rays stop when they hit an OCCUPIED voxel.

Camera model
------------
The SO-101 wrist camera is **mounted to the left of the gripper** and
angled slightly to the right so its optical axis converges with the
gripper's pointing direction.  We model this as a fixed rigid transform
from the FK end-effector frame to the camera frame::

    T_cam_from_ee = [R_offset | t_offset]

where *t_offset* shifts ~2 cm left (−Y in EE frame) and *R_offset*
rotates ~5° rightward (positive yaw around the EE's Z-axis).

The FK function ``ee_pose(joints)`` returns a 4×4 camera-to-world
that already includes this offset, so we use it directly.  The
simulator mirrors that convention: the provided ``cam_pos`` is the
camera, the ``cam_dir`` is the **negated** FK z-axis (= optical axis
into the scene).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .utils import UNKNOWN, FREE, OCCUPIED

logger = logging.getLogger(__name__)


# =====================================================================
# Camera-from-EE offset  (for pure-simulation FK substitute)
# =====================================================================
# In the real robot ``ee_pose()`` already bakes this in.
# We only use these constants in the *simulated FK* path (when
# so101_kinematics is unavailable and we generate poses synthetically).

# Camera is ~2 cm to the left, ~0.5 cm forward, 0 cm up relative to
# the gripper's tool centre point.
CAM_OFFSET_EE = np.array([0.0, -0.02, -0.005], dtype=np.float64)

# Inward yaw of ~5° (camera looks slightly right towards gripper axis)
CAM_YAW_RAD = np.deg2rad(5.0)
_cy, _sy = math.cos(CAM_YAW_RAD), math.sin(CAM_YAW_RAD)
CAM_ROT_FROM_EE = np.array([
    [ _cy, _sy, 0.0],
    [-_sy, _cy, 0.0],
    [ 0.0, 0.0, 1.0],
], dtype=np.float64)


# =====================================================================
# Primitive shapes
# =====================================================================

@dataclass
class Primitive:
    """Base class for a voxelisable shape."""
    center: np.ndarray   # (3,)  world metres
    def contains(self, pts: np.ndarray) -> np.ndarray:
        """Return bool mask for which of pts (N,3) are inside."""
        raise NotImplementedError


@dataclass
class Box(Primitive):
    half_extents: np.ndarray = field(default_factory=lambda: np.array([0.03, 0.03, 0.03]))

    def contains(self, pts: np.ndarray) -> np.ndarray:
        d = np.abs(pts - self.center)
        return np.all(d <= self.half_extents, axis=1)


@dataclass
class Sphere(Primitive):
    radius: float = 0.04

    def contains(self, pts: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pts - self.center, axis=1) <= self.radius


@dataclass
class Cylinder(Primitive):
    """Axis-aligned cylinder (axis = Z by default)."""
    radius: float = 0.03
    half_height: float = 0.05
    axis: int = 2    # 0=X, 1=Y, 2=Z

    def contains(self, pts: np.ndarray) -> np.ndarray:
        d = pts - self.center
        # Axes perpendicular to the cylinder axis
        perp = [i for i in range(3) if i != self.axis]
        r2 = d[:, perp[0]] ** 2 + d[:, perp[1]] ** 2
        along = np.abs(d[:, self.axis])
        return (r2 <= self.radius ** 2) & (along <= self.half_height)


# =====================================================================
# Scene generation
# =====================================================================

def generate_random_scene(
    rng: np.random.Generator,
    grid_res: int = 64,
    half_extent: float = 0.3,
    scene_center: np.ndarray | None = None,
    n_objects: tuple[int, int] = (3, 8),
    size_range: tuple[float, float] = (0.02, 0.08),
) -> tuple[np.ndarray, np.ndarray, float, list[Primitive]]:
    """Create a ground-truth occupancy grid with random primitives.

    Parameters
    ----------
    rng : numpy random Generator
    grid_res : int
        Voxel grid resolution per axis (default 64).
    half_extent : float
        Half side-length of the cubic scene volume (metres).
    scene_center : (3,) or None
        Centre of the scene volume.  Defaults to (0.15, 0.0, 0.0) —
        roughly where objects sit in front of the SO-101.
    n_objects : (min, max)
        Uniform-random number of primitives to place.
    size_range : (min, max)
        Range for random half-extents / radii (metres).

    Returns
    -------
    ground_truth : (G, G, G) uint8 — OCCUPIED where objects are,
                   FREE everywhere else.
    grid_origin  : (3,) world coordinate of voxel [0,0,0] corner.
    voxel_size   : float, metres per voxel.
    objects      : list of Primitive instances placed in the scene.
    """
    if scene_center is None:
        scene_center = np.array([0.15, 0.0, 0.0], dtype=np.float64)
    else:
        scene_center = np.asarray(scene_center, dtype=np.float64)

    grid_origin = scene_center - half_extent
    voxel_size = (2.0 * half_extent) / grid_res

    # Voxel centre coordinates
    idx = np.arange(grid_res)
    ix, iy, iz = np.meshgrid(idx, idx, idx, indexing="ij")
    # (G³, 3) world coords of voxel centres
    voxel_centres = grid_origin + (np.stack([ix, iy, iz], axis=-1).reshape(-1, 3) + 0.5) * voxel_size

    n = rng.integers(n_objects[0], n_objects[1] + 1)
    objects: list[Primitive] = []
    occupied_mask = np.zeros(len(voxel_centres), dtype=bool)

    shape_types = [Box, Sphere, Cylinder]

    for _ in range(n):
        # Random position within the inner 60 % of the volume
        inner = half_extent * 0.6
        pos = scene_center + rng.uniform(-inner, inner, size=3)
        s = rng.uniform(size_range[0], size_range[1])

        shape_cls = rng.choice(shape_types)
        if shape_cls is Box:
            he = np.array([s, s, s]) * rng.uniform(0.5, 1.5, size=3)
            obj = Box(center=pos.copy(), half_extents=he)
        elif shape_cls is Sphere:
            obj = Sphere(center=pos.copy(), radius=s)
        else:
            axis = int(rng.integers(0, 3))
            obj = Cylinder(center=pos.copy(), radius=s * 0.7,
                           half_height=s * 1.2, axis=axis)
        objects.append(obj)
        occupied_mask |= obj.contains(voxel_centres)

    # Build grid: everything starts FREE; objects are OCCUPIED
    gt = np.full(grid_res ** 3, FREE, dtype=np.uint8)
    gt[occupied_mask] = OCCUPIED
    gt = gt.reshape(grid_res, grid_res, grid_res)

    # ── Floor plane ────────────────────────────────────────────────
    # Mark the bottom slice(s) as OCCUPIED so the agent learns that
    # looking *through* the table yields no information, and the
    # planner never sends the arm below the surface.
    # Convention: Z-axis is the 3rd grid axis (index 2).
    # We fill the lowest ``floor_thickness`` voxel layers.
    floor_thickness = max(1, grid_res // 32)      # 1–2 voxels
    gt[:, :, :floor_thickness] = OCCUPIED

    logger.info("Generated scene: %d objects, %d/%d voxels occupied (%.1f%%) "
                "[incl. floor %d layers]",
                n, (gt == OCCUPIED).sum(), gt.size,
                100.0 * (gt == OCCUPIED).sum() / gt.size,
                floor_thickness)
    return gt, grid_origin.astype(np.float32), float(voxel_size), objects


# =====================================================================
# Visibility simulator (DDA ray-casting with occlusion)
# =====================================================================

class VisibilitySimulator:
    """Simulate what a camera sees via DDA ray-casting against a
    ground-truth occupancy grid.

    The camera is modelled as an SO-101 wrist camera:
      • Offset slightly left of the EE
      • Angled ~5° rightward
      • 60° horizontal FOV, 47° vertical FOV  (640×480 webcam, fx≈554)
      • 640×480 image → we cast rays at every 4th pixel (130×98)

    After construction, call :meth:`observe_from` to reveal voxels from
    a given camera pose.  Intermediate voxels along each ray become FREE
    (empty space); the first OCCUPIED voxel hit becomes OCCUPIED in the
    observed grid (and the ray stops — natural occlusion).
    """

    def __init__(
        self,
        ground_truth: np.ndarray,          # (G, G, G) uint8
        grid_origin: np.ndarray,           # (3,)
        voxel_size: float,
        hfov_deg: float = 60.0,           # horizontal FOV  (real webcam ~60°)
        vfov_deg: float = 47.0,           # vertical FOV    (real webcam ~47°)
        max_range: float = 0.5,           # metres (matches scoring fn)
        ray_w: int = 130,                 # 518/4 ≈ 130  (stride-4 on VGGT W)
        ray_h: int = 98,                  # 392/4 =  98  (stride-4 on VGGT H)
    ):
        self.gt = ground_truth.copy()
        self.grid_origin = np.asarray(grid_origin, dtype=np.float64)
        self.voxel_size = float(voxel_size)
        self.G = ground_truth.shape[0]
        self.max_range = max_range

        # Build normalised ray direction templates in *camera* frame
        # Camera convention: X-right, Y-down, Z-forward (OpenCV style).
        hfov = np.deg2rad(hfov_deg)
        vfov = np.deg2rad(vfov_deg)
        u = np.linspace(-np.tan(hfov / 2), np.tan(hfov / 2), ray_w)
        v = np.linspace(-np.tan(vfov / 2), np.tan(vfov / 2), ray_h)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        # Each ray: (x_right, y_down, z_forward=1)
        rays_cam = np.stack([uu.ravel(), vv.ravel(),
                             np.ones(ray_w * ray_h)], axis=-1)
        # Normalise
        rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
        self._rays_cam = rays_cam.astype(np.float64)  # (N_rays, 3)
        self._n_rays = len(rays_cam)

        # The "observed" grid starts all UNKNOWN
        self.observed = np.full_like(ground_truth, UNKNOWN)

    def reset(self) -> None:
        """Clear all observations (start a new episode)."""
        self.observed = np.full_like(self.gt, UNKNOWN)

    @property
    def explored_fraction(self) -> float:
        known = np.count_nonzero(self.observed != UNKNOWN)
        return known / self.observed.size

    def observe_from(
        self,
        cam_pos: np.ndarray,     # (3,)  world metres
        cam_dir: np.ndarray,     # (3,)  forward viewing direction (unit)
        cam_up: np.ndarray | None = None,  # (3,)  optional up vector
    ) -> float:
        """Cast rays from the given pose and reveal voxels in ``self.observed``.

        Parameters
        ----------
        cam_pos : (3,)
            Camera position in world frame.
        cam_dir : (3,)
            Forward viewing direction (unit vector).  This is the optical
            axis — the negated FK z-axis for the SO-101.
        cam_up : (3,) or None
            Up direction in world frame.  If None, uses world-Y as up
            (with fallback to world-Z if cam_dir ≈ Y).

        Returns
        -------
        float
            Δ explored_fraction (increase from this observation).
        """
        explored_before = self.explored_fraction
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        cam_dir = np.asarray(cam_dir, dtype=np.float64)
        cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)

        # Build camera-to-world rotation (OpenCV: X-right, Y-down, Z-fwd)
        if cam_up is None:
            world_up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(cam_dir, world_up)) > 0.95:
                world_up = np.array([0.0, 0.0, 1.0])
        else:
            world_up = np.asarray(cam_up, dtype=np.float64)

        z_cam = cam_dir                                     # forward
        x_cam = np.cross(z_cam, world_up)
        x_cam /= np.linalg.norm(x_cam) + 1e-12             # right
        y_cam = np.cross(z_cam, x_cam)                      # down
        R_c2w = np.column_stack([x_cam, y_cam, z_cam])      # 3×3

        # Transform template rays to world frame
        rays_world = (R_c2w @ self._rays_cam.T).T           # (N, 3)

        # Endpoints at max_range
        endpoints = cam_pos + rays_world * self.max_range    # (N, 3)

        # DDA ray-casting with occlusion against ground truth
        self._cast_rays_occluded(cam_pos, endpoints)

        return self.explored_fraction - explored_before

    # ── DDA kernel with occlusion ─────────────────────────────────
    def _cast_rays_occluded(
        self,
        origin_world: np.ndarray,   # (3,)
        endpoints: np.ndarray,       # (N, 3)
    ) -> None:
        """Cast rays, using the *ground truth* to simulate occlusion.

        For each ray:
          • March through voxels from origin to endpoint.
          • If GT says FREE  → mark observed as FREE, continue.
          • If GT says OCCUPIED → mark observed as OCCUPIED, **stop ray**.
          • If we exit the grid → stop.

        This gives realistic occlusion: objects block the view of things
        behind them.
        """
        G = self.G
        vs = self.voxel_size
        vo = self.grid_origin
        gt = self.gt
        obs = self.observed

        o = (origin_world - vo) / vs   # origin in voxel coords

        for n in range(len(endpoints)):
            e = (endpoints[n] - vo) / vs
            d = e - o
            length = np.linalg.norm(d)
            if length < 1e-8:
                continue
            d = d / length

            ix, iy, iz = int(np.floor(o[0])), int(np.floor(o[1])), int(np.floor(o[2]))

            step_x = 1 if d[0] >= 0 else -1
            step_y = 1 if d[1] >= 0 else -1
            step_z = 1 if d[2] >= 0 else -1

            tmax_x = ((ix + (1 if step_x > 0 else 0)) - o[0]) / d[0] if abs(d[0]) > 1e-12 else 1e30
            tmax_y = ((iy + (1 if step_y > 0 else 0)) - o[1]) / d[1] if abs(d[1]) > 1e-12 else 1e30
            tmax_z = ((iz + (1 if step_z > 0 else 0)) - o[2]) / d[2] if abs(d[2]) > 1e-12 else 1e30
            tdelta_x = abs(1.0 / d[0]) if abs(d[0]) > 1e-12 else 1e30
            tdelta_y = abs(1.0 / d[1]) if abs(d[1]) > 1e-12 else 1e30
            tdelta_z = abs(1.0 / d[2]) if abs(d[2]) > 1e-12 else 1e30

            for _ in range(3 * G):
                if 0 <= ix < G and 0 <= iy < G and 0 <= iz < G:
                    gt_val = gt[ix, iy, iz]
                    if gt_val == OCCUPIED:
                        # Hit a surface → mark occupied and STOP (occlusion)
                        obs[ix, iy, iz] = OCCUPIED
                        break
                    else:
                        # Empty space → mark as FREE
                        obs[ix, iy, iz] = FREE
                else:
                    # Outside grid → stop ray
                    break

                # Advance DDA
                if tmax_x < tmax_y:
                    if tmax_x < tmax_z:
                        ix += step_x; tmax_x += tdelta_x
                    else:
                        iz += step_z; tmax_z += tdelta_z
                else:
                    if tmax_y < tmax_z:
                        iy += step_y; tmax_y += tdelta_y
                    else:
                        iz += step_z; tmax_z += tdelta_z


# =====================================================================
# Numba-accelerated version (optional, large speedup)
# =====================================================================

try:
    import numba

    @numba.njit(cache=True)
    def _dda_occluded_kernel(
        ox: float, oy: float, oz: float,
        ex: float, ey: float, ez: float,
        G: int,
        gt: np.ndarray,
        obs: np.ndarray,
    ) -> None:
        """Single-ray DDA with ground-truth occlusion, compiled with Numba."""
        dx = ex - ox
        dy = ey - oy
        dz = ez - oz
        length = (dx*dx + dy*dy + dz*dz) ** 0.5
        if length < 1e-8:
            return
        dx /= length; dy /= length; dz /= length

        ix = int(np.floor(ox))
        iy = int(np.floor(oy))
        iz = int(np.floor(oz))

        step_x = 1 if dx >= 0 else -1
        step_y = 1 if dy >= 0 else -1
        step_z = 1 if dz >= 0 else -1

        tmax_x = ((ix + (1 if step_x > 0 else 0)) - ox) / dx if abs(dx) > 1e-12 else 1e30
        tmax_y = ((iy + (1 if step_y > 0 else 0)) - oy) / dy if abs(dy) > 1e-12 else 1e30
        tmax_z = ((iz + (1 if step_z > 0 else 0)) - oz) / dz if abs(dz) > 1e-12 else 1e30
        tdelta_x = abs(1.0 / dx) if abs(dx) > 1e-12 else 1e30
        tdelta_y = abs(1.0 / dy) if abs(dy) > 1e-12 else 1e30
        tdelta_z = abs(1.0 / dz) if abs(dz) > 1e-12 else 1e30

        for _ in range(3 * G):
            if 0 <= ix < G and 0 <= iy < G and 0 <= iz < G:
                gt_val = gt[ix, iy, iz]
                if gt_val == 2:   # OCCUPIED
                    obs[ix, iy, iz] = 2
                    return        # occluded — stop
                else:
                    obs[ix, iy, iz] = 1   # FREE
            else:
                return

            if tmax_x < tmax_y:
                if tmax_x < tmax_z:
                    ix += step_x; tmax_x += tdelta_x
                else:
                    iz += step_z; tmax_z += tdelta_z
            else:
                if tmax_y < tmax_z:
                    iy += step_y; tmax_y += tdelta_y
                else:
                    iz += step_z; tmax_z += tdelta_z

    @numba.njit(parallel=True, cache=True)
    def _cast_rays_occluded_numba(
        origin: np.ndarray,     # (3,)
        endpoints: np.ndarray,  # (N, 3)
        vo: np.ndarray,         # (3,)
        vs: float,
        G: int,
        gt: np.ndarray,
        obs: np.ndarray,
    ) -> None:
        N = endpoints.shape[0]
        ox = (origin[0] - vo[0]) / vs
        oy = (origin[1] - vo[1]) / vs
        oz = (origin[2] - vo[2]) / vs
        for n in numba.prange(N):
            exi = (endpoints[n, 0] - vo[0]) / vs
            eyi = (endpoints[n, 1] - vo[1]) / vs
            ezi = (endpoints[n, 2] - vo[2]) / vs
            _dda_occluded_kernel(ox, oy, oz, exi, eyi, ezi, G, gt, obs)

    # Monkey-patch the simulator to use numba when available
    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


def _patch_simulator_numba(sim: VisibilitySimulator) -> None:
    """Replace the Python DDA with the Numba-accelerated version."""
    if not _NUMBA_AVAILABLE:
        return

    def _fast_cast(origin_world, endpoints):
        _cast_rays_occluded_numba(
            origin_world, endpoints,
            sim.grid_origin, sim.voxel_size, sim.G,
            sim.gt, sim.observed,
        )

    sim._cast_rays_occluded = _fast_cast


# =====================================================================
# Simulated FK  (when so101_kinematics is not available)
# =====================================================================

def simulated_fk_bank(
    candidates: list[list[float]],
    scene_center: np.ndarray | None = None,
    arm_reach: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate FK for the viewpoint bank without so101_kinematics.

    Generates plausible camera positions on a hemisphere around the scene
    centre, with orientations looking inward.  The J0 (base rotation) and
    J4 (wrist roll) joints map directly to azimuth and roll.

    This is only used when the real FK module is unavailable (e.g. in a
    pure-simulation training run on a server without the arm SDK).

    Returns: (positions, directions, wrist_rolls)  same shape as
             ``build_fk_bank``.
    """
    if scene_center is None:
        scene_center = np.array([0.15, 0.0, 0.0], dtype=np.float64)

    positions = []
    directions = []
    wrist_rolls = []

    for vp in candidates:
        j0, j1, j2, j3, j4, j5 = vp
        # Azimuth from J0 (base rotation)
        azimuth = np.deg2rad(j0)
        # Elevation: higher J1+J2 → lower elevation
        # Map J1 ∈ [-90, 0] and J2 ∈ [20, 110] to elevation
        elevation = np.deg2rad(30 + (j1 + 90) * 0.3 + (j2 - 60) * 0.2)
        elevation = np.clip(elevation, np.deg2rad(5), np.deg2rad(75))

        # Distance from scene centre (roughly arm_reach)
        dist = arm_reach * (0.8 + 0.4 * (j2 - 20) / 90.0)

        # Position on hemisphere
        x = scene_center[0] + dist * np.cos(elevation) * np.cos(azimuth)
        y = scene_center[1] + dist * np.cos(elevation) * np.sin(azimuth)
        z = scene_center[2] + dist * np.sin(elevation)

        # ── Floor constraint ──────────────────────────────────────
        # The table surface sits at z = scene_center[2] - half_extent
        # (bottom of the voxel volume).  Keep camera at least 3 cm
        # above the floor so the arm never dips below the table.
        floor_z = scene_center[2] - 0.3   # default half_extent
        z = max(z, floor_z + 0.03)

        pos = np.array([x, y, z])

        # Camera offset (left of gripper, angled right)
        # In the EE frame: shift left (-Y), rotate 5° right
        ee_fwd = (scene_center - pos)
        ee_fwd /= np.linalg.norm(ee_fwd) + 1e-12
        ee_right = np.cross(ee_fwd, np.array([0.0, 0.0, 1.0]))
        ee_right /= np.linalg.norm(ee_right) + 1e-12
        ee_up = np.cross(ee_right, ee_fwd)

        # Apply offset: 2 cm left, angle 5° right
        cam_pos = pos - ee_right * 0.02  # shift left
        cam_dir = ee_fwd + ee_right * np.tan(CAM_YAW_RAD)
        cam_dir /= np.linalg.norm(cam_dir) + 1e-12

        positions.append(cam_pos.astype(np.float32))
        directions.append(cam_dir.astype(np.float32))
        wrist_rolls.append(float(j4))

    return (
        np.array(positions, dtype=np.float32),
        np.array(directions, dtype=np.float32),
        np.array(wrist_rolls, dtype=np.float32),
    )
