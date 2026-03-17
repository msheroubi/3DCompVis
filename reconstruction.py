"""
3D reconstruction via VGGT with per‑region uncertainty estimation.

Wraps the VGGT model (facebook/VGGT-1B) to produce:
  • dense point clouds in world coordinates
  • per‑pixel depth + confidence
  • a voxelised uncertainty volume used by the planner
"""
from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Occupancy voxel states
# ---------------------------------------------------------------------------
UNKNOWN: int = 0
FREE: int = 1
OCCUPIED: int = 2

# ---------------------------------------------------------------------------
# Lazy imports so the module can be loaded even without vggt installed.
# The actual import happens inside VGGTReconstructor.load_model().
# ---------------------------------------------------------------------------
_vggt_available: Optional[bool] = None


def _check_vggt() -> bool:
    global _vggt_available
    if _vggt_available is None:
        try:
            import vggt  # noqa: F401
            _vggt_available = True
        except ImportError:
            # Try adding the parent directory (where vggt/ lives) to sys.path
            import sys
            vggt_dir = str(Path(__file__).resolve().parent.parent / "vggt")
            if vggt_dir not in sys.path:
                sys.path.insert(0, vggt_dir)
            try:
                import vggt  # noqa: F401
                _vggt_available = True
            except ImportError:
                _vggt_available = False
    return _vggt_available


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Everything produced by a single reconstruction run."""

    # Dense geometry  (all np.ndarray unless noted)
    world_points: np.ndarray          # (S, H, W, 3) – point map per view
    world_points_conf: np.ndarray     # (S, H, W)    – confidence [0, 1]
    depth_maps: np.ndarray            # (S, H, W)    – metric depth
    depth_conf: np.ndarray            # (S, H, W)    – depth confidence

    # Camera parameters (predicted by VGGT *or* overridden by arm poses)
    extrinsics: np.ndarray            # (S, 3, 4)
    intrinsics: np.ndarray            # (S, 3, 3)

    # Fused point cloud (after confidence filtering)
    fused_xyz: np.ndarray             # (N, 3)
    fused_rgb: np.ndarray             # (N, 3)  uint8
    fused_conf: np.ndarray            # (N,)

    # Voxel‑level uncertainty (used by the planner)
    voxel_uncertainty: np.ndarray     # (Gx, Gy, Gz) – mean inverse confidence
    voxel_origin: np.ndarray          # (3,) – world coord of voxel [0,0,0]
    voxel_size: float                 # metres per voxel

    # Tri‑state occupancy grid: UNKNOWN=0, FREE=1, OCCUPIED=2
    voxel_occupancy: np.ndarray       # (Gx, Gy, Gz)  dtype=uint8

    processing_time_s: float = 0.0
    n_input_frames: int = 0


# ---------------------------------------------------------------------------
# Reconstruction engine
# ---------------------------------------------------------------------------

class VGGTReconstructor:
    """Stateful wrapper around the VGGT model."""

    def __init__(self, device: str = "cuda", voxel_resolution: int = 64,
                 confidence_threshold: float = 0.3) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.voxel_resolution = voxel_resolution
        self.conf_threshold = confidence_threshold
        self.model = None
        self._dtype = None

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load the VGGT‑1B checkpoint (downloads on first call)."""
        if self.model is not None:
            return
        if not _check_vggt():
            raise ImportError(
                "vggt is not installed. "
                "Clone https://github.com/facebookresearch/vggt and `pip install -e .`"
            )
        from vggt.models.vggt import VGGT  # type: ignore

        logger.info("Loading VGGT‑1B model …")
        t0 = time.time()
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.model.eval()
        cap = torch.cuda.get_device_capability() if self.device == "cuda" else (0, 0)
        self._dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        logger.info("VGGT loaded in %.1f s  (device=%s, dtype=%s)",
                     time.time() - t0, self.device, self._dtype)

    # ------------------------------------------------------------------
    def _prepare_images(self, bgr_frames: list[np.ndarray]) -> torch.Tensor:
        """Convert a list of BGR numpy frames to the tensor VGGT expects.

        VGGT wants (S, 3, H, W) float32 in [0, 1] with H, W divisible by 14.
        We resize width to 518 and adjust height accordingly.
        """
        TARGET_W = 518
        processed = []
        for bgr in bgr_frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            new_h = int(round(h * TARGET_W / w / 14)) * 14
            resized = cv2.resize(rgb, (TARGET_W, new_h), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            processed.append(tensor)

        # All images must be the same size — pad shorter ones
        max_h = max(t.shape[1] for t in processed)
        padded = []
        for t in processed:
            if t.shape[1] < max_h:
                pad = torch.zeros(3, max_h - t.shape[1], TARGET_W)
                t = torch.cat([t, pad], dim=1)
            padded.append(t)

        return torch.stack(padded)  # (S, 3, H, W)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, bgr_frames: list[np.ndarray]) -> ReconstructionResult:
        """Run full reconstruction on a list of BGR frames.

        Returns a ReconstructionResult with all geometry + uncertainty.
        """
        self.load_model()
        t0 = time.perf_counter()

        images = self._prepare_images(bgr_frames).to(self.device)
        S = images.shape[0]
        H, W = images.shape[2], images.shape[3]

        # ---- VGGT forward pass ------------------------------------------
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
        from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore

        with torch.cuda.amp.autocast(dtype=self._dtype):
            preds = self.model(images)

        # Decode poses
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            preds["pose_enc"], (H, W)
        )
        preds["extrinsic"] = extrinsic
        preds["intrinsic"] = intrinsic

        # Move everything to numpy (squeeze batch dim)
        np_preds: dict[str, np.ndarray] = {}
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                np_preds[k] = v.cpu().float().numpy().squeeze(0)
            else:
                np_preds[k] = v

        # ---- Build per‑pixel world points from depth --------------------
        depth_maps = np_preds["depth"]           # (S, H, W, 1)
        depth_conf = np_preds["depth_conf"]      # (S, H, W)
        world_pts  = np_preds["world_points"]    # (S, H, W, 3)
        world_conf = np_preds["world_points_conf"]  # (S, H, W)
        extr       = np_preds["extrinsic"]       # (S, 3, 4)
        intr       = np_preds["intrinsic"]       # (S, 3, 3)

        depth_maps_sq = depth_maps.squeeze(-1) if depth_maps.ndim == 4 else depth_maps

        # ---- Fuse into a single filtered point cloud --------------------
        fused_xyz, fused_rgb, fused_conf = self._fuse_point_cloud(
            world_pts, world_conf, bgr_frames, images.cpu().numpy().squeeze(0) if images.ndim == 5 else images.cpu().numpy()
        )

        # ---- Voxelise uncertainty ---------------------------------------
        voxel_unc, voxel_origin, voxel_size = self._voxelise_uncertainty(
            fused_xyz, fused_conf
        )

        # ---- Occupancy grid via ray‑casting -----------------------------
        voxel_occ = self._build_occupancy_grid(
            extr, intr, world_pts, world_conf,
            voxel_origin, voxel_size,
        )

        dt = time.perf_counter() - t0
        logger.info("Reconstruction done: %d frames, %.1f s, %d points",
                     S, dt, len(fused_xyz))

        return ReconstructionResult(
            world_points=world_pts,
            world_points_conf=world_conf,
            depth_maps=depth_maps_sq,
            depth_conf=depth_conf,
            extrinsics=extr,
            intrinsics=intr,
            fused_xyz=fused_xyz,
            fused_rgb=fused_rgb,
            fused_conf=fused_conf,
            voxel_uncertainty=voxel_unc,
            voxel_origin=voxel_origin,
            voxel_size=voxel_size,
            voxel_occupancy=voxel_occ,
            processing_time_s=dt,
            n_input_frames=S,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fuse_point_cloud(
        self,
        world_pts: np.ndarray,     # (S, H, W, 3)
        world_conf: np.ndarray,    # (S, H, W)
        bgr_frames: list[np.ndarray],
        images_np: np.ndarray,     # (S, 3, H, W) float [0,1]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten per‑view point maps into one cloud, filtered by confidence."""
        S, H, W, _ = world_pts.shape

        all_xyz, all_rgb, all_conf = [], [], []
        for s in range(S):
            conf = world_conf[s]                      # (H, W)
            mask = conf > self.conf_threshold         # bool (H, W)

            pts = world_pts[s][mask]                  # (N, 3)
            # get colour from the preprocessed image tensor
            rgb = (images_np[s].transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
            colours = rgb[mask]                       # (N, 3)

            all_xyz.append(pts)
            all_rgb.append(colours)
            all_conf.append(conf[mask])

        return (
            np.concatenate(all_xyz, axis=0) if all_xyz else np.zeros((0, 3)),
            np.concatenate(all_rgb, axis=0) if all_rgb else np.zeros((0, 3), dtype=np.uint8),
            np.concatenate(all_conf, axis=0) if all_conf else np.zeros((0,)),
        )

    def _voxelise_uncertainty(
        self,
        xyz: np.ndarray,   # (N, 3)
        conf: np.ndarray,  # (N,)
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Build a voxel grid of mean *inverse* confidence (= uncertainty).

        Voxels with no observations get maximum uncertainty (1.0).
        Returns (grid, origin, voxel_size).
        """
        G = self.voxel_resolution
        if len(xyz) == 0:
            return np.ones((G, G, G), dtype=np.float32), np.zeros(3), 1.0

        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        extent = hi - lo
        # add a 5 % border so edge points aren't on voxel boundaries
        margin = extent * 0.05 + 1e-6
        origin = lo - margin
        span = extent + 2 * margin
        voxel_size = float(span.max() / G)

        # quantise each point to a voxel
        idx = ((xyz - origin) / voxel_size).astype(np.int32)
        idx = np.clip(idx, 0, G - 1)

        # accumulate inverse confidence per voxel
        sum_grid = np.zeros((G, G, G), dtype=np.float64)
        count_grid = np.zeros((G, G, G), dtype=np.float64)
        inv_conf = 1.0 - np.clip(conf, 0.0, 1.0)

        np.add.at(sum_grid, (idx[:, 0], idx[:, 1], idx[:, 2]), inv_conf)
        np.add.at(count_grid, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

        # mean inverse confidence (unobserved voxels → 1.0 = max uncertainty)
        with np.errstate(divide="ignore", invalid="ignore"):
            unc = np.where(count_grid > 0, sum_grid / count_grid, 1.0).astype(np.float32)

        return unc, origin.astype(np.float32), voxel_size

    # ------------------------------------------------------------------
    def _build_occupancy_grid(
        self,
        extrinsics: np.ndarray,        # (S, 3, 4)
        intrinsics: np.ndarray,        # (S, 3, 3)
        world_points: np.ndarray,      # (S, H, W, 3)
        world_points_conf: np.ndarray, # (S, H, W)
        voxel_origin: np.ndarray,      # (3,)
        voxel_size: float,
        ray_stride: int = 4,
    ) -> np.ndarray:
        """Build a tri‑state occupancy grid via 3D DDA ray‑casting.

        For every sub‑sampled pixel with sufficient confidence, a ray is
        traced from the camera origin through the voxel grid to the
        observed surface point.  Intermediate voxels are marked FREE;
        the surface voxel is marked OCCUPIED.  Unvisited voxels remain
        UNKNOWN.

        Returns
        -------
        grid : np.ndarray, dtype=uint8, shape (G, G, G)
            Values are UNKNOWN (0), FREE (1), or OCCUPIED (2).
        """
        G = self.voxel_resolution
        grid = np.zeros((G, G, G), dtype=np.uint8)  # UNKNOWN = 0

        S, H, W, _ = world_points.shape

        for s in range(S):
            # Camera origin in world coordinates:
            # extrinsic = [R | t]  (cam‑from‑world)
            # cam_pos_world = -R^T @ t
            R = extrinsics[s, :3, :3]       # (3, 3)
            t = extrinsics[s, :3, 3]        # (3,)
            cam_origin = -R.T @ t           # (3,) world

            conf = world_points_conf[s]     # (H, W)
            pts  = world_points[s]          # (H, W, 3)

            # Sub‑sample pixels with stride
            rows = np.arange(0, H, ray_stride)
            cols = np.arange(0, W, ray_stride)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            rr, cc = rr.ravel(), cc.ravel()

            # Confidence mask
            mask = conf[rr, cc] > self.conf_threshold
            rr, cc = rr[mask], cc[mask]

            if len(rr) == 0:
                continue

            endpoints = pts[rr, cc]          # (N, 3)

            # Cast each ray through the voxel grid
            _cast_rays_dda(
                cam_origin, endpoints, voxel_origin,
                voxel_size, G, grid,
            )

        n_free = int((grid == FREE).sum())
        n_occ  = int((grid == OCCUPIED).sum())
        n_unk  = int(grid.size - n_free - n_occ)
        logger.info(
            "Occupancy grid: %d FREE (%.1f%%), %d OCCUPIED (%.1f%%), %d UNKNOWN (%.1f%%)",
            n_free, 100.0 * n_free / grid.size,
            n_occ,  100.0 * n_occ  / grid.size,
            n_unk,  100.0 * n_unk  / grid.size,
        )
        return grid


# ======================================================================
# 3‑D DDA ray traversal  (Amanatides & Woo)
# ======================================================================

def _cast_rays_dda(
    origin_world: np.ndarray,       # (3,)  camera centre
    endpoints: np.ndarray,          # (N, 3)  surface points
    voxel_origin: np.ndarray,       # (3,)
    voxel_size: float,
    G: int,
    grid: np.ndarray,               # (G, G, G) uint8 — modified in place
) -> None:
    """Cast many rays through a voxel grid using 3‑D DDA.

    Each ray starts at *origin_world* and ends at the corresponding row
    of *endpoints*.  Intermediate voxels → FREE, terminal voxel → OCCUPIED.
    """
    # Try numba fast‑path; fall back to pure‑numpy if unavailable.
    try:
        _cast_rays_numba(origin_world, endpoints, voxel_origin,
                         voxel_size, G, grid)
    except Exception:                # numba not installed or JIT failed
        _cast_rays_python(origin_world, endpoints, voxel_origin,
                          voxel_size, G, grid)


# ------------------------------------------------------------------
# Pure‑Python / NumPy fallback
# ------------------------------------------------------------------

def _cast_rays_python(
    origin: np.ndarray,
    endpoints: np.ndarray,
    voxel_origin: np.ndarray,
    vs: float,
    G: int,
    grid: np.ndarray,
) -> None:
    """Scalar DDA – correct but slow.  Used when numba is not available."""
    o = (origin - voxel_origin) / vs          # origin in voxel coords
    eps = endpoints                            # (N, 3) world
    for n in range(len(eps)):
        e = (eps[n] - voxel_origin) / vs      # endpoint in voxel coords
        d = e - o
        length = np.linalg.norm(d)
        if length < 1e-8:
            continue
        d = d / length                        # unit direction

        # starting voxel
        ix, iy, iz = int(np.floor(o[0])), int(np.floor(o[1])), int(np.floor(o[2]))
        # ending voxel
        ex, ey, ez = int(np.floor(e[0])), int(np.floor(e[1])), int(np.floor(e[2]))

        step_x = 1 if d[0] >= 0 else -1
        step_y = 1 if d[1] >= 0 else -1
        step_z = 1 if d[2] >= 0 else -1

        # tMax — distance along ray to the next voxel boundary
        # tDelta — distance along ray to cross one full voxel
        def _t_max(o_comp, d_comp, i_comp, step):
            if abs(d_comp) < 1e-12:
                return 1e30
            boundary = (i_comp + (1 if step > 0 else 0))
            return (boundary - o_comp) / d_comp

        def _t_delta(d_comp):
            return abs(1.0 / d_comp) if abs(d_comp) > 1e-12 else 1e30

        tmax_x = _t_max(o[0], d[0], ix, step_x)
        tmax_y = _t_max(o[1], d[1], iy, step_y)
        tmax_z = _t_max(o[2], d[2], iz, step_z)
        tdelta_x = _t_delta(d[0])
        tdelta_y = _t_delta(d[1])
        tdelta_z = _t_delta(d[2])

        max_steps = 3 * G   # safety cap
        for _ in range(max_steps):
            if 0 <= ix < G and 0 <= iy < G and 0 <= iz < G:
                # Check if we've reached the endpoint voxel
                if ix == ex and iy == ey and iz == ez:
                    grid[ix, iy, iz] = OCCUPIED
                    break
                # Only mark as FREE if not already OCCUPIED
                if grid[ix, iy, iz] != OCCUPIED:
                    grid[ix, iy, iz] = FREE

            # Advance to next voxel boundary
            if tmax_x < tmax_y:
                if tmax_x < tmax_z:
                    ix += step_x
                    tmax_x += tdelta_x
                else:
                    iz += step_z
                    tmax_z += tdelta_z
            else:
                if tmax_y < tmax_z:
                    iy += step_y
                    tmax_y += tdelta_y
                else:
                    iz += step_z
                    tmax_z += tdelta_z

            # Stop if we've left the grid entirely (all 3 axes)
            if (ix < 0 or ix >= G) and (iy < 0 or iy >= G) and (iz < 0 or iz >= G):
                break
        else:
            # If we exited the loop without hitting endpoint,
            # still mark the endpoint voxel as OCCUPIED if inside grid
            if 0 <= ex < G and 0 <= ey < G and 0 <= ez < G:
                grid[ex, ey, ez] = OCCUPIED


# ------------------------------------------------------------------
# Numba‑accelerated version (optional)
# ------------------------------------------------------------------
try:
    import numba

    @numba.njit(cache=True)
    def _dda_kernel(
        ox: float, oy: float, oz: float,
        ex: float, ey: float, ez: float,
        G: int,
        grid: np.ndarray,
    ) -> None:
        """Single‑ray 3D DDA kernel compiled with Numba."""
        dx = ex - ox
        dy = ey - oy
        dz = ez - oz
        length = (dx*dx + dy*dy + dz*dz) ** 0.5
        if length < 1e-8:
            return
        dx /= length
        dy /= length
        dz /= length

        ix = int(np.floor(ox))
        iy = int(np.floor(oy))
        iz = int(np.floor(oz))
        iex = int(np.floor(ex))
        iey = int(np.floor(ey))
        iez = int(np.floor(ez))

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
                if ix == iex and iy == iey and iz == iez:
                    grid[ix, iy, iz] = 2  # OCCUPIED
                    return
                if grid[ix, iy, iz] != 2:
                    grid[ix, iy, iz] = 1  # FREE

            if tmax_x < tmax_y:
                if tmax_x < tmax_z:
                    ix += step_x
                    tmax_x += tdelta_x
                else:
                    iz += step_z
                    tmax_z += tdelta_z
            else:
                if tmax_y < tmax_z:
                    iy += step_y
                    tmax_y += tdelta_y
                else:
                    iz += step_z
                    tmax_z += tdelta_z

            if ix < -1 or ix > G or iy < -1 or iy > G or iz < -1 or iz > G:
                break

        # Mark endpoint even if we didn't land exactly on it
        if 0 <= iex < G and 0 <= iey < G and 0 <= iez < G:
            grid[iex, iey, iez] = 2

    @numba.njit(parallel=True, cache=True)
    def _cast_rays_numba(
        origin_world: np.ndarray,
        endpoints: np.ndarray,
        voxel_origin: np.ndarray,
        vs: float,
        G: int,
        grid: np.ndarray,
    ) -> None:
        N = endpoints.shape[0]
        ox = (origin_world[0] - voxel_origin[0]) / vs
        oy = (origin_world[1] - voxel_origin[1]) / vs
        oz = (origin_world[2] - voxel_origin[2]) / vs
        for n in numba.prange(N):
            exi = (endpoints[n, 0] - voxel_origin[0]) / vs
            eyi = (endpoints[n, 1] - voxel_origin[1]) / vs
            ezi = (endpoints[n, 2] - voxel_origin[2]) / vs
            _dda_kernel(ox, oy, oz, exi, eyi, ezi, G, grid)

except ImportError:
    # numba not available — _cast_rays_dda will use the Python fallback
    pass
