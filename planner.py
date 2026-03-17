"""
Next‑best‑view planner.

Given the voxelised uncertainty volume from reconstruction, proposes a
ranked list of 6‑DoF camera poses the robot arm should visit to reduce
reconstruction uncertainty the most.

Strategy
--------
1.  Identify the *k* highest‑uncertainty voxel clusters.
2.  For each cluster centroid, compute a viewpoint that looks at it from a
    distance proportional to the scene extent, avoiding previously visited
    poses.
3.  Return the viewpoints as a ranked list of ArmPose commands.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from collector import ArmPose
from reconstruction import UNKNOWN, FREE, OCCUPIED

logger = logging.getLogger(__name__)


@dataclass
class ViewpointCommand:
    """One suggested viewpoint for the robot arm."""
    pose: ArmPose
    target_world: np.ndarray        # (3,) – the world point we're looking at
    expected_uncertainty: float      # mean uncertainty of the target cluster
    priority: int = 0               # 0 = highest priority


@dataclass
class PlannerConfig:
    n_viewpoints: int = 5
    """How many viewpoints to propose per planning cycle."""
    view_distance_factor: float = 1.5
    """Multiplier on median scene radius for camera standoff distance."""
    min_distance_between_views_m: float = 0.05
    """Don't propose two viewpoints closer than this."""
    up_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    """World up direction (for computing look‑at quaternions)."""


class NextBestViewPlanner:
    """Compute next‑best views from a voxelised uncertainty grid."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.cfg = config or PlannerConfig()

    # ------------------------------------------------------------------
    def plan(
        self,
        voxel_uncertainty: np.ndarray,   # (Gx, Gy, Gz)
        voxel_origin: np.ndarray,        # (3,)
        voxel_size: float,
        previous_poses: list[ArmPose] | None = None,
        scene_center: np.ndarray | None = None,
        voxel_occupancy: np.ndarray | None = None,  # (Gx, Gy, Gz) uint8
    ) -> list[ViewpointCommand]:
        """Return a ranked list of viewpoint commands.

        If *voxel_occupancy* is provided the planner first targets UNKNOWN
        frontier voxels (adjacent to FREE space) and only falls back to
        uncertainty‑based clusters when no frontier remains.
        """
        G = voxel_uncertainty.shape[0]

        # ---- 1. Find target cluster centroids ---------------------------
        # Prefer frontier clusters (UNKNOWN voxels next to FREE) when an
        # occupancy grid is available; fall back to uncertainty clusters.
        centroids = np.zeros((0, 3))
        unc_values = np.zeros((0,))
        strategy = "uncertainty"  # default

        if voxel_occupancy is not None:
            centroids, unc_values = self._find_frontier_clusters(
                voxel_occupancy, voxel_uncertainty,
                voxel_origin, voxel_size,
                k=self.cfg.n_viewpoints * 2,
            )
            if len(centroids) > 0:
                strategy = "frontier"

        if len(centroids) == 0:
            centroids, unc_values = self._find_uncertain_clusters(
                voxel_uncertainty, voxel_origin, voxel_size,
                k=self.cfg.n_viewpoints * 2,  # oversample, then prune
            )

        if len(centroids) == 0:
            logger.info("No uncertain regions found — scene looks complete.")
            return []

        logger.info("Planning with strategy='%s', %d candidate centroids",
                    strategy, len(centroids))

        # ---- 2. Compute scene geometry ----------------------------------
        if scene_center is None:
            scene_center = voxel_origin + 0.5 * G * voxel_size

        scene_radius = 0.5 * G * voxel_size
        standoff = scene_radius * self.cfg.view_distance_factor

        # ---- 3. Generate candidate viewpoints ---------------------------
        prev_positions = (
            np.array([p.translation() for p in previous_poses])
            if previous_poses else np.zeros((0, 3))
        )

        commands: list[ViewpointCommand] = []
        used_positions: list[np.ndarray] = []

        for centroid, unc in zip(centroids, unc_values):
            # viewpoint sits on the line from scene_center through centroid,
            # at *standoff* distance from the centroid
            direction = centroid - scene_center
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm

            cam_pos = centroid + direction * standoff

            # skip if too close to a previous or already‑proposed pose
            if len(prev_positions) > 0:
                dists_prev = np.linalg.norm(prev_positions - cam_pos, axis=1)
                if dists_prev.min() < self.cfg.min_distance_between_views_m:
                    continue
            if any(np.linalg.norm(p - cam_pos) < self.cfg.min_distance_between_views_m
                   for p in used_positions):
                continue

            quat = self._look_at_quaternion(cam_pos, centroid, self.cfg.up_vector)
            pose = ArmPose(
                x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(cam_pos[2]),
                qx=float(quat[0]), qy=float(quat[1]),
                qz=float(quat[2]), qw=float(quat[3]),
            )
            commands.append(ViewpointCommand(
                pose=pose,
                target_world=centroid,
                expected_uncertainty=float(unc),
                priority=len(commands),
            ))
            used_positions.append(cam_pos)

            if len(commands) >= self.cfg.n_viewpoints:
                break

        logger.info("Planner proposed %d viewpoints", len(commands))
        return commands

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_uncertain_clusters(
        self,
        grid: np.ndarray,
        origin: np.ndarray,
        voxel_size: float,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the top‑k uncertain voxel centroids (world coords) and their values.

        Instead of full clustering we simply take the *k* voxels with highest
        uncertainty that are not direct neighbours (26‑connected suppression).
        """
        G = grid.shape[0]
        flat = grid.ravel().copy()
        order = np.argsort(flat)[::-1]  # descending uncertainty

        selected_ijk: list[np.ndarray] = []
        selected_unc: list[float] = []

        for idx in order:
            if flat[idx] <= 0.0:
                break  # remaining voxels have zero uncertainty
            ijk = np.array(np.unravel_index(idx, grid.shape))

            # suppress neighbours of already selected voxels
            too_close = False
            for prev in selected_ijk:
                if np.abs(ijk - prev).max() <= 2:
                    too_close = True
                    break
            if too_close:
                continue

            selected_ijk.append(ijk)
            selected_unc.append(float(flat[idx]))
            if len(selected_ijk) >= k:
                break

        if not selected_ijk:
            return np.zeros((0, 3)), np.zeros((0,))

        centroids = origin + (np.array(selected_ijk) + 0.5) * voxel_size
        return centroids, np.array(selected_unc)

    def _find_frontier_clusters(
        self,
        occupancy: np.ndarray,     # (G, G, G) uint8
        uncertainty: np.ndarray,   # (G, G, G) float
        origin: np.ndarray,
        voxel_size: float,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find clusters of UNKNOWN voxels adjacent to FREE space (the frontier).

        Strategy:
        1. Identify all UNKNOWN voxels that have at least one 6‑connected
           FREE neighbour.  These form the *exploration frontier*.
        2. Rank them by uncertainty (high → low) and apply 26‑connected
           non‑max suppression (same as ``_find_uncertain_clusters``).
        """
        G = occupancy.shape[0]

        # Build binary masks
        is_unknown = occupancy == UNKNOWN
        is_free = occupancy == FREE

        # Dilate FREE by 1 voxel along 6‑connected axes to find its boundary
        free_dilated = np.zeros_like(is_free)
        free_dilated[1:, :, :] |= is_free[:-1, :, :]
        free_dilated[:-1, :, :] |= is_free[1:, :, :]
        free_dilated[:, 1:, :] |= is_free[:, :-1, :]
        free_dilated[:, :-1, :] |= is_free[:, 1:, :]
        free_dilated[:, :, 1:] |= is_free[:, :, :-1]
        free_dilated[:, :, :-1] |= is_free[:, :, 1:]

        frontier = is_unknown & free_dilated

        frontier_indices = np.argwhere(frontier)  # (M, 3)
        if len(frontier_indices) == 0:
            return np.zeros((0, 3)), np.zeros((0,))

        # Score each frontier voxel by its uncertainty
        scores = uncertainty[frontier]
        order = np.argsort(scores)[::-1]

        selected_ijk: list[np.ndarray] = []
        selected_score: list[float] = []

        for idx in order:
            ijk = frontier_indices[idx]
            too_close = False
            for prev in selected_ijk:
                if np.abs(ijk - prev).max() <= 2:
                    too_close = True
                    break
            if too_close:
                continue
            selected_ijk.append(ijk)
            selected_score.append(float(scores[idx]))
            if len(selected_ijk) >= k:
                break

        if not selected_ijk:
            return np.zeros((0, 3)), np.zeros((0,))

        centroids = origin + (np.array(selected_ijk) + 0.5) * voxel_size
        logger.info("Frontier: %d voxels, selected %d clusters",
                    len(frontier_indices), len(centroids))
        return centroids, np.array(selected_score)

    @staticmethod
    def _look_at_quaternion(
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
    ) -> np.ndarray:
        """Compute a quaternion (qx, qy, qz, qw) for a camera at *eye* looking at *target*.

        Uses the OpenCV camera convention (z‑forward, x‑right, y‑down).
        """
        forward = target - eye
        forward /= np.linalg.norm(forward) + 1e-12

        right = np.cross(forward, up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            # degenerate — pick an arbitrary right vector
            right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
            rn = np.linalg.norm(right)
        right /= rn

        down = np.cross(forward, right)
        down /= np.linalg.norm(down) + 1e-12

        # rotation matrix columns → (right, down, forward) = OpenCV convention
        R = np.stack([right, down, forward], axis=1)  # (3, 3)

        return _rotation_matrix_to_quaternion(R)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to quaternion (qx, qy, qz, qw)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)
