"""
Occupancy grid visualisation utilities.

Exports the tri‑state voxel grid as a trimesh Scene (.glb) with:
  • OCCUPIED voxels  → solid red cubes
  • FREE voxels      → semi‑transparent green cubes (optional)
  • UNKNOWN frontier → semi‑transparent grey cubes (optional)

Also provides a lightweight scatter‑based alternative that is much
faster to generate for 64³ grids.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Occupancy constants (must match reconstruction.py)
UNKNOWN = 0
FREE = 1
OCCUPIED = 2


# ======================================================================
# GLB / trimesh visualisation
# ======================================================================

def occupancy_to_glb(
    voxel_occupancy: np.ndarray,        # (G, G, G) uint8
    voxel_origin: np.ndarray,           # (3,)
    voxel_size: float,
    *,
    show_free: bool = False,
    show_frontier: bool = True,
    point_cloud_xyz: Optional[np.ndarray] = None,  # (N, 3)
    point_cloud_rgb: Optional[np.ndarray] = None,  # (N, 3) uint8
    output_path: Optional[str] = None,
) -> "trimesh.Scene":                   # type: ignore[name-defined]
    """Convert an occupancy grid to a trimesh Scene and optionally save as GLB.

    Parameters
    ----------
    voxel_occupancy : (G, G, G) uint8
        Tri‑state grid.
    voxel_origin : (3,) float
        World coordinate of voxel [0, 0, 0].
    voxel_size : float
        Side length of each voxel in metres.
    show_free : bool
        If True, render FREE voxels as translucent green cubes.
    show_frontier : bool
        If True, render frontier UNKNOWN voxels (adjacent to FREE)
        as translucent grey cubes.
    point_cloud_xyz, point_cloud_rgb : optional
        If provided, overlay the fused point cloud on the scene.
    output_path : optional str
        If given, save the scene as a .glb file.

    Returns
    -------
    trimesh.Scene
    """
    import trimesh

    scene = trimesh.Scene()

    G = voxel_occupancy.shape[0]
    half = voxel_size / 2.0

    # --- Occupied voxels (red) ---
    occ_ijk = np.argwhere(voxel_occupancy == OCCUPIED)
    if len(occ_ijk) > 0:
        centres = voxel_origin + (occ_ijk + 0.5) * voxel_size
        _add_voxel_boxes(scene, centres, half, color=[200, 40, 40, 255],
                         name="occupied")
        logger.info("Visualised %d OCCUPIED voxels", len(occ_ijk))

    # --- Free voxels (green, translucent) ---
    if show_free:
        free_ijk = np.argwhere(voxel_occupancy == FREE)
        if len(free_ijk) > 0:
            centres = voxel_origin + (free_ijk + 0.5) * voxel_size
            _add_voxel_boxes(scene, centres, half, color=[40, 200, 40, 50],
                             name="free")
            logger.info("Visualised %d FREE voxels", len(free_ijk))

    # --- Frontier voxels (grey, translucent) ---
    if show_frontier:
        frontier = _compute_frontier(voxel_occupancy)
        front_ijk = np.argwhere(frontier)
        if len(front_ijk) > 0:
            centres = voxel_origin + (front_ijk + 0.5) * voxel_size
            _add_voxel_boxes(scene, centres, half, color=[180, 180, 180, 80],
                             name="frontier")
            logger.info("Visualised %d frontier voxels", len(front_ijk))

    # --- Optional point cloud overlay ---
    if point_cloud_xyz is not None and len(point_cloud_xyz) > 0:
        colors = point_cloud_rgb if point_cloud_rgb is not None else None
        pc = trimesh.PointCloud(point_cloud_xyz, colors=colors)
        scene.add_geometry(pc, geom_name="point_cloud")

    if output_path:
        scene.export(output_path)
        logger.info("Saved occupancy GLB to %s", output_path)

    return scene


# ======================================================================
# Helpers
# ======================================================================

def _add_voxel_boxes(
    scene: "trimesh.Scene",
    centres: np.ndarray,         # (M, 3)
    half_size: float,
    color: list[int],
    name: str,
) -> None:
    """Add instanced cube geometry to the scene for a set of voxel centres."""
    import trimesh

    # Create one canonical cube
    box = trimesh.creation.box(extents=[half_size * 2] * 3)
    box.visual.face_colors = color

    # Instance it at each centre via a single concatenated mesh
    # (much faster than adding thousands of individual geometries)
    meshes = []
    for c in centres:
        b = box.copy()
        b.apply_translation(c)
        meshes.append(b)

    if meshes:
        combined = trimesh.util.concatenate(meshes)
        scene.add_geometry(combined, geom_name=name)


def _compute_frontier(occupancy: np.ndarray) -> np.ndarray:
    """Return a boolean mask of UNKNOWN voxels that are 6‑connected to FREE."""
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


# ======================================================================
# Lightweight scatter plot (no trimesh required)
# ======================================================================

def occupancy_scatter_data(
    voxel_occupancy: np.ndarray,
    voxel_origin: np.ndarray,
    voxel_size: float,
) -> dict[str, np.ndarray]:
    """Return centres + labels for plotting with matplotlib / plotly / etc.

    Returns
    -------
    dict with keys:
      - 'occupied_xyz'  : (M1, 3)
      - 'free_xyz'      : (M2, 3)
      - 'frontier_xyz'  : (M3, 3)
    """
    occ_ijk = np.argwhere(voxel_occupancy == OCCUPIED)
    free_ijk = np.argwhere(voxel_occupancy == FREE)
    frontier = _compute_frontier(voxel_occupancy)
    front_ijk = np.argwhere(frontier)

    to_world = lambda ijk: voxel_origin + (ijk + 0.5) * voxel_size

    return {
        "occupied_xyz": to_world(occ_ijk) if len(occ_ijk) > 0 else np.zeros((0, 3)),
        "free_xyz": to_world(free_ijk) if len(free_ijk) > 0 else np.zeros((0, 3)),
        "frontier_xyz": to_world(front_ijk) if len(front_ijk) > 0 else np.zeros((0, 3)),
    }
