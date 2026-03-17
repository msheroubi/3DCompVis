#!/usr/bin/env python3
"""
Offline test for the 3D occupancy grid pipeline.

Usage (from robot_vision_bridge/):
    python test_occupancy_offline.py --images ../vggt/imgs/armv2/

What it does:
  1. Loads all JPEGs from the specified folder (sorted).
  2. Runs VGGT reconstruction (loads model on GPU if available).
  3. Builds the tri-state occupancy grid via 3-D DDA ray-casting.
  4. Runs the frontier-based next-best-view planner.
  5. Prints a concise stats summary.
  6. Exports occupancy_grid.glb  — open with any GLTF viewer (e.g. model-viewer.dev).
  7. Exports occupancy_scatter.png — a quick 3-panel matplotlib figure.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline occupancy grid test")
    p.add_argument(
        "--images", default="../vggt/imgs/armv2/",
        help="Folder containing input JPEG/PNG frames (default: ../vggt/imgs/armv2/)",
    )
    p.add_argument(
        "--max-frames", type=int, default=20,
        help="Max frames to feed into VGGT (default: 20)",
    )
    p.add_argument(
        "--conf-threshold", type=float, default=0.3,
        help="Confidence threshold for point/ray filtering (default: 0.3)",
    )
    p.add_argument(
        "--ray-stride", type=int, default=4,
        help="Sub-sampling stride for DDA rays (default: 4)",
    )
    p.add_argument(
        "--no-glb", action="store_true",
        help="Skip GLB export (faster, no trimesh needed)",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib scatter plot",
    )
    p.add_argument(
        "--out-dir", default=".",
        help="Directory for output files (default: current dir)",
    )
    return p.parse_args()


# ======================================================================
# Helpers
# ======================================================================

def load_frames(folder: Path, max_frames: int) -> list[np.ndarray]:
    """Load all images from *folder*, sorted by filename, up to *max_frames*."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not paths:
        sys.exit(f"ERROR: no images found in {folder}")
    paths = paths[:max_frames]
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  WARNING: could not read {p.name}, skipping")
            continue
        frames.append(img)
        print(f"  Loaded {p.name}  ({img.shape[1]}×{img.shape[0]})")
    return frames


def print_stats(result) -> None:
    occ = result.voxel_occupancy
    unc = result.voxel_uncertainty
    G = occ.shape[0]
    total = occ.size

    n_free = int((occ == 1).sum())
    n_occ  = int((occ == 2).sum())
    n_unk  = int((occ == 0).sum())

    print()
    print("=" * 55)
    print("  RECONSTRUCTION SUMMARY")
    print("=" * 55)
    print(f"  Frames processed    : {result.n_input_frames}")
    print(f"  Processing time     : {result.processing_time_s:.1f} s")
    print(f"  Fused point count   : {len(result.fused_xyz):,}")
    print(f"  Mean confidence     : {result.fused_conf.mean():.3f}")
    if len(result.fused_xyz) > 0:
        mn = result.fused_xyz.min(axis=0)
        mx = result.fused_xyz.max(axis=0)
        print(f"  Scene bounds (m)    : [{mn[0]:.2f},{mn[1]:.2f},{mn[2]:.2f}] → "
              f"[{mx[0]:.2f},{mx[1]:.2f},{mx[2]:.2f}]")
    print()
    print(f"  Voxel grid          : {G}³  ({total:,} voxels)")
    print(f"  Voxel size          : {result.voxel_size:.4f} m")
    print(f"  FREE   (1)          : {n_free:6,}  ({100.0*n_free/total:5.1f}%)")
    print(f"  OCCUPIED (2)        : {n_occ:6,}  ({100.0*n_occ/total:5.1f}%)")
    print(f"  UNKNOWN (0)         : {n_unk:6,}  ({100.0*n_unk/total:5.1f}%)")
    print(f"  Explored fraction   : {100.0*(n_free+n_occ)/total:5.1f}%")
    print()
    print(f"  Mean uncertainty    : {unc.mean():.3f}")
    print(f"  Max  uncertainty    : {unc.max():.3f}")
    print("=" * 55)


def export_glb(result, out_dir: Path) -> None:
    try:
        import trimesh  # noqa — tested at call site
    except ImportError:
        print("  trimesh not found — skipping GLB export (pip install trimesh)")
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from occupancy_vis import occupancy_to_glb

    out_path = out_dir / "occupancy_grid.glb"
    print(f"  Exporting GLB → {out_path} …")
    t0 = time.perf_counter()
    occupancy_to_glb(
        result.voxel_occupancy,
        result.voxel_origin,
        result.voxel_size,
        show_free=True,
        show_frontier=True,
        point_cloud_xyz=result.fused_xyz if len(result.fused_xyz) > 0 else None,
        point_cloud_rgb=result.fused_rgb if len(result.fused_rgb) > 0 else None,
        output_path=str(out_path),
    )
    print(f"  GLB exported in {time.perf_counter()-t0:.1f} s  ({out_path.stat().st_size//1024} KB)")
    print(f"  Open at: https://modelviewer.dev/editor/ (drag and drop)")


def export_plot(result, planned_views, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
    except ImportError:
        print("  matplotlib not found — skipping plot (pip install matplotlib)")
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from occupancy_vis import occupancy_scatter_data

    data = occupancy_scatter_data(
        result.voxel_occupancy, result.voxel_origin, result.voxel_size
    )

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("Occupancy Grid — offline test", fontsize=13, fontweight="bold")

    # ---- Panel 1: OCCUPIED + UNKNOWN frontier ----
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title("OCCUPIED (red) + Frontier (grey)")
    if len(data["occupied_xyz"]) > 0:
        ax1.scatter(*data["occupied_xyz"].T, c="red", s=6, alpha=0.8, label="Occupied")
    if len(data["frontier_xyz"]) > 0:
        # Subsample frontier for speed
        idx = np.random.choice(len(data["frontier_xyz"]),
                               min(2000, len(data["frontier_xyz"])), replace=False)
        ax1.scatter(*data["frontier_xyz"][idx].T, c="grey", s=3, alpha=0.3, label="Frontier")
    ax1.legend(fontsize=7)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    # ---- Panel 2: FREE voxels ----
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_title("FREE voxels (green)")
    if len(data["free_xyz"]) > 0:
        idx = np.random.choice(len(data["free_xyz"]),
                               min(3000, len(data["free_xyz"])), replace=False)
        ax2.scatter(*data["free_xyz"][idx].T, c="green", s=3, alpha=0.3)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    # ---- Panel 3: Fused point cloud + planned viewpoints ----
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_title("Point cloud + NBV plan")
    if len(result.fused_xyz) > 0:
        rgb_norm = result.fused_rgb.astype(float) / 255.0
        idx = np.random.choice(len(result.fused_xyz),
                               min(5000, len(result.fused_xyz)), replace=False)
        ax3.scatter(*result.fused_xyz[idx].T, c=rgb_norm[idx], s=2, alpha=0.5)
    for i, vc in enumerate(planned_views):
        pos = np.array([vc.pose.x, vc.pose.y, vc.pose.z])
        tgt = vc.target_world
        ax3.scatter(*pos, c="blue", s=40, marker="^", zorder=10)
        ax3.plot(*zip(pos, tgt), c="blue", alpha=0.4, linewidth=1)
        ax3.text(pos[0], pos[1], pos[2], f"P{i}", fontsize=7, color="blue")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")

    out_path = out_dir / "occupancy_scatter.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved → {out_path}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    img_folder = Path(args.images).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure we can import bridge modules and the vggt package
    bridge_dir = Path(__file__).parent.resolve()
    vggt_dir = bridge_dir.parent / "vggt"
    for p in [str(bridge_dir), str(vggt_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    print(f"\nLoading images from: {img_folder}")
    frames = load_frames(img_folder, args.max_frames)
    print(f"\n{len(frames)} frame(s) loaded")

    # ---- Reconstruction ----
    from reconstruction import VGGTReconstructor
    recon = VGGTReconstructor(confidence_threshold=args.conf_threshold)
    # Patch the ray stride so we can configure it from CLI
    _orig_build = recon._build_occupancy_grid
    recon._build_occupancy_grid = lambda *a, **kw: _orig_build(
        *a, ray_stride=args.ray_stride, **kw
    )

    print("\nRunning VGGT reconstruction (model loads on first call) …")
    t0 = time.perf_counter()
    result = recon.reconstruct(frames)
    print(f"Reconstruction + occupancy grid: {time.perf_counter()-t0:.1f} s")

    print_stats(result)

    # ---- Planning ----
    from planner import NextBestViewPlanner
    planner = NextBestViewPlanner()
    planned_views = planner.plan(
        voxel_uncertainty=result.voxel_uncertainty,
        voxel_origin=result.voxel_origin,
        voxel_size=result.voxel_size,
        voxel_occupancy=result.voxel_occupancy,
    )

    if planned_views:
        print(f"\n  Next-best-view plan ({len(planned_views)} viewpoints):")
        for vc in planned_views:
            p = vc.pose
            print(f"    [P{vc.priority}] pos=({p.x:.3f},{p.y:.3f},{p.z:.3f})  "
                  f"target=({vc.target_world[0]:.3f},{vc.target_world[1]:.3f},"
                  f"{vc.target_world[2]:.3f})  unc={vc.expected_uncertainty:.3f}")
    else:
        print("\n  No viewpoints planned (scene may be fully explored).")

    # ---- Exports ----
    if not args.no_glb:
        print()
        export_glb(result, out_dir)

    if not args.no_plot:
        print()
        export_plot(result, planned_views, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
