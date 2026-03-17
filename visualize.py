#!/usr/bin/env python3
"""
Visualize reconstruction + occupancy grid from a running server.

Usage:
    python visualize.py                               # connects to localhost:8765
    python visualize.py --server http://IP:8765
    python visualize.py --server http://IP:8765 --glb out.glb
    python visualize.py --server http://IP:8765 --save scatter.png
    python visualize.py --server http://IP:8765 --viser        # interactive 3-D viewer
    python visualize.py --server http://IP:8765 --snapshot before.npz  # save snapshot
    python visualize.py --server http://IP:8765 --before before.npz --viser  # before/after

Options:
    --server       Server base URL (default: http://localhost:8765)
    --glb          Export occupancy grid to a .glb file (requires trimesh)
    --save         Save scatter plot to file instead of showing interactively
    --no-free      Hide free voxels (show only occupied + frontier)
    --viser        Open interactive viser 3-D viewer (open browser at localhost:VISER_PORT)
    --viser-port   Port for the viser server (default 8080)
    --snapshot     Save current reconstruction as a .npz snapshot file
    --before       Path to a previously saved .npz snapshot for before/after comparison
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
import requests

# ── Occupancy constants (must match reconstruction.py) ──────────────────────
UNKNOWN = 0
FREE = 1
OCCUPIED = 2


def fetch_result(base: str) -> dict:
    r = requests.get(base.rstrip("/") + "/reconstruct/result", timeout=30)
    if r.status_code == 404:
        print("ERROR: No reconstruction result on server. Run a reconstruction first.")
        sys.exit(1)
    r.raise_for_status()
    return r.json()


def fetch_occupancy(base: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (grid, voxel_origin, voxel_size)."""
    r = requests.get(base.rstrip("/") + "/occupancy", timeout=30)
    if r.status_code == 404:
        print("ERROR: No occupancy grid on server.")
        sys.exit(1)
    r.raise_for_status()

    shape = tuple(int(d) for d in r.headers["X-Grid-Shape"].split(","))
    voxel_size = float(r.headers["X-Voxel-Size"])
    origin = np.array([float(v) for v in r.headers["X-Voxel-Origin"].split(",")])

    # The server sets Content-Encoding: gzip — requests may auto-decompress
    # or not, depending on version. Handle both:
    raw = r.content
    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass  # already decompressed by requests

    grid = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    return grid, origin, voxel_size


def fetch_plan(base: str) -> list[dict]:
    r = requests.get(base.rstrip("/") + "/plan", timeout=30)
    r.raise_for_status()
    return r.json().get("viewpoints", [])


def compute_frontier(occupancy: np.ndarray) -> np.ndarray:
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


def scatter_plot(
    grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    result: dict,
    plan: list[dict],
    *,
    show_free: bool = True,
    save_path: str | None = None,
) -> None:
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def centres(ijk: np.ndarray) -> np.ndarray:
        return origin + (ijk + 0.5) * voxel_size

    occ_ijk = np.argwhere(grid == OCCUPIED)
    free_ijk = np.argwhere(grid == FREE)
    front_ijk = np.argwhere(compute_frontier(grid))

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("VGGT Reconstruction — Occupancy Grid", fontsize=13, fontweight="bold")

    # ── Left: 3-D scatter ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.set_title("3-D Occupancy")

    if show_free and len(free_ijk) > 0:
        xyz = centres(free_ijk)
        # subsample to keep rendering fast
        idx = np.random.choice(len(xyz), min(len(xyz), 2000), replace=False)
        ax3.scatter(*xyz[idx].T, c="limegreen", s=2, alpha=0.15, label=f"FREE ({len(free_ijk):,})")

    if len(front_ijk) > 0:
        xyz = centres(front_ijk)
        idx = np.random.choice(len(xyz), min(len(xyz), 3000), replace=False)
        ax3.scatter(*xyz[idx].T, c="grey", s=3, alpha=0.35, label=f"Frontier ({len(front_ijk):,})")

    if len(occ_ijk) > 0:
        xyz = centres(occ_ijk)
        ax3.scatter(*xyz.T, c="crimson", s=6, alpha=0.9, label=f"OCCUPIED ({len(occ_ijk):,})")

    # Plot NBV viewpoints
    for i, vp in enumerate(plan[:5]):
        pose = vp["pose"]  # flat dict: {x, y, z, qx, qy, qz, qw}
        px, py, pz = pose["x"], pose["y"], pose["z"]
        tgt = vp.get("target_world", [px, py, pz])
        ax3.scatter(px, py, pz,
                    c="gold", s=80, marker="^", zorder=5,
                    label="NBV" if i == 0 else "_")
        ax3.plot([px, tgt[0]], [py, tgt[1]], [pz, tgt[2]],
                 "y--", linewidth=0.8, alpha=0.6)

    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.legend(loc="upper left", fontsize=7, markerscale=2)

    # ── Right: stats panel ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")

    occ_info = result.get("occupancy", {})
    unc_info = result.get("uncertainty", {})
    total = grid.size

    lines = [
        ("Frames", str(result.get("n_input_frames", "?"))),
        ("Fused points", f"{result.get('n_fused_points', 0):,}"),
        ("Processing time", f"{result.get('processing_time_s', 0):.2f} s"),
        ("Mean confidence", f"{result.get('mean_confidence', 0):.4f}"),
        ("", ""),
        ("Grid shape", str(occ_info.get("grid_shape", "?"))),
        ("Voxel size", f"{occ_info.get('voxel_size_m', 0)*100:.1f} cm"),
        ("", ""),
        ("FREE voxels", f"{occ_info.get('n_free', 0):,}  ({occ_info.get('free_fraction', 0):.1%})"),
        ("OCCUPIED voxels", f"{occ_info.get('n_occupied', 0):,}  ({occ_info.get('occupied_fraction', 0):.1%})"),
        ("UNKNOWN voxels", f"{total - occ_info.get('n_free', 0) - occ_info.get('n_occupied', 0):,}"),
        ("Explored", f"{occ_info.get('explored_fraction', 0):.1%}"),
        ("", ""),
        ("Mean uncertainty", f"{unc_info.get('mean_uncertainty', 0):.4f}"),
        ("Max uncertainty", f"{unc_info.get('max_uncertainty', 0):.4f}"),
        ("High-unc fraction", f"{unc_info.get('high_uncertainty_fraction', 0):.1%}"),
        ("", ""),
        ("NBV viewpoints", str(len(plan))),
    ]
    col_left = 0.05
    col_right = 0.55
    y = 0.97
    dy = 0.056
    for label, val in lines:
        if not label:
            y -= dy * 0.4
            continue
        ax2.text(col_left, y, label + ":", transform=ax2.transAxes,
                 fontsize=9, va="top", color="#444")
        ax2.text(col_right, y, val, transform=ax2.transAxes,
                 fontsize=9, va="top", fontweight="bold")
        y -= dy

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Scatter plot saved to {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Point-cloud fetch + snapshot helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_pointcloud(base: str) -> dict:
    """Fetch fused point cloud + camera poses from /pointcloud.

    Returns dict with keys: xyz, rgb, conf, extrinsics, intrinsics, images
    """
    r = requests.get(base.rstrip("/") + "/pointcloud", timeout=60)
    if r.status_code == 404:
        print("ERROR: No point cloud on server.")
        sys.exit(1)
    r.raise_for_status()

    raw = r.content
    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass

    import io
    npz = np.load(io.BytesIO(raw))
    return {k: npz[k] for k in npz.files}


def save_snapshot(data: dict, path: str) -> None:
    np.savez_compressed(path, **data)
    print(f"Snapshot saved to {path}")


def load_snapshot(path: str) -> dict:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def _invert_extrinsics(ext: np.ndarray) -> np.ndarray:
    """Convert (S, 3, 4) world-to-cam [R|t] to cam-to-world.

    cam-to-world: R_cw = R.T,  t_cw = -R.T @ t
    """
    S = ext.shape[0]
    c2w = np.zeros_like(ext)  # (S, 3, 4)
    R = ext[:, :3, :3]        # (S, 3, 3)
    t = ext[:, :3, 3]         # (S, 3)
    c2w[:, :3, :3] = R.transpose(0, 2, 1)
    c2w[:, :3, 3] = -(R.transpose(0, 2, 1) @ t[:, :, None])[:, :, 0]
    return c2w


# ─────────────────────────────────────────────────────────────────────────────
# Viser 3-D interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

def viser_view(
    current: dict,
    *,
    before: dict | None = None,
    plan: list[dict] | None = None,
    grid: np.ndarray | None = None,
    grid_origin: np.ndarray | None = None,
    grid_voxel_size: float = 0.02,
    port: int = 8080,
) -> None:
    """Open an interactive viser viewer.

    current / before: dicts with keys xyz, rgb, conf, extrinsics, intrinsics, images.
    The 'before' cloud is shown in a cool blue tint for easy comparison.
    """
    import viser
    import viser.transforms as vtf

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # ── Scene center = mean of current cloud ─────────────────────────────────
    xyz_cur = current["xyz"].astype(np.float32)
    rgb_cur = current["rgb"].astype(np.uint8)
    conf_cur = current["conf"].astype(np.float32)
    center = xyz_cur.mean(axis=0) if len(xyz_cur) > 0 else np.zeros(3)

    def recentered(xyz: np.ndarray) -> np.ndarray:
        return xyz - center

    # ── GUI controls ─────────────────────────────────────────────────────────
    gui_conf = server.gui.add_slider(
        "Confidence threshold %", min=0, max=100, step=1, initial_value=10)
    gui_show_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
    gui_show_occ = server.gui.add_checkbox("Show occupancy grid", initial_value=True)
    gui_show_bounds = server.gui.add_checkbox("Show grid bounding box", initial_value=True)
    gui_show_unknown = server.gui.add_checkbox("Show UNKNOWN voxels", initial_value=False)
    if before is not None:
        gui_show_before = server.gui.add_checkbox("Show BEFORE cloud", initial_value=True)
        gui_show_after  = server.gui.add_checkbox("Show AFTER cloud",  initial_value=True)

    # ── AFTER (current) point cloud ───────────────────────────────────────────
    def _mask(conf: np.ndarray, pct: float) -> np.ndarray:
        thr = np.percentile(conf, pct)
        return conf >= thr

    pc_after = server.scene.add_point_cloud(
        "after",
        points=recentered(xyz_cur),
        colors=rgb_cur,
        point_size=0.002,
        point_shape="circle",
    )

    # ── BEFORE point cloud (blue-tinted) ─────────────────────────────────────
    pc_before = None
    if before is not None:
        xyz_bef = before["xyz"].astype(np.float32)
        conf_bef = before["conf"].astype(np.float32)
        # Tint grey-blue to distinguish from current
        blue_rgb = np.tile(np.array([[100, 160, 255]], dtype=np.uint8), (len(xyz_bef), 1))
        pc_before = server.scene.add_point_cloud(
            "before",
            points=recentered(xyz_bef),
            colors=blue_rgb,
            point_size=0.002,
            point_shape="circle",
        )

    # ── Camera frustums ───────────────────────────────────────────────────────
    cam_handles: list = []

    def _add_cameras(pc_data: dict, label: str, color: list[int]) -> None:
        extr = pc_data["extrinsics"]  # world-to-cam (S, 3, 4)
        intr = pc_data["intrinsics"]  # (S, 3, 3)
        c2w  = _invert_extrinsics(extr)
        imgs = pc_data.get("images")  # (S, H, W, 3) uint8 or None
        S = len(extr)
        for i in range(S):
            R = c2w[i, :3, :3]
            t = c2w[i, :3, 3]
            wxyz = vtf.SO3.from_matrix(R).wxyz
            frame = server.scene.add_frame(
                f"{label}/cam_{i}",
                wxyz=wxyz, position=t - center,
                axes_length=0.03, axes_radius=0.001, origin_radius=0.001,
            )
            fx = intr[i, 0, 0]
            fy = intr[i, 1, 1]
            H  = int(round(intr[i, 1, 2] * 2))
            W  = int(round(intr[i, 0, 2] * 2))
            fov = float(2 * np.arctan2(H / 2, fy)) if fy > 0 else 0.7
            asp = W / H if H > 0 else 1.0
            thumb = imgs[i] if imgs is not None and len(imgs) > i else None
            server.scene.add_camera_frustum(
                f"{label}/cam_{i}/frustum",
                fov=fov, aspect=asp, scale=0.04,
                image=thumb, line_width=1.0,
            )
            cam_handles.append(frame)

    _add_cameras(current, "after", [255, 200, 0])
    if before is not None:
        _add_cameras(before, "before", [100, 160, 255])

    # ── Occupancy grid overlay (OCCUPIED + frontier + optional UNKNOWN) ─────────
    occ_pc = None
    unknown_pc = None
    bounds_handle = None
    if grid is not None and grid_origin is not None:
        G = grid.shape[0]
        occ_ijk   = np.argwhere(grid == OCCUPIED)
        front_ijk = np.argwhere(compute_frontier(grid))
        occ_xyz   = grid_origin + (occ_ijk   + 0.5) * grid_voxel_size
        front_xyz = grid_origin + (front_ijk + 0.5) * grid_voxel_size
        all_occ_xyz = np.vstack([occ_xyz, front_xyz]) if len(front_xyz) > 0 else occ_xyz
        colors_occ = np.vstack([
            np.tile([220, 40, 40, 200], (len(occ_xyz), 1)),
            np.tile([160, 160, 160, 100], (len(front_xyz), 1)),
        ]).astype(np.uint8)[:, :3] if (len(occ_xyz) + len(front_xyz)) > 0 else np.zeros((0, 3), dtype=np.uint8)
        if len(all_occ_xyz) > 0:
            occ_pc = server.scene.add_point_cloud(
                "occupancy",
                points=recentered(all_occ_xyz),
                colors=colors_occ,
                point_size=grid_voxel_size * 0.8,
                point_shape="square",
            )

        # UNKNOWN voxels (sparse sample so the browser doesn't choke)
        unk_ijk = np.argwhere(grid == UNKNOWN)
        if len(unk_ijk) > 0:
            rng = np.random.default_rng(0)
            sample = rng.choice(len(unk_ijk), min(len(unk_ijk), 8000), replace=False)
            unk_xyz = grid_origin + (unk_ijk[sample] + 0.5) * grid_voxel_size
            unknown_pc = server.scene.add_point_cloud(
                "unknown",
                points=recentered(unk_xyz),
                colors=np.tile([80, 80, 120], (len(unk_xyz), 1)).astype(np.uint8),
                point_size=grid_voxel_size * 0.4,
                point_shape="square",
            )
            unknown_pc.visible = False   # hidden by default

        # Grid bounding-box wireframe – 12 edges of the cube
        lo = grid_origin - center
        hi = grid_origin + G * grid_voxel_size - center
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom face
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top face
        ])
        edges = np.array([
            [corners[0], corners[1]], [corners[1], corners[2]],
            [corners[2], corners[3]], [corners[3], corners[0]],  # bottom loop
            [corners[4], corners[5]], [corners[5], corners[6]],
            [corners[6], corners[7]], [corners[7], corners[4]],  # top loop
            [corners[0], corners[4]], [corners[1], corners[5]],
            [corners[2], corners[6]], [corners[3], corners[7]],  # verticals
        ], dtype=np.float32)  # (12, 2, 3)
        bounds_handle = server.scene.add_line_segments(
            "grid_bounds",
            points=edges,
            colors=np.array([255, 255, 100], dtype=np.uint8),
            line_width=1.5,
        )

    # ── NBV viewpoints ────────────────────────────────────────────────────────
    if plan:
        for i, vp in enumerate(plan[:5]):
            pose = vp["pose"]
            pos = np.array([pose["x"], pose["y"], pose["z"]]) - center
            tgt_raw = vp.get("target_world")
            if tgt_raw:
                tgt = np.array(tgt_raw) - center
                server.scene.add_line_segments(
                    f"nbv/ray_{i}",
                    points=np.array([[pos, tgt]]),
                    colors=np.array([255, 220, 0], dtype=np.uint8),
                    line_width=1.5,
                )
            server.scene.add_icosphere(
                f"nbv/point_{i}",
                radius=0.008,
                position=pos,
                color=(255, 220, 0),
            )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    @gui_conf.on_update
    def _(_) -> None:
        pct = gui_conf.value
        m = _mask(conf_cur, pct)
        pc_after.points = recentered(xyz_cur[m])
        pc_after.colors = rgb_cur[m]
        if pc_before is not None:
            mb = _mask(before["conf"], pct)  # type: ignore[index]
            pc_before.points = recentered(before["xyz"][mb])  # type: ignore[index]

    @gui_show_cameras.on_update
    def _(_) -> None:
        for h in cam_handles:
            h.visible = gui_show_cameras.value

    if occ_pc is not None:
        @gui_show_occ.on_update
        def _(_) -> None:
            occ_pc.visible = gui_show_occ.value  # type: ignore[union-attr]

    if bounds_handle is not None:
        @gui_show_bounds.on_update
        def _(_) -> None:
            bounds_handle.visible = gui_show_bounds.value  # type: ignore[union-attr]

    if unknown_pc is not None:
        @gui_show_unknown.on_update
        def _(_) -> None:
            unknown_pc.visible = gui_show_unknown.value  # type: ignore[union-attr]

    if before is not None:
        @gui_show_before.on_update
        def _(_) -> None:
            if pc_before is not None:
                pc_before.visible = gui_show_before.value

        @gui_show_after.on_update
        def _(_) -> None:
            pc_after.visible = gui_show_after.value

    label = "AFTER" if before is None else "AFTER  (orange cameras)"
    before_label = "BEFORE (blue cameras)"
    print(f"\nViser viewer at http://localhost:{port}")
    print(f"  {label}:  {len(xyz_cur):,} points")
    if before is not None:
        print(f"  {before_label}: {len(before['xyz']):,} points")
    print("  Use the controls panel in the browser to toggle layers.")
    print("  Press Ctrl+C to quit.\n")

    import time
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


def export_glb(
    grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    output_path: str,
    show_free: bool = True,
) -> None:
    # Re-use occupancy_vis from this package
    bridge_dir = Path(__file__).resolve().parent
    if str(bridge_dir) not in sys.path:
        sys.path.insert(0, str(bridge_dir))
    from occupancy_vis import occupancy_to_glb
    occupancy_to_glb(
        grid, origin, voxel_size,
        show_free=show_free,
        show_frontier=True,
        output_path=output_path,
    )
    print(f"GLB saved to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8765",
                    help="Server base URL")
    ap.add_argument("--glb", metavar="OUT.glb",
                    help="Export occupancy to a GLB file")
    ap.add_argument("--save", metavar="OUT.png",
                    help="Save scatter plot to PNG instead of showing interactively")
    ap.add_argument("--no-free", action="store_true",
                    help="Hide free voxels in the scatter plot / GLB")
    ap.add_argument("--viser", action="store_true",
                    help="Open viser interactive 3-D viewer")
    ap.add_argument("--viser-port", type=int, default=8080,
                    help="Port for the viser server (default 8080)")
    ap.add_argument("--snapshot", metavar="FILE.npz",
                    help="Save current reconstruction as a .npz snapshot")
    ap.add_argument("--before", metavar="FILE.npz",
                    help="Previous snapshot for before/after comparison in viser")
    args = ap.parse_args()

    print(f"Fetching data from {args.server} …")
    result = fetch_result(args.server)
    grid, origin, voxel_size = fetch_occupancy(args.server)
    plan = fetch_plan(args.server)

    occ = result.get("occupancy", {})
    print(f"  Grid: {grid.shape}  voxel={voxel_size*100:.1f} cm")
    print(f"  FREE={occ.get('n_free',0):,}  OCCUPIED={occ.get('n_occupied',0):,}"
          f"  UNKNOWN={grid.size - occ.get('n_free',0) - occ.get('n_occupied',0):,}")
    print(f"  Explored: {occ.get('explored_fraction', 0):.1%}   NBV viewpoints: {len(plan)}")

    if args.glb:
        export_glb(grid, origin, voxel_size, args.glb, show_free=not args.no_free)

    # ── Snapshot: save & optionally start viser ────────────────────────────
    if args.snapshot or args.viser:
        print("Fetching point cloud …")
        pc = fetch_pointcloud(args.server)
        if args.snapshot:
            save_snapshot(pc, args.snapshot)

        if args.viser:
            before_data = load_snapshot(args.before) if args.before else None
            viser_view(
                pc,
                before=before_data,
                plan=plan,
                grid=grid,
                grid_origin=origin,
                grid_voxel_size=voxel_size,
                port=args.viser_port,
            )
            return  # viser loop blocks; skip the 2-D plot

    # ── 2-D matplotlib scatter (default) ──────────────────────────────────
    scatter_plot(
        grid, origin, voxel_size,
        result=result,
        plan=plan,
        show_free=not args.no_free,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
