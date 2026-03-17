"""
Robot Vision GPU Bridge — server with VGGT 3D reconstruction.

Endpoints
---------
GET  /health                  – liveness + device info
POST /session/start           – begin a new collection session
POST /session/stop            – finish collection, save to disk
GET  /session/status          – current session stats
POST /reconstruct             – run VGGT on collected (or specified) session
GET  /reconstruct/status      – poll reconstruction progress
GET  /reconstruct/result      – fetch latest reconstruction summary
GET  /plan                    – get next-best-view commands
WS   /ws/vision               – streaming: client sends JPEG+pose, server
                                 returns vision analytics + (optional)
                                 viewpoint commands
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import time
import threading
from typing import Optional

import cv2
import numpy as np
import orjson
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from collector import ArmPose, CollectionSession, SessionManager
from reconstruction import VGGTReconstructor, ReconstructionResult, UNKNOWN, FREE, OCCUPIED
from planner import NextBestViewPlanner, PlannerConfig, ViewpointCommand

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ======================================================================
# FastAPI app
# ======================================================================
app = FastAPI(title="Robot Vision GPU Bridge", version="0.2.0")


# ======================================================================
# Global state (initialised in __main__)
# ======================================================================

class ServerState:
    """Holds all mutable server state so it can be initialised cleanly."""
    def __init__(self, prefer_cuda: bool = True) -> None:
        self.device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"

        # Original lightweight vision engine (kept for real-time stream)
        self.vision = _LightVisionEngine(self.device)

        # Session / collection
        self.sessions = SessionManager()

        # VGGT reconstruction (lazy-loaded on first use)
        self.reconstructor = VGGTReconstructor(device=self.device)
        self.reconstruction_result: Optional[ReconstructionResult] = None
        self.reconstruction_running: bool = False
        self.reconstruction_error: Optional[str] = None

        # Next-best-view planner
        self.planner = NextBestViewPlanner()

        # Latest planned viewpoints
        self.planned_views: list[ViewpointCommand] = []

        # Fixed scene bounds (set via /scene/init before exploration)
        self.scene_center: Optional[np.ndarray] = None
        self.scene_half_extent: float = 0.4          # metres
        self.scene_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None  # (lo, hi)
        self.scene_bounds_type: str = "hemisphere"    # "hemisphere" | "cube"
        self.scene_forward: Optional[np.ndarray] = None  # unit forward dir (for hemisphere)


state: Optional[ServerState] = None


# ======================================================================
# Lightweight real-time vision (original logic, extracted)
# ======================================================================

class _LightVisionEngine:
    """Quick per-frame analytics (brightness / confidence) on the GPU."""

    def __init__(self, device: str) -> None:
        self.device = device

    def process(self, frame_bgr: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb)
            .to(self.device, non_blocking=True)
            .float() / 255.0
        )
        luminance = (
            0.2126 * tensor[..., 0]
            + 0.7152 * tensor[..., 1]
            + 0.0722 * tensor[..., 2]
        )
        brightness = float(luminance.mean().cpu().item())
        variance = float(torch.var(luminance).cpu().item())
        confidence = min(1.0, max(0.0, variance * 8.0))

        if brightness < 0.25:
            suggested = "increase_exposure_or_move_to_light"
        elif confidence < 0.15:
            suggested = "hold_position"
        else:
            suggested = "proceed_tracking"

        h, w = frame_bgr.shape[:2]
        return {
            "vision": {
                "scene_brightness": brightness,
                "feature_confidence": confidence,
                "image_size": [int(w), int(h)],
            },
            "control": {
                "suggested_action": suggested,
                "target_pose": {
                    "x": 0.0, "y": 0.0, "z": 0.0,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                },
            },
            "meta": {
                "device": self.device,
                "ts": time.time(),
            },
        }


# ======================================================================
# REST endpoints
# ======================================================================

@app.get("/health")
def health() -> JSONResponse:
    device = state.device if state else "uninitialized"
    active_session = state.sessions.active.session_id if state and state.sessions.active else None
    return JSONResponse({
        "status": "ok",
        "device": device,
        "active_session": active_session,
        "reconstruction_running": state.reconstruction_running if state else False,
    })


# ---- Session management ----

@app.post("/session/start")
def session_start(max_frames: int = 50, min_baseline_m: float = 0.02) -> JSONResponse:
    assert state is not None
    if state.sessions.active is not None:
        return JSONResponse({"error": "A session is already active. Stop it first."},
                            status_code=409)
    session = state.sessions.new_session(max_frames=max_frames,
                                         min_baseline_m=min_baseline_m)
    logger.info("Started session %s (max_frames=%d)", session.session_id, max_frames)
    return JSONResponse({
        "session_id": session.session_id,
        "status": session.status,
        "max_frames": max_frames,
    })


@app.post("/session/stop")
def session_stop() -> JSONResponse:
    assert state is not None
    finished = state.sessions.finish_collection()
    if finished is None:
        return JSONResponse({"error": "No active session."}, status_code=404)
    logger.info("Session %s stopped with %d frames",
                finished.session_id, len(finished.frames))
    return JSONResponse({
        "session_id": finished.session_id,
        "n_frames": len(finished.frames),
        "status": finished.status,
    })


@app.get("/session/status")
def session_status() -> JSONResponse:
    assert state is not None
    s = state.sessions.active
    if s is None:
        return JSONResponse({"active": False})
    return JSONResponse({
        "active": True,
        "session_id": s.session_id,
        "n_frames": len(s.frames),
        "max_frames": s.max_frames,
        "status": s.status,
    })


@app.post("/reset")
def reset_server() -> JSONResponse:
    """Clear all sessions, reconstruction results, and scene bounds."""
    assert state is not None
    n_sessions = len(state.sessions.history)
    state.sessions.history.clear()
    state.sessions.active = None
    state.reconstruction_result = None
    state.reconstruction_error = None
    state.planned_views = []
    state.scene_center = None
    state.scene_bounds = None
    state.scene_forward = None
    state.scene_bounds_type = "hemisphere"
    logger.info("Server reset: cleared %d sessions", n_sessions)
    return JSONResponse({"cleared_sessions": n_sessions, "status": "reset"})


# ---- Scene initialisation ----

from pydantic import BaseModel

class SceneInitRequest(BaseModel):
    center: list[float]                   # [x, y, z]
    forward: list[float] | None = None    # [fx, fy, fz]
    half_extent: float = 0.4
    bounds_type: str = "hemisphere"        # "hemisphere" | "cube"


@app.post("/scene/init")
def scene_init(body: SceneInitRequest) -> JSONResponse:
    """Set a fixed bounding volume for the scene.

    JSON body fields
    ----------------
    center : [x, y, z]
        Scene centre in world coordinates (usually the FK-derived camera
        position at the arm's starting pose).
    forward : [fx, fy, fz]
        Unit forward direction the arm faces (used for hemisphere mode).
    half_extent : float
        Half side-length (cube) or radius (hemisphere) in metres.
    bounds_type : str
        ``"hemisphere"`` (default) — only the half-space in front of the
        arm is included.  ``"cube"`` — full axis-aligned cube.
    """
    assert state is not None

    # ── Clear all previous sessions and reconstruction results ──
    state.sessions.history.clear()
    state.sessions.active = None
    state.reconstruction_result = None
    state.reconstruction_error = None
    state.planned_views = []
    logger.info("Scene init: cleared %s previous sessions and reconstruction",
                "all")

    c = np.array(body.center, dtype=np.float64)
    state.scene_center = c
    state.scene_half_extent = body.half_extent
    state.scene_bounds_type = body.bounds_type

    if body.forward is not None:
        fwd = np.array(body.forward, dtype=np.float64)
        fwd_norm = np.linalg.norm(fwd)
        state.scene_forward = fwd / fwd_norm if fwd_norm > 1e-8 else np.array([1.0, 0.0, 0.0])
    else:
        state.scene_forward = np.array([1.0, 0.0, 0.0])  # default

    # Compute axis-aligned bounding box that encloses the chosen volume.
    # For the hemisphere we still store a full cube as the grid bounds,
    # but the is_inside callback will clip to the half-space at voxel level.
    lo = c - body.half_extent
    hi = c + body.half_extent
    state.scene_bounds = (lo, hi)

    logger.info("Scene init: center=%s, half_extent=%.3f, type=%s, forward=%s",
                c.tolist(), body.half_extent, body.bounds_type,
                state.scene_forward.tolist())
    return JSONResponse({
        "center": c.tolist(),
        "half_extent": body.half_extent,
        "bounds_type": body.bounds_type,
        "forward": state.scene_forward.tolist(),
        "bbox_lo": lo.tolist(),
        "bbox_hi": hi.tolist(),
    })


# ---- Reconstruction ----

def _run_reconstruction(session: CollectionSession) -> None:
    """Run VGGT reconstruction in a background thread."""
    assert state is not None
    try:
        state.reconstruction_running = True
        state.reconstruction_error = None
        session.status = "reconstructing"

        subset = session.select_subset(n=20)
        bgr_frames = [f.frame_bgr for f in subset]

        result = state.reconstructor.reconstruct(
            bgr_frames,
            scene_bounds=state.scene_bounds,
            scene_bounds_type=state.scene_bounds_type,
            scene_forward=state.scene_forward,
            scene_center=state.scene_center,
        )
        state.reconstruction_result = result

        # Run planner immediately
        arm_poses = [f.arm_pose for f in subset]
        state.planned_views = state.planner.plan(
            voxel_uncertainty=result.voxel_uncertainty,
            voxel_origin=result.voxel_origin,
            voxel_size=result.voxel_size,
            previous_poses=arm_poses,
            voxel_occupancy=result.voxel_occupancy,
            arm_base_position=state.scene_center,
        )

        session.status = "done"
        logger.info("Reconstruction + planning complete for session %s", session.session_id)

    except Exception as exc:
        state.reconstruction_error = str(exc)
        session.status = "error"
        logger.exception("Reconstruction failed")
    finally:
        state.reconstruction_running = False


def _run_reconstruction_merged(sessions: list[CollectionSession]) -> None:
    """Run VGGT reconstruction on merged frames from ALL sessions.

    This is the key to iterative exploration: each reconstruction round
    uses *every* frame captured so far, producing a progressively more
    complete 3-D model.
    """
    assert state is not None
    try:
        state.reconstruction_running = True
        state.reconstruction_error = None
        for s in sessions:
            s.status = "reconstructing"

        # Merge all frames across sessions
        all_frames: list = []
        for s in sessions:
            all_frames.extend(s.frames)

        if not all_frames:
            raise ValueError("No frames to reconstruct")

        # Build a temporary merged session for subset selection
        merged = CollectionSession(session_id="merged")
        merged.frames = all_frames
        subset = merged.select_subset(n=20)
        bgr_frames = [f.frame_bgr for f in subset]

        logger.info("Merged reconstruction: %d sessions, %d total frames, "
                    "%d selected", len(sessions), len(all_frames), len(bgr_frames))

        result = state.reconstructor.reconstruct(
            bgr_frames,
            scene_bounds=state.scene_bounds,
            scene_bounds_type=state.scene_bounds_type,
            scene_forward=state.scene_forward,
            scene_center=state.scene_center,
        )
        state.reconstruction_result = result

        # Run planner immediately
        arm_poses = [f.arm_pose for f in subset]
        state.planned_views = state.planner.plan(
            voxel_uncertainty=result.voxel_uncertainty,
            voxel_origin=result.voxel_origin,
            voxel_size=result.voxel_size,
            previous_poses=arm_poses,
            voxel_occupancy=result.voxel_occupancy,
            arm_base_position=state.scene_center,
        )

        for s in sessions:
            s.status = "done"
        logger.info("Merged reconstruction + planning complete (%d sessions)",
                    len(sessions))

    except Exception as exc:
        state.reconstruction_error = str(exc)
        for s in sessions:
            s.status = "error"
        logger.exception("Merged reconstruction failed")
    finally:
        state.reconstruction_running = False


@app.post("/reconstruct")
def reconstruct(session_id: Optional[str] = None, merge_all: bool = False) -> JSONResponse:
    """Trigger reconstruction on a finished session (or the most recent one).

    If *merge_all* is ``True``, frames from **all** historical sessions are
    merged and reconstructed together — essential for iterative exploration.
    """
    assert state is not None
    if state.reconstruction_running:
        return JSONResponse({"error": "Reconstruction already in progress."}, status_code=409)

    # ── Merged reconstruction (all sessions) ──────────────────────────
    if merge_all:
        sessions = [
            s for s in state.sessions.history.values()
            if s.status in ("ready", "done") and len(s.frames) > 0
        ]
        if not sessions:
            return JSONResponse({"error": "No sessions with frames found."}, status_code=404)

        total_frames = sum(len(s.frames) for s in sessions)
        thread = threading.Thread(
            target=_run_reconstruction_merged, args=(sessions,), daemon=True)
        thread.start()

        return JSONResponse({
            "session_ids": [s.session_id for s in sessions],
            "n_sessions": len(sessions),
            "n_total_frames": total_frames,
            "status": "reconstruction_started",
        })

    # ── Single-session reconstruction (original behaviour) ────────────
    # Find the session
    if session_id:
        session = state.sessions.history.get(session_id)
    else:
        # Use the most recent finished session
        session = None
        for sid, s in reversed(list(state.sessions.history.items())):
            if s.status in ("ready", "done"):
                session = s
                break

    if session is None or len(session.frames) == 0:
        return JSONResponse({"error": "No session with frames found."}, status_code=404)

    # Launch in background thread so this endpoint returns immediately
    thread = threading.Thread(target=_run_reconstruction, args=(session,), daemon=True)
    thread.start()

    return JSONResponse({
        "session_id": session.session_id,
        "n_frames": len(session.frames),
        "status": "reconstruction_started",
    })


@app.get("/reconstruct/status")
def reconstruct_status() -> JSONResponse:
    assert state is not None
    return JSONResponse({
        "running": state.reconstruction_running,
        "error": state.reconstruction_error,
        "has_result": state.reconstruction_result is not None,
    })


@app.get("/reconstruct/result")
def reconstruct_result() -> JSONResponse:
    """Return a summary of the latest reconstruction (without the huge arrays)."""
    assert state is not None
    r = state.reconstruction_result
    if r is None:
        return JSONResponse({"error": "No reconstruction result available."}, status_code=404)

    # Compute per-voxel uncertainty stats
    unc = r.voxel_uncertainty
    high_unc_frac = float((unc > 0.7).sum() / max(unc.size, 1))

    # Compute occupancy stats
    occ = r.voxel_occupancy
    n_free = int((occ == FREE).sum())
    n_occupied = int((occ == OCCUPIED).sum())
    n_unknown = int((occ == UNKNOWN).sum())
    total = max(occ.size, 1)

    return JSONResponse({
        "n_input_frames": r.n_input_frames,
        "processing_time_s": round(r.processing_time_s, 2),
        "n_fused_points": int(len(r.fused_xyz)),
        "point_cloud_bounds": {
            "min": r.fused_xyz.min(axis=0).tolist() if len(r.fused_xyz) > 0 else [0, 0, 0],
            "max": r.fused_xyz.max(axis=0).tolist() if len(r.fused_xyz) > 0 else [0, 0, 0],
        },
        "mean_confidence": float(r.fused_conf.mean()) if len(r.fused_conf) > 0 else 0.0,
        "uncertainty": {
            "voxel_grid_shape": list(unc.shape),
            "voxel_size_m": r.voxel_size,
            "origin": r.voxel_origin.tolist(),
            "mean_uncertainty": float(unc.mean()),
            "max_uncertainty": float(unc.max()),
            "high_uncertainty_fraction": round(high_unc_frac, 4),
        },
        "occupancy": {
            "grid_shape": list(occ.shape),
            "voxel_size_m": r.voxel_size,
            "origin": r.voxel_origin.tolist(),
            "n_free": n_free,
            "n_occupied": n_occupied,
            "n_unknown": n_unknown,
            "free_fraction": round(n_free / total, 4),
            "occupied_fraction": round(n_occupied / total, 4),
            "explored_fraction": round((n_free + n_occupied) / total, 4),
        },
    })


# ---- Planning / next-best-view ----

@app.get("/occupancy")
def occupancy_grid() -> JSONResponse:
    """Return the full occupancy grid as a gzip‑compressed binary.

    The grid is a flat uint8 array of shape (G, G, G) in C‑order.
    The response includes the grid shape, origin, and voxel_size in headers.
    """
    import gzip
    from fastapi.responses import Response

    assert state is not None
    r = state.reconstruction_result
    if r is None:
        return JSONResponse({"error": "No reconstruction result."}, status_code=404)

    occ = r.voxel_occupancy
    compressed = gzip.compress(occ.tobytes(), compresslevel=6)

    return Response(
        content=compressed,
        media_type="application/octet-stream",
        headers={
            "Content-Encoding": "gzip",
            "X-Grid-Shape": ",".join(str(d) for d in occ.shape),
            "X-Voxel-Size": str(r.voxel_size),
            "X-Voxel-Origin": ",".join(str(v) for v in r.voxel_origin.tolist()),
        },
    )


@app.get("/pointcloud")
def pointcloud() -> JSONResponse:
    """Return the fused point cloud + camera poses as a gzip-compressed .npz blob.

    Arrays in the npz:
      xyz        (N, 3) float32  – world-space positions (fused, confidence-filtered)
      rgb        (N, 3) uint8    – colours
      conf       (N,)  float32   – per-point confidence
      extrinsics (S, 3, 4) float32 – world-to-cam (OpenCV convention)
      intrinsics (S, 3, 3) float32
      images     (S, H_t, W_t, 3) uint8 – small thumbnails for frustum overlays
    Headers: X-N-Frames, X-N-Points
    """
    import gzip
    import io
    import cv2  # noqa: F401 – used for thumbnail resize
    from fastapi.responses import Response

    assert state is not None
    r = state.reconstruction_result
    if r is None:
        return JSONResponse({"error": "No reconstruction result."}, status_code=404)

    # Build thumbnail images from the stored world_points input images
    # We store world_points as (S,H,W,3) — derive thumbnails from them if available,
    # otherwise generate blank placeholders.
    S = r.n_input_frames
    THUMB_W = 128
    if r.world_points is not None and len(r.world_points) > 0:
        # world_points is geometry; access the original images via fused_rgb mapping
        # We don't store raw images in the result — generate grey placeholders
        thumb_h = THUMB_W
        thumbs = np.full((S, thumb_h, THUMB_W, 3), 128, dtype=np.uint8)
    else:
        thumbs = np.zeros((max(S, 1), THUMB_W, THUMB_W, 3), dtype=np.uint8)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        xyz=r.fused_xyz.astype(np.float32),
        rgb=r.fused_rgb.astype(np.uint8),
        conf=r.fused_conf.astype(np.float32),
        extrinsics=r.extrinsics.astype(np.float32),
        intrinsics=r.intrinsics.astype(np.float32),
        images=thumbs,
    )
    buf.seek(0)
    compressed = gzip.compress(buf.read(), compresslevel=6)

    return Response(
        content=compressed,
        media_type="application/octet-stream",
        headers={
            "Content-Encoding": "gzip",
            "X-N-Frames": str(S),
            "X-N-Points": str(len(r.fused_xyz)),
        },
    )


@app.get("/plan")
def plan() -> JSONResponse:
    """Return the latest set of next-best-view commands."""
    assert state is not None
    if not state.planned_views:
        return JSONResponse({"viewpoints": [], "message": "No plan available. Run /reconstruct first."})

    views = []
    for vc in state.planned_views:
        views.append({
            "priority": vc.priority,
            "pose": vc.pose.to_dict(),
            "target_world": vc.target_world.tolist(),
            "expected_uncertainty": round(vc.expected_uncertainty, 4),
        })
    return JSONResponse({"viewpoints": views})


# ======================================================================
# WebSocket — streaming vision + collection
# ======================================================================

async def sender_loop(websocket: WebSocket, out_queue: asyncio.Queue) -> None:
    while True:
        payload = await out_queue.get()
        await websocket.send_bytes(orjson.dumps(payload))


@app.websocket("/ws/vision")
async def ws_vision(websocket: WebSocket) -> None:
    """Accept frames (+ optional arm pose) and return real-time analytics.

    Binary protocol:
      bytes[0:28]  -> 7 x float32  (x y z qx qy qz qw) -- arm pose
      bytes[28:]   -> JPEG payload

    If the message is shorter than 29 bytes or the quaternion norm check
    fails we assume legacy format (all JPEG, no pose).
    """
    await websocket.accept()
    assert state is not None

    out_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    sender = asyncio.create_task(sender_loop(websocket, out_queue))

    try:
        while True:
            raw = await websocket.receive_bytes()

            # --- Decode pose + image from binary message -----------------
            arm_pose = ArmPose()
            POSE_HEADER = 7 * 4  # 28 bytes
            if len(raw) > POSE_HEADER + 2:
                # Try interpreting the first 28 bytes as 7 floats
                maybe_pose = np.frombuffer(raw[:POSE_HEADER], dtype=np.float32)
                # Simple heuristic: valid quaternion has norm ~ 1
                quat_norm = np.linalg.norm(maybe_pose[3:7])
                if 0.9 < quat_norm < 1.1:
                    arm_pose = ArmPose(
                        x=float(maybe_pose[0]), y=float(maybe_pose[1]), z=float(maybe_pose[2]),
                        qx=float(maybe_pose[3]), qy=float(maybe_pose[4]),
                        qz=float(maybe_pose[5]), qw=float(maybe_pose[6]),
                    )
                    jpeg_bytes = raw[POSE_HEADER:]
                else:
                    jpeg_bytes = raw
            else:
                jpeg_bytes = raw

            np_buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # --- Real-time analytics -------------------------------------
            t0 = time.perf_counter()
            result = state.vision.process(frame)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            result["meta"]["processing_ms"] = dt_ms
            result["meta"]["e2e_server_ts"] = time.time()

            # --- If a session is active, try to add the frame ------------
            session = state.sessions.active
            if session is not None:
                accepted = session.add_frame(frame, arm_pose)
                result["collection"] = {
                    "active": True,
                    "session_id": session.session_id,
                    "n_frames": len(session.frames),
                    "frame_accepted": accepted,
                }
            else:
                result["collection"] = {"active": False}

            # --- Attach latest plan if available -------------------------
            if state.planned_views:
                top = state.planned_views[0]
                result["control"]["target_pose"] = top.pose.to_dict()
                result["control"]["suggested_action"] = "move_to_viewpoint"
                result["control"]["planned_viewpoints_remaining"] = len(state.planned_views)

            # --- Send ---------------------------------------------------
            if out_queue.full():
                _ = out_queue.get_nowait()
            await out_queue.put(result)

    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot vision GPU bridge server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--no-preload", action="store_true",
                        help="Skip preloading the VGGT model at startup")
    parser.add_argument("--sessions-dir", default="sessions",
                        help="Directory to save collection sessions")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    state = ServerState(prefer_cuda=not args.cpu)
    state.sessions.sessions_dir = args.sessions_dir
    logger.info("Server starting -- device=%s", state.device)

    if not args.no_preload:
        logger.info("Pre-loading VGGT model (use --no-preload to skip) …")
        state.reconstructor.load_model()
        logger.info("Model ready – starting HTTP server")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
