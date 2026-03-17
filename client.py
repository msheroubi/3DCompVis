"""
Robot Vision Bridge — client with reconstruction workflow.

Modes
-----
stream   – continuous camera streaming with optional pose data
collect  – start a session, stream frames+poses, stop, reconstruct, get plan
"""
from __future__ import annotations

import argparse
import asyncio
import struct
import time

import cv2
import numpy as np
import orjson
import websockets
import requests


POSE_STRUCT = struct.Struct("<7f")  # x y z qx qy qz qw  (28 bytes, little-endian)


# ======================================================================
# Arm pose source (stub — replace with your actual robot SDK)
# ======================================================================

class ArmPoseSource:
    """Abstract interface to the robot arm's current end-effector pose.

    Replace the body of `get_pose()` with your actual robot SDK call,
    e.g. reading from ROS tf, a KUKA iiwa driver, or a UR RTDE stream.
    """

    def get_pose(self) -> tuple[float, ...]:
        """Return (x, y, z, qx, qy, qz, qw) of the camera mount."""
        # --- STUB: returns identity pose ---
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robot vision bridge client")
    p.add_argument("--server", required=True,
                   help="Server base URL, e.g. http://SERVER_IP:8765")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--max-fps", type=float, default=20.0)
    p.add_argument("--show", action="store_true", help="Show local camera preview")

    sub = p.add_subparsers(dest="mode", help="Operating mode")

    # stream — simple continuous streaming (original behaviour + pose)
    sub.add_parser("stream", help="Continuous streaming")

    # collect — full reconstruction workflow
    c = sub.add_parser("collect", help="Collect -> reconstruct -> plan")
    c.add_argument("--max-frames", type=int, default=50)
    c.add_argument("--min-baseline", type=float, default=0.02,
                   help="Min translation (m) between accepted frames")
    c.add_argument("--duration", type=float, default=30.0,
                   help="Collection duration in seconds (0 = until 'q' key)")
    c.add_argument("--auto-reconstruct", action="store_true", default=True,
                   help="Automatically trigger reconstruction after collection")

    return p.parse_args()


# ======================================================================
# Helpers
# ======================================================================

def ws_url(base: str) -> str:
    """Convert http(s)://... to ws(s)://... /ws/vision."""
    return base.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws/vision"


def api(base: str, path: str, method: str = "GET", **kwargs) -> dict:
    """Call a REST endpoint, return JSON."""
    url = base.rstrip("/") + path
    r = getattr(requests, method.lower())(url, **kwargs)
    r.raise_for_status()
    return r.json()


def encode_message(jpeg_bytes: bytes, pose: tuple[float, ...]) -> bytes:
    """Pack arm pose + JPEG into the binary wire format."""
    header = POSE_STRUCT.pack(*pose)
    return header + jpeg_bytes


# ======================================================================
# Stream mode
# ======================================================================

async def stream_loop(args: argparse.Namespace) -> None:
    arm = ArmPoseSource()
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")

    frame_period = 1.0 / max(args.max_fps, 1e-6)
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    async with websockets.connect(ws_url(args.server), max_size=2**24) as ws:
        print(f"[stream] Connected to {args.server}")
        last_sent = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.01)
                continue

            now = time.perf_counter()
            wait = frame_period - (now - last_sent)
            if wait > 0:
                await asyncio.sleep(wait)

            ok, encoded = cv2.imencode(".jpg", frame, jpeg_params)
            if not ok:
                continue

            pose = arm.get_pose()
            msg = encode_message(encoded.tobytes(), pose)
            await ws.send(msg)
            last_sent = time.perf_counter()

            response_raw = await ws.recv()
            payload = orjson.loads(response_raw)

            ctrl = payload.get("control", {})
            meta = payload.get("meta", {})
            vis = payload.get("vision", {})
            coll = payload.get("collection", {})

            status = (
                f"action={ctrl.get('suggested_action')}"
                f"  bright={vis.get('scene_brightness', 0):.3f}"
                f"  conf={vis.get('feature_confidence', 0):.3f}"
                f"  proc_ms={meta.get('processing_ms', 0):.1f}"
                f"  device={meta.get('device')}"
            )
            if coll.get("active"):
                status += f"  frames={coll.get('n_frames')}"
            if ctrl.get("planned_viewpoints_remaining"):
                status += f"  plan_remaining={ctrl['planned_viewpoints_remaining']}"
            print(status)

            if args.show:
                cv2.imshow("local_camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


# ======================================================================
# Collect mode — full reconstruction workflow
# ======================================================================

async def collect_workflow(args: argparse.Namespace) -> None:
    """
    1. POST /session/start
    2. Stream frames + poses for --duration seconds
    3. POST /session/stop
    4. POST /reconstruct  (if --auto-reconstruct)
    5. Poll /reconstruct/status
    6. GET /reconstruct/result + GET /plan
    """
    base = args.server

    # 1. Start session
    resp = api(base, "/session/start", "POST",
               params={"max_frames": args.max_frames,
                       "min_baseline_m": args.min_baseline})
    sid = resp["session_id"]
    print(f"[collect] Session started: {sid}")

    # 2. Stream frames
    arm = ArmPoseSource()
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")

    frame_period = 1.0 / max(args.max_fps, 1e-6)
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    t_start = time.time()
    n_accepted = 0

    async with websockets.connect(ws_url(base), max_size=2**24) as ws:
        print(f"[collect] Streaming to {base} for {args.duration}s ...")
        last_sent = 0.0
        while True:
            elapsed = time.time() - t_start
            if args.duration > 0 and elapsed > args.duration:
                break

            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.01)
                continue

            now = time.perf_counter()
            wait = frame_period - (now - last_sent)
            if wait > 0:
                await asyncio.sleep(wait)

            ok, encoded = cv2.imencode(".jpg", frame, jpeg_params)
            if not ok:
                continue

            pose = arm.get_pose()
            msg = encode_message(encoded.tobytes(), pose)
            await ws.send(msg)
            last_sent = time.perf_counter()

            response_raw = await ws.recv()
            payload = orjson.loads(response_raw)

            coll = payload.get("collection", {})
            if coll.get("frame_accepted"):
                n_accepted += 1
            print(f"\r[collect] {elapsed:.0f}s  frames={coll.get('n_frames', '?')}"
                  f"  accepted_total={n_accepted}", end="", flush=True)

            if args.show:
                cv2.imshow("collecting", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print()

    # 3. Stop session
    resp = api(base, "/session/stop", "POST")
    print(f"[collect] Session stopped: {resp['n_frames']} frames saved")

    # 4. Trigger reconstruction
    if args.auto_reconstruct:
        print("[collect] Triggering reconstruction ...")
        resp = api(base, "/reconstruct", "POST",
                   params={"session_id": sid})
        print(f"[collect] Reconstruction started on {resp['n_frames']} frames")

        # 5. Poll until done
        while True:
            await asyncio.sleep(2.0)
            status = api(base, "/reconstruct/status")
            if not status["running"]:
                if status.get("error"):
                    print(f"[collect] ERROR: {status['error']}")
                    return
                break
            print("[collect] ... reconstructing ...")

        # 6. Show results
        result = api(base, "/reconstruct/result")
        print("\n=== Reconstruction Result ===")
        print(f"  Input frames:    {result['n_input_frames']}")
        print(f"  Processing time: {result['processing_time_s']}s")
        print(f"  Fused points:    {result['n_fused_points']}")
        print(f"  Mean confidence: {result['mean_confidence']:.4f}")
        unc = result["uncertainty"]
        print(f"  Mean uncertainty: {unc['mean_uncertainty']:.4f}")
        print(f"  Max uncertainty:  {unc['max_uncertainty']:.4f}")
        print(f"  High-unc fraction: {unc['high_uncertainty_fraction']:.2%}")

        occ = result.get("occupancy", {})
        if occ:
            print(f"\n  --- Occupancy Grid ---")
            print(f"  Grid shape:      {occ.get('grid_shape')}")
            print(f"  Voxel size:      {occ.get('voxel_size_m', 0):.4f} m")
            print(f"  FREE:            {occ.get('n_free', 0):,}  ({occ.get('free_fraction', 0):.1%})")
            print(f"  OCCUPIED:        {occ.get('n_occupied', 0):,}  ({occ.get('occupied_fraction', 0):.1%})")
            print(f"  UNKNOWN:         {occ.get('n_unknown', 0):,}")
            print(f"  Explored:        {occ.get('explored_fraction', 0):.1%}")

        plan = api(base, "/plan")
        views = plan.get("viewpoints", [])
        if views:
            print(f"\n=== Next-Best-View Plan ({len(views)} viewpoints) ===")
            for v in views:
                p = v["pose"]
                print(f"  #{v['priority']}  uncertainty={v['expected_uncertainty']:.3f}"
                      f"  pos=({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f})"
                      f"  target=({v['target_world'][0]:.3f}, "
                      f"{v['target_world'][1]:.3f}, {v['target_world'][2]:.3f})")
        else:
            print("\n[collect] No additional viewpoints needed — scene looks complete!")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    cli_args = parse_args()
    mode = cli_args.mode or "stream"
    if mode == "stream":
        asyncio.run(stream_loop(cli_args))
    elif mode == "collect":
        asyncio.run(collect_workflow(cli_args))
    else:
        print(f"Unknown mode: {mode}")
