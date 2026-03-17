#!/usr/bin/env python3
"""
Collect occupancy-grid transitions for offline RL training.

Runs the existing heuristic exploration pipeline (``auto_nbv_explore``)
and records every (occupancy_before, action, occupancy_after) transition
into a ``.npz`` file that ``CachedViewpointEnv`` can replay.

This script drives the **real arm** and talks to the **GPU server** —
it is essentially a data-logging wrapper around the existing pipeline.

Usage
-----
    python -m rl_viewpoint.collect \
        --server http://GPU_IP:8765 \
        --port /dev/ttyACM1 \
        --camera-id 2 \
        --episodes 5 \
        --output transitions.npz

Each episode runs a full exploration loop and stores one transition per
viewpoint visit (i.e. per reconstruction round × viewpoints-per-round).

Output ``.npz`` keys
--------------------
  occ_before  : (N, 64, 64, 64)  uint8
  occ_after   : (N, 64, 64, 64)  uint8
  action_xyz  : (N, 3)           float32 — camera position of chosen vp
  action_roll : (N,)             float32 — wrist roll (J4) in degrees
  cam_pos     : (N, 3)           float32 — camera pos before acting
  cam_dir     : (N, 3)           float32 — camera dir before acting
  episode_ids : (N,)             int32   — which episode this belongs to
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import logging
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Re-use constants / functions from the main pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from active_capture import (  # noqa: E402
    DEFAULT_VIEWPOINTS,
    REST_POSITION,
    UNKNOWN,
    FREE,
    OCCUPIED,
    POSE_STRUCT,
    _HAS_FK,
    _HAS_LEROBOT,
    _precompute_fk_data,
    api,
    encode_message,
    generate_viewpoint_bank,
    joint_to_pose7,
    move_arm,
    read_joint_positions,
    return_to_rest,
    validate_joint_angles,
    ws_url,
)


def _fetch_occupancy(base: str) -> tuple[np.ndarray | None, np.ndarray | None, float]:
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
    grid = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    return grid, origin, voxel_size


async def collect_episode(
    args: argparse.Namespace,
    robot,
    episode_id: int,
    transitions: dict[str, list],
) -> int:
    """Run one exploration episode and append transitions to *transitions*.

    Returns number of transitions collected.
    """
    import websockets
    import orjson

    base = args.server
    n_collected = 0

    # ── Init scene bounds (same as active_capture) ────────────────
    if _HAS_FK:
        from so101_kinematics import ee_pose
        ref_vp = DEFAULT_VIEWPOINTS[0]
        move_arm(robot, ref_vp)
        time.sleep(0.5)
        T_c2w = ee_pose(np.array(ref_vp, dtype=float))
        cam_pos = T_c2w[:3, 3].copy()
        cam_fwd = -T_c2w[:3, 2].copy()
        cam_fwd /= np.linalg.norm(cam_fwd) + 1e-12
        scene_center = cam_pos + cam_fwd * 0.25
        scene_forward = cam_fwd.copy()

        try:
            requests.post(base.rstrip("/") + "/scene/init", timeout=30, json={
                "center": scene_center.tolist(),
                "forward": scene_forward.tolist(),
                "half_extent": getattr(args, "scene_extent", 2.4),
                "bounds_type": getattr(args, "bounds_type", "hemisphere"),
            }).raise_for_status()
        except Exception as e:
            logger.warning("Scene init failed: %s", e)
    else:
        scene_center = None
        scene_forward = None

    # ── Build viewpoint bank ──────────────────────────────────────
    candidates = generate_viewpoint_bank(n=args.bank_size, include_defaults=True)
    cam_positions, cam_directions = _precompute_fk_data(candidates)
    visited: set[int] = set()

    # Initial occupancy = all UNKNOWN
    current_occ = np.zeros((64, 64, 64), dtype=np.uint8)

    # ── Seed phase ────────────────────────────────────────────────
    seed_count = min(args.seed_views, len(candidates))
    next_batch = list(range(seed_count))

    round_n = 0
    while round_n < args.max_rounds and next_batch:
        round_n += 1
        print(f"  [Ep {episode_id}] Round {round_n}: visiting {next_batch}")

        # Store occ_before for each viewpoint in the batch
        occ_before_snap = current_occ.copy()

        # Record camera state before acting
        joints_before = read_joint_positions(robot)
        if _HAS_FK:
            T_before = ee_pose(joints_before)
            pos_before = T_before[:3, 3].astype(np.float32).copy()
            dir_before = (-T_before[:3, 2]).copy()
            dir_before /= np.linalg.norm(dir_before) + 1e-12
            dir_before = dir_before.astype(np.float32)
        else:
            pos_before = np.zeros(3, dtype=np.float32)
            dir_before = np.array([0, 0, 1], dtype=np.float32)

        # ── Capture batch ─────────────────────────────────────────
        resp = api(base, "/session/start", "POST",
                   params={"max_frames": 50, "min_baseline_m": 0.02})
        sid = resp["session_id"]
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        async with websockets.connect(ws_url(base), max_size=2**24) as ws:
            for vi in next_batch:
                vp = candidates[vi]
                move_arm(robot, vp)
                visited.add(vi)
                time.sleep(0.2)

                obs = robot.get_observation()
                frame = obs.get("wrist_cam")
                if frame is None:
                    continue
                joint_deg = read_joint_positions(robot)
                pose = joint_to_pose7(joint_deg)
                ok, encoded = cv2.imencode(".jpg", frame, jpeg_params)
                if ok:
                    await ws.send(encode_message(encoded.tobytes(), pose))
                    _ = await ws.recv()

        api(base, "/session/stop", "POST")

        # ── Reconstruct ───────────────────────────────────────────
        api(base, "/reconstruct", "POST",
            params={"session_id": sid, "merge_all": True})
        for _ in range(60):
            time.sleep(2.0)
            status = api(base, "/reconstruct/status")
            if not status["running"]:
                break

        # ── Fetch new occupancy ───────────────────────────────────
        new_occ, _, _ = _fetch_occupancy(base)
        if new_occ is None:
            print(f"  [Ep {episode_id}] No occupancy returned — skipping round")
            continue
        new_occ = new_occ.copy()

        # ── Store one transition per viewpoint in the batch ───────
        for vi in next_batch:
            vp = candidates[vi]
            transitions["occ_before"].append(occ_before_snap)
            transitions["occ_after"].append(new_occ)
            transitions["action_xyz"].append(cam_positions[vi].astype(np.float32))
            transitions["action_roll"].append(float(vp[4]))  # J4
            transitions["cam_pos"].append(pos_before)
            transitions["cam_dir"].append(dir_before)
            transitions["episode_ids"].append(episode_id)
            n_collected += 1

        current_occ = new_occ

        # ── Check explored fraction ───────────────────────────────
        explored = np.count_nonzero(current_occ != UNKNOWN) / current_occ.size
        print(f"  [Ep {episode_id}] Round {round_n} explored: {explored:.1%}")
        if explored >= args.stop_explored:
            print(f"  [Ep {episode_id}] Target reached.")
            break

        # ── Pick next batch (simple: take next unvisited) ─────────
        unvisited = [i for i in range(len(candidates)) if i not in visited]
        if not unvisited:
            break
        next_batch = unvisited[:args.views_per_round]

    return n_collected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect occupancy transitions for offline RL training")
    parser.add_argument("--server", required=True)
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="kowalski_follower")
    parser.add_argument("--camera-id", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of exploration episodes to record")
    parser.add_argument("--output", default="transitions.npz",
                        help="Output .npz file path")

    # Exploration params (same as active_capture)
    parser.add_argument("--bank-size", type=int, default=200)
    parser.add_argument("--seed-views", type=int, default=3)
    parser.add_argument("--views-per-round", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--stop-explored", type=float, default=0.90)
    parser.add_argument("--scene-extent", type=float, default=2.4)
    parser.add_argument("--bounds-type", default="hemisphere")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not _HAS_LEROBOT:
        print("ERROR: lerobot not installed.")
        sys.exit(1)

    # Connect robot
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

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

    transitions: dict[str, list] = {
        "occ_before": [],
        "occ_after": [],
        "action_xyz": [],
        "action_roll": [],
        "cam_pos": [],
        "cam_dir": [],
        "episode_ids": [],
    }

    total = 0
    try:
        for ep in range(args.episodes):
            print(f"\n{'='*60}")
            print(f"  COLLECTION EPISODE {ep + 1}/{args.episodes}")
            print(f"{'='*60}")

            # Reset server between episodes
            try:
                api(args.server, "/reset", "POST")
            except Exception:
                pass

            n = asyncio.run(collect_episode(args, robot, ep, transitions))
            total += n
            print(f"  Episode {ep + 1} collected {n} transitions (total: {total})")

            # Return to rest between episodes
            return_to_rest(robot)
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        return_to_rest(robot)
        robot.disconnect()

    # Save
    if total > 0:
        np.savez_compressed(
            args.output,
            occ_before=np.array(transitions["occ_before"], dtype=np.uint8),
            occ_after=np.array(transitions["occ_after"], dtype=np.uint8),
            action_xyz=np.array(transitions["action_xyz"], dtype=np.float32),
            action_roll=np.array(transitions["action_roll"], dtype=np.float32),
            cam_pos=np.array(transitions["cam_pos"], dtype=np.float32),
            cam_dir=np.array(transitions["cam_dir"], dtype=np.float32),
            episode_ids=np.array(transitions["episode_ids"], dtype=np.int32),
        )
        print(f"\nSaved {total} transitions to {args.output}")
    else:
        print("\nNo transitions collected.")


if __name__ == "__main__":
    main()
