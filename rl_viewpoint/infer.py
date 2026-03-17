#!/usr/bin/env python3
"""
Inference-time viewpoint selection using a trained RL policy.

Loads a PPO checkpoint and replaces the heuristic ``_select_next_viewpoints``
from ``active_capture.py`` with the learned policy.  Talks to the real arm
+ GPU server.

Usage
-----
    python -m rl_viewpoint.infer \
        --checkpoint checkpoints/viewpoint_ppo/best_model.pt \
        --server http://GPU_IP:8765 \
        --port /dev/ttyACM1 \
        --camera-id 2 \
        --max-rounds 10

The loop structure mirrors ``auto_nbv_explore`` but the next-viewpoint
decision comes from the policy instead of information-gain scoring.
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
import torch

logger = logging.getLogger(__name__)

# Parent pipeline imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from active_capture import (  # noqa: E402
    DEFAULT_VIEWPOINTS,
    REST_POSITION,
    UNKNOWN,
    POSE_STRUCT,
    _HAS_FK,
    _HAS_LEROBOT,
    _precompute_fk_data,
    api,
    encode_message,
    generate_viewpoint_bank,
    joint_to_pose7,
    move_arm,
    print_reconstruction_result,
    read_joint_positions,
    return_to_rest,
    ws_url,
)

from rl_viewpoint.policy import ViewpointActorCritic
from rl_viewpoint.utils import (
    build_fk_bank,
    downsample_occupancy,
    match_to_bank,
    action_to_cartesian,
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


def load_policy(
    checkpoint_path: str,
    obs_res: int = 16,
    hidden_dim: int = 128,
    device: torch.device | None = None,
) -> ViewpointActorCritic:
    """Load a trained ViewpointActorCritic from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViewpointActorCritic(
        grid_res=obs_res, hidden_dim=hidden_dim, action_dim=4).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded policy from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"avg_reward={ckpt.get('avg_reward', '?')})")
    return model


def policy_select_viewpoint(
    model: ViewpointActorCritic,
    occupancy: np.ndarray,          # (64, 64, 64)
    cam_pos: np.ndarray,            # (3,)
    cam_dir: np.ndarray,            # (3,)
    bank_positions: np.ndarray,     # (N, 3)
    bank_rolls: np.ndarray,         # (N,)
    pos_low: np.ndarray,
    pos_high: np.ndarray,
    roll_low: float,
    roll_high: float,
    obs_res: int = 16,
    device: torch.device | None = None,
    deterministic: bool = True,
) -> int:
    """Use the policy to select the best bank candidate index.

    Returns the index into the FK bank.
    """
    if device is None:
        device = next(model.parameters()).device

    ds_occ = downsample_occupancy(occupancy, obs_res)
    explored = np.count_nonzero(occupancy != UNKNOWN) / occupancy.size

    obs_t = {
        "occupancy": torch.tensor(
            ds_occ[np.newaxis, np.newaxis], dtype=torch.float32, device=device),
        "cam_pos": torch.tensor(
            cam_pos[np.newaxis], dtype=torch.float32, device=device),
        "cam_dir": torch.tensor(
            cam_dir[np.newaxis], dtype=torch.float32, device=device),
        "explored": torch.tensor(
            [[explored]], dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        action, _, _ = model.act(obs_t, deterministic=deterministic)
    action_np = action.cpu().numpy().squeeze(0)

    target_xyz, target_roll = action_to_cartesian(
        action_np, pos_low, pos_high, roll_low, roll_high)
    bank_idx = match_to_bank(target_xyz, target_roll, bank_positions, bank_rolls)
    return bank_idx


async def rl_explore(args: argparse.Namespace, robot) -> None:
    """RL-policy-driven exploration loop (replaces auto_nbv_explore)."""
    import websockets
    import orjson

    base = args.server
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load policy ───────────────────────────────────────────────
    model = load_policy(
        args.checkpoint,
        obs_res=args.obs_res,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    # ── Init scene ────────────────────────────────────────────────
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

        try:
            requests.post(base.rstrip("/") + "/scene/init", timeout=30, json={
                "center": scene_center.tolist(),
                "forward": cam_fwd.tolist(),
                "half_extent": args.scene_extent,
                "bounds_type": args.bounds_type,
            }).raise_for_status()
        except Exception as e:
            logger.warning("Scene init failed: %s", e)

    # ── Build FK bank ─────────────────────────────────────────────
    candidates = generate_viewpoint_bank(n=args.bank_size, include_defaults=True)
    bank_pos, bank_dir, bank_rolls = build_fk_bank(candidates)
    pos_low = bank_pos.min(axis=0) - 0.02
    pos_high = bank_pos.max(axis=0) + 0.02
    roll_low = float(bank_rolls.min()) - 1.0
    roll_high = float(bank_rolls.max()) + 1.0

    visited: set[int] = set()
    current_occ = np.zeros((64, 64, 64), dtype=np.uint8)

    # ── Seed: capture reference frame ─────────────────────────────
    print("[RL] Capturing reference frame ...")
    resp = api(base, "/session/start", "POST",
               params={"max_frames": 5, "min_baseline_m": 0.0})
    ref_sid = resp["session_id"]
    async with websockets.connect(ws_url(base), max_size=2**24) as ws:
        obs = robot.get_observation()
        frame = obs.get("wrist_cam")
        if frame is not None:
            joints = read_joint_positions(robot)
            pose = joint_to_pose7(joints)
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                await ws.send(encode_message(enc.tobytes(), pose))
                _ = await ws.recv()
    api(base, "/session/stop", "POST")

    # ── Main exploration loop ─────────────────────────────────────
    for round_n in range(1, args.max_rounds + 1):
        # Get current camera state
        joints_now = read_joint_positions(robot)
        if _HAS_FK:
            T_now = ee_pose(joints_now)
            cur_pos = T_now[:3, 3].astype(np.float32).copy()
            cur_dir = (-T_now[:3, 2]).copy()
            cur_dir /= np.linalg.norm(cur_dir) + 1e-12
            cur_dir = cur_dir.astype(np.float32)
        else:
            cur_pos = np.zeros(3, dtype=np.float32)
            cur_dir = np.array([0, 0, 1], dtype=np.float32)

        # ── Policy picks viewpoints ───────────────────────────────
        batch: list[int] = []
        for _ in range(args.views_per_round):
            idx = policy_select_viewpoint(
                model, current_occ, cur_pos, cur_dir,
                bank_pos, bank_rolls,
                pos_low, pos_high, roll_low, roll_high,
                obs_res=args.obs_res, device=device,
                deterministic=not args.stochastic,
            )
            # Avoid revisiting the same candidate in this batch
            attempts = 0
            while idx in visited and attempts < 20:
                # Add noise and re-query
                noisy_occ = current_occ.copy()
                idx = policy_select_viewpoint(
                    model, noisy_occ, cur_pos, cur_dir,
                    bank_pos, bank_rolls,
                    pos_low, pos_high, roll_low, roll_high,
                    obs_res=args.obs_res, device=device,
                    deterministic=False,
                )
                attempts += 1
            batch.append(idx)
            visited.add(idx)

        print(f"\n{'='*60}")
        print(f"  RL EXPLORE  round={round_n}  batch={batch}  "
              f"visited={len(visited)}/{len(candidates)}")
        print(f"{'='*60}")

        # ── Capture frames at selected viewpoints ─────────────────
        resp = api(base, "/session/start", "POST",
                   params={"max_frames": 50, "min_baseline_m": 0.02})
        sid = resp["session_id"]
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        async with websockets.connect(ws_url(base), max_size=2**24) as ws:
            for vi in batch:
                vp = candidates[vi]
                print(f"  → VP {vi}: {vp}")
                move_arm(robot, vp)
                time.sleep(0.2)
                obs = robot.get_observation()
                frame = obs.get("wrist_cam")
                if frame is None:
                    continue
                joints = read_joint_positions(robot)
                pose = joint_to_pose7(joints)
                ok, enc = cv2.imencode(".jpg", frame, jpeg_params)
                if ok:
                    await ws.send(encode_message(enc.tobytes(), pose))
                    _ = await ws.recv()

        api(base, "/session/stop", "POST")

        # ── Reconstruct ───────────────────────────────────────────
        print("[reconstruct] Running VGGT (merge all) ...")
        api(base, "/reconstruct", "POST",
            params={"session_id": sid, "merge_all": True})
        for _ in range(60):
            await asyncio.sleep(2.0)
            status = api(base, "/reconstruct/status")
            if not status["running"]:
                break

        result = api(base, "/reconstruct/result")
        print_reconstruction_result(result)

        # ── Fetch updated occupancy ───────────────────────────────
        new_occ, _, _ = _fetch_occupancy(base)
        if new_occ is not None:
            current_occ = new_occ

        explored = result.get("occupancy", {}).get("explored_fraction", 0)
        if explored >= args.stop_explored:
            print(f"\n  Explored {explored:.1%} — target reached. Done.")
            break

        unvisited = [i for i in range(len(candidates)) if i not in visited]
        if not unvisited:
            print("  All candidates visited.")
            break

    print(f"\nRL exploration finished: {len(visited)}/{len(candidates)} "
          f"visited in {round_n} rounds.")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL-policy viewpoint selection (inference)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--server", required=True)
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="kowalski_follower")
    parser.add_argument("--camera-id", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--obs-res", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--bank-size", type=int, default=200)
    parser.add_argument("--views-per-round", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--stop-explored", type=float, default=0.90)
    parser.add_argument("--scene-extent", type=float, default=2.4)
    parser.add_argument("--bounds-type", default="hemisphere")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (sample) instead of mean")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not _HAS_LEROBOT:
        print("ERROR: lerobot not installed.")
        sys.exit(1)

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

    try:
        asyncio.run(rl_explore(args, robot))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        return_to_rest(robot)
        robot.disconnect()


if __name__ == "__main__":
    main()
