"""
Frame + robot pose collector for 3D reconstruction sessions.

Manages a buffer of (image, arm_pose, timestamp) tuples, selects
a diverse subset for reconstruction, and persists sessions to disk.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import orjson


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArmPose:
    """6‑DoF pose of the robot end‑effector (camera mount)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw],
                        dtype=np.float64)

    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_dict(d: dict) -> "ArmPose":
        return ArmPose(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
            qx=float(d.get("qx", 0.0)),
            qy=float(d.get("qy", 0.0)),
            qz=float(d.get("qz", 0.0)),
            qw=float(d.get("qw", 1.0)),
        )

    def to_dict(self) -> dict:
        return dict(x=self.x, y=self.y, z=self.z,
                    qx=self.qx, qy=self.qy, qz=self.qz, qw=self.qw)


@dataclass
class CapturedFrame:
    """A single captured frame with its associated metadata."""
    frame_bgr: np.ndarray
    arm_pose: ArmPose
    timestamp: float
    frame_id: int = 0


@dataclass
class CollectionSession:
    """Manages one reconstruction session's worth of captured data."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    frames: list[CapturedFrame] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "collecting"  # collecting | ready | reconstructing | done

    # Selection config
    max_frames: int = 50
    min_baseline_m: float = 0.02  # minimum translation between selected frames

    def add_frame(self, frame_bgr: np.ndarray, arm_pose: ArmPose) -> bool:
        """Add a frame. Returns True if it was accepted (enough baseline)."""
        if len(self.frames) >= self.max_frames:
            return False

        # Always accept the first frame
        if len(self.frames) == 0:
            cf = CapturedFrame(
                frame_bgr=frame_bgr,
                arm_pose=arm_pose,
                timestamp=time.time(),
                frame_id=0,
            )
            self.frames.append(cf)
            return True

        # Check spatial diversity — reject frames too close to the last
        last_pos = self.frames[-1].arm_pose.translation()
        cur_pos = arm_pose.translation()
        baseline = float(np.linalg.norm(cur_pos - last_pos))

        if baseline < self.min_baseline_m:
            return False

        cf = CapturedFrame(
            frame_bgr=frame_bgr,
            arm_pose=arm_pose,
            timestamp=time.time(),
            frame_id=len(self.frames),
        )
        self.frames.append(cf)
        return True

    def select_subset(self, n: int = 20) -> list[CapturedFrame]:
        """Select a spatially diverse subset of *n* frames for reconstruction.

        Uses farthest‑point sampling on the arm translations
        so the selected views maximise coverage of the workspace.
        """
        if len(self.frames) <= n:
            return list(self.frames)

        positions = np.array([f.arm_pose.translation() for f in self.frames])
        selected_idx: list[int] = [0]  # start with the first frame
        min_dists = np.full(len(self.frames), np.inf)

        for _ in range(n - 1):
            last = positions[selected_idx[-1]]
            dists = np.linalg.norm(positions - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            # mask already selected
            min_dists[selected_idx] = -1.0
            next_idx = int(np.argmax(min_dists))
            selected_idx.append(next_idx)

        selected_idx.sort()
        return [self.frames[i] for i in selected_idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, base_dir: str = "sessions") -> Path:
        """Save frames as JPEGs + a metadata JSON."""
        out = Path(base_dir) / self.session_id
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        meta_entries = []
        for cf in self.frames:
            fname = f"frame_{cf.frame_id:04d}.jpg"
            cv2.imwrite(str(img_dir / fname), cf.frame_bgr)
            meta_entries.append({
                "frame_id": cf.frame_id,
                "file": fname,
                "timestamp": cf.timestamp,
                "arm_pose": cf.arm_pose.to_dict(),
            })

        meta = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "n_frames": len(self.frames),
            "frames": meta_entries,
        }
        (out / "meta.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        return out

    @staticmethod
    def load(session_dir: str) -> "CollectionSession":
        """Load a previously saved session from disk."""
        p = Path(session_dir)
        meta = orjson.loads((p / "meta.json").read_bytes())
        session = CollectionSession(
            session_id=meta["session_id"],
            created_at=meta["created_at"],
        )
        for entry in meta["frames"]:
            img = cv2.imread(str(p / "images" / entry["file"]))
            cf = CapturedFrame(
                frame_bgr=img,
                arm_pose=ArmPose.from_dict(entry["arm_pose"]),
                timestamp=entry["timestamp"],
                frame_id=entry["frame_id"],
            )
            session.frames.append(cf)
        session.status = "ready"
        return session


# ---------------------------------------------------------------------------
# Global session manager (used by the server)
# ---------------------------------------------------------------------------

class SessionManager:
    """Thin wrapper that keeps track of the active session."""

    def __init__(self, sessions_dir: str = "sessions") -> None:
        self.sessions_dir = sessions_dir
        self.active: Optional[CollectionSession] = None
        self.history: dict[str, CollectionSession] = {}

    def new_session(self, max_frames: int = 50, min_baseline_m: float = 0.02) -> CollectionSession:
        session = CollectionSession(max_frames=max_frames, min_baseline_m=min_baseline_m)
        self.active = session
        self.history[session.session_id] = session
        return session

    def finish_collection(self) -> Optional[CollectionSession]:
        if self.active is None:
            return None
        self.active.status = "ready"
        self.active.save(self.sessions_dir)
        finished = self.active
        self.active = None
        return finished
