"""
Shared helpers for the RL viewpoint pipeline.

Provides:
  • ``downsample_occupancy``  — pool a 64³ voxel grid down to 16³
  • ``build_fk_bank``         — precompute FK positions/directions for
                                the entire candidate joint-angle bank
  • ``match_to_bank``         — find the nearest bank entry to a
                                (x, y, z, wrist_roll) Cartesian target
  • Joint-limit constants re-exported from active_capture
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Occupancy constants (must match reconstruction.py)
# ---------------------------------------------------------------------------
UNKNOWN: int = 0
FREE: int = 1
OCCUPIED: int = 2


# ---------------------------------------------------------------------------
# Joint limits (mirrored from active_capture.py)
# ---------------------------------------------------------------------------
SO101_JOINT_LIMITS: list[tuple[float, float]] = [
    (-90.0,  90.0),   # J0: base rotation
    (-90.0,   0.0),   # J1: shoulder
    ( 20.0, 110.0),   # J2: elbow
    (-60.0,  10.0),   # J3: wrist pitch
    (-90.0,  90.0),   # J4: wrist roll
    (  0.0,  50.0),   # J5: gripper / camera tilt
]


# ---------------------------------------------------------------------------
# Occupancy downsampling  (64³ → target_res³, default 16³)
# ---------------------------------------------------------------------------

def downsample_occupancy(
    grid: np.ndarray,
    target_res: int = 16,
) -> np.ndarray:
    """Down-sample a voxel occupancy grid with priority-aware pooling.

    Pooling rule per block:
      OCCUPIED (2) > FREE (1) > UNKNOWN (0)

    i.e. if *any* voxel in the block is OCCUPIED the output voxel is
    OCCUPIED; otherwise if *any* is FREE it is FREE; else UNKNOWN.

    Parameters
    ----------
    grid : np.ndarray, shape (G, G, G), dtype uint8
        Full-resolution occupancy grid.
    target_res : int
        Output resolution per axis (must evenly divide G).

    Returns
    -------
    np.ndarray, shape (target_res, target_res, target_res), dtype uint8
    """
    G = grid.shape[0]
    assert grid.shape == (G, G, G), f"Expected cubic grid, got {grid.shape}"
    assert G % target_res == 0, (
        f"Grid size {G} must be divisible by target_res {target_res}")
    factor = G // target_res

    # Reshape to (target_res, factor, target_res, factor, target_res, factor)
    reshaped = grid.reshape(
        target_res, factor,
        target_res, factor,
        target_res, factor,
    )

    # Priority pooling via max over the block axes (1, 3, 5)
    # Since OCCUPIED=2 > FREE=1 > UNKNOWN=0, np.max does exactly the
    # right thing.
    out = reshaped.max(axis=(1, 3, 5)).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# FK bank: precompute Cartesian positions + viewing directions
# ---------------------------------------------------------------------------

def build_fk_bank(
    candidates: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FK for every candidate and return structured arrays.

    Returns
    -------
    positions  : (N, 3)  camera position in world frame
    directions : (N, 3)  *forward* viewing direction (negated FK z-axis)
    wrist_rolls: (N,)    J4 joint angle in degrees (the action wrist_roll)
    """
    try:
        from so101_kinematics import ee_pose
    except ImportError:
        raise ImportError(
            "so101_kinematics is required for FK. "
            "Make sure it is on PYTHONPATH.")

    positions = []
    directions = []
    wrist_rolls = []
    for vp in candidates:
        T_c2w = ee_pose(np.array(vp, dtype=float))
        positions.append(T_c2w[:3, 3].copy())
        # FK z-axis points away from scene → negate for forward direction
        fwd = -T_c2w[:3, 2].copy()
        fwd /= np.linalg.norm(fwd) + 1e-12
        directions.append(fwd)
        wrist_rolls.append(vp[4])  # J4 = wrist roll
    return (
        np.array(positions, dtype=np.float32),
        np.array(directions, dtype=np.float32),
        np.array(wrist_rolls, dtype=np.float32),
    )


def match_to_bank(
    target_xyz: np.ndarray,        # (3,)
    target_wrist_roll: float,      # degrees
    bank_positions: np.ndarray,    # (N, 3)
    bank_wrist_rolls: np.ndarray,  # (N,)
    pos_weight: float = 1.0,
    roll_weight: float = 0.01,     # 1 deg ≈ 1 cm
) -> int:
    """Return the index of the bank entry closest to the Cartesian target.

    The distance metric is:
        d = pos_weight * ||pos - target_xyz||
          + roll_weight * |roll - target_wrist_roll|

    Parameters
    ----------
    target_xyz : (3,)
        Desired end-effector position in world frame (metres).
    target_wrist_roll : float
        Desired wrist roll in degrees.
    bank_positions : (N, 3)
    bank_wrist_rolls : (N,)
    pos_weight, roll_weight : float
        Relative weighting between position and roll distance.

    Returns
    -------
    int — index into the bank.
    """
    pos_dist = np.linalg.norm(bank_positions - target_xyz[None, :], axis=1)
    roll_dist = np.abs(bank_wrist_rolls - target_wrist_roll)
    combined = pos_weight * pos_dist + roll_weight * roll_dist
    return int(np.argmin(combined))


# ---------------------------------------------------------------------------
# Normalisation helpers for the action space
# ---------------------------------------------------------------------------

def action_to_cartesian(
    action: np.ndarray,
    pos_low: np.ndarray,
    pos_high: np.ndarray,
    roll_low: float,
    roll_high: float,
) -> tuple[np.ndarray, float]:
    """Map a normalised [-1, 1]⁴ action to (xyz_metres, wrist_roll_deg).

    action[0:3] → position  (linearly scaled to [pos_low, pos_high])
    action[3]   → wrist_roll (linearly scaled to [roll_low, roll_high])
    """
    a = np.clip(action, -1.0, 1.0)
    xyz = pos_low + (a[:3] + 1.0) * 0.5 * (pos_high - pos_low)
    wrist = roll_low + (a[3] + 1.0) * 0.5 * (roll_high - roll_low)
    return xyz.astype(np.float32), float(wrist)


def cartesian_to_action(
    xyz: np.ndarray,
    wrist_roll: float,
    pos_low: np.ndarray,
    pos_high: np.ndarray,
    roll_low: float,
    roll_high: float,
) -> np.ndarray:
    """Inverse of ``action_to_cartesian``: Cartesian → normalised [-1,1]⁴."""
    a_pos = 2.0 * (xyz - pos_low) / (pos_high - pos_low + 1e-8) - 1.0
    a_roll = 2.0 * (wrist_roll - roll_low) / (roll_high - roll_low + 1e-8) - 1.0
    return np.clip(np.append(a_pos, a_roll), -1.0, 1.0).astype(np.float32)
