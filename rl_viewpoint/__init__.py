"""
rl_viewpoint — RL-based next-best-view selection for SO-101 exploration.

This package lives in a separate subfolder so the existing heuristic
pipeline in `active_capture.py` is untouched.

Modules
-------
env         Gymnasium environment wrapping the reconstruction server.
policy      3D-CNN actor-critic network for PPO.
train       PPO training loop.
collect     Offline data-collection script (cache occupancy transitions).
infer       Inference-time wrapper that loads a checkpoint and picks views.
utils       Shared helpers (occupancy downsampling, FK bank, etc.).
"""
