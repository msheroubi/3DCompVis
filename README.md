# Robot Vision Bridge — 3D Reconstruction with VGGT

GPU-accelerated server that accepts a camera stream from a robot arm, reconstructs the 3D scene using [VGGT](https://github.com/facebookresearch/vggt), identifies uncertain regions, and sends the arm back to collect more views.

## Architecture

```
 Local machine (arm + camera)              GPU server
┌─────────────────────────┐          ┌────────────────────────────┐
│  client.py              │          │  server.py                 │
│  ┌───────────┐          │   WS     │  ┌──────────────────────┐  │
│  │ Camera    ├──JPEG+───┼──────────┼──► Real-time analytics  │  │
│  └───────────┘  pose    │          │  │ (brightness, conf)   │  │
│  ┌───────────┐          │          │  └──────────┬───────────┘  │
│  │ Robot arm ├──pose────┘          │             │              │
│  │ (SDK)     ◄──commands─ ─ ─ ─ ─ ┤  ┌──────────▼───────────┐  │
│  └───────────┘           REST      │  │ collector.py         │  │
│                                    │  │ Session buffer       │  │
│                                    │  └──────────┬───────────┘  │
│                                    │             │              │
│                                    │  ┌──────────▼───────────┐  │
│                                    │  │ reconstruction.py    │  │
│                                    │  │ VGGT-1B              │  │
│                                    │  │ → point cloud        │  │
│                                    │  │ → depth + confidence │  │
│                                    │  │ → uncertainty voxels │  │
│                                    │  └──────────┬───────────┘  │
│                                    │             │              │
│                                    │  ┌──────────▼───────────┐  │
│                                    │  │ planner.py           │  │
│                                    │  │ Next-best-view       │  │
│                                    │  │ → arm pose commands  │  │
│                                    │  └──────────────────────┘  │
│                                    └────────────────────────────┘
```

## Workflow

### 1. Install

**Server (GPU machine):**
```bash
pip install -r requirements.txt

# Install VGGT (needed only for reconstruction, not for streaming)
git clone https://github.com/facebookresearch/vggt.git
cd vggt && pip install -e .
```

**Client (local machine):**
```bash
pip install opencv-python websockets orjson requests
```

### 2. Start the server
```bash
python server.py --host 0.0.0.0 --port 8765
# Add --cpu to force CPU mode
```

### 3. Run the client

#### Simple streaming (real-time analytics only)
```bash
python client.py --server http://SERVER_IP:8765 stream --show
```

#### Full reconstruction workflow
```bash
python client.py --server http://SERVER_IP:8765 collect \
    --max-frames 50 \
    --min-baseline 0.02 \
    --duration 30 \
    --auto-reconstruct
```

This will:
1. **Start a collection session** on the server
2. **Stream frames + arm poses** for 30 seconds (or until 50 diverse frames collected)
3. **Stop collection** and save frames to disk
4. **Run VGGT reconstruction** (takes ~3-10s depending on frame count)
5. **Print results**: point cloud stats, uncertainty analysis, and next-best-view commands

### 4. REST API (for manual/programmatic control)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check + device info |
| `/session/start?max_frames=50` | POST | Begin collecting frames |
| `/session/stop` | POST | Finish collection |
| `/session/status` | GET | Current session stats |
| `/reconstruct` | POST | Trigger VGGT reconstruction |
| `/reconstruct/status` | GET | Poll reconstruction progress |
| `/reconstruct/result` | GET | Reconstruction summary + uncertainty |
| `/plan` | GET | Next-best-view commands |
| `/docs` | GET | Interactive Swagger UI |

### 5. Binary wire protocol (WebSocket)

Each message from client → server is:

```
bytes[0:28]   7 × float32 (little-endian): x y z qx qy qz qw  (arm pose)
bytes[28:]    JPEG-encoded frame
```

Legacy clients that send only JPEG bytes are still supported (pose defaults to identity).

## Key modules

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server — streaming, REST endpoints, orchestration |
| `client.py` | Camera capture + arm pose → server, with collect/reconstruct workflow |
| `collector.py` | Frame+pose buffer, spatial diversity filtering, session persistence |
| `reconstruction.py` | VGGT wrapper — 3D point clouds, depth, confidence, voxel uncertainty |
| `planner.py` | Next-best-view — picks camera poses that maximally reduce uncertainty |

## Integrating your robot arm

Edit `ArmPoseSource.get_pose()` in `client.py` to return the real end-effector pose from your robot SDK. The return value should be `(x, y, z, qx, qy, qz, qw)` in metres and unit quaternion.

Example for a UR robot via RTDE:
```python
class ArmPoseSource:
    def __init__(self):
        from rtde_receive import RTDEReceiveInterface
        self.rtde = RTDEReceiveInterface("192.168.1.100")

    def get_pose(self):
        tcp = self.rtde.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        # Convert axis-angle to quaternion ...
        return (x, y, z, qx, qy, qz, qw)
```

## How uncertainty-driven exploration works

1. **VGGT outputs** per-pixel `world_points_conf` and `depth_conf` — these indicate how certain the model is about each 3D point.

2. **Voxelisation** — we discretise the scene into a 64³ voxel grid and compute the mean *inverse confidence* per voxel. Empty/unobserved voxels get maximum uncertainty (1.0).

3. **Cluster selection** — the planner finds the top-*k* high-uncertainty voxel clusters using greedy non-maximum suppression.

4. **Viewpoint generation** — for each uncertain cluster centroid, we place a camera on the ray from scene centre through the centroid, at a standoff distance proportional to the scene extent, with a look-at quaternion pointing at the centroid.

5. **Redundancy filtering** — viewpoints too close to previously visited poses are discarded so the arm always moves to a genuinely new vantage point.

The planned viewpoints are automatically injected into the WebSocket stream as `control.target_pose`, so the arm can start moving immediately.
