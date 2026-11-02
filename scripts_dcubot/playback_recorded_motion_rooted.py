"""Like playback_recorded_motion.py, but also replays root translation/rotation
(the original script discards these because dcubot.urdf's base_link has no
freejoint -- it adds one at load time via MjSpec instead). Without this, the
robot walks in place and the feet swing through an unanchored arc that looks
wrong, because the real recorded motion has the body moving ~3.7m forward
over the clip while the stance foot stays planted near-constant height.

Usage:
    uv run --with mujoco python scripts_dcubot/playback_recorded_motion_rooted.py [recording.json]
"""
import json
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

URDF = "/home/ahn/dcubot/DCUBOT/reference_motion_generator/open_duck_reference_motion_generator/robots/dcubot/dcubot.urdf"
RECORDING = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ahn/dcubot/DCUBOT/reference_motion_generator/motions_dcubot/forward_medium_0.02_0.0_0.0.json"

with open(RECORDING) as f:
    data = json.load(f)

frames = np.array(data["Frames"])
offs = data["Frame_offset"][0]
sizes = data["Frame_size"][0]
joint_names = data["Joints"]
fps = data["FPS"]

root_pos = frames[:, offs["root_pos"]:offs["root_pos"] + sizes["root_pos"]]
root_quat_xyzw = frames[:, offs["root_quat"]:offs["root_quat"] + sizes["root_quat"]]
joints_pos = frames[:, offs["joints_pos"]:offs["joints_pos"] + sizes["joints_pos"]]

spec = mujoco.MjSpec.from_file(URDF)
base = spec.body("base_link")
base.add_freejoint()
# Bare URDF has no ground -- add one so forward motion reads against a fixed
# reference (the robot-only MJCF scene files add this too, but this script
# loads the raw robot URDF directly).
spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[5, 5, 0.01], rgba=[0.6, 0.6, 0.65, 1],
)
m = spec.compile()
d = mujoco.MjData(m)

joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in joint_names]
qadrs = [m.jnt_qposadr[jid] for jid in joint_ids]

print("재생할 joint 순서 (전부 라디안):", joint_names)
print(f"root_pos x 범위: {root_pos[:, 0].min():.2f} ~ {root_pos[:, 0].max():.2f} m "
      "(이번엔 실제로 전진합니다)")

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Ground-fixed camera (does NOT follow the robot, unlike the earlier
    # version) -- framed once, from the side, wide enough to show the whole
    # recorded path so forward motion reads against the fixed floor, same as
    # the Placo/meshcat view.
    x_mid = (root_pos[:, 0].min() + root_pos[:, 0].max()) / 2
    x_span = max(root_pos[:, 0].max() - root_pos[:, 0].min(), 0.3)
    viewer.cam.lookat[:] = [x_mid, root_pos[0, 1], 0.15]
    viewer.cam.distance = x_span * 1.6 + 0.5
    viewer.cam.azimuth = 90   # looking along -Y, so +X (forward) goes left->right
    viewer.cam.elevation = -15

    i = 0
    n = len(frames)
    while viewer.is_running():
        d.qpos[0:3] = root_pos[i]
        x, y, z, w = root_quat_xyzw[i]
        d.qpos[3:7] = [w, x, y, z]  # mujoco freejoint quat order is wxyz
        for qadr, val in zip(qadrs, joints_pos[i]):
            d.qpos[qadr] = val
        mujoco.mj_forward(m, d)
        viewer.sync()
        time.sleep(1.0 / fps)
        i = (i + 1) % n
