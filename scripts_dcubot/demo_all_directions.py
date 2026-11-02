"""Cycle through forward / backward / strafe left / strafe right / turn left /
turn right, each for a fixed duration, ground-fixed camera (not following the
robot) reframed per segment -- same setup as playback_recorded_motion_rooted.py,
extended to sequence multiple recordings in one viewer session.

Usage:
    cd reference_motion_generator
    .venv/bin/python scripts_dcubot/demo_all_directions.py [seconds_per_segment]
"""
import json
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

URDF = "/home/ahn/dcubot/DCUBOT/reference_motion_generator/open_duck_reference_motion_generator/robots/dcubot/dcubot.urdf"
MOTIONS_DIR = "/home/ahn/dcubot/DCUBOT/reference_motion_generator/motions_dcubot"
SECONDS_PER_SEGMENT = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

SEGMENTS = [
    ("forward", "forward_medium_0.02_0.0_0.0.json"),
    ("backward", "backward_-0.04_0.0_0.0.json"),
    ("strafe_left", "strafe_left_0.0_0.04_0.0.json"),
    ("strafe_right", "strafe_right_0.0_-0.04_0.0.json"),
    ("turn_left", "turn_left_0.0_0.0_0.5.json"),
    ("turn_right", "turn_right_0.0_0.0_-0.5.json"),
]


def load_motion(path):
    with open(path) as f:
        data = json.load(f)
    frames = np.array(data["Frames"])
    offs = data["Frame_offset"][0]
    sizes = data["Frame_size"][0]
    fps = data["FPS"]
    n_keep = min(len(frames), int(SECONDS_PER_SEGMENT * fps))
    frames = frames[:n_keep]
    return {
        "joint_names": data["Joints"],
        "fps": fps,
        "root_pos": frames[:, offs["root_pos"]:offs["root_pos"] + sizes["root_pos"]],
        "root_quat_xyzw": frames[:, offs["root_quat"]:offs["root_quat"] + sizes["root_quat"]],
        "joints_pos": frames[:, offs["joints_pos"]:offs["joints_pos"] + sizes["joints_pos"]],
    }


motions = {name: load_motion(f"{MOTIONS_DIR}/{fname}") for name, fname in SEGMENTS}

spec = mujoco.MjSpec.from_file(URDF)
base = spec.body("base_link")
base.add_freejoint()
spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[5, 5, 0.01], rgba=[0.6, 0.6, 0.65, 1],
)
m = spec.compile()
d = mujoco.MjData(m)

joint_ids_by_motion = {
    name: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in mo["joint_names"]]
    for name, mo in motions.items()
}
qadrs_by_motion = {
    name: [m.jnt_qposadr[jid] for jid in jids] for name, jids in joint_ids_by_motion.items()
}

print(f"segments ({SECONDS_PER_SEGMENT:.0f}s each): {[s[0] for s in SEGMENTS]}")

with mujoco.viewer.launch_passive(m, d) as viewer:
    seg_idx = 0
    frame_idx = 0
    seg_start = time.time()

    def frame_camera(mo):
        rp = mo["root_pos"]
        x_mid = (rp[:, 0].min() + rp[:, 0].max()) / 2
        y_mid = (rp[:, 1].min() + rp[:, 1].max()) / 2
        span = max(rp[:, 0].max() - rp[:, 0].min(), rp[:, 1].max() - rp[:, 1].min(), 0.3)
        viewer.cam.lookat[:] = [x_mid, y_mid, 0.15]
        viewer.cam.distance = span * 1.8 + 0.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

    name, _ = SEGMENTS[seg_idx]
    print(f"[{name}]")
    frame_camera(motions[name])

    while viewer.is_running():
        name, _ = SEGMENTS[seg_idx]
        mo = motions[name]
        n = len(mo["root_pos"])

        d.qpos[0:3] = mo["root_pos"][frame_idx]
        x, y, z, w = mo["root_quat_xyzw"][frame_idx]
        d.qpos[3:7] = [w, x, y, z]
        for qadr, val in zip(qadrs_by_motion[name], mo["joints_pos"][frame_idx]):
            d.qpos[qadr] = val
        mujoco.mj_forward(m, d)
        viewer.sync()
        time.sleep(1.0 / mo["fps"])

        frame_idx = (frame_idx + 1) % n
        if time.time() - seg_start >= SECONDS_PER_SEGMENT:
            seg_idx = (seg_idx + 1) % len(SEGMENTS)
            frame_idx = 0
            seg_start = time.time()
            name, _ = SEGMENTS[seg_idx]
            print(f"[{name}]")
            frame_camera(motions[name])
