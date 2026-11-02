import json
import time
import numpy as np
import mujoco
import mujoco.viewer

URDF = "/home/ahn/dcubot/DCUBOT/reference_motion_generator/open_duck_reference_motion_generator/robots/dcubot/dcubot.urdf"
RECORDING = "/home/ahn/dcubot/DCUBOT/reference_motion_generator/motions_dcubot/forward_medium_0.02_0.0_0.0.json"

with open(RECORDING) as f:
    data = json.load(f)

frames = np.array(data["Frames"])
offs = data["Frame_offset"][0]
sizes = data["Frame_size"][0]
joint_names = data["Joints"]
fps = data["FPS"]

root_pos = frames[:, offs["root_pos"]:offs["root_pos"]+sizes["root_pos"]]
root_quat_xyzw = frames[:, offs["root_quat"]:offs["root_quat"]+sizes["root_quat"]]
joints_pos = frames[:, offs["joints_pos"]:offs["joints_pos"]+sizes["joints_pos"]]

m = mujoco.MjModel.from_xml_path(URDF)
d = mujoco.MjData(m)

# NOTE: this dcubot.urdf has no freejoint (fixed base), so we only replay the
# leg/head joint angles (all in RADIANS, matching MuJoCo's Joint tab units) --
# not the root translation/rotation.
joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in joint_names]
qadrs = [m.jnt_qposadr[jid] for jid in joint_ids]

print("재생할 joint 순서 (전부 라디안):", joint_names)

with mujoco.viewer.launch_passive(m, d) as viewer:
    i = 0
    n = len(frames)
    while viewer.is_running():
        for qadr, val in zip(qadrs, joints_pos[i]):
            d.qpos[qadr] = val
        mujoco.mj_forward(m, d)
        viewer.sync()
        time.sleep(1.0 / fps)
        i = (i + 1) % n
