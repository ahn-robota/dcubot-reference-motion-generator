"""Play back a recorded (and mirrored) dcubot motion JSON in meshcat via Placo,
so you can inspect it the same way as the live gait_playground/gait_generator
views, but for the already-fixed, saved motion file."""
import json
import sys
import time
import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz

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

joints_pos = frames[:, offs["joints_pos"]:offs["joints_pos"]+sizes["joints_pos"]]

robot = placo.HumanoidRobot(URDF)
robot.update_kinematics()

viz = robot_viz(robot)
print(f"meshcat URL: {viz.viewer.url()}")
viz.viewer.open()
print(f"재생 중: {RECORDING}  ({len(frames)} frames @ {fps}fps)")

while True:
    for i in range(len(frames)):
        for jn, val in zip(joint_names, joints_pos[i]):
            robot.set_joint(jn, val)
        robot.update_kinematics()
        viz.display(robot.state.q)
        robot_frame_viz(robot, "trunk")
        robot_frame_viz(robot, "left_foot")
        robot_frame_viz(robot, "right_foot")
        time.sleep(1.0 / fps)
