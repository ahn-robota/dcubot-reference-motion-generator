"""Load the OLD (pre-remodel) and NEW dcubot URDF side by side in one meshcat
viewer, for visual comparison of coordinate axes / proportions / pose.

Both robots are shown at their zero-joint ("legs straight") pose. The NEW
robot is offset sideways so the two don't overlap.

Usage:
    uv run python scripts_dcubot/compare_old_new_meshcat.py

Then open the printed meshcat URL in a browser.
"""

import tempfile
import time

import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import get_viewer, robot_viz

OLD_URDF_SRC = "/home/ahn/dcubot_old_model_1358g_backup/dcubot/dcubot.urdf"
OLD_ASSETS_DIR = "/home/ahn/dcubot_old_model_1358g_backup/dcubot/assets"
# The backup URDF's own <mesh> tags are absolute paths into the LIVE generator
# directory (not self-contained), which now holds the NEW model's meshes --
# rewrite them to point at the backup's own assets/ before loading, so the
# OLD robot actually shows its own (1.358kg) mesh shapes.
GENERATOR_LIVE_ASSETS = "/home/ahn/dcubot/DCUBOT/reference_motion_generator/open_duck_reference_motion_generator/robots/dcubot/assets"
NEW_URDF = "/home/ahn/dcubot/DCUBOT/reference_motion_generator/open_duck_reference_motion_generator/robots/dcubot/dcubot.urdf"
OFFSET_X = 0.5  # meters, separation between the two robots in the viewer

old_urdf_text = open(OLD_URDF_SRC).read().replace(GENERATOR_LIVE_ASSETS, OLD_ASSETS_DIR)
old_urdf_tmp = tempfile.NamedTemporaryFile(
    mode="w", suffix=".urdf", prefix="dcubot_old_", delete=False
)
old_urdf_tmp.write(old_urdf_text)
old_urdf_tmp.close()
OLD_URDF = old_urdf_tmp.name

old_robot = placo.HumanoidRobot(OLD_URDF)
new_robot = placo.HumanoidRobot(NEW_URDF)
old_robot.update_kinematics()
new_robot.update_kinematics()

viewer = get_viewer()
print(f"Viewer URL: {viewer.url()}")

# Default meshcat camera framing is often pointed away from a ~0.25m robot at
# the origin, making the scene look empty until you manually zoom/pan. Set an
# explicit camera position looking at the midpoint between the two robots.
viewer["/Background"].set_property("top_color", [0.9, 0.9, 0.9])
viewer["/Background"].set_property("bottom_color", [0.7, 0.7, 0.7])
viewer["/Cameras/default/rotated/<object>"].set_transform(
    tf.translation_matrix([OFFSET_X / 2, -1.0, 0.3])
    @ tf.euler_matrix(np.pi / 2.2, 0, 0)
)

viz_old = robot_viz(old_robot, name="old_1358g")
viz_new = robot_viz(new_robot, name="new_2072g")

viz_old.display(old_robot.state.q)
viz_new.display(new_robot.state.q)

viewer["new_2072g"].set_transform(tf.translation_matrix([OFFSET_X, 0, 0]))

print("old_1358g = OLD model (1.358kg), new_2072g = NEW model (2.072kg)")
print(f"new model is shifted +{OFFSET_X}m in X so they don't overlap")
print("Both shown at zero-joint (legs straight) pose. Ctrl+C to exit.")

while True:
    time.sleep(1)
