"""Play back the FITTED polynomial reference motion pkl (what joystick.py
actually uses for the imitation reward right now) in meshcat via Placo --
unlike playback_recorded_motion_placo.py, this reads the .pkl directly
(there is no raw recording .json matching this pkl's specific dx values),
reconstructing the joint trajectory by sampling the fitted polynomial over
one gait period.

Usage:
    .venv/bin/python scripts_dcubot/playback_poly_pkl_placo.py [--pkl PATH] [--key KEY]

  --pkl   defaults to the pkl playground/dcubot/joystick.py currently points
          at (DCUBOT_REFERENCE_PKL): polynomial_coefficients_slow.pkl
  --key   one of the pkl's dx_dy_dtheta keys (run with --list to see them);
          defaults to the fastest forward speed in the pkl.
"""
import argparse
import pickle
import sys
import time

import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz

URDF = "/home/ahn/dcubot/DCUBOT/reference_motion_generator/open_duck_reference_motion_generator/robots/dcubot/dcubot.urdf"
DEFAULT_PKL = "/home/ahn/dcubot/DCUBOT/rl_training/playground/dcubot/data/polynomial_coefficients_slow.pkl"

# Placo recording order (confirmed via scripts/inspect_reference.py reading
# a raw recording's own "Joints" list): left leg(5), head(3), right leg(5).
# The pkl's fitted vector is [joints_pos(13), joints_vel(13), foot_contacts(2),
# base_linear_vel(3), base_angular_vel(3)] -- only joints_pos[0:13] is needed
# here, in this exact order.
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def sample_polynomial(t, coeffs):
    # coeffs stored lowest-degree-first (fit_poly.py's np.flip(np.polyfit(...)));
    # flip once to highest-degree-first for np.polyval, matching
    # playground/common/poly_reference_motion.py's own load-time convention.
    return np.array([np.polyval(np.flip(np.array(c)), t) for c in coeffs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    data = pickle.load(open(args.pkl, "rb"))
    keys = sorted(data.keys(), key=lambda k: float(k.split("_")[0]))

    if args.list:
        print(f"keys in {args.pkl}:")
        for k in keys:
            print(" ", k)
        sys.exit(0)

    key = args.key or keys[-1]
    if key not in data:
        print(f"key {key} not found. Available: {keys}")
        sys.exit(1)

    entry = data[key]
    period, fps = entry["period"], entry["fps"]
    nb_steps = int(period * fps)
    coeffs = list(entry["coefficients"].values())  # dim_0..dim_33, in order

    robot = placo.HumanoidRobot(URDF)
    robot.update_kinematics()

    viz = robot_viz(robot)
    print(f"meshcat URL: {viz.viewer.url()}")
    viz.viewer.open()
    print(f"재생 중: {args.pkl}  key={key}  ({nb_steps} steps/period @ {fps}fps)")

    while True:
        for i in range(nb_steps):
            t = (i % nb_steps) / nb_steps
            frame = sample_polynomial(t, coeffs)
            joints_pos = frame[0:13]
            for jn, val in zip(JOINT_NAMES, joints_pos):
                robot.set_joint(jn, val)
            robot.update_kinematics()
            viz.display(robot.state.q)
            robot_frame_viz(robot, "trunk")
            robot_frame_viz(robot, "left_foot")
            robot_frame_viz(robot, "right_foot")
            time.sleep(1.0 / fps)


if __name__ == "__main__":
    main()
