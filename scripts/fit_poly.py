import numpy as np
import json
from glob import glob
import os
import argparse
import pickle
import re

parser = argparse.ArgumentParser()
parser.add_argument("--ref_motion", type=str, default="ref_motion")
# Use the LAST full period of each recording instead of the first period after the
# start double support. The first period still contains the start-up transient
# (measured on dcubot: knee 24.5 vs 30.8 deg L/R at t=1.5-1.9 s, steady knee ~31 deg).
parser.add_argument("--window", type=str, default="start", choices=["start", "last"])
args = parser.parse_args()

all_files = glob(f"{args.ref_motion}/*.json")


def fit_ref_motion(file):
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    period = float(os.environ.get("FIT_PERIOD", data["Placo"]["period"]))  # FIT_PERIOD: true cycle (dcubot: 0.400 s, Placo records 0.408 s)
    fps = data["FPS"]
    frame_offsets = data["Frame_offset"][0]
    startend_double_support_ratio = data["Placo"]["startend_double_support_ratio"]
    start_offset = int(startend_double_support_ratio * fps)
    nb_steps_in_period = int(period * fps)
    if args.window == "last":
        start_offset = len(Y_all) - int(nb_steps_in_period)
    _Y = Y_all[start_offset : start_offset + int(nb_steps_in_period)]
    joints_pos = _Y[:, frame_offsets["joints_pos"] : frame_offsets["left_toe_pos"]]
    joints_vel = _Y[:, frame_offsets["joints_vel"] : frame_offsets["left_toe_vel"]]
    foot_contacts = _Y[
        :, frame_offsets["foot_contacts"] : frame_offsets["foot_contacts"] + 2
    ]
    base_linear_vel = _Y[
        :, frame_offsets["world_linear_vel"] : frame_offsets["world_angular_vel"]
    ]
    base_angular_vel = _Y[
        :, frame_offsets["world_angular_vel"] : frame_offsets["joints_vel"]
    ]

    Y = np.concatenate(
        [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel],
        axis=1,
    ).astype(np.float64)

    # Generate time feature
    X = np.linspace(0, 1, Y.shape[0]).reshape(-1, 1).astype(np.float64)  # Time variable (float64: float32 breaks deg>=10 fits)

    # Polynomial degree
    degree = int(os.environ.get("FIT_DEGREE", 15))

    # Store coefficients
    coefficients = {}

    # ====== Fit Polynomial Regression per Dimension ======
    for dim in range(Y.shape[1]):
        coeffs = np.polyfit(X.flatten(), Y[:, dim], degree)
        coefficients[f"dim_{dim}"] = list(np.flip(coeffs))

    ret_data = {
        "coefficients": coefficients,
        "period": period,
        "fps": fps,
        "frame_offsets": frame_offsets,
        "start_offset": start_offset,
        "nb_steps_in_period": nb_steps_in_period,
        "startend_double_support_ratio": startend_double_support_ratio,
    }

    return ret_data


all_coefficients = {}
for file in all_files:
    # Files are named "<anything>_<x_vel>_<y_vel>_<theta_vel>.json"; take the
    # last three numeric tokens so prefixes like "forward_medium" (which contain
    # underscores) don't shift the split indices.
    stem = os.path.splitext(os.path.basename(file))[0]
    m = re.search(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$", stem)
    if m is None:
        print(f"skip (cannot parse x_y_theta from name): {file}")
        continue
    name = "_".join(m.groups())

    all_coefficients[name] = fit_ref_motion(file)


pickle.dump(all_coefficients, open("polynomial_coefficients.pkl", "wb"))