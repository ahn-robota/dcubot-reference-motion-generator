import os
import numpy as np
from glob import glob
import json
import matplotlib.pyplot as plt
import argparse
import pickle
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--coefficients",
    type=str,
    default="polynomial_coefficients.pkl",
    help="Path to polynomial coefficients file",
)
parser.add_argument("-n", type=int, default=1, help="number of periods to sample")
parser.add_argument("--ref_motion", type=str, default="recordings")
parser.add_argument("--save_dir", type=str, default=None, help="save PNGs here instead of plt.show()")
args = parser.parse_args()

# poly_data = json.load(open(args.coefficients))
poly_data = pickle.load(open(args.coefficients, "rb"))
# pickle.load(args.coefficients)

all_ref_files = glob(f"{args.ref_motion}/*.json")


def sample_polynomial(time: float, dimension: int, coefficients) -> float:
    """
    Evaluate the polynomial at a given time for a specific dimension.
    """
    if not (0 <= time <= 1):
        raise ValueError("Time must be between 0 and 1.")

    coeffs = coefficients.get(f"dim_{dimension}")
    if coeffs is None:
        raise ValueError(f"Dimension {dimension} not found in stored coefficients.")

    return sum(c * (time**i) for i, c in enumerate(coeffs))


for file in all_ref_files:
    stem = os.path.splitext(os.path.basename(file))[0]
    m = re.search(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$", stem)
    if m is None:
        print(f"skip (cannot parse x_y_theta from name): {file}")
        continue
    name = "_".join(m.groups())
    banner = f"=== {name} ==="
    spacer = "=" * len(banner)
    print(spacer)
    print(banner)
    print(spacer)

    period = poly_data[name]["period"]
    fps = poly_data[name]["fps"]
    frame_offsets = poly_data[name]["frame_offsets"]
    startend_double_support_ratio = poly_data[name]["startend_double_support_ratio"]
    start_offset = int(startend_double_support_ratio * fps)
    nb_steps_in_period = int(period * fps)

    # Load reference motion data
    ref_data = json.load(open(file))
    Y_all = np.array(ref_data["Frames"])
    _Y = Y_all[start_offset : start_offset + int(nb_steps_in_period) * args.n]
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
    )

    jn = ref_data["Joints"]
    dimensions_names = (
        [f"pos {j}" for j in jn]
        + [f"vel {j}" for j in jn]
        + ["foot_contacts left", "foot_contacts right"]
        + [f"base_linear_vel {a}" for a in "xyz"]
        + [f"base_angular_vel {a}" for a in "xyz"]
    )
    nb_dim = Y.shape[1]

    # Generate time feature
    X = np.linspace(0, 1, Y.shape[0]).reshape(-1, 1)  # Time variable

    # Get coefficients for this motion
    coefficients = poly_data[name]["coefficients"]

    dimensions = []

    # ====== Plot Fitting for Each Dimension ======
    for dimension in range(nb_dim):
        poly_samples = []
        ref_samples = []
        # times = X.flatten()  # Use original time values for accurate mapping
        times = np.arange(len(Y)) % nb_steps_in_period / nb_steps_in_period

        for i, t in enumerate(times):
            if i >= Y.shape[0]:
                break  # Prevent out-of-bounds errors

            poly_sample = sample_polynomial(t, dimension, coefficients)
            ref = Y[i, dimension]

            poly_samples.append(poly_sample)
            ref_samples.append(ref)
        dimensions.append((dimension, ref_samples, poly_samples))

    # ====== Plotting ======
    nb_rows = int(np.sqrt(nb_dim))
    nb_cols = int(np.ceil(nb_dim / nb_rows))

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(12, 8))
    axs = axs.flatten()

    for i, (dim, ref, poly) in enumerate(dimensions):
        ax = axs[i]
        ax.plot(ref, label="Original Data", alpha=0.5)
        ax.plot(poly, label="Polynomial Fit", color="red")
        ax.set_title(f"{dimensions_names[dim]}", fontsize=7)
        rmse = float(np.sqrt(np.mean((np.array(ref) - np.array(poly)) ** 2)))
        rng = float(np.ptp(ref)) or 1.0
        print(f"{dimensions_names[dim]:28s} rmse={rmse:.4f}  range={rng:.4f}  rmse/range={rmse/rng:.3f}")

    plt.tight_layout()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out = os.path.join(args.save_dir, f"{name}.png")
        plt.savefig(out, dpi=90)
        plt.close(fig)
        print("saved", out)
    else:
        plt.show()
