"""Regenerate every motion in motions_dcubot/ against the CURRENT dcubot.urdf
and placo_defaults.json.

Why this exists: motions_dcubot/*.json were generated on 2026-09-24, before
the 2026-10-04/05 URDF corrections (recenter_base_origin.py, set_foot_planner_frame.py).
Replaying those old files on the corrected skeleton produces visibly wrong
motion (verified 2026-10-14). This regenerates all 9 with the same commanded
(dx, dy, dtheta) but against the current engine, overwriting the stale files
in place so filenames/semantics stay unchanged.

Usage:
    cd reference_motion_generator
    ../.venv/bin/python scripts_dcubot/regenerate_motions_dcubot.py
"""
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(SCRIPT_DIR, "..", "open_duck_reference_motion_generator")
DEFAULTS = os.path.join(GEN_DIR, "robots/dcubot/placo_defaults.json")
MOTIONS_DIR = os.path.join(SCRIPT_DIR, "..", "motions_dcubot")
TMP_PRESET = "/tmp/dcubot_regen_preset.json"
TMP_OUT = "/tmp/dcubot_regen_out"

# (output name without .json, dx, dy, dtheta) -- matches the existing
# motions_dcubot/ filenames exactly.
MOTIONS = [
    ("standing_0.0_0.0_0.0", 0.0, 0.0, 0.0),
    # slow/medium/fast dx values, measured empirically against the REAL
    # simulated velocity (gait_generator.py's "avg lin_vel_x", not the
    # "computed xvel" formula label -- that one is just dx run through
    # steps_to_vel() and does not reflect what the robot actually achieves).
    # Sweep on 2026-10-15: dx->real velocity is linear up to dx~0.12-0.13,
    # where it SATURATES at the robot's real kinematic max (~0.59 m/s) --
    # dx=0.12 (auto_gait.json's own "medium"/walk_max_dx_forward) is already
    # AT that ceiling, so it produced an identical gait to the old dx=0.49
    # (both saturate to 0.59 m/s) -- that's why the first fix looked unchanged.
    # These three stay in the verified-linear, non-saturated range, matching
    # the ~0.05-0.1 m/s scale of the actually-used training reference
    # (polynomial_coefficients_slow.pkl, built from ~slow_0.049-0.098 recordings):
    #   dx=0.01 -> ~0.049 m/s, dx=0.02 -> ~0.098 m/s, dx=0.04 -> ~0.195 m/s
    #
    # Separately, dx only sets stride LENGTH. Step CADENCE (how fast the legs
    # swing, independent of stride) comes from single_support_duration --
    # confirmed visually 2026-10-15 that the old value (0.17s, period 0.408s)
    # still looked fast even with a short stride. Raised to 0.25s (period
    # 0.6s) in placo_defaults.json itself, so every motion below inherits it.
    ("forward_slow_0.01_0.0_0.0", 0.01, 0.0, 0.0),
    ("forward_medium_0.02_0.0_0.0", 0.02, 0.0, 0.0),
    ("forward_fast_0.04_0.0_0.0", 0.04, 0.0, 0.0),
    # backward/strafe/turn had the exact same saturation problem as forward,
    # confirmed by the same sweep-and-measure method (2026-10-15):
    #   dx     -0.08 to -0.392 all saturate at the SAME -0.266 m/s (walk_max_dx_backward
    #           =0.08 is again exactly the ceiling, not a safe value). Linear below
    #           that; -0.04 -> -0.133 m/s measured, matches forward_fast's magnitude.
    #   dy      0.1 to 0.245 all saturate near 0.16 m/s (walk_max_dy=0.1 is the ceiling).
    #           Linear below that; 0.04 -> 0.062 m/s measured, matches forward_medium.
    #   dtheta  saturates near 0.23-0.24 above dtheta~0.8-1.0 (walk_max_dtheta=1.0 is
    #           the ceiling). Below that it's nonlinear (roughly quadratic), not just
    #           scaled down -- 0.5 -> 0.0765 measured "avg yaw", clearly sub-saturation.
    ("backward_-0.04_0.0_0.0", -0.04, 0.0, 0.0),
    ("strafe_left_0.0_0.04_0.0", 0.0, 0.04, 0.0),
    ("strafe_right_0.0_-0.04_0.0", 0.0, -0.04, 0.0),
    ("turn_left_0.0_0.0_0.5", 0.0, 0.0, 0.5),
    ("turn_right_0.0_0.0_-0.5", 0.0, 0.0, -0.5),
]


def main():
    os.makedirs(TMP_OUT, exist_ok=True)
    with open(DEFAULTS) as f:
        base = json.load(f)

    for out_name, dx, dy, dtheta in MOTIONS:
        preset = dict(base)
        preset["dx"], preset["dy"], preset["dtheta"] = dx, dy, dtheta
        with open(TMP_PRESET, "w") as f:
            json.dump(preset, f, indent=2)

        print(f"--- regenerating {out_name} (dx={dx}, dy={dy}, dtheta={dtheta}) ---")
        result = subprocess.run(
            [sys.executable, os.path.join(GEN_DIR, "gait_generator.py"),
             "--duck", "dcubot", "--preset", TMP_PRESET,
             "-n", "regen", "-o", TMP_OUT, "-l", "8"],
            cwd=GEN_DIR, env={**os.environ, "RECORD_FPS": "50"},
            capture_output=True, text=True,
        )
        produced = [f for f in os.listdir(TMP_OUT) if f.startswith("regen_")]
        if not produced:
            print(result.stdout[-2000:], result.stderr[-2000:])
            sys.exit(f"FAILED: no output for {out_name}")
        src = os.path.join(TMP_OUT, produced[0])
        dst = os.path.join(MOTIONS_DIR, out_name + ".json")
        os.replace(src, dst)
        print(f"  wrote {dst}")

    print("done -- all 9 motions_dcubot/*.json regenerated against the current URDF")


if __name__ == "__main__":
    main()
