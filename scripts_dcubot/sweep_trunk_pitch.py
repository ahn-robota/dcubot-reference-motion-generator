"""Sweep walk_trunk_pitch x walk_com_height and measure hip_pitch range, symmetry, stability.

Headless: runs Placo's gait engine (same tick path as gait_generator.py) at a fixed
forward command, records joints and body pose after a warm-up, and reports:
  - hip_pitch L/R min, max, range (deg)
  - left/right hip_pitch range difference (asymmetry, deg)
  - lateral sway of the body (mm)
  - max body tilt (deg) and whether the body fell (tilt > 30 deg or z < 0.1 m)
  - support steps (count of single-support switches) per side

Usage (repo root):
    .venv/bin/python scripts_dcubot/sweep_trunk_pitch.py
    .venv/bin/python scripts_dcubot/sweep_trunk_pitch.py --dx 0.1 --seconds 6
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "open_duck_reference_motion_generator")
sys.path.insert(0, PKG)
from placo_walk_engine import PlacoWalkEngine  # noqa: E402

ASSET = os.path.join(PKG, "robots", "dcubot") + "/"
DT = 0.001
WARMUP_S = 1.0
DCUBOT_BIAS = 0.0001  # same bias gait_generator.py uses for dcubot


def urdf_limits(urdf_path):
    """Revolute joint limits (rad) from the URDF, keyed by joint name."""
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    out = {}
    for j in root.findall("joint"):
        if j.get("type") == "revolute":
            lim = j.find("limit")
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def run_case(base_cfg, trunk_pitch_deg, com_height, dx, seconds, urdf=None, joint_limits=False):
    cfg = dict(base_cfg)
    cfg["walk_trunk_pitch"] = trunk_pitch_deg
    cfg["walk_com_height"] = com_height
    # absolute urdf path overrides the default (os.path.join keeps an absolute second arg)
    pwe = PlacoWalkEngine(ASSET, urdf or "dcubot.urdf", cfg)
    if joint_limits:
        # engine sets knees to [0.2, 0.01] rad, which the crouched start pose (knee ~ -0.94)
        # already violates -> infeasible QP. Relax knees to the URDF range, then enable limits.
        pwe.robot.set_joint_limits("left_knee", -np.pi / 2, np.pi / 2)
        pwe.robot.set_joint_limits("right_knee", -np.pi / 2, np.pi / 2)
        pwe.solver.enable_joint_limits(True)   # engine disables this in __init__
    lims = urdf_limits(urdf or os.path.join(ASSET, "dcubot.urdf"))
    sat_hits, sat_total = 0, 0
    pwe.set_traj(dx, 0.0, DCUBOT_BIAS)

    hip_l, hip_r, sway_y, tilt, z_min = [], [], [], [], []
    steps = {"left": 0, "right": 0}
    last_phase = None
    n_ticks = int((WARMUP_S + seconds) / DT)
    for _ in range(n_ticks):
        pwe.tick(DT)
        if pwe.t < WARMUP_S:
            continue
        ang = pwe.get_angles()
        # fraction of samples where any leg joint is within 0.5 deg of its URDF limit
        near = False
        for name, v in ang.items():
            if name in lims:
                lo, hi = lims[name]
                if v <= lo + np.radians(0.5) or v >= hi - np.radians(0.5):
                    near = True
        sat_total += 1
        sat_hits += int(near)
        hip_l.append(np.degrees(ang["left_hip_pitch"]))
        hip_r.append(np.degrees(ang["right_hip_pitch"]))
        T = pwe.robot.get_T_world_fbase()
        sway_y.append(T[1, 3])
        z_min.append(T[2, 3])
        up = T[2, 2]  # body z-axis world z component
        tilt.append(np.degrees(np.arccos(np.clip(up, -1, 1))))
        phase = tuple(pwe.get_current_support_phase())
        if phase == (1, 0) and last_phase != (1, 0):
            steps["left"] += 1
        if phase == (0, 1) and last_phase != (0, 1):
            steps["right"] += 1
        last_phase = phase

    hip_l, hip_r = np.array(hip_l), np.array(hip_r)
    sway_y = np.array(sway_y)
    tilt = np.array(tilt)
    z_min = np.array(z_min)
    fell = bool(tilt.max() > 30.0 or z_min.min() < 0.10)
    return {
        "trunk_pitch_deg": trunk_pitch_deg,
        "com_height_m": com_height,
        "hip_pitch_L_min": float(hip_l.min()), "hip_pitch_L_max": float(hip_l.max()),
        "hip_pitch_L_range": float(hip_l.max() - hip_l.min()),
        "hip_pitch_R_min": float(hip_r.min()), "hip_pitch_R_max": float(hip_r.max()),
        "hip_pitch_R_range": float(hip_r.max() - hip_r.min()),
        "asym_range_deg": float(abs((hip_l.max() - hip_l.min()) - (hip_r.max() - hip_r.min()))),
        "sway_mm": float((sway_y.max() - sway_y.min()) * 1000),
        "max_tilt_deg": float(tilt.max()),
        "min_body_z_m": float(z_min.min()),
        "left_steps": steps["left"], "right_steps": steps["right"],
        "limit_touch_pct": 100.0 * sat_hits / max(sat_total, 1),
        "fell": fell,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.1)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--trunk", type=float, nargs="+", default=[0, -2, -4, -6, -8])
    ap.add_argument("--com", type=float, nargs="+", default=[0.16, 0.17])
    ap.add_argument("--json", default=None)
    ap.add_argument("--urdf", default=None, help="absolute path to a URDF to compare against")
    ap.add_argument("--joint-limits", action="store_true", help="enable Placo solver joint limits")
    args = ap.parse_args()

    with open(os.path.join(ASSET, "placo_defaults.json")) as f:
        base_cfg = json.load(f)

    rows = []
    for com in args.com:
        for tp in args.trunk:
            r = run_case(base_cfg, tp, com, args.dx, args.seconds, args.urdf, args.joint_limits)
            rows.append(r)
            print(f"trunk {tp:+5.1f}  com {com:.2f}: hipL [{r['hip_pitch_L_min']:+6.1f},{r['hip_pitch_L_max']:+6.1f}] "
                  f"range {r['hip_pitch_L_range']:5.1f} | hipR [{r['hip_pitch_R_min']:+6.1f},{r['hip_pitch_R_max']:+6.1f}] "
                  f"range {r['hip_pitch_R_range']:5.1f} | asym {r['asym_range_deg']:4.1f} | "
                  f"sway {r['sway_mm']:5.1f}mm | tilt {r['max_tilt_deg']:4.1f} | "
                  f"steps L/R {r['left_steps']}/{r['right_steps']} | limit-touch {r['limit_touch_pct']:4.1f}% | {'FELL' if r['fell'] else 'ok'}",
                  flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
