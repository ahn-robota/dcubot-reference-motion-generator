"""Gait metrics for one configuration (headless, engine with joint limits on).

Reports over the full gait cycle (after warm-up):
  - knee min/max (deg) for left and right
  - swing-foot ground clearance: sole height above the floor while a foot is in swing
    (per step: max height; overall: min/mean of the per-step maxima)
  - straight-leg vertical distance hip_pitch axis -> sole (knee = 0, all leg joints 0)

Usage (repo root):
    .venv/bin/python scripts_dcubot/measure_gait_metrics.py --com 0.23 --dx 0.1
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
BIAS = 0.0001


def straight_leg_mm(pwe):
    """Vertical distance hip_pitch axis -> sole, with every leg joint at 0 (knee straight)."""
    r = pwe.robot
    for side in ("left", "right"):
        for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
            r.set_joint(f"{side}_{j}", 0.0)
    r.update_kinematics()
    out = {}
    for side, L in (("left", "L"), ("right", "R")):
        hip_z = r.get_T_world_frame(f"thigh_{L}_1")[2, 3]   # hip_pitch axis (thigh frame origin)
        sole_z = r.get_T_world_frame(f"{side}_foot")[2, 3]  # sole-centre planner frame
        out[side] = (hip_z - sole_z) * 1000
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--com", type=float, default=0.23)
    ap.add_argument("--dx", type=float, default=0.1)
    ap.add_argument("--trunk", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ASSET, "placo_defaults.json")))
    cfg["walk_com_height"] = args.com
    cfg["walk_trunk_pitch"] = args.trunk
    pwe = PlacoWalkEngine(ASSET, "dcubot.urdf", cfg)
    pwe.set_traj(args.dx, 0.0, BIAS)

    straight = straight_leg_mm(pwe)
    # restore walking start pose by re-running the engine's own init
    pwe = PlacoWalkEngine(ASSET, "dcubot.urdf", cfg)
    pwe.set_traj(args.dx, 0.0, BIAS)

    knee_l, knee_r = [], []
    sole_l, sole_r, phases = [], [], []
    for _ in range(int((WARMUP_S + args.seconds) / DT)):
        pwe.tick(DT)
        if pwe.t < WARMUP_S:
            continue
        ang = pwe.get_angles()
        knee_l.append(np.degrees(ang["left_knee"]))
        knee_r.append(np.degrees(ang["right_knee"]))
        sole_l.append(pwe.robot.get_T_world_left()[2, 3] * 1000)
        sole_r.append(pwe.robot.get_T_world_right()[2, 3] * 1000)
        phases.append(tuple(pwe.get_current_support_phase()))

    knee_l, knee_r = np.array(knee_l), np.array(knee_r)
    sole_l, sole_r = np.array(sole_l), np.array(sole_r)

    # swing clearance: per swing window, the max sole height of the swinging foot
    clearances = []
    cur_side, cur_max = None, -1e9
    for k, ph in enumerate(phases):
        if ph == (1, 0):
            side = "right"
        elif ph == (0, 1):
            side = "left"
        else:
            side = None
        if side != cur_side:
            if cur_side is not None:
                clearances.append((cur_side, cur_max))
            cur_side, cur_max = side, -1e9
        if side == "right":
            cur_max = max(cur_max, sole_r[k])
        elif side == "left":
            cur_max = max(cur_max, sole_l[k])
    if cur_side is not None:
        clearances.append((cur_side, cur_max))
    cl = np.array([c for _, c in clearances if c > -1e8])

    print(f"== com {args.com:.3f} m, dx {args.dx}, trunk {args.trunk} deg, knee limits from URDF ==")
    print(f"knee LEFT  min {knee_l.min():+7.2f} deg   max {knee_l.max():+7.2f} deg")
    print(f"knee RIGHT min {knee_r.min():+7.2f} deg   max {knee_r.max():+7.2f} deg")
    print(f"swing clearance (sole height above floor, per step): "
          f"min {cl.min():.2f} mm   mean {cl.mean():.2f} mm   max {cl.max():.2f} mm   (n={len(cl)})")
    print(f"sole height during stance (should be ~0): L min {sole_l.min():+.2f}  R min {sole_r.min():+.2f} mm")
    print(f"straight leg hip_pitch -> sole (knee 0): L {straight['left']:.1f} mm   R {straight['right']:.1f} mm")
    print(f"com_height {args.com*1000:.0f} mm vs straight-leg hip height {np.mean(list(straight.values())):.1f} mm "
          f"-> ratio {args.com*1000/np.mean(list(straight.values())):.0%}")


if __name__ == "__main__":
    main()
