"""Fine com_height sweep with two gates (dx and trunk_pitch fixed).

Gate 1 - floor contact: the lower sole must stay within +-FLOOR_TOL_MM of the floor
          over the whole cycle (sinking or hovering both fail).
Gate 2 - joint room: hip_pitch must stay inside +-30 deg with at least MARGIN_DEG
          to spare on both sides (joint limits are on in the engine).

Usage (repo root):
    .venv/bin/python scripts_dcubot/sweep_com_floor.py --dx 0.03
    .venv/bin/python scripts_dcubot/sweep_com_floor.py --dx 0.03 --com 0.190 0.195 0.200
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
HIP_LIMIT = 30.0
FLOOR_TOL_MM = 3.0
MARGIN_DEG = 5.0


def run(cfg, com, dx, trunk, seconds):
    tilts = []
    cfg = dict(cfg)
    cfg["walk_com_height"] = com
    cfg["walk_trunk_pitch"] = trunk
    pwe = PlacoWalkEngine(ASSET, "dcubot.urdf", cfg)
    pwe.set_traj(dx, 0.0, BIAS)
    lower_sole, hip_l, hip_r = [], [], []
    for _ in range(int((WARMUP_S + seconds) / DT)):
        pwe.tick(DT)
        if pwe.t < WARMUP_S:
            continue
        zl = pwe.robot.get_T_world_left()[2, 3] * 1000
        zr = pwe.robot.get_T_world_right()[2, 3] * 1000
        lower_sole.append(min(zl, zr))
        tilts.append(np.degrees(np.arccos(np.clip(pwe.robot.get_T_world_fbase()[2, 2], -1, 1))))
        ang = pwe.get_angles()
        hip_l.append(np.degrees(ang["left_hip_pitch"]))
        hip_r.append(np.degrees(ang["right_hip_pitch"]))
    ls = np.array(lower_sole)
    hl, hr = np.array(hip_l), np.array(hip_r)
    hip_min = min(hl.min(), hr.min())
    hip_max = max(hl.max(), hr.max())
    margin = min(HIP_LIMIT - hip_max, hip_min + HIP_LIMIT)
    floor_ok = bool(ls.min() >= -FLOOR_TOL_MM and ls.max() <= FLOOR_TOL_MM)
    room_ok = bool(margin >= MARGIN_DEG)
    return {
        "com": com, "trunk": trunk, "tilt_max_deg": float(np.max(tilts)),
        "lower_sole_min_mm": float(ls.min()), "lower_sole_max_mm": float(ls.max()),
        "hip_min": float(hip_min), "hip_max": float(hip_max),
        "hip_margin_deg": float(margin),
        "asym_deg": float(abs((hl.max() - hl.min()) - (hr.max() - hr.min()))),
        "floor_ok": floor_ok, "room_ok": room_ok, "pass": floor_ok and room_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.03)
    ap.add_argument("--trunk", type=float, nargs="+", default=[0.0])
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--com", type=float, nargs="+",
                    default=[round(0.17 + 0.005 * i, 3) for i in range(9)])  # 0.170 .. 0.210
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ASSET, "placo_defaults.json")))
    rows = []
    for tp in args.trunk:
     for com in args.com:
        r = run(cfg, com, args.dx, tp, args.seconds)
        rows.append(r)
        print(f"trunk {tp:+5.2f} com {com:.3f} tilt {r['tilt_max_deg']:4.1f}: sole[{r['lower_sole_min_mm']:+6.2f},{r['lower_sole_max_mm']:+6.2f}] mm "
              f"{'OK ' if r['floor_ok'] else 'NG '}| hip [{r['hip_min']:+6.1f},{r['hip_max']:+6.1f}] "
              f"margin {r['hip_margin_deg']:+6.1f} deg {'OK ' if r['room_ok'] else 'NG '}| "
              f"asym {r['asym_deg']:4.1f} | {'PASS' if r['pass'] else 'fail'}", flush=True)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)
        print("saved", args.json)


if __name__ == "__main__":
    main()
