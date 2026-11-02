"""Move Placo's foot reference from the ankle axis to the sole centre.

Placo treats foot_L_1 / foot_R_1 (the ankle axis) as the foot position. The real
contact surface (sole) sits at a fixed offset from it, identical for both feet
(measured from the foot collision mesh bounding box, in the foot_*_1 frame):

    x +12.2 mm   y +18.5 mm   z -38.8 mm   (sole bottom-face centre)

This only changes the fixed `left_foot_frame` / `right_foot_frame` joints, which
Placo uses as the foot targets. Geometry and the MuJoCo model are unchanged.

Usage:
    python scripts_dcubot/set_foot_planner_frame.py            # dry run
    python scripts_dcubot/set_foot_planner_frame.py --apply    # write in place
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

URDF = Path(__file__).resolve().parents[1] / "open_duck_reference_motion_generator/robots/dcubot/dcubot.urdf"
SOLE_OFFSET_M = (0.0122, 0.0185, -0.0388)
FRAMES = {"left_foot_frame": "left_foot", "right_foot_frame": "right_foot"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--urdf", default=str(URDF))
    args = ap.parse_args()

    tree = ET.parse(args.urdf)
    joints = {j.get("name"): j for j in tree.getroot().findall("joint")}
    for name, child in FRAMES.items():
        j = joints[name]
        if j.find("child").get("link") != child:
            sys.exit(f"{name}: child is {j.find('child').get('link')}, expected {child}")
        o = j.find("origin")
        old = o.get("xyz")
        o.set("xyz", " ".join(f"{v:.6g}" for v in SOLE_OFFSET_M))
        print(f"{name:18s} origin xyz  {old:>20s} -> {o.get('xyz')}")

    if args.apply:
        tree.write(args.urdf, encoding="utf-8", xml_declaration=True)
        print(f"wrote {args.urdf}")
    else:
        print("dry run -- nothing written (use --apply)")


if __name__ == "__main__":
    sys.exit(main())
