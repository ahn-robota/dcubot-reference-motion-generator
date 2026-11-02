"""Move the URDF base frame onto the leg midline (frame-only change, no geometry edit).

The CAD export put the base_link origin about 27 mm off the centre of the two
hips.  This script re-expresses the robot in a base frame translated to the
hip midline: every origin attached directly to base_link (its own visual /
collision / inertial, and its child joints and fixed frames) gets the same
+d shift in y.  Deeper links are relative to their parents and are untouched,
so the physical robot is identical -- only the reference frame moves.

Usage:
    python scripts_dcubot/recenter_base_origin.py            # dry run, prints numbers
    python scripts_dcubot/recenter_base_origin.py --apply    # writes the URDF in place
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

URDF = Path(__file__).resolve().parents[1] / "open_duck_reference_motion_generator/robots/dcubot/dcubot.urdf"
HIP_YAW = ("left_hip_yaw", "right_hip_yaw")


def xyz_of(elem):
    return [float(v) for v in elem.get("xyz").split()]


def set_xyz(elem, xyz):
    elem.set("xyz", " ".join(f"{v:.6g}" for v in xyz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--urdf", default=str(URDF))
    args = ap.parse_args()

    tree = ET.parse(args.urdf)
    root = tree.getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}

    hip_y = [xyz_of(joints[n].find("origin"))[1] for n in HIP_YAW]
    midline_y = sum(hip_y) / 2.0
    d = -midline_y  # shift added to every base-attached origin in y
    print(f"hip_yaw y: left {hip_y[0]*1000:+.3f} mm, right {hip_y[1]*1000:+.3f} mm")
    print(f"hip midline y (base frame): {midline_y*1000:+.3f} mm  -> shift d = {d*1000:+.3f} mm")

    changed = []
    # child joints / fixed frames attached to base_link
    for j in root.findall("joint"):
        if j.find("parent").get("link") == "base_link":
            o = j.find("origin")
            xyz = xyz_of(o)
            xyz[1] += d
            changed.append((f"joint {j.get('name')}", o, xyz))
    # base_link's own geometry origins
    base = next(l for l in root.findall("link") if l.get("name") == "base_link")
    for tag in ("inertial", "visual", "collision"):
        for el in base.findall(tag):
            o = el.find("origin")
            if o is None:
                continue
            xyz = xyz_of(o)
            xyz[1] += d
            changed.append((f"base_link {tag}", o, xyz))

    for label, o, xyz in changed:
        old = o.get("xyz")
        set_xyz(o, xyz)
        print(f"  {label:32s} {old:>28s} -> {o.get('xyz')}")

    # verify: hip midline is now zero, and hip spacing is unchanged
    new_hip = [xyz_of(joints[n].find("origin"))[1] for n in HIP_YAW]
    print(f"after: hip_yaw y left {new_hip[0]*1000:+.3f} mm, right {new_hip[1]*1000:+.3f} mm, "
          f"midline {(sum(new_hip)/2)*1000:+.6f} mm, spacing {(new_hip[0]-new_hip[1])*1000:.3f} mm")

    if args.apply:
        tree.write(args.urdf, encoding="utf-8", xml_declaration=True)
        print(f"wrote {args.urdf}")
    else:
        print("dry run -- nothing written (use --apply)")


if __name__ == "__main__":
    sys.exit(main())
