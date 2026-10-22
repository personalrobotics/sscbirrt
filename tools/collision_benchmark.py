# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Per-check cost of the native MuJoCo validator with a planning margin, all pairs against the pairs that can
change (#254).

The scene is the demo world (UR5e, Robotiq 2F-85, table) with free cans resting on the table in a tight grid, so a
margin makes every can touch the table and its neighbours: environment contacts that the policy discards. For each
margin it times ``is_valid`` over the same seeded random arm configurations with two validators on one snapshot:
``filter_passive_pairs=False`` (a full ``mj_collision``, the reference) and the default, which collides only pairs
involving the arm or a held object. Blocks of configurations alternate between the two so both see the same machine
state. It also records, untimed, that the two give the same decisions and contact lists on every configuration, and
the mean number of contacts MuJoCo generates per check.

Timings belong to one host: run it as a single process on a quiet machine and compare only within one report.

    uv run python tools/collision_benchmark.py --output collision_benchmark.json
    uv run python tools/collision_benchmark.py --configs 2000 --cans 16
    uv run python tools/collision_benchmark.py --mjcf geodude.xml --joints j1 j2 ...   # another world, as compiled
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MARGINS = (0.0, 0.005)
RATIO_LIMIT = 1.35  # #254: at 5 mm, the filtered check costs at most 1.35x the margin-0 check
BLOCK = 500


def _demo_model(cans: int):
    from sscbirrt.demo.scene import CAN_RADIUS, build_scene, can_position

    pitch = 2 * CAN_RADIUS + 0.004  # neighbours 4 mm apart: inside a 5 mm margin
    cols = 4
    grid = {
        f"can{i}": can_position([0.62 + (i // cols) * pitch, (i % cols - (cols - 1) / 2) * pitch]) for i in range(cans)
    }
    return build_scene(cans=grid)


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "src", "cpp", "tools")),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "versions": {p: importlib.metadata.version(p) for p in ("sscbirrt", "mujoco", "numpy")},
        "python": platform.python_version(),
    }


def run(configs: int, seed: int, cans: int, mjcf: str | None, joints: list[str] | None) -> dict[str, Any]:
    import mujoco

    from sscbirrt.backends import native_mujoco as nm
    from sscbirrt.demo.scene import UR5E_JOINTS

    joints = joints or list(UR5E_JOINTS)

    def model():
        return mujoco.MjModel.from_xml_path(mjcf) if mjcf else _demo_model(cans)

    probe = nm.NativeScene.from_model(model(), joints)
    arm = {b for b in range(probe.native.nbody) if probe.native.is_arm_body(b)}
    lo, hi = probe.joint_limits
    lo, hi = np.maximum(lo, -np.pi), np.minimum(hi, np.pi)
    qs = [list(map(float, q)) for q in np.random.default_rng(seed).uniform(lo, hi, size=(configs, len(joints)))]

    variants: dict[str, Any] = {}
    agreement: dict[str, Any] = {}
    for margin in MARGINS:
        m = model()
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in arm:
                m.geom_margin[g] = margin  # clearance on the environment, as a consumer sets it
        scene = nm.NativeScene.from_model(m, joints)
        snap = nm.Snapshot.capture(scene, mujoco.MjData(m))
        mod = nm._load()
        pair = {
            "all": mod.SceneValidator(scene.native, snap.native, filter_passive_pairs=False),
            "filtered": mod.SceneValidator(scene.native, snap.native),
        }
        # Untimed: the two must agree on every configuration.
        same_decision = same_contacts = 0
        raw = {k: 0 for k in pair}
        valid = {k: 0 for k in pair}
        for q in qs:
            a, f = pair["all"].invalid_contacts(q), pair["filtered"].invalid_contacts(q)
            key = [(c.body1, c.body2, c.geom1, c.geom2, c.dist, c.kind) for c in a]
            same_contacts += key == [(c.body1, c.body2, c.geom1, c.geom2, c.dist, c.kind) for c in f]
            same_decision += (not a) == (not f)
            for k, v in pair.items():
                raw[k] += v.raw_contact_count(q)
                valid[k] += not (a if k == "all" else f)
        # Timed: alternating blocks, is_valid only.
        seconds = {k: 0.0 for k in pair}
        for start in range(0, configs, BLOCK):
            block = qs[start : start + BLOCK]
            order = list(pair) if (start // BLOCK) % 2 == 0 else list(reversed(pair))
            for k in order:
                v = pair[k]
                t = time.perf_counter()
                for q in block:
                    v.is_valid(q)
                seconds[k] += time.perf_counter() - t
        label = f"{margin * 1000:g}mm"
        for k in pair:
            variants[f"{k}_{label}"] = {
                "pairs": k,
                "margin_m": margin,
                "per_check_us": 1e6 * seconds[k] / configs,
                "valid_fraction": valid[k] / configs,
                "mean_contacts_generated": raw[k] / configs,
            }
        agreement[label] = {"decisions": same_decision == configs, "contacts": same_contacts == configs}

    us = {k: v["per_check_us"] for k, v in variants.items()}
    ratio = us["filtered_5mm"] / us["filtered_0mm"]
    return {
        "artifact": "sscbirrt collision benchmark",
        "issue": "https://github.com/personalrobotics/sscbirrt/issues/254",
        "provenance": _provenance(),
        "scene": {
            "source": mjcf or f"sscbirrt.demo.scene.build_scene with {cans} resting cans",
            "joints": joints,
            "nbody": int(probe.native.nbody),
            "ngeom": int(probe.native.ngeom),
            "arm_bodies": len(arm),
        },
        "configurations": configs,
        "seed": seed,
        "variants": variants,
        "filtered_equals_all": agreement,
        "ratios": {
            "filtered_5mm_over_filtered_0mm": ratio,
            "all_5mm_over_all_0mm": us["all_5mm"] / us["all_0mm"],
            "filtered_5mm_over_all_5mm": us["filtered_5mm"] / us["all_5mm"],
        },
        "verdict": {"limit": RATIO_LIMIT, "pass": bool(ratio <= RATIO_LIMIT)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cans", type=int, default=8)
    parser.add_argument("--mjcf", help="time another compiled world instead of the demo scene")
    parser.add_argument("--joints", nargs="+", help="its planned joints (default: the UR5e's)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run(args.configs, args.seed, args.cans, args.mjcf, args.joints)
    text = json.dumps(report, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    for name, v in report["variants"].items():
        print(f"{name:14s} {v['per_check_us']:8.2f} us/check  valid {v['valid_fraction']:.3f}  "
              f"contacts {v['mean_contacts_generated']:.1f}")  # fmt: skip
    print("filtered equals all:", report["filtered_equals_all"])
    print("ratios:", {k: round(v, 3) for k, v in report["ratios"].items()}, "verdict:", report["verdict"])
    return 0 if all(a["decisions"] and a["contacts"] for a in report["filtered_equals_all"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
