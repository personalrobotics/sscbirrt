# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Per-check cost of the native MuJoCo validator with a planning clearance (#254, #256).

The scene is the demo world (UR5e, Robotiq 2F-85, table) with free cans resting on the table in a tight grid, so a
clearance makes every can come within it of the table and its neighbours: environment pairs the policy discards.
Over the same seeded random arm configurations, on one native scene, it times ``is_valid`` for:

- the first-class clearance (``NativeCollisionChecker(..., clearance=c)``) at 0 and 5 mm, each with the passive
  pairs filtered (the default) and with a full ``mj_collision`` (``filter_passive_pairs=False``, the reference);
- the model-margin reference at 5 mm: the same world with 5 mm margins on every environment geom in the model
  itself (``body_margin`` recomputed as MuJoCo's compiler does), checked at clearance 0. With no held object and
  no arm margins it collides the same pairs with the same margins, so its cost against the first-class 5 mm check
  separates the implementation's overhead from the extra collision work a clearance causes.

Blocks of configurations alternate between the variants so all see the same machine state. Untimed, it records
that the filtered and full variants agree, and that the first-class and model-margin 5 mm checks agree, on every
decision and ordered contact list (distances included), and the mean number of contacts MuJoCo generates per check.

Timings belong to one host: run it as a single process on a quiet machine and compare only within one report.

    uv run python tools/collision_benchmark.py --output collision_benchmark.json
    uv run python tools/collision_benchmark.py --configs 2000 --cans 16
    uv run python tools/collision_benchmark.py --mjcf world.xml --joints j1 j2 ...   # another world, as compiled
"""

from __future__ import annotations

import argparse
import hashlib
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

CLEARANCE = 0.005
RATIO_LIMIT = 1.35  # #256: at 5 mm, about 1.35x the clearance-0 check on Geodude's scene; diagnostic, not a gate
BLOCK = 500


def _demo_model(cans: int):
    from sscbirrt.demo.scene import CAN_RADIUS, build_scene, can_position

    pitch = 2 * CAN_RADIUS + 0.004  # neighbours 4 mm apart: inside a 5 mm clearance
    cols = 4
    grid = {
        f"can{i}": can_position([0.62 + (i // cols) * pitch, (i % cols - (cols - 1) / 2) * pitch]) for i in range(cans)
    }
    return build_scene(cans=grid)


def _with_model_margins(model, arm_bodies: set[int], margin: float) -> None:
    """``margin`` on every environment geom of ``model``, and ``body_margin`` as MuJoCo's compiler computes it (the
    maximum over a body's geoms of margin + gap). Writing ``geom_margin`` alone leaves ``body_margin`` stale, and
    MuJoCo's BVH midphase would then prune pairs of multi-geom bodies with the old margin."""
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in arm_bodies:
            model.geom_margin[g] = max(model.geom_margin[g], margin)
    for b in range(model.nbody):
        geoms = range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b])
        model.body_margin[b] = max([model.geom_margin[g] + model.geom_gap[g] for g in geoms], default=0.0)


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None  # not a checkout (an installed tool or an archive)

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


def _key(validator, q) -> list:
    return [(c.body1, c.body2, c.geom1, c.geom2, c.dist, c.kind) for c in validator.invalid_contacts(q)]


def run(configs: int, seed: int, cans: int, mjcf: str | None, joints: list[str] | None) -> dict[str, Any]:
    import mujoco

    from sscbirrt.backends import native_mujoco as nm
    from sscbirrt.demo.scene import UR5E_JOINTS

    joints = joints or list(UR5E_JOINTS)

    def model():
        return mujoco.MjModel.from_xml_path(mjcf) if mjcf else _demo_model(cans)

    base = model()
    scene = nm.NativeScene.from_model(base, joints)
    arm = {b for b in range(scene.native.nbody) if scene.native.is_arm_body(b)}
    lo, hi = scene.joint_limits
    lo, hi = np.maximum(lo, -np.pi), np.minimum(hi, np.pi)
    qs = [list(map(float, q)) for q in np.random.default_rng(seed).uniform(lo, hi, size=(configs, len(joints)))]
    snap = nm.Snapshot.capture(scene, mujoco.MjData(base))
    mod = nm._load()

    margined = model()
    _with_model_margins(margined, arm, CLEARANCE)
    margin_scene = nm.NativeScene.from_model(margined, joints)
    margin_snap = nm.Snapshot.capture(margin_scene, mujoco.MjData(margined))

    variants = {
        "all_0mm": mod.SceneValidator(scene.native, snap.native, filter_passive_pairs=False),
        "filtered_0mm": mod.SceneValidator(scene.native, snap.native),
        "all_5mm": mod.SceneValidator(scene.native, snap.native, filter_passive_pairs=False, clearance=CLEARANCE),
        "filtered_5mm": mod.SceneValidator(scene.native, snap.native, clearance=CLEARANCE),
        "model_margin_5mm": mod.SceneValidator(margin_scene.native, margin_snap.native),
    }

    # Untimed: agreement on every configuration, validity, and contacts generated.
    same = {"filtered_equals_all_0mm": 0, "filtered_equals_all_5mm": 0, "first_class_equals_model_margin_5mm": 0}
    valid = {k: 0 for k in variants}
    raw = {k: 0 for k in variants}
    for q in qs:
        keys = {k: _key(v, q) for k, v in variants.items()}
        same["filtered_equals_all_0mm"] += keys["filtered_0mm"] == keys["all_0mm"]
        same["filtered_equals_all_5mm"] += keys["filtered_5mm"] == keys["all_5mm"]
        same["first_class_equals_model_margin_5mm"] += keys["filtered_5mm"] == keys["model_margin_5mm"]
        for k, v in variants.items():
            valid[k] += not keys[k]
            raw[k] += v.raw_contact_count(q)

    # Timed: alternating blocks, is_valid only.
    seconds = {k: 0.0 for k in variants}
    names = list(variants)
    for start in range(0, configs, BLOCK):
        block = qs[start : start + BLOCK]
        order = names if (start // BLOCK) % 2 == 0 else list(reversed(names))
        for k in order:
            v = variants[k]
            t = time.perf_counter()
            for q in block:
                v.is_valid(q)
            seconds[k] += time.perf_counter() - t

    report_variants = {
        k: {
            "pairs": "all" if k.startswith("all") else "filtered",
            "clearance_m": CLEARANCE if k in ("all_5mm", "filtered_5mm") else 0.0,
            "model_margin_m": CLEARANCE if k == "model_margin_5mm" else 0.0,
            "per_check_us": 1e6 * seconds[k] / configs,
            "valid_fraction": valid[k] / configs,
            "mean_contacts_generated": raw[k] / configs,
        }
        for k in variants
    }
    us = {k: v["per_check_us"] for k, v in report_variants.items()}
    ratio = us["filtered_5mm"] / us["filtered_0mm"]
    return {
        "artifact": "sscbirrt collision benchmark",
        "issues": [
            "https://github.com/personalrobotics/sscbirrt/issues/254",
            "https://github.com/personalrobotics/sscbirrt/issues/256",
        ],
        "provenance": _provenance(),
        "scene": {
            "source": mjcf or f"sscbirrt.demo.scene.build_scene with {cans} resting cans",
            "mjb_sha256": scene.provenance["mjb_sha256"],
            "joints": joints,
            "nbody": int(scene.native.nbody),
            "ngeom": int(scene.native.ngeom),
            "arm_bodies": len(arm),
        },
        "configurations": configs,
        "seed": seed,
        "variants": report_variants,
        "agreement": {k: n == configs for k, n in same.items()},
        "ratios": {
            "filtered_5mm_over_filtered_0mm": ratio,
            "filtered_5mm_over_model_margin_5mm": us["filtered_5mm"] / us["model_margin_5mm"],
            "model_margin_5mm_over_filtered_0mm": us["model_margin_5mm"] / us["filtered_0mm"],
            "all_5mm_over_all_0mm": us["all_5mm"] / us["all_0mm"],
        },
        "diagnostic": {"limit": RATIO_LIMIT, "within": bool(ratio <= RATIO_LIMIT)},
        "model_sha256": hashlib.sha256(nm.export_mjb(base)).hexdigest(),
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
        print(f"{name:17s} {v['per_check_us']:8.2f} us/check  valid {v['valid_fraction']:.3f}  "
              f"contacts {v['mean_contacts_generated']:.1f}")  # fmt: skip
    print("agreement:", report["agreement"])
    print("ratios:", {k: round(v, 3) for k, v in report["ratios"].items()}, "diagnostic:", report["diagnostic"])
    return 0 if all(report["agreement"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
