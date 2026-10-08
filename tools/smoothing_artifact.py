# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""An inspectable record of path smoothing (#207) on the pick and transport demos.

For each problem and seed it plans with ``smooth=True`` and records:
- the planned polyline and its true corners;
- every corner's blend attempts (size, midpoint deviation, why rejected) and outcome;
- the stops (corners kept, where a retimer comes to rest);
- the smoothed path sampled densely;
- an independent re-check of those samples with the scene's collision checker;
- path lengths;
- for transport, the held can's tilt along the smoothed path against the upright constraint (the residual).

    uv run python tools/smoothing_artifact.py                 # writes tests/reference/smoothing_artifact.json
    uv run python tools/smoothing_artifact.py --check         # regenerate and compare the outcomes

Paths depend on the platform's floating point, so ``--check`` compares outcomes on the machine that wrote the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "tests" / "reference" / "smoothing_artifact.json"
RECHECK = 0.01  # rad between the samples the re-check tests
RECORD = 0.05  # rad between the samples stored in the record


def record(problem: str, seed: int) -> dict[str, Any]:
    import mujoco

    from sscbirrt.backends.native_mujoco import NativeCollisionChecker, NativeScene, Snapshot
    from sscbirrt.demo.scenarios import all_scenarios, transport
    from sscbirrt.mujoco import _attachments, plan
    from sscbirrt.smoothing import coalesce

    p = all_scenarios()[problem].problem()
    r = plan(p.model, p.data, p.arm, config=p.config, seed=seed, smooth=True, **p.kwargs)
    sp = r.smooth_path
    scene = NativeScene.from_model(p.model, p.arm.joints)
    attachments = _attachments(p.model, p.data, p.arm, p.kwargs.get("holding"))
    checker = NativeCollisionChecker(scene, Snapshot.capture(scene, p.data, attachments))
    smoothed = sp.to_polyline(RECHECK)
    out: dict[str, Any] = {
        "problem": problem,
        "seed": seed,
        "planned": {
            "waypoints": len(r.path),
            "length": sp.report.original_length,
            "corners": [q.tolist() for q in coalesce(r.path)],
        },  # fmt: skip
        "corners": [
            {
                "index": c.index,
                "turn_deg": float(np.degrees(c.turn)),
                "outcome": c.outcome,
                "attempts": [{**a, "d": float(a["d"]), "deviation": float(a["deviation"])} for a in c.attempts],
            }  # fmt: skip
            for c in sp.report.corners
        ],
        "stops": [q.tolist() for q in sp.stops],
        "segments": len(sp.segments),
        "smoothed": {
            "length": sp.report.smoothed_length,
            "samples": [q.tolist() for q in sp.to_polyline(RECORD)],
            "endpoints_exact": bool(np.array_equal(sp.start, r.path[0]) and np.array_equal(sp.goal, r.path[-1])),
            "recheck_spacing": RECHECK,
            "recheck_collision_free": bool(all(checker.is_valid(q) for q in smoothed)),
        },
    }
    if problem == "transport":
        view = mujoco.MjData(p.model)
        tilts = [float(np.degrees(transport.tilt(p.model, view, q))) for q in smoothed]
        limit = float(np.degrees(np.hypot(transport.TILT_LIMIT, transport.TILT_LIMIT)))
        out["constraint"] = {"name": "upright (roll and pitch each within TILT_LIMIT)", "tilt_limit_deg": limit,
                             "max_tilt_deg_smoothed": max(tilts),
                             "max_tilt_deg_planned": max(float(np.degrees(transport.tilt(p.model, view, q)))
                                                         for q in r.path)}  # fmt: skip
    return out


def build(seeds: list[int]) -> dict[str, Any]:
    import importlib.metadata

    return {
        "issue": 207,
        "sscbirrt": importlib.metadata.version("sscbirrt"),
        "records": [record(problem, seed) for problem in ("pick", "transport") for seed in seeds],
    }


def outcomes(artifact: dict[str, Any]) -> list[Any]:
    return [(r["problem"], r["seed"], [c["outcome"] for c in r["corners"]], r["segments"]) for r in artifact["records"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--output", default=str(ARTIFACT))
    ap.add_argument("--check", action="store_true", help="regenerate and compare outcomes with the stored file")
    args = ap.parse_args(argv)
    artifact = build(list(range(args.seeds)))
    if args.check:
        stored = json.loads(Path(args.output).read_text())
        same = outcomes(stored) == outcomes(artifact)
        print("outcomes match" if same else "outcomes differ")
        return 0 if same else 1
    Path(args.output).write_text(json.dumps(artifact, indent=1) + "\n")
    blended = sum(c["outcome"] == "blended" for r in artifact["records"] for c in r["corners"])
    total = sum(len(r["corners"]) for r in artifact["records"])
    print(f"wrote {args.output}: {len(artifact['records'])} records, {blended}/{total} corners blended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
