# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The sstsr conformance corpus for the native TSR runtime (#87, #127).

Generates ``tests/reference/tsr_conformance.json`` from sstsr with a fixed
seed: a set of regions (identity and non-identity frames, zero-width rows,
full rotations, outer rotational intervals, near-singular pitch) and, for
each, probe transforms on and off the region with sstsr's ``contains``,
``distance``, and ``closest_transform`` results. ``tests/test_tsr_conformance.py``
runs the native ``TSR`` on every probe. Regenerate with

    uv run python tools/tsr_conformance.py
    uv run python tools/tsr_conformance.py --check   # semantics unchanged?
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tsr import TSR
from tsr.sampling import _interval_sum

ARTIFACT = Path(__file__).resolve().parent.parent / "tests" / "reference" / "tsr_conformance.json"
SEED = 20260929


def _frame(x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    return TSR.xyzrpy_to_trans(np.array([x, y, z, roll, pitch, yaw], dtype=float))


def regions() -> list[tuple[str, TSR]]:
    pi = np.pi
    box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-pi, pi]])
    return [
        ("identity_box_free_yaw", TSR(np.eye(4), np.eye(4), box)),
        ("offset_frames", TSR(_frame(1.0, 2.0, 0.0, 0.0, 0.0, 0.5), _frame(0.1, 0.0, 0.0), box)),
        (
            "tilted_frame_band",
            TSR(
                _frame(0.5, -0.2, 0.3, 0.3, -0.4, 1.2),
                np.eye(4),
                np.array([[-2, 2], [-0.6, 0.6], [0, 0], [0, 0], [0, 0], [-pi, pi]]),
            ),
        ),
        ("point_pose", TSR(_frame(0.2, 0.2, 0.2, 0.1, 0.2, 0.3), np.eye(4), np.zeros((6, 2)))),
        (
            "full_rotations",
            TSR(np.eye(4), np.eye(4), np.array([[0, 0.1], [0, 0], [0, 0], [-pi, pi], [-pi, pi], [-pi, pi]])),
        ),
        (
            "outer_yaw_interval",
            TSR(np.eye(4), np.eye(4), np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [3 * pi / 4, -3 * pi / 4]])),
        ),
        (
            "outer_roll_and_pitch_band",
            TSR(
                _frame(0, 0, 1.0),
                _frame(0, 0, -0.1),
                np.array([[-0.3, 0.3], [-0.3, 0.3], [-0.1, 0.1], [2.5, -2.5], [-0.2, 0.2], [-1.0, 1.0]]),
            ),
        ),
        (
            "near_singular_pitch",
            TSR(
                np.eye(4),
                np.eye(4),
                np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.5, 0.5], [pi / 2 - 1e-3, pi / 2], [-0.5, 0.5]]),
            ),
        ),
        (
            "wide_roll_window",
            TSR(
                _frame(0, 0, 0, 0, 0, -2.0),
                _frame(0, 0.05, 0),
                np.array([[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02], [-3.0, 3.0], [-0.3, 0.3], [-0.1, 0.1]]),
            ),
        ),
    ]


def probes(t: TSR, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for _ in range(6):
        out.append(t.sample(rng=rng))
    for _ in range(6):
        s = t.sample(rng=rng)
        off = _frame(*(rng.uniform(-0.3, 0.3, 3)), *(rng.uniform(-1.0, 1.0, 3)))
        out.append(off @ s)
    for pitch in (np.pi / 2 - 1e-7, -np.pi / 2 + 1e-7, np.pi / 2, -np.pi / 2):
        out.append(t.T0_w @ _frame(0.01, -0.01, 0.0, 0.2, pitch, 0.4) @ t.Tw_e)
    out.append(t.T0_w @ _frame(0.0, 0.0, 0.0, np.pi, 0.0, np.pi) @ t.Tw_e)  # an RPY-equivalent representation
    out.append(t.T0_w @ _frame(0.0, 0.0, 0.0, 0.0, 0.0, -np.pi) @ t.Tw_e)  # yaw exactly at the seam
    # sstsr 4 (#190, sscbirrt #274): membership is exact, so poses just past a bound are refused with that distance.
    # Probe the old 1e-3 band: on, 2e-4, 9e-4 and 1.1e-3 past the upper bound of the first nonzero-width translation
    # row and of the first rotation row narrower than a full turn, from a pose at the region's centre.
    cont = np.asarray(t._Bw_cont)
    centre = (cont[:, 0] + cont[:, 1]) / 2
    for rows in (range(3), range(3, 6)):
        row = next((i for i in rows if 0 < cont[i, 1] - cont[i, 0] < 2 * np.pi), None)
        if row is None:
            continue
        for past in (0.0, 2e-4, 9e-4, 1.1e-3):
            c = centre.copy()
            c[row] = cont[row, 1] + past
            out.append(t.T0_w @ _frame(*c) @ t.Tw_e)
    return out


def record(name: str, t: TSR, rng: np.random.Generator) -> dict[str, Any]:
    entries = []
    for T in probes(t, rng):
        dist, closest = t.closest_transform(T)
        entries.append(
            {
                "T": T.tolist(),
                "contains": bool(t.contains(T)),
                "distance": float(dist),
                "closest": closest.tolist(),
                "closest_contained": bool(t.contains(closest)),
            }
        )
    xy = t.sample_xyzrpy(rng=np.random.default_rng(7))
    return {
        "name": name,
        "T0_w": t.T0_w.tolist(),
        "Tw_e": t.Tw_e.tolist(),
        "Bw": t.Bw.tolist(),
        "continuous_bounds": t._Bw_cont.tolist(),
        "volume": _interval_sum(t.Bw),
        "sample_xyzrpy_seed7_python": xy.tolist(),
        "probes": entries,
    }


def generate() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    return {
        "artifact": "sstsr conformance corpus for the native TSR runtime",
        "issue": "https://github.com/personalrobotics/sscbirrt/issues/87",
        "seed": SEED,
        "versions": {"sstsr": importlib.metadata.version("sstsr"), "numpy": np.__version__},
        "regions": [record(name, t, rng) for name, t in regions()],
    }


ATOL = 1e-9  # distances and transforms compare within this; sstsr on another platform differs in the last bits


def mismatches(stored: dict[str, Any], fresh: dict[str, Any], atol: float = ATOL) -> list[str]:
    """Where a fresh sstsr run disagrees with the stored corpus: booleans exactly, floats within atol."""
    out = []
    by_name = {r["name"]: r for r in stored["regions"]}
    for r in fresh["regions"]:
        old = by_name.get(r["name"])
        if old is None:
            out.append(f"{r['name']}: not in the stored corpus")
            continue
        if len(old["probes"]) != len(r["probes"]):
            out.append(f"{r['name']}: {len(old['probes'])} stored probes vs {len(r['probes'])} fresh")
            continue
        for i, (x, y) in enumerate(zip(old["probes"], r["probes"])):
            where = f"{r['name']}[{i}]"
            if x["contains"] != y["contains"]:
                out.append(f"{where}.contains: stored={x['contains']} fresh={y['contains']}")
            if abs(x["distance"] - y["distance"]) > atol:
                out.append(f"{where}.distance: stored={x['distance']!r} fresh={y['distance']!r}")
            if not np.allclose(x["closest"], y["closest"], atol=atol, rtol=0.0):
                diff = np.abs(np.array(x["closest"]) - np.array(y["closest"])).max()
                out.append(f"{where}.closest: max |diff| = {diff:.3g}")
            if x["closest_contained"] != y["closest_contained"]:
                out.append(f"{where}.closest_contained: stored={x['closest_contained']} fresh={y['closest_contained']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    fresh = generate()
    if args.check:
        stored = json.loads(args.output.read_text())
        bad = mismatches(stored, fresh)
        if bad:
            print("MISMATCH: sstsr's results on the conformance corpus changed:", file=sys.stderr)
            for line in bad:
                print(f"  {line}", file=sys.stderr)
            return 1
        n = sum(len(r["probes"]) for r in fresh["regions"])
        print(
            f"conformance corpus unchanged ({len(fresh['regions'])} regions, {n} probes, within {ATOL}; "
            f"versions {fresh['versions']})"
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    n = sum(len(r["probes"]) for r in fresh["regions"])
    print(f"wrote {args.output} ({len(fresh['regions'])} regions, {n} probes, versions {fresh['versions']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
