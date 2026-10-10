# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Benchmark for #214: bounded-acceleration trajectory smoothing against blends + TOPP-RA, under #214's gates.

For pick and transport, over paired seeds, at equal limits (1.57 rad/s and 4 rad/s^2 per joint), it builds three
kinds of trajectory and records duration, smoothing time, planning time and every validation:

- ``raw``: the planned polyline at rest at every corner (exactly the planned geometry; the safe fallback);
- ``blend_toppra``: PR #211's blended segments retimed rest to rest by TOPP-RA (needs ``toppra``). If its 2 ms
  command fails the runtime checks, it is re-run with limits x0.98, x0.95, x0.9, and the scale used is recorded;
- ``smoother_N``: time-optimized shortcuts (``sscbirrt._trajectory``) with an N-iteration budget.

Validation, applied to every result:

- the source trajectory re-checked through the planning boundary at 0.01 rad (samples and chords);
- the runtime command (positions every 2 ms, integer ns) checked with ssrobot's speed and acceleration rules, ported
  exactly from ``ssrobot.validation._check_speeds`` and ``_check_accelerations``. They are exact ``Fraction``
  arithmetic with a 1e-9 relative allowance and rest at both ends. The command is also checked for collision and
  path constraint along its own linear chords;
- exact endpoints and rest at both ends.

    uv run python tools/trajectory_benchmark.py run --seeds 100 --output benchmarks/trajectory/run.json
    uv run python tools/trajectory_benchmark.py gates benchmarks/trajectory/run.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VMAX, AMAX = 1.57, 4.0
TICK_NS = 2_000_000  # the runtime's control period: 2 ms, as ssrobot's example arm
DENSE = 0.01  # rad between samples of the dense diagnostic (not a gate: the declared resolution is the guarantee)
BUDGETS = (25, 100, 400)


# --------------------------------------------------------------------------------------------------------------------
# ssrobot's runtime checks, ported exactly (ssrobot/src/ssrobot/validation.py, _check_speeds and _check_accelerations)
# --------------------------------------------------------------------------------------------------------------------


def runtime_violations(times_ns: list[int], positions: np.ndarray, vmax: float, amax: float) -> list[str]:
    """Every speed and acceleration violation of a linearly interpolated command, as ssrobot computes them."""
    out: list[str] = []
    allowance = 1 + Fraction(1, 10**9)
    durations = [times_ns[i] - times_ns[i - 1] for i in range(1, len(times_ns))]
    if times_ns[0] != 0 or any(d <= 0 for d in durations):
        return ["times must start at 0 and strictly increase"]
    for k in range(positions.shape[1]):
        col = [Fraction(float(x)) for x in positions[:, k]]
        for i, d in enumerate(durations):
            speed = abs(col[i + 1] - col[i]) * 10**9 / d
            if speed > Fraction(vmax) * allowance:
                out.append(f"joint {k} speed {float(speed):.6g} on segment {i}")
        speeds = [Fraction(0)] + [(col[i + 1] - col[i]) / d for i, d in enumerate(durations)] + [Fraction(0)]
        spans = [0, *durations, 0]
        for i in range(len(times_ns)):
            mean = Fraction(spans[i] + spans[i + 1], 2)
            acc = abs(speeds[i + 1] - speeds[i]) / mean * 10**18
            if acc > Fraction(amax) * allowance:
                out.append(f"joint {k} acceleration {float(acc):.6g} at waypoint {i}")
    return out


# --------------------------------------------------------------------------------------------------------------------
# Problems and validation
# --------------------------------------------------------------------------------------------------------------------


def load(name: str):
    """``(solve(seed) -> (result, planner, problem))`` for a demo problem, capturing the problem sscbirrt built."""
    from sscbirrt import CBiRRT
    from sscbirrt.demo.scenarios import all_scenarios
    from sscbirrt.mujoco import plan

    p = all_scenarios()[name].problem()
    captured: dict[str, Any] = {}
    original = CBiRRT.solve

    def solve(seed: int):
        def spy(self, problem, seed=None, smooth=None):
            captured["planner"], captured["problem"] = self, problem
            return original(self, problem, seed=seed, smooth=smooth)

        CBiRRT.solve = spy
        try:
            t = time.perf_counter()
            result = plan(p.model, p.data, p.arm, config=p.config, seed=seed, **p.kwargs)
            wall = time.perf_counter() - t
        finally:
            CBiRRT.solve = original
        return result, wall, captured["planner"], captured["problem"]

    return solve


def check_source(positions: np.ndarray, planner, problem) -> str | None:
    """The source trajectory, densely sampled, through the planning boundary (samples and chords)."""
    from sscbirrt._trajectory import validate_samples

    return validate_samples(positions, planner._motion_validator(problem), lambda q: planner._admissible(problem, q))


def arc_samples(positions: np.ndarray, spacing: float) -> np.ndarray:
    """Points along the polyline through ``positions`` at arc-length spacing ``spacing``, from the first point, plus
    the last: the motion as a validator at that resolution sees it, wherever the polyline's own vertices fall."""
    seg = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.append(np.arange(0.0, cum[-1], spacing), cum[-1])
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    frac = np.where(seg[idx] > 0, (s - cum[idx]) / np.where(seg[idx] > 0, seg[idx], 1.0), 0.0)
    return positions[idx] + (positions[idx + 1] - positions[idx]) * frac[:, None]


def resample(positions: np.ndarray, spacing: float) -> np.ndarray:
    """Insert points so no two consecutive ones are farther apart than ``spacing`` (linear between samples)."""
    out = [positions[0]]
    for a, b in zip(positions, positions[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(out)


def evaluate(times_ns, positions, source_positions, planner, problem, start, goal) -> dict[str, Any]:
    """#214's gates at one declared resolution r (the problem's edge_resolution): the source trajectory and the
    runtime command are each checked at arc-length spacing r, as the planner checks an edge. The runtime command's
    speeds and accelerations are checked exactly, as ssrobot does. Denser checks are recorded as diagnostics."""
    r = planner.config.edge_resolution
    v = runtime_violations(times_ns, positions, VMAX, AMAX)
    src = check_source(arc_samples(source_positions, r), planner, problem)
    cmd = check_source(arc_samples(positions, r), planner, problem)
    dense = check_source(resample(source_positions, DENSE), planner, problem)
    return {
        "resolution": r,
        "dense_ok": dense is None,
        "dense_reason": dense,
        "runtime_ok": not v,
        "runtime_violations": v[:5],
        "source_ok": src is None,
        "source_reason": src,
        "command_ok": cmd is None,
        "command_reason": cmd,
        "endpoints_ok": bool(
            np.allclose(positions[0], start, atol=1e-9) and np.allclose(positions[-1], goal, atol=1e-9)
        ),
    }


def raw_and_smoother(result, planner, problem, seed: int) -> list[dict[str, Any]]:
    from sscbirrt._trajectory import ShortcutOptions, initial_trajectory, shortcut
    from sscbirrt.smoothing import coalesce

    margin = 1 - 1e-6  # float rounding of sampled positions must stay within ssrobot's exact check
    vmax, amax = np.full(len(result.path[0]), VMAX * margin), np.full(len(result.path[0]), AMAX * margin)
    traj0 = initial_trajectory(coalesce(result.path), vmax, amax)
    rows = []

    def record(method, traj, seconds):
        times, q = traj.sample(TICK_NS)
        src = traj.evaluate(np.linspace(0.0, traj.duration, max(2, int(np.ceil(traj.duration / 0.002)) + 1)))[0]
        row = {"method": method, "duration": traj.duration, "smoothing_seconds": seconds}
        row.update(evaluate(times, q, src, planner, problem, result.path[0], result.path[-1]))
        rows.append(row)

    record("raw", traj0, 0.0)
    for budget in BUDGETS:
        t = time.perf_counter()
        traj = shortcut(
            traj0,
            motion_validator=planner._motion_validator(problem),
            admissible=lambda q: planner._admissible(problem, q),
            resolution=planner.config.edge_resolution,
            options=ShortcutOptions(iterations=budget, seed=seed),
        )
        record(f"smoother_{budget}", traj, time.perf_counter() - t)
        rows[-1]["accepted"] = traj.diagnostics["accepted"]
    return rows


def blend_toppra(result, planner, problem) -> dict[str, Any]:
    """PR #211's segments retimed rest to rest by TOPP-RA, re-run with scaled limits until the command passes."""
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint

    from sscbirrt.smoothing import Blend

    ta.setup_logging("CRITICAL")
    t0 = time.perf_counter()
    sp = planner.smooth(problem, result.path)
    dof = len(result.path[0])

    def grid(seg, per_line, per_blend):
        br = seg.breaks
        return np.unique(np.concatenate([np.linspace(br[i], br[i + 1], per_blend if isinstance(pc, Blend) else per_line)
                                         for i, pc in enumerate(seg.pieces)]))  # fmt: skip

    def retime(seg, scale):
        vl = np.tile([-VMAX * scale, VMAX * scale], (dof, 1))
        al = np.tile([-AMAX * scale, AMAX * scale], (dof, 1))
        for per_line, per_blend in ((60, 40), (20, 20), (6, 10)):
            limits = [constraint.JointVelocityConstraint(vl), constraint.JointAccelerationConstraint(al)]
            g = grid(seg, per_line, per_blend)
            inst = algo.TOPPRA(limits, seg, gridpoints=g, parametrizer="ParametrizeConstAccel")
            traj = inst.compute_trajectory(0, 0)
            if traj is not None:
                return traj
        return None

    for scale in (1.0, 0.98, 0.95, 0.9):
        trajs = [retime(seg, scale) for seg in sp.segments]
        if any(t is None for t in trajs):
            return {"method": "blend_toppra", "failed": "TOPP-RA could not parameterize a segment", "scale": scale}
        # one command: segments back to back, each rest to rest
        times, rows, src, offset = [0], [trajs[0](0.0)], [], 0
        for tr in trajs:
            end_ns = int(np.ceil(tr.duration * 1e9))
            ts = list(range(TICK_NS, end_ns, TICK_NS)) + [end_ns]
            rows += [tr(min(t * 1e-9, tr.duration)) for t in ts]
            times += [offset + t for t in ts]
            src.append(tr(np.linspace(0, tr.duration, max(2, int(np.ceil(tr.duration / 0.002)) + 1))))
            offset += end_ns
        q = np.array(rows)
        if not runtime_violations(times, q, VMAX, AMAX):
            break
    seconds = time.perf_counter() - t0
    row = {"method": "blend_toppra", "duration": offset * 1e-9, "smoothing_seconds": seconds, "scale": scale}
    row.update(evaluate(times, q, np.vstack(src), planner, problem, result.path[0], result.path[-1]))
    return row


def _run(args) -> None:
    out = {"issue": 214, "limits": {"velocity": VMAX, "acceleration": AMAX}, "tick_ns": TICK_NS, "budgets": BUDGETS,
           "seeds": args.seeds, "records": []}  # fmt: skip
    for name in args.problems:
        solve = load(name)
        for seed in range(args.seeds):
            result, wall, planner, problem = solve(seed)
            stats = result.stats or {}
            solve_s = float(stats.get("seconds_roots", 0.0)) + float(result.planning_time)
            base = {"problem": name, "seed": seed, "planning_seconds": wall, "solve_seconds": solve_s,
                    "plan_success": result.success}  # fmt: skip
            if not result.success:
                out["records"].append({**base, "rows": []})
                continue
            rows = raw_and_smoother(result, planner, problem, seed)
            if not args.no_toppra:
                rows.append(blend_toppra(result, planner, problem))
            out["records"].append({**base, "rows": rows})
            if not args.quiet:
                d = {r["method"]: round(r.get("duration", float("nan")), 2) for r in rows}
                print(f"{name} s{seed}: {d}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"wrote {args.output}")


# --------------------------------------------------------------------------------------------------------------------
# Gates (#214)
# --------------------------------------------------------------------------------------------------------------------


def _boot(fn, *arrays, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    idx = rng.integers(0, m, size=(n, m))
    vals = np.array([fn(*(a[i] for a in arrays)) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _gates(args) -> None:
    """#214's gates. The smoothing-time gate is reported against two readings of "planning wall time": the whole
    ``sscbirrt.mujoco.plan`` call (scene construction included) and the solver alone (roots plus search)."""
    data = json.loads(Path(args.run).read_text())
    for name in dict.fromkeys(r["problem"] for r in data["records"]):
        recs = [r for r in data["records"] if r["problem"] == name and r["rows"]]
        by: dict[str, dict[int, dict[str, Any]]] = {}
        for r in recs:
            for row in r["rows"]:
                by.setdefault(row["method"], {})[r["seed"]] = row
        print(f"\n== {name}: {len(recs)} planned seeds")
        for method, seeds in by.items():
            rows = list(seeds.values())
            valid = [r for r in rows if _ok(r)]
            durs = [r["duration"] for r in valid] or [float("nan")]
            smooth = [r.get("smoothing_seconds", 0.0) for r in valid] or [0.0]
            print(f"  {method:14s} valid {len(valid)}/{len(rows)}  median {np.median(durs):.2f}s  "
                  f"p90 {np.percentile(durs, 90):.2f}s  smoothing p90 {np.percentile(smooth, 90):.3f}s")  # fmt: skip
            for r in [r for r in rows if not _ok(r)][:3]:
                print(f"    invalid: {_why(r)}")
        plan_p90 = float(np.percentile([r["planning_seconds"] for r in recs], 90))
        solve_p90 = float(np.percentile([r["solve_seconds"] for r in recs], 90))
        base = by.get("blend_toppra", {})
        for budget in data["budgets"]:
            prop = by.get(f"smoother_{budget}", {})
            pairs = [(prop[s], base[s]) for s in prop if s in base and _ok(prop[s]) and _ok(base[s])]
            if not pairs:
                continue
            a = np.array([p["duration"] for p, _ in pairs])
            b = np.array([q["duration"] for _, q in pairs])
            lo_d, hi_d = _boot(lambda x, y: np.median(x - y), a, b)
            lo_r, hi_r = _boot(lambda x, y: np.percentile(x, 90) / np.percentile(y, 90), a, b)
            sp90 = float(np.percentile([p["smoothing_seconds"] for p, _ in pairs], 90))
            hard = all(_ok(r) for r in prop.values())
            speed = hi_d < 0 and hi_r <= 1.0
            head = f"  smoother_{budget} vs blend_toppra, {len(pairs)} pairs:"
            print(f"{head} median diff CI [{lo_d:+.2f}, {hi_d:+.2f}] s, p90 ratio CI [{lo_r:.2f}, {hi_r:.2f}]")
            print(f"    hard gates {_pf(hard)}, duration gates {_pf(speed)}")
            print(f"    smoothing p90 {sp90:.3f}s vs plan() p90 {plan_p90:.3f}s {_pf(sp90 <= plan_p90)}; "
                  f"vs solver-only p90 {solve_p90:.3f}s {_pf(sp90 <= solve_p90)}")  # fmt: skip


def _pf(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _why(r: dict[str, Any]) -> Any:
    return r.get("failed") or r.get("source_reason") or r.get("command_reason") or r.get("runtime_violations")


def _ok(r: dict[str, Any]) -> bool:
    return not r.get("failed") and r["runtime_ok"] and r["source_ok"] and r["command_ok"] and r["endpoints_ok"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--problems", nargs="+", default=["pick", "transport"])
    r.add_argument("--seeds", type=int, default=100)
    r.add_argument("--output", default="trajectory_benchmark.json")
    r.add_argument("--no-toppra", action="store_true", help="skip the blends + TOPP-RA baseline (no toppra installed)")
    r.add_argument("--quiet", action="store_true")
    g = sub.add_parser("gates")
    g.add_argument("run")
    args = ap.parse_args(argv)
    {"run": _run, "gates": _gates}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
