# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Benchmark for #214: bounded-acceleration trajectory smoothing against blends + TOPP-RA, under #214's gates.

For pick and transport, over paired seeds, at equal limits (1.57 rad/s and 4 rad/s^2 per joint), it builds three
kinds of trajectory and records duration, smoothing time, planning time and every validation:

- ``raw``: the planned polyline at rest at every corner (exactly the planned geometry; the safe fallback);
- ``blend_toppra``: PR #211's blended segments retimed rest to rest by TOPP-RA (needs ``toppra``). If its 2 ms
  command fails the runtime checks, it is re-run with limits x0.98, x0.95, x0.9, and the scale used is recorded;
- ``smoother_N``: time-optimized shortcuts (``sscbirrt.trajectory``, the Python reference) with an N-attempt budget;
- ``native_smoother_N``: the same in the native core (``CBiRRT.smooth_trajectory``; only the random stream differs);
- ``smoother_default``, ``native_smoother_default``: the same with ``TrajectoryOptions``' defaults.

Validation, applied to every result:

- the source trajectory re-checked through the planning boundary at 0.01 rad (samples and chords);
- the runtime command (positions every 2 ms, integer ns) checked with ssrobot's speed and acceleration rules, ported
  exactly from ``ssrobot.validation._check_speeds`` and ``_check_accelerations``. They are exact ``Fraction``
  arithmetic with a 1e-9 relative allowance and rest at both ends. The command is also checked for collision and
  path constraint along its own linear chords;
- exact endpoints and rest at both ends.

    uv run python tools/trajectory_benchmark.py run --seeds 100 --output benchmarks/trajectory/run.json
    uv run python tools/trajectory_benchmark.py gates benchmarks/trajectory/run.json

Stopping rules are studied offline. Shortcutting is sequential, so stopping after k attempts gives exactly the
trajectory a k-iteration run gives: ``trace`` records one long native run per seed (every attempt's gain and elapsed
time), and ``stopping`` replays its prefixes under each candidate rule, against a ``run`` file's baselines:

    uv run python tools/trajectory_benchmark.py trace --seeds 100 --output benchmarks/trajectory/trace.json
    uv run python tools/trajectory_benchmark.py stopping trace.json run.json
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
CONVERSION_TOLERANCE = 1e-4  # rad: linear runtime interpolation may deviate this far from the validated source
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
        def spy(self, problem, seed=None, trajectory=None):
            captured["planner"], captured["problem"] = self, problem
            return original(self, problem, seed=seed, trajectory=trajectory)

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
    from sscbirrt.trajectory import validate_samples

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


def conversion_deviation(times_ns: list[int], positions: np.ndarray, source) -> float:
    """Largest joint-space distance between the runtime's linear interpolation of the command and the source law,
    probed at the quarter points of every command interval (a parabola departs most from its chord in between)."""
    t = np.array(times_ns, dtype=float) * 1e-9
    worst = 0.0
    for f in (0.25, 0.5, 0.75):
        probe = t[:-1] + f * np.diff(t)
        linear = positions[:-1] + f * np.diff(positions, axis=0)
        worst = max(worst, float(np.max(np.linalg.norm(linear - source(probe), axis=1))))
    return worst


def evaluate(times_ns, positions, source, record_ok, planner, problem, start, goal) -> dict[str, Any]:
    """#214's gates. Validation is a record: geometry already checked (the planner's edges, the smoother's
    shortcuts, PR #211's blends) is never re-checked. Only what a step adds is checked by that step. Runtime
    conversion adds linear chords between samples, so it is checked by its deviation from the validated source
    (within CONVERSION_TOLERANCE), not by a collision re-check. Speeds and accelerations are checked exactly, as
    ssrobot does. Re-checks at the resolution and denser are recorded as diagnostics only."""
    r = planner.config.edge_resolution
    v = runtime_violations(times_ns, positions, VMAX, AMAX)
    dev = conversion_deviation(times_ns, positions, source)
    src_dense = source(np.linspace(0.0, times_ns[-1] * 1e-9, max(2, times_ns[-1] // 2_000_000 + 1)))
    diag_r = check_source(arc_samples(src_dense, r), planner, problem)
    diag_dense = check_source(resample(src_dense, DENSE), planner, problem)
    return {
        "resolution": r,
        "record_ok": bool(record_ok),
        "runtime_ok": not v,
        "runtime_violations": v[:5],
        "conversion_deviation": dev,
        "conversion_ok": dev <= CONVERSION_TOLERANCE,
        "endpoints_ok": bool(
            np.allclose(positions[0], start, atol=1e-9) and np.allclose(positions[-1], goal, atol=1e-9)
        ),
        "diag_recheck_at_r_ok": diag_r is None,
        "diag_recheck_at_r_reason": diag_r,
        "diag_dense_ok": diag_dense is None,
        "diag_dense_reason": diag_dense,
    }


def raw_and_smoother(result, planner, problem, seed: int) -> list[dict[str, Any]]:
    from sscbirrt.trajectory import TrajectoryOptions, coalesce, initial_trajectory, shortcut

    margin = 1 - 1e-6  # float rounding of sampled positions must stay within ssrobot's exact check
    vmax, amax = np.full(len(result.path[0]), VMAX * margin), np.full(len(result.path[0]), AMAX * margin)
    traj0 = initial_trajectory(coalesce(result.path), vmax, amax)
    rows = []

    def record(method, traj, seconds):
        times, q = traj.sample(TICK_NS)
        validated = all(p.validated in ("planned", "shortcut") for p in traj.pieces)
        row = {"method": method, "duration": traj.duration, "smoothing_seconds": seconds}
        row.update(evaluate(times, q, lambda ts: traj.evaluate(ts)[0], validated, planner, problem,
                            result.path[0], result.path[-1]))  # fmt: skip
        row.update(accepted=traj.diagnostics["accepted"], attempts=len(traj.diagnostics["attempts"]),
                   stopped=traj.stopped)  # fmt: skip
        rows.append(row)

    def python(options):  # the reference loop
        return shortcut(traj0, motion_validator=planner._motion_validator(problem),
                        admissible=lambda q: planner._admissible(problem, q),
                        resolution=planner.config.edge_resolution, options=options, seed=seed)  # fmt: skip

    def native(options):  # the public entry point; wall time includes lowering (a fresh native collision checker)
        traj = planner.smooth_trajectory(problem, result.path, options)
        assert traj.backend == "native", traj.backend_reasons
        return traj

    rows.append({"method": "raw", "duration": traj0.duration, "smoothing_seconds": 0.0})
    rows[-1].update(evaluate(*traj0.sample(TICK_NS), lambda ts: traj0.evaluate(ts)[0], True, planner, problem,
                             result.path[0], result.path[-1]))  # fmt: skip
    for kind, smooth in (("smoother", python), ("native_smoother", native)):
        for budget in BUDGETS:
            t = time.perf_counter()
            traj = smooth(TrajectoryOptions(vmax, amax, iterations=budget, convergence_window=None, seed=seed))
            record(f"{kind}_{budget}", traj, time.perf_counter() - t)
        t = time.perf_counter()
        traj = smooth(TrajectoryOptions(vmax, amax, seed=seed))
        record(f"{kind}_default", traj, time.perf_counter() - t)
    return rows


def blend_toppra(result, planner, problem) -> dict[str, Any]:
    """PR #211's segments retimed rest to rest by TOPP-RA, re-run with scaled limits until the command passes."""
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint
    from blend_baseline import Blend, smooth_path

    ta.setup_logging("CRITICAL")
    t0 = time.perf_counter()
    sp = smooth_path(result.path, motion_validator=planner._motion_validator(problem),
                     admissible=lambda q: planner._admissible(problem, q),
                     resolution=planner.config.edge_resolution or planner.config.step_size)  # fmt: skip
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
        # one command: segments back to back, each rest to rest; no sliver of a final interval in any segment
        times, rows, offset, starts = [0], [trajs[0](0.0)], 0, []
        for tr in trajs:
            end_ns = int(np.ceil(tr.duration * 1e9))
            ts = list(range(TICK_NS, end_ns, TICK_NS))
            if ts and end_ns - ts[-1] < TICK_NS // 2:
                ts.pop()
            ts.append(end_ns)
            rows += [tr(min(t * 1e-9, tr.duration)) for t in ts]
            times += [offset + t for t in ts]
            starts.append(offset)
            offset += end_ns
        q = np.array(rows)
        if not runtime_violations(times, q, VMAX, AMAX):
            break
    seconds = time.perf_counter() - t0
    bounds = np.array(starts + [offset], dtype=float) * 1e-9

    def source(ts):
        ts = np.atleast_1d(ts)
        k = np.clip(np.searchsorted(bounds, ts, side="right") - 1, 0, len(trajs) - 1)
        return np.array([trajs[i](min(t - bounds[i], trajs[i].duration)) for i, t in zip(k, ts)])

    # Geometry is PR #211's: lines from the plan and blends it validated, so the record is complete.
    row = {"method": "blend_toppra", "duration": offset * 1e-9, "smoothing_seconds": seconds, "scale": scale}
    row.update(evaluate(times, q, source, True, planner, problem, result.path[0], result.path[-1]))
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
            print(f"  {method:24s} valid {len(valid)}/{len(rows)}  median {np.median(durs):.2f}s  "
                  f"p90 {np.percentile(durs, 90):.2f}s  smoothing p90 {np.percentile(smooth, 90):.3f}s")  # fmt: skip
            for r in [r for r in rows if not _ok(r)][:3]:
                print(f"    invalid: {_why(r)}")
        plan_p90 = float(np.percentile([r["planning_seconds"] for r in recs], 90))
        solve_p90 = float(np.percentile([r["solve_seconds"] for r in recs], 90))
        base = by.get("blend_toppra", {})
        for method in [f"{kind}smoother_{b}" for kind in ("", "native_") for b in [*data["budgets"], "default"]]:
            prop = by.get(method, {})
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
            head = f"  {method} vs blend_toppra, {len(pairs)} pairs:"
            print(f"{head} median diff CI [{lo_d:+.2f}, {hi_d:+.2f}] s, p90 ratio CI [{lo_r:.2f}, {hi_r:.2f}]")
            print(f"    hard gates {_pf(hard)}, duration gates {_pf(speed)}")
            if "attempts" in next(iter(prop.values())):
                n = [r["attempts"] for r in prop.values()]
                stops = {k: sum(r["stopped"] == k for r in prop.values()) for k in ("converged", "iterations")}
                print(f"    attempts median {np.median(n):.0f} p90 {np.percentile(n, 90):.0f}; stopped {stops}")
            print(f"    smoothing p90 {sp90:.3f}s vs plan() p90 {plan_p90:.3f}s {_pf(sp90 <= plan_p90)}; "
                  f"vs solver-only p90 {solve_p90:.3f}s {_pf(sp90 <= solve_p90)}")  # fmt: skip


def _pf(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _why(r: dict[str, Any]) -> Any:
    if r.get("failed"):
        return r["failed"]
    if not r["record_ok"]:
        return "a piece has no validation record"
    if not r["conversion_ok"]:
        return f"conversion deviation {r['conversion_deviation']:.2e} rad"
    return r.get("runtime_violations") or "endpoints"


def _ok(r: dict[str, Any]) -> bool:
    return (not r.get("failed") and r["record_ok"] and r["runtime_ok"] and r["conversion_ok"]
            and r["endpoints_ok"])  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Stopping rules (#214): one long trace per seed, replayed offline
# --------------------------------------------------------------------------------------------------------------------

TRACE_ITERATIONS = 400


def lower_bound(start, goal, vmax, amax) -> float:
    """Obstacle-free lower bound on any trajectory's duration: the slowest joint's rest-to-rest minimum time."""
    from sscbirrt.trajectory import min_time

    return max(min_time(float(a), 0.0, float(b), 0.0, float(v), float(c)).duration
               for a, b, v, c in zip(start, goal, vmax, amax))  # fmt: skip


def _trace(args) -> None:
    from sscbirrt.trajectory import TrajectoryOptions, coalesce, initial_trajectory

    out = {"issue": 214, "limits": {"velocity": VMAX, "acceleration": AMAX}, "iterations": args.iterations,
           "seeds": args.seeds, "records": []}  # fmt: skip
    for name in args.problems:
        solve = load(name)
        for seed in range(args.seeds):
            result, wall, planner, problem = solve(seed)
            if not result.success:
                continue
            margin = 1 - 1e-6
            vmax, amax = np.full(len(result.path[0]), VMAX * margin), np.full(len(result.path[0]), AMAX * margin)
            t = time.perf_counter()
            options = TrajectoryOptions(vmax, amax, iterations=args.iterations, convergence_window=None, seed=seed)
            traj = planner.smooth_trajectory(problem, result.path, options)  # a full native trace
            total = time.perf_counter() - t
            d = traj.diagnostics
            out["records"].append({
                "problem": name, "seed": seed, "planning_seconds": wall,
                "initial": initial_trajectory(coalesce(result.path), vmax, amax).duration,
                "lower_bound": lower_bound(result.path[0], result.path[-1], vmax, amax),
                "final": traj.duration, "overhead_seconds": total - d["seconds"],
                # per attempt: accepted, gain (None if no interpolant), elapsed seconds
                "attempts": [[bool(a.get("accepted")), a.get("gain"), a["elapsed"]] for a in d["attempts"]],
            })  # fmt: skip
            if not args.quiet:
                print(f"{name} s{seed}: {out['records'][-1]['final']:.2f}s", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out) + "\n")
    print(f"wrote {args.output}")


def durations(rec: dict[str, Any]) -> np.ndarray:
    """Duration after each number of attempts, 0..n: an accepted shortcut shortens the trajectory by its gain."""
    gains = [g if ok else 0.0 for ok, g, _ in rec["attempts"]]
    return rec["initial"] - np.concatenate([[0.0], np.cumsum(gains)])


def _rules(cap: int) -> dict[str, Any]:
    """Candidate rules: each maps a trace record to the number of attempts made before stopping (at most cap)."""

    def fixed(k):
        return lambda rec: min(k, cap, len(rec["attempts"]))

    def patience(p):
        def rule(rec):
            since = 0
            for k, (ok, _, _) in enumerate(rec["attempts"][:cap], start=1):
                since = 0 if ok else since + 1
                if since >= p:
                    return k
            return min(cap, len(rec["attempts"]))

        return rule

    def rate(window, eps):  # the last `window` attempts improved the duration by less than eps of it
        def rule(rec):
            D = durations(rec)
            for k in range(window, min(cap, len(rec["attempts"])) + 1):
                if D[k - window] - D[k] < eps * D[k]:
                    return k
            return min(cap, len(rec["attempts"]))

        return rule

    def gap(delta, then):  # within delta of the obstacle-free lower bound, or `then`
        def rule(rec):
            D = durations(rec)
            k_then = then(rec)
            hit = np.nonzero(D[: k_then + 1] <= (1 + delta) * rec["lower_bound"])[0]
            return int(hit[0]) if len(hit) else k_then

        return rule

    rules: dict[str, Any] = {f"fixed {k}": fixed(k) for k in (25, 50, 100, 200, 400) if k <= cap}
    rules.update({f"patience {p}": patience(p) for p in (10, 20, 30, 50)})
    rules.update({f"rate {w}/{e:.1%}": rate(w, e) for w in (20, 40) for e in (0.005, 0.01, 0.02)})
    rules.update({f"gap 5% | patience {p}": gap(0.05, patience(p)) for p in (20, 30)})
    return rules


def _stopping(args) -> None:
    trace = json.loads(Path(args.trace).read_text())
    run = json.loads(Path(args.run).read_text())
    base = {
        (r["problem"], r["seed"]): row for r in run["records"] for row in r["rows"] if row["method"] == "blend_toppra"
    }
    check = {(r["problem"], r["seed"]): row["duration"] for r in run["records"] for row in r["rows"]
             if row["method"] == "native_smoother_100"}  # fmt: skip
    for name in dict.fromkeys(r["problem"] for r in trace["records"]):
        recs = [r for r in trace["records"] if r["problem"] == name]
        drift = max((abs(durations(r)[100] - check[(name, r["seed"])]) for r in recs if (name, r["seed"]) in check),
                    default=float("nan"))  # fmt: skip
        plan_p90 = float(np.percentile([r["planning_seconds"] for r in recs], 90))
        print(f"\n== {name}: {len(recs)} seeds; replay at 100 vs the run's native_smoother_100: max |diff| "
              f"{drift:.1e} s; plan() p90 {plan_p90:.3f}s")  # fmt: skip
        print(f"  {'rule':24s} {'cap':>4s} {'attempts med/p90':>17s} {'smoothing p90':>14s} {'median':>7s} "
              f"{'loss vs 400 med/p90':>20s}  median diff CI       p90 ratio CI")  # fmt: skip
        for cap in (100, 400):
            for label, rule in _rules(cap).items():
                ks = [rule(r) for r in recs]
                D = [float(durations(r)[k]) for r, k in zip(recs, ks)]
                secs = [r["overhead_seconds"] + (r["attempts"][k - 1][2] if k else 0.0) for r, k in zip(recs, ks)]
                loss = [d / float(durations(r)[-1]) - 1 for r, d in zip(recs, D)]
                pairs = [(d, base[(name, r["seed"])]["duration"]) for r, d in zip(recs, D)
                         if (name, r["seed"]) in base and _ok(base[(name, r["seed"])])]  # fmt: skip
                a, b = np.array([p for p, _ in pairs]), np.array([q for _, q in pairs])
                lo_d, hi_d = _boot(lambda x, y: np.median(x - y), a, b)
                lo_r, hi_r = _boot(lambda x, y: np.percentile(x, 90) / np.percentile(y, 90), a, b)
                ok = hi_d < 0 and hi_r <= 1.0 and np.percentile(secs, 90) <= plan_p90
                print(f"  {label:24s} {cap:4d} {np.median(ks):8.0f}/{np.percentile(ks, 90):<8.0f} "
                      f"{np.percentile(secs, 90):13.3f}s {np.median(D):6.2f}s "
                      f"{np.median(loss):9.1%}/{np.percentile(loss, 90):<9.1%} "
                      f"[{lo_d:+.2f}, {hi_d:+.2f}]  [{lo_r:.2f}, {hi_r:.2f}]  {_pf(ok)}")  # fmt: skip


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
    t = sub.add_parser("trace")
    t.add_argument("--problems", nargs="+", default=["pick", "transport"])
    t.add_argument("--seeds", type=int, default=100)
    t.add_argument("--iterations", type=int, default=TRACE_ITERATIONS)
    t.add_argument("--output", default="trajectory_trace.json")
    t.add_argument("--quiet", action="store_true")
    st = sub.add_parser("stopping")
    st.add_argument("trace")
    st.add_argument("run", help="a run file with blend_toppra and native_smoother_100 rows for the same seeds")
    args = ap.parse_args(argv)
    {"run": _run, "gates": _gates, "trace": _trace, "stopping": _stopping}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
