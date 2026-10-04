# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Planning-time benchmark on realistic problems, for comparing planner changes (#196, #186).

Runs each problem over many seeds on the native backend and records, per run, the outcome, the time, the work
(machine-independent counts), and the path. Time alone is not trusted: a loaded machine slows every run (one run
this week was 8x slow across the board), so conclusions rest on the work counts, and wall time is cross-checked by a
fixed calibration workload, independent of the planner, before and after each problem and, for an A/B comparison,
by interleaving blocks of seeds between the two installs so that both see the same machine state.

The primary time is ``solve_seconds``: root collection plus search plus smoothing. ``planning_time`` alone excludes
the initial root collection, so it would flatter a configuration that collects more roots up front.

    uv run python tools/planning_benchmark.py run --label main --output benchmarks/planning_main.json
    uv run python tools/planning_benchmark.py run --label new --output new.json \\
        --interleave /path/to/baseline/.venv/bin/python --other-label base --other-output base.json
    uv run python tools/planning_benchmark.py compare base.json new.json

Problems: the three demo planning calls (``pick``, ``transport``: the upright carry, ``door``: the chain-constrained
opening) and the UR5e cases of the reference artifact.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEMO_PROBLEMS = ("pick", "transport", "door")
ARTIFACT_PROBLEMS = (
    "ur5e_tsr_goal_union_with_path_tsr",
    "ur5e_mujoco_tsr_goal_among_obstacles",
    "ur5e_mujoco_held_object",
    "ur5e_mujoco_tsr_chain_crank",
)
PROBLEMS = DEMO_PROBLEMS + ARTIFACT_PROBLEMS
WORK_KEYS = ("iterations", "state_checks", "edge_checks", "set_samples", "tree_nodes", "roots", "search_roots")
BLOCK = 10  # seeds per interleaved block


def _artifact_tool():
    spec = importlib.util.spec_from_file_location("reference_artifact", ROOT / "tools" / "reference_artifact.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_artifact"] = module
    spec.loader.exec_module(module)
    return module


def load(name: str):
    """``(solve(seed) -> PlanResult, timeout_seconds, goal_labels or None)`` for one problem, built once."""
    if name in DEMO_PROBLEMS:
        from sscbirrt.demo.scenarios import all_scenarios
        from sscbirrt.mujoco import plan

        p = all_scenarios()[name].problem()
        labels = None
        if name == "pick":
            from sscbirrt.demo.scenarios.pick import owners

            labels = owners()

        def solve(seed: int):
            return plan(p.model, p.data, p.arm, config=p.config, seed=seed, **p.kwargs)

        return solve, float(p.config.timeout), labels
    if "MUJOCO_MENAGERIE_PATH" not in os.environ:
        try:
            import sscbirrt_assets

            os.environ["MUJOCO_MENAGERIE_PATH"] = str(sscbirrt_assets.menagerie_path())
        except ImportError:
            pass
    case = next(c for c in _artifact_tool().cases() if c["name"] == name)
    if case.get("skip"):
        raise RuntimeError(f"{name}: {case['skip']}")
    planner, problem = case["planner"], case["problem"]
    return (lambda seed: planner.solve(problem, seed=seed)), float(case["config"].timeout), None


def _roots(tree) -> int:
    return 0 if tree is None else sum(1 for n in tree.nodes if n.parent is None)


def measure(result, wall: float) -> dict[str, Any]:
    s = result.stats or {}
    path = result.path
    length = None if path is None else float(sum(np.linalg.norm(np.subtract(b, a)) for a, b in zip(path, path[1:])))
    reason = result.failure_reason or ""
    return {
        "success": bool(result.success),
        "failure": None if result.success else reason.split(" ")[0],
        "backend": result.backend,
        "wall_seconds": wall,
        "planning_seconds": float(result.planning_time),
        "roots_seconds": float(s.get("seconds_roots", 0.0)),
        "solve_seconds": float(s.get("seconds_roots", 0.0)) + float(result.planning_time),
        "smoothing_seconds": float(s.get("seconds_smoothing", 0.0)),
        "iterations": int(result.iterations),
        "state_checks": int(s.get("state_checks", 0)),
        "edge_checks": int(s.get("edge_checks", 0)),
        "set_samples": int(s.get("set_samples", 0)),
        "tree_nodes": int(sum(result.tree_sizes)) if result.tree_sizes else 0,
        "roots": _roots(result.tree_start) + _roots(result.tree_goal),
        "search_roots": int(s.get("search_roots", 0)),
        "path_length": length,
        "waypoints": None if path is None else len(path),
        "start_index": result.start_index,
        "goal_index": result.goal_index,
    }


def calibrate(repeats: int = 5) -> float:
    """Median seconds of a fixed workload that no planner version touches (dense linear algebra), so that A and B
    calibrations measure the machine, not the code under test."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((300, 300))
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(20):
            np.linalg.solve(a + 300 * np.eye(300), a @ a)
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def run_block(name: str, seeds: range) -> list[dict[str, Any]]:
    solve, _, _ = load(name)
    runs = []
    for seed in seeds:
        t = time.perf_counter()
        result = solve(seed)
        runs.append({"seed": seed, **measure(result, time.perf_counter() - t)})
    return runs


def describe() -> dict[str, Any]:
    import sscbirrt

    versions = {}
    for pkg in ("sscbirrt", "sstsr", "ssik", "mujoco", "numpy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    sha = subprocess.run(
        ["git", "-C", os.path.dirname(sscbirrt.__file__), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "versions": versions,
        "git_sha": sha or None,
        "sscbirrt_path": os.path.dirname(sscbirrt.__file__),
        "machine": {"platform": platform.platform(), "cpus": os.cpu_count(), "python": platform.python_version()},
    }


def _worker(args) -> None:
    """Internal: one block or a calibration, as JSON on stdout (used by --interleave)."""
    if args.what == "calibrate":
        out: Any = calibrate()
    elif args.what == "describe":
        out = describe()
    else:
        out = run_block(args.what, range(args.start, args.stop))
    sys.stdout.write("\n@@RESULT@@" + json.dumps(out))


def _call(python: str, *argv: str) -> Any:
    proc = subprocess.run([python, str(Path(__file__).resolve()), "worker", *argv], capture_output=True, text=True)
    if proc.returncode != 0 or "@@RESULT@@" not in proc.stdout:
        raise RuntimeError(f"worker {python} {argv} failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.rsplit("@@RESULT@@", 1)[1])


def _ci_median(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    boots = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(runs: list[dict[str, Any]], timeout: float, labels: list[str] | None) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    ok = [r for r in runs if r["success"]]
    # Failures count at the timeout, so a slower median is never hidden by dropping the runs that gave up.
    t = np.array([r["solve_seconds"] if r["success"] else timeout for r in runs])
    summary = {
        "runs": len(runs),
        "success_rate": len(ok) / len(runs),
        "failures": dict(Counter(r["failure"] for r in runs if not r["success"])),
        "solve_seconds": {
            "median": float(np.median(t)),
            "median_ci95": _ci_median(t, rng),
            "p90": float(np.percentile(t, 90)),
            "max": float(t.max()),
        },
        "work_median": {k: float(np.median([r[k] for r in runs])) for k in WORK_KEYS},
        "path_length_median": float(np.median([r["path_length"] for r in ok])) if ok else None,
        "backends": dict(Counter(r["backend"] for r in runs)),
    }
    goals = Counter(r["goal_index"] for r in ok)
    summary["goals"] = {(labels[k] if labels else str(k)): n for k, n in sorted(goals.items(), key=lambda kv: kv[0])}
    if labels:
        summary["goals_by_label"] = dict(Counter(labels[r["goal_index"]] for r in ok))
    return summary


def _run(args) -> None:
    problems = args.problems or list(PROBLEMS)
    other = args.interleave
    me = sys.executable
    out_a = {"label": args.label, **describe(), "seeds": args.seeds, "problems": {}}
    out_b = None
    if other:
        out_b = {"label": args.other_label, **_call(other, "describe"), "seeds": args.seeds, "problems": {}}
    for name in problems:
        _, timeout, labels = load(name)
        runs_a: list[dict[str, Any]] = []
        runs_b: list[dict[str, Any]] = []
        cal_a = [_call(me, "calibrate") if other else calibrate()]
        cal_b = [_call(other, "calibrate")] if other else []
        for lo in range(0, args.seeds, BLOCK):
            block = (str(lo), str(min(lo + BLOCK, args.seeds)))
            if other:
                # Alternate which install goes first, so neither always runs on the warmer machine.
                pair = [(other, runs_b), (me, runs_a)] if (lo // BLOCK) % 2 else [(me, runs_a), (other, runs_b)]
                for python, runs in pair:
                    runs += _call(python, name, *block)
            else:
                runs_a += run_block(name, range(int(block[0]), int(block[1])))
        cal_a.append(_call(me, "calibrate") if other else calibrate())
        if other:
            cal_b.append(_call(other, "calibrate"))
        out_a["problems"][name] = {"timeout": timeout, "calibration_seconds": cal_a, "runs": runs_a,
                                   "summary": summarize(runs_a, timeout, labels)}  # fmt: skip
        if other:
            out_b["problems"][name] = {"timeout": timeout, "calibration_seconds": cal_b, "runs": runs_b,
                                       "summary": summarize(runs_b, timeout, labels)}  # fmt: skip
        if not args.quiet:
            line = _line(name, out_a["problems"][name]["summary"])
            if other:
                line += "  |  " + _line("", out_b["problems"][name]["summary"])
            print(line, flush=True)
    _write(out_a, args.output)
    if other:
        _write(out_b, args.other_output)


def _line(name: str, s: dict[str, Any]) -> str:
    t = s["solve_seconds"]
    ok = f"ok {s['success_rate']:5.0%}"
    return f"{name:38s} {ok}  median {t['median']:7.3f}s  p90 {t['p90']:7.3f}s  max {t['max']:7.2f}s"


def _write(out: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")


def _compare(args) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    print(f"A = {a['label']} ({a['versions']['sscbirrt']}, {str(a['git_sha'])[:8]})   "
          f"B = {b['label']} ({b['versions']['sscbirrt']}, {str(b['git_sha'])[:8]})")  # fmt: skip
    rng = np.random.default_rng(0)
    for name in a["problems"]:
        if name not in b["problems"]:
            continue
        pa, pb = a["problems"][name], b["problems"][name]
        sa, sb = pa["summary"], pb["summary"]
        ta = np.array([r["solve_seconds"] if r["success"] else pa["timeout"] for r in pa["runs"]])
        tb = np.array([r["solve_seconds"] if r["success"] else pb["timeout"] for r in pb["runs"]])
        ratios = [
            np.median(rng.choice(tb, len(tb))) / max(np.median(rng.choice(ta, len(ta))), 1e-12) for _ in range(2000)
        ]
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        print(f"\n{name}  (calibration A {np.median(pa['calibration_seconds']) * 1e3:.1f} ms, "
              f"B {np.median(pb['calibration_seconds']) * 1e3:.1f} ms)")  # fmt: skip
        rows = [
            ("success", f"{sa['success_rate']:.0%}", f"{sb['success_rate']:.0%}", ""),
            ("solve median s", f"{sa['solve_seconds']['median']:.3f}", f"{sb['solve_seconds']['median']:.3f}",
             f"x{np.median(tb) / max(np.median(ta), 1e-12):.2f} [{lo:.2f}, {hi:.2f}]"),
            ("solve p90 s", f"{sa['solve_seconds']['p90']:.3f}", f"{sb['solve_seconds']['p90']:.3f}", ""),
            ("solve max s", f"{sa['solve_seconds']['max']:.2f}", f"{sb['solve_seconds']['max']:.2f}", ""),
        ]  # fmt: skip
        for k in WORK_KEYS:
            va, vb = sa["work_median"][k], sb["work_median"][k]
            rows.append((f"{k} median", f"{va:g}", f"{vb:g}", f"x{vb / va:.2f}" if va else ""))
        pla, plb = sa["path_length_median"], sb["path_length_median"]
        rows.append(("path length median", f"{pla:.2f}" if pla else "-", f"{plb:.2f}" if plb else "-", ""))
        if "goals_by_label" in sa or "goals_by_label" in sb:
            rows.append(("goals", str(sa.get("goals_by_label")), str(sb.get("goals_by_label")), ""))
        for row in rows:
            print(f"  {row[0]:20s} {row[1]:>14s} {row[2]:>14s}  {row[3]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="benchmark this install (and optionally another, interleaved)")
    r.add_argument("--problems", nargs="*", choices=PROBLEMS)
    r.add_argument("--seeds", type=int, default=100)
    r.add_argument("--label", default="this")
    r.add_argument("--output", default="planning_benchmark.json")
    r.add_argument("--interleave", metavar="PYTHON", help="another install's python, run in alternating blocks")
    r.add_argument("--other-label", default="other")
    r.add_argument("--other-output", default="planning_benchmark_other.json")
    r.add_argument("--quiet", action="store_true")
    c = sub.add_parser("compare", help="print B against A")
    c.add_argument("a")
    c.add_argument("b")
    w = sub.add_parser("worker", help=argparse.SUPPRESS)
    w.add_argument("what")
    w.add_argument("start", type=int, nargs="?", default=0)
    w.add_argument("stop", type=int, nargs="?", default=0)
    args = ap.parse_args(argv)
    {"run": _run, "compare": _compare, "worker": _worker}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
