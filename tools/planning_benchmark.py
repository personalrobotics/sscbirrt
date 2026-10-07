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
    uv run python tools/planning_benchmark.py run --problems pick --set step_size=0.4 num_tree_roots=5
    uv run python tools/planning_benchmark.py sweep spec.json --problems pick --seeds 0-19 --output sweep.jsonl
    uv run python tools/planning_benchmark.py analyze sweep.jsonl

``sweep`` runs a parameter study (#186): a full factorial (``"factors"``) or a list of settings (``"cells"``) on
top of ``"fixed"`` overrides, with ``"baseline"`` naming today's defaults. Every (setting, seed) pair runs in a
shuffled order on one install, so machine drift cannot line up with a factor, and each run is appended to a
JSON-lines file as it finishes, so an interrupted sweep resumes. ``analyze`` applies the decision rule fixed before
the data (#186): among settings that always succeed, the lowest median solve time whose p90 is not worse than the
baseline's and whose median path is at most 10% longer; ties within the CI go to the setting nearest the baseline.

Problems: the three demo planning calls (``pick``, ``transport``: the upright carry, ``door``: the chain-constrained
opening) and the UR5e cases of the reference artifact.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import importlib.util
import itertools
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
    """``(solve(seed, overrides=None) -> PlanResult, timeout_seconds, goal_labels or None)`` for one problem, built
    once. ``overrides`` replaces CBiRRTConfig fields for that solve only (``dataclasses.replace``)."""
    if name in DEMO_PROBLEMS:
        from sscbirrt.demo.scenarios import all_scenarios
        from sscbirrt.mujoco import plan

        p = all_scenarios()[name].problem()
        labels = None
        if name == "pick":
            from sscbirrt.demo.scenarios.pick import owners

            labels = owners()

        def solve(seed: int, overrides: dict[str, Any] | None = None):
            config = dataclasses.replace(p.config, **(overrides or {}))
            return plan(p.model, p.data, p.arm, config=config, seed=seed, **p.kwargs)

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
    base = planner.config

    def solve_case(seed: int, overrides: dict[str, Any] | None = None):
        planner.config = dataclasses.replace(base, **(overrides or {}))
        try:
            return planner.solve(problem, seed=seed)
        finally:
            planner.config = base

    return solve_case, float(case["config"].timeout), None


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


def run_block(name: str, seeds: range, overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    solve, _, _ = load(name)
    runs = []
    for seed in seeds:
        t = time.perf_counter()
        result = solve(seed, overrides)
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
        out = run_block(args.what, range(args.start, args.stop), json.loads(args.overrides))
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
    overrides = parse_overrides(args.set)
    out_a["overrides"] = overrides
    if out_b is not None:
        out_b["overrides"] = overrides
    for name in problems:
        _, timeout, labels = load(name)
        timeout = float(overrides.get("timeout", timeout))
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
                    runs += _call(python, name, *block, "--overrides", json.dumps(overrides))
            else:
                runs_a += run_block(name, range(int(block[0]), int(block[1])), overrides)
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


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """``key=value`` pairs to CBiRRTConfig overrides; values parse as JSON (``0.4``, ``5``, ``null``)."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, _, value = item.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _seed_range(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi) + 1)) if hi else list(range(int(lo)))


def _cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The settings a spec names: the factorial of ``factors`` or the listed ``cells``, each on top of ``fixed``,
    plus the ``baseline`` if it is not among them."""
    if "factors" in spec:
        names = list(spec["factors"])
        cells = [dict(zip(names, levels)) for levels in itertools.product(*(spec["factors"][n] for n in names))]
    else:
        cells = [dict(c) for c in spec["cells"]]
    if spec.get("baseline") is not None and spec["baseline"] not in cells:
        cells.append(dict(spec["baseline"]))
    return cells


def _overrides(cell: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    out = {**fixed}
    for key, value in cell.items():
        if key == "sample_probability":  # both coins move together
            out["start_sample_probability"] = out["goal_sample_probability"] = value
        else:
            out[key] = value
    return out


def _key(cell: dict[str, Any]) -> str:
    return json.dumps(cell, sort_keys=True)


def _sweep(args) -> None:
    spec = json.loads(Path(args.spec).read_text())
    cells, fixed, seeds = _cells(spec), spec.get("fixed", {}), _seed_range(args.seeds)
    out = Path(args.output)
    done: set[tuple[str, str, int]] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if "cell" in row:
                done.add((row["problem"], _key(row["cell"]), row["seed"]))
    else:
        out.write_text(json.dumps({"header": {**describe(), "spec": spec, "seeds": seeds}}) + "\n")
    rng = np.random.default_rng(args.order_seed)
    for name in args.problems:
        solve, timeout, _ = load(name)
        todo = [(c, s) for c in cells for s in seeds if (name, _key(c), s) not in done]
        order = rng.permutation(len(todo))
        with out.open("a") as f:
            f.write(json.dumps({"problem": name, "calibration_seconds": calibrate(), "remaining": len(todo)}) + "\n")
            for i, k in enumerate(order, 1):
                cell, seed = todo[k]
                ov = _overrides(cell, fixed)
                t = time.perf_counter()
                result = solve(seed, ov)
                wall = time.perf_counter() - t
                row = {"problem": name, "cell": cell, "seed": seed, "timeout": float(ov.get("timeout", timeout))}
                row.update(measure(result, wall))
                f.write(json.dumps(row) + "\n")
                f.flush()
                if i % 50 == 0:
                    f.write(json.dumps({"problem": name, "calibration_seconds": calibrate(), "after": i}) + "\n")
                    if not args.quiet:
                        print(f"{name}: {i}/{len(todo)}", flush=True)
    print(f"wrote {out}")


def _paired_ratio_ci(a: np.ndarray, b: np.ndarray, stat, rng: np.random.Generator, n: int = 2000) -> tuple:
    """CI of stat(b)/stat(a), resampling seeds jointly (a and b share seeds)."""
    idx = rng.integers(0, len(a), size=(n, len(a)))
    ratios = np.array([stat(b[i]) / max(stat(a[i]), 1e-12) for i in idx])
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def analyze(rows: list[dict[str, Any]], spec: dict[str, Any], problem: str) -> dict[str, Any]:
    """Per-setting summaries and the decision rule of #186, fixed before the data."""
    rng = np.random.default_rng(0)
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("problem") == problem and "cell" in r:
            by.setdefault(_key(r["cell"]), []).append(r)
    cells = {}
    for key, runs in by.items():
        runs.sort(key=lambda r: r["seed"])
        t = np.array([r["solve_seconds"] if r["success"] else r["timeout"] for r in runs])
        ok = [r["path_length"] for r in runs if r["success"]]
        cells[key] = {
            "cell": runs[0]["cell"],
            "seeds": [r["seed"] for r in runs],
            "t": t,
            "path": np.array([r["path_length"] if r["success"] else np.nan for r in runs]),
            "success_rate": float(np.mean([r["success"] for r in runs])),
            "median": float(np.median(t)),
            "p90": float(np.percentile(t, 90)),
            "path_median": float(np.median(ok)) if ok else float("nan"),
            "work": {k: float(np.median([r[k] for r in runs])) for k in WORK_KEYS},
        }
    base_key = _key(spec["baseline"]) if spec.get("baseline") is not None else None
    base = cells.get(base_key)
    levels = {}
    for c in cells.values():
        for f, v in c["cell"].items():
            levels.setdefault(f, set()).add(v)
    rank = {f: {v: i for i, v in enumerate(sorted(vs, key=lambda v: (v is None, v)))} for f, vs in levels.items()}

    def distance(cell):
        if base is None:
            return (0, 0)
        diff = [f for f in cell if cell[f] != base["cell"].get(f)]
        return (len(diff), sum(abs(rank[f][cell[f]] - rank[f][base["cell"][f]]) for f in diff))

    eligible = []
    for key, c in cells.items():
        c["eligible"], c["why_not"] = True, []
        if c["success_rate"] < 1.0:
            c["eligible"] = False
            c["why_not"].append("not 100% success")
        if base is not None and key != base_key and c["seeds"] == base["seeds"]:
            lo, _ = _paired_ratio_ci(base["t"], c["t"], lambda x: np.percentile(x, 90), rng)
            c["p90_ratio_ci_low"] = lo
            if lo > 1.0:
                c["eligible"] = False
                c["why_not"].append("p90 worse than baseline")
            if c["path_median"] > 1.10 * base["path_median"]:
                c["eligible"] = False
                c["why_not"].append("path >10% longer")
        if c["eligible"]:
            eligible.append(c)
    eligible.sort(key=lambda c: c["median"])
    winner = None
    if eligible:
        best = eligible[0]
        tied = [best]
        for c in eligible[1:]:
            if c["seeds"] != best["seeds"]:
                continue
            lo, hi = _paired_ratio_ci(best["t"], c["t"], np.median, rng)
            if lo <= 1.0 <= hi:  # not distinguishable from the fastest
                tied.append(c)
        winner = min(tied, key=lambda c: (distance(c["cell"]), c["median"]))
    effects = {}
    for f, vs in levels.items():
        effects[f] = {
            str(v): {
                "median_of_medians": float(np.median([c["median"] for c in cells.values() if c["cell"].get(f) == v])),
                "mean_success": float(np.mean([c["success_rate"] for c in cells.values() if c["cell"].get(f) == v])),
            }
            for v in sorted(vs, key=lambda v: (v is None, v))
        }
    return {"cells": cells, "baseline": base, "eligible": eligible, "winner": winner, "effects": effects}


def _fmt_cell(cell: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(cell.items()))


def _analyze(args) -> None:
    lines = [json.loads(line) for line in Path(args.sweep).read_text().splitlines()]
    spec = lines[0]["header"]["spec"]
    rows = lines[1:]
    cal = [r["calibration_seconds"] * 1e3 for r in rows if "calibration_seconds" in r]
    print(f"calibration: {min(cal):.1f}-{max(cal):.1f} ms over {len(cal)} readings")
    for problem in dict.fromkeys(r["problem"] for r in rows if "cell" in r):
        res = analyze(rows, spec, problem)
        base = res["baseline"]
        print(f"\n== {problem}: {len(res['cells'])} settings, {len(res['eligible'])} eligible")
        if base is not None:
            print(f"baseline  {_fmt_cell(base['cell'])}: success {base['success_rate']:.0%}, median "
                  f"{base['median']:.3f} s, p90 {base['p90']:.3f} s, path {base['path_median']:.2f}")  # fmt: skip
        print("top eligible by median solve time:")
        for c in res["eligible"][: args.top]:
            print(f"  {c['median']:.3f} s  p90 {c['p90']:.3f}  path {c['path_median']:.2f}  "
                  f"iters {c['work']['iterations']:.0f}  {_fmt_cell(c['cell'])}")  # fmt: skip
        w = res["winner"]
        print("decision: " + (f"{_fmt_cell(w['cell'])} (median {w['median']:.3f} s)" if w else "keep the baseline"))
        print("main effects (median of setting medians, mean success):")
        for f, by_level in res["effects"].items():
            parts = [f"{lv}: {e['median_of_medians']:.3f}s/{e['mean_success']:.0%}" for lv, e in by_level.items()]
            print(f"  {f:28s} " + "  ".join(parts))


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
    r.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="CBiRRTConfig overrides for every run")
    s = sub.add_parser("sweep", help="a parameter study: every (setting, seed) pair, shuffled, resumable")
    s.add_argument("spec", help="JSON: fixed, factors or cells, baseline")
    s.add_argument("--problems", nargs="+", choices=PROBLEMS, default=["pick"])
    s.add_argument("--seeds", default="0-19", help="N (0..N-1) or LO-HI inclusive")
    s.add_argument("--output", default="sweep.jsonl")
    s.add_argument("--order-seed", type=int, default=1234)
    s.add_argument("--quiet", action="store_true")
    a = sub.add_parser("analyze", help="summaries and the decision rule for a sweep")
    a.add_argument("sweep")
    a.add_argument("--top", type=int, default=10)
    c = sub.add_parser("compare", help="print B against A")
    c.add_argument("a")
    c.add_argument("b")
    w = sub.add_parser("worker", help=argparse.SUPPRESS)
    w.add_argument("what")
    w.add_argument("start", type=int, nargs="?", default=0)
    w.add_argument("stop", type=int, nargs="?", default=0)
    w.add_argument("--overrides", default="{}")
    args = ap.parse_args(argv)
    {"run": _run, "sweep": _sweep, "analyze": _analyze, "compare": _compare, "worker": _worker}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
