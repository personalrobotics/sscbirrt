# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The reduced Geodude edge of personalrobotics/ssrobot#132, for #230's evidence.

ssrobot's reference-robot scenario (ssrobot 6f660eb, CI run 38499490297) planned Geodude's right arm back to
``ready`` with sscbirrt 3.7.0. Its timed trajectory crosses the left gripper for a few milliseconds inside a
shortcut, and ssrobot's runtime check caught it. This tool reruns that scenario inside ssrobot, captures the
planning scene, and records what sscbirrt's validators say about it. It needs ssrobot at that commit with the pinned
``geodude_assets`` installed, so it does not run in sscbirrt's CI.

    # in a venv with ssrobot (uv sync --locked) and sscbirrt 3.7.0:
    python tools/geodude_132_evidence.py capture SSROBOT_DIR OUT_DIR CI_RESULT_JSON
    # in a venv with ssrobot and this sscbirrt:
    python tools/geodude_132_evidence.py evaluate SSROBOT_DIR OUT_DIR

``capture`` writes ``capture.json``: the configuration plan's snapshot and result, whether the result equals the CI
artifact byte for byte, a dense recheck of its trajectory, the 3.7 shortcut rule on every piece, and whether the 3.7
smoother rerun on the path reproduces the trajectory. ``evaluate`` rebuilds that planning scene from the snapshot
(same fingerprint) and writes ``evidence.json``: this sscbirrt's validator on the recorded path and trajectory
(Python and native), then the same scenario planned with this sscbirrt, its planning-time reports against the public
calls, and its dense recheck.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tomllib

DENSE = 20001  # states of the dense diagnostic recheck over the whole trajectory


def _scenario(ssrobot_dir: Path, out: Path) -> tuple[dict, Any, Any, Any, Any]:
    """Run ssrobot's scenario for Geodude, capturing the configuration plan's snapshot, result and planning scene."""
    sys.path.insert(0, str(ssrobot_dir / "scripts"))
    import check_reference_robots as cr
    import ssrobot.planning as sp
    from ssrobot import RobotContext

    captured: dict[str, Any] = {}
    prepare = sp.prepare
    planning_scene = RobotContext.planning_scene

    def scene_spy(self, snapshot, group):
        scene = planning_scene(self, snapshot, group)
        captured.setdefault("scenes", []).append((snapshot, scene))
        return scene

    def prepare_spy(result, scene, **kw):
        if result.trajectory is not None and hasattr(result.request.goal, "configurations"):
            captured["result"], captured["scene"] = result, scene
            captured["snapshot"] = next(s for s, sc in captured["scenes"] if sc is scene)
        return prepare(result, scene, **kw)

    sp.prepare = prepare_spy
    RobotContext.planning_scene = scene_spy
    pins = tomllib.loads((ssrobot_dir / "references/robots.toml").read_text())
    spec = next(r for r in pins["robots"] if r["name"] == "geodude")["scenario"]
    spec = {**spec, "scene": str(ssrobot_dir / spec["scene"])}
    from ssrobot import load_installed_package

    package = load_installed_package("geodude_assets")
    record = cr.scenario(package, spec, None, out)
    return record, captured["result"], captured["scene"], captured["snapshot"], package


def _trajectory(record_traj, path):
    from sscbirrt.trajectory import Piece, Ramp, Trajectory

    pieces = [
        Piece(
            tuple(
                Ramp(r.position, r.velocity, tuple((i.duration, i.acceleration) for i in r.intervals)) for r in p.ramps
            ),
            p.validation.value,
        )
        for p in record_traj.pieces
    ]
    traj = Trajectory(
        pieces, np.array(record_traj.limits.effective_velocity), np.array(record_traj.limits.effective_acceleration)
    )
    traj.path = [np.array(q) for q in path]
    return traj


def _dense(traj, checker, scene) -> dict:
    ts = np.linspace(0.0, traj.duration, DENSE)
    qs, _, _ = traj.evaluate(ts)
    bad = [i for i, q in enumerate(qs) if not checker.is_valid(q)]
    agree = all(checker.is_valid(q) == scene.is_valid(tuple(q)) for q in qs[:: max(1, DENSE // 2000)])
    out: dict[str, Any] = {"states": DENSE, "invalid": len(bad), "scene_agrees": agree}
    if bad:
        breaks = traj.breaks
        out["window_s"] = [float(ts[bad[0]]), float(ts[bad[-1]])]
        out["pieces"] = sorted({int(np.searchsorted(breaks, ts[i], side="right") - 1) for i in bad})
        deepest = min((c for i in bad for c in checker.invalid_contacts(qs[i])), key=lambda c: c["dist"])
        out["deepest_contact"] = {k: deepest[k] for k in ("body1", "body2", "dist")}
    return out


def _space(package, joints):
    from sscbirrt import JointSpace

    lo = [package.description.joint(j).limits.lower for j in joints]
    hi = [package.description.joint(j).limits.upper for j in joints]
    return JointSpace(np.array(lo), np.array(hi))


def _smooth_again(space, checker, path, request):
    """The installed sscbirrt's smoother on ``path``, exactly as ssrobot's adapter calls it."""
    from sscbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem, TrajectoryOptions

    class Model:
        dof = space.dof
        joint_limits = (space.lower, space.upper)

        def forward_kinematics(self, q):
            return np.eye(4)

    planner = CBiRRT(Model(), collision_checker=checker, config=CBiRRTConfig(edge_resolution=0.05), backend="native")
    problem = PlanningProblem(
        space=planner.space, start=FiniteSet([path[0]]), goal=FiniteSet([path[-1]]), validator=checker
    )
    limits = request.limits
    options = TrajectoryOptions(
        np.array(limits.effective_velocity), np.array(limits.effective_acceleration), seed=request.seed
    )
    return planner, problem, planner.smooth_trajectory(problem, path, options)


def _same_law(a, b) -> bool:
    return len(a.pieces) == len(b.pieces) and all(
        len(p.ramps) == len(q.ramps)
        and all(r.x0 == s.x0 and r.v0 == s.v0 and r.pieces == s.pieces for r, s in zip(p.ramps, q.ramps))
        for p, q in zip(a.pieces, b.pieces)
    )


def capture(ssrobot_dir: Path, out: Path, ci_result: Path) -> None:
    from ssrobot import dumps

    from sscbirrt.motion import DiscreteMotionValidator
    from sscbirrt.trajectory import piece_samples, validate_samples

    record, result, scene, snapshot, package = _scenario(ssrobot_dir, out / "scenario-3.7")
    text = dumps(result) + "\n"
    checker = scene.native()
    traj = _trajectory(result.trajectory, result.path.waypoints)
    space = _space(package, result.request.joints)

    def admissible(q):
        ok = bool(space.contains(q) and checker.is_valid(q))
        return ok, None if ok else "invalid"

    motion = DiscreteMotionValidator(space, lambda q: admissible(q)[0], 0.05)
    old_rule = [validate_samples(piece_samples(p, 0.05, traj.velocity_limits), motion, admissible) for p in traj.pieces]
    _, _, again = _smooth_again(space, checker, traj.path, result.request)
    data = {
        "sscbirrt": md.version("sscbirrt"),
        "ssrobot_prepared": record["plans"]["configuration"].get("prepared"),
        "result_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "result_equals_ci_artifact": text == ci_result.read_text(),
        "snapshot": json.loads(dumps(snapshot)),
        "result": json.loads(text),
        "dense": _dense(traj, checker, scene),
        "rule_3_7_per_piece": [None if why is None else why for why in old_rule],
        "smoother_rerun_reproduces_trajectory": _same_law(again, traj),
    }
    (out / "capture.json").write_text(json.dumps(data, indent=1) + "\n")


def _report(r) -> dict:
    d = r.to_dict()
    d.pop("provenance", None)
    return d


def evaluate(ssrobot_dir: Path, out: Path) -> None:
    from ssrobot import PlanResult, RobotContext, SceneSnapshot, loads
    from ssrobot.mujoco import MujocoRuntime

    from sscbirrt.validation import ValidationPolicy, Validator

    cap = json.loads((out / "capture.json").read_text())
    snapshot = loads(json.dumps(cap["snapshot"]), SceneSnapshot)
    recorded = loads(json.dumps(cap["result"]), PlanResult)
    record, result, scene, _, package = _scenario(ssrobot_dir, out / "scenario-main")

    pins = tomllib.loads((ssrobot_dir / "references/robots.toml").read_text())
    spec = next(r for r in pins["robots"] if r["name"] == "geodude")["scenario"]
    runtime = MujocoRuntime(package, keyframe=None, scene=str(ssrobot_dir / spec["scene"]))
    with RobotContext(package.description, runtime) as ctx:
        old_scene = ctx.planning_scene(snapshot, recorded.request.group)
        joints = recorded.request.joints
        space = _space(package, joints)
        old_checker = old_scene.native()
        old_traj = _trajectory(recorded.trajectory, recorded.path.waypoints)
        policy = ValidationPolicy(0.05)
        on_recorded = {}
        for backend in ("python", "native"):
            v = Validator(space, old_checker, policy=policy, backend=backend)
            on_recorded[backend] = {
                "path": _report(v.validate_path(old_traj.path)),
                "trajectory": _report(v.validate_trajectory(old_traj)),
            }
        dense_recorded = _dense(old_traj, old_checker, old_scene)

    checker = scene.native()
    space = _space(package, result.request.joints)
    path = [np.array(q) for q in result.path.waypoints]
    traj = _trajectory(result.trajectory, path)
    _, _, again = _smooth_again(space, checker, path, result.request)
    public = Validator(space, checker, policy=ValidationPolicy(0.05), backend="native")
    planned = {
        "ssrobot_prepared": record["plans"]["configuration"].get("prepared"),
        "path_waypoints": len(path),
        "trajectory_pieces": len(traj.pieces),
        "smoother_rerun_reproduces_trajectory": _same_law(again, traj),
        "planning_trajectory_report": _report(again.validation),
        "public_trajectory_report": _report(public.validate_trajectory(traj)),
        "public_path_report": _report(public.validate_path(path)),
        "dense": _dense(traj, checker, scene),
    }
    planned["planning_equals_public"] = planned["planning_trajectory_report"] == planned["public_trajectory_report"]
    evidence = {
        "issue": "https://github.com/personalrobotics/sscbirrt/issues/230",
        "case": "personalrobotics/ssrobot#132: Geodude's right arm back to ready, ssrobot 6f660eb, CI run 38499490297",
        "versions": {
            "captured_with": cap["sscbirrt"],
            "evaluated_with": md.version("sscbirrt"),
            "ssik": md.version("ssik"),
            "mujoco": md.version("mujoco"),
        },
        "recorded_plan": {
            "result_sha256": cap["result_sha256"],
            "result_equals_ci_artifact": cap["result_equals_ci_artifact"],
            "snapshot_fingerprint_matches": snapshot.fingerprint() == recorded.request.snapshot,
            "ssrobot_prepared": cap["ssrobot_prepared"],
            "dense_recheck_3_7": cap["dense"],
            "dense_recheck_rebuilt_scene": dense_recorded,
            "rule_3_7_per_piece": cap["rule_3_7_per_piece"],
            "smoother_3_7_rerun_reproduces_trajectory": cap["smoother_rerun_reproduces_trajectory"],
            "this_validator": on_recorded,
            "python_equals_native": on_recorded["python"]["trajectory"] | {"backend": None}
            == on_recorded["native"]["trajectory"] | {"backend": None},
        },
        "planned_with_this_sscbirrt": planned,
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
    if argv[:1] == ["capture"] and len(argv) == 4:
        capture(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    elif argv[:1] == ["evaluate"] and len(argv) == 3:
        evaluate(Path(argv[1]), Path(argv[2]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
