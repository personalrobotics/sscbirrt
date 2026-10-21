# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Evidence for #230: the public validator and the planner give one answer, and it is inspectable.

On a small native MuJoCo world (a two-link arm and a wall), for both backends, it records:

- an accepted path and a rejected one, with the rejection's segment, fraction, failure and contacts;
- an exact reparameterization of a path (its trajectory along the path) and an interpolation that changes the
  trace (a time-optimal shortcut through the wall), each validated as what it is;
- the reports obtained during planning (``PlanResult.validation``, ``Trajectory.validation``) next to the public
  calls on the same path and trajectory (``sscbirrt.mujoco.validator``);
- the same queries repeated and in reverse order;
- the inputs: the scene's provenance (model and snapshot hashes), every path, the trajectory's pieces, and the policy.

For #247 it records, on a second world with one native scene, snapshots that differ only in an attachment's allowed
bodies: each snapshot's hash and the scene's provenance, and for both backends the state's verdict with its contacts
in MuJoCo's order and their kinds. A held object's contact with a fixture or a stationary robot body is ignored only
when that exact body is allowed; the arm's contacts and two held objects' contacts stay checked; a fixed ``<exclude>``
pair stays excluded.

``tests/reference/geodude_132_evidence.json`` is the reduced Geodude edge of ssrobot#132, made by
``tools/geodude_132_evidence.py`` inside ssrobot.

    uv run python tools/validation_artifact.py            # write tests/reference/validation_evidence.json
    uv run python tools/validation_artifact.py --check    # compare a fresh run's verdicts with it
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT = Path(__file__).resolve().parent.parent / "tests" / "reference" / "validation_evidence.json"

WORLD = """
<mujoco><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="3 3 0.1"/>
  <geom name="wall" type="box" pos="0.47 0 0.3" size="0.05 0.5 0.3"/>
  <body name="arm/base" pos="0 0 0.1"><joint name="j0" type="hinge" axis="0 0 1" range="-3 3"/>
    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
    <body name="arm/link1" pos="0 0 0.1"><joint name="j1" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0.4 0 0"/>
      <body name="arm/hand" pos="0.4 0 0">
        <geom type="box" size="0.02 0.03 0.02"/><site name="tip" pos="0.02 0 0"/>
      </body>
    </body>
  </body>
</worldbody></mujoco>"""

START, GOAL = [1.5, -0.3], [-1.5, -0.3]  # the hand must swing past the wall (at j0 = 0 when j1 is near 0)
THROUGH = [[1.0, 0.0], [-1.0, 0.0]]  # straight through the wall


# #247: an arm that yaws (j0) and extends its gripper (j1) at z = 0.5, carrying a can 0.1 ahead of the gripper.
# Around it at the can's radius: a table (a box, so MuJoCo reports can -> table), a post (a sphere, so post -> can),
# and a stationary robot body outside the planned group. Nearer in, a bin the arm's link overlaps with a fixed
# <exclude>, and a crate it overlaps without one.
ALLOWANCE_WORLD = """
<mujoco><compiler angle="radian"/>
  <contact><exclude body1="arm/link" body2="bin"/></contact>
  <worldbody>
    <body name="table" pos="0.555 0 0.5"><geom type="box" size="0.03 0.08 0.08"/></body>
    <body name="post" pos="0 0.555 0.5"><geom type="sphere" size="0.03"/></body>
    <body name="other/link" pos="-0.555 0 0.5"><joint name="other/j" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.03 0.08 0.08"/></body>
    <body name="bin" pos="-0.17678 -0.17678 0.5" euler="0 0 0.7854"><geom type="box" size="0.03 0.03 0.03"/></body>
    <body name="crate" pos="-0.17678 0.17678 0.5" euler="0 0 -0.7854"><geom type="box" size="0.03 0.03 0.03"/></body>
    <body name="arm/link" pos="0 0 0.5"><joint name="j0" type="hinge" axis="0 0 1" range="-3.3 3.3"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.38 0 0"/>
      <body name="arm/gripper/base" pos="0.4 0 0"><joint name="j1" type="slide" axis="1 0 0" range="0 0.2"/>
        <geom type="box" size="0.02 0.03 0.02"/><site name="tip" pos="0.02 0 0"/></body>
    </body>
    <body name="can" pos="2 0 0.5"><freejoint/><geom type="cylinder" size="0.03 0.05"/></body>
    <body name="box2" pos="2 1 0.5"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>
  </worldbody>
</mujoco>"""
GRIPPER = "arm/gripper/base"
AHEAD, ABOVE = [0.1, 0.0, 0.0], [0.1, 0.0, 0.25]  # the can in the gripper's frame
BESIDE = [0.1, 0.045, 0.0]  # box2 in the gripper's frame: touching the can's side

# name: (configuration, {held body: offset}, [(snapshot label, {held body: extra allowed bodies})])
# Every snapshot also allows the gripper's subtree (the derived default); "strict" allows only that.
ALLOWANCE_CASES = {
    "can_on_table": ([0.0, 0.0], {"can": AHEAD}, [("strict", {}), ("table", {"can": ["table"]}), ("strict_again", {})]),
    "can_on_post": ([np.pi / 2, 0.0], {"can": AHEAD}, [("strict", {}), ("table", {"can": ["table"]}),
                                                         ("post", {"can": ["post"]})]),
    "can_on_stationary_robot": ([np.pi, 0.0], {"can": AHEAD}, [("strict", {}), ("other", {"can": ["other/link"]})]),
    "arm_on_table": ([0.0, 0.15], {"can": ABOVE}, [("strict", {}), ("table", {"can": ["table"]})]),
    "held_on_held": ([-np.pi / 2, 0.0], {"can": AHEAD, "box2": BESIDE},
                     [("strict", {}), ("each_other", {"can": ["box2"], "box2": ["can"]})]),
    "excluded_pair": ([-3 * np.pi / 4, 0.0], {"can": AHEAD}, [("strict", {}), ("bin", {"can": ["bin"]})]),
    "crate_not_excluded": ([3 * np.pi / 4, 0.0], {"can": AHEAD}, [("strict", {}), ("crate", {"can": ["crate"]})]),
}  # fmt: skip


def _allowance_snapshot(scene, qpos, held, extra):
    """A snapshot of ``qpos`` holding ``held`` with the gripper's subtree plus ``extra`` allowed."""
    from sscbirrt.backends import native_mujoco as nm

    mod, native = nm._load(), scene.native
    default = nm.gripper_allowed_bodies(scene, GRIPPER)
    attachments = []
    for obj, offset in held.items():
        T = np.eye(4)
        T[:3, 3] = offset
        allowed = default + [native.body_id(b) for b in extra.get(obj, [])]
        attachments.append(mod.Attachment(native.body_id(obj), native.body_id(GRIPPER), T.tolist(), allowed))
    return nm.Snapshot(mod.Snapshot(native, list(map(float, qpos)), [], [], attachments), scene)


def _allowances() -> dict[str, Any]:
    import mujoco

    from sscbirrt import CBiRRT, CBiRRTConfig
    from sscbirrt.backends.mujoco import MuJoCoRobotModel
    from sscbirrt.backends.native_mujoco import NativeCollisionChecker, NativeScene, _NoIK

    model = mujoco.MjModel.from_xml_string(ALLOWANCE_WORLD)
    data = mujoco.MjData(model)
    scene = NativeScene.from_model(model, ["j0", "j1"])
    robot = MuJoCoRobotModel(model, data, "tip", ["j0", "j1"])
    out: dict[str, Any] = {"mjcf": ALLOWANCE_WORLD.strip(), "gripper": GRIPPER, "cases": {}}
    for name, (q, held, snapshots) in ALLOWANCE_CASES.items():
        case: dict[str, Any] = {"q": list(map(float, q)), "held": held, "snapshots": {}}
        for label, extra in snapshots:
            snap = _allowance_snapshot(scene, data.qpos, held, extra)
            checker = NativeCollisionChecker(scene, snap)
            rec: dict[str, Any] = {
                "extra_allowed": extra,
                "sha256": snap.sha256,
                "scene": scene.provenance,
                "backends": {},
            }
            for backend in ("python", "native"):
                planner = CBiRRT(robot, _NoIK(), checker, CBiRRTConfig(), backend=backend)
                rec["backends"][backend] = _report(planner.validator().validate_state(q))
            case["snapshots"][label] = rec
        out["cases"][name] = case
    return out


def _allowance_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """A state's verdict with its contacts in MuJoCo's order and their kinds."""
    w = report["witness"]
    # each pair once, in MuJoCo's order (the number of contact points per pair is not part of the verdict)
    pairs = dict.fromkeys((c["body1"], c["body2"], c["kind"]) for c in ([] if w is None else w["contacts"]))
    return {"valid": report["valid"], "contacts": [list(p) for p in pairs]}


def _allowance_view(section: dict[str, Any]) -> dict[str, Any]:
    """The comparable part of the #247 section: verdicts per backend, which snapshots share a hash, and whether
    every snapshot reports one scene provenance."""
    view: dict[str, Any] = {}
    provenances = set()
    for name, case in section["cases"].items():
        labels = list(case["snapshots"])
        hashes = [case["snapshots"][k]["sha256"] for k in labels]
        view[name] = {
            "verdicts": {
                k: {b: _allowance_verdict(r) for b, r in s["backends"].items()} for k, s in case["snapshots"].items()
            },
            "same_hash": [[a for a in labels if hashes[labels.index(a)] == h] for h in dict.fromkeys(hashes)],
        }
        provenances |= {json.dumps(s["scene"], sort_keys=True) for s in case["snapshots"].values()}
    view["one_scene_provenance"] = len(provenances) == 1
    return view


def _report(r) -> dict[str, Any]:
    return r.to_dict()


def _trajectory(traj) -> list:
    return [
        {"validated": p.validated, "ramps": [[r.x0, r.v0, [list(s) for s in r.pieces]] for r in p.ramps]}
        for p in traj.pieces
    ]


def generate() -> dict[str, Any]:
    import mujoco

    from sscbirrt import TrajectoryOptions
    from sscbirrt.mujoco import Arm, plan, validator
    from sscbirrt.trajectory import Piece, Trajectory, synchronize

    model = mujoco.MjModel.from_xml_string(WORLD)
    data = mujoco.MjData(model)
    data.qpos[:2] = START
    mujoco.mj_forward(model, data)
    arm = Arm(model, ["j0", "j1"], "tip")
    options = TrajectoryOptions(velocity_limits=1.57, acceleration_limits=4.0)
    out: dict[str, Any] = {
        "artifact": "sscbirrt public validation evidence",
        "issue": "https://github.com/personalrobotics/sscbirrt/issues/230",
        "versions": {p: importlib.metadata.version(p) for p in ("sscbirrt", "mujoco", "numpy")},
        "world": {"mjcf": WORLD.strip(), "start": START, "goal": GOAL},
        "backends": {},
    }
    for backend in ("python", "native"):
        result = plan(model, data, arm, goal=GOAL, seed=0, backend=backend, trajectory=options)
        v = validator(model, data, arm, backend=backend)
        traj = result.trajectory
        # An interpolation that changes the trace: a rest-to-rest time-optimal motion straight through the wall.
        vmax, amax = traj.velocity_limits, traj.acceleration_limits
        piece = synchronize(np.array(THROUGH[0]), np.zeros(2), np.array(THROUGH[1]), np.zeros(2), vmax, amax)
        through = Trajectory([Piece(piece.ramps, "shortcut")], vmax, amax)
        queries = [
            ("accepted_path", lambda: v.validate_path(result.path)),
            ("rejected_path", lambda: v.validate_path(THROUGH)),
            ("reparameterization", lambda: v.validate_trajectory(traj)),
            ("trace_changing", lambda: v.validate_trajectory(through)),
            ("state_in_wall", lambda: v.validate_state([0.0, 0.0])),
        ]
        first = {name: _report(q()) for name, q in queries}
        again = {name: _report(q()) for name, q in queries}
        reversed_order = {name: _report(q()) for name, q in reversed(queries)}
        out["backends"][backend] = {
            "plan": {
                "success": result.success,
                "backend": result.backend,
                "path": [list(map(float, q)) for q in result.path],
                "trajectory": _trajectory(traj),
            },
            "during_planning": {"path": _report(result.validation), "trajectory": _report(traj.validation)},
            "public": first,
            "planning_equals_public": {
                "path": _report(result.validation) == first["accepted_path"],
                "trajectory": _report(traj.validation) == first["reparameterization"],
            },
            "repeated_equal": again == first,
            "reordered_equal": reversed_order == first,
            "through_the_wall": {"pieces": _trajectory(through)},
        }
    py, nat = out["backends"]["python"]["public"], out["backends"]["native"]["public"]
    out["python_equals_native_verdicts"] = {name: _verdict(py[name]) == _verdict(nat[name]) for name in py}
    out["allowances"] = _allowances()
    return out


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    """What must not change between runs, platforms or backends: no floats beyond the fraction, no backend."""
    w = report["witness"]
    return {
        "subject": report["subject"],
        "valid": report["valid"],
        "policy": report["policy"]["name"],
        "resolution": report["policy"]["resolution"],
        "witness": None
        if w is None
        else {
            "failure": w["failure"],
            "segment": w["segment"],
            "contacts": sorted({(c["body1"], c["body2"]) for c in w["contacts"]}),
        },
    }


def semantic_view(artifact: dict[str, Any]) -> dict[str, Any]:
    """The comparable part: verdicts and equalities, per backend; no paths, checks or versions (a plan's path
    depends on platform rounding, as in the reference artifact)."""
    view: dict[str, Any] = {
        "python_equals_native_verdicts": artifact["python_equals_native_verdicts"],
        "allowances": _allowance_view(artifact["allowances"]),
    }
    for backend, b in artifact["backends"].items():
        view[backend] = {
            "success": b["plan"]["success"],
            "planning_equals_public": b["planning_equals_public"],
            "repeated_equal": b["repeated_equal"],
            "reordered_equal": b["reordered_equal"],
            "public": {
                name: {**_verdict(r), "witness": None if r["witness"] is None else {**_verdict(r)["witness"]}}
                for name, r in b["public"].items()
            },
        }
    return view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare a fresh run's verdicts with the stored artifact")
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    fresh = generate()
    if args.check:
        stored = json.loads(args.output.read_text())
        a, b = semantic_view(stored), semantic_view(json.loads(json.dumps(fresh)))
        if a != b:
            print(f"validation evidence differs:\n  stored: {a}\n  fresh:  {b}")
            return 1
        print(
            "validation evidence matches (verdicts, planning = public, repeats and reorders, python = native, "
            "attachment allowances)"
        )
        return 0
    args.output.write_text(json.dumps(fresh, indent=1) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
