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

For #256 it records, on a third world with one native scene, checkers at clearance 0 and 5 mm: each case's verdict on
both backends, just inside and just outside the clearance (a primitive, a two-geom mesh body that only BVH
traversal reaches, a stationary robot body, a held object near the environment), a held object near its own arm, an
allowed contact, and a predefined ``<pair>``; which bodies each validator moves and the margins it collides with;
the override and flex refusals; that the caller's model and the scene are unchanged; that the two checkers interleave
without affecting each other; that with no attachment the 5 mm checker equals a model compiled with 5 mm margins,
contact for contact; and the policies and provenance of plans at both clearances.

For #267 it records, on the first world and both backends, ``CBiRRT.time_path`` on a planned path: with
``geometry="keep"`` (how far its samples lie from the given polyline) and ``"shortcut"``, each report next to the
public validator's; a rebase with a clear straight bridge prepended; one whose bridge crosses the wall
(``InvalidPath``, its witness); and, on a 5 mm checker, the clearance in the reports and a bridge from a state
that is valid at 0 but within 5 mm of the wall.

``tests/reference/geodude_132_evidence.json`` is the reduced Geodude edge of ssrobot#132, made by
``tools/geodude_132_evidence.py`` inside ssrobot.

    uv run python tools/validation_artifact.py            # write tests/reference/validation_evidence.json
    uv run python tools/validation_artifact.py --check    # compare a fresh run's verdicts with it
"""

from __future__ import annotations

import argparse
import hashlib
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
    from sscbirrt.backends.native_mujoco import Attachment, Snapshot, gripper_allowed_bodies

    default = gripper_allowed_bodies(scene, GRIPPER)
    attachments = []
    for obj, offset in held.items():
        T = np.eye(4)
        T[:3, 3] = offset
        attachments.append(Attachment(obj, GRIPPER, T, allowed_bodies=[*default, *extra.get(obj, [])]))
    return Snapshot.from_state(scene, qpos, attachments=attachments)


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


# #256: a gantry tip (jx, jy) at z = 0.5 whose body has two geoms, so every pair it is in is traversed by BVH. Near
# faces at x = 0.47: a box, a stationary robot body, and (at y = -0.3) a two-geom body with a mesh plate at x = 0.5;
# a sphere post (y = 0.6) whose pair with the tip is predefined with margin 0.
CLEARANCE = 0.005
_PLATE = " ".join(f"{x} {y} {z}" for x in (0, 0.02) for y in (-0.1, 0.1) for z in (-0.1, 0.1))
CLEARANCE_WORLD = f"""
<mujoco><compiler angle="radian"/>
  <asset><mesh name="plate" vertex="{_PLATE}"/></asset>
  <contact><pair geom1="tip" geom2="post_geom" margin="0"/></contact>
  <worldbody>
    <body name="block" pos="0.5 0.3 0.5"><geom name="block_geom" type="box" size="0.03 0.05 0.05"/></body>
    <body name="shelf" pos="0.5 -0.3 0.5"><geom name="shelf_plate" type="mesh" mesh="plate"/>
      <geom name="shelf_back" type="box" pos="0.12 0 0" size="0.01 0.01 0.01"/></body>
    <body name="other/link" pos="0.5 0 0.5"><joint name="other/j" type="hinge" axis="0 0 1"/>
      <geom name="other_geom" type="box" size="0.03 0.05 0.05"/></body>
    <body name="post" pos="0.5 0.6 0.5"><geom name="post_geom" type="sphere" size="0.03"/></body>
    <body name="arm" pos="0 0 0.5"><joint name="jx" type="slide" axis="1 0 0" range="-1 1"/>
      <joint name="jy" type="slide" axis="0 1 0" range="-1 1"/>
      <geom name="link" type="capsule" fromto="-0.2 0 0 -0.1 0 0" size="0.01"/>
      <geom name="tip" type="sphere" size="0.02"/><site name="tip_site"/></body>
    <body name="can" pos="2 2 0.5"><freejoint/><geom name="can_geom" type="cylinder" size="0.02 0.03"/></body>
  </worldbody>
</mujoco>"""
CLEARANCE_JOINTS = ["jx", "jy"]
_FACE = 0.47
_AHEAD = [0.045, 0.0, 0.0]  # a held can 5 mm ahead of the tip
_BESIDE = [0.0, 0.043, 0.0]  # a held can 3 mm beside the tip
_IN, _OUT = 0.0049, 0.0051  # just inside and just outside 5 mm
# name: (configuration, held can offset or None, its allowed bodies: None = the derived rule)
CLEARANCE_CASES = {
    "primitive_inside": ([_FACE - 0.02 - _IN, 0.3], None, None),
    "primitive_outside": ([_FACE - 0.02 - _OUT, 0.3], None, None),
    "mesh_inside": ([0.5 - 0.02 - _IN, -0.3], None, None),
    "mesh_outside": ([0.5 - 0.02 - _OUT, -0.3], None, None),
    "stationary_robot_inside": ([_FACE - 0.02 - _IN, 0.0], None, None),
    "stationary_robot_outside": ([_FACE - 0.02 - _OUT, 0.0], None, None),
    "held_near_environment_inside": ([_FACE - 0.065 - _IN, 0.3], _AHEAD, None),
    "held_near_environment_outside": ([_FACE - 0.065 - _OUT, 0.3], _AHEAD, None),
    "held_near_own_arm": ([0.0, -0.6], _BESIDE, ()),
    "held_on_allowed_body": ([_FACE - 0.065 + 0.001, 0.3], _AHEAD, ["block"]),
    "held_on_body_not_allowed": ([_FACE - 0.065 + 0.001, 0.3], _AHEAD, ()),
    "predefined_pair_inside": ([0.5 - 0.03 - 0.02 - _IN, 0.6], None, None),
    "predefined_pair_outside": ([0.5 - 0.03 - 0.02 - _OUT, 0.6], None, None),
}
_OVERRIDE_WORLD = CLEARANCE_WORLD.replace(
    '<compiler angle="radian"/>', '<compiler angle="radian"/><option><flag override="enable"/></option>'
)
_FLEX = (
    '<flexcomp name="rope" type="grid" count="3 1 1" spacing="0.05 0.05 0.05" pos="-0.5 -0.8 0.2" dim="1" '
    'radius="0.01"{}'
)
_FLEX_WORLD = CLEARANCE_WORLD.replace("  </worldbody>", "    " + _FLEX.format("/>") + "\n  </worldbody>")
_INERT_FLEX_WORLD = CLEARANCE_WORLD.replace(
    "  </worldbody>", "    " + _FLEX.format('><contact contype="0" conaffinity="0"/></flexcomp>') + "\n  </worldbody>"
)


def _clearance_snapshot(scene, model, held, allowed):
    import mujoco

    from sscbirrt.backends.native_mujoco import Attachment, Snapshot

    attachments = []
    if held is not None:
        T = np.eye(4)
        T[:3, 3] = held
        attachments = [Attachment("can", "arm", T, allowed_bodies=allowed)]
    return Snapshot.from_state(scene, mujoco.MjData(model).qpos, attachments=attachments)


def _refusal(xml: str) -> dict[str, Any]:
    """Whether a positive clearance is refused on ``xml``, and clearance 0 accepted."""
    import mujoco

    from sscbirrt.backends.native_mujoco import NativeCollisionChecker, NativeScene, Snapshot

    model = mujoco.MjModel.from_xml_string(xml)
    scene = NativeScene.from_model(model, CLEARANCE_JOINTS)
    snap = Snapshot.capture(scene, mujoco.MjData(model))
    NativeCollisionChecker(scene, snap)  # clearance 0 is accepted
    try:
        NativeCollisionChecker(scene, snap, clearance=CLEARANCE)
    except ValueError as e:
        return {"zero_accepted": True, "positive_refused": True, "error": str(e)}
    return {"zero_accepted": True, "positive_refused": False, "error": None}


def _contacts_key(validator, q) -> list:
    return [[c.body1, c.body2, c.geom1, c.geom2, c.dist, c.kind] for c in validator.invalid_contacts(q)]


def _clearances() -> dict[str, Any]:
    import mujoco

    from sscbirrt import CBiRRT, CBiRRTConfig
    from sscbirrt.backends.mujoco import MuJoCoRobotModel
    from sscbirrt.backends.native_mujoco import NativeCollisionChecker, NativeScene, _NoIK, export_mjb
    from sscbirrt.mujoco import Arm, plan

    model = mujoco.MjModel.from_xml_string(CLEARANCE_WORLD)
    data = mujoco.MjData(model)
    model_before = hashlib.sha256(export_mjb(model)).hexdigest()
    scene = NativeScene.from_model(model, CLEARANCE_JOINTS)
    robot = MuJoCoRobotModel(model, data, "tip_site", CLEARANCE_JOINTS)
    clearances = (0.0, CLEARANCE)
    out: dict[str, Any] = {
        "mjcf": CLEARANCE_WORLD.strip(),
        "clearances_m": list(clearances),
        "scene": scene.provenance,
        "cases": {},
    }
    nat = scene.native
    for name, (q, held, allowed) in CLEARANCE_CASES.items():
        snap = _clearance_snapshot(scene, model, held, allowed)
        case: dict[str, Any] = {"q": q, "held": held, "allowed": allowed, "snapshot_sha256": snap.sha256, "checks": {}}
        for c in clearances:
            checker = NativeCollisionChecker(scene, snap, clearance=c)
            reports = {}
            for backend in ("python", "native"):
                planner = CBiRRT(robot, _NoIK(), checker, CBiRRTConfig(), backend=backend)
                reports[backend] = _report(planner.validator().validate_state(q))
            case["checks"][f"{c:g}"] = reports
        out["cases"][name] = case

    # What each validator moves and the margins it collides with, empty-handed and holding the can.
    out["classification"] = {}
    for label, held in (("empty_hand", None), ("holding_can", _AHEAD)):
        snap = _clearance_snapshot(scene, model, held, None)
        views = {f"{c:g}": NativeCollisionChecker(scene, snap, clearance=c).native for c in clearances}
        v0 = views["0"]
        out["classification"][label] = {
            "moving_bodies": [nat.body_name(b) for b in v0.moving_bodies()],
            "geoms": [
                {
                    "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                    "body": nat.body_name(int(model.geom_bodyid[g])),
                    "authored_margin": float(model.geom_margin[g]),
                    "margin": {k: float(v.geom_margins()[g]) for k, v in views.items()},
                }
                for g in range(model.ngeom)
            ],
            "pair_margin": {k: [float(x) for x in v.pair_margins()] for k, v in views.items()},
        }

    out["refusals"] = {
        "override": _refusal(_OVERRIDE_WORLD),
        "flex_that_can_touch_the_moving_set": _refusal(_FLEX_WORLD),
        "flex_with_collisions_off": _refusal(_INERT_FLEX_WORLD),
    }

    # Two checkers over one scene, interleaved query by query, answer as each does alone.
    snap = _clearance_snapshot(scene, model, None, None)
    alone = {c: NativeCollisionChecker(scene, snap, clearance=c) for c in clearances}
    paired = {c: NativeCollisionChecker(scene, snap, clearance=c) for c in clearances}
    qs = [q for q, _, _ in CLEARANCE_CASES.values()]
    expected = {c: [_contacts_key(alone[c].native, q) for q in qs] for c in clearances}
    interleaved = {c: [] for c in clearances}
    for q in qs:
        for c in clearances:
            interleaved[c].append(_contacts_key(paired[c].native, q))
    out["interleaved_equal_alone"] = all(interleaved[c] == expected[c] for c in clearances)

    # With no attachment and no arm margin, 5 mm equals a model compiled with 5 mm on every environment geom.
    oracle_xml = CLEARANCE_WORLD.replace('  <contact><pair geom1="tip" geom2="post_geom" margin="0"/></contact>\n', "")
    arm_geoms = ('name="link"', 'name="tip"')
    margined = "\n".join(
        line if any(a in line for a in arm_geoms) else line.replace("<geom ", f'<geom margin="{CLEARANCE:g}" ')
        for line in oracle_xml.splitlines()
    )
    m_first, m_compiled = mujoco.MjModel.from_xml_string(oracle_xml), mujoco.MjModel.from_xml_string(margined)
    s_first, s_compiled = (
        NativeScene.from_model(m_first, CLEARANCE_JOINTS),
        NativeScene.from_model(m_compiled, CLEARANCE_JOINTS),
    )
    first = NativeCollisionChecker(s_first, _clearance_snapshot(s_first, m_first, None, None), clearance=CLEARANCE)
    compiled = NativeCollisionChecker(s_compiled, _clearance_snapshot(s_compiled, m_compiled, None, None))
    rng = np.random.default_rng(256)
    oracle_qs = [list(map(float, q)) for q in rng.uniform([-0.2, -0.75], [0.55, 0.75], size=(2000, 2))]
    agree = sum(_contacts_key(first.native, q) == _contacts_key(compiled.native, q) for q in oracle_qs)
    out["model_margin_oracle"] = {
        "configurations": len(oracle_qs),
        "identical_contact_lists": int(agree),
        "invalid_configurations": int(sum(not compiled.is_valid(np.array(q)) for q in oracle_qs)),
        "compiled_margin_m": CLEARANCE,
    }

    # Plans at both clearances: their policies and provenance.
    arm = Arm(model, CLEARANCE_JOINTS, "tip_site")
    out["plans"] = {}
    for backend in ("python", "native"):
        for c in clearances:
            result = plan(model, data, arm, start=[-0.3, 0.0], goal=[-0.3, 0.45], seed=0, backend=backend, clearance=c)
            out["plans"][f"{backend}_{c:g}"] = {
                "success": result.success,
                "backend": result.backend,
                "policy": result.validation.policy.to_dict(),
                "provenance": {
                    k: result.provenance.get(k) for k in ("clearance_m", "scene_mjb_sha256", "snapshot_sha256")
                },
            }

    out["mutation"] = {
        "caller_model_unchanged": hashlib.sha256(export_mjb(model)).hexdigest() == model_before,
        "scene_model_unchanged": nat.current_mjb_sha256() == scene.provenance["mjb_sha256"],
    }
    return out


def _clearance_view(section: dict[str, Any]) -> dict[str, Any]:
    """The comparable part of the #256 section."""
    cases = {
        name: {c: {b: _allowance_verdict(r) for b, r in reports.items()} for c, reports in case["checks"].items()}
        for name, case in section["cases"].items()
    }
    plans = section["plans"]
    return {
        "cases": cases,
        "classification": {
            label: {
                "moving_bodies": cls["moving_bodies"],
                "geoms": [(g["geom"], g["body"], g["margin"]) for g in cls["geoms"]],
                "pair_margin": cls["pair_margin"],
            }
            for label, cls in section["classification"].items()
        },
        "refusals": {k: (v["zero_accepted"], v["positive_refused"]) for k, v in section["refusals"].items()},
        "interleaved_equal_alone": section["interleaved_equal_alone"],
        "oracle_agrees": section["model_margin_oracle"]["identical_contact_lists"]
        == section["model_margin_oracle"]["configurations"],
        "plans": {
            k: (p["success"], p["policy"]["parameters"], p["provenance"]["clearance_m"]) for k, p in plans.items()
        },
        "one_scene_for_every_plan": len({p["provenance"]["scene_mjb_sha256"] for p in plans.values()}) == 1,
        "mutation": section["mutation"],
    }


# #267: rebase bridges for the first world's plan, prepended to the planned path.
CLEAR_BRIDGE_FROM = [1.7, -0.3]  # beside the start, away from the wall
BLOCKED_BRIDGE_FROM = [-1.0, 0.0]  # its straight line to the start sweeps the hand through the wall
NEAR_WALL = [0.0, -0.2745]  # valid at clearance 0, within 5 mm of the wall


def _polyline_distance(points: np.ndarray, corners: list[np.ndarray]) -> float:
    """The largest distance from ``points`` to the polyline through ``corners`` (joint space)."""
    best = np.full(len(points), np.inf)
    for a, b in zip(corners[:-1], corners[1:]):
        d = b - a
        t = np.clip(((points - a) @ d) / max(float(d @ d), 1e-30), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * d), axis=1))
    return float(best.max())


def _time_paths() -> dict[str, Any]:
    import mujoco

    from sscbirrt import CBiRRT, CBiRRTConfig, InvalidPath, TrajectoryOptions
    from sscbirrt.backends.mujoco import MuJoCoRobotModel
    from sscbirrt.backends.native_mujoco import NativeCollisionChecker, NativeScene, Snapshot, _NoIK

    model = mujoco.MjModel.from_xml_string(WORLD)
    data = mujoco.MjData(model)
    scene = NativeScene.from_model(model, ["j0", "j1"])
    snap = Snapshot.capture(scene, data)
    robot = MuJoCoRobotModel(model, data, "tip", ["j0", "j1"])
    options = TrajectoryOptions(velocity_limits=1.57, acceleration_limits=4.0)
    out: dict[str, Any] = {}
    for backend in ("python", "native"):
        for clearance in (0.0, 0.005):
            checker = NativeCollisionChecker(scene, snap, clearance=clearance)
            planner = CBiRRT(robot, _NoIK(), checker, CBiRRTConfig(), backend=backend)
            result = planner.plan(start=np.array(START), goal=np.array(GOAL), seed=0, return_details=True)
            path = [np.asarray(q, dtype=float) for q in result.path]
            public = planner.validator()
            rec: dict[str, Any] = {"plan_success": result.success, "path_waypoints": len(path)}
            for geometry in ("keep", "shortcut"):
                traj = planner.time_path(path, options, geometry=geometry)
                samples = traj.sample(10_000_000)[1]  # every 10 ms
                rec[geometry] = {
                    "report": _report(traj.validation),
                    "report_equals_public": traj.validation == public.validate_trajectory(traj),
                    "duration_s": traj.duration,
                    "pieces": len(traj.pieces),
                    "max_distance_from_path": _polyline_distance(samples, path),
                }
            for name, q0 in (("clear_bridge", CLEAR_BRIDGE_FROM), ("blocked_bridge", BLOCKED_BRIDGE_FROM),
                             ("near_wall_bridge", NEAR_WALL)):  # fmt: skip
                rebased = [np.array(q0, dtype=float), *path]
                try:
                    traj = planner.time_path(rebased, options, geometry="keep")
                    rec[name] = {"timed": True, "report": _report(traj.validation)}
                except InvalidPath as e:
                    rec[name] = {"timed": False, "error": type(e).__name__, "report": _report(e.report)}
            out[f"{backend}_{clearance:g}"] = rec
    return out


def _time_path_view(section: dict[str, Any]) -> dict[str, Any]:
    """The comparable part of the #267 section: decisions, report equalities, geometry kept or not."""
    view = {}
    for key, rec in section.items():
        v: dict[str, Any] = {"plan_success": rec["plan_success"]}
        for geometry in ("keep", "shortcut"):
            g = rec[geometry]
            v[geometry] = {
                "valid": g["report"]["valid"],
                "clearance_m": g["report"]["policy"]["parameters"][0],
                "report_equals_public": g["report_equals_public"],
                "on_the_path": g["max_distance_from_path"] < 1e-9,
            }
        v["shortcut_no_slower"] = rec["shortcut"]["duration_s"] <= rec["keep"]["duration_s"] + 1e-9
        for name in ("clear_bridge", "blocked_bridge", "near_wall_bridge"):
            r = rec[name]
            w = r["report"]["witness"]
            v[name] = {
                "timed": r["timed"],
                "error": r.get("error"),
                "witness": None
                if w is None
                else {"failure": w["failure"], "segment": w["segment"],
                      "bodies": sorted({b for c in w["contacts"] for b in (c["body1"], c["body2"])})},
            }  # fmt: skip
        view[key] = v
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
    out["clearance"] = _clearances()
    out["time_path"] = _time_paths()
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
        "clearance": _clearance_view(artifact["clearance"]),
        "time_path": _time_path_view(artifact["time_path"]),
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
            "attachment allowances, clearance, time_path)"
        )
        return 0
    args.output.write_text(json.dumps(fresh, indent=1) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
