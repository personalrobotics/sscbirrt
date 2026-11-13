# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The decision-parity corpus for the native MuJoCo validator (#84, #138).

Builds small MuJoCo worlds covering self-collision, robot-environment, held-object-environment, allowed
gripper-object contact, mocap bodies, contact margins, cylinders, box-box, and plane-mesh contact. For each
scenario it records configurations near and across contact with:

- mj_manipulator's ``CollisionChecker`` decision, the policy the native validator reproduces;
- sscbirrt's legacy ``MuJoCoCollisionChecker`` decision where it applies (no attachments). That checker
  counts every contact in the world, environment against environment included, so it rejects any world
  with resting contacts; it is a one-sided check (it never accepts what the policy rejects);
- whether ``mj_kinematics`` followed by ``mj_collision`` yields the same contact geom pairs as ``mj_forward``.

``tests/test_mujoco_collision_corpus.py`` replays the native validator against the recorded decisions.
Regenerating needs mj_manipulator, which is retired and archived (personalrobotics/mj_manipulator#182). The
corpus was recorded with 2.0.0; its final tag, v2.1.0, reproduces every recorded decision. Install it on Python
3.11 or later with its workspace sources ignored. The overlay brings its own pins (sstsr 3.x, sscbirrt 3.x from
PyPI, no sscbirrt-assets), so use it for this tool only. It writes
``tests/reference/mujoco_collision_corpus.json``:

    MJM="mj_manipulator @ git+https://github.com/personalrobotics/mj_manipulator@v2.1.0"
    uv run --no-sources --with "$MJM" python tools/mujoco_collision_corpus.py          # write the reference
    uv run --no-sources --with "$MJM" python tools/mujoco_collision_corpus.py --check  # regenerate, compare
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ARTIFACT = Path(__file__).resolve().parent.parent / "tests" / "reference" / "mujoco_collision_corpus.json"
SEED = 1707
JOINTS = ["arm/j0", "arm/j1", "arm/j2"]

# A three-joint arm with a two-finger gripper, a table with a contact margin, a can with a free joint, a
# mocap sphere, an inline tetrahedral mesh resting on the floor plane, and a second box for box-box contact.
WORLD = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <asset>
    <mesh name="tet" vertex="0 0 0  0.12 0 0  0 0.12 0  0 0 0.12"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <geom name="table" type="box" pos="0.55 0 0.15" size="0.15 0.3 0.15" margin="{margin}"/>
    <geom name="crate" type="box" pos="0.2 -0.45 0.1" size="0.08 0.08 0.1"/>
    <body name="tetra" pos="-0.4 0.4 -0.01"><geom type="mesh" mesh="tet"/></body>
    <body name="arm/base" pos="0 0 0.1">
      <joint name="arm/j0" type="hinge" axis="0 0 1" limited="true" range="-3 3"/>
      <geom name="arm/link0" type="capsule" size="0.035" fromto="0 0 0 0 0 0.15"/>
      <body name="arm/link1" pos="0 0 0.15">
        <joint name="arm/j1" type="hinge" axis="0 1 0" limited="true" range="-2.5 2.5"/>
        <geom name="arm/link1_geom" type="capsule" size="0.03" fromto="0 0 0 0.3 0 0"/>
        <body name="arm/link2" pos="0.3 0 0">
          <joint name="arm/j2" type="hinge" axis="0 1 0" limited="true" range="-2.8 2.8"/>
          <geom name="arm/link2_geom" type="capsule" size="0.025" fromto="0 0 0 0.25 0 0"/>
          <body name="arm/gripper/base" pos="0.25 0 0">
            <geom name="arm/gripper/palm" type="box" size="0.02 0.04 0.02"/>
            <body name="arm/gripper/left_finger" pos="0.06 0.035 0">
              <geom name="arm/gripper/lf" type="box" size="0.04 0.006 0.015"/>
            </body>
            <body name="arm/gripper/right_finger" pos="0.06 -0.035 0">
              <geom name="arm/gripper/rf" type="box" size="0.04 0.006 0.015"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="can" pos="0.62 0 0.36"><freejoint/><geom name="can/geom" type="cylinder" size="0.031 0.06"/></body>
    <body name="ball" mocap="true" pos="0.3 0.35 0.3"><geom name="ball/geom" type="sphere" size="0.06"/></body>
  </worldbody>
</mujoco>
"""


def scenarios() -> list[dict[str, Any]]:
    """Each: an MJCF string, a live-state setup, and whether the can is grasped."""
    return [
        {"name": "free_arm_no_margin", "margin": 0.0, "grasp": False, "mocap": [0.3, 0.35, 0.3]},
        {"name": "free_arm_table_margin_2cm", "margin": 0.02, "grasp": False, "mocap": [0.3, 0.35, 0.3]},
        {"name": "mocap_ball_in_the_workspace", "margin": 0.0, "grasp": False, "mocap": [0.35, 0.0, 0.32]},
        {"name": "holding_the_can", "margin": 0.0, "grasp": True, "mocap": [0.3, 0.35, 0.3]},
        {"name": "holding_the_can_table_margin", "margin": 0.02, "grasp": True, "mocap": [0.3, 0.35, 0.3]},
    ]


def _grasp_pose(model, data):
    """Put the can between the fingers at the current configuration and return T_gripper_object."""
    mujoco.mj_forward(model, data)
    gid = model.body("arm/gripper/base").id
    T_wg = np.eye(4)
    T_wg[:3, :3] = data.xmat[gid].reshape(3, 3)
    T_wg[:3, 3] = data.xpos[gid]
    T_go = np.eye(4)
    T_go[:3, 3] = [0.08, 0.0, 0.0]  # centered in the fingers' span (x 0.02 to 0.10), clear of the forearm's rounded end
    T_wo = T_wg @ T_go
    can = model.body("can").id
    adr = model.jnt_qposadr[model.body_jntadr[can]]
    data.qpos[adr : adr + 3] = T_wo[:3, 3]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T_wo[:3, :3].reshape(9))
    data.qpos[adr + 3 : adr + 7] = q
    mujoco.mj_forward(model, data)
    return T_go


def _contact_pairs(model, data):
    return sorted((int(min(c.geom[0], c.geom[1])), int(max(c.geom[0], c.geom[1]))) for c in data.contact[: data.ncon])


def _forward_vs_kinematics_agree(model, data_template, q, adrs) -> bool:
    d1, d2 = mujoco.MjData(model), mujoco.MjData(model)
    for d in (d1, d2):
        d.qpos[:] = data_template.qpos
        d.mocap_pos[:] = data_template.mocap_pos
        d.mocap_quat[:] = data_template.mocap_quat
        for i, a in enumerate(adrs):
            d.qpos[a] = q[i]
    mujoco.mj_forward(model, d1)
    mujoco.mj_kinematics(model, d2)
    mujoco.mj_collision(model, d2)
    return _contact_pairs(model, d1) == _contact_pairs(model, d2)


def _configs(rng, model, joint_ids, decision, n_random=48) -> list[list[float]]:
    lo = model.jnt_range[joint_ids, 0]
    hi = model.jnt_range[joint_ids, 1]
    qs = [rng.uniform(lo, hi) for _ in range(n_random)]
    # bisect between random pairs of opposite decision to land within 1e-3 of a boundary
    out = list(qs)
    for a, b in zip(qs[:-1], qs[1:]):
        if decision(a) == decision(b):
            continue
        x, y = np.array(a), np.array(b)
        for _ in range(12):
            m = 0.5 * (x + y)
            if decision(m) == decision(a):
                x = m
            else:
                y = m
        out.extend([x, y])
    return [list(map(float, q)) for q in out]


def build(spec) -> tuple[Any, Any, dict[str, Any] | None]:
    model = mujoco.MjModel.from_xml_string(WORLD.replace("{margin}", str(spec["margin"])))
    data = mujoco.MjData(model)
    data.mocap_pos[0] = spec["mocap"]
    attachments = None
    if spec["grasp"]:
        data.qpos[: len(JOINTS)] = [0.0, -0.6, 0.9]
        T = _grasp_pose(model, data)
        attachments = {"can": ("arm/gripper/base", T)}
    mujoco.mj_forward(model, data)
    return model, data, attachments


def generate() -> dict[str, Any]:
    try:
        from mj_manipulator.collision import CollisionChecker as MjmChecker
    except ImportError:
        print(
            "mj_manipulator is not installed; it is needed to regenerate the corpus. It is retired: run this "
            'tool with uv run --no-sources --with "mj_manipulator @ '
            "git+https://github.com/personalrobotics/mj_manipulator@v2.1.0\" (see this module's docstring)",
            file=sys.stderr,
        )
        raise
    from sscbirrt.backends.mujoco import MuJoCoCollisionChecker

    rng = np.random.default_rng(SEED)
    out = []
    for spec in scenarios():
        model, data, attachments = build(spec)
        joint_ids = np.array([model.joint(j).id for j in JOINTS])
        adrs = model.jnt_qposadr[joint_ids]
        grasped = frozenset({("can", "arm")}) if attachments else frozenset()
        mjm = MjmChecker(model, mujoco.MjData(model), JOINTS, grasped_objects=grasped, attachments=attachments or {})
        # snapshot mode: its private data must carry the live state (qpos of everything, mocap)
        mjm.data.qpos[:] = data.qpos
        mjm.data.mocap_pos[:] = data.mocap_pos
        mjm.data.mocap_quat[:] = data.mocap_quat
        pyc = None if attachments else MuJoCoCollisionChecker(model, mujoco.MjData(model), JOINTS)
        if pyc is not None:
            pyc.data.qpos[:] = data.qpos
            pyc.data.mocap_pos[:] = data.mocap_pos
            pyc.data.mocap_quat[:] = data.mocap_quat
        configs = _configs(rng, model, joint_ids, lambda q: bool(mjm.is_valid(np.asarray(q))))
        records = []
        for q in configs:
            records.append(
                {
                    "q": q,
                    "mj_manipulator_valid": bool(mjm.is_valid(np.asarray(q))),
                    "sscbirrt_valid": None if pyc is None else bool(pyc.is_valid(np.asarray(q))),
                    "forward_vs_kinematics_agree": _forward_vs_kinematics_agree(model, data, q, adrs),
                    "invalid_contacts_mjm": sorted(f"{a}|{b}" for a, b, _ in mjm.get_contacts(np.asarray(q))),
                }
            )
        out.append(
            {
                "name": spec["name"],
                "xml": WORLD.replace("{margin}", str(spec["margin"])),
                "joints": JOINTS,
                "qpos": data.qpos.tolist(),
                "mocap_pos": data.mocap_pos.reshape(-1).tolist(),
                "mocap_quat": data.mocap_quat.reshape(-1).tolist(),
                "attachments": {k: [v[0], v[1].tolist()] for k, v in (attachments or {}).items()},
                "configurations": records,
            }
        )
    return {
        "artifact": "decision parity between the native MuJoCo validator and mj_manipulator's CollisionChecker",
        "issue": "https://github.com/personalrobotics/sscbirrt/issues/84",
        "seed": SEED,
        "versions": {
            "mujoco": mujoco.__version__,
            "mj_manipulator": importlib.metadata.version("mj_manipulator"),
            "numpy": np.__version__,
        },
        "scenarios": out,
    }


def decisions(artifact) -> list[tuple[str, int, bool]]:
    return [
        (s["name"], i, r["mj_manipulator_valid"])
        for s in artifact["scenarios"]
        for i, r in enumerate(s["configurations"])
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    fresh = generate()
    n = sum(len(s["configurations"]) for s in fresh["scenarios"])
    if args.check:
        stored = json.loads(args.output.read_text())
        if decisions(stored) != decisions(fresh):
            print("MISMATCH: mj_manipulator's decisions on the corpus changed", file=sys.stderr)
            return 1
        print(
            f"corpus decisions unchanged ({len(fresh['scenarios'])} scenarios, {n} configurations; "
            f"versions {fresh['versions']})"
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    disagreements = sum(not r["forward_vs_kinematics_agree"] for s in fresh["scenarios"] for r in s["configurations"])
    print(
        f"wrote {args.output} ({len(fresh['scenarios'])} scenarios, {n} configurations, "
        f"{disagreements} forward/kinematics disagreements; versions {fresh['versions']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
