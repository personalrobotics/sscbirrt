# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generate or check the deterministic Python reference artifact (#94).

The artifact is the definition of "compatible with the Python reference" for
the native backends (#82, #92). It runs a fixed matrix of planning problems on
the planar reference arm with fixed seeds and records, per case:

- the problem specification as data;
- the seed and the implementation versions;
- the result: status, failure category, provenance, iterations, path;
- an independent validation report computed from the problem alone, without
  the planner: every waypoint in the joint space, the first waypoint in the
  declared start set and the last in the declared goal set, every waypoint
  admissible (validator and path constraint), every consecutive pair
  validated at ``edge_resolution`` along the space's direction, and raw joint
  continuity on angular joints.

The semantic oracle is the status, category, provenance, endpoint membership,
and validation report. Paths and iteration counts are recorded for
inspection, not compared, so the artifact is stable across platforms while
the planner's observable behavior is pinned.

Usage:
    python tools/reference_artifact.py            # regenerate tests/reference/python_reference.json
    python tools/reference_artifact.py --check    # regenerate in memory and compare semantics with the file
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tsr import TSR, TSRChain

from sscbirrt import CBiRRT, CBiRRTConfig, PlanningProblem
from sscbirrt.sets import AllOf, AnyOf, FiniteSet, MostViolatedProjection, PredicateSet, explicit_samples
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK, Wall  # noqa: F401
from sscbirrt.tsr_set import TSRConfigurationSet, tsr_weights

ARTIFACT = Path(__file__).resolve().parent.parent / "tests" / "reference" / "python_reference.json"
SEMANTIC_KEYS = ("status", "failure_category", "start_source", "goal_source", "validation")


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------


def _frame(x: float, y: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    return T


BOX = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
Y_BAND = np.array([[-2.0, 2.0], [-0.6, 0.6], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])


def _planner(config: CBiRRTConfig, validator=None) -> CBiRRT:
    robot = PlanarArm()
    return CBiRRT(robot, PlanarIK(robot), validator or NoCollision(), config)


def _finite(planner: CBiRRT, configs) -> FiniteSet:
    """Finite set under the planner's space metric, as the legacy lowering builds them (matters for angular joints)."""
    return FiniteSet(configs, tolerance=planner.config.membership_tolerance, metric=planner.space.distance)


def _tsr_set(planner: CBiRRT, tsr) -> TSRConfigurationSet:
    return TSRConfigurationSet(
        tsr, planner.robot, planner.ik, planner.space, tolerance=planner.config.membership_tolerance
    )


def cases() -> list[dict[str, Any]]:
    """Each case: name, description, seed, config, planner, problem, spec (as data)."""
    out: list[dict[str, Any]] = []

    def add(name, description, seed, config, problem, spec, planner):
        out.append(
            {
                "name": name,
                "description": description,
                "seed": seed,
                "config": config,
                "problem": problem,
                "spec": spec,
                "planner": planner,
            }
        )

    base = dict(smooth_path=True, step_size=0.2, connection_tolerance=1e-3, edge_resolution=0.05, timeout=30.0)

    # 1. fixed start to fixed goal
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    q0, q1 = np.zeros(2), np.array([1.0, 0.5])
    add(
        "fixed_to_fixed",
        "one fixed start, one fixed goal, no constraint",
        0,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, [q0]), goal=_finite(p, [q1]), validator=p.collision),
        {"start": [q0.tolist()], "goal": [q1.tolist()]},
        p,
    )

    # 2. multiple roots
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    starts = [np.array([0.0, 0.0]), np.array([0.1, 0.1])]
    goals = [np.array([1.0, 0.5]), np.array([-1.0, 0.5])]
    add(
        "multiple_roots",
        "two fixed starts, two fixed goals; provenance reports which pair connected",
        1,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, starts), goal=_finite(p, goals), validator=p.collision),
        {"start": [s.tolist() for s in starts], "goal": [g.tolist() for g in goals]},
        p,
    )

    # 3. nested finite goal with a predicate filter
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    near, far, bad = np.array([0.2, 0.1]), np.array([2.5, 2.5]), np.array([0.2, -0.5])
    goal = AnyOf([AllOf([FiniteSet([bad, near]), PredicateSet(lambda q: q[1] > 0, name="q1>0")]), FiniteSet([far])])
    add(
        "nested_finite_goal",
        "AnyOf(AllOf(FiniteSet, predicate), FiniteSet): the filtered member is expected, provenance (0, 1)",
        2,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, [np.zeros(2)]), goal=goal, validator=p.collision),
        {"goal": "AnyOf([AllOf([FiniteSet([bad, near]), q1>0]), FiniteSet([far])])", "near": near.tolist()},
        p,
    )

    # 4. wrapped joint across the seam
    cfg = CBiRRTConfig(continuous_joints=(True, False), **base)
    p = _planner(cfg)
    q0, q1 = np.array([3.0, 0.0]), np.array([-3.0, 0.0])
    add(
        "wrapped_seam",
        "joint 0 angular; start and goal 0.28 rad apart across the seam; output continuous in raw values",
        3,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, [q0]), goal=_finite(p, [q1]), validator=p.collision),
        {"angular_joints": [True, False], "start": [q0.tolist()], "goal": [q1.tolist()]},
        p,
    )

    # 5. rejection-only path constraint
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    half = PredicateSet(lambda q: q[1] >= -1e-9, name="q1>=0")
    add(
        "rejection_constraint",
        "membership-only path constraint q1 >= 0, enforced by rejection",
        4,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.zeros(2)]),
            goal=_finite(p, [np.array([1.0, 0.3])]),
            validator=p.collision,
            path_constraint=half,
        ),
        {"path_constraint": "PredicateSet(q1>=0)"},
        p,
    )

    # 6. projected path constraint (TSR band)
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    band = _tsr_set(p, TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=Y_BAND))
    q0, q1 = np.array([0.0, 0.6]), np.array([-0.3, 0.9])
    add(
        "projected_constraint",
        "TSR path constraint: end effector y within ±0.6, projected each step",
        5,
        cfg,
        PlanningProblem(
            space=p.space, start=_finite(p, [q0]), goal=_finite(p, [q1]), validator=p.collision, path_constraint=band
        ),
        {"path_constraint": {"tsr_Bw": Y_BAND.tolist()}},
        p,
    )

    # 7. TSR goal union
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    a = _tsr_set(p, TSR(T0_w=_frame(1.9, 0.3), Tw_e=np.eye(4), Bw=BOX))
    b = _tsr_set(p, TSR(T0_w=_frame(-1.5, 1.0), Tw_e=np.eye(4), Bw=BOX))
    add(
        "tsr_goal_union",
        "AnyOf of two TSR regions weighted by volume; provenance reports which",
        6,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.zeros(2)]),
            goal=AnyOf([a, b], weights=tsr_weights([a, b])),
            validator=p.collision,
        ),
        {"goal": "AnyOf([TSR@(1.9,0.3), TSR@(-1.5,1.0)])"},
        p,
    )

    # 8. TSR chain goal
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    hinge = TSR(T0_w=_frame(1.0, 0.5), Tw_e=np.eye(4), Bw=np.array([[0, 0]] * 5 + [[-np.pi, np.pi]]))
    handle = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=np.array([[0.3, 0.3]] + [[0, 0]] * 4 + [[-np.pi, np.pi]]))
    chain = _tsr_set(p, TSRChain(TSRs=[hinge, handle]))
    add(
        "tsr_chain_goal",
        "door chain: hinge at (1.0, 0.5) with free yaw, handle 0.3 along the door",
        7,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, [np.zeros(2)]), goal=chain, validator=p.collision),
        {"goal": "TSRChain(hinge@(1.0,0.5) yaw free, handle x=0.3 yaw free)"},
        p,
    )

    # 9. two-TSR intersection constraint with the named strategy
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    x_band = _tsr_set(
        p,
        TSR(
            T0_w=_frame(1.5, 0.0),
            Tw_e=np.eye(4),
            Bw=np.array([[-0.5, 0.5], [-2, 2], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]]),
        ),
    )
    y_band = _tsr_set(p, TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=Y_BAND))
    both = AllOf([x_band, y_band], projection=MostViolatedProjection())
    add(
        "allof_constraint",
        "AllOf of two TSR bands projected with MostViolatedProjection",
        8,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.array([0.0, 0.6])]),
            goal=_finite(p, [np.array([-0.3, 0.9])]),
            validator=p.collision,
            path_constraint=both,
        ),
        {"path_constraint": "AllOf([x in [1,2], y in [-0.6,0.6]], MostViolatedProjection)"},
        p,
    )

    # 10. timeout: the smallest positive timeout, on a problem no single iteration can solve, so the
    # outcome does not depend on the clock's resolution (a coarse clock may let one iteration run)
    cfg = CBiRRTConfig(**{**base, "timeout": 1e-9})
    timeout_wall = Wall(axis=0, lo=0.45, hi=0.55)
    p = _planner(cfg, timeout_wall)
    add(
        "timeout",
        "a one-nanosecond timeout with a wall between start and goal: the deadline reports a timeout",
        9,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.zeros(2)]),
            goal=_finite(p, [np.array([1.0, 0.0])]),
            validator=timeout_wall,
        ),
        {"timeout": 1e-9, "validator": "Wall(q0 in (0.45, 0.55))"},
        p,
    )

    # 11. cancellation
    cfg = CBiRRTConfig(**{**base, "abort_fn": lambda: True})
    p = _planner(cfg)
    add(
        "cancellation",
        "abort_fn always true: the first iteration reports an abort",
        10,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.zeros(2)]),
            goal=_finite(p, [np.array([1.0, 0.5])]),
            validator=p.collision,
        ),
        {"abort": True},
        p,
    )

    # 12. unreachable: a wall across all of joint 1 between start and goal
    cfg = CBiRRTConfig(**{**base, "max_iterations": 300, "timeout": 30.0})
    wall = Wall(axis=0, lo=0.45, hi=0.55)
    p = _planner(cfg, wall)
    add(
        "unreachable",
        "a wall at q0 in (0.45, 0.55) across all q1 separates start and goal; max iterations reached",
        11,
        cfg,
        PlanningProblem(
            space=p.space, start=_finite(p, [np.zeros(2)]), goal=_finite(p, [np.array([1.0, 0.0])]), validator=wall
        ),
        {"validator": "Wall(q0 in (0.45, 0.55))", "max_iterations": 300},
        p,
    )

    out.extend(ur5e_cases(base))
    out.extend(ur5e_mujoco_cases(base))
    return out


def _skipped(name: str, description: str, seed: int, reason: str) -> dict[str, Any]:
    return {"name": name, "description": description, "seed": seed, "skipped": reason}


def ur5e_mujoco_cases(base: dict[str, Any]) -> list[dict[str, Any]]:
    """The v1.7.0 release cases (#88): the UR5e with the Robotiq gripper in the example's MuJoCo scene, a
    TSR-goal query among the table obstacles and a held-object query, both from a snapshot with native
    collision checking and SSIK. Recorded as skipped when the Menagerie, mujoco, ssik, or the native scene
    is unavailable, so the artifact keeps every case name in every environment."""
    names = [
        (
            "ur5e_mujoco_tsr_goal_among_obstacles",
            "UR5e + Robotiq in MuJoCo: finite start, top-down grasp TSR above the cylinder on the table",
            20,
        ),
        (
            "ur5e_mujoco_held_object",
            "UR5e + Robotiq holding the cylinder: finite start, place TSR over the table, gripper contact allowed",
            21,
        ),
        (
            "ur5e_mujoco_tsr_chain_crank",
            "UR5e + Robotiq turning a crank 60 degrees: start, goal, and path constraint are one TSR chain (hinge, "
            "then a grasp free to spin about the handle's bar), native since sstsr 3.3 (#184)",
            0,
        ),
    ]
    reason = None
    menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if not menagerie:
        reason = "MUJOCO_MENAGERIE_PATH is not set"
    try:
        import mujoco  # noqa: F401
        import ssik

        from sscbirrt.backends import native_mujoco
        from sscbirrt.backends.mujoco import MuJoCoRobotModel, site_offset_in_body
        from sscbirrt.backends.ssik import SSIKSolver
    except ImportError as e:
        reason = reason or f"missing dependency: {e}"
    if reason is None and not native_mujoco.available():
        reason = native_mujoco.unavailable_reason()
    if reason is not None:
        print(f"note: UR5e MuJoCo cases skipped: {reason}", file=sys.stderr)
        return [_skipped(n, d, s, reason) for n, d, s in names]

    from sscbirrt.testing.ur5e import create_grasp_tsr, create_scene

    joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    model = create_scene(Path(menagerie), free_cylinder=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = ssik.Manipulator.from_mjcf(
        str(Path(menagerie) / "universal_robots_ur5e" / "ur5e.xml"), base="world", ee="wrist_3_link"
    )
    ik = SSIKSolver(arm, T_ee=site_offset_in_body(model, "attachment_site"))
    robot = MuJoCoRobotModel(model, data, "attachment_site", joints)
    scene = native_mujoco.NativeScene.from_model(model, joints)
    cfg = CBiRRTConfig(**{**base, "num_tree_roots": 20, "timeout": 60.0})
    home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    cylinder_pos = model.body("cylinder").pos.copy()
    out: list[dict[str, Any]] = []

    # A. grasp TSR above the resting cylinder, everything else an obstacle
    snap_a = native_mujoco.Snapshot.capture(scene, data)
    checker_a = native_mujoco.NativeCollisionChecker(scene, snap_a)
    planner_a = CBiRRT(robot, ik, checker_a, cfg)
    out.append(
        {
            "name": names[0][0],
            "description": names[0][1],
            "seed": names[0][2],
            "config": cfg,
            "problem": PlanningProblem(
                space=planner_a.space,
                start=_finite(planner_a, [home]),
                goal=_tsr_set(planner_a, create_grasp_tsr(cylinder_pos)),
                validator=checker_a,
            ),
            "spec": {
                "robot": "menagerie ur5e + robotiq_2f85",
                "ssik": importlib.metadata.version("ssik"),
                "mujoco": mujoco.__version__,
                "solver_name": arm.solver_name,
                "scene_mjb_sha256": scene.provenance["mjb_sha256"],
                "snapshot_sha256": snap_a.sha256,
                "start": [home.tolist()],
                "goal": "grasp TSR above the cylinder",
            },
            "planner": planner_a,
        }
    )

    # B. holding the cylinder: attach it in the gripper, plan to a place TSR over the other side of the table
    q_hold = home.copy()
    for i, a in enumerate(model.jnt_qposadr[[model.joint(j).id for j in joints]]):
        data.qpos[a] = q_hold[i]
    mujoco.mj_forward(model, data)
    T_go = np.eye(4)
    T_go[:3, 3] = [0.0, 0.0, 0.16]  # along the gripper's approach axis, between the pads and past the wrist
    attachments = {"cylinder": ("gripper_base_mount", T_go)}
    snap_b = native_mujoco.Snapshot.capture(scene, data, attachments=attachments)
    checker_b = native_mujoco.NativeCollisionChecker(scene, snap_b)
    planner_b = CBiRRT(robot, ik, checker_b, cfg)
    place = create_grasp_tsr(np.array([0.5, -0.25, 0.47]))
    out.append(
        {
            "name": names[1][0],
            "description": names[1][1],
            "seed": names[1][2],
            "config": cfg,
            "problem": PlanningProblem(
                space=planner_b.space,
                start=_finite(planner_b, [q_hold]),
                goal=_tsr_set(planner_b, place),
                validator=checker_b,
            ),
            "spec": {
                "robot": "menagerie ur5e + robotiq_2f85",
                "ssik": importlib.metadata.version("ssik"),
                "mujoco": mujoco.__version__,
                "solver_name": arm.solver_name,
                "scene_mjb_sha256": scene.provenance["mjb_sha256"],
                "snapshot_sha256": snap_b.sha256,
                "attachments": {"cylinder": ["gripper_base_mount", T_go.tolist()]},
                "start": [q_hold.tolist()],
                "goal": "place TSR over the table",
            },
            "planner": planner_b,
        }
    )

    # C. a TSR chain as start, goal, and path constraint: turn a crank about a vertical axis above the table. The
    # flange holds the handle from above and may spin about the handle's vertical bar; that axis is offset from the
    # hinge, so the chain does not reduce to a single TSR. Its distance is sstsr's bounded inverse in both backends.
    def crank(lo: float, hi: float) -> TSRChain:
        hub = np.eye(4)
        hub[:3, 3] = [0.35, -0.2, 0.6]
        to_handle = np.eye(4)
        to_handle[:3, :3] = np.diag([1.0, -1.0, -1.0])  # flange pointing down
        to_handle[:3, 3] = [0.12, 0.0, 0.0]
        hinge = TSR(T0_w=hub, Tw_e=to_handle, Bw=[[0, 0]] * 5 + [[lo, hi]])
        spin = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=[[0, 0]] * 5 + [[-np.pi, np.pi]])
        return TSRChain(TSRs=[hinge, spin])

    turn = np.pi / 3
    cfg_c = CBiRRTConfig(**{**base, "step_size": 0.1, "edge_resolution": 0.05, "num_tree_roots": 20, "timeout": 120.0})
    planner_c = CBiRRT(robot, ik, checker_a, cfg_c)
    out.append(
        {
            "name": names[2][0],
            "description": names[2][1],
            "seed": names[2][2],
            "config": cfg_c,
            "problem": PlanningProblem(
                space=planner_c.space,
                start=_tsr_set(planner_c, crank(0.0, 0.0)),
                goal=_tsr_set(planner_c, crank(turn, turn)),
                path_constraint=_tsr_set(planner_c, crank(-0.02, turn + 0.02)),
                validator=checker_a,
            ),
            "spec": {
                "robot": "menagerie ur5e + robotiq_2f85",
                "ssik": importlib.metadata.version("ssik"),
                "mujoco": mujoco.__version__,
                "solver_name": arm.solver_name,
                "scene_mjb_sha256": scene.provenance["mjb_sha256"],
                "snapshot_sha256": snap_a.sha256,
                "start": "crank chain at 0",
                "goal": "crank chain at 60 degrees",
                "path_constraint": "crank chain within [-0.02, 60 degrees + 0.02]",
            },
            "planner": planner_c,
        }
    )
    return out


def ur5e_cases(base: dict[str, Any]) -> list[dict[str, Any]]:
    """The v1.6.0 end-to-end case (#91): a UR5e with a finite start, an AnyOf of two goal TSRs by volume,
    and a path TSR, planned through SSIK. Skipped, with a notice, when ssik is not installed."""
    try:
        import ssik

        from sscbirrt.backends.ssik import SSIKRobotModel, SSIKSolver
    except ImportError:
        print("note: ssik is not installed; the UR5e cases are omitted from this run", file=sys.stderr)
        return []

    manipulator = ssik.Manipulator.from_prebuilt("ur5e")
    ik = SSIKSolver(manipulator)
    robot = SSIKRobotModel(ik)
    cfg = CBiRRTConfig(**{**base, "num_tree_roots": 20})
    planner = CBiRRT(robot, ik, NoCollision(), cfg)
    q0 = np.array([0.0, -1.2, 1.0, -1.4, -1.57, 0.0])
    qa = np.array([0.8, -1.0, 0.8, -1.3, -1.57, 0.4])
    qb = np.array([-0.9, -1.4, 1.2, -1.2, -1.57, -0.3])
    box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])

    def goal_set(q):
        return _tsr_set(planner, TSR(T0_w=robot.forward_kinematics(q), Tw_e=np.eye(4), Bw=box))

    a, b = goal_set(qa), goal_set(qb)
    above = _tsr_set(
        planner,
        TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([[-1.2, 1.2], [-1.2, 1.2], [0.05, 1.5], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]),
        ),
    )
    out: list[dict[str, Any]] = []
    out.append(
        {
            "name": "ur5e_tsr_goal_union_with_path_tsr",
            "description": "UR5e through SSIK: finite start, AnyOf of two grasp TSRs by volume, a workspace path TSR",
            "seed": 12,
            "config": cfg,
            "problem": PlanningProblem(
                space=planner.space,
                start=_finite(planner, [q0]),
                goal=AnyOf([a, b], weights=tsr_weights([a, b])),
                validator=planner.collision,
                path_constraint=above,
            ),
            "spec": {
                "robot": "ssik prebuilt ur5e",
                "ssik": importlib.metadata.version("ssik"),
                "solver_name": manipulator.solver_name,
                "start": [q0.tolist()],
                "goal": "AnyOf([TSR@fk(qa), TSR@fk(qb)])",
                "path_constraint": "TSR workspace box, z in [0.05, 1.5], rotation free",
            },
            "planner": planner,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Independent validation
# ---------------------------------------------------------------------------


def validate(problem: PlanningProblem, config: CBiRRTConfig, path: list[np.ndarray], planned=None) -> dict[str, Any]:
    """Check a path against the problem with its own operations and the public validator (#230).

    ``path_valid`` is ``Validator.for_problem(problem, config).validate_path(path)``, the Python reference,
    whichever backend planned. ``report_matches_planning`` holds the planner's own report (``planned``, from
    ``PlanResult.validation``) equal to it: the same verdict, policy, and number of states checked.
    """
    from sscbirrt.validation import Validator  # here, so the benchmark's older installs can still import this tool

    space = problem.space

    def admissible(q):
        if not space.contains(q):
            return False
        if not problem.validator.is_valid(q):
            return False
        return problem.path_constraint is None or problem.path_constraint.contains(q)

    report = Validator.for_problem(problem, config, backend="python").validate_path(path)
    max_raw_step = max(
        (float(np.abs(np.asarray(b) - np.asarray(a)).max()) for a, b in zip(path[:-1], path[1:])), default=0.0
    )

    def verdict(r):
        return None if r is None else (r.subject, r.valid, r.intervals, r.checks, r.policy, r.witness)

    return {
        "all_in_space": all(space.contains(q) for q in path),
        "first_in_start_set": bool(problem.start.contains(path[0])),
        "last_in_goal_set": bool(problem.goal.contains(path[-1])),
        "all_admissible": all(admissible(q) for q in path),
        "path_valid": report.valid,
        "report_matches_planning": verdict(planned) == verdict(report),
        "raw_steps_within_step_size": max_raw_step <= config.step_size + 1e-9,
        "max_raw_step": max_raw_step,
        "waypoints": len(path),
        "states_checked": report.checks,
    }


def failure_category(reason: str | None) -> str | None:
    if reason is None:
        return None
    for prefix, category in (("Timeout", "timeout"), ("Aborted", "aborted"), ("Max iterations", "max_iterations")):
        if reason.startswith(prefix):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Generation and comparison
# ---------------------------------------------------------------------------


def versions() -> dict[str, str]:
    def v(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "sscbirrt": v("sscbirrt"),
        "sstsr": v("sstsr"),
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def run_case(case: dict[str, Any], backend: str = "python") -> dict[str, Any]:
    """Run one case. With backend="native", a case the native core cannot lower is recorded as unsupported."""
    if "skipped" in case:
        return {
            "name": case["name"],
            "description": case["description"],
            "seed": case["seed"],
            "spec": None,
            "status": "skipped",
            "skipped_reason": case["skipped"],
            "failure_category": None,
            "start_source": [],
            "goal_source": [],
            "iterations": 0,
            "path": None,
            "validation": None,
        }
    planner: CBiRRT = case["planner"]
    planner.backend = "python" if backend == "native" else backend  # the native branch below drives the core itself
    if backend == "native":
        from sscbirrt.backends import native

        try:
            lowered = native.lower(case["problem"], case["config"])
        except native.NativeUnsupported as e:
            return {
                "name": case["name"],
                "description": case["description"],
                "seed": case["seed"],
                "spec": case["spec"],
                "status": "unsupported",
                "failure_category": None,
                "start_source": [],
                "goal_source": [],
                "iterations": 0,
                "path": None,
                "validation": None,
                "unsupported_reasons": list(e.reasons),
            }
        result, python_calls = _native_solve_counting_python_calls(lowered, case)
    else:
        result = planner.solve(case["problem"], seed=case["seed"])
        python_calls = None
    record = {
        "name": case["name"],
        "description": case["description"],
        "seed": case["seed"],
        "spec": case["spec"],
        "status": "success" if result.success else "failure",
        "failure_category": failure_category(result.failure_reason),
        "start_source": list(result.start_source),
        "goal_source": list(result.goal_source),
        "iterations": result.iterations,
        "path": [q.tolist() for q in result.path] if result.success else None,
        "validation": (
            validate(case["problem"], case["config"], result.path, result.validation) if result.success else None
        ),
    }
    if python_calls is not None:
        record["native_python_calls"] = python_calls  # the no-callback proof: Python functions entered during the solve
    if backend == "auto":
        record["backend"] = result.backend  # which implementation the default selection chose, and why not native
        record["backend_reasons"] = list(result.backend_reasons)
    return record


def _native_solve_counting_python_calls(lowered, case):
    """Run the bare native solve under a profile hook that counts Python function entries (#91).

    abort_fn cases use the wrapper (the token is polled from Python by design) and record no count.
    """
    from sscbirrt.backends import native

    if case["config"].abort_fn is not None:
        return native.solve(lowered, case["seed"], case["config"].abort_fn), None
    planner = native._native.Planner(lowered.config)
    calls: list[str] = []

    def profiler(frame, event, arg):
        if event == "call":
            calls.append(frame.f_code.co_name)

    sys.setprofile(profiler)
    try:
        r = planner.solve(lowered.problem, case["seed"], None, True)
    finally:
        sys.setprofile(None)
    return native.convert(r), len(calls)


def generate(backend: str = "python") -> dict[str, Any]:
    return {
        "artifact": f"sscbirrt {backend} behavior on the reference matrix",
        "issue": "https://github.com/personalrobotics/sscbirrt/issues/94",
        "backend": backend,
        "versions": versions(),
        "cases": [run_case(c, backend) for c in cases()],
    }


def uniquely_rooted(case_name: str) -> bool:
    """Whether both roles of the named case have exactly one admissible root, so provenance is comparable."""
    for c in cases():
        if c["name"] == case_name:
            return len(explicit_samples(c["problem"].start)) == 1 and len(explicit_samples(c["problem"].goal)) == 1
    raise KeyError(case_name)


def parity_mismatches(python_artifact: dict[str, Any], native_artifact: dict[str, Any]) -> list[str]:
    """Where a native run disagrees with the Python artifact on the semantic view, for the cases native supports.

    Provenance is compared only where the reached root is unique (docs/native-design.md, parity note).
    """
    by_name = {c["name"]: c for c in semantic_view(python_artifact)}
    out = []
    for nat in semantic_view(native_artifact):
        if nat["status"] in ("unsupported", "skipped"):
            continue
        py = by_name[nat["name"]]
        if py["status"] == "skipped":
            continue
        keys = ["status", "failure_category", "validation"]
        if uniquely_rooted(nat["name"]):
            keys += ["start_source", "goal_source"]
        for k in keys:
            if nat[k] != py[k]:
                out.append(f"{nat['name']}.{k}: python={py[k]!r} native={nat[k]!r}")
    return out


def semantic_view(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """The comparable part: no paths, no iteration counts, no versions."""
    view = []
    for case in artifact["cases"]:
        entry = {k: case[k] for k in ("name",) + SEMANTIC_KEYS}
        if entry["validation"] is not None:
            entry["validation"] = {
                k: v for k, v in entry["validation"].items() if k not in ("max_raw_step", "waypoints", "states_checked")
            }
        view.append(entry)
    return view


def semantic_mismatches(stored: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Where a fresh Python run disagrees with the stored artifact. A case skipped in either run (an
    environment without the Menagerie, for instance) is not compared; a case missing on one side is."""
    a = {c["name"]: c for c in semantic_view(stored)}
    b = {c["name"]: c for c in semantic_view(fresh)}
    out = [f"{n}: missing from the {'fresh' if n in a else 'stored'} run" for n in sorted(set(a) ^ set(b))]
    for name in sorted(set(a) & set(b)):
        if a[name]["status"] == "skipped" or b[name]["status"] == "skipped":
            continue
        if a[name] != b[name]:
            out.append(f"{name}:\n    stored: {a[name]}\n    fresh:  {b[name]}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare a fresh run's semantics with the stored artifact")
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument(
        "--backend",
        choices=["python", "native", "auto"],
        default="python",
        help="which implementation to run; with --check, native and auto are compared with the stored Python artifact",
    )
    args = parser.parse_args(argv)

    fresh = generate(args.backend)
    if args.check and args.backend == "auto":
        stored = json.loads(ARTIFACT.read_text())
        ran = [c for c in fresh["cases"] if c["status"] != "skipped"]
        # Cases the default ran natively are held to the parity rules (provenance only where uniquely rooted);
        # cases it ran in Python must reproduce the stored Python artifact exactly.
        by_python = {c["name"] for c in ran if c["backend"] == "python"}
        bad = parity_mismatches(stored, {"cases": [c for c in ran if c["backend"] == "native"]})
        bad += semantic_mismatches(
            {"cases": [c for c in stored["cases"] if c["name"] in by_python]},
            {"cases": [c for c in ran if c["backend"] == "python"]},
        )
        # Default selection must be explicit: Python only with a stated reason, native whenever it could.
        bad += [
            f"{c['name']}: Python was selected with no reason"
            for c in ran
            if c["backend"] == "python" and not c["backend_reasons"]
        ]
        bad += [
            f"{c['name']}: reasons recorded but the native backend ran"
            for c in ran
            if c["backend"] == "native" and c["backend_reasons"]
        ]
        if bad:
            print("MISMATCH: default backend selection disagrees with the Python artifact:", file=sys.stderr)
            for line in bad:
                print(f"  {line}", file=sys.stderr)
            return 1
        native_names = [c["name"] for c in ran if c["backend"] == "native"]
        python_names = {c["name"]: c["backend_reasons"] for c in ran if c["backend"] == "python"}
        print(
            f"default selection matches the Python artifact on {len(ran)} cases; "
            f"native on {len(native_names)}: {', '.join(native_names)}"
        )
        for name, why in python_names.items():
            print(f"  Python for {name}: {'; '.join(why)}")
        return 0
    if args.check and args.backend == "native":
        stored = json.loads(ARTIFACT.read_text())
        mismatches = parity_mismatches(stored, fresh)
        supported = [c["name"] for c in fresh["cases"] if c["status"] not in ("unsupported", "skipped")]
        if mismatches:
            print("PARITY MISMATCH between the native core and the Python artifact:", file=sys.stderr)
            for m in mismatches:
                print(f"  {m}", file=sys.stderr)
            return 1
        print(f"native matches the Python artifact on {len(supported)} supported cases: {', '.join(supported)}")
        return 0
    if args.check:
        stored = json.loads(args.output.read_text())
        bad = semantic_mismatches(stored, fresh)
        if bad:
            print("MISMATCH: the Python planner's semantics on the reference matrix changed", file=sys.stderr)
            for line in bad:
                print(f"  {line}", file=sys.stderr)
            return 1
        exact = json.dumps(stored, sort_keys=True) == json.dumps(fresh, sort_keys=True)
        print(f"semantics match ({len(fresh['cases'])} cases); bit-for-bit: {exact} (versions {fresh['versions']})")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({len(fresh['cases'])} cases, versions {fresh['versions']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
