# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Plan for an arm in a MuJoCo world in one call.

    from sscbirrt.mujoco import Arm, plan

    arm = Arm(model, joints=["shoulder_pan_joint", ...], ee_site="grasp_site", mjcf="ur5e.xml")
    result = plan(model, data, arm, goal=grasp_regions)        # a PlanResult
    result = plan(model, data, arm, goal=q_place, constraint=upright, holding="can")

``start`` defaults to where the arm is now. ``start``, ``goal`` and ``constraint`` each accept a
configuration, a list of configurations, a ``tsr.TSR`` (or ``TSRChain``), a list of TSRs (a union for start
and goal, an intersection for a constraint), or any sscbirrt set. Collision checking uses the native MuJoCo
scene (a grasped object may touch its gripper); planning runs natively when every piece has a native form and
in Python otherwise (``result.backend``, ``result.backend_reasons``).

Requires ``sscbirrt[mujoco]``; ``mjcf=`` also needs ``sscbirrt[ssik]``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from os import PathLike
from typing import Any

import mujoco
import numpy as np
from tsr import TSR, TSRChain

from sscbirrt.backends.mujoco import MuJoCoRobotModel, _joint_ids, _site_id
from sscbirrt.backends.native_mujoco import NativeCollisionChecker, NativeScene, Snapshot, _NoIK
from sscbirrt.config import CBiRRTConfig
from sscbirrt.exceptions import NativeUnsupported
from sscbirrt.legacy import legacy_problem
from sscbirrt.planner import CBiRRT, PlanResult, as_configurations
from sscbirrt.sets import StateSet
from sscbirrt.smoothing import SmoothingOptions

__all__ = ["Arm", "plan"]


def _pose(xpos: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(xmat).reshape(3, 3)
    T[:3, 3] = xpos
    return T


class Arm:
    """The arm to plan for: its joints in order, the end-effector site, and (optionally) its IK.

    Args:
        model: The compiled ``mujoco.MjModel`` of the whole world.
        joints: The arm's joint names, in the order configurations use.
        ee_site: The site whose pose TSRs constrain (for a gripper, its grasp frame).
        mjcf: Path to the arm's own MJCF (for example Menagerie's ``ur5e.xml``). When given, an analytical
            SSIK solver is built from it, with the frames that relate it to ``model`` computed from ``model``
            itself and checked. Needed for TSR starts, goals and constraints unless ``ik`` is given.
        ik: Any ``IKSolver``, instead of ``mjcf``.
        ik_end_body: The name, in ``mjcf``, of the body that carries the last joint; default: its name in
            ``model`` (they differ when the arm was attached to the world with a name prefix).
        joint_limits: Optional ``(lower, upper)`` planning limits replacing the model's.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joints: Sequence[str],
        ee_site: str,
        *,
        mjcf: str | PathLike | None = None,
        ik: Any = None,
        ik_end_body: str | None = None,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        if isinstance(joints, str):
            raise TypeError("joints must be a list of joint names, not a single string")
        self.model = model
        self.joints = list(joints)
        self.ee_site = ee_site
        self.joint_limits = joint_limits
        _site_id(model, ee_site)
        self._qpos = np.array([model.jnt_qposadr[j] for j in _joint_ids(model, self.joints)], dtype=int)
        if ik is not None and mjcf is not None:
            raise ValueError("pass mjcf= (to build SSIK) or ik=, not both")
        self.ik = ik if ik is not None else (self._ssik(mjcf, ik_end_body) if mjcf is not None else None)

    @property
    def dof(self) -> int:
        return len(self.joints)

    def configuration(self, data: mujoco.MjData) -> np.ndarray:
        """The arm's configuration in ``data`` now."""
        return np.array(data.qpos[self._qpos], dtype=float)

    def robot_model(self, data: mujoco.MjData) -> MuJoCoRobotModel:
        return MuJoCoRobotModel(self.model, data, self.ee_site, self.joints, joint_limits=self.joint_limits)

    def _ssik(self, mjcf, ik_end_body: str | None):
        try:
            import ssik
        except ImportError:
            raise ImportError('Arm(mjcf=...) builds SSIK IK: pip install "sscbirrt[ssik]"') from None
        from sscbirrt.backends.ssik import SSIKSolver

        model = self.model
        end = model.body(int(model.jnt_bodyid[model.joint(self.joints[-1]).id])).name  # in this world
        end_in_mjcf = (
            ik_end_body or end
        )  # the same body in the arm's own file (names differ when attached with a prefix)
        try:
            arm = ssik.Manipulator.from_mjcf(str(mjcf), base="world", ee=end_in_mjcf)
        except Exception as e:
            raise ValueError(
                f"could not build SSIK from {mjcf} ending at body {end_in_mjcf!r} ({e}); if the arm was attached to "
                f"this world with a name prefix, pass ik_end_body= the body's name in {mjcf}"
            ) from e

        # Relate SSIK's frames to the world's from the model itself: T_base places SSIK's base where the arm
        # stands in this world, and T_ee carries SSIK's last body to the end-effector site.
        data = mujoco.MjData(model)

        def world_poses(q):
            data.qpos[self._qpos] = q
            mujoco.mj_kinematics(model, data)
            return _pose(data.body(end).xpos, data.body(end).xmat), _pose(
                data.site(self.ee_site).xpos, data.site(self.ee_site).xmat
            )

        q0 = np.zeros(self.dof)
        T_end, T_site = world_poses(q0)
        T_base = T_end @ np.linalg.inv(np.asarray(arm.fk(q0), dtype=float))
        solver = SSIKSolver(arm, T_base=T_base, T_ee=np.linalg.inv(T_end) @ T_site)

        q1 = np.random.default_rng(0).uniform(-1.0, 1.0, self.dof)  # check the relation away from where it was fit
        err = np.abs(solver.fk(q1) - world_poses(q1)[1]).max()
        if err > 1e-6:
            raise ValueError(
                f"the SSIK model from {mjcf} does not match this arm in the world (forward kinematics differ by "
                f"{err:.2g}); check that mjcf is this arm's file, that joints are its joints in chain order, and "
                f"that ik_end_body ({end_in_mjcf!r}) names its last body"
            )
        return solver


def _role(value, dof: int, name: str):
    """Split a start/goal argument into (configurations, TSRs, set): exactly one kind is used."""
    if isinstance(value, (TSR, TSRChain)):
        return None, [value], None
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (TSR, TSRChain)) for v in value):
        return None, list(value), None
    if not isinstance(value, (list, tuple, np.ndarray)) and isinstance(value, StateSet):
        return None, None, value
    if isinstance(value, (list, tuple)) and any(isinstance(v, (TSR, TSRChain)) for v in value):
        configs = [v for v in value if not isinstance(v, (TSR, TSRChain))]
        tsrs = [v for v in value if isinstance(v, (TSR, TSRChain))]
        return [c for q in configs for c in as_configurations(q, dof, name)], tsrs, None
    return as_configurations(value, dof, name), None, None


def _constraint(value):
    if value is None:
        return None, None
    if isinstance(value, (TSR, TSRChain)):
        return [value], None
    if isinstance(value, (list, tuple)) and all(isinstance(v, (TSR, TSRChain)) for v in value):
        return list(value), None
    if isinstance(value, StateSet):
        return None, value
    raise TypeError(f"constraint: expected a TSR, a list of TSRs, or a set; got {type(value).__name__}")


def _attachments(model, data, arm: Arm, holding) -> dict[str, tuple[str, np.ndarray]] | None:
    """``holding`` as the scene's attachments: each held body rigidly fixed to the gripper as it is now."""
    if holding is None:
        return None
    if isinstance(holding, dict):
        return holding
    names = [holding] if isinstance(holding, str) else list(holding)
    gripper = model.body(int(model.site_bodyid[model.site(arm.ee_site).id])).name
    mujoco.mj_kinematics(model, data)
    T_gripper = _pose(data.body(gripper).xpos, data.body(gripper).xmat)
    out = {}
    for name in names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f"holding: body '{name}' not found in the model")
        T_obj = _pose(data.body(name).xpos, data.body(name).xmat)
        out[name] = (gripper, np.linalg.inv(T_gripper) @ T_obj)
    return out


def plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: Arm,
    *,
    goal,
    start=None,
    constraint=None,
    holding: str | Sequence[str] | dict | None = None,
    config: CBiRRTConfig | None = None,
    seed: int | None = None,
    backend: str = "auto",
    extra_arm_bodies: Sequence[str] = (),
    smooth: "bool | SmoothingOptions | None" = None,
) -> PlanResult:
    """Plan a collision-free path for ``arm`` in the world ``data`` holds now.

    Args:
        goal: Where to go: a configuration, configurations, a TSR, TSRs (any of them), or a set.
        start: Where to start, in the same forms; default: the arm's configuration in ``data``.
        constraint: A TSR every waypoint must satisfy, several (all of them), or a set.
        holding: A body (or bodies) the gripper holds; it moves with the gripper and may touch it. The grasp
            is taken from the current poses in ``data``. A ``{body: (gripper_body, T_gripper_body)}`` dict is
            also accepted.
        config: Planner settings; default ``CBiRRTConfig()``.
        seed: Seed for a reproducible plan.
        backend: ``"auto"`` (native when possible, otherwise Python), ``"native"`` (raise if not), or
            ``"python"``.
        extra_arm_bodies: Bodies outside the joints' subtrees that move with the arm (see ``NativeScene``).
        smooth: True or ``SmoothingOptions`` to also blend the path's corners where admissible, for execution:
            ``result.smooth_path`` (``result.path`` stays the planned polyline). See ``CBiRRT.smooth``.

    Returns:
        A ``PlanResult``: ``success``, ``path``, ``failure_reason``, which member of each set was used
        (``start_index``, ``goal_index``), ``backend``, and provenance.
    """
    if not isinstance(arm, Arm):
        raise TypeError(f"arm must be an sscbirrt.mujoco.Arm, got {type(arm).__name__}")
    if arm.model is not model:
        raise ValueError("arm was built for a different model than the one passed to plan")
    start_cfgs, start_tsrs, start_set = _role(arm.configuration(data) if start is None else start, arm.dof, "start")
    goal_cfgs, goal_tsrs, goal_set = _role(goal, arm.dof, "goal")
    constraint_tsrs, constraint_set = _constraint(constraint)
    if arm.ik is None and (start_tsrs or goal_tsrs or constraint_tsrs):
        raise ValueError("TSR starts, goals and constraints need IK: build the Arm with mjcf= (SSIK) or ik=")

    robot = arm.robot_model(data)
    try:
        scene = NativeScene.from_model(model, arm.joints, tuple(extra_arm_bodies))
    except NativeUnsupported as e:
        raise NativeUnsupported(
            [f"sscbirrt.mujoco.plan checks collisions with the native MuJoCo scene ({r})" for r in e.reasons]
        ) from None
    checker = NativeCollisionChecker(scene, Snapshot.capture(scene, data, _attachments(model, data, arm, holding)))
    planner = CBiRRT(
        robot, arm.ik if arm.ik is not None else _NoIK(), checker, config or CBiRRTConfig(), backend=backend
    )

    problem = legacy_problem(
        robot, planner.ik, checker, planner.space, planner.config,
        start_cfgs, goal_cfgs, start_tsrs, goal_tsrs, constraint_tsrs,
    )  # fmt: skip
    overrides = {"start": start_set, "goal": goal_set, "path_constraint": constraint_set}
    problem = dataclasses.replace(problem, **{k: v for k, v in overrides.items() if v is not None})
    return planner.solve(problem, seed=seed, smooth=smooth)
