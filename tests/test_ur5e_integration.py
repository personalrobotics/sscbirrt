# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Integration check on a real arm: constrained planning with the UR5e in MuJoCo, SSIK IK.

Skipped unless mujoco and ssik are installed and MUJOCO_MENAGERIE_PATH points
at a clone of google-deepmind/mujoco_menagerie. This is the check that caught
the root-sampling regression in #33: the start region below has several IK
branches per pose of which only some are collision-free, and never the first.
With SSIK every in-limit winding is returned as well (#36, #63).
"""

import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
ssik = pytest.importorskip("ssik")
MENAGERIE = os.environ.get("MUJOCO_MENAGERIE_PATH")
if not MENAGERIE or not (Path(MENAGERIE) / "universal_robots_ur5e").exists():
    pytest.skip("MUJOCO_MENAGERIE_PATH not set to a mujoco_menagerie clone", allow_module_level=True)

from tsr import TSR  # noqa: E402

from sscbirrt import CBiRRT, CBiRRTConfig  # noqa: E402
from sscbirrt.backends.mujoco import MuJoCoCollisionChecker, MuJoCoRobotModel, site_offset_in_body  # noqa: E402
from sscbirrt.backends.ssik import SSIKSolver  # noqa: E402
from sscbirrt.space import JointSpace  # noqa: E402
from sscbirrt.testing.ur5e import create_grasp_tsr, create_scene  # noqa: E402
from sscbirrt.tsr_set import TSRConfigurationSet  # noqa: E402

JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@pytest.fixture(scope="module")
def ur5e():
    model = create_scene(Path(MENAGERIE))
    data = mujoco.MjData(model)
    robot = MuJoCoRobotModel(model, data, "attachment_site", JOINTS)
    collision = MuJoCoCollisionChecker(model, data, JOINTS)
    arm = ssik.Manipulator.from_mjcf(
        Path(MENAGERIE) / "universal_robots_ur5e" / "ur5e.xml", base="world", ee="wrist_3_link"
    )
    ik = SSIKSolver(arm, T_ee=site_offset_in_body(model, "attachment_site"))
    # The UR5e's joints are bounded intervals (±2π, elbow ±π), not continuous circles (#35).
    config = CBiRRTConfig(max_iterations=5000, step_size=0.2, sample_draws=100)
    return robot, collision, ik, CBiRRT(robot, ik, collision, config)


def gripper_down_everywhere():
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T[:3, 3] = [0.0, 0.0, 0.6]
    bounds = np.array([[-0.9, 0.9], [-0.9, 0.9], [-0.3, 0.5], [-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]])
    return TSR(T0_w=T, Tw_e=np.eye(4), Bw=bounds)


def test_ssik_and_mujoco_forward_kinematics_agree(ur5e):
    """The frame contract: SSIK's FK (with T_ee) equals MuJoCo's attachment-site FK."""
    robot, _, ik, _ = ur5e
    rng = np.random.default_rng(0)
    for _ in range(25):
        q = rng.uniform(-np.pi, np.pi, 6)
        assert np.allclose(ik.fk(q), robot.forward_kinematics(q), atol=1e-9)


def test_ik_solutions_reach_the_mujoco_pose_and_include_windings(ur5e):
    robot, _, ik, _ = ur5e
    q = np.array([0.1, -1.2, 1.0, -0.5, 0.3, 0.2])
    T = robot.forward_kinematics(q)
    sols = ik.solve(T)
    assert len(sols) > 8  # eight geometric branches, each with in-limit windings on the ±2π joints
    lo, hi = robot.joint_limits
    for s in sols:
        assert np.all((s >= lo) & (s <= hi))
        assert np.allclose(robot.forward_kinematics(s), T, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_constrained_transport_keeps_gripper_down(ur5e, seed):
    robot, collision, _, planner = ur5e
    start_tsr = create_grasp_tsr(np.array([0.55, -0.35, 0.47]))  # hard region: few collision-free branches
    goal_tsr = create_grasp_tsr(np.array([-0.30, 0.45, 0.47]))
    upright = gripper_down_everywhere()

    result = planner.plan(
        start_tsrs=[start_tsr], goal_tsrs=[goal_tsr], constraint_tsrs=[upright], seed=seed, return_details=True
    )
    assert result.success, result.failure_reason
    for q in result.path:
        assert collision.is_valid(q)
        assert upright.distance(robot.forward_kinematics(q))[0] <= planner.config.membership_tolerance
    assert start_tsr.distance(robot.forward_kinematics(result.path[0]))[0] <= planner.config.membership_tolerance
    assert goal_tsr.distance(robot.forward_kinematics(result.path[-1]))[0] <= planner.config.membership_tolerance
    # Executable as raw joint values: no waypoint outside the real limits, no jump larger than one step
    lo, hi = robot.joint_limits
    P = np.array(result.path)
    assert np.all((P >= lo) & (P <= hi))
    assert np.abs(np.diff(P, axis=0)).max() <= planner.config.step_size + 1e-9


def test_planner_keeps_a_collision_free_branch_when_others_collide(ur5e):
    """Root seeding must find the admissible IK branches even when the first ones collide."""
    robot, collision, ik, planner = ur5e
    start_tsr = create_grasp_tsr(np.array([0.55, -0.35, 0.47]))
    space = JointSpace(*robot.joint_limits)
    s = TSRConfigurationSet(start_tsr, robot, ik, space)
    rng = np.random.default_rng(0)
    mixed_draws = 0
    for _ in range(20):
        cands = s.sample(rng)
        if not cands:
            continue
        valid = [collision.is_valid(c.q) for c in cands]
        if any(valid) and not all(valid):
            mixed_draws += 1
    assert mixed_draws > 0  # the region really is adversarial: colliding and free candidates share a pose
    from sscbirrt import FiniteSet, PlanningProblem

    prob = PlanningProblem(space=planner.space, start=s, goal=FiniteSet([np.zeros(6)]), validator=collision)
    roots = planner._roots(prob, s, "Start")
    assert roots and all(collision.is_valid(r.q) for r in roots)


def test_unconstrained_baseline_violates_constraint(ur5e):
    """Sanity check that the constraint above is not vacuous for this start/goal pair.

    Whether one unconstrained plan tilts depends on the roots and detours its seed draws: about half do
    (16 of 30 seeds after #196). Five seeds were too few: on Linux all five stayed upright, while on macOS
    two of them tilted. So the claim is over up to 20 seeds, stopping at the first plan that tilts.
    """
    robot, _, _, planner = ur5e
    upright = gripper_down_everywhere()
    tilts = []
    for seed in range(20):
        result = planner.plan(
            start_tsrs=[create_grasp_tsr(np.array([0.55, -0.35, 0.47]))],
            goal_tsrs=[create_grasp_tsr(np.array([-0.30, 0.45, 0.47]))],
            seed=seed,
            return_details=True,
        )
        assert result.success
        tilts.append(max(upright.distance(robot.forward_kinematics(q))[0] for q in result.path))
        if tilts[-1] > 1.0:
            break
    assert max(tilts) > 1.0, tilts


def test_projection_near_a_winding_does_not_take_a_full_turn(ur5e):
    """The #36 regression: a configuration near one winding must project to that winding, not the principal one."""
    robot, _, ik, planner = ur5e
    q = np.array([0.1, -1.2, 1.0, -0.5, 0.3, 0.2 - 2 * np.pi])  # wrist 3 on its other in-limit winding
    assert planner.space.contains(q)
    box = np.array([[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]])
    T = robot.forward_kinematics(q)
    tsr = TSR(T0_w=T, Tw_e=np.eye(4), Bw=box)
    s = TSRConfigurationSet(tsr, robot, ik, planner.space)
    q_off = q + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_off[1] += 0.3  # push the pose out of the box so projection has to solve IK
    assert not s.contains(q_off)
    q_proj = s.project(q, q_off)
    assert q_proj is not None and s.contains(q_proj)
    assert planner.space.distance(q_proj, q) < 1.0  # stayed on the nearby winding, not 2π away


def test_door_handle_chain_goal_on_the_ur5e(ur5e):
    """A two-TSR chain goal: a door hinge with limited swing, then the handle with a top-down grasp (#7)."""
    from tsr import TSRChain

    robot, collision, _, planner = ur5e
    # Hinge frame near the table edge; the door may swing ±0.4 rad about z
    T_hinge = np.eye(4)
    T_hinge[:3, 3] = [0.35, -0.25, 0.60]
    hinge = TSR(T0_w=T_hinge, Tw_e=np.eye(4), Bw=np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [-0.4, 0.4]]))
    # Handle 0.25 m along the door's x, grasped from above (gripper z down), any yaw about the handle
    T_grasp = np.eye(4)
    T_grasp[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    handle = TSR(
        T0_w=np.eye(4), Tw_e=T_grasp, Bw=np.array([[0.25, 0.25], [0, 0], [0.05, 0.10], [0, 0], [0, 0], [-np.pi, np.pi]])
    )
    chain = TSRChain(TSRs=[hinge, handle])

    home = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
    result = planner.plan(start=home, goal_tsrs=[chain], seed=0, return_details=True)
    assert result.success, result.failure_reason
    T_end = robot.forward_kinematics(result.path[-1])
    assert chain.contains(T_end)
    # The handle is on the door's swing arc: 0.25 m from the hinge axis in the xy plane
    assert np.linalg.norm(T_end[:2, 3] - T_hinge[:2, 3]) == pytest.approx(0.25, abs=2e-3)
    assert all(collision.is_valid(q) for q in result.path)


class TestMuJoCoDifferentialIK:
    """MuJoCo's differential IK on the full UR5e, the non-SSIK path, exercised even where SSIK is installed (#65).

    Moved from the removed example tests (#166)."""

    HOME = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

    def _world(self):
        from sscbirrt.backends.mujoco import MuJoCoIKSolver

        model = create_scene(Path(MENAGERIE))
        data = mujoco.MjData(model)
        robot = MuJoCoRobotModel(model, data, "attachment_site", JOINTS)
        collision = MuJoCoCollisionChecker(model, data, JOINTS)
        ik = MuJoCoIKSolver(model, data, "attachment_site", JOINTS, collision_checker=collision, seed=0)
        return robot, collision, ik

    def test_solves_a_nontrivial_pose_from_a_seed(self):
        robot, _, ik = self._world()
        target = robot.forward_kinematics(self.HOME + np.array([0.3, 0.2, -0.2, 0.1, 0.1, 0.4]))
        sols = ik.solve(target, q_init=self.HOME)  # requires iterative updates from the seed
        assert sols
        assert np.linalg.norm(robot.forward_kinematics(sols[0])[:3, 3] - target[:3, 3]) < 5e-3

    def test_plans_to_a_grasp_region(self):
        robot, collision, ik = self._world()
        cfg = CBiRRTConfig(timeout=60.0, goal_sample_probability=0.15, sample_draws=100)
        planner = CBiRRT(robot, ik, collision, cfg)
        result = planner.plan(start=self.HOME, goal_tsrs=[create_grasp_tsr(np.array([0.45, 0.15, 0.47]))], seed=0,
                              return_details=True)  # fmt: skip
        assert result.success, result.failure_reason
        assert all(collision.is_valid(q) for q in result.path)


def test_ssik_windings_share_a_key_and_the_hinge_scene_declares_invariance(ur5e):
    """#200 on the real stack: SSIK's windings of one arm pose share a key, and both MuJoCo checkers declare that
    full turns cannot change their verdict (every UR5e joint is a hinge)."""
    from sscbirrt.backends.native_mujoco import NativeCollisionChecker, NativeScene, Snapshot, available

    robot, collision, ik, planner = ur5e
    s = TSRConfigurationSet(create_grasp_tsr(np.array([0.45, 0.15, 0.47])), robot, ik, planner.space)
    cands = s.sample(np.random.default_rng(0))
    keys = {c.key for c in cands}
    assert len(cands) > len(keys) > 1  # many windings, a few physical configurations
    for c in cands:
        twin = next(d for d in cands if d.key == c.key)
        assert np.allclose(robot.forward_kinematics(c.q), robot.forward_kinematics(twin.q), atol=1e-9)
    assert collision.full_turn_invariant
    if available():
        scene = NativeScene.from_model(collision.model, JOINTS)
        checker = NativeCollisionChecker(scene, Snapshot.capture(scene, collision.data))
        assert checker.full_turn_invariant
