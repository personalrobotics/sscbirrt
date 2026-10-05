# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Each tolerance has one meaning; this file exercises them independently."""

import numpy as np
import pytest
from tsr import TSR

from sscbirrt import CBiRRT, CBiRRTConfig, PlanningProblem
from sscbirrt.legacy import legacy_problem
from sscbirrt.sets import AllOf, FiniteSet
from sscbirrt.tree import RRTree
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel

BOX = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])


def make_planner(ik=None, **cfg):
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    return CBiRRT(robot, ik or MockIKSolver(robot, collision), collision, CBiRRTConfig(**cfg))


class HalfwayIK:
    """An IK solver that only moves halfway toward each solution from q_init.

    Makes projection genuinely iterative, so progress tolerances matter.
    """

    def __init__(self, inner):
        self.inner = inner

    def solve(self, pose, q_init=None):
        sols = self.inner.solve(pose, q_init)
        if q_init is None:
            return sols
        return [q_init + 0.5 * (s - q_init) for s in sols]


def unconstrained(planner):
    q = np.zeros(2)
    return PlanningProblem(space=planner.space, start=FiniteSet([q]), goal=FiniteSet([q]), validator=planner.collision)


def lowered(planner, **kw):
    return legacy_problem(
        planner.robot,
        planner.ik,
        planner.collision,
        planner.space,
        planner.config,
        kw.get("start"),
        kw.get("goal"),
        kw.get("start_tsrs"),
        kw.get("goal_tsrs"),
        kw.get("constraint_tsrs"),
    )


class TestConnectionTolerance:
    def test_reached_within_connection_tolerance_connects_exactly(self):
        """Within tolerance the target is connected by one validated edge, not merely declared reached."""
        planner = make_planner(connection_tolerance=0.1)
        tree = RRTree(np.zeros(2))
        target = np.array([0.05, 0.0])
        idx, reached = planner._grow(unconstrained(planner), tree, target)
        assert reached and len(tree) == 2
        assert np.array_equal(tree.nodes[idx].config, target)

    def test_tight_connection_tolerance_grows(self):
        planner = make_planner(connection_tolerance=1e-3)
        tree = RRTree(np.zeros(2))
        idx, reached = planner._grow(unconstrained(planner), tree, np.array([0.05, 0.0]))
        assert reached and len(tree) > 1

    def test_connection_tolerance_does_not_affect_membership(self):
        planner = make_planner(connection_tolerance=0.5, membership_tolerance=1e-3)
        problem = lowered(planner, start=[np.zeros(2)], goal=[np.zeros(2)], constraint_tsrs=[TSR(Bw=BOX)])
        assert problem.path_constraint.tolerance == 1e-3


class TestEdgeResolution:
    def test_none_resolution_is_step_size(self):
        planner = make_planner(step_size=0.3, edge_resolution=None)
        tree = RRTree(np.zeros(2))
        planner._extend_along_edge(unconstrained(planner), tree, 0, np.array([0.3, 0.0]))
        assert len(tree) == 2  # one node added: the endpoint

    def test_finer_resolution_adds_intermediate_nodes(self):
        planner = make_planner(step_size=0.3, edge_resolution=0.1)
        tree = RRTree(np.zeros(2))
        planner._extend_along_edge(unconstrained(planner), tree, 0, np.array([0.3, 0.0]))
        assert len(tree) == 4  # 0.1, 0.2, 0.3

    def test_finer_resolution_catches_thin_obstacle(self):
        class Wall:
            """Invalid only in a thin slab that a coarse check steps over."""

            def is_valid(self, q):
                return not (0.14 < q[0] < 0.16)

        q0, q1 = np.zeros(2), np.array([0.3, 0.0])

        coarse = make_planner(step_size=0.3, edge_resolution=None)
        tree = RRTree(q0)
        problem = PlanningProblem(space=coarse.space, start=FiniteSet([q0]), goal=FiniteSet([q1]), validator=Wall())
        _, ok = coarse._extend_along_edge(problem, tree, 0, q1)
        assert ok  # steps straight over the wall

        fine = make_planner(step_size=0.3, edge_resolution=0.01)
        tree = RRTree(q0)
        problem = PlanningProblem(space=fine.space, start=FiniteSet([q0]), goal=FiniteSet([q1]), validator=Wall())
        _, ok = fine._extend_along_edge(problem, tree, 0, q1)
        assert not ok

    def test_non_positive_resolution_rejected(self):
        with pytest.raises(ValueError):
            CBiRRTConfig(edge_resolution=0.0)


class TestMembershipTolerance:
    def test_lowering_passes_membership_tolerance_to_sets(self):
        planner = make_planner(membership_tolerance=0.02)
        problem = lowered(
            planner, start=[np.zeros(2)], goal_tsrs=[TSR(Bw=BOX)], constraint_tsrs=[TSR(Bw=BOX), TSR(Bw=BOX)]
        )
        assert problem.path_constraint.children[0].tolerance == 0.02
        assert problem.goal.children[0].tolerance == 0.02
        assert problem.start.tolerance == 0.02  # FiniteSet

    def test_membership_tolerance_widens_containment(self):
        robot = MockRobotModel()
        T0_w = np.eye(4)
        T0_w[0, 3] = 2.0  # EE at (2, 0) for q = 0; the box is ±5cm
        q_edge = np.array([0.0, 0.0])
        q_edge[0] = 0.0
        tsr = TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=BOX)
        # A configuration 3cm outside the box along y
        q = np.array([0.04, 0.0])
        dist, _ = tsr.distance(robot.forward_kinematics(q))
        assert 0.02 < dist < 0.05
        tight = lowered(make_planner(membership_tolerance=1e-3), start=[q_edge], goal=[q_edge], constraint_tsrs=[tsr])
        loose = lowered(make_planner(membership_tolerance=0.05), start=[q_edge], goal=[q_edge], constraint_tsrs=[tsr])
        assert not tight.path_constraint.contains(q)
        assert loose.path_constraint.contains(q)


class TestProjectionProgressTolerance:
    def test_lowering_passes_projection_progress_tolerance(self):
        planner = make_planner(projection_progress_tolerance=0.5, progress_tolerance=1e-6)
        problem = lowered(planner, start=[np.zeros(2)], goal=[np.zeros(2)], constraint_tsrs=[TSR(Bw=BOX), TSR(Bw=BOX)])
        assert isinstance(problem.path_constraint, AllOf)
        assert problem.path_constraint.projection.progress_tolerance == 0.5
        assert problem.path_constraint.children[0].progress_tolerance == 0.5

    def test_projection_progress_is_independent_of_growth_progress(self):
        T0_w = np.eye(4)
        T0_w[0, 3], T0_w[1, 3] = 1.2, 0.8
        tsr = TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=BOX)
        q = np.array([0.3, 0.9])  # ~0.38 outside
        robot = MockRobotModel()
        halfway = HalfwayIK(MockIKSolver(robot, MockCollisionChecker()))

        # Halving the violation each iteration converges within the 50-iteration budget
        default = lowered(make_planner(ik=halfway), start=[q], goal=[q], constraint_tsrs=[tsr])
        assert default.path_constraint.contains(default.path_constraint.project(q, q))

        # Demanding more progress per iteration than halving can deliver makes projection give up
        strict = lowered(
            make_planner(ik=halfway, projection_progress_tolerance=0.3), start=[q], goal=[q], constraint_tsrs=[tsr]
        )
        assert strict.path_constraint.project(q, q) is None

        # Growth progress tolerance does not affect projection
        growth = lowered(make_planner(ik=halfway, progress_tolerance=0.3), start=[q], goal=[q], constraint_tsrs=[tsr])
        assert growth.path_constraint.project(q, q) is not None


class TestDeprecatedAlias:
    def test_tsr_tolerance_sets_both_and_warns(self):
        with pytest.warns(DeprecationWarning, match="tsr_tolerance"):
            cfg = CBiRRTConfig(tsr_tolerance=0.02)
        assert cfg.membership_tolerance == 0.02
        assert cfg.connection_tolerance == 0.02
        with pytest.warns(DeprecationWarning, match="tsr_tolerance"):
            assert cfg.tsr_tolerance == 0.02

    def test_reading_alias_without_setting_it(self):
        cfg = CBiRRTConfig(membership_tolerance=0.007)
        with pytest.warns(DeprecationWarning, match="tsr_tolerance"):
            assert cfg.tsr_tolerance == 0.007

    def test_replace_keeps_the_replaced_value(self):
        """#176: the alias used to be a stored field mirroring membership_tolerance, so replace() copied the stale
        mirror back over the new value."""
        import dataclasses

        with pytest.warns(DeprecationWarning):
            cfg = CBiRRTConfig(tsr_tolerance=0.02)
        assert dataclasses.replace(cfg, membership_tolerance=0.005).membership_tolerance == 0.005
        assert dataclasses.replace(CBiRRTConfig(), membership_tolerance=0.005).membership_tolerance == 0.005

    def test_defaults_unchanged(self):
        cfg = CBiRRTConfig()
        assert cfg.membership_tolerance == 1e-3
        assert cfg.connection_tolerance == 1e-3
        assert cfg.edge_resolution == 0.05  # since 3.4 (#204); None (step_size) before
        assert cfg.progress_tolerance == 1e-6
        assert cfg.projection_progress_tolerance == 1e-6
