# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Every geometric operation in solve(problem) uses problem.space, not the planner's default (#45)."""

import inspect

import numpy as np
import pytest

from sscbirrt import CBiRRT, CBiRRTConfig, PlanningProblem
from sscbirrt.sets import FiniteSet
from sscbirrt.space import JointSpace
from sscbirrt.tree import RRTree
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel

LIMITS = (np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi]))


def make_planner(angular, smooth=False):
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    cfg = CBiRRTConfig(continuous_joints=angular, smooth_path=smooth)
    return CBiRRT(robot, MockIKSolver(robot, collision), collision, cfg)


def problem_with(planner, space, start, goal):
    return PlanningProblem(space=space, start=FiniteSet([start]), goal=FiniteSet([goal]), validator=planner.collision)


class TestNearestUsesQuerySpace:
    """Tree: nodes at (3, 0) and (0, 0). Target (-3, 0).

    Under a wrapping joint 0 the nearest node is (3, 0), 0.28 rad away the short way.
    Under a bounded joint 0 the nearest node is (0, 0), 3 rad away.
    """

    tree_nodes = [np.array([3.0, 0.0]), np.array([0.0, 0.0])]
    target = np.array([-3.0, 0.0])

    def tree(self):
        t = RRTree(self.tree_nodes[0])
        t.add_node(self.tree_nodes[1], 0)
        return t

    def test_problem_space_wraps_but_planner_default_does_not(self):
        planner = make_planner(None)
        wrapping = JointSpace(*LIMITS, continuous_joints=(True, False))
        assert planner._nearest_node(wrapping, self.tree(), self.target) == 0

    def test_planner_default_wraps_but_problem_space_does_not(self):
        planner = make_planner((True, False))
        bounded = JointSpace(*LIMITS)
        assert planner._nearest_node(bounded, self.tree(), self.target) == 1

    def test_grow_extends_from_the_query_space_nearest(self):
        """With a wrapping query space, growth toward (-3, 0) starts from (3, 0) and crosses the seam."""
        planner = make_planner(None)  # planner default is bounded
        wrapping = JointSpace(*LIMITS, continuous_joints=(True, False))
        prob = problem_with(planner, wrapping, self.tree_nodes[0], self.target)
        tree = self.tree()
        idx, reached = planner._grow(prob, tree, self.target)
        assert reached
        # The new nodes descend from node 0 (3, 0), not node 1 (0, 0)
        assert tree.nodes[idx].parent is not None
        root = idx
        while tree.nodes[root].parent is not None:
            root = tree.nodes[root].parent
        assert root == 0

    def test_solve_with_wrapping_query_space_takes_the_short_way(self):
        # RRT-Connect in free space connects through a random sample on the first
        # iteration, so the raw path is long under any metric; shortcutting under
        # the query metric is what exposes the short way across the seam.
        planner = make_planner(None, smooth=True)
        wrapping = JointSpace(*LIMITS, continuous_joints=(True, False))
        start, goal = np.array([3.0, 0.0]), np.array([-3.0, 0.0])
        result = planner.solve(problem_with(planner, wrapping, start, goal), seed=0)
        assert result.success
        length = sum(wrapping.distance(a, b) for a, b in zip(result.path[:-1], result.path[1:]))
        assert length < 1.0  # short way is 0.28; the long way is 6

    def test_solve_with_bounded_query_space_takes_the_long_way(self):
        planner = make_planner((True, False))  # planner default wraps
        bounded = JointSpace(*LIMITS)
        start, goal = np.array([3.0, 0.0]), np.array([-3.0, 0.0])
        result = planner.solve(problem_with(planner, bounded, start, goal), seed=0)
        assert result.success
        length = sum(bounded.distance(a, b) for a, b in zip(result.path[:-1], result.path[1:]))
        assert length > 5.0
        assert all(bounded.contains(q) for q in result.path)


class TestNoSelfSpaceInSearch:
    @pytest.mark.parametrize(
        "method",
        [
            "solve",
            "_roots",
            "_draw_roots",
            "_admissible",
            "_grow",
            "_extend_along_edge",
            "_nearest_node",
            "_smooth_path",
            "_try_shortcut",
            "_extract_path",
        ],
    )
    def test_search_methods_do_not_reference_planner_space(self, method):
        src = inspect.getsource(getattr(CBiRRT, method))
        assert "self.space" not in src, f"{method} uses self.space"
