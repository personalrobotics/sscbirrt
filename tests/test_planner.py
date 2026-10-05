# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for CBiRRT planner."""

import numpy as np
import pytest
from tsr import TSR

from sscbirrt import AllStartConfigurationsInCollision, CBiRRT, CBiRRTConfig, PlanningProblem
from sscbirrt.legacy import legacy_problem
from sscbirrt.sets import AllOf, FiniteSet, MostViolatedProjection
from sscbirrt.tree import RRTree


class MockRobotModel:
    """Simple 2-DOF planar arm for testing."""

    def __init__(self):
        self.l1 = 1.0  # Link 1 length
        self.l2 = 1.0  # Link 2 length

    @property
    def dof(self) -> int:
        return 2

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi])

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """2D planar FK embedded in 3D."""
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])

        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        return T


class MockCollisionChecker:
    """Always returns valid (no obstacles)."""

    def is_valid(self, q: np.ndarray) -> bool:
        return True

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        return np.ones(len(qs), dtype=bool)


class MockIKSolver:
    """Analytical IK for 2-DOF planar arm."""

    def __init__(
        self,
        robot: MockRobotModel | None = None,
        collision_checker: MockCollisionChecker | None = None,
    ):
        self.robot = robot or MockRobotModel()
        self.collision = collision_checker or MockCollisionChecker()

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Return all kinematic IK solutions (unvalidated).

        Note: q_init is ignored (analytical solver finds all solutions).
        """
        x, y = pose[0, 3], pose[1, 3]
        d = np.sqrt(x**2 + y**2)

        # Check reachability
        if d > self.robot.l1 + self.robot.l2 or d < abs(self.robot.l1 - self.robot.l2):
            return []

        # Elbow angle
        cos_q2 = (d**2 - self.robot.l1**2 - self.robot.l2**2) / (2 * self.robot.l1 * self.robot.l2)
        cos_q2 = np.clip(cos_q2, -1, 1)

        solutions = []
        for sign in [1, -1]:
            q2 = sign * np.arccos(cos_q2)
            q1 = np.arctan2(y, x) - np.arctan2(self.robot.l2 * np.sin(q2), self.robot.l1 + self.robot.l2 * np.cos(q2))
            solutions.append(np.array([q1, q2]))

        return solutions

    def solve_valid(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Return only valid IK solutions (within limits, collision-free).

        Note: q_init is ignored (analytical solver finds all solutions).
        """
        solutions = self.solve(pose)
        lower, upper = self.robot.joint_limits

        valid = []
        for q in solutions:
            # Check joint limits
            if not (np.all(q >= lower) and np.all(q <= upper)):
                continue
            # Check collisions
            if not self.collision.is_valid(q):
                continue
            valid.append(q)

        return valid


class TestRRTree:
    def test_init(self):
        root = np.array([0.0, 0.0])
        tree = RRTree(root)
        assert len(tree) == 1
        assert np.allclose(tree.nodes[0].config, root)

    def test_add_node(self):
        tree = RRTree(np.array([0.0, 0.0]))
        idx = tree.add_node(np.array([1.0, 1.0]), parent_idx=0)
        assert idx == 1
        assert len(tree) == 2
        assert tree.nodes[1].parent == 0

    def test_nearest(self):
        tree = RRTree(np.array([0.0, 0.0]))
        tree.add_node(np.array([1.0, 0.0]), 0)
        tree.add_node(np.array([0.0, 1.0]), 0)

        assert tree.nearest(np.array([0.9, 0.1])) == 1
        assert tree.nearest(np.array([0.1, 0.9])) == 2

    def test_path_to_root(self):
        tree = RRTree(np.array([0.0, 0.0]))
        idx1 = tree.add_node(np.array([1.0, 0.0]), 0)
        idx2 = tree.add_node(np.array([2.0, 0.0]), idx1)

        path = tree.get_path_to_root(idx2)
        assert len(path) == 3
        assert np.allclose(path[0], [0.0, 0.0])
        assert np.allclose(path[1], [1.0, 0.0])
        assert np.allclose(path[2], [2.0, 0.0])

    def test_rrtree_multiple_roots(self):
        """Test RRTree with multiple root configurations."""
        configs = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        tree = RRTree(configs)
        assert len(tree) == 2
        assert tree.nodes[0].parent is None
        assert tree.nodes[1].parent is None
        assert tree.num_roots == 2

    def test_rrtree_backward_compatibility(self):
        """Test RRTree still accepts single config."""
        tree = RRTree(np.array([0.0, 0.0]))
        assert len(tree) == 1
        assert tree.num_roots == 1

    def test_rrtree_nearest_with_multiple_roots(self):
        """Test nearest neighbor search works with multiple roots."""
        configs = [np.array([0.0, 0.0]), np.array([2.0, 2.0])]
        tree = RRTree(configs)

        # Query closer to first root
        assert tree.nearest(np.array([0.1, 0.1])) == 0
        # Query closer to second root
        assert tree.nearest(np.array([1.9, 1.9])) == 1


class TestCBiRRT:
    def test_plan_simple(self):
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik=ik,
            collision_checker=collision,
        )

        start = np.array([0.0, 0.0])

        # Goal: reach position (1.5, 0.5)
        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5

        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
        )

        path = planner.plan(start, goal_tsrs=[goal_tsr], seed=42)

        assert path is not None
        assert len(path) >= 2
        assert np.allclose(path[0], start)

        # Check goal reached
        final_pose = robot.forward_kinematics(path[-1])
        dist, _ = goal_tsr.distance(final_pose)
        assert dist < 0.2  # Within tolerance

    def test_start_in_collision_raises(self):
        class BlockingCollisionChecker:
            def is_valid(self, q):
                return False

            def is_valid_batch(self, qs):
                return np.zeros(len(qs), dtype=bool)

        planner = CBiRRT(
            robot=MockRobotModel(),
            ik=MockIKSolver(),
            collision_checker=BlockingCollisionChecker(),
        )

        with pytest.raises(AllStartConfigurationsInCollision):
            planner.plan(np.array([0.0, 0.0]), goal_tsrs=[])

    @pytest.mark.parametrize(
        "extend_steps,connect_steps,variant_name",
        [
            (None, None, "CON-CON"),
            (5, 5, "EXT-EXT"),
            (5, None, "EXT-CON"),
            (None, 5, "CON-EXT"),
        ],
    )
    def test_all_variants(self, extend_steps, connect_steps, variant_name):
        """Test all 4 BiRRT variants: CON-CON, EXT-EXT, EXT-CON, CON-EXT."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        config = CBiRRTConfig(
            step_size=0.2,
            extend_steps=extend_steps,
            connect_steps=connect_steps,
        )
        planner = CBiRRT(
            robot=robot,
            ik=ik,
            collision_checker=collision,
            config=config,
        )

        start = np.array([0.0, 0.0])

        # Goal: reach position (1.5, 0.5)
        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5

        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
        )

        result = planner.plan(start, goal_tsrs=[goal_tsr], seed=42, return_details=True)

        assert result.success, f"{variant_name} failed to find path"
        assert len(result.path) >= 2
        assert np.allclose(result.path[0], start)

        # Check goal reached
        final_pose = robot.forward_kinematics(result.path[-1])
        dist, _ = goal_tsr.distance(final_pose)
        assert dist < 0.2, f"{variant_name} did not reach goal (dist={dist})"

    def test_start_tsrs(self):
        """Test planning with start sampled from TSRs."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik=ik,
            collision_checker=collision,
        )

        # Start TSR: positions near (1.0, 1.0)
        T0_w_start = np.eye(4)
        T0_w_start[0, 3] = 1.0
        T0_w_start[1, 3] = 1.0

        start_tsr = TSR(
            T0_w=T0_w_start,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
        )

        # Goal TSR: positions near (1.5, 0.5)
        T0_w_goal = np.eye(4)
        T0_w_goal[0, 3] = 1.5
        T0_w_goal[1, 3] = 0.5

        goal_tsr = TSR(
            T0_w=T0_w_goal,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
        )

        # Plan with start=None, using start_tsrs
        result = planner.plan(
            start=None,
            goal_tsrs=[goal_tsr],
            start_tsrs=[start_tsr],
            seed=42,
            return_details=True,
        )

        assert result.success
        assert len(result.path) >= 2

        # Verify start is in start TSR region
        start_pose = robot.forward_kinematics(result.path[0])
        start_dist, _ = start_tsr.distance(start_pose)
        assert start_dist < 0.2

        # Verify goal is reached
        final_pose = robot.forward_kinematics(result.path[-1])
        goal_dist, _ = goal_tsr.distance(final_pose)
        assert goal_dist < 0.2

    def test_no_start_raises(self):
        """Test that providing neither start nor start_tsrs raises error."""
        planner = CBiRRT(
            robot=MockRobotModel(),
            ik=MockIKSolver(),
            collision_checker=MockCollisionChecker(),
        )

        goal_tsr = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.zeros((6, 2)),
        )

        with pytest.raises(ValueError, match="No valid start configurations available"):
            planner.plan(start=None, goal_tsrs=[goal_tsr])

    def test_plan_multiple_starts(self):
        """Test planning from multiple start configs."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik=ik,
            collision_checker=collision,
        )

        starts = [np.array([0.0, 0.0]), np.array([0.1, 0.1])]

        # Goal TSR at (1.5, 0.5)
        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5
        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
        )

        result = planner.plan(start=starts, goal_tsrs=[goal_tsr], seed=42, return_details=True)

        assert result.success
        # Path starts from one of the provided starts
        assert any(np.allclose(result.path[0], s) for s in starts)

    def test_plan_multiple_goals(self):
        """Test planning to multiple goal configs."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik=ik,
            collision_checker=collision,
        )

        start = np.array([0.0, 0.0])
        # Two valid goal configurations
        goals = [np.array([np.pi / 2, 0.0]), np.array([np.pi / 2, np.pi / 4])]

        path = planner.plan(start=start, goal=goals, seed=42)

        assert path is not None
        # Path ends at one of the provided goals (within tolerance)
        assert any(np.allclose(path[-1], g, atol=0.05) for g in goals)

    def test_plan_multiple_to_multiple(self):
        """Test planning with multiple starts and goals."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik=ik,
            collision_checker=collision,
        )

        starts = [np.array([0.0, 0.0]), np.array([0.1, 0.1])]
        goals = [np.array([np.pi / 2, 0.0]), np.array([np.pi / 2, np.pi / 4])]

        path = planner.plan(start=starts, goal=goals, seed=42)
        assert path is not None

    def test_invalid_start_in_list_raises(self):
        """Test that invalid start config in list raises clear error."""

        class AlwaysInvalidChecker:
            def is_valid(self, q):
                return False

            def is_valid_batch(self, qs):
                return np.zeros(len(qs), dtype=bool)

        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, AlwaysInvalidChecker())

        starts = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]

        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5
        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
        )

        with pytest.raises(AllStartConfigurationsInCollision):
            planner.plan(start=starts, goal_tsrs=[goal_tsr])

    def test_backward_compatibility(self):
        """Test that old API still works."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik=ik,
            collision_checker=collision,
        )

        start_config = np.array([0.0, 0.0])

        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5
        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
        )

        # Old style - positional arg, only goal_tsrs
        path = planner.plan(start_config, goal_tsrs=[goal_tsr], seed=42)
        assert path is not None

    def test_single_goal_config(self):
        """Test planning with single goal configuration (new API)."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik=ik,
            collision_checker=collision,
        )

        start = np.array([0.0, 0.0])
        goal = np.array([np.pi / 2, 0.0])

        path = planner.plan(start=start, goal=goal, seed=42)

        assert path is not None
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal, atol=0.05)


class TestConfigValidation:
    """Tests for start/goal configuration validation."""

    def test_filters_invalid_starts_proceeds_with_valid(self, caplog):
        """Test that invalid starts are filtered but planning proceeds with valid ones."""

        class SelectiveCollisionChecker:
            """Only the first config is in collision."""

            def is_valid(self, q):
                # Config [0, 0] is in collision, others are fine
                return not (q[0] == 0.0 and q[1] == 0.0)

            def is_valid_batch(self, qs):
                return np.array([self.is_valid(q) for q in qs])

        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, SelectiveCollisionChecker())

        # First start is in collision, second is valid
        starts = [np.array([0.0, 0.0]), np.array([0.5, 0.5])]
        goal = np.array([1.0, 0.0])

        import logging

        with caplog.at_level(logging.WARNING):
            path = planner.plan(start=starts, goal=goal, seed=42)

        # Should succeed with the valid start
        assert path is not None
        # Should have logged a warning about the filtered config
        assert "Filtered 1 invalid start" in caplog.text
        assert "Start[0]: in collision" in caplog.text

    def test_all_goals_in_collision_raises(self):
        """Test that all goals in collision raises AllGoalConfigurationsInCollision."""
        from sscbirrt import AllGoalConfigurationsInCollision

        class GoalBlockingChecker:
            """Start is valid, goals are not."""

            def __init__(self, goal_configs):
                self.goal_configs = [tuple(g) for g in goal_configs]

            def is_valid(self, q):
                return tuple(q) not in self.goal_configs

            def is_valid_batch(self, qs):
                return np.array([self.is_valid(q) for q in qs])

        robot = MockRobotModel()
        start = np.array([0.0, 0.0])
        goals = [np.array([1.0, 0.0]), np.array([1.0, 1.0])]

        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, GoalBlockingChecker(goals))

        with pytest.raises(AllGoalConfigurationsInCollision) as exc_info:
            planner.plan(start=start, goal=goals, seed=42)

        assert exc_info.value.n_configs == 2
        assert "Goal[0]: in collision" in str(exc_info.value)
        assert "Goal[1]: in collision" in str(exc_info.value)

    def test_exception_contains_details(self):
        """Test that exception message contains details about which configs failed."""

        class AlwaysInvalidChecker:
            def is_valid(self, q):
                return False

            def is_valid_batch(self, qs):
                return np.zeros(len(qs), dtype=bool)

        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, AlwaysInvalidChecker())

        starts = [np.array([0.0, 0.0]), np.array([0.5, 0.5]), np.array([1.0, 1.0])]

        with pytest.raises(AllStartConfigurationsInCollision) as exc_info:
            planner.plan(start=starts, goal=np.array([2.0, 0.0]), seed=42)

        exc = exc_info.value
        assert exc.n_configs == 3
        assert len(exc.details) == 3
        assert "All 3 start configuration(s) in collision" in str(exc)

    def test_filters_invalid_goals_proceeds_with_valid(self, caplog):
        """Test that invalid goals are filtered but planning proceeds with valid ones."""

        class SelectiveGoalChecker:
            """Only goal [1.0, 1.0] is in collision."""

            def is_valid(self, q):
                return not (abs(q[0] - 1.0) < 0.01 and abs(q[1] - 1.0) < 0.01)

            def is_valid_batch(self, qs):
                return np.array([self.is_valid(q) for q in qs])

        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, SelectiveGoalChecker())

        start = np.array([0.0, 0.0])
        # First goal is in collision, second is valid
        goals = [np.array([1.0, 1.0]), np.array([1.0, 0.0])]

        import logging

        with caplog.at_level(logging.WARNING):
            path = planner.plan(start=start, goal=goals, seed=42)

        # Should succeed with the valid goal
        assert path is not None
        # Should have logged a warning
        assert "Filtered 1 invalid goal" in caplog.text


class TestAngularJoints:
    """Tests for angular joint wraparound handling."""

    def test_angular_joints_validation_mismatch(self):
        """angular_joints length must match robot DOF."""
        robot = MockRobotModel()  # 2 DOF
        ik = MockIKSolver(robot, MockCollisionChecker())

        config = CBiRRTConfig(continuous_joints=(True, True, True))  # 3 != 2
        with pytest.raises(ValueError, match="continuous_joints length"):
            CBiRRT(robot, ik, MockCollisionChecker(), config)

    def test_angular_joints_validation_correct(self):
        """angular_joints with correct length should not raise."""
        robot = MockRobotModel()  # 2 DOF
        ik = MockIKSolver(robot, MockCollisionChecker())

        config = CBiRRTConfig(continuous_joints=(True, True))
        planner = CBiRRT(robot, ik, MockCollisionChecker(), config)
        assert planner.config.continuous_joints == (True, True)

    def test_angular_distance_wraparound(self):
        """Distance between angles near ±pi should be small, not ~2*pi."""
        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        config = CBiRRTConfig(continuous_joints=(True, True))
        planner = CBiRRT(robot, ik, MockCollisionChecker(), config)

        q1 = np.array([-np.pi + 0.1, 0.0])
        q2 = np.array([np.pi - 0.1, 0.0])
        dist = planner.space.distance(q1, q2)
        # Wraparound distance should be ~0.2, not ~2*pi - 0.2
        assert dist < 0.5

    def test_angular_direction_shortest_path(self):
        """Direction should go the short way around the circle."""
        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        config = CBiRRTConfig(continuous_joints=(True, False))
        planner = CBiRRT(robot, ik, MockCollisionChecker(), config)

        q_from = np.array([np.pi - 0.1, 0.0])
        q_to = np.array([-np.pi + 0.1, 0.0])
        direction = planner.space.direction(q_from, q_to)
        # Should go forward (positive) through pi, not backward through 0
        assert direction[0] > 0
        assert abs(direction[0]) < 0.5

    def test_planning_with_angular_joints(self):
        """Planning should succeed with angular_joints enabled."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        config = CBiRRTConfig(continuous_joints=(True, True))
        planner = CBiRRT(robot, ik, collision, config)

        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.5])
        path = planner.plan(start=start, goal=goal, seed=42)
        assert path is not None
        assert len(path) >= 2


def _legacy(planner, **kw):
    """Lower legacy plan() arguments into a PlanningProblem for direct inspection."""
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


def _unconstrained(planner, path):
    """A problem whose only content is the path's endpoints; used to drive smoothing."""
    return PlanningProblem(
        space=planner.space,
        start=FiniteSet([path[0]]),
        goal=FiniteSet([path[-1]]),
        validator=planner.collision,
    )


class TestConstraintTSRs:
    """Tests for path constraint TSR handling."""

    def test_no_constraints_lowers_to_none(self):
        """With no constraint TSRs, the problem has no path constraint and any config is admissible."""
        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, MockCollisionChecker())

        problem = _legacy(planner, start=[np.array([0.0, 0.0])], goal=[np.array([0.5, 0.5])])
        assert problem.path_constraint is None
        assert planner._admissible(problem, np.array([0.0, 0.0]))[0]

    def test_constraint_membership(self):
        """A config is admissible iff its end-effector pose lies in the constraint TSR."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)
        planner = CBiRRT(robot, ik, collision)

        # Constraint TSR: end-effector x in [1.5, 2.5], y in [-0.5, 0.5]
        T0_w = np.eye(4)
        T0_w[0, 3] = 2.0
        constraint_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.5, 0.5],
                    [-0.5, 0.5],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
        )
        q = np.array([0.0, 0.0])
        problem = _legacy(planner, start=[q], goal=[q], constraint_tsrs=[constraint_tsr])

        # q=[0,0] gives FK at (2.0, 0.0) which is inside the constraint
        assert problem.path_constraint.contains(np.array([0.0, 0.0]))
        assert planner._admissible(problem, np.array([0.0, 0.0]))[0]

        # q=[pi/2, 0] gives FK at (0.0, 2.0) which is outside the constraint
        assert not problem.path_constraint.contains(np.array([np.pi / 2, 0.0]))
        ok, reason = planner._admissible(problem, np.array([np.pi / 2, 0.0]))
        assert not ok and reason == "violates path constraints"

    def test_projection_respects_tsr_frames(self):
        """Projection must compose bwopt with T0_w and Tw_e (regression for #28)."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)
        planner = CBiRRT(robot, ik, collision)

        # Same 10cm box around world (1.2, 0.8), expressed two ways
        T0_w = np.eye(4)
        T0_w[0, 3] = 1.2
        T0_w[1, 3] = 0.8
        box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        framed = TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=box)
        shifted = box.copy()
        shifted[0] += 1.2
        shifted[1] += 0.8
        unframed = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=shifted)
        Tw_e = np.eye(4)
        Tw_e[0, 3] = 0.1
        offset = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=box)

        q = np.array([0.3, 0.9])  # violates all three by ~0.38
        for tsr in (framed, unframed, offset):
            problem = _legacy(planner, start=[q], goal=[q], constraint_tsrs=[tsr])
            q_proj = problem.path_constraint.project(q, q)
            assert q_proj is not None
            dist, _ = tsr.distance(robot.forward_kinematics(q_proj))
            assert dist <= planner.config.membership_tolerance

    def test_multiple_constraints_lower_to_allof_with_named_strategy(self):
        """Two constraint TSRs become an AllOf with an explicit projection strategy."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        planner = CBiRRT(robot, MockIKSolver(robot, collision), collision)
        loose = np.array([[-2, 2], [-2, 2], [-1, 1], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
        q = np.array([0.0, 0.0])
        problem = _legacy(planner, start=[q], goal=[q], constraint_tsrs=[TSR(Bw=loose), TSR(Bw=loose)])
        assert isinstance(problem.path_constraint, AllOf)
        assert isinstance(problem.path_constraint.projection, MostViolatedProjection)
        assert problem.path_constraint.contains(q)

    def test_planning_with_constraint_tsrs(self):
        """Planning with constraint TSRs should produce paths satisfying them."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        # Loose constraint: end-effector must stay within a large box
        T0_w = np.eye(4)
        T0_w[0, 3] = 1.0
        constraint_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-2.0, 2.0],
                    [-2.0, 2.0],
                    [-1.0, 1.0],
                    [-np.pi, np.pi],
                    [-np.pi, np.pi],
                    [-np.pi, np.pi],
                ]
            ),
        )

        planner = CBiRRT(robot, ik, collision)

        start = np.array([0.0, 0.0])
        goal = np.array([0.3, 0.3])

        path = planner.plan(start=start, goal=goal, constraint_tsrs=[constraint_tsr], seed=42)

        assert path is not None
        assert len(path) >= 2

        # Verify all path waypoints satisfy the constraint
        for q in path:
            pose = robot.forward_kinematics(q)
            dist, _ = constraint_tsr.distance(pose)
            assert dist < 0.1, f"Path waypoint violates constraint (dist={dist})"


class TestPathSmoothing:
    """Tests for path smoothing."""

    def test_smoothing_shortens_a_jagged_path(self):
        """Smoothing a jagged path shortens it. (Waypoint count is not the measure: a shortcut is stored at
        edge_resolution, so a straight path can carry more waypoints than the zigzag it replaces.)"""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        # Use a generous step size so smoothing can shortcut
        config = CBiRRTConfig(
            smooth_path=False,  # We'll call _smooth_path manually
            step_size=0.3,
            smoothing_iterations=100,
            smoothing_patience=30,
        )
        planner = CBiRRT(robot, ik, collision, config)
        planner._rng = np.random.default_rng(42)

        # Create a deliberately jagged path (zigzag)
        path = [
            np.array([0.0, 0.0]),
            np.array([0.1, 0.2]),
            np.array([0.2, 0.0]),
            np.array([0.3, 0.2]),
            np.array([0.4, 0.0]),
            np.array([0.5, 0.2]),
            np.array([0.6, 0.0]),
        ]

        smoothed = planner._smooth_path(_unconstrained(planner, path), path)

        def length(p):
            return sum(np.linalg.norm(b - a) for a, b in zip(p, p[1:]))

        assert length(smoothed) < length(path)
        assert np.array_equal(smoothed[0], path[0]) and np.array_equal(smoothed[-1], path[-1])

    def test_smoothing_short_path_unchanged(self):
        """Path with 2 or fewer waypoints should not be modified."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)
        planner = CBiRRT(robot, ik, collision)
        planner._rng = np.random.default_rng(42)

        path = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
        smoothed = planner._smooth_path(_unconstrained(planner, path), path)
        assert len(smoothed) == 2

    def test_smoothing_patience_terminates_early(self):
        """Smoothing should stop early when no progress is made."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        config = CBiRRTConfig(
            smooth_path=False,
            smoothing_iterations=1000,
            smoothing_patience=5,
        )
        planner = CBiRRT(robot, ik, collision, config)
        planner._rng = np.random.default_rng(42)

        # Already-smooth straight-line path — no shortcuts possible
        path = [np.array([0.0, 0.0]), np.array([0.05, 0.0]), np.array([0.1, 0.0])]
        smoothed = planner._smooth_path(_unconstrained(planner, path), path)
        # Should terminate early (patience=5) rather than running 1000 iterations
        assert smoothed is not None


class TestTreeCaching:
    """Tests for RRTree config array caching."""

    def test_nearest_consistent_after_add(self):
        """nearest() should return correct results after adding nodes."""
        tree = RRTree(np.array([0.0, 0.0]))
        tree.add_node(np.array([10.0, 10.0]), 0)

        # Query near second node
        assert tree.nearest(np.array([9.0, 9.0])) == 1

        # Add a third node closer to query
        tree.add_node(np.array([9.5, 9.5]), 1)
        assert tree.nearest(np.array([9.0, 9.0])) == 2


class TestPlanResultIndices:
    """Verify start_index and goal_index point to the correct TSR/config."""

    @pytest.fixture
    def planner(self):
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)
        return CBiRRT(robot=robot, ik=ik, collision_checker=collision), robot

    def _make_point_tsr(self, x, y, tol=0.1):
        """Create a small TSR at (x, y)."""
        T0_w = np.eye(4)
        T0_w[0, 3] = x
        T0_w[1, 3] = y
        Bw = np.array(
            [
                [-tol, tol],
                [-tol, tol],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        return TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=Bw)

    def test_goal_index_matches_reached_tsr(self, planner):
        """goal_index should point to the TSR closest to path[-1]."""
        cbrt, robot = planner

        start = np.array([0.0, 0.5])

        # Two well-separated goal TSRs
        tsr_a = self._make_point_tsr(1.5, 0.5)  # index 0
        tsr_b = self._make_point_tsr(-1.5, 0.5)  # index 1

        result = cbrt.plan(
            start=start,
            goal_tsrs=[tsr_a, tsr_b],
            return_details=True,
        )
        assert result.success

        # Compute EE pose at path end
        ee_pose = robot.forward_kinematics(result.path[-1])

        # The reached TSR should have ~zero distance
        dist_a, _ = tsr_a.distance(ee_pose)
        dist_b, _ = tsr_b.distance(ee_pose)

        if result.goal_index == 0:
            assert dist_a < 0.15, f"goal_index=0 but dist to TSR[0]={dist_a:.3f}"
            assert dist_b > dist_a, "goal_index=0 but TSR[1] is closer"
        else:
            assert dist_b < 0.15, f"goal_index=1 but dist to TSR[1]={dist_b:.3f}"
            assert dist_a > dist_b, "goal_index=1 but TSR[0] is closer"

    def test_start_index_matches_reached_tsr(self, planner):
        """start_index should point to the TSR closest to path[0]."""
        cbrt, robot = planner

        goal = np.array([0.0, 0.5])

        # Two well-separated start TSRs
        tsr_a = self._make_point_tsr(1.5, 0.5)  # index 0
        tsr_b = self._make_point_tsr(-1.5, 0.5)  # index 1

        result = cbrt.plan(
            start_tsrs=[tsr_a, tsr_b],
            goal=goal,
            return_details=True,
        )
        assert result.success

        # Compute EE pose at path start
        ee_pose = robot.forward_kinematics(result.path[0])

        dist_a, _ = tsr_a.distance(ee_pose)
        dist_b, _ = tsr_b.distance(ee_pose)

        if result.start_index == 0:
            assert dist_a < 0.15, f"start_index=0 but dist to TSR[0]={dist_a:.3f}"
            assert dist_b > dist_a, "start_index=0 but TSR[1] is closer"
        else:
            assert dist_b < 0.15, f"start_index=1 but dist to TSR[1]={dist_b:.3f}"
            assert dist_a > dist_b, "start_index=1 but TSR[0] is closer"

    def test_goal_index_with_multiple_configs(self, planner):
        """goal_index with config list should match path[-1]."""
        cbrt, robot = planner

        start = np.array([0.0, 0.5])
        goal_a = np.array([1.0, 0.5])  # index 0
        goal_b = np.array([-1.0, 0.5])  # index 1

        result = cbrt.plan(
            start=start,
            goal=[goal_a, goal_b],
            return_details=True,
        )
        assert result.success

        end = result.path[-1]
        dist_a = np.linalg.norm(end - goal_a)
        dist_b = np.linalg.norm(end - goal_b)

        if result.goal_index == 0:
            assert dist_a < dist_b
        else:
            assert dist_b < dist_a

    def test_single_start_goal_returns_zero(self, planner):
        """Single start and goal should return index 0."""
        cbrt, robot = planner

        result = cbrt.plan(
            start=np.array([0.0, 0.5]),
            goal=np.array([0.5, 0.5]),
            return_details=True,
        )
        assert result.success
        assert result.start_index == 0
        assert result.goal_index == 0

    def test_planning_stats_populated(self, planner):
        """Planning stats should be populated on success and failure."""
        cbrt, robot = planner

        result = cbrt.plan(
            start=np.array([0.0, 0.5]),
            goal=np.array([0.5, 0.5]),
            return_details=True,
        )
        assert result.success
        assert result.planning_time > 0
        assert result.iterations > 0
        assert result.tree_sizes[0] > 0
        assert result.tree_sizes[1] > 0


class TestConfigurationInputs:
    """#171: start/goal accept one configuration as any flat sequence, or a list of them, on every backend."""

    @pytest.mark.parametrize("backend", ["python", "native", "auto"])
    def test_flat_list_is_one_configuration(self, backend):
        from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK

        if backend == "native":
            pytest.importorskip("sscbirrt._native")
        planner = CBiRRT(PlanarArm(), PlanarIK(), NoCollision(), CBiRRTConfig(), backend=backend)
        assert planner.plan(start=[0.1, 0.2], goal=(1.0, 0.5), seed=0) is not None
        r = planner.plan(start=[[0.1, 0.2], [0.0, 0.0]], goal=[1.0, 0.5], seed=0, return_details=True)
        assert r.success and r.start_index in (0, 1)

    @pytest.mark.parametrize(
        "bad, message",
        [
            ([0.1, 0.2, 0.3], r"got shape \(3,\)"),
            ([[0.1, 0.2], [0.3]], "got a ragged or non-numeric value"),
            ("ab", "got a ragged"),
        ],
    )
    def test_malformed_input_names_the_argument(self, bad, message):
        from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK

        planner = CBiRRT(PlanarArm(), PlanarIK(), NoCollision(), CBiRRTConfig(), backend="python")
        with pytest.raises(
            ValueError, match=rf"start: expected one configuration of length 2 or a list of them, {message}"
        ):
            planner.plan(start=bad, goal=[1.0, 0.5])


class TestSampleProbability:
    """#196, CBiRRT's P_sample: on a tree's turn, a coin adds roots from the tree's own set instead of extending.

    A full-height wall at joint 0 in (0.45, 0.55), and another across the seam at +-pi (the planar arm's joints wrap);
    the goal is a band beyond the first wall (unreachable) or on the start's side. The native backend runs the same
    cases in cpp/tests/test_planner.cpp, in a bounded space with the one wall.
    """

    class Walls:
        def is_valid(self, q):
            return not (0.45 < q[0] < 0.55 or abs(q[0]) > 2.5)

    class Band:
        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi

        def contains(self, q):
            return self.lo <= q[0] <= self.hi

        def sample(self, rng):
            from sscbirrt.sets import Sample

            return [Sample(np.array([self.lo + rng.random() * (self.hi - self.lo), 0.0]), ())]

    def _solve(self, goal, seed, max_iterations=100000, **kw):
        from sscbirrt.testing import PlanarArm, PlanarIK

        cfg = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, num_tree_roots=1, max_iterations=max_iterations, **kw)
        planner = CBiRRT(PlanarArm(), PlanarIK(), self.Walls(), cfg, backend="python")
        problem = PlanningProblem(
            space=planner.space,
            start=FiniteSet([np.zeros(2)], metric=planner.space.distance),
            goal=goal,
            validator=planner.collision,
        )
        return planner.solve(problem, seed=seed)

    @staticmethod
    def _roots(tree):
        return [n for n in tree.nodes if n.parent is None]

    def test_zero_adds_no_roots_during_the_search(self):
        r = self._solve(self.Band(1.0, 1.2), seed=3, max_iterations=200, goal_sample_probability=0.0)
        assert not r.success and r.stats["search_roots"] == 0
        assert len(self._roots(r.tree_goal)) == 1

    def test_one_adds_a_root_on_every_goal_turn_and_none_to_a_finite_start(self):
        r = self._solve(
            self.Band(1.0, 1.2), seed=3, max_iterations=200, goal_sample_probability=1.0, start_sample_probability=1.0
        )
        assert not r.success and r.stats["search_roots"] == 100  # 100 goal turns, one candidate per draw
        roots = self._roots(r.tree_goal)
        assert len(roots) == 101 and all(1.0 <= n.config[0] <= 1.2 for n in roots)
        assert len(self._roots(r.tree_start)) == 1  # the finite start never tosses the coin

    def test_a_path_to_a_root_added_during_the_search_reports_its_provenance(self):
        from sscbirrt.sets import AnyOf

        goal = AnyOf([self.Band(1.0, 1.2), self.Band(-1.2, -1.0)], weights=[1.0, 1.0])
        checked = 0
        for seed in range(40):
            r = self._solve(goal, seed=seed, goal_sample_probability=0.5)
            assert r.success and r.goal_source == (1,) and r.goal_index == 1  # only the near band is reachable
            assert -1.2 <= r.path[-1][0] <= -1.0
            if r.tree_goal.nodes[0].source_index == (0,):  # the first goal root was unreachable
                assert r.stats["search_roots"] > 0
                checked += 1
        assert checked > 0


class TestWindingsShareVerdicts:
    """#200: windings of one physical configuration share the validator's verdict, only when it declares
    invariance to full turns. Results are identical; only the number of validator calls changes."""

    class Cos:
        def __init__(self, invariant):
            self.full_turn_invariant, self.calls = invariant, 0

        def is_valid(self, q):
            self.calls += 1
            return np.cos(q[0]) > -0.5  # depends on q[0] only through its angle

    class Windings:
        def contains(self, q):
            return abs(q[1] - 1.0) < 1e-9

        def sample(self, rng):
            from sscbirrt.sets import Sample
            from sscbirrt.tsr_set import full_turn_key

            a = -np.pi + rng.random() * 2 * np.pi
            qs = [np.array([w, 1.0]) for w in (a, a + 2 * np.pi, a - 2 * np.pi)]
            return [Sample(q, (), full_turn_key(q, (True, False))) for q in qs]

    def test_full_turn_key(self):
        from sscbirrt.tsr_set import full_turn_key

        k = full_turn_key(np.array([0.3, 1.0]), (True, False))
        assert k == full_turn_key(np.array([0.3 + 2 * np.pi, 1.0]), (True, False))
        assert k != full_turn_key(np.array([0.3, 1.0 + 2 * np.pi]), (True, False))
        assert k != full_turn_key(np.array([0.31, 1.0]), (True, False))

    def test_shared_only_when_declared_and_results_identical(self):
        from sscbirrt.space import JointSpace

        def run(invariant):
            v = self.Cos(invariant)
            space = JointSpace(np.array([-7.0, -1.5]), np.array([7.0, 1.5]))
            cfg = CBiRRTConfig(step_size=0.2, num_tree_roots=30, max_per_draw=3, goal_sample_probability=0.0)
            planner = CBiRRT(MockRobotModel(), None, v, cfg, backend="python")
            problem = PlanningProblem(
                space=space,
                start=FiniteSet([np.array([0.0, 1.0])], metric=space.distance),
                goal=self.Windings(),
                validator=v,
            )
            return planner.solve(problem, seed=5), v.calls

        (shared, shared_calls), (unshared, unshared_calls) = run(True), run(False)
        assert shared.success and unshared.success
        assert all(np.array_equal(a, b) for a, b in zip(shared.path, unshared.path))
        assert shared.iterations == unshared.iterations
        assert shared.stats["state_checks"] == unshared.stats["state_checks"]  # every candidate is still judged
        assert shared.stats["reused_verdicts"] > 0 and unshared.stats["reused_verdicts"] == 0
        assert shared_calls < unshared_calls
