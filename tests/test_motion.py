# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Local-motion validation is an explicit, replaceable boundary (#46).

Every tree edge, the final connection between trees, and every shortcut pass
through ``PlanningProblem.motion_validator``. The default reproduces the
discretized behavior; a custom validator may be stricter but never looser.
"""

import numpy as np
import pytest

from sscbirrt import CBiRRT, CBiRRTConfig, DiscreteMotionValidator, LocalMotion, MotionContractError, PlanningProblem
from sscbirrt.sets import FiniteSet, PredicateSet
from sscbirrt.space import JointSpace
from sscbirrt.tree import RRTree
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel


def make_planner(**cfg):
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    return CBiRRT(robot, MockIKSolver(robot, collision), collision, CBiRRTConfig(**cfg))


def problem(planner, start, goal, motion_validator=None, path_constraint=None):
    return PlanningProblem(
        space=planner.space,
        start=FiniteSet([start]),
        goal=FiniteSet([goal]),
        validator=planner.collision,
        path_constraint=path_constraint,
        motion_validator=motion_validator,
    )


class Spy:
    """Wraps a validator, recording every (q_from, q_to) it was asked about."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def validate(self, q_from, q_to):
        self.calls.append((np.array(q_from), np.array(q_to)))
        return self.inner.validate(q_from, q_to)


class NoCrossing:
    """Rejects any motion that crosses the line q[0] = c, even if both endpoints are valid."""

    def __init__(self, inner, c):
        self.inner, self.c = inner, c

    def validate(self, q_from, q_to):
        if (q_from[0] - self.c) * (q_to[0] - self.c) < 0:
            return LocalMotion()
        return self.inner.validate(q_from, q_to)


# ---------------------------------------------------------------------------
# DiscreteMotionValidator
# ---------------------------------------------------------------------------


class TestDiscreteMotionValidator:
    space = JointSpace(np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi]))

    def test_samples_every_resolution_and_ends_exactly(self):
        v = DiscreteMotionValidator(self.space, lambda q: True, 0.01)
        m = v.validate(np.zeros(2), np.array([0.05, 0.0]))
        assert m.reached and len(m.configs) == 5
        assert np.array_equal(m.configs[-1], [0.05, 0.0])
        assert np.allclose(m.configs[0], [0.01, 0.0])

    def test_partial_prefix_on_first_invalid_sample(self):
        v = DiscreteMotionValidator(self.space, lambda q: q[0] < 0.035, 0.01)
        m = v.validate(np.zeros(2), np.array([0.05, 0.0]))
        assert not m.reached and len(m.configs) == 3  # 0.01, 0.02, 0.03 valid; 0.04 not

    def test_zero_length_motion(self):
        v = DiscreteMotionValidator(self.space, lambda q: True, 0.01)
        m = v.validate(np.zeros(2), np.zeros(2))
        assert m.reached and m.configs == []

    def test_out_of_space_target(self):
        v = DiscreteMotionValidator(self.space, lambda q: True, 0.01)
        m = v.validate(np.zeros(2), np.array([10.0, 0.0]))
        assert not m.reached and m.configs == []

    def test_wraps_for_angular_joints(self):
        s = JointSpace(np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi]), continuous_joints=(True, False))
        v = DiscreteMotionValidator(s, lambda q: True, 0.1)
        m = v.validate(np.array([3.0, 0.0]), np.array([-3.0, 0.0]))
        assert m.reached and len(m.configs) == 3  # short way: 0.28 rad at 0.1 -> 3 samples
        assert np.array_equal(m.configs[-1], [-3.0, 0.0])
        assert m.configs[0][0] > 3.0  # interior samples go the short way past pi

    def test_bad_resolution(self):
        with pytest.raises(ValueError):
            DiscreteMotionValidator(self.space, lambda q: True, 0.0)


# ---------------------------------------------------------------------------
# Planner integration
# ---------------------------------------------------------------------------


class TestDefaultReproducesBehavior:
    def test_default_and_explicit_discrete_give_identical_trees(self):
        planner = make_planner(edge_resolution=0.01, connection_tolerance=0.1, smooth_path=False)
        q0, q1 = np.zeros(2), np.array([0.05, 0.02])
        default = problem(planner, q0, q1)
        tree_a = RRTree(q0)
        planner._grow(default, tree_a, q1)

        explicit = DiscreteMotionValidator(planner.space, lambda q: planner._admissible(default, q)[0], 0.01)
        with_explicit = problem(planner, q0, q1, motion_validator=explicit)
        tree_b = RRTree(q0)
        planner._grow(with_explicit, tree_b, q1)

        assert len(tree_a) == len(tree_b)
        assert all(np.array_equal(a.config, b.config) for a, b in zip(tree_a.nodes, tree_b.nodes))

    def test_default_uses_edge_resolution_then_step_size(self):
        planner = make_planner(step_size=0.3)
        v = planner._motion_validator(problem(planner, np.zeros(2), np.zeros(2)))
        assert isinstance(v, DiscreteMotionValidator) and v.resolution == 0.3
        planner = make_planner(step_size=0.3, edge_resolution=0.05)
        v = planner._motion_validator(problem(planner, np.zeros(2), np.zeros(2)))
        assert v.resolution == 0.05


class TestCustomValidator:
    def test_rejects_motion_with_valid_endpoints(self):
        """A crossing-aware validator blocks the direct segment even though every sample is valid."""
        planner = make_planner(connection_tolerance=0.5, smooth_path=False)
        q0, q1 = np.array([0.4, 0.0]), np.array([0.6, 0.0])
        base = planner._motion_validator(problem(planner, q0, q1))
        prob = problem(planner, q0, q1, motion_validator=NoCrossing(base, 0.5))
        tree = RRTree(q0)
        idx, reached = planner._grow(prob, tree, q1)
        assert not reached and len(tree) == 1

    def test_solve_respects_custom_validator_on_every_segment(self):
        """With crossing forbidden, no returned segment crosses the line and the plan still succeeds around it.

        The line is only forbidden for |q[1]| < 1 (a wall with an opening), so a path exists.
        """
        planner = make_planner(step_size=0.2, connection_tolerance=0.1, edge_resolution=0.05)
        q0, q1 = np.array([0.4, 0.0]), np.array([0.6, 0.0])

        def hits_wall(a, b):
            """Whether the straight segment a->b crosses x = 0.5 at a point with |y| < 1."""
            if (a[0] - 0.5) * (b[0] - 0.5) >= 0:
                return False
            t = (0.5 - a[0]) / (b[0] - a[0])
            return abs(a[1] + t * (b[1] - a[1])) < 1.0

        class Wall:
            def __init__(self, inner):
                self.inner = inner

            def validate(self, q_from, q_to):
                if hits_wall(q_from, q_to):
                    return LocalMotion()
                return self.inner.validate(q_from, q_to)

        base = planner._motion_validator(problem(planner, q0, q1))
        prob = problem(planner, q0, q1, motion_validator=Wall(base))
        for seed in range(3):
            result = planner.solve(prob, seed=seed)
            assert result.success
            # Any sub-segment of a validated edge crosses where the edge crosses, so this holds per waypoint pair
            for a, b in zip(result.path[:-1], result.path[1:]):
                assert not hits_wall(a, b)

    def test_looser_validator_cannot_store_inadmissible_configs(self):
        """Configurations a custom validator returns are still checked; the invariant survives a bad validator."""
        planner = make_planner(connection_tolerance=0.5, smooth_path=False)
        q0, q1 = np.zeros(2), np.array([0.3, 0.0])

        class Liar:
            def validate(self, q_from, q_to):
                return LocalMotion(configs=[np.array([10.0, 0.0]), np.array(q_to)], reached=True)

        prob = problem(planner, q0, q1, motion_validator=Liar())
        tree = RRTree(q0)
        idx, reached = planner._grow(prob, tree, q1)
        assert not reached and len(tree) == 1
        assert all(planner.space.contains(n.config) for n in tree.nodes)

    def test_validator_claiming_reached_must_end_at_target(self):
        planner = make_planner(connection_tolerance=0.5, smooth_path=False)
        q0, q1 = np.zeros(2), np.array([0.3, 0.0])

        class OffByABit:
            def validate(self, q_from, q_to):
                return LocalMotion(configs=[np.array([0.29, 0.0])], reached=True)

        prob = problem(planner, q0, q1, motion_validator=OffByABit())
        tree = RRTree(q0)
        with pytest.raises(MotionContractError, match="instead of the exact target"):
            planner._grow(prob, tree, q1)
        assert len(tree) == 1  # nothing stored before the contract check

    def test_custom_validator_still_subject_to_path_constraint(self):
        planner = make_planner(connection_tolerance=0.5, smooth_path=False)
        q0, q1 = np.zeros(2), np.array([0.3, 0.0])
        half = PredicateSet(lambda q: q[0] < 0.2)

        class PassEverything:
            def validate(self, q_from, q_to):
                return LocalMotion(configs=[np.array([0.1, 0.0]), np.array([0.25, 0.0]), np.array(q_to)], reached=True)

        prob = problem(planner, q0, q1, motion_validator=PassEverything(), path_constraint=half)
        tree = RRTree(q0)
        idx, reached = planner._grow(prob, tree, q1)
        # A claimed success containing a state outside the path constraint is rejected whole (#54)
        assert not reached and len(tree) == 1


class TestSingleBoundary:
    def test_growth_connection_and_shortcut_all_go_through_the_validator(self):
        planner = make_planner(step_size=0.1, connection_tolerance=0.05, smoothing_iterations=20)
        q0, q1 = np.zeros(2), np.array([0.33, 0.0])
        base = planner._motion_validator(problem(planner, q0, q1))
        spy = Spy(base)
        prob = problem(planner, q0, q1, motion_validator=spy)

        # Growth: several step edges, then the final exact connection to q1
        tree = RRTree(q0)
        idx, reached = planner._grow(prob, tree, q1)
        assert reached
        assert len(spy.calls) >= 4
        assert np.array_equal(spy.calls[-1][1], q1)  # the last call is the exact connection

        # Shortcut: goes through the same validator
        n = len(spy.calls)
        shortcut = planner._try_shortcut(prob, q0, q1)
        assert shortcut is not None and len(spy.calls) > n

        # Full solve with smoothing: every stored edge came through the validator
        spy.calls.clear()
        result = planner.solve(prob, seed=0)
        assert result.success and len(spy.calls) > 0

    def test_partial_motion_keeps_progress(self):
        planner = make_planner(step_size=0.3, edge_resolution=0.01, connection_tolerance=0.5, smooth_path=False)
        q0, q1 = np.zeros(2), np.array([0.05, 0.0])
        base = planner._motion_validator(problem(planner, q0, q1))

        class StopsShort:
            def validate(self, q_from, q_to):
                m = base.validate(q_from, q_to)
                return LocalMotion(configs=m.configs[:2], reached=False)

        prob = problem(planner, q0, q1, motion_validator=StopsShort())
        tree = RRTree(q0)
        idx, reached = planner._grow(prob, tree, q1)
        assert not reached and len(tree) == 3
        assert np.allclose(tree.nodes[idx].config, [0.02, 0.0])


# ---------------------------------------------------------------------------
# LocalMotion contract enforcement (#54)
# ---------------------------------------------------------------------------


class EmptyButReached:
    def validate(self, q_from, q_to):
        return LocalMotion(configs=[], reached=True)


class TestLocalMotionContract:
    def setup_method(self):
        self.planner = make_planner(connection_tolerance=0.5, step_size=0.1)
        self.q0, self.q1 = np.zeros(2), np.array([0.3, 0.0])

    def test_empty_reached_on_nonzero_motion_raises_in_grow(self):
        prob = problem(self.planner, self.q0, self.q1, motion_validator=EmptyButReached())
        tree = RRTree(self.q0)
        with pytest.raises(MotionContractError, match="no configurations"):
            self.planner._grow(prob, tree, self.q1)
        assert len(tree) == 1

    def test_empty_reached_raises_during_bidirectional_connection(self):
        prob = problem(self.planner, self.q0, self.q1, motion_validator=EmptyButReached())
        with pytest.raises(MotionContractError):
            self.planner.solve(prob, seed=0)

    def test_empty_reached_raises_during_shortcut_smoothing(self):
        base = self.planner._motion_validator(problem(self.planner, self.q0, self.q1))

        class LiesOnlyForShortcuts:
            """Honest for growth, but claims an empty success once smoothing asks for a long shortcut."""

            def validate(self, q_from, q_to):
                if np.linalg.norm(np.asarray(q_to) - np.asarray(q_from)) > 0.25:
                    return LocalMotion(configs=[], reached=True)
                return base.validate(q_from, q_to)

        prob = problem(self.planner, self.q0, self.q1, motion_validator=LiesOnlyForShortcuts())
        path = [np.array([0.0, 0.0]), np.array([0.1, 0.1]), np.array([0.2, 0.0]), np.array([0.3, 0.0])]
        self.planner._rng = np.random.default_rng(0)
        with pytest.raises(MotionContractError):
            self.planner._smooth_path(prob, path)

    def test_claimed_success_with_inadmissible_state_is_rejected_whole(self):
        class BadMiddle:
            def validate(self, q_from, q_to):
                return LocalMotion(configs=[np.array([0.1, 0.0]), np.array([10.0, 0.0]), np.array(q_to)], reached=True)

        prob = problem(self.planner, self.q0, self.q1, motion_validator=BadMiddle())
        tree = RRTree(self.q0)
        idx, reached = self.planner._grow(prob, tree, self.q1)
        assert not reached and len(tree) == 1  # the admissible first config was not stored either

    def test_partial_result_keeps_admissible_prefix(self):
        class PartialWithBadTail:
            def validate(self, q_from, q_to):
                return LocalMotion(configs=[np.array([0.1, 0.0]), np.array([10.0, 0.0])], reached=False)

        prob = problem(self.planner, self.q0, self.q1, motion_validator=PartialWithBadTail())
        tree = RRTree(self.q0)
        idx, reached = self.planner._grow(prob, tree, self.q1)
        assert not reached and len(tree) == 2
        assert np.array_equal(tree.nodes[idx].config, [0.1, 0.0])

    def test_zero_length_motion_succeeds_without_node_regardless_of_payload(self):
        class ReturnsTargetForZeroMotion:
            def validate(self, q_from, q_to):
                return LocalMotion(configs=[np.array(q_to)], reached=True)

        prob = problem(self.planner, self.q0, self.q0, motion_validator=ReturnsTargetForZeroMotion())
        tree = RRTree(self.q0)
        idx, reached = self.planner._grow(prob, tree, self.q0)
        assert reached and idx == 0 and len(tree) == 1
        # And the empty-but-reached payload is fine for a zero motion
        prob = problem(self.planner, self.q0, self.q0, motion_validator=EmptyButReached())
        tree = RRTree(self.q0)
        assert self.planner._grow(prob, tree, self.q0) == (0, True)

    def test_default_validator_satisfies_the_contract_everywhere(self):
        """The default never trips the checks across many seeded plans with smoothing."""
        planner = make_planner(step_size=0.2, connection_tolerance=0.1, edge_resolution=0.05)
        for seed in range(5):
            result = planner.solve(problem(planner, np.zeros(2), np.array([1.0, 0.5])), seed=seed)
            assert result.success
            assert np.array_equal(result.path[-1], [1.0, 0.5])

    def test_wrong_shape_final_config_raises(self):
        class WrongShape:
            def validate(self, q_from, q_to):
                return LocalMotion(configs=[np.array([0.3, 0.0, 0.0])], reached=True)

        prob = problem(self.planner, self.q0, self.q1, motion_validator=WrongShape())
        with pytest.raises(MotionContractError):
            self.planner._grow(prob, RRTree(self.q0), self.q1)


# ---------------------------------------------------------------------------
# Replacement vs composition (#56)
# ---------------------------------------------------------------------------


class TestReplacementVsComposition:
    """A bare custom validator replaces the default; RestrictedMotionValidator composes with it."""

    class Wall:
        """Invalid inside a thin slab that a validator checking only the endpoint would miss."""

        def is_valid(self, q):
            return not (0.14 < q[0] < 0.16)

    def make(self):
        robot = MockRobotModel()
        wall = self.Wall()
        cfg = CBiRRTConfig(step_size=0.3, edge_resolution=0.01, connection_tolerance=0.5, smooth_path=False)
        planner = CBiRRT(robot, MockIKSolver(robot, MockCollisionChecker()), wall, cfg)
        return planner, wall

    def test_bare_custom_validator_is_trusted_for_the_interior(self):
        """Documented behavior: the planner does not sample between a custom validator's configurations."""
        planner, wall = self.make()
        q0, q1 = np.zeros(2), np.array([0.3, 0.0])

        class EndpointOnly:
            def validate(self, q_from, q_to):
                return LocalMotion(configs=[np.array(q_to)], reached=True)

        prob = problem(planner, q0, q1, motion_validator=EndpointOnly())
        tree = RRTree(q0)
        idx, reached = planner._grow(prob, tree, q1)
        assert reached  # trusted: the wall in the interior is not detected by the planner
        assert wall.is_valid(q1)

    def test_default_catches_the_wall(self):
        planner, _ = self.make()
        q0, q1 = np.zeros(2), np.array([0.3, 0.0])
        tree = RRTree(q0)
        idx, reached = planner._grow(problem(planner, q0, q1), tree, q1)
        assert not reached

    def test_restricted_keeps_default_checks_and_adds_a_restriction(self):
        from sscbirrt import RestrictedMotionValidator

        planner, _ = self.make()
        q0 = np.zeros(2)
        prob0 = problem(planner, q0, q0)
        base = planner.default_motion_validator(prob0)

        # Restriction: no motion may change q[1] by more than 0.05
        restricted = RestrictedMotionValidator(base, accepts=lambda a, b: abs(b[1] - a[1]) <= 0.05)

        # Still catches the wall (default checks retained)
        q_wall = np.array([0.3, 0.0])
        prob = problem(planner, q0, q_wall, motion_validator=restricted)
        assert not planner._grow(prob, RRTree(q0), q_wall)[1]

        # Rejects a wall-free motion that violates the restriction, and stores nothing
        q_jump = np.array([0.0, 0.2])
        prob = problem(planner, q0, q_jump, motion_validator=restricted)
        tree = RRTree(q0)
        idx, reached = planner._grow(prob, tree, q_jump)
        assert not reached and len(tree) == 1

        # Accepts a wall-free motion within the restriction, with the default's samples
        q_ok = np.array([0.0, 0.04])
        prob = problem(planner, q0, q_ok, motion_validator=restricted)
        tree = RRTree(q0)
        idx, reached = planner._grow(prob, tree, q_ok)
        assert reached and len(tree) == 1 + 4 and np.array_equal(tree.nodes[idx].config, q_ok)

    def test_default_motion_validator_is_public_and_matches_internal(self):
        planner, _ = self.make()
        prob = problem(planner, np.zeros(2), np.zeros(2))
        assert isinstance(planner.default_motion_validator(prob), DiscreteMotionValidator)
        assert planner.default_motion_validator(prob).resolution == planner._motion_validator(prob).resolution

    def test_goal_tree_edges_are_validated_from_the_goal_side(self):
        """Documents the reversibility requirement: half the calls have q_from nearer the goal."""
        robot = MockRobotModel()
        cfg = CBiRRTConfig(step_size=0.1, connection_tolerance=0.05, smooth_path=False)
        planner = CBiRRT(robot, MockIKSolver(robot, MockCollisionChecker()), MockCollisionChecker(), cfg)
        q0, q1 = np.zeros(2), np.array([1.0, 0.0])
        base = planner.default_motion_validator(problem(planner, q0, q1))
        spy = Spy(base)
        result = planner.solve(problem(planner, q0, q1, motion_validator=spy), seed=0)
        assert result.success
        # Some motions start at the start root (start tree, execution direction) and some start at the
        # goal root (goal tree, validated in the reverse of execution direction)
        from_start_root = sum(1 for a, _ in spy.calls if np.array_equal(a, q0))
        from_goal_root = sum(1 for a, _ in spy.calls if np.array_equal(a, q1))
        assert from_start_root > 0 and from_goal_root > 0
