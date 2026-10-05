# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""PlanningProblem.sampler replaces the free-space target sampler; the default is the space (#110)."""

import numpy as np

from sscbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem, SpaceSampler
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK, Wall

START, GOAL = np.array([-0.5, 0.5]), np.array([0.5, 0.5])


def _planner(collision=None, **kw):
    cfg = CBiRRTConfig(step_size=0.1, smooth_path=False, **kw)
    return CBiRRT(PlanarArm(), PlanarIK(), collision or NoCollision(), cfg)


def _problem(planner, sampler=None):
    return PlanningProblem(
        space=planner.space,
        start=FiniteSet([START], metric=planner.space.distance),
        goal=FiniteSet([GOAL], metric=planner.space.distance),
        validator=planner.collision,
        sampler=sampler,
    )


class Recording:
    """Uniform, but records every draw."""

    def __init__(self, space):
        self.space, self.draws = space, []

    def sample(self, rng):
        q = self.space.sample(rng)
        self.draws.append(q)
        return q


class AlwaysGoal:
    def sample(self, rng):
        return GOAL.copy()


def test_joint_space_is_a_space_sampler():
    assert isinstance(_planner().space, SpaceSampler)


def test_default_is_the_space_itself():
    planner = _planner()
    a = planner.solve(_problem(planner), seed=4)
    b = planner.solve(_problem(planner, sampler=planner.space), seed=4)
    assert a.success and b.success
    assert all(np.array_equal(x, y) for x, y in zip(a.path, b.path))
    assert a.iterations == b.iterations


def test_custom_sampler_is_consulted_once_per_unbiased_iteration():
    planner = _planner()
    rec = Recording(planner.space)
    result = planner.solve(_problem(planner, sampler=rec), seed=0)
    assert result.success
    assert len(rec.draws) == result.iterations  # no bias, so every iteration draws from the sampler


def test_a_sampler_that_proposes_the_goal_connects_in_one_iteration():
    planner = _planner()
    result = planner.solve(_problem(planner, sampler=AlwaysGoal()), seed=0)
    assert result.success
    assert result.iterations == 1
    assert np.array_equal(result.path[0], START) and np.array_equal(result.path[-1], GOAL)
    # straight line in joint space: every waypoint lies on the segment
    for q in result.path:
        assert abs(q[1] - 0.5) < 1e-12


def test_a_sampler_outside_the_space_degrades_to_no_growth():
    class Outside:
        def sample(self, rng):
            return np.array([10.0, 10.0])  # beyond the ±π limits

    # A wall between start and goal, so the connect step alone cannot join the trees;
    # with every proposed target outside the space, no tree ever explores around it.
    planner = _planner(collision=Wall(axis=0, lo=-0.2, hi=0.2), max_iterations=50)
    result = planner.solve(_problem(planner, sampler=Outside()), seed=0)
    assert not result.success
    assert result.failure_reason.startswith("Max iterations")
    start_tree_q0 = [n.config[0] for n in result.tree_start.nodes]
    assert (
        max(start_tree_q0) < -0.2 and min(start_tree_q0) >= -0.5
    )  # only straight toward the goal root, never past the wall
