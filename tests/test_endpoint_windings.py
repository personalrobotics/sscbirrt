# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""#212: a path ends on the nearest winding of a reached start or goal set member."""

import importlib.util
import os

import numpy as np
import pytest

from sscbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem
from sscbirrt.sets import Sample
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK

TWO_PI = 2 * np.pi
NEAR = np.array([0.5, 0.3])
FAR = NEAR - np.array([TWO_PI, 0.0])  # the same pose a full turn of joint 0 away


class WideArm(PlanarArm):
    """Joint 0 bounded at +-7 rad, so it admits full turns; joint 1 at +-1.5, so it does not."""

    @property
    def joint_limits(self):
        return np.array([-7.0, -1.5]), np.array([7.0, 1.5])


class FarWinding:
    """Members: ``target`` up to full turns of joint 0. The sampler offers only ``far``."""

    def __init__(self, target, far):
        self.target, self.far = np.asarray(target), np.asarray(far)

    def contains(self, q):
        d0 = np.remainder(q[0] - self.target[0] + np.pi, TWO_PI) - np.pi
        return abs(d0) < 1e-9 and abs(q[1] - self.target[1]) < 1e-9

    def sample(self, rng):
        return [Sample(self.far.copy(), ())]


class Box:
    """Invalid inside a box in raw joint values (so not full-turn invariant)."""

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)

    def is_valid(self, q):
        return not bool(np.all(q > self.lo) and np.all(q < self.hi))


def _planner(validator=None, **config):
    cfg = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, **config)
    return CBiRRT(WideArm(), PlanarIK(), validator or NoCollision(), cfg, backend="python")


def _solve(planner, start, goal, seed=1):
    problem = PlanningProblem(space=planner.space, start=start, goal=goal, validator=planner.collision)
    return planner.solve(problem, seed=seed), problem


def _length(path):
    return sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))


def _fixed(planner, q):
    return FiniteSet([np.asarray(q, dtype=float)], metric=planner.space.distance)


class TestGoal:
    def test_moves_to_the_nearest_winding_and_keeps_its_provenance(self):
        on, off = _planner(), _planner(shorten_endpoints=False)
        a, problem = _solve(on, _fixed(on, [0.0, 0.3]), FarWinding(NEAR, FAR))
        b, _ = _solve(off, _fixed(off, [0.0, 0.3]), FarWinding(NEAR, FAR))
        assert a.success and b.success
        assert np.array_equal(b.path[-1], FAR)
        assert np.allclose(a.path[-1], NEAR, atol=1e-12) and problem.goal.contains(a.path[-1])
        assert _length(a.path) < _length(b.path) - 1.0
        assert a.goal_source == b.goal_source and np.array_equal(a.path[0], b.path[0])
        assert all(on._admissible(problem, q)[0] for q in a.path)

    def test_a_finite_goal_never_moves_and_nothing_else_changes(self):
        on, off = _planner(), _planner(shorten_endpoints=False)
        a, _ = _solve(on, _fixed(on, [0.0, 0.3]), _fixed(on, FAR), seed=2)
        b, _ = _solve(off, _fixed(off, [0.0, 0.3]), _fixed(off, FAR), seed=2)
        assert a.success and np.array_equal(a.path[-1], FAR)
        assert len(a.path) == len(b.path) and all(np.array_equal(x, y) for x, y in zip(a.path, b.path))

    def test_a_blocked_near_winding_is_never_used(self):
        block = Box([0.3, 0.1], [0.7, 0.5])  # around NEAR, in raw joint values; FAR is outside it
        planner = _planner(validator=block)
        r, problem = _solve(planner, _fixed(planner, [0.0, 0.3]), FarWinding(NEAR, FAR))
        assert r.success and np.array_equal(r.path[-1], FAR)
        assert all(planner._admissible(problem, q)[0] for q in r.path)

    def test_a_declared_invariant_validator_skips_rechecks_but_not_results(self):
        class Invariant:
            full_turn_invariant = True

            def is_valid(self, q):
                return True

        plain, inv = _planner(), _planner(validator=Invariant())
        a, _ = _solve(plain, _fixed(plain, [0.0, 0.3]), FarWinding(NEAR, FAR), seed=3)
        b, _ = _solve(inv, _fixed(inv, [0.0, 0.3]), FarWinding(NEAR, FAR), seed=3)
        assert np.allclose(a.path[-1], NEAR) and np.allclose(b.path[-1], NEAR)
        assert len(a.path) == len(b.path) and all(np.allclose(x, y) for x, y in zip(a.path, b.path))
        assert b.stats["edge_checks"] <= a.stats["edge_checks"]

    def test_without_smoothing_nothing_moves(self):
        planner = _planner(smooth_path=False)
        r, _ = _solve(planner, _fixed(planner, [0.0, 0.3]), FarWinding(NEAR, FAR))
        assert np.array_equal(r.path[-1], FAR)

    def test_deterministic_for_a_seed(self):
        a, _ = _solve(_planner(), _fixed(_planner(), [0.0, 0.3]), FarWinding(NEAR, FAR), seed=5)
        b, _ = _solve(_planner(), _fixed(_planner(), [0.0, 0.3]), FarWinding(NEAR, FAR), seed=5)
        assert len(a.path) == len(b.path) and all(np.array_equal(x, y) for x, y in zip(a.path, b.path))


class TestStart:
    def test_a_start_set_moves_symmetrically_and_a_fixed_start_never_does(self):
        planner = _planner()
        r, problem = _solve(planner, FarWinding(NEAR, FAR), _fixed(planner, [0.0, 0.3]))
        assert r.success and np.allclose(r.path[0], NEAR, atol=1e-12) and problem.start.contains(r.path[0])
        assert np.array_equal(r.path[-1], [0.0, 0.3])


class TestWindings:
    def test_only_a_full_turn_that_unwinds_is_proposed(self):
        space = _planner().space
        start = np.array([0.0, 0.3])
        # joint 0 at 0.5 is within pi of the start: no turn to unwind, so nothing is proposed
        assert CBiRRT._windings(space, np.array([0.5, 0.3]), start) == []
        # at 0.5 - 2*pi it is a turn away: the one candidate unwinds it
        (only,) = CBiRRT._windings(space, np.array([0.5 - TWO_PI, 0.3]), start)
        assert only[0] == pytest.approx(0.5) and only[1] == 0.3
        # the turn that would unwind leaves the limits: nothing
        assert CBiRRT._windings(space, np.array([6.0, 0.3]), np.array([6.9, 0.3])) == []
        # +pi against -pi is a tie, not a turn to unwind
        assert CBiRRT._windings(space, np.array([np.pi, 0.3]), np.array([0.0, 0.3])) == []

    def test_continuous_joints_are_never_offset(self):
        planner = CBiRRT(
            WideArm(), PlanarIK(), NoCollision(), CBiRRTConfig(continuous_joints=(True, False)), backend="python"
        )
        assert CBiRRT._windings(planner.space, np.array([0.5 - TWO_PI, 0.3]), np.array([0.0, 0.3])) == []

    def test_an_abort_leaves_the_path_as_it_is(self):
        planner = _planner(abort_fn=lambda: True)
        problem = PlanningProblem(
            space=planner.space,
            start=_fixed(planner, [0.0, 0.3]),
            goal=FarWinding(NEAR, FAR),
            validator=planner.collision,
        )
        path = [np.array([0.0, 0.3]), FAR.copy()]
        assert planner._shorten_endpoints(problem, path) is path


def _have_demo():
    if importlib.util.find_spec("mujoco") is None:
        return False
    try:
        from sscbirrt.demo.scene import menagerie_path

        menagerie_path()
    except ImportError:
        return os.environ.get("MUJOCO_MENAGERIE_PATH") is not None
    return True


@pytest.mark.skipif(not _have_demo(), reason="needs sscbirrt[demo] (MuJoCo and the UR5e model)")
class TestUR5e:
    """The native core on the pick demo: #212's motivating case, seed 0's spinning wrist."""

    @pytest.mark.parametrize("seed", [0, 1, 6, 12])
    def test_pick_ends_on_a_nearer_winding_of_the_same_grasp(self, seed):
        import dataclasses

        from sscbirrt.demo.scenarios import pick
        from sscbirrt.mujoco import plan

        p = pick.problem()
        off = plan(p.model, p.data, p.arm, config=dataclasses.replace(p.config, shorten_endpoints=False), seed=seed,
                   **p.kwargs)  # fmt: skip
        p = pick.problem()
        on = plan(p.model, p.data, p.arm, config=p.config, seed=seed, **p.kwargs)
        assert on.backend == "native" and on.success and off.success
        assert on.goal_index == off.goal_index
        assert _length(on.path) < _length(off.path) - 1.0
        turns = (on.path[-1] - off.path[-1]) / TWO_PI  # the same search, so the same member, another winding
        assert np.allclose(turns, np.round(turns), atol=1e-9) and np.any(np.round(turns) != 0)
