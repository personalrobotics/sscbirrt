# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The public validator (#230): one implementation, used by the planner and by anyone checking its output."""

import importlib.metadata
import math

import numpy as np
import pytest

from sscbirrt import CBiRRT, CBiRRTConfig, JointSpace
from sscbirrt.backends import native
from sscbirrt.sets import AllOf, FiniteSet
from sscbirrt.testing import NoCollision, Wall
from sscbirrt.testing.planar import PlanarArm
from sscbirrt.trajectory import (
    Piece,
    Ramp,
    Trajectory,
    TrajectoryOptions,
    coalesce,
    initial_trajectory,
    synchronize,
)
from sscbirrt.validation import MalformedTrajectory, ValidationPolicy, Validator

BACKENDS = ["python"] + (["native"] if native.available() else [])


def plane(continuous=(False, False)):
    return JointSpace(np.full(2, -2.0), np.full(2, 2.0), continuous_joints=list(continuous))


def verdict(report):
    """What both implementations must agree on; floats compared separately."""
    w = report.witness
    head = (report.subject, report.valid, report.intervals, report.checks, report.policy, report.implementation)
    if w is None:
        return head, None
    return head, (w.failure, w.segment, w.fraction, w.constraint, w.contacts, w.configuration)


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


class TestPolicy:
    def test_every_setting_is_reported(self):
        p = ValidationPolicy(0.02)
        d = p.to_dict()
        assert (d["name"], d["mechanism"], d["guarantee"], d["resolution"]) == (
            "joint_l2_subdivision", "joint_subdivision", "discrete", 0.02,
        )  # fmt: skip
        assert d["parameters"] == sorted(d["parameters"]) and d["summary"]

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
    def test_a_resolution_must_be_positive_and_finite(self, bad):
        with pytest.raises(ValueError, match="resolution"):
            ValidationPolicy(bad)

    def test_the_planner_policy_comes_from_its_config(self):
        assert ValidationPolicy.from_config(CBiRRTConfig(edge_resolution=0.03)).resolution == 0.03
        assert ValidationPolicy.from_config(CBiRRTConfig(edge_resolution=None, step_size=0.2)).resolution == 0.2


class TestReports:
    def test_a_valid_motion_counts_its_states(self, backend):
        v = Validator(plane(), NoCollision(), policy=ValidationPolicy(0.1), backend=backend)
        r = v.validate_motion([0, 0], [1, 0])  # 10 intervals: q0 plus 10 states
        assert (r.valid, r.subject, r.intervals, r.checks, r.witness) == (True, "motion", 1, 11, None)
        assert r.backend == backend and r.interpolation == "raw_linear"

    def test_the_first_invalid_state_is_located(self, backend):
        v = Validator(plane(), Wall(0, 0.45, 0.55), policy=ValidationPolicy(0.1), backend=backend)
        r = v.validate_path([[0, 0], [0, 1], [1, 1]])
        w = r.witness
        assert not r.valid and r.intervals == 2
        assert (w.failure, w.segment, w.fraction, w.detail) == ("collision", 1, 0.5, "in collision")
        assert w.configuration == pytest.approx((0.5, 1.0))
        assert r.checks == 1 + 10 + 5  # q0, the first segment, then up to the hit

    def test_an_invalid_first_waypoint_is_segment_zero_fraction_zero(self, backend):
        v = Validator(plane(), Wall(0, -0.1, 0.1), backend=backend)
        w = v.validate_path([[0, 0], [1, 0]]).witness
        assert (w.segment, w.fraction, v.validate_path([[0, 0]]).witness.segment) == (0, 0.0, None)

    def test_limits_and_constraint_members_are_named(self, backend):
        v = Validator(plane(), NoCollision(), backend=backend)
        w = v.validate_state([2.5, 0]).witness
        assert w.failure == "position_limits" and "joint 0" in w.detail
        inside = AllOf([FiniteSet([[0, 0], [1, 0]], tolerance=0.5), FiniteSet([[0, 0]], tolerance=0.5)])
        v = Validator(plane(), NoCollision(), inside, backend=backend)
        assert v.validate_state([0, 0]).valid
        w = v.validate_state([1, 0]).witness
        assert (w.failure, w.constraint) == ("path_constraint", 1)
        assert Validator(plane(), NoCollision(), inside.children[0], backend=backend).validate_state([5, 5]).witness

    @pytest.mark.parametrize("bad", [[0.0], [0.0, math.nan], "ab"])
    def test_malformed_input_raises(self, backend, bad):
        v = Validator(plane(), NoCollision(), backend=backend)
        with pytest.raises(ValueError):
            v.validate_state(bad)
        with pytest.raises(ValueError):
            v.validate_motion([0, 0], bad)
        with pytest.raises(ValueError, match="empty"):
            v.validate_path([])

    def test_a_report_is_json_ready(self):
        import json

        r = Validator(plane(), Wall(0, -0.1, 0.1), backend="python").validate_motion([-1, 0], [1, 0])
        d = json.loads(json.dumps(r.to_dict()))
        assert d["witness"]["failure"] == "collision" and d["policy"]["guarantee"] == "discrete"


class TestRawLinear:
    """#230: the motion validated is the straight line in raw joint values, never the short way round."""

    def test_a_long_turn_is_validated_along_the_long_raw_line(self, backend):
        space = plane(continuous=(True, False))
        q0, q1 = [-0.75 * math.pi, 0.0], [0.75 * math.pi, 0.0]  # 1.5 turns of pi raw; the short way crosses +-pi
        on_long_arc = Validator(space, Wall(0, -0.1, 0.1), backend=backend)
        on_short_arc = Validator(space, Wall(0, 2.9, 3.3), backend=backend)
        assert not on_long_arc.validate_motion(q0, q1).valid
        assert on_short_arc.validate_motion(q0, q1).valid
        # the planner's edge rule takes the short way; the validator does not
        assert space.distance(q0, q1) < 0.51 * math.pi


class TestBackends:
    @pytest.mark.skipif(not native.available(), reason="needs sscbirrt._native")
    def test_native_and_python_agree(self):
        rng = np.random.default_rng(0)
        space = plane(continuous=(True, False))
        constraint = AllOf([FiniteSet([[0, 0]], tolerance=10.0), FiniteSet([[0, 0]], tolerance=1.9)])
        py = Validator(space, Wall(1, 0.3, 0.4), constraint, policy=ValidationPolicy(0.07), backend="python")
        nat = Validator(space, Wall(1, 0.3, 0.4), constraint, policy=ValidationPolicy(0.07), backend="native")
        assert (py.backend, nat.backend) == ("python", "native")
        for _ in range(200):
            path = [rng.uniform(-1.9, 1.9, 2) + [4 * rng.integers(-1, 2), 0] for _ in range(rng.integers(1, 5))]
            a, b = py.validate_path(path), nat.validate_path(path)
            (ha, wa), (hb, wb) = verdict(a), verdict(b)
            assert ha == hb
            assert (wa is None) == (wb is None)
            if wa is not None:
                assert wa[:5] == wb[:5]
                assert np.allclose(wa[5], wb[5], atol=1e-12)

    @pytest.mark.skipif(not native.available(), reason="needs sscbirrt._native")
    def test_the_native_core_is_this_release(self):
        assert native._native.__version__ == importlib.metadata.version("sscbirrt")

    def test_auto_falls_back_with_reasons(self):
        class PythonChecker:
            def is_valid(self, q):
                return True

        v = Validator(plane(), PythonChecker())
        assert v.backend == "python"
        if native.available():
            assert any("PythonChecker" in r for r in v.backend_reasons)

    def test_repeated_and_reordered_queries_agree(self, backend):
        v = Validator(plane(), Wall(0, 0.2, 0.3), backend=backend)
        motions = [([0, 0], [1, 1]), ([-1, 0], [0, 0]), ([0.25, 0], [0.25, 1]), ([1, 1], [1, -1])]
        first = [v.validate_motion(*m) for m in motions]
        again = [v.validate_motion(*m) for m in reversed(motions)][::-1]
        assert first == again


class TestPlanner:
    """A plan succeeds only on a path the public validator accepts, and its report is the public call's."""

    def test_the_result_carries_the_public_report(self, backend):
        config = CBiRRTConfig(timeout=10.0, edge_resolution=0.03)
        checker = Wall(0, -0.1, 0.1, extent=1.0)
        planner = CBiRRT(PlanarArm(), collision_checker=checker, config=config, backend=backend)
        r = planner.plan(start=np.array([-1.0, 0.0]), goal=np.array([1.0, 0.0]), seed=1, return_details=True)
        assert r.success and r.backend == backend
        v = Validator(planner.space, checker, policy=ValidationPolicy.from_config(config), backend=backend)
        assert r.validation == v.validate_path(r.path)
        assert r.validation.valid and r.validation.intervals == len(r.path) - 1

    def test_a_path_the_validator_rejects_is_not_returned(self, backend):
        """A checker over raw values on a continuous joint is not full-turn invariant. The trees meet across +-pi, so
        their nodes near the goal are stored around -3.0, where every wrapped edge sample is valid; the path returned
        is unwrapped, which moves those waypoints a turn up into the raw wall. The plan fails with the report rather
        than return that path."""
        config = CBiRRTConfig(continuous_joints=[True, False], timeout=5.0, smooth_path=False)
        checker = Wall(0, 3.25, 3.3)
        planner = CBiRRT(PlanarArm(), collision_checker=checker, config=config, backend=backend)
        r = planner.plan(start=np.array([3.0, 0.0]), goal=np.array([-3.0, 0.0]), seed=0, return_details=True)
        assert not r.success and r.path is None
        assert r.failure_reason.startswith("Returned path failed final validation: collision at segment")
        w = r.validation.witness
        assert (w.failure, w.fraction) == ("collision", 1.0)
        assert 3.25 < w.configuration[0] < 3.3


ARM_XML = """
<mujoco><compiler angle="radian"/><worldbody>
  <geom name="wall" type="box" pos="0.47 0 0.3" size="0.05 0.5 0.3"/>
  <body name="arm/base" pos="0 0 0.1"><joint name="j0" type="hinge" axis="0 0 1" limited="true" range="-3 3"/>
    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
    <body name="arm/link1" pos="0 0 0.1"><joint name="j1" type="hinge" axis="0 1 0" limited="true" range="-2 2"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0.4 0 0"/>
      <body name="arm/gripper" pos="0.4 0 0"><geom type="box" size="0.02 0.03 0.02"/></body>
    </body>
  </body>
</worldbody></mujoco>"""


def test_a_collision_witness_names_the_contacts(backend):
    mujoco = pytest.importorskip("mujoco")
    from sscbirrt.backends import native_mujoco as nm

    if not nm.available():
        pytest.skip(nm.unavailable_reason())
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    scene = nm.NativeScene.from_model(model, ["j0", "j1"])
    checker = nm.NativeCollisionChecker(scene, nm.Snapshot.capture(scene, mujoco.MjData(model)))
    space = JointSpace(np.array([-3.0, -2.0]), np.array([3.0, 2.0]))
    v = Validator(space, checker, backend=backend)
    assert v.backend == backend
    r = v.validate_motion([1.5, 0.0], [0.0, 0.0])  # swings the arm into the wall
    w = r.witness
    assert w.failure == "collision" and w.segment == 0 and 0 < w.fraction <= 1
    assert w.contacts and all(c.kind == "robot_environment" and c.distance < 0.02 for c in w.contacts)
    assert {"arm/gripper", "arm/link1"} & {c.body1 for c in w.contacts} | {c.body2 for c in w.contacts}
    assert dict(r.provenance)["snapshot_sha256"] == checker.snapshot.sha256
    if backend == "native":  # the Python reference reaches the same witness through invalid_contacts
        assert verdict(Validator(space, checker, backend="python").validate_motion([1.5, 0.0], [0.0, 0.0])) == verdict(
            r
        )


def test_a_scene_plan_report_is_the_public_report(backend):
    """With a native MuJoCo checker the report carries the snapshot it validated against, on either backend."""
    mujoco = pytest.importorskip("mujoco")
    from sscbirrt.backends import native_mujoco as nm

    if not nm.available():
        pytest.skip(nm.unavailable_reason())
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    scene = nm.NativeScene.from_model(model, ["j0", "j1"])
    checker = nm.NativeCollisionChecker(scene, nm.Snapshot.capture(scene, mujoco.MjData(model)))
    config = CBiRRTConfig(timeout=10.0, edge_resolution=0.02)
    planner = CBiRRT(PlanarArm(), collision_checker=checker, config=config, backend=backend)
    r = planner.plan(start=np.array([1.5, -1.0]), goal=np.array([-1.5, -1.0]), seed=0, return_details=True)
    assert r.success and r.backend == backend
    public = Validator(planner.space, checker, policy=ValidationPolicy.from_config(config), backend=backend)
    assert r.validation == public.validate_path(r.path)
    assert dict(r.validation.provenance)["snapshot_sha256"] == checker.snapshot.sha256


def test_a_custom_motion_validator_owns_its_motions():
    """It replaces the edge rule, so no ValidationPolicy describes the path: no report, and for_problem refuses."""
    import dataclasses

    from sscbirrt import FiniteSet as Finite
    from sscbirrt import PlanningProblem

    planner = CBiRRT(PlanarArm(), collision_checker=NoCollision(), config=CBiRRTConfig(timeout=5.0), backend="python")
    base = PlanningProblem(
        space=planner.space, start=Finite([[-1.0, 0.0]]), goal=Finite([[1.0, 0.0]]), validator=NoCollision()
    )
    problem = dataclasses.replace(base, motion_validator=planner.default_motion_validator(base))
    r = planner.solve(problem, seed=0)
    assert r.success and r.validation is None
    with pytest.raises(ValueError, match="replaces the default"):
        Validator.for_problem(problem)


def test_a_path_of_planner_samples_checks_no_new_state(backend):
    """The planner stores every edge sample as a node: pieces exactly one resolution long. Recomputed from the
    nodes, a piece can come out a rounding error longer; it must still be one interval, not a midpoint nobody
    checked (found by the door benchmark: a TSR-chain path rejected at fraction 0.5)."""
    rng = np.random.default_rng(3)
    v = Validator(JointSpace(np.full(6, -7.0), np.full(6, 7.0)), NoCollision(), policy=ValidationPolicy(0.05),
                  backend=backend)  # fmt: skip
    for _ in range(200):
        a = rng.uniform(-3, 3, 6)
        d = rng.normal(size=6)
        d *= 0.1 / np.linalg.norm(d)  # one planner step: two samples, each exactly one resolution
        path = [a, a + 0.5 * d, a + d]
        assert v.validate_path(path).checks == 3


# ----- timed trajectories --------------------------------------------------------------------------------------------

V2, A2 = np.array([1.0, 1.0]), np.array([2.0, 2.0])


def dense(points, step=0.05):
    """A planner-like dense path through ``points``: every edge cut into samples at most ``step`` apart."""
    from sscbirrt.validation import line_point, subdivisions

    out = [np.asarray(points[0], dtype=float)]
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = subdivisions(float(np.linalg.norm(b - a)), step)
        out += [b if i == n else line_point(a, b - a, i, n) for i in range(1, n + 1)]
    return out


def timed(path):
    traj = initial_trajectory(coalesce(path), V2, A2)
    traj.path = path
    return traj


class TestTrajectories:
    def test_pieces_on_the_source_path_reuse_its_states(self, backend):
        path = dense([[-1.0, 0.0], [0.0, 0.5], [1.0, 0.5]])
        v = Validator(plane(), NoCollision(), backend=backend)
        r = v.validate_trajectory(timed(path))
        assert r.valid and r.subject == "trajectory" and r.law == "piecewise_constant_acceleration"
        assert r.intervals == 2 and r.duration_s > 0
        assert r.checks == v.validate_path(path).checks  # exactly the path's states, no new one

    def test_without_its_path_a_trajectory_is_sampled_in_time(self, backend):
        path = dense([[-1.0, 0.0], [1.0, 0.0]])
        traj = timed(path)
        traj.path = None
        r = Validator(plane(), NoCollision(), backend=backend).validate_trajectory(traj)
        speed = np.linalg.norm(V2)
        assert r.valid and r.checks == 1 + math.ceil(traj.duration * speed / 0.05 - 1e-9)

    def test_a_shortcut_through_an_obstacle_is_found_in_time(self, backend):
        piece = synchronize(np.array([-1.0, 0.0]), np.zeros(2), np.array([1.0, 0.3]), np.zeros(2), V2, A2)
        traj = Trajectory([Piece(piece.ramps, "shortcut")], V2, A2)
        r = Validator(plane(), Wall(0, -0.05, 0.05), backend=backend).validate_trajectory(traj)
        w = r.witness
        assert not r.valid and w.failure == "collision" and w.segment == 0
        assert 0 < w.time_s < traj.duration
        assert np.allclose(traj.state(w.time_s)[0], w.configuration, atol=1e-12)
        assert -0.05 < w.configuration[0] < 0.05

    def test_position_limits_hold_between_samples(self, backend):
        # Joint 0 rises from 1.9 to 2.1 (beyond its limit of 2) at t = 2 s and returns, at rest at both ends. A coarse
        # policy samples only the end; the turning point is checked exactly.
        a = 0.2
        up = Ramp(1.9, 0.0, ((1.0, a), (2.0, -a), (1.0, a)))
        still = Ramp(0.0, 0.0, ((1.0, 0.0), (2.0, 0.0), (1.0, 0.0)))
        traj = Trajectory([Piece((up, still), "shortcut")], V2, A2)
        r = Validator(plane(), NoCollision(), policy=ValidationPolicy(100.0), backend=backend).validate_trajectory(traj)
        w = r.witness
        assert (w.failure, w.time_s) == ("position_limits", pytest.approx(2.0))
        assert w.configuration[0] == pytest.approx(2.1)

    @pytest.mark.parametrize("breakage", ["jump", "moving", "fast", "ramps"])
    def test_a_trajectory_that_breaks_its_own_invariants_is_refused(self, backend, breakage):
        p1 = synchronize(np.zeros(2), np.zeros(2), np.array([0.5, 0.0]), np.zeros(2), V2, A2)
        p2 = synchronize(np.array([0.5, 0.0]), np.zeros(2), np.array([1.0, 0.0]), np.zeros(2), V2, A2)
        pieces = [p1, p2]
        if breakage == "jump":
            pieces[1] = synchronize(np.array([0.6, 0.0]), np.zeros(2), np.array([1.0, 0.0]), np.zeros(2), V2, A2)
        if breakage == "moving":
            pieces = [p1, Piece(tuple(r.split(r.duration / 2)[0] for r in p2.ramps))]
        if breakage == "fast":
            pieces = [p1, p2]
            V = V2 / 10
        if breakage == "ramps":
            pieces[1] = Piece(p2.ramps[:1])
        traj = Trajectory(pieces, V if breakage == "fast" else V2, A2)
        with pytest.raises(MalformedTrajectory):
            Validator(plane(), NoCollision(), backend=backend).validate_trajectory(traj)

    @pytest.mark.skipif(not native.available(), reason="needs sscbirrt._native")
    def test_native_and_python_agree_on_trajectories(self):
        rng = np.random.default_rng(1)
        for _ in range(40):
            pts = [rng.uniform(-1.5, 1.5, 2) for _ in range(rng.integers(2, 5))]
            traj = timed(dense(pts))
            if rng.random() < 0.5:  # a shortcut spliced in, as the smoother does
                t1, t2 = np.sort(rng.uniform(0, traj.duration, 2))
                qa, va, _ = traj.state(t1)
                qb, vb, _ = traj.state(t2)
                piece = synchronize(qa, va, qb, vb, V2, A2)
                if piece is not None and piece.duration < t2 - t1:
                    traj = traj.splice(t1, t2, Piece(piece.ramps, "shortcut"))
            wall = Wall(0, rng.uniform(-1, 1), 0.0)
            wall.hi = wall.lo + 0.1
            a = Validator(plane(), wall, backend="python").validate_trajectory(traj)
            b = Validator(plane(), wall, backend="native").validate_trajectory(traj)
            (ha, wa), (hb, wb) = verdict(a), verdict(b)
            assert ha == hb and (wa is None) == (wb is None)
            if wa is not None:
                assert wa[:3] == wb[:3] and np.allclose(wa[5], wb[5], atol=1e-12)
                assert a.witness.time_s == pytest.approx(b.witness.time_s, abs=1e-12)


class TestTimedPlans:
    def test_a_returned_trajectory_carries_the_public_report(self, backend):
        config = CBiRRTConfig(timeout=10.0, edge_resolution=0.03)
        checker = Wall(0, -0.1, 0.1, extent=1.0)
        planner = CBiRRT(PlanarArm(), collision_checker=checker, config=config, backend=backend)
        options = TrajectoryOptions(velocity_limits=1.0, acceleration_limits=2.0)
        r = planner.plan(start=np.array([-1.0, 0.0]), goal=np.array([1.0, 0.0]), seed=2, trajectory=options,
                         return_details=True)  # fmt: skip
        traj = r.trajectory
        assert r.success and traj.backend == backend and traj.diagnostics["accepted"] > 0
        assert traj.validation.valid and traj.path is not None
        v = Validator(planner.space, checker, policy=ValidationPolicy.from_config(config), backend=backend)
        assert traj.validation == v.validate_trajectory(traj)


def test_the_spread_order_visits_every_state_once_middle_first():
    from sscbirrt.validation import spread_order

    for n in range(1, 300):
        order = spread_order(n)
        assert sorted(order) == list(range(n))
        if native.available():
            assert native._native.spread_order(n) == order
    assert [i + 1 for i in spread_order(9)] == [5, 3, 7, 2, 4, 6, 8, 1, 9]


def test_a_spread_check_has_the_same_verdict_and_finds_an_obstacle_sooner(backend):
    piece = synchronize(np.array([-1.0, 0.0]), np.zeros(2), np.array([1.0, 0.3]), np.zeros(2), V2, A2)
    v = Validator(plane(), Wall(0, -0.05, 0.05), policy=ValidationPolicy(0.01), backend="python")
    earliest = v.check_piece(piece, 0, 0.0, V2, None)  # a report: refined to the earliest invalid state
    whole = v.check_piece(piece, 0, 0.0, V2, None, first=False)  # the verdict only
    assert earliest[1].failure == whole[1].failure == "collision"
    assert whole[0] < earliest[0]  # the wall is mid-way: the first probe lands near it
