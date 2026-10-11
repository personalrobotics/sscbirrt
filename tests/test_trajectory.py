# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""#214: timed trajectories with time-optimized shortcuts (sscbirrt.trajectory), in both backends."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from sscbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem, PredicateSet
from sscbirrt.backends import native
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK, Wall
from sscbirrt.trajectory import (
    Trajectory,
    TrajectoryOptions,
    coalesce,
    fixed_time,
    initial_trajectory,
    min_time,
    piece_from_native,
    ramp_from_native,
    shortcut,
    synchronize,
    verify,
)

ROOT = Path(__file__).resolve().parent.parent


def _tool():
    spec = importlib.util.spec_from_file_location("trajectory_benchmark", ROOT / "tools" / "trajectory_benchmark.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules["trajectory_benchmark"] = tool
    spec.loader.exec_module(tool)
    return tool


def _cases(n, seed, moving=True):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        vmax, amax = rng.uniform(0.3, 3), rng.uniform(0.5, 8)
        x0, x1 = rng.uniform(-3, 3, 2)
        v0, v1 = rng.uniform(-vmax, vmax, 2) if moving else (0.0, 0.0)
        yield float(x0), float(v0), float(x1), float(v1), vmax, amax


class TestRamps:
    def test_min_time_is_valid_and_no_valid_ramp_is_faster(self):
        for x0, v0, x1, v1, vmax, amax in _cases(3000, 0):
            r = min_time(x0, v0, x1, v1, vmax, amax)
            assert r is not None and verify(r, x1, v1, vmax, amax)
            for f in (0.9, 0.98):  # an independent solver finds nothing faster
                assert fixed_time(x0, v0, x1, v1, f * r.duration, vmax, amax) is None

    def test_fixed_time_is_exact_when_it_answers_and_complete_from_rest(self):
        for moving in (False, True):
            for x0, v0, x1, v1, vmax, amax in _cases(2000, 1, moving):
                T = min_time(x0, v0, x1, v1, vmax, amax).duration
                for f in (1.1, 1.5, 3.0):
                    r = fixed_time(x0, v0, x1, v1, f * T, vmax, amax)
                    if r is not None:
                        assert verify(r, x1, v1, vmax, amax, f * T)
                    else:
                        assert moving  # from rest to rest every longer duration is feasible

    def test_split_preserves_the_motion(self):
        r = min_time(0.0, 0.5, 2.0, -0.3, 1.0, 2.0)
        a, b = r.split(r.duration * 0.37)
        assert a.duration + b.duration == pytest.approx(r.duration)
        assert a.end() == pytest.approx((b.x0, b.v0))
        assert b.end() == pytest.approx(r.end())


class TestSynchronization:
    def test_every_joint_shares_one_duration_within_its_limits(self):
        rng = np.random.default_rng(2)
        vmax, amax = np.array([1.0, 2.0, 0.5]), np.array([3.0, 1.0, 2.0])
        for _ in range(300):
            q0, q1 = rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)
            v0, v1 = rng.uniform(-1, 1, 3) * vmax, rng.uniform(-1, 1, 3) * vmax
            piece = synchronize(q0, v0, q1, v1, vmax, amax)
            if piece is None:
                continue
            T = piece.duration
            for j, ramp in enumerate(piece.ramps):
                assert verify(ramp, q1[j], v1[j], vmax[j], amax[j], T)


def _planner(backend="auto", **config):
    config = CBiRRTConfig(step_size=0.1, edge_resolution=0.02, **config)
    return CBiRRT(PlanarArm(), PlanarIK(), NoCollision(), config, backend=backend)


def _problem(planner, path, validator=None, constraint=None):
    return PlanningProblem(
        space=planner.space,
        start=FiniteSet([path[0]], metric=planner.space.distance),
        goal=FiniteSet([path[-1]], metric=planner.space.distance),
        validator=validator or NoCollision(),
        path_constraint=constraint,
    )


def _dense(*corners, step=0.05):
    out = [np.asarray(corners[0], dtype=float)]
    for a, b in zip(corners, corners[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(1, math.ceil(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return out


L2 = np.array([1.0, 2.0])
VMAX, AMAX = np.array([1.0, 1.0]), np.array([2.0, 2.0])
U_PATH = _dense([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])  # three straight runs, two corners


def _opts(**kw):
    return TrajectoryOptions(VMAX, AMAX, **kw)


def _shortcut(planner, problem, traj, options, seed=0):
    """The Python reference loop, whatever the planner's backend."""
    return shortcut(
        traj,
        motion_validator=planner._motion_validator(problem),
        admissible=lambda q: planner._admissible(problem, q),
        resolution=planner.config.edge_resolution,
        options=options,
        seed=seed,
    )


def _smooth(planner, problem, traj, iterations=200, seed=0):  # a fixed budget in the reference
    return _shortcut(planner, problem, traj, _opts(iterations=iterations, convergence_window=None), seed)


class TestTrajectory:
    def test_initial_trajectory_is_the_polyline_at_rest_at_each_corner(self):
        corners = coalesce(U_PATH)
        traj = initial_trajectory(corners, VMAX, AMAX)
        q, v, a = traj.evaluate(np.linspace(0, traj.duration, 400))
        on_path = [min(np.linalg.norm(np.cross(np.append(c1 - c0, 0), np.append(x - c0, 0))) for c0, c1 in
                       zip(corners, corners[1:])) for x in q]  # fmt: skip
        assert max(on_path) < 1e-9 and np.abs(v).max() <= 1.0 + 1e-9 and np.abs(a).max() <= 2.0 + 1e-9
        for t in traj.breaks:  # at rest at every corner
            assert np.abs(traj.state(t)[1]).max() < 1e-9

    def test_shortcuts_are_faster_valid_and_keep_endpoints_at_rest(self):
        planner = _planner()
        problem = _problem(planner, U_PATH)
        traj0 = initial_trajectory(coalesce(U_PATH), VMAX, AMAX)
        traj = _smooth(planner, problem, traj0)
        assert traj.duration < traj0.duration and traj.diagnostics["accepted"] > 0
        (q0, v0, _), (q1, v1, _) = traj.state(0.0), traj.state(traj.duration)
        assert np.allclose(q0, U_PATH[0]) and np.allclose(q1, U_PATH[-1])
        assert np.abs(v0).max() < 1e-9 and np.abs(v1).max() < 1e-9
        _, v, a = traj.evaluate(np.linspace(0, traj.duration, 2000))
        assert np.abs(v).max() <= 1.0 * (1 + 1e-9) and np.abs(a).max() <= 2.0 * (1 + 1e-9)

    def test_a_shortcut_never_crosses_an_obstacle(self):
        """The guarantee is the planner's: samples at most edge_resolution apart are valid, so a motion can graze an
        obstacle's corner between samples but never get deeper into it than that resolution."""
        planner = _planner()
        res = planner.config.edge_resolution

        class Block:  # fills the inside of the U except a margin around the path
            def is_valid(self, q):
                return not (0.1 < q[0] < 0.9 and 0.1 < q[1] < 0.9)

        problem = _problem(planner, U_PATH, validator=Block())
        traj = _smooth(planner, problem, initial_trajectory(coalesce(U_PATH), VMAX, AMAX), iterations=300)
        q, _, _ = traj.evaluate(np.linspace(0, traj.duration, 4000))
        deep = [x for x in q if 0.1 + res < x[0] < 0.9 - res and 0.1 + res < x[1] < 0.9 - res]
        assert deep == []
        assert any("in collision" in str(a.get("rejected")) for a in traj.diagnostics["attempts"])

    def test_path_constraint_is_respected(self):
        planner = _planner()
        band = PredicateSet(lambda q: q[1] >= -0.02 and q[1] <= 1.02 and q[0] >= -0.02 and q[0] <= 1.02)
        problem = _problem(planner, U_PATH, constraint=band)
        traj = _smooth(planner, problem, initial_trajectory(coalesce(U_PATH), VMAX, AMAX), iterations=300)
        q, _, _ = traj.evaluate(np.linspace(0, traj.duration, 4000))
        assert all(band.contains(x) for x in q)

    def test_deterministic_for_a_seed(self):
        planner = _planner()
        problem = _problem(planner, U_PATH)
        traj0 = initial_trajectory(coalesce(U_PATH), VMAX, AMAX)
        a, b = _smooth(planner, problem, traj0, seed=3), _smooth(planner, problem, traj0, seed=3)
        ts = np.linspace(0, a.duration, 50)
        assert a.duration == b.duration and np.array_equal(a.evaluate(ts)[0], b.evaluate(ts)[0])

    def test_every_piece_carries_its_validation_record(self):
        """Validation is a record: planned pieces were checked by the planner, shortcut pieces by the smoother."""
        planner = _planner()
        problem = _problem(planner, U_PATH)
        traj0 = initial_trajectory(coalesce(U_PATH), VMAX, AMAX)
        assert {p.validated for p in traj0.pieces} == {"planned"}
        traj = _smooth(planner, problem, traj0)
        assert {p.validated for p in traj.pieces} <= {"planned", "shortcut"} and "shortcut" in {
            p.validated for p in traj.pieces
        }

    def test_sample_never_leaves_a_sliver_of_a_final_interval(self):
        traj = initial_trajectory(coalesce(U_PATH), VMAX, AMAX)
        for dt in (2_000_000, 3_000_000, 7_000_000):
            times, _ = traj.sample(dt)
            assert times[-1] - times[-2] >= dt // 2

    def test_sample_gives_integer_ns_ending_at_the_goal(self):
        traj = initial_trajectory(coalesce(U_PATH), VMAX, AMAX)
        times, q = traj.sample(2_000_000)
        assert times[0] == 0 and all(isinstance(t, int) for t in times) and all(b > a for a, b in zip(times, times[1:]))
        assert times[-1] == math.ceil(traj.duration * 1e9) and np.allclose(q[-1], U_PATH[-1])


class TestRuntimeCommand:
    def test_sampled_smoother_output_passes_the_runtime_checks(self):
        tool = _tool()
        planner = _planner()
        problem = _problem(planner, U_PATH)
        traj = _smooth(planner, problem, initial_trajectory(coalesce(U_PATH), VMAX, AMAX))
        for dt in (2_000_000, 1_000_000, 7_000_000):
            times, q = traj.sample(dt)
            assert tool.runtime_violations(times, q, 1.0, 2.0) == []

    def test_the_port_catches_violations(self):
        tool = _tool()
        times = [0, 1_000_000_000, 2_000_000_000]
        assert tool.runtime_violations(times, np.array([[0.0], [2.0], [2.0]]), 1.0, 10.0)  # 2 rad/s > 1
        assert tool.runtime_violations(times, np.array([[0.0], [0.9], [0.9]]), 1.0, 0.5)  # jerky start

    def test_port_agrees_with_ssrobot(self):
        validation = pytest.importorskip("ssrobot.validation")
        from types import SimpleNamespace

        tool = _tool()
        rng = np.random.default_rng(4)
        for _ in range(200):
            n = int(rng.integers(3, 8))
            times = [0] + list(np.cumsum(rng.integers(1_000_000, 50_000_000, n - 1)).astype(int).tolist())
            q = np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)
            q[0] = 0.0
            ours = bool(tool.runtime_violations(times, q, 1.0, 4.0))
            desc = SimpleNamespace(
                joint=lambda j: SimpleNamespace(limits=SimpleNamespace(velocity=1.0, acceleration=4.0))
            )
            traj = SimpleNamespace(joints=("a", "b"), time_from_start_ns=tuple(times), positions=tuple(map(tuple, q)))
            theirs = False
            try:
                validation._check_speeds(desc, traj)
                validation._check_accelerations(desc, traj)
            except validation.ValidationError:
                theirs = True
            assert ours == theirs


needs_native = pytest.mark.skipif(not native.available(), reason="sscbirrt._native is not built")


def _same_ramp(a, b):
    if a is None or b is None:
        return a is None and b is None
    pairs = list(zip((a.x0, a.v0, *(x for p in a.pieces for x in p)), (b.x0, b.v0, *(x for p in b.pieces for x in p))))
    return len(a.pieces) == len(b.pieces) and all(abs(x - y) <= 1e-12 for x, y in pairs)


@needs_native
class TestNativeParity:
    """One contract, two implementations: the native ramps, synchronization and corners equal Python's."""

    def test_ramps_match(self):
        n = native._native
        for x0, v0, x1, v1, vmax, amax in _cases(2000, 5):
            py = min_time(x0, v0, x1, v1, vmax, amax)
            nat = n.ramp_min_time(x0, v0, x1, v1, vmax, amax)
            assert _same_ramp(py, None if nat is None else ramp_from_native(nat))
            for f in (0.95, 1.0, 1.03, 1.5, 3.0):  # below the minimum (None), at it, and above
                T = f * py.duration
                nat = n.ramp_fixed_time(x0, v0, x1, v1, T, vmax, amax)
                assert _same_ramp(
                    fixed_time(x0, v0, x1, v1, T, vmax, amax), None if nat is None else ramp_from_native(nat)
                )

    def test_synchronization_and_corners_match(self):
        rng = np.random.default_rng(6)
        vmax, amax = np.array([1.0, 2.0, 0.5]), np.array([3.0, 1.0, 2.0])
        for _ in range(500):
            q0, q1 = rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)
            v0, v1 = rng.uniform(-1, 1, 3) * vmax, rng.uniform(-1, 1, 3) * vmax
            py = synchronize(q0, v0, q1, v1, vmax, amax)
            nat = native._native.synchronize(*(list(map(float, x)) for x in (q0, v0, q1, v1, vmax, amax)))
            assert (py is None) == (nat is None)
            if py is not None:
                assert all(_same_ramp(a, b) for a, b in zip(py.ramps, piece_from_native(nat).ramps))
        path = U_PATH + [U_PATH[-1]]  # a repeated waypoint is dropped
        assert np.array_equal(np.array(native._native.coalesce([list(q) for q in path])), np.array(coalesce(path)))


SHIFTED_U = [q + np.array([0.0, -0.5]) for q in U_PATH]  # around the block Wall(0, 0.1, 0.9, extent=0.4)


def _native_smooth(planner, problem, path=U_PATH, iterations=200, seed=0):  # a fixed budget, natively
    traj = _planner(backend="native").smooth_trajectory(
        problem, path, _opts(iterations=iterations, convergence_window=None, seed=seed)
    )
    assert traj.backend == "native"
    return traj


@needs_native
class TestNativeSmoother:
    def test_faster_valid_recorded_and_at_rest(self):
        tool = _tool()
        planner = _planner()
        traj = _native_smooth(planner, _problem(planner, U_PATH))
        assert traj.duration < initial_trajectory(coalesce(U_PATH), VMAX, AMAX).duration
        assert traj.diagnostics["accepted"] > 0 and len(traj.diagnostics["attempts"]) == 200
        records = {p.validated for p in traj.pieces}
        assert records <= {"planned", "shortcut"} and "shortcut" in records
        (q0, v0, _), (q1, v1, _) = traj.state(0.0), traj.state(traj.duration)
        assert np.allclose(q0, U_PATH[0], atol=1e-12) and np.allclose(q1, U_PATH[-1], atol=1e-9)
        assert np.abs(v0).max() < 1e-9 and np.abs(v1).max() < 1e-9
        times, q = traj.sample(2_000_000)
        assert tool.runtime_violations(times, q, 1.0, 2.0) == []

    def test_deterministic_for_a_seed(self):
        planner = _planner()
        problem = _problem(planner, U_PATH)
        a, b = _native_smooth(planner, problem, seed=4), _native_smooth(planner, problem, seed=4)
        ts = np.linspace(0, a.duration, 50)
        assert a.duration == b.duration and np.array_equal(a.evaluate(ts)[0], b.evaluate(ts)[0])

    def test_never_deeper_into_an_obstacle_than_the_resolution(self):
        planner = _planner()
        res = planner.config.edge_resolution
        problem = _problem(planner, SHIFTED_U, validator=Wall(0, 0.1, 0.9, extent=0.4))
        traj = _native_smooth(planner, problem, path=SHIFTED_U, iterations=300)
        q, _, _ = traj.evaluate(np.linspace(0, traj.duration, 4000))
        assert not [x for x in q if 0.1 + res < x[0] < 0.9 - res and abs(x[1]) < 0.4 - res]
        assert any("in collision" in str(a.get("rejected")) for a in traj.diagnostics["attempts"])

    def test_durations_match_the_reference_in_distribution(self):
        """Only the random stream differs, so over seeds the two smoothers reach similar durations."""
        planner = _planner()
        problem = _problem(planner, SHIFTED_U, validator=Wall(0, 0.1, 0.9, extent=0.4))
        traj0 = initial_trajectory(coalesce(SHIFTED_U), VMAX, AMAX)
        py = [_smooth(planner, problem, traj0, iterations=100, seed=s).duration for s in range(20)]
        nat = [_native_smooth(planner, problem, path=SHIFTED_U, iterations=100, seed=s).duration for s in range(20)]
        assert abs(np.median(py) - np.median(nat)) < 0.1 * np.median(py)


class TestStopping:
    """The defaults stop on convergence, a rule that depends only on the trajectory, never on the machine."""

    def _check(self, traj):
        d, o = traj.diagnostics, traj.options
        n = len(d["attempts"])
        assert traj.stopped == "converged" and o.convergence_window <= n < o.iterations
        gains = [a.get("gain", 0.0) if a.get("accepted") else 0.0 for a in d["attempts"]]
        recent = sum(gains[n - o.convergence_window :])  # what the last window of attempts saved
        assert recent < o.convergence_tolerance * traj.duration * (1 + 1e-9)

    def test_defaults_stop_once_converged_in_both_backends(self):
        planner = _planner()
        problem = _problem(planner, SHIFTED_U, validator=Wall(0, 0.1, 0.9, extent=0.4))
        traj0 = initial_trajectory(coalesce(SHIFTED_U), VMAX, AMAX)
        self._check(_shortcut(planner, problem, traj0, _opts()))
        if native.available():
            nat = planner.smooth_trajectory(problem, SHIFTED_U, _opts())
            assert nat.backend == "native"
            self._check(nat)
            again = planner.smooth_trajectory(problem, SHIFTED_U, _opts())
            assert again.duration == nat.duration and len(again.diagnostics["attempts"]) == len(
                nat.diagnostics["attempts"]
            )

    def test_disabled_runs_every_iteration(self):
        planner = _planner()
        traj = _smooth(planner, _problem(planner, U_PATH), initial_trajectory(coalesce(U_PATH), VMAX, AMAX), 150)
        assert len(traj.diagnostics["attempts"]) == 150 and traj.stopped == "iterations"

    @needs_native
    def test_native_defaults_equal_the_reference(self):
        from dataclasses import asdict

        ours = asdict(TrajectoryOptions(1.0, 1.0))
        theirs = native._native.shortcut_defaults()
        for name in ("velocity_limits", "acceleration_limits", "seed"):  # required, or resolved at the call
            ours.pop(name)
        assert theirs.pop("seed") == 0  # a direct call's seed
        assert theirs == ours


class Block:  # a Python validator: no native form
    def is_valid(self, q):
        return not (0.1 < q[0] < 0.9 and 0.1 < q[1] < 0.9)


def _valid(traj, problem, planner):
    """Every piece recorded, at rest at both ends, within the limits, and every dense sample admissible."""
    assert all(p.validated in ("planned", "shortcut") for p in traj.pieces)
    (_, v0), (_, v1) = traj.start, traj.goal
    assert np.abs(v0).max() < 1e-9 and np.abs(v1).max() < 1e-9
    q, v, a = traj.evaluate(np.linspace(0, traj.duration, 2000))
    assert (np.abs(v) <= traj.velocity_limits * (1 + 1e-9)).all()
    assert (np.abs(a) <= traj.acceleration_limits * (1 + 1e-9)).all()
    r = traj.resolution
    deep = [x for x in q if not planner._admissible(problem, x)[0]]
    return deep, r


class TestPublicAPI:
    """CBiRRT.smooth_trajectory and trajectory=... on solve and plan: the path, the trajectory and a sampled
    command stay distinct (#214)."""

    START, GOAL = np.array([-2.0, 0.5]), np.array([2.0, 0.5])

    def _plan_problem(self, planner, validator=None):
        return PlanningProblem(
            space=planner.space,
            start=FiniteSet([self.START], metric=planner.space.distance),
            goal=FiniteSet([self.GOAL], metric=planner.space.distance),
            validator=validator or Wall(0, -0.2, 0.2, extent=1.0),
        )

    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_solve_attaches_a_trajectory_and_leaves_the_path_alone(self, backend):
        planner = _planner(backend)
        problem = self._plan_problem(planner)
        plain = planner.solve(problem, seed=3)
        timed = planner.solve(problem, seed=3, trajectory=_opts())
        assert plain.success and timed.success and plain.trajectory is None
        assert len(plain.path) == len(timed.path) and all(np.array_equal(a, b) for a, b in zip(plain.path, timed.path))
        traj = timed.trajectory
        assert isinstance(traj, Trajectory) and traj.backend == timed.backend
        assert np.allclose(traj.start[0], self.START, atol=1e-12) and np.allclose(traj.goal[0], self.GOAL, atol=1e-9)
        assert traj.seed == 3 and traj.resolution == 0.02 and traj.stopped in ("converged", "iterations")
        assert np.array_equal(traj.velocity_limits, VMAX) and np.array_equal(traj.acceleration_limits, AMAX)
        assert traj.duration < initial_trajectory(coalesce(timed.path), VMAX, AMAX).duration

    def test_a_failed_solve_has_no_trajectory(self):
        planner = _planner(max_iterations=1, connect_steps=1, extend_steps=1)
        result = planner.solve(self._plan_problem(planner), seed=0, trajectory=_opts())
        assert not result.success and result.trajectory is None

    def test_plan_needs_return_details(self):
        planner = _planner()
        with pytest.raises(ValueError, match="return_details"):
            planner.plan(start=self.START, goal=self.GOAL, trajectory=_opts())
        result = planner.plan(start=self.START, goal=self.GOAL, seed=1, return_details=True, trajectory=_opts())
        assert result.trajectory is not None

    def test_seed_comes_from_the_solve_unless_set(self):
        planner = _planner()
        problem = self._plan_problem(planner)
        a, b = (planner.solve(problem, seed=7, trajectory=_opts()).trajectory for _ in range(2))
        assert a.seed == b.seed == 7 and a.duration == b.duration
        assert planner.solve(problem, seed=7, trajectory=_opts(seed=11)).trajectory.seed == 11
        assert planner.solve(problem, trajectory=_opts()).trajectory.seed == 0

    @pytest.mark.parametrize("bad, match", [
        (np.array([1.0]), "one per joint"), (np.array([1.0, 0.0]), "joint 1"), (-1.0, "joint 0"),
        (np.array([np.inf, 1.0]), "joint 0"), (np.array([1.0, np.nan]), "joint 1"),
    ])  # fmt: skip
    def test_bad_limits_fail_before_planning(self, bad, match):
        planner = _planner()
        for options in (TrajectoryOptions(bad, AMAX), TrajectoryOptions(VMAX, bad)):
            with pytest.raises(ValueError, match=match):
                planner.solve(self._plan_problem(planner), seed=0, trajectory=options)

    def test_scalar_limits_apply_to_every_joint(self):
        planner = _planner()
        traj = planner.smooth_trajectory(_problem(planner, U_PATH), U_PATH, TrajectoryOptions(1.0, 2.0))
        assert np.array_equal(traj.velocity_limits, [1.0, 1.0]) and np.array_equal(traj.acceleration_limits, [2.0, 2.0])

    def test_options_must_be_trajectory_options(self):
        planner = _planner()
        with pytest.raises(TypeError):
            planner.solve(self._plan_problem(planner), trajectory={"velocity_limits": 1.0})

    def test_auto_falls_back_to_python_and_says_why(self):
        planner = _planner()
        problem = _problem(planner, U_PATH, validator=Block())
        traj = planner.smooth_trajectory(problem, U_PATH, _opts())
        assert traj.backend == "python" and any("validator" in r for r in traj.backend_reasons)
        deep, _ = _valid(traj, problem, planner)
        assert not [x for x in deep if 0.1 + 0.02 < x[0] < 0.9 - 0.02 and 0.1 + 0.02 < x[1] < 0.9 - 0.02]

    def test_solve_lowers_once_and_shares_its_reasons(self, monkeypatch):
        calls = []
        original = native.lower
        monkeypatch.setattr(native, "lower", lambda *a, **k: calls.append(1) or original(*a, **k))
        planner = _planner()
        result = planner.solve(self._plan_problem(planner, validator=Block()), seed=0, trajectory=_opts())
        assert result.success and result.backend == "python" and len(calls) == 1
        assert result.trajectory.backend == "python" and result.trajectory.backend_reasons == result.backend_reasons

    @needs_native
    def test_native_never_falls_back(self):
        planner = _planner("native")
        with pytest.raises(native.NativeUnsupported):
            planner.smooth_trajectory(_problem(planner, U_PATH, validator=Block()), U_PATH, _opts())

    def test_python_backend_stays_python(self, monkeypatch):
        monkeypatch.setattr(native, "lower", lambda *a, **k: pytest.fail("python backend lowered the problem"))
        traj = _planner("python").smooth_trajectory(_problem(_planner(), U_PATH), U_PATH, _opts())
        assert traj.backend == "python" and traj.backend_reasons == ()

    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_start_is_goal_and_two_point_paths(self, backend):
        planner = _planner(backend)
        here = np.array([0.5, 1.5])
        still = planner.smooth_trajectory(_problem(planner, [here, here]), [here, here], _opts())
        assert still.duration == 0.0 and np.array_equal(still.start[0], here) and np.array_equal(still.goal[0], here)
        assert still.sample(2_000_000)[0] == [0]
        line = [np.array([0.0, 0.0]), np.array([1.0, 0.5])]
        traj = planner.smooth_trajectory(_problem(planner, line), line, _opts())
        assert traj.duration == pytest.approx(initial_trajectory(line, VMAX, AMAX).duration)  # nothing to shortcut

    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_a_continuous_joint_crosses_pi(self, backend):
        config = CBiRRTConfig(step_size=0.1, edge_resolution=0.02, continuous_joints=[True, False])
        planner = CBiRRT(PlanarArm(), PlanarIK(), NoCollision(), config, backend=backend)
        path = _dense([2.8, 0.0], [3.0, 0.5], [3.4, 0.0])  # unwrapped across +pi
        problem = _problem(planner, path)
        traj = planner.smooth_trajectory(problem, path, _opts())
        deep, _ = _valid(traj, problem, planner)
        assert deep == [] and np.allclose(traj.goal[0], path[-1], atol=1e-9)

    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_a_zero_time_budget_returns_the_planned_polyline(self, backend):
        planner = _planner(backend)
        traj = planner.smooth_trajectory(_problem(planner, U_PATH), U_PATH, _opts(time_budget=0.0))
        assert traj.stopped == "time_budget" and {p.validated for p in traj.pieces} == {"planned"}
        assert traj.duration == pytest.approx(initial_trajectory(coalesce(U_PATH), VMAX, AMAX).duration)

    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_abort_stops_with_a_valid_trajectory(self, backend):
        calls = {"n": 0}

        def abort():
            calls["n"] += 1
            return calls["n"] > 5

        planner = _planner(backend, abort_fn=abort)
        problem = _problem(planner, U_PATH)
        traj = planner.smooth_trajectory(problem, U_PATH, _opts(convergence_window=None, iterations=10_000))
        assert traj.stopped == "aborted" and len(traj.diagnostics["attempts"]) < 10_000
        assert _valid(traj, problem, planner)[0] == []

    @pytest.mark.parametrize("dt", [0, -1, 2e6, True, None])
    def test_sample_needs_a_positive_int_period(self, dt):
        with pytest.raises(ValueError, match="dt_ns"):
            initial_trajectory(coalesce(U_PATH), VMAX, AMAX).sample(dt)

    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_runtime_command_passes_at_exact_limits(self, backend):
        tool = _tool()
        planner = _planner(backend)
        traj = planner.smooth_trajectory(_problem(planner, U_PATH), U_PATH, _opts())
        for dt in (1_000_000, 2_000_000, 7_000_000):
            times, q = traj.sample(dt)
            assert tool.runtime_violations(times, q, 1.0, 2.0) == []
