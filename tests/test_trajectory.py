# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""#214 prototype: bounded-acceleration trajectories with time-optimized shortcuts (sscbirrt._trajectory)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from sscbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem, PredicateSet
from sscbirrt._trajectory import (
    ShortcutOptions,
    fixed_time,
    initial_trajectory,
    min_time,
    piece_from_native,
    ramp_from_native,
    shortcut,
    smooth_native,
    synchronize,
    verify,
)
from sscbirrt.backends import native
from sscbirrt.smoothing import coalesce
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK, Wall

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


def _planner(**config):
    return CBiRRT(PlanarArm(), PlanarIK(), NoCollision(), CBiRRTConfig(step_size=0.1, edge_resolution=0.02, **config))


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


def _smooth(planner, problem, traj, iterations=200, seed=0):
    return shortcut(
        traj,
        motion_validator=planner._motion_validator(problem),
        admissible=lambda q: planner._admissible(problem, q),
        resolution=planner.config.edge_resolution,
        options=ShortcutOptions(iterations=iterations, seed=seed),
    )


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


def _native_smooth(planner, problem, path=U_PATH, iterations=200, seed=0):
    return smooth_native(planner, problem, path, VMAX, AMAX, ShortcutOptions(iterations=iterations, seed=seed))


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

    def test_python_objects_have_no_native_form(self):
        planner = _planner()

        class Block:
            def is_valid(self, q):
                return True

        with pytest.raises(native.NativeUnsupported):
            _native_smooth(planner, _problem(planner, U_PATH, validator=Block()))
