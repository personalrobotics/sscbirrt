# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Smooth geometric paths (#207): quintic corner blends validated through the planning problem's own boundary."""

import math

import numpy as np
import pytest

from sscbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem, PredicateSet, SmoothingOptions
from sscbirrt.smoothing import Blend, Line, SmoothSegment, blend_for, coalesce
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK


def _dense(*corners, step=0.05):
    """A planned-looking path: straight runs between corners with a waypoint every ``step``."""
    out = [np.asarray(corners[0], dtype=float)]
    for a, b in zip(corners, corners[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(1, math.ceil(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return out


def _planner(**config):
    cfg = CBiRRTConfig(step_size=0.1, edge_resolution=0.02, **config)
    return CBiRRT(PlanarArm(), PlanarIK(), NoCollision(), cfg, backend="python")


def _problem(planner, path, validator=None, constraint=None, motion_validator=None):
    return PlanningProblem(
        space=planner.space,
        start=FiniteSet([path[0]], metric=planner.space.distance),
        goal=FiniteSet([path[-1]], metric=planner.space.distance),
        validator=validator or NoCollision(),
        path_constraint=constraint,
        motion_validator=motion_validator,
    )


L_PATH = _dense([0.0, 0.0], [1.0, 0.0], [1.0, 1.0])  # one right-angle corner at (1, 0)
CORNER = np.array([1.0, 0.0])


class TestGeometry:
    def test_blend_is_tangent_with_zero_curvature_at_both_ends(self):
        u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.6, 0.8])
        b = Blend(np.array([1.0, 2.0, 3.0]), u, v, 0.3)
        ends = np.array([0.0, b.length])
        q, dq, ddq = (b.evaluate(ends, k) for k in (0, 1, 2))
        assert np.allclose(q[0], b.corner - 0.3 * u) and np.allclose(q[1], b.corner + 0.3 * v)
        assert np.allclose(dq[0], u) and np.allclose(dq[1], v)  # unit speed, along the lines
        assert np.abs(ddq).max() < 1e-12  # zero second derivative, matching the lines

    def test_derivatives_match_finite_differences(self):
        b = Blend(np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.4)
        s, h = np.linspace(0.02, b.length - 0.02, 9), 1e-6
        fd1 = (b.evaluate(s + h, 0) - b.evaluate(s - h, 0)) / (2 * h)
        fd2 = (b.evaluate(s + h, 1) - b.evaluate(s - h, 1)) / (2 * h)
        assert np.allclose(fd1, b.evaluate(s, 1), atol=1e-8) and np.allclose(fd2, b.evaluate(s, 2), atol=1e-6)

    @pytest.mark.parametrize("turn_deg", [10.0, 90.0, 150.0])
    def test_deviation_is_honoured_and_blends_never_overlap(self, turn_deg):
        t = math.radians(turn_deg)
        before, corner, after = np.array([-1.0, 0.0]), np.zeros(2), np.array([math.cos(t), math.sin(t)]) * 0.3
        b = blend_for(corner, before, after, max_deviation=0.05)
        assert b.deviation <= 0.05 + 1e-12
        assert b.d <= 0.5 * min(1.0, 0.3) + 1e-12  # at most halfway along each neighbouring chord
        assert b.deviation == pytest.approx(39 / 64 * b.d * math.sin(t / 2))

    def test_segment_is_c2_across_junctions(self):
        P0, P2 = np.array([0.0, 0.0]), np.array([1.0, 1.0])
        b = blend_for(CORNER, P0, P2, 0.1)
        seg = SmoothSegment((Line(P0, b.control[0]), b, Line(b.control[-1], P2)))
        for s in seg.breaks[1:-1]:
            for order, tol in ((0, 1e-8), (1, 1e-8), (2, 1e-4)):
                assert np.abs(seg(s - 1e-9, order) - seg(s + 1e-9, order)).max() < tol
        assert seg(np.linspace(0, seg.length, 5), 2).shape == (5, 2) and seg(0.3, 1).shape == (2,)

    def test_blend_samples_are_at_most_resolution_apart(self):
        b = Blend(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 0.7)
        pts = b.samples(0.05)
        assert max(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])) <= 0.05

    def test_coalesce_keeps_only_the_true_corners(self):
        pts = coalesce(L_PATH + [L_PATH[-1]])  # a repeated waypoint too
        assert len(pts) == 3 and np.array_equal(pts[0], L_PATH[0]) and np.array_equal(pts[-1], L_PATH[-1])
        assert np.allclose(pts[1], CORNER)


class SpyValidator:
    """The problem's motion validator: records every chord it is asked about."""

    def __init__(self, base):
        self.base, self.chords = base, []

    def validate(self, a, b):
        self.chords.append((np.array(a), np.array(b)))
        return self.base.validate(a, b)


class TestSmoothing:
    def test_free_corner_is_blended_through_the_problems_motion_validator(self):
        planner = _planner()
        base = planner.default_motion_validator(_problem(planner, L_PATH))
        spy = SpyValidator(base)
        sp = planner.smooth(_problem(planner, L_PATH, motion_validator=spy), L_PATH)
        assert len(sp.segments) == 1 and sp.stops == [] and sp.report.blended == 1
        assert np.array_equal(sp.start, L_PATH[0]) and np.array_equal(sp.goal, L_PATH[-1])
        assert sp.continuity.startswith("C2")
        blend = next(p for p in sp.segments[0].pieces if isinstance(p, Blend))
        assert len(spy.chords) == len(blend.samples(0.02)) - 1  # every chord went to the problem's validator
        assert sp.report.smoothed_length < sp.report.original_length

    def test_blend_shrinks_away_from_an_obstacle(self):
        planner = _planner()
        full = blend_for(CORNER, np.zeros(2), np.array([1.0, 1.0]), 0.1)
        mid = full.evaluate(np.array([full.length / 2]), 0)[0]

        class Disk:  # in the way of the full-size blend, clear of the path and of half-size blends
            def is_valid(self, q):
                return np.linalg.norm(np.asarray(q) - mid) > 0.4 * full.deviation

        sp = planner.smooth(_problem(planner, L_PATH, validator=Disk()), L_PATH)
        (corner,) = sp.report.corners
        assert corner.attempts[0]["rejected"] and "in collision" in corner.attempts[0]["rejected"]
        assert corner.outcome == "blended" and corner.attempts[-1]["d"] < corner.attempts[0]["d"]

    def test_unblendable_corner_is_kept_as_a_valid_stop(self):
        planner = _planner()

        class OnThePath:  # only the planned polyline itself is free
            def is_valid(self, q):
                x, y = q
                return (abs(y) < 1e-9 and -1e-9 <= x <= 1 + 1e-9) or (abs(x - 1) < 1e-9 and -1e-9 <= y <= 1 + 1e-9)

        sp = planner.smooth(_problem(planner, L_PATH, validator=OnThePath()), L_PATH)
        assert sp.report.stops == 1 and len(sp.report.corners[0].attempts) == SmoothingOptions().max_attempts
        assert len(sp.segments) == 2 and np.allclose(sp.stops[0], CORNER)  # the original corner, at rest
        assert all(OnThePath().is_valid(q) for q in sp.to_polyline(0.01))

    def test_inequality_path_constraint_is_never_violated(self):
        planner = _planner()

        def near_path(q):  # within 0.02 of the polyline: blends must shrink to fit
            x, y = q
            to_first = abs(y) if x <= 1 else math.hypot(x - 1, y)
            to_second = abs(x - 1) if y >= 0 else math.hypot(x - 1, y)
            return min(to_first, to_second) <= 0.02

        sp = planner.smooth(_problem(planner, L_PATH, constraint=PredicateSet(near_path)), L_PATH)
        (corner,) = sp.report.corners
        assert corner.outcome == "blended" and "violates path constraints" in corner.attempts[0]["rejected"]
        blend = next(p for p in sp.segments[0].pieces if isinstance(p, Blend))
        assert all(near_path(q) for q in blend.samples(0.02))

    def test_equality_path_constraint_gives_stops_never_violations(self):
        planner = _planner()

        def on_path(q):
            x, y = q
            return (abs(y) < 1e-9 and x <= 1 + 1e-9) or (abs(x - 1) < 1e-9 and y >= -1e-9)

        sp = planner.smooth(_problem(planner, L_PATH, constraint=PredicateSet(on_path)), L_PATH)
        assert sp.report.blended == 0 and all(on_path(q) for q in sp.to_polyline(0.01))

    def test_continuous_joint_corner_has_no_full_turn_excursion(self):
        planner = _planner(continuous_joints=(True, False))
        path = _dense([2.9, 0.0], [3.5, 0.0], [3.5, 0.6])  # unwrapped, past +pi on the continuous joint
        sp = planner.smooth(_problem(planner, path), path)
        pts = sp.to_polyline(0.02)
        assert sp.report.blended == 1
        assert max(np.abs(b - a).max() for a, b in zip(pts, pts[1:])) <= 0.03  # no 2*pi jump between samples

    def test_deterministic(self):
        planner = _planner()
        problem = _problem(planner, L_PATH)
        a, b = planner.smooth(problem, L_PATH), planner.smooth(problem, L_PATH)
        assert all(np.array_equal(x, y) for x, y in zip(a.to_polyline(0.01), b.to_polyline(0.01)))
        assert [c.attempts for c in a.report.corners] == [c.attempts for c in b.report.corners]

    def test_reversal_is_a_stop(self):
        planner = _planner()
        path = _dense([0.0, 0.0], [1.0, 0.0], [0.2, 0.02])
        sp = planner.smooth(_problem(planner, path), path)
        assert sp.report.stops == 1 and "reversal" in sp.report.corners[0].attempts[0]["rejected"]


class TestEntryPoints:
    def test_solve_and_plan_attach_a_smooth_path_only_when_asked(self):
        planner = _planner()
        problem = _problem(planner, L_PATH)
        assert planner.solve(problem, seed=0).smooth_path is None
        r = planner.solve(problem, seed=0, smooth=SmoothingOptions(max_deviation=0.05))
        assert r.success and r.smooth_path is not None
        assert np.array_equal(r.smooth_path.start, r.path[0]) and np.array_equal(r.smooth_path.goal, r.path[-1])
        with pytest.raises(ValueError, match="return_details=True"):
            planner.plan(start=L_PATH[0], goal=L_PATH[-1], smooth=True)
        assert planner.plan(start=L_PATH[0], goal=L_PATH[-1], smooth=True, return_details=True).smooth_path
        with pytest.raises(TypeError, match="smooth must be"):
            planner.solve(problem, seed=0, smooth="yes")

    def test_mujoco_plan_smooths_the_pick(self):
        pytest.importorskip("mujoco")
        pytest.importorskip("ssik")
        pytest.importorskip("sscbirrt_assets")
        from sscbirrt.demo.scenarios import pick
        from sscbirrt.mujoco import plan

        p = pick.problem()
        r = plan(p.model, p.data, p.arm, config=p.config, seed=1, smooth=True, **p.kwargs)
        sp = r.smooth_path
        assert r.success and sp.report.blended + sp.report.stops == len(sp.report.corners) >= 1
        assert len(sp.stops) == sp.report.stops and len(sp.segments) == sp.report.stops + 1
        seg = sp.segments[0]
        s = np.linspace(*seg.path_interval, 7)
        assert all(seg(s, k).shape == (7, 6) for k in (0, 1, 2))


def test_smoothing_artifact_records_what_review_needs(tmp_path):
    """#207's inspectable artifact: original and smoothed paths, stops, validation, lengths, constraint residuals."""
    pytest.importorskip("mujoco")
    pytest.importorskip("ssik")
    pytest.importorskip("sscbirrt_assets")
    import importlib.util
    import json
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "smoothing_artifact.py"
    spec = importlib.util.spec_from_file_location("smoothing_artifact", path)
    tool = importlib.util.module_from_spec(spec)
    sys.modules["smoothing_artifact"] = tool
    spec.loader.exec_module(tool)
    out = tmp_path / "artifact.json"
    assert tool.main(["--seeds", "1", "--output", str(out)]) == 0
    records = json.loads(out.read_text())["records"]
    assert [r["problem"] for r in records] == ["pick", "transport"]
    for r in records:
        assert r["smoothed"]["endpoints_exact"] and r["smoothed"]["recheck_collision_free"]
        assert len(r["stops"]) == sum(c["outcome"] == "stop" for c in r["corners"]) == r["segments"] - 1
        assert r["smoothed"]["length"] <= r["planned"]["length"] + 1e-9
    c = records[1]["constraint"]
    assert c["max_tilt_deg_smoothed"] <= c["tilt_limit_deg"] + 0.1
    assert tool.main(["--seeds", "1", "--output", str(out), "--check"]) == 0
