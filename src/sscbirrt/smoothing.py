# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Smooth geometric paths for execution: constraint-preserving corner blends (#207).

A planned path is a valid polyline. At each corner its tangent jumps, so a retimer under finite acceleration must
stop there. This module replaces each corner, where it can, with a quintic blend whose curvature is continuous, and
keeps every corner it cannot blend as an explicit stop. It changes geometry, so it validates every blend through the
planning problem's own admissibility boundary; it assigns no timing.

Three operations, kept distinct:

- **Shortening** (the planner's shortcut smoother) replaces runs of the path with shorter validated edges. The path
  stays a polyline.
- **Smoothing** (this module) replaces corners with blends. The result is a ``SmoothPath``: segments that are C2 in
  their parameter, separated by stops.
- **Time parameterization** (downstream, for example TOPP-RA) assigns a timing ``s(t)`` to each segment under robot
  limits without moving it, coming to rest at every stop.

What is guaranteed. A blend is accepted only if samples along it, spaced at most ``resolution`` apart in joint space,
are all admissible (joint space, state validator, path constraint), and the problem's motion validator accepts every
chord between consecutive samples. That is the planner's own discrete guarantee for an edge, no stronger: an obstacle
thinner than ``resolution``, or a region the curve enters between samples, can be missed. Line pieces are the original
path's geometry. Equality path constraints (zero-width regions) generally reject blends, because no projection is
applied; such corners become stops.

The blend. For a corner ``C`` with incoming unit direction ``u``, outgoing ``v`` and half-size ``d``, the quintic
Bezier with control points ``C - d u, C - 3d/4 u, C - d/2 u, C + d/2 v, C + 3d/4 v, C + d v`` leaves and rejoins the
lines tangentially with zero second derivative. Reparameterized to unit speed at its ends, it joins arc-length lines
with continuous position, first and second derivative: each segment is C2 in its parameter ``s``, which is exactly arc
length on line pieces and close to it in blends. Its midpoint lies ``(39/64) d sin(theta/2)`` from the corner, where
``theta`` is the turning angle.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Midpoint distance from the corner per unit half-size and per unit sin(theta/2) (see the module docstring).
_MIDPOINT_GAIN = 39.0 / 64.0


@dataclass(frozen=True)
class SmoothingOptions:
    """How corners are blended.

    Attributes:
        max_deviation: Largest joint-space distance (rad) a blend's midpoint may lie from the corner it replaces.
            The blend starts at this size, limited so that neighbouring blends never overlap.
        shrink: Factor applied to a blend's size after it fails validation.
        max_attempts: Sizes tried per corner before it is kept as a stop.
        collinear_tolerance: Consecutive chords whose directions differ by less than this angle (rad) are one line.
        max_turn: Corners that turn by more than this angle (rad), near a reversal, are kept as stops.
    """

    max_deviation: float = 0.1
    shrink: float = 0.5
    max_attempts: int = 6
    collinear_tolerance: float = 1e-9
    max_turn: float = math.radians(170.0)

    def __post_init__(self):
        if not self.max_deviation > 0:
            raise ValueError(f"max_deviation must be positive, got {self.max_deviation!r}")
        if not 0 < self.shrink < 1:
            raise ValueError(f"shrink must be in (0, 1), got {self.shrink!r}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts!r}")
        if not self.collinear_tolerance >= 0:
            raise ValueError(f"collinear_tolerance must be nonnegative, got {self.collinear_tolerance!r}")
        if not 0 < self.max_turn < math.pi:
            raise ValueError(f"max_turn must be in (0, pi), got {self.max_turn!r}")


def coalesce(path: Sequence[np.ndarray], tolerance: float = 1e-9) -> list[np.ndarray]:
    """The path's true corners: its endpoints and every waypoint where the direction turns by at least ``tolerance``.

    Planned paths are dense (a waypoint every edge resolution along straight runs); this merges those runs.
    Repeated waypoints are dropped. Endpoints are returned exactly.
    """
    pts = [np.asarray(path[0], dtype=float)]
    for q in path[1:]:
        q = np.asarray(q, dtype=float)
        if np.linalg.norm(q - pts[-1]) > 0:
            pts.append(q)
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a = pts[i] - out[-1]
        b = pts[i + 1] - pts[i]
        if _angle(a, b) >= tolerance:
            out.append(pts[i])
    out.append(pts[-1])
    return out


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two nonzero vectors, accurate near zero."""
    ua, ub = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(2.0 * math.atan2(np.linalg.norm(ua - ub), np.linalg.norm(ua + ub)))


# --------------------------------------------------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    """A straight piece from ``a`` to ``b``, parameterized by arc length ``s`` in ``[0, length]``."""

    a: np.ndarray
    b: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    def evaluate(self, s: np.ndarray, order: int) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        L = self.length
        if L == 0.0:  # a single-point path (start equals goal)
            return np.tile(self.a if order == 0 else np.zeros_like(self.a), (len(s), 1))
        if order == 0:
            return self.a + np.outer(s / L, self.b - self.a)
        if order == 1:
            return np.tile((self.b - self.a) / L, (len(s), 1))
        return np.zeros((len(s), len(self.a)))


@dataclass(frozen=True)
class Blend:
    """A quintic corner blend with half-size ``d``, reparameterized to unit speed at its ends (``s`` in
    ``[0, 5d/4]``). ``corner``, ``u`` and ``v`` are the corner and the unit directions in and out."""

    corner: np.ndarray
    u: np.ndarray
    v: np.ndarray
    d: float

    @property
    def control(self) -> np.ndarray:
        c, u, v, d = self.corner, self.u, self.v, self.d
        return np.array([c - d * u, c - 0.75 * d * u, c - 0.5 * d * u, c + 0.5 * d * v, c + 0.75 * d * v, c + d * v])

    @property
    def length(self) -> float:
        """The parameter range, 5d/4 (not the arc length, which is slightly less)."""
        return 1.25 * self.d

    @property
    def deviation(self) -> float:
        """Distance of the blend's midpoint from the corner."""
        return float(np.linalg.norm(self.evaluate(np.array([self.length / 2]), 0)[0] - self.corner))

    def evaluate(self, s: np.ndarray, order: int) -> np.ndarray:
        t = np.asarray(s, dtype=float) / self.length
        P = self.control
        if order == 0:
            return _bernstein(5, t) @ P
        if order == 1:
            return (5.0 * (_bernstein(4, t) @ np.diff(P, axis=0))) / self.length
        return (20.0 * (_bernstein(3, t) @ np.diff(P, n=2, axis=0))) / self.length**2

    def samples(self, resolution: float) -> list[np.ndarray]:
        """Points along the blend, ends included, no two consecutive ones farther apart than ``resolution``.

        The curve's derivative is a convex combination of ``5 (P[i+1] - P[i])``, so ``n`` uniform parameter steps
        with ``n >= 5 max |P[i+1] - P[i]| / resolution`` bound every chord by ``resolution``.
        """
        P = self.control
        bound = 5.0 * float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        n = max(1, math.ceil(bound / resolution))
        return list(self.evaluate(np.linspace(0.0, self.length, n + 1), 0))


def _bernstein(n: int, t: np.ndarray) -> np.ndarray:
    k = np.arange(n + 1)
    coeff = np.array([math.comb(n, i) for i in k], dtype=float)
    t = t[:, None]
    return coeff * t**k * (1.0 - t) ** (n - k)


def blend_for(corner: np.ndarray, before: np.ndarray, after: np.ndarray, max_deviation: float) -> Blend:
    """The largest blend at ``corner`` whose midpoint is within ``max_deviation`` and that reaches at most halfway
    along each neighbouring chord (so neighbouring blends never overlap)."""
    a, b = corner - before, after - corner
    u, v = a / np.linalg.norm(a), b / np.linalg.norm(b)
    half_turn = math.sin(_angle(a, b) / 2.0)
    d = 0.5 * min(np.linalg.norm(a), np.linalg.norm(b))
    if half_turn > 0:
        d = min(d, max_deviation / (_MIDPOINT_GAIN * half_turn))
    return Blend(np.asarray(corner, dtype=float), u, v, float(d))


# --------------------------------------------------------------------------------------------------------------------
# Segments and the smooth path
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothSegment:
    """A run of line and blend pieces with continuous position, first and second derivative in its parameter ``s``.

    The call shape follows TOPP-RA's geometric paths: ``segment(s, order)`` with ``order`` 0, 1 or 2 returns an
    ``(len(s), dof)`` array (a ``(dof,)`` array for a scalar ``s``). A retimer starts and ends each segment at rest.
    """

    pieces: tuple[Line | Blend, ...]

    @property
    def breaks(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum([p.length for p in self.pieces])])

    @property
    def length(self) -> float:
        return float(self.breaks[-1])

    @property
    def path_interval(self) -> np.ndarray:
        return np.array([0.0, self.length])

    @property
    def dof(self) -> int:
        return len(self(0.0))

    @property
    def start(self) -> np.ndarray:
        p = self.pieces[0]
        return p.a if isinstance(p, Line) else p.control[0]

    @property
    def end(self) -> np.ndarray:
        p = self.pieces[-1]
        return p.b if isinstance(p, Line) else p.control[-1]

    def __call__(self, s: float | np.ndarray, order: int = 0) -> np.ndarray:
        if order not in (0, 1, 2):
            raise ValueError(f"order must be 0, 1 or 2, got {order!r}")
        scalar = np.ndim(s) == 0
        s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, self.length)
        breaks = self.breaks
        idx = np.clip(np.searchsorted(breaks, s, side="right") - 1, 0, len(self.pieces) - 1)
        out = np.empty((len(s), len(self.start)))
        for i in np.unique(idx):
            mask = idx == i
            out[mask] = self.pieces[i].evaluate(s[mask] - breaks[i], order)
        return out[0] if scalar else out

    def sample(self, spacing: float) -> list[np.ndarray]:
        """Points along the segment, ends included, at parameter spacing of at most ``spacing``."""
        n = max(1, math.ceil(self.length / spacing))
        return list(self(np.linspace(0.0, self.length, n + 1)))


@dataclass
class CornerReport:
    """What happened at one corner: the blends tried, largest first, and the outcome."""

    index: int
    config: np.ndarray
    turn: float
    outcome: str = "stop"  # "blended" or "stop"
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SmoothingReport:
    corners: list[CornerReport]
    original_length: float
    smoothed_length: float

    @property
    def blended(self) -> int:
        return sum(c.outcome == "blended" for c in self.corners)

    @property
    def stops(self) -> int:
        return sum(c.outcome == "stop" for c in self.corners)


@dataclass
class SmoothPath:
    """A path for execution: segments, C2 in their parameter, separated by stops where a retimer comes to rest.

    ``segments[0]`` starts at the planned start and ``segments[-1]`` ends at the planned goal, exactly. ``stops`` are
    the configurations between consecutive segments: corners that could not be blended.
    """

    segments: list[SmoothSegment]
    report: SmoothingReport
    continuity: str = "C2 within a segment; at rest at every stop"

    @property
    def start(self) -> np.ndarray:
        return self.segments[0].start

    @property
    def goal(self) -> np.ndarray:
        return self.segments[-1].end

    @property
    def stops(self) -> list[np.ndarray]:
        return [seg.end for seg in self.segments[:-1]]

    def to_polyline(self, spacing: float) -> list[np.ndarray]:
        """The path as dense waypoints, segment by segment, for executors that interpolate positions linearly."""
        out: list[np.ndarray] = []
        for seg in self.segments:
            pts = seg.sample(spacing)
            out.extend(pts if not out else pts[1:])
        return out


# --------------------------------------------------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------------------------------------------------

Admissible = Callable[[np.ndarray], "tuple[bool, str | None]"]


def validate_blend(blend: Blend, resolution: float, motion_validator: Any, admissible: Admissible) -> str | None:
    """Why ``blend`` is not admissible under the planning problem's boundary, or ``None`` if it is.

    Every sample (at most ``resolution`` apart) must be admissible, and the motion validator must accept every chord
    between consecutive samples; each configuration it returns is checked for admissibility too, as the planner
    checks the configurations of every edge.
    """
    pts = blend.samples(resolution)
    ok, why = admissible(pts[0])
    if not ok:
        return f"sample 0: {why}"
    for i in range(len(pts) - 1):
        motion = motion_validator.validate(pts[i], pts[i + 1])
        if not motion.reached:
            # Name the cause when the chord's end is itself inadmissible (the default validator checks that)
            ok, why = admissible(pts[i + 1])
            return f"chord {i}: {why}" if not ok else f"chord {i}: motion validator rejected it"
        for q in motion.configs:
            ok, why = admissible(q)
            if not ok:
                return f"chord {i}: {why}"
    return None


def smooth_path(
    path: Sequence[np.ndarray],
    *,
    motion_validator: Any,
    admissible: Admissible,
    resolution: float,
    options: SmoothingOptions | None = None,
) -> SmoothPath:
    """Blend each corner of a valid planned ``path`` that can be blended admissibly; keep the rest as stops.

    ``admissible(q) -> (ok, reason)`` and ``motion_validator`` are the planning problem's own boundary (see
    ``CBiRRT.smooth``); ``resolution`` is its edge resolution. Deterministic: corners are independent, sizes shrink by
    a fixed factor, and nothing is random.
    """
    options = options or SmoothingOptions()
    pts = coalesce(path, options.collinear_tolerance)
    original = float(sum(np.linalg.norm(b - a) for a, b in zip(path[1:], path[:-1])))
    corners: list[CornerReport] = []
    blends: dict[int, Blend] = {}
    for i in range(1, len(pts) - 1):
        turn = _angle(pts[i] - pts[i - 1], pts[i + 1] - pts[i])
        report = CornerReport(index=i, config=pts[i], turn=turn)
        corners.append(report)
        if turn > options.max_turn:
            report.attempts.append({"d": 0.0, "deviation": 0.0, "rejected": "turns too sharply (near a reversal)"})
            continue
        blend = blend_for(pts[i], pts[i - 1], pts[i + 1], options.max_deviation)
        for _ in range(options.max_attempts):
            why = validate_blend(blend, resolution, motion_validator, admissible)
            report.attempts.append({"d": blend.d, "deviation": blend.deviation, "rejected": why})
            if why is None:
                report.outcome = "blended"
                blends[i] = blend
                break
            blend = Blend(blend.corner, blend.u, blend.v, blend.d * options.shrink)

    segments: list[SmoothSegment] = []
    pieces: list[Line | Blend] = []
    here = pts[0]
    for i in range(1, len(pts) - 1):
        if i in blends:
            b = blends[i]
            entry = b.control[0]
            if np.linalg.norm(entry - here) > 0:
                pieces.append(Line(here, entry))
            pieces.append(b)
            here = b.control[-1]
        else:  # a stop: end this segment at the corner, start the next one there
            pieces.append(Line(here, pts[i]))
            segments.append(SmoothSegment(tuple(pieces)))
            pieces, here = [], pts[i]
    if np.linalg.norm(pts[-1] - here) > 0 or not pieces:
        pieces.append(Line(here, pts[-1]))
    segments.append(SmoothSegment(tuple(pieces)))
    smoothed = float(sum(_arc_length(seg) for seg in segments))
    return SmoothPath(segments, SmoothingReport(corners, original, smoothed))


def _arc_length(seg: SmoothSegment, n: int = 64) -> float:
    total = 0.0
    for p in seg.pieces:
        if isinstance(p, Line):
            total += p.length
        else:
            pts = p.evaluate(np.linspace(0.0, p.length, n + 1), 0)
            total += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return total
