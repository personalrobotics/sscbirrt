# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Prototype (#214, not public API): bounded-acceleration trajectory smoothing with time-optimized shortcuts.

After Hauser and Ng-Thow-Hing, "Fast smoothing of manipulator trajectories using optimal bounded-acceleration
shortcuts" (ICRA 2010), the design of OpenRAVE's ParabolicSmoother.

- A **ramp** is one joint's motion over an interval: up to three constant-acceleration pieces.
- A **piece** is one interval in which every joint follows its own ramp, all of the same duration.
- A **trajectory** is a sequence of pieces. Position and velocity are continuous; acceleration is piecewise constant.

Each shortcut is time-optimal between its two boundary states under the per-joint limits. Randomized shortcutting
does not make the whole trajectory globally time-optimal. Every ramp is verified independently of the formulas
that built it (endpoints, acceleration and velocity bounds), so a mistake in the algebra can only reject a
shortcut, never admit an invalid one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_EPS_T = 1e-12  # times this close to zero are zero
_TOL_X = 1e-9  # endpoint tolerance (joint units)
_TOL_REL = 1e-9  # relative slack on velocity and acceleration bounds in verification


# --------------------------------------------------------------------------------------------------------------------
# One joint
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ramp:
    """One joint's motion: start state and constant-acceleration pieces ``(duration, acceleration)``."""

    x0: float
    v0: float
    pieces: tuple[tuple[float, float], ...]

    @property
    def duration(self) -> float:
        return float(sum(d for d, _ in self.pieces))

    def state(self, t: float) -> tuple[float, float, float]:
        """Position, velocity and acceleration at ``t`` (clamped to the ramp)."""
        x, v = self.x0, self.v0
        a = self.pieces[0][1] if self.pieces else 0.0
        remaining = min(max(t, 0.0), self.duration)
        for d, a in self.pieces:
            step = min(d, remaining)
            x, v = x + v * step + 0.5 * a * step * step, v + a * step
            remaining -= step
            if remaining <= 0.0:
                break
        return x, v, a

    def evaluate(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized position, velocity and acceleration at times ``ts`` (each clamped to the ramp)."""
        ts = np.clip(np.asarray(ts, dtype=float), 0.0, self.duration)
        x = np.full_like(ts, self.x0)
        v = np.full_like(ts, self.v0)
        a = np.zeros_like(ts)
        start, xs, vs = 0.0, self.x0, self.v0
        done = np.zeros(ts.shape, dtype=bool)
        for i, (d, acc) in enumerate(self.pieces):
            last = i == len(self.pieces) - 1
            mask = ~done & ((ts <= start + d) | last)
            tau = ts[mask] - start
            x[mask] = xs + vs * tau + 0.5 * acc * tau * tau
            v[mask] = vs + acc * tau
            a[mask] = acc
            done |= mask
            xs, vs, start = xs + vs * d + 0.5 * acc * d * d, vs + acc * d, start + d
        return x, v, a

    def end(self) -> tuple[float, float]:
        x, v, _ = self.state(self.duration)
        return x, v

    def split(self, t: float) -> tuple[Ramp, Ramp]:
        """The ramp before and after ``t``."""
        before, after, elapsed = [], [], 0.0
        for d, a in self.pieces:
            if elapsed + d <= t:
                before.append((d, a))
            elif elapsed >= t:
                after.append((d, a))
            else:
                before.append((t - elapsed, a))
                after.append((elapsed + d - t, a))
            elapsed += d
        x, v, _ = self.state(t)
        return Ramp(self.x0, self.v0, tuple(before)), Ramp(x, v, tuple(after))


def verify(ramp: Ramp, x1: float, v1: float, vmax: float, amax: float, duration: float | None = None) -> bool:
    """Independent check: the ramp ends at ``(x1, v1)``, stays within ``|v| <= vmax`` and ``|a| <= amax``, and lasts
    ``duration`` if given. Velocity is extreme at piece boundaries, since acceleration is constant on each piece."""
    if any(d < -_EPS_T for d, _ in ramp.pieces):
        return False
    if any(abs(a) > amax * (1 + _TOL_REL) for _, a in ramp.pieces):
        return False
    v = ramp.v0
    if abs(v) > vmax * (1 + _TOL_REL):
        return False
    for d, a in ramp.pieces:
        v += a * d
        if abs(v) > vmax * (1 + _TOL_REL):
            return False
    xe, ve = ramp.end()
    scale = max(1.0, abs(x1))
    if abs(xe - x1) > _TOL_X * scale or abs(ve - v1) > 1e-7 * max(1.0, vmax):
        return False
    return duration is None or abs(ramp.duration - duration) <= 1e-9 * max(1.0, duration)


def _clean(pieces: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return tuple((max(d, 0.0), a) for d, a in pieces if d > _EPS_T)


def min_time(x0: float, v0: float, x1: float, v1: float, vmax: float, amax: float) -> Ramp | None:
    """The time-optimal ramp from ``(x0, v0)`` to ``(x1, v1)`` under ``|v| <= vmax`` and ``|a| <= amax``.

    Candidates: accelerate one way then the other (P+P-, P-P+). If the switch velocity would exceed ``vmax``, cruise
    at ``±vmax`` between them (P+L P-). The switch velocity satisfies ``v_m**2 = a dx + (v0**2 + v1**2) / 2``.
    """
    dx = x1 - x0
    if abs(dx) <= _TOL_X and abs(v1 - v0) <= 1e-12:
        return Ramp(x0, v0, ())
    best: Ramp | None = None
    for s in (1.0, -1.0):
        a = s * amax
        rad = a * dx + 0.5 * (v0 * v0 + v1 * v1)
        if rad < 0:
            continue
        vm = s * math.sqrt(rad)
        if abs(vm) <= vmax:
            t1, t2 = (vm - v0) / a, (vm - v1) / a
            if t1 < -_EPS_T or t2 < -_EPS_T:
                continue
            ramp = Ramp(x0, v0, _clean([(t1, a), (t2, -a)]))
        else:
            vc = s * vmax
            t1, t3 = (vc - v0) / a, (vc - v1) / a
            if t1 < -_EPS_T or t3 < -_EPS_T:
                continue
            cruise = dx - ((vc * vc - v0 * v0) + (vc * vc - v1 * v1)) / (2 * a)
            t2 = cruise / vc
            if t2 < -_EPS_T:
                continue
            ramp = Ramp(x0, v0, _clean([(t1, a), (t2, 0.0), (t3, -a)]))
        if verify(ramp, x1, v1, vmax, amax) and (best is None or ramp.duration < best.duration):
            best = ramp
    return best


def fixed_time(x0: float, v0: float, x1: float, v1: float, T: float, vmax: float, amax: float) -> Ramp | None:
    """A ramp from ``(x0, v0)`` to ``(x1, v1)`` lasting exactly ``T`` within the bounds, or ``None``.

    Least acceleration first: a P P shape whose acceleration ``a`` solves ``a**2 T**2 - 4 a D - dv**2 = 0`` with
    ``D = dx - (v0 + v1) T / 2``. If its switch velocity exceeds ``vmax``, cruise at ``±vmax``, which fixes the
    acceleration magnitude. Feasible durations can have gaps when the boundary velocities are nonzero, so ``None``
    means "not at this T".
    """
    if T <= _EPS_T:
        return Ramp(x0, v0, ()) if abs(x1 - x0) <= _TOL_X and abs(v1 - v0) <= 1e-12 else None
    dx, dv = x1 - x0, v1 - v0
    D = dx - 0.5 * (v0 + v1) * T
    candidates: list[Ramp] = []
    if abs(D) <= 1e-14 * max(1.0, abs(dx)) and abs(dv) <= 1e-14:
        candidates.append(Ramp(x0, v0, ((T, 0.0),)))
    else:
        disc = 4 * D * D + T * T * dv * dv
        for a in ((2 * D + math.sqrt(disc)) / (T * T), (2 * D - math.sqrt(disc)) / (T * T)):
            if a == 0.0:
                continue
            t1 = (a * T + dv) / (2 * a)
            if t1 < -_EPS_T or t1 > T + _EPS_T:
                continue
            t1 = min(max(t1, 0.0), T)
            candidates.append(Ramp(x0, v0, _clean([(t1, a), (T - t1, -a)])))
    best = None
    for ramp in sorted(candidates, key=lambda r: max((abs(a) for _, a in r.pieces), default=0.0)):
        if verify(ramp, x1, v1, vmax, amax, T):
            best = ramp
            break
    if best is not None:
        return best
    # Cruise at +-vmax: dx = vc T - s ((vc - v0)**2 + (vc - v1)**2) / (2 A)
    for s in (1.0, -1.0):
        vc = s * vmax
        denom = 2.0 * (vc * T - dx)
        if s * denom <= 0:
            continue
        A = s * ((vc - v0) ** 2 + (vc - v1) ** 2) / denom
        if not 0 < A <= amax * (1 + _TOL_REL):
            continue
        a1 = math.copysign(A, vc - v0) if vc != v0 else A
        a3 = math.copysign(A, v1 - vc) if vc != v1 else A
        t1, t3 = abs(vc - v0) / A, abs(vc - v1) / A
        t2 = T - t1 - t3
        if t2 < -_EPS_T:
            continue
        ramp = Ramp(x0, v0, _clean([(t1, a1), (t2, 0.0), (t3, a3)]))
        if verify(ramp, x1, v1, vmax, amax, T):
            return ramp
    # Full acceleration with a cruise at an intermediate speed vc (the remaining shape): with directions
    # s1 = sign(vc - v0) and s3 = sign(v1 - vc), dx = vc T - s1 (vc - v0)**2 / (2 A) + s3 (v1 - vc)**2 / (2 A).
    A = amax
    for s1 in (1.0, -1.0):
        for s3 in (1.0, -1.0):
            qa = (s3 - s1) / (2 * A)
            qb = T + (s1 * v0 - s3 * v1) / A
            qc = (s3 * v1 * v1 - s1 * v0 * v0) / (2 * A) - dx
            if abs(qa) < 1e-15:
                roots = [-qc / qb] if qb != 0 else []
            else:
                disc = qb * qb - 4 * qa * qc
                if disc < 0:
                    continue
                roots = [(-qb + math.sqrt(disc)) / (2 * qa), (-qb - math.sqrt(disc)) / (2 * qa)]
            for vc in roots:
                if abs(vc) > vmax * (1 + _TOL_REL):
                    continue
                if (vc - v0) * s1 < -1e-12 or (v1 - vc) * s3 < -1e-12:
                    continue
                t1, t3 = abs(vc - v0) / A, abs(v1 - vc) / A
                t2 = T - t1 - t3
                if t2 < -_EPS_T:
                    continue
                ramp = Ramp(x0, v0, _clean([(t1, s1 * A), (t2, 0.0), (t3, s3 * A)]))
                if verify(ramp, x1, v1, vmax, amax, T):
                    return ramp
    return None


# --------------------------------------------------------------------------------------------------------------------
# Many joints
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """One interval in which every joint follows its own ramp, all of duration ``duration``."""

    ramps: tuple[Ramp, ...]

    @property
    def duration(self) -> float:
        return max((r.duration for r in self.ramps), default=0.0)

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = [r.state(t) for r in self.ramps]
        return tuple(np.array([x[k] for x in s]) for k in range(3))  # type: ignore[return-value]

    def split(self, t: float) -> tuple[Piece, Piece]:
        pairs = [r.split(t) for r in self.ramps]
        return Piece(tuple(p[0] for p in pairs)), Piece(tuple(p[1] for p in pairs))


def synchronize(
    q0: np.ndarray, v0: np.ndarray, q1: np.ndarray, v1: np.ndarray, vmax: np.ndarray, amax: np.ndarray
) -> Piece | None:
    """The fastest common duration at which every joint can move between its boundary states within its limits.

    Start from the slowest joint's minimum time. Where some joint cannot match a duration (its feasible durations
    can have gaps), step the duration up by 2%, to at most 4x. ``None`` when no common duration is found.
    """
    mins = [min_time(*args) for args in zip(q0, v0, q1, v1, vmax, amax)]
    if any(m is None for m in mins):
        return None
    T0 = max(m.duration for m in mins)  # type: ignore[union-attr]
    if T0 <= _EPS_T:
        return Piece(tuple(Ramp(float(x), float(v), ()) for x, v in zip(q0, v0)))
    T = T0
    while T <= 4.0 * T0:
        ramps = []
        for j in range(len(q0)):
            m = mins[j]
            if m is not None and abs(m.duration - T) <= 1e-12 * max(1.0, T):
                ramps.append(m)
                continue
            r = fixed_time(q0[j], v0[j], q1[j], v1[j], T, vmax[j], amax[j])
            if r is None:
                break
            ramps.append(r)
        else:
            return Piece(tuple(ramps))
        T *= 1.02
    return None


# --------------------------------------------------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A timed trajectory: pieces in sequence, exact ``q(t)``, ``qdot(t)``, ``qddot(t)``.

    Position and velocity are continuous. Acceleration is piecewise constant, and its discontinuities are at piece
    and ramp boundaries. ``sample`` discretizes the trajectory for a runtime: an approximation of this source law,
    not the law itself.
    """

    pieces: list[Piece]
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    law: str = "piecewise-parabolic"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def breaks(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum([p.duration for p in self.pieces])])

    @property
    def duration(self) -> float:
        return float(self.breaks[-1])

    def _locate(self, t: float) -> tuple[int, float]:
        br = self.breaks
        i = int(np.clip(np.searchsorted(br, t, side="right") - 1, 0, len(self.pieces) - 1))
        return i, t - br[i]

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i, tau = self._locate(min(max(t, 0.0), self.duration))
        return self.pieces[i].state(tau)

    def evaluate(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized ``q``, ``qdot``, ``qddot`` at times ``ts``: arrays of shape ``(len(ts), dof)``."""
        ts = np.clip(np.asarray(ts, dtype=float), 0.0, self.duration)
        br = self.breaks
        idx = np.clip(np.searchsorted(br, ts, side="right") - 1, 0, len(self.pieces) - 1)
        dof = len(self.pieces[0].ramps)
        q, v, a = (np.empty((len(ts), dof)) for _ in range(3))
        for i in np.unique(idx):
            mask = idx == i
            for j, ramp in enumerate(self.pieces[i].ramps):
                q[mask, j], v[mask, j], a[mask, j] = ramp.evaluate(ts[mask] - br[i])
        return q, v, a

    def splice(self, t1: float, t2: float, piece: Piece) -> Trajectory:
        """This trajectory with ``[t1, t2]`` replaced by ``piece``."""
        i1, tau1 = self._locate(t1)
        i2, tau2 = self._locate(t2)
        head = self.pieces[:i1] + [self.pieces[i1].split(tau1)[0]]
        tail = [self.pieces[i2].split(tau2)[1]] + self.pieces[i2 + 1 :]
        pieces = [p for p in head + [piece] + tail if p.duration > _EPS_T]
        return Trajectory(pieces, self.velocity_limits, self.acceleration_limits, self.law, self.diagnostics)

    def sample(self, dt_ns: int) -> tuple[list[int], np.ndarray]:
        """A runtime command: integer-nanosecond times from 0 every ``dt_ns``, ending exactly at the duration
        (rounded up to a whole nanosecond), and the positions there."""
        end_ns = math.ceil(self.duration * 1e9)
        times = list(range(0, end_ns, dt_ns)) + [end_ns]
        if len(times) >= 2 and times[-1] == times[-2]:
            times.pop()
        q, _, _ = self.evaluate(np.array(times, dtype=float) * 1e-9)
        return times, q


def straight_line(q0: np.ndarray, q1: np.ndarray, vmax: np.ndarray, amax: np.ndarray) -> Piece:
    """Rest-to-rest motion along the chord ``q0 -> q1``: one scalar trapezoid shared by every joint, so the motion
    is exactly the chord. Speed and acceleration along it are limited by the most constrained joint."""
    d = np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)
    L = float(np.linalg.norm(d))
    if L == 0.0:
        return Piece(tuple(Ramp(float(x), 0.0, ()) for x in q0))
    u = np.abs(d) / L
    with np.errstate(divide="ignore"):
        vs = float(np.min(np.where(u > 0, vmax / u, np.inf)))
        As = float(np.min(np.where(u > 0, amax / u, np.inf)))
    s = min_time(0.0, 0.0, L, 0.0, vs, As)
    if s is None:  # unreachable for positive limits; kept as an explicit failure
        raise ValueError("no trapezoid along the chord")
    scaled = [tuple((dur, acc * d[j] / L) for dur, acc in s.pieces) for j in range(len(d))]
    return Piece(tuple(Ramp(float(q0[j]), 0.0, scaled[j]) for j in range(len(d))))


def initial_trajectory(corners: Sequence[np.ndarray], vmax: np.ndarray, amax: np.ndarray) -> Trajectory:
    """The planned polyline (its true corners), at rest at every corner: the safe fallback, exactly the planned
    geometry."""
    pieces = [straight_line(a, b, vmax, amax) for a, b in zip(corners, corners[1:])]
    return Trajectory([p for p in pieces if p.duration > _EPS_T] or pieces[:1], np.asarray(vmax), np.asarray(amax))


# --------------------------------------------------------------------------------------------------------------------
# Shortcuts in time
# --------------------------------------------------------------------------------------------------------------------

Admissible = Callable[[np.ndarray], "tuple[bool, str | None]"]


def validate_samples(points: np.ndarray, motion_validator: Any, admissible: Admissible) -> str | None:
    """Why a sampled motion is inadmissible, or ``None``: every sample admissible, every chord accepted by the
    problem's motion validator, every configuration it returns admissible (the planner's edge rule)."""
    ok, why = admissible(points[0])
    if not ok:
        return f"sample 0: {why}"
    for i in range(len(points) - 1):
        motion = motion_validator.validate(points[i], points[i + 1])
        if not motion.reached:
            ok, why = admissible(points[i + 1])
            return f"chord {i}: {why}" if not ok else f"chord {i}: motion validator rejected it"
        for q in motion.configs:
            ok, why = admissible(q)
            if not ok:
                return f"chord {i}: {why}"
    return None


def piece_samples(piece: Piece, resolution: float, vmax: np.ndarray) -> np.ndarray:
    """Positions along ``piece`` no farther apart than ``resolution`` in joint space: the joint-space speed is at
    most ``|vmax|``, so a time step of ``resolution / |vmax|`` bounds every chord."""
    T = piece.duration
    n = max(1, math.ceil(T * float(np.linalg.norm(vmax)) / resolution))
    ts = np.linspace(0.0, T, n + 1)
    return np.stack([r.evaluate(ts)[0] for r in piece.ramps], axis=1)


@dataclass(frozen=True)
class ShortcutOptions:
    iterations: int = 100
    patience: int | None = None  # stop after this many attempts without improvement
    time_budget: float | None = None  # seconds
    seed: int = 0
    min_interval: float = 1e-3  # seconds; shorter windows are not worth an attempt


def shortcut(
    traj: Trajectory,
    *,
    motion_validator: Any,
    admissible: Admissible,
    resolution: float,
    options: ShortcutOptions | None = None,
) -> Trajectory:
    """Randomized time-optimized shortcuts: replace ``[t1, t2]`` by the synchronized time-optimal interpolant between
    the states there when it is faster and admissible under the planning problem's boundary. Deterministic for a
    seed (and an iteration budget; a wall-time budget is not)."""
    options = options or ShortcutOptions()
    rng = np.random.default_rng(options.seed)
    vmax, amax = traj.velocity_limits, traj.acceleration_limits
    attempts: list[dict[str, Any]] = []
    started, since_gain = time.perf_counter(), 0
    for it in range(options.iterations):
        if options.time_budget is not None and time.perf_counter() - started > options.time_budget:
            break
        if options.patience is not None and since_gain >= options.patience:
            break
        T = traj.duration
        t1, t2 = np.sort(rng.uniform(0.0, T, 2))
        record: dict[str, Any] = {"iteration": it, "t1": float(t1), "t2": float(t2)}
        attempts.append(record)
        since_gain += 1
        if t2 - t1 < options.min_interval:
            record["rejected"] = "window too short"
            continue
        qa, va, _ = traj.state(float(t1))
        qb, vb, _ = traj.state(float(t2))
        piece = synchronize(qa, va, qb, vb, vmax, amax)
        if piece is None:
            record["rejected"] = "no common duration"
            continue
        gain = (t2 - t1) - piece.duration
        record["gain"] = float(gain)
        if gain <= 1e-9:
            record["rejected"] = "not faster"
            continue
        why = validate_samples(piece_samples(piece, resolution, vmax), motion_validator, admissible)
        if why is not None:
            record["rejected"] = why
            continue
        traj = traj.splice(float(t1), float(t2), piece)
        record["accepted"] = True
        since_gain = 0
    traj.diagnostics = {
        "attempts": attempts,
        "accepted": sum(bool(a.get("accepted")) for a in attempts),
        "seconds": time.perf_counter() - started,
        "options": options,
    }
    return traj
