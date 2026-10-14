# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Motion validation: one implementation, used by the planner and by anyone checking its output (#230).

A ``Validator`` is bound to a joint space, a collision checker, and an optional path constraint, under a
``ValidationPolicy``. It answers for a state, one motion, or a whole path with a ``ValidationReport``: valid
under the reported policy, or the first invalid state found and why.

**Motions are raw-linear.** ``validate_motion(q0, q1)`` validates the straight line in raw joint values, the
motion a runtime executes when it interpolates them. A continuous joint is not wrapped: a half turn or more is
validated as the long raw line it is, never as the short way round.

**The guarantee is discrete.** The policy samples states at most ``resolution`` apart (Euclidean, in joint
space) and claims nothing between them. #222 owns choosing a stronger policy; the report always says which one
answered.

**Timed trajectories** (``validate_trajectory``) are checked over their whole law, not their knots. A piece whose
trace lies on the trajectory's source path (``Trajectory.path``), an exact reparameterization of part of it, is
checked at that path's own states within its span: the evidence is reused only once the equivalence is
established numerically. Every other piece is checked at uniform time steps short enough that no joint-space step
exceeds the resolution under the trajectory's velocity limits, and position limits are checked exactly at every
ramp's turning point. A trajectory that breaks its own invariants (continuity, limits, rest at both ends) is
malformed: ``MalformedTrajectory``.

**The planner uses this.** The default motion validator samples tree edges with the same rule, a planned path is
returned only after ``validate_path`` accepts it (``PlanResult.validation``), the trajectory smoother accepts a
shortcut only if every piece it creates passes the trajectory rule, and a timed trajectory is returned only after
``validate_trajectory`` accepts it (``Trajectory.validation``). The native core has the
same implementation (``sscbirrt::Validator``); ``Validator(..., backend="auto")`` runs it when every component
has a native form.

**Threading.** A validator is not thread-safe: use one per thread. Nothing is cached, so no earlier call can
change a later answer.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sscbirrt.space import JointSpace

SUBJECTS = ("state", "motion", "path", "trajectory")
LAW = "piecewise_constant_acceleration"  # how a Trajectory's position evolves: quadratic on every ramp segment
ON_PATH = 1e-8  # a trajectory point within this of the source path (relative, at least 1 joint unit) lies on it
FAILURES = ("position_limits", "collision", "path_constraint", "unsupported")

# A length within this relative tolerance of a whole number of resolutions counts as that number. The planner stores
# every edge sample as a node, so a path's pieces are often exactly one resolution long; recomputed from the stored
# nodes, one may come out a rounding error longer, and a plain ceil would then check a midpoint nobody planned.
ROUNDING = 1e-9


def subdivisions(distance: float, resolution: float) -> int:
    """How many intervals a straight motion of length ``distance`` is split into:
    ``max(1, ceil(distance / resolution - ROUNDING))``."""
    return max(1, math.ceil(distance / resolution - ROUNDING))


def line_point(q0: np.ndarray, d: np.ndarray, i: int, n: int) -> np.ndarray:
    """The ``i``-th of ``n`` subdivision points of the line from ``q0`` along ``d``: ``q0 + (i/n) d``."""
    return q0 + (i / n) * d


@dataclass(frozen=True)
class ValidationPolicy:
    """What a validator checks and what a valid answer guarantees.

    Today's policy is the planner's edge rule: states along each raw-linear motion at most ``resolution`` apart
    in joint space (``n = max(1, ceil(|q1 - q0| / resolution))`` intervals), both endpoints included. The
    guarantee is discrete: nothing is claimed between the checked states. There is no hidden setting; every
    parameter is on this object and in every report.
    """

    resolution: float = 0.05

    name: str = field(default="joint_l2_subdivision", init=False)
    mechanism: str = field(default="joint_subdivision", init=False)
    guarantee: str = field(default="discrete", init=False)
    ambiguity_band_m: float = field(default=0.0, init=False)

    def __post_init__(self):
        if not (isinstance(self.resolution, (int, float)) and math.isfinite(self.resolution) and self.resolution > 0):
            raise ValueError(f"resolution must be a positive finite number, got {self.resolution!r}")
        object.__setattr__(self, "resolution", float(self.resolution))

    @classmethod
    def from_config(cls, config) -> ValidationPolicy:
        """The policy a planner with this ``CBiRRTConfig`` applies: ``edge_resolution``, else ``step_size``."""
        return cls(resolution=config.edge_resolution or config.step_size)

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        """Every other setting of the mechanism, as sorted ``(key, value)`` pairs."""
        return (
            ("endpoints", "checked"),
            ("interpolation", "raw_linear"),
            ("rounding", f"{ROUNDING:g}"),
            ("trajectory", "source_path_states_or_speed_bounded_time_steps"),
            ("trajectory_limits", "exact"),
        )

    @property
    def summary(self) -> str:
        return (
            f"Valid at states at most {self.resolution:g} apart (Euclidean, joint space, to a relative "
            f"{ROUNDING:g}) along each raw-linear motion, endpoints included; nothing is claimed between them."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "guarantee": self.guarantee,
            "resolution": self.resolution,
            "ambiguity_band_m": self.ambiguity_band_m,
            "parameters": [list(p) for p in self.parameters],
            "summary": self.summary,
        }


@dataclass(frozen=True)
class Contact:
    """A disallowed contact at the invalid state: bodies by name, geoms by id, signed distance (negative is
    penetration), and the kind of pair (for the MuJoCo scene, ``self_collision`` or ``robot_environment``)."""

    kind: str
    body1: str
    body2: str
    geom1: int
    geom2: int
    distance: float


@dataclass(frozen=True)
class InvalidWitness:
    """The first invalid state found, where it is, and why.

    Attributes:
        failure: One of ``FAILURES``.
        configuration: The invalid configuration.
        segment: The interval it lies on: 0 for a motion, the segment for a path, the piece for a trajectory; None
            for a state.
        fraction: Its position on that interval, ``i/n``, in [0, 1], for a motion or path.
        time_s: Its time, for a trajectory.
        contacts: The disallowed contacts, when the collision checker can list them.
        constraint: For a path-constraint failure, the index of the first violated member of an ``AllOf`` path
            constraint (0 for any other constraint).
        detail: The reason in words.
    """

    failure: str
    configuration: tuple[float, ...]
    segment: int | None = None
    fraction: float | None = None
    time_s: float | None = None
    contacts: tuple[Contact, ...] = ()
    constraint: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class ValidationReport:
    """One validator's answer for one subject.

    Attributes:
        subject: ``state``, ``motion``, ``path``, or ``trajectory``.
        valid: Whether the subject is valid under ``policy``.
        intervals: Intervals in the subject: 0 for a state, 1 for a motion, waypoints minus one for a path, pieces
            for a trajectory.
        checks: States checked, the invalid one included.
        policy: The policy applied.
        implementation: ``sscbirrt X.Y.Z``.
        backend: ``python`` or ``native``: which implementation of the one contract answered.
        witness: The first invalid state; present exactly when not valid.
        interpolation: ``raw_linear`` for a motion or path; None otherwise.
        law: ``piecewise_constant_acceleration`` for a trajectory; None otherwise.
        duration_s: A trajectory's duration; None otherwise.
        provenance: What the collision checker validated against (scene and snapshot hashes for the native
            MuJoCo scene), as sorted ``(key, value)`` pairs.
    """

    subject: str
    valid: bool
    intervals: int
    checks: int
    policy: ValidationPolicy
    implementation: str
    backend: str
    witness: InvalidWitness | None = None
    interpolation: str | None = None
    law: str | None = None
    duration_s: float | None = None
    provenance: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready form, for artifacts and traces."""
        out = asdict(self)
        out["policy"] = self.policy.to_dict()
        out["provenance"] = [list(p) for p in self.provenance]
        if self.witness is not None:
            out["witness"]["configuration"] = list(self.witness.configuration)
            out["witness"]["contacts"] = [asdict(c) for c in self.witness.contacts]
        return out


class MalformedTrajectory(ValueError):
    """A trajectory that breaks its own invariants: a shape, finiteness, duration, continuity, limit or rest
    condition. It is refused, never validated more weakly."""


def implementation() -> str:
    """``sscbirrt X.Y.Z`` for the installed package."""
    from sscbirrt.planner import versions

    return f"sscbirrt {versions().get('sscbirrt', 'unknown')}"


def classify(
    space: JointSpace, checker, path_constraint, q: np.ndarray, valid: bool | None = None
) -> tuple[str, str] | None:
    """Why ``q`` may not appear on a path, as ``(failure, detail)``, or None if it may.

    The one state rule, in order: membership in the joint space (limits), then the collision checker, then the
    path constraint. ``valid`` is the checker's verdict when it is already known (#200). The planner's
    admissibility and every validator answer come from here.
    """
    why = space.why_invalid(q)
    if why is not None:
        return "position_limits", f"outside joint space ({why})"
    if not (checker.is_valid(q) if valid is None else valid):
        return "collision", "in collision"
    if path_constraint is not None and not path_constraint.contains(q):
        return "path_constraint", "violates path constraints"
    return None


def _config(q, dof: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(q, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not numeric: {q!r}") from None
    if arr.shape != (dof,):
        raise ValueError(f"{name}: shape {arr.shape} != ({dof},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: non-finite entries: {arr}")
    return arr


class Validator:
    """The validator for one joint space, collision checker, and path constraint, under one policy.

    Args:
        space: The joint space (limits and topology).
        checker: The collision checker (``is_valid(q)``; ``invalid_contacts(q)`` too, for contact witnesses).
        path_constraint: A set every state must be a member of, or None.
        policy: The ``ValidationPolicy``; default ``ValidationPolicy()``.
        backend: ``"auto"``: the native core when every component has a native form, else this Python
            reference; ``"native"``: the native core or ``NativeUnsupported``; ``"python"``: the reference.

    Malformed input (wrong shape, non-finite values, an empty path) raises ``ValueError``; it is never coerced
    into a weaker check.
    """

    def __init__(
        self,
        space: JointSpace,
        checker,
        path_constraint=None,
        *,
        policy: ValidationPolicy | None = None,
        backend: str = "auto",
    ):
        if backend not in ("python", "native", "auto"):
            raise ValueError(f'backend must be "python", "native", or "auto", got {backend!r}')
        self.space = space
        self.checker = checker
        self.path_constraint = path_constraint
        self.policy = policy or ValidationPolicy()
        self.backend = "python"
        self.backend_reasons: tuple[str, ...] = ()
        self._native = None
        provenance = getattr(checker, "provenance", None)
        self.provenance = tuple(sorted((str(k), str(v)) for k, v in dict(provenance or {}).items()))
        if backend != "python":
            from sscbirrt.backends import native

            try:
                self._native = native.lower_validator(space, checker, path_constraint, self.policy)
                self.backend = "native"
            except native.NativeUnsupported as e:
                if backend == "native":
                    raise
                self.backend_reasons = tuple(e.reasons)

    @classmethod
    def for_problem(cls, problem, config=None, *, backend: str = "auto") -> Validator:
        """The validator a planner with ``config`` (default ``CBiRRTConfig()``) applies to ``problem``."""
        from sscbirrt.config import CBiRRTConfig

        if problem.motion_validator is not None:
            raise ValueError(
                f"the problem's motion validator ({type(problem.motion_validator).__name__}) replaces the default "
                "edge rule, so no ValidationPolicy describes its motions"
            )
        return cls(
            problem.space,
            problem.validator,
            problem.path_constraint,
            policy=ValidationPolicy.from_config(config or CBiRRTConfig()),
            backend=backend,
        )

    # ----- public API ----------------------------------------------------------------------------------------------

    def validate_state(self, q) -> ValidationReport:
        """Whether one configuration is valid."""
        q = _config(q, self.space.dof, "q")
        if self._native is not None:
            return self._from_native(self._native.validate_state(q.tolist()))
        witness = self._witness(q, None, None)
        return self._report("state", 0, 1, witness)

    def validate_motion(self, q0, q1) -> ValidationReport:
        """Whether the raw-linear motion from ``q0`` to ``q1`` is valid, both endpoints included."""
        q0 = _config(q0, self.space.dof, "q0")
        q1 = _config(q1, self.space.dof, "q1")
        if self._native is not None:
            return self._from_native(self._native.validate_motion(q0.tolist(), q1.tolist()))
        witness = self._witness(q0, 0, 0.0)
        checks = 1
        if witness is None:
            more, witness = self._segment(q0, q1, 0)
            checks += more
        return self._report("motion", 1, checks, witness)

    def validate_path(self, path: Sequence) -> ValidationReport:
        """Whether a path is valid: its first waypoint, then each raw-linear segment in order."""
        if len(path) == 0:
            raise ValueError("path: empty")
        qs = [_config(q, self.space.dof, f"path[{i}]") for i, q in enumerate(path)]
        if self._native is not None:
            return self._from_native(self._native.validate_path([q.tolist() for q in qs]))
        witness = self._witness(qs[0], 0 if len(qs) > 1 else None, 0.0 if len(qs) > 1 else None)
        checks = 1
        for k in range(len(qs) - 1):
            if witness is not None:
                break
            more, witness = self._segment(qs[k], qs[k + 1], k)
            checks += more
        return self._report("path", len(qs) - 1, checks, witness)

    def validate_trajectory(self, trajectory) -> ValidationReport:
        """Whether a timed trajectory is valid over its whole law: its start, then every piece in time order.

        Raises ``MalformedTrajectory`` for a trajectory that breaks its own invariants (see the module docstring).
        The witness of an invalid one gives its piece (``segment``) and ``time_s``.
        """
        vmax, amax = well_formed(trajectory, self.space.dof)
        path = getattr(trajectory, "path", None)
        if self._native is not None:
            from sscbirrt.backends.native import trajectory_data

            pieces = trajectory_data(trajectory)
            native_path = None if path is None else [list(map(float, q)) for q in path]
            return self._from_native(
                self._native.validate_trajectory(pieces, vmax.tolist(), amax.tolist(), native_path)
            )
        evidence = None if path is None else PathEvidence(path, self.policy.resolution)
        breaks = trajectory.breaks
        start = np.array([r.x0 for r in trajectory.pieces[0].ramps])
        witness = self._witness(start, 0, None, 0.0)
        checks = 1
        for k, piece in enumerate(trajectory.pieces):
            if witness is not None:
                break
            more, witness = self.check_piece(piece, k, float(breaks[k]), vmax, evidence)
            checks += more
        return self._report("trajectory", len(trajectory.pieces), checks, witness, float(trajectory.duration))

    def check_piece(self, piece, index: int, t0: float, vmax: np.ndarray, evidence: PathEvidence | None):
        """Check the states of one piece after its start, in time order: (states checked, the first invalid one).

        A piece whose trace lies on the source path is checked at the path's states within its span and at its end;
        any other piece at ``n = max(1, ceil(T |vmax| / resolution - ROUNDING))`` uniform time steps, with every
        bounded joint's out-of-limits turning point added. The smoother accepts a shortcut only if each piece it
        creates passes this, so a returned trajectory passes ``validate_trajectory``.
        """
        T = piece.duration
        on_path = None if evidence is None else evidence.straight_span(piece)
        if on_path is not None:
            states, ref = on_path
            for i, (_, q) in enumerate(states):
                witness = self._witness(q, index, None, None)
                if witness is not None:
                    t = t0 + (T if i == len(states) - 1 else _time_at(piece.ramps[ref], q[ref], T))
                    return i + 1, dataclasses.replace(witness, time_s=t)
            return len(states), None
        ts = _time_steps(piece, self.policy.resolution, vmax, self.space)
        qs = np.stack([r.evaluate(ts)[0] for r in piece.ramps], axis=1)
        for i, (t, q) in enumerate(zip(ts, qs)):
            witness = self._witness(q, index, None, t0 + float(t))
            if witness is not None:
                return i + 1, witness
        return len(ts), None

    # ----- the reference implementation ----------------------------------------------------------------------------

    def _segment(self, q0: np.ndarray, q1: np.ndarray, segment: int) -> tuple[int, InvalidWitness | None]:
        """Check the states after ``q0`` on the raw line to ``q1``: (states checked, the first invalid one)."""
        d = q1 - q0
        n = subdivisions(float(np.linalg.norm(d)), self.policy.resolution)
        for i in range(1, n + 1):
            q = q1 if i == n else line_point(q0, d, i, n)
            witness = self._witness(q, segment, i / n)
            if witness is not None:
                return i, witness
        return n, None

    def _witness(
        self, q: np.ndarray, segment: int | None, fraction: float | None, time_s: float | None = None
    ) -> InvalidWitness | None:
        why = classify(self.space, self.checker, self.path_constraint, q)
        if why is None:
            return None
        failure, detail = why
        contacts: tuple[Contact, ...] = ()
        constraint = None
        if failure == "collision" and hasattr(self.checker, "invalid_contacts"):
            contacts = tuple(
                Contact(str(c["kind"]), str(c["body1"]), str(c["body2"]), int(c["geom1"]), int(c["geom2"]),
                        float(c["dist"]))
                for c in self.checker.invalid_contacts(q)
            )  # fmt: skip
        if failure == "path_constraint":
            constraint = _violated_member(self.path_constraint, q)
        return InvalidWitness(
            failure=failure,
            configuration=tuple(float(x) for x in q),
            segment=segment,
            fraction=fraction,
            time_s=time_s,
            contacts=contacts,
            constraint=constraint,
            detail=detail,
        )

    def _report(
        self, subject: str, intervals: int, checks: int, witness: InvalidWitness | None, duration: float | None = None
    ) -> ValidationReport:
        return ValidationReport(
            subject=subject,
            valid=witness is None,
            intervals=intervals,
            checks=checks,
            policy=self.policy,
            implementation=implementation(),
            backend="python",
            witness=witness,
            interpolation="raw_linear" if subject in ("motion", "path") else None,
            law=LAW if subject == "trajectory" else None,
            duration_s=duration,
            provenance=self.provenance,
        )

    def _from_native(self, r) -> ValidationReport:
        from sscbirrt.backends.native import report_from_native

        return report_from_native(r, self.provenance)


def _violated_member(path_constraint, q: np.ndarray) -> int:
    from sscbirrt.sets import AllOf

    if isinstance(path_constraint, AllOf):
        for i, child in enumerate(path_constraint.children):
            if not child.contains(q):
                return i
    return 0


# ----- trajectories ------------------------------------------------------------------------------------------------

_REL = 1e-9  # relative slack on velocity and acceleration bounds, as the smoother's own ramp verification
_TOL_X = 1e-9  # position continuity (joint units, relative to at least 1)
_TOL_V = 1e-7  # velocity continuity and rest (relative to max(1, vmax))


def well_formed(trajectory, dof: int) -> tuple[np.ndarray, np.ndarray]:
    """The trajectory's velocity and acceleration limits, after checking its own invariants, or
    ``MalformedTrajectory`` naming the first one broken: shapes and finiteness, every ramp of a piece lasting the
    piece's duration, ``|v| <= vmax`` and ``|a| <= amax``, position and velocity continuous across pieces, and rest
    at both ends."""
    try:
        vmax = np.asarray(trajectory.velocity_limits, dtype=float)
        amax = np.asarray(trajectory.acceleration_limits, dtype=float)
        pieces = list(trajectory.pieces)
    except (AttributeError, TypeError, ValueError) as e:
        raise MalformedTrajectory(f"not a trajectory: {e}") from None
    for name, lim in (("velocity_limits", vmax), ("acceleration_limits", amax)):
        if lim.shape != (dof,) or not np.all(np.isfinite(lim)) or not np.all(lim > 0):
            raise MalformedTrajectory(f"{name} must be {dof} positive finite values, got {lim}")
    if not pieces:
        raise MalformedTrajectory("no pieces")
    previous = None
    for k, piece in enumerate(pieces):
        if len(piece.ramps) != dof:
            raise MalformedTrajectory(f"piece {k}: {len(piece.ramps)} ramps for {dof} joints")
        T = piece.duration
        for j, r in enumerate(piece.ramps):
            values = [r.x0, r.v0] + [x for seg in r.pieces for x in seg]
            if not all(math.isfinite(x) for x in values) or any(d < -1e-12 for d, _ in r.pieces):
                raise MalformedTrajectory(f"piece {k}, joint {j}: non-finite values or a negative duration")
            if abs(r.duration - T) > 1e-9 * max(1.0, T):
                raise MalformedTrajectory(f"piece {k}, joint {j}: lasts {r.duration}, the piece {T}")
            v = r.v0
            if abs(v) > vmax[j] * (1 + _REL):
                raise MalformedTrajectory(f"piece {k}, joint {j}: velocity {v} beyond {vmax[j]}")
            for d, a in r.pieces:
                if abs(a) > amax[j] * (1 + _REL):
                    raise MalformedTrajectory(f"piece {k}, joint {j}: acceleration {a} beyond {amax[j]}")
                v = v + a * d
                if abs(v) > vmax[j] * (1 + _REL):
                    raise MalformedTrajectory(f"piece {k}, joint {j}: velocity {v} beyond {vmax[j]}")
        if previous is not None:
            for j, r in enumerate(piece.ramps):
                x, v = previous[j]
                if abs(r.x0 - x) > _TOL_X * max(1.0, abs(x)) or abs(r.v0 - v) > _TOL_V * max(1.0, vmax[j]):
                    raise MalformedTrajectory(f"piece {k}, joint {j}: discontinuous with the piece before")
        previous = [r.end() for r in piece.ramps]
    for j in range(dof):
        if abs(pieces[0].ramps[j].v0) > _TOL_V * max(1.0, vmax[j]) or abs(previous[j][1]) > _TOL_V * max(1.0, vmax[j]):
            raise MalformedTrajectory(f"joint {j}: not at rest at both ends")
    return vmax, amax


def _time_steps(piece, resolution: float, vmax: np.ndarray, space: JointSpace) -> np.ndarray:
    """The times after a piece's start that are checked: uniform steps over which no joint-space step exceeds the
    resolution (``|q(t + dt) - q(t)| <= dt |vmax|``), and every bounded joint's turning point that leaves its limits,
    so position limits hold exactly along the law."""
    T = piece.duration
    n = max(1, math.ceil(T * float(np.linalg.norm(vmax)) / resolution - ROUNDING))
    ts = list(np.linspace(0.0, T, n + 1)[1:])
    bounded = np.ones(space.dof, dtype=bool) if space.continuous_joints is None else ~space.continuous_joints
    extra = []
    for j, r in enumerate(piece.ramps):
        if not bounded[j]:
            continue
        start, x, v = 0.0, r.x0, r.v0
        for d, a in r.pieces:
            if a != 0.0:
                tau = -v / a
                if 0.0 < tau < d:
                    peak = x + v * tau + 0.5 * a * tau * tau
                    if not space.lower[j] <= peak <= space.upper[j]:
                        extra.append(start + tau)
            x, v, start = x + v * d + 0.5 * a * d * d, v + a * d, start + d
    return np.array(sorted(ts + extra), dtype=float)


def _time_at(ramp, x: float, T: float) -> float:
    """The time a monotone ramp reaches position ``x``, by bisection (60 halvings of ``[0, T]``)."""
    lo, hi = 0.0, T
    x0 = float(ramp.evaluate(np.array([0.0]))[0][0])
    x1 = float(ramp.evaluate(np.array([T]))[0][0])
    rising = x1 >= x0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        xm = float(ramp.evaluate(np.array([mid]))[0][0])
        if (xm < x) == rising:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class PathEvidence:
    """The states ``validate_path`` checks on a trajectory's source path, each with its path parameter
    ``k + i/n``, and the test that a straight piece's trace lies on the path."""

    def __init__(self, path: Sequence, resolution: float):
        self.nodes = [np.asarray(q, dtype=float) for q in path]
        if not self.nodes:
            raise MalformedTrajectory("the source path is empty")
        self.states: list[tuple[float, np.ndarray]] = [(0.0, self.nodes[0])]
        for k in range(len(self.nodes) - 1):
            q0, q1 = self.nodes[k], self.nodes[k + 1]
            d = q1 - q0
            n = subdivisions(float(np.linalg.norm(d)), resolution)
            for i in range(1, n + 1):
                self.states.append((k + i / n, q1 if i == n else line_point(q0, d, i, n)))

    def locate(self, x: np.ndarray) -> float | None:
        """The path parameter of the first point of the path within ``ON_PATH`` of ``x``, or None."""
        tol = ON_PATH * max(1.0, float(np.max(np.abs(x))))
        if len(self.nodes) == 1:
            return 0.0 if float(np.linalg.norm(self.nodes[0] - x)) <= tol else None
        for k in range(len(self.nodes) - 1):
            p, s = self.nodes[k], self.nodes[k + 1] - self.nodes[k]
            ss = float(np.dot(s, s))
            f = 0.0 if ss == 0.0 else min(1.0, max(0.0, float(np.dot(x - p, s)) / ss))
            if float(np.linalg.norm(p + f * s - x)) <= tol:
                return k + f
        return None

    def straight_span(self, piece):
        """For a piece that moves along a straight line lying on the path, forward: the states to check after its
        start (the path's states strictly inside its span, then its end) and its reference joint. None otherwise."""
        a = np.array([float(r.evaluate(np.array([0.0]))[0][0]) for r in piece.ramps])
        b = np.array([float(r.evaluate(np.array([piece.duration]))[0][0]) for r in piece.ramps])
        ref = straight_reference(piece, a, b)
        if ref is None:
            return None
        la, lb = self.locate(a), self.locate(b)
        if la is None or lb is None or lb < la:
            return None
        D = b - a
        DD = float(np.dot(D, D))
        tol = ON_PATH * max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        for k in range(math.floor(la) + 1, math.ceil(lb)):  # every node strictly inside the span lies on a -> b
            p = self.nodes[k]
            f = min(1.0, max(0.0, float(np.dot(p - a, D)) / DD))
            if float(np.linalg.norm(a + f * D - p)) > tol:
                return None
        inside = [(lam, q) for lam, q in self.states if la < lam < lb]
        return inside + [(lb, b)], ref


def straight_reference(piece, a: np.ndarray, b: np.ndarray) -> int | None:
    """The reference joint of a piece that moves monotonically along the straight line ``a -> b``: every joint's
    ramp is the reference joint's scaled by ``(b_j - a_j) / (b_r - a_r)`` (same segment durations, proportional
    start velocity and accelerations), and the reference never moves backwards. None otherwise."""
    D = b - a
    ref = int(np.argmax(np.abs(D)))
    if D[ref] == 0.0:
        return None
    r0 = piece.ramps[ref]
    durations = [d for d, _ in r0.pieces]
    for j, r in enumerate(piece.ramps):
        if [d for d, _ in r.pieces] != durations:
            return None
        ratio = D[j] / D[ref]
        if abs(r.v0 - ratio * r0.v0) > 1e-9 * max(1.0, abs(r0.v0)):
            return None
        for (_, aj), (_, ar) in zip(r.pieces, r0.pieces):
            if abs(aj - ratio * ar) > 1e-9 * max(1.0, abs(ar)):
                return None
    sign = 1.0 if D[ref] > 0 else -1.0
    v = r0.v0
    if sign * v < -1e-9 * max(1.0, abs(v)):
        return None
    for d, a in r0.pieces:
        v = v + a * d
        if sign * v < -1e-9 * max(1.0, abs(a * d)):
            return None
    return ref
