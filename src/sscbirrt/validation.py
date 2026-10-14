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

**The planner uses this.** The default motion validator samples tree edges with the same rule, and a planned
path is returned only after ``validate_path`` accepts it (``PlanResult.validation``). The native core has the
same implementation (``sscbirrt::Validator``); ``Validator(..., backend="auto")`` runs it when every component
has a native form.

**Threading.** A validator is not thread-safe: use one per thread. Nothing is cached, so no earlier call can
change a later answer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sscbirrt.space import JointSpace

SUBJECTS = ("state", "motion", "path")
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
        return (("endpoints", "checked"), ("interpolation", "raw_linear"), ("rounding", f"{ROUNDING:g}"))

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
        segment: The interval it lies on, for a motion (always 0) or a path; None for a state.
        fraction: Its position on that interval, ``i/n``, in [0, 1]; None for a state.
        contacts: The disallowed contacts, when the collision checker can list them.
        constraint: For a path-constraint failure, the index of the first violated member of an ``AllOf`` path
            constraint (0 for any other constraint).
        detail: The reason in words.
    """

    failure: str
    configuration: tuple[float, ...]
    segment: int | None = None
    fraction: float | None = None
    contacts: tuple[Contact, ...] = ()
    constraint: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class ValidationReport:
    """One validator's answer for one subject.

    Attributes:
        subject: ``state``, ``motion``, or ``path``.
        valid: Whether the subject is valid under ``policy``.
        intervals: Intervals in the subject: 0 for a state, 1 for a motion, waypoints minus one for a path.
        checks: States checked, the invalid one included.
        policy: The policy applied.
        implementation: ``sscbirrt X.Y.Z``.
        backend: ``python`` or ``native``: which implementation of the one contract answered.
        witness: The first invalid state; present exactly when not valid.
        interpolation: ``raw_linear`` for a motion or path; None for a state.
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

    def _witness(self, q: np.ndarray, segment: int | None, fraction: float | None) -> InvalidWitness | None:
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
            contacts=contacts,
            constraint=constraint,
            detail=detail,
        )

    def _report(self, subject: str, intervals: int, checks: int, witness: InvalidWitness | None) -> ValidationReport:
        return ValidationReport(
            subject=subject,
            valid=witness is None,
            intervals=intervals,
            checks=checks,
            policy=self.policy,
            implementation=implementation(),
            backend="python",
            witness=witness,
            interpolation=None if subject == "state" else "raw_linear",
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
