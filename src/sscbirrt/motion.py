# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Local-motion validation: the boundary every tree edge passes through.

A state validator answers whether one configuration is valid. A motion
validator answers whether the local motion between two configurations is
valid, and returns the configurations to store along it. The planner uses
one motion validator for ordinary growth, the final connection between
trees, and shortcut smoothing.

**Replacement, not composition.** A custom ``PlanningProblem.motion_validator``
*replaces* the default ``DiscreteMotionValidator`` and owns the validity of
the complete motion between ``q_from`` and ``q_to``, interior included. The
planner does not run the default checks alongside it. This is what lets a
backend supply continuous collision checking without paying for redundant
discrete samples, and it means a custom validator that skips the interior
is trusted to have checked it. To *add* a restriction while keeping the
default discrete checks, compose explicitly with ``RestrictedMotionValidator``.

**What the planner guarantees regardless of validator.** Every
configuration a validator returns is checked for admissibility (joint
space, state validator, path constraint) before it is stored as a node, and
a ``LocalMotion`` must satisfy its contract (see ``LocalMotion``; violations
raise ``MotionContractError``). Node admissibility says nothing about the
motion *between* nodes; that is the validator's responsibility.

**Two modes** (#230). ``validate`` answers *how far from the start* the motion is
valid: its prefix becomes tree nodes, so it checks in order. Where only *whether
the whole motion* is valid matters (path shortcuts, whose tree is discarded on a
no), the planner calls ``validate_whole`` instead (see ``validate_whole``): the
same states, visited in van der Corput order, which reaches an obstacle sooner.
A validator without ``validate_whole`` answers through ``validate``.

**Reversibility.** The search is bidirectional: the goal tree grows toward
the start, so its edges are validated with ``q_from`` on the goal side and
executed in the opposite direction. A custom validator must therefore treat
a motion as valid independently of direction, or accept that half of the
path's edges were validated in reverse.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from sscbirrt.space import JointSpace
from sscbirrt.validation import line_point, spread_order, subdivisions


@dataclass
class LocalMotion:
    """The validated part of a local motion from ``q_from`` toward ``q_to``.

    Attributes:
        configs: Validated configurations after ``q_from``, in order, to be
            stored as consecutive tree nodes. Empty if the first sample was
            invalid. If ``reached`` is True the last entry is ``q_to`` exactly
            and, for a nonzero motion, ``configs`` is nonempty; the planner
            raises ``MotionContractError`` otherwise.
        reached: Whether the whole motion was valid and ends at ``q_to``.
    """

    configs: list[np.ndarray] = field(default_factory=list)
    reached: bool = False


@runtime_checkable
class MotionValidator(Protocol):
    """Validate the local motion between two configurations.

    Returns the validated prefix of the motion (see ``LocalMotion``); a
    partially valid motion returns the configurations up to the first invalid
    one with ``reached=False``, which lets tree growth keep its progress.
    """

    def validate(self, q_from: np.ndarray, q_to: np.ndarray) -> LocalMotion: ...


def validate_whole(validator, q_from: np.ndarray, q_to: np.ndarray) -> list[np.ndarray] | None:
    """Whether the whole motion is valid: the configurations ``validator.validate`` would return with
    ``reached=True``, or None. Uses the validator's own ``validate_whole`` (optional; the same states in any order,
    the same configurations on a yes) when it has one, else ``validate``."""
    method = getattr(validator, "validate_whole", None)
    if method is not None:
        return method(q_from, q_to)
    motion = validator.validate(q_from, q_to)
    return list(motion.configs) if motion.reached else None


class RestrictedMotionValidator:
    """Compose a base validator with an extra motion predicate.

    The base validator (typically the planner's default, from
    ``CBiRRT.default_motion_validator(problem)``) decides validity and
    produces the configurations to store; the motion is then accepted only
    if ``accepts(q_from, q_to)`` is also true. A rejected motion returns an
    empty ``LocalMotion``. This is the explicit way to be stricter than the
    default while keeping its discrete checks; a bare custom validator
    replaces the default instead.
    """

    def __init__(self, base: MotionValidator, accepts: Callable[[np.ndarray, np.ndarray], bool]):
        self.base = base
        self.accepts = accepts

    def validate(self, q_from: np.ndarray, q_to: np.ndarray) -> LocalMotion:
        if not self.accepts(np.asarray(q_from), np.asarray(q_to)):
            return LocalMotion()
        return self.base.validate(q_from, q_to)

    def validate_whole(self, q_from: np.ndarray, q_to: np.ndarray) -> list[np.ndarray] | None:
        if not self.accepts(np.asarray(q_from), np.asarray(q_to)):
            return None
        return validate_whole(self.base, q_from, q_to)


class DiscreteMotionValidator:
    """Straight-line motion, checked every ``resolution`` along the segment.

    Samples ``q_from + (i/n) * direction`` for ``i = 1..n`` with
    ``n = ceil(distance / resolution)`` under the space's wrapped direction;
    the last sample is ``q_to`` exactly. Each sample must satisfy
    ``is_admissible``. Validity is therefore discrete: an obstacle thinner
    than ``resolution`` can be stepped over. Choose the resolution against
    the thinnest feature you must not miss.

    The subdivision is ``sscbirrt.validation``'s, so an edge is checked at the
    states ``Validator.validate_motion`` checks on the raw line from ``q_from``
    along the wrapped direction; the planner returns its path in those raw
    values (``JointSpace.unwrap_path``) and validates it whole before success.
    """

    def __init__(self, space: JointSpace, is_admissible: Callable[[np.ndarray], bool], resolution: float):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.space = space
        self.is_admissible = is_admissible
        self.resolution = resolution

    def validate(self, q_from: np.ndarray, q_to: np.ndarray) -> LocalMotion:
        space = self.space
        if not space.contains(q_to):
            return LocalMotion()
        q_from = np.asarray(q_from, dtype=float)
        direction = space.direction(q_from, q_to)
        distance = float(np.linalg.norm(direction))
        if distance == 0.0:
            return LocalMotion(configs=[], reached=True)
        n_steps = subdivisions(distance, self.resolution)
        configs: list[np.ndarray] = []
        for i in range(1, n_steps + 1):
            q = np.array(q_to, dtype=float) if i == n_steps else line_point(q_from, direction, i, n_steps)
            if not self.is_admissible(q):
                return LocalMotion(configs=configs, reached=False)
            configs.append(q)
        return LocalMotion(configs=configs, reached=True)

    def samples(self, q_from: np.ndarray, q_to: np.ndarray) -> list[np.ndarray] | None:
        """The configurations ``validate`` checks and, on success, returns, unchecked; None if ``q_to`` is outside
        the space (``validate`` then returns an empty ``LocalMotion``)."""
        if not self.space.contains(q_to):
            return None
        q_from = np.asarray(q_from, dtype=float)
        direction = self.space.direction(q_from, q_to)
        distance = float(np.linalg.norm(direction))
        if distance == 0.0:
            return []
        n = subdivisions(distance, self.resolution)
        return [np.array(q_to, dtype=float) if i == n else line_point(q_from, direction, i, n) for i in range(1, n + 1)]

    def validate_whole(self, q_from: np.ndarray, q_to: np.ndarray) -> list[np.ndarray] | None:
        """``validate``'s configurations if every one is admissible, else None: the same samples, checked in van der
        Corput order (``spread_order``), stopping at the first inadmissible one found."""
        configs = self.samples(q_from, q_to)
        if configs is None:
            return None
        for k in spread_order(len(configs)):
            if not self.is_admissible(configs[k]):
                return None
        return configs
