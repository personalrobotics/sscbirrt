# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""State sets: membership semantics, Boolean composition, and optional capabilities.

A state set is a subset of the configuration space. Its only required
operation is membership (``contains``). Sampling, distance evaluation, and
projection are optional capabilities that a concrete set may or may not
provide. Composition via ``AnyOf`` (union) and ``AllOf`` (intersection) is
explicit and nests arbitrarily.

Capability rules for composites:

- A single-child ``AnyOf`` or ``AllOf`` delegates every capability to that child.
- A multi-child ``AnyOf`` samples only with an explicit mixture policy (``weights``),
  and projects by taking the nearest successful child projection.
- A multi-child ``AllOf`` projects only with an explicit named strategy such as
  ``MostViolatedProjection``, and samples only with one such as
  ``RejectionSampling``.

Use ``supports(s, Capability)`` to query what a set can do. Calling an
unsupported capability raises ``UnsupportedCapability``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from sscbirrt.exceptions import UnsupportedCapability

Metric = Callable[[np.ndarray, np.ndarray], float]


def euclidean(q1: np.ndarray, q2: np.ndarray) -> float:
    """Default metric: Euclidean distance between configurations."""
    return float(np.linalg.norm(np.asarray(q2) - np.asarray(q1)))


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class StateSet(Protocol):
    """A subset of configuration space. Membership is the only required operation."""

    def contains(self, q: np.ndarray) -> bool: ...


@dataclass(frozen=True)
class Sample:
    """A configuration drawn from a set, with provenance.

    Attributes:
        q: The sampled configuration.
        source: Indices identifying which alternative produced the sample,
            outermost choice first. ``AnyOf`` prepends the index of the child
            it chose; ``FiniteSet`` contributes the index of the member it
            drew; other leaves contribute nothing.
        key: Equal for candidates of one draw that differ only by full turns
            of revolute joints, so the same physical configuration; a validator
            that declares ``full_turn_invariant`` judges them alike (#200).
            ``None``: no key.
    """

    q: np.ndarray
    source: tuple[int, ...] = field(default=())
    key: tuple[int, ...] | None = None


@runtime_checkable
class SetSampler(Protocol):
    """Capability: draw candidate members of the set.

    One draw may yield several candidates, for example every IK branch of
    one sampled pose. The caller validates them and keeps what it wants;
    the set never applies collision or other external filters itself.
    """

    def sample(self, rng: np.random.Generator) -> list[Sample]:
        """Return the candidates of one draw; empty if the draw failed."""
        ...


@runtime_checkable
class SetDistance(Protocol):
    """Capability: measure violation of the set.

    ``distance(q)`` is nonnegative and is at most the set's membership
    tolerance exactly when ``contains(q)`` holds. It need not be a metric
    distance to the set; for composites it is a violation measure.
    """

    def distance(self, q: np.ndarray) -> float: ...


@runtime_checkable
class SetViolation(Protocol):
    """Capability: how far outside the set a configuration is, on the set's own scale.

    ``violation(q)`` is nonnegative, exactly zero when ``contains(q)`` holds,
    and positive otherwise. Unlike ``distance`` it already accounts for the
    set's membership tolerance, so ``violation(q) == 0`` and ``contains(q)``
    always agree. Strategies that compare violations across children, such
    as ``MostViolatedProjection``, require the children's violations to be
    on a comparable scale (for example, all residual TSR distances in the
    same units); the strategy cannot check that, so the caller must.
    """

    def violation(self, q: np.ndarray) -> float: ...


@runtime_checkable
class SetProjector(Protocol):
    """Capability: move a configuration onto the set.

    ``q_previous`` is the configuration the proposal was extended from. A
    projector may use it to seed an iterative solve or to reject results
    that moved too far from where the extension started.
    """

    def project(self, q_previous: np.ndarray, q_proposed: np.ndarray) -> np.ndarray | None: ...


class IntersectionProjection(Protocol):
    """A named strategy for projecting onto an intersection of sets.

    A strategy may additionally define ``requires(children)`` to validate
    child capabilities eagerly; ``AllOf`` calls it at construction if present.
    """

    def project(
        self,
        children: Sequence[StateSet],
        q_previous: np.ndarray,
        q_proposed: np.ndarray,
    ) -> np.ndarray | None: ...


class IntersectionSampling(Protocol):
    """A named strategy for sampling an intersection of sets.

    May define ``requires(children)`` like ``IntersectionProjection``.
    """

    def sample(self, children: Sequence[StateSet], rng: np.random.Generator) -> list[Sample]: ...


def supports(s: object, capability: type) -> bool:
    """Whether ``s`` provides ``capability`` (``SetSampler``, ``SetDistance``, ``SetViolation``, ``SetProjector``).

    Composites declare their capabilities explicitly via a ``capabilities``
    attribute; leaves are checked structurally.
    """
    declared = getattr(s, "capabilities", None)
    if declared is not None:
        return capability in declared
    return isinstance(s, capability)


# ---------------------------------------------------------------------------
# Leaf sets
# ---------------------------------------------------------------------------


class FiniteSet:
    """A finite collection of configurations.

    Supports membership (within ``tolerance`` under ``metric``), distance to
    the nearest member, violation (distance beyond the tolerance), and
    uniform sampling. Provides no projector.
    """

    def __init__(
        self,
        configs: Sequence[np.ndarray],
        tolerance: float = 1e-6,
        metric: Metric = euclidean,
    ):
        if len(configs) == 0:
            raise ValueError("FiniteSet requires at least one configuration")
        self.configs = [np.array(c, dtype=float) for c in configs]
        self.tolerance = tolerance
        self.metric = metric

    def __len__(self) -> int:
        return len(self.configs)

    def distance(self, q: np.ndarray) -> float:
        return min(self.metric(q, c) for c in self.configs)

    def violation(self, q: np.ndarray) -> float:
        return max(0.0, self.distance(q) - self.tolerance)

    def contains(self, q: np.ndarray) -> bool:
        return self.distance(q) <= self.tolerance

    def sample(self, rng: np.random.Generator) -> list[Sample]:
        i = int(rng.integers(len(self.configs)))
        return [Sample(self.configs[i].copy(), (i,))]


class EmptySet:
    """The empty set: finite with no members. Useful as a placeholder role."""

    def contains(self, q: np.ndarray) -> bool:
        return False

    def __repr__(self) -> str:
        return "EmptySet()"


class PredicateSet:
    """A set defined by an arbitrary membership predicate. Membership only."""

    def __init__(self, predicate: Callable[[np.ndarray], bool], name: str | None = None):
        self.predicate = predicate
        self.name = name

    def contains(self, q: np.ndarray) -> bool:
        return bool(self.predicate(q))

    def __repr__(self) -> str:
        return f"PredicateSet({self.name or self.predicate!r})"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _all_support(children: Sequence[StateSet], capability: type) -> bool:
    return all(supports(c, capability) for c in children)


class _Composite:
    """Shared machinery for ``AnyOf`` and ``AllOf``."""

    _kind: str = ""

    def __init__(self, children: Sequence[StateSet]):
        if len(children) == 0:
            raise ValueError(f"{self._kind} requires at least one child set")
        self.children = list(children)
        self.capabilities: frozenset[type] = frozenset()

    def _require(self, capability: type) -> None:
        if capability not in self.capabilities:
            raise UnsupportedCapability(
                f"{self._kind} with {len(self.children)} children does not support "
                f"{capability.__name__}: {self._why_unsupported(capability)}"
            )

    def _why_unsupported(self, capability: type) -> str:  # pragma: no cover - overridden
        return "capability not available"

    def __repr__(self) -> str:
        return f"{self._kind}({self.children!r})"


class AnyOf(_Composite):
    """Union: ``q`` is a member iff it is a member of at least one child.

    Capabilities:
        distance: min over children, if every child supports distance.
        violation: min over children, if every child supports violation
            (zero iff some child contains the point).
        sample: single child delegates. Multiple children require ``weights``
            (the mixture policy); the chosen child's index is prepended to
            each candidate's ``source``.
        project: single child delegates. Multiple children: project onto each
            child and return the successful result nearest ``q_proposed``
            under ``metric``. Requires every child to support projection.
    """

    _kind = "AnyOf"

    def __init__(
        self,
        children: Sequence[StateSet],
        weights: Sequence[float] | None = None,
        metric: Metric = euclidean,
    ):
        super().__init__(children)
        self.metric = metric
        self.weights: np.ndarray | None = None
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.shape != (len(self.children),):
                raise ValueError(f"weights length ({len(w)}) must match number of children ({len(self.children)})")
            if np.any(w < 0) or w.sum() <= 0:
                raise ValueError("weights must be nonnegative and sum to a positive value")
            self.weights = w / w.sum()

        caps = {StateSet}
        if _all_support(self.children, SetDistance):
            caps.add(SetDistance)
        if _all_support(self.children, SetViolation):
            caps.add(SetViolation)
        if len(self.children) == 1:
            if supports(self.children[0], SetSampler):
                caps.add(SetSampler)
            if supports(self.children[0], SetProjector):
                caps.add(SetProjector)
        else:
            if self.weights is not None and _all_support(self.children, SetSampler):
                caps.add(SetSampler)
            if _all_support(self.children, SetProjector):
                caps.add(SetProjector)
        self.capabilities = frozenset(caps)

    def _why_unsupported(self, capability: type) -> str:
        if capability is SetSampler and len(self.children) > 1 and self.weights is None:
            return "a multi-child union needs an explicit mixture policy (pass weights=)"
        return "every child must support it"

    def contains(self, q: np.ndarray) -> bool:
        return any(c.contains(q) for c in self.children)

    def distance(self, q: np.ndarray) -> float:
        self._require(SetDistance)
        return min(c.distance(q) for c in self.children)

    def violation(self, q: np.ndarray) -> float:
        self._require(SetViolation)
        return min(c.violation(q) for c in self.children)

    def sample(self, rng: np.random.Generator) -> list[Sample]:
        self._require(SetSampler)
        if len(self.children) == 1:
            i = 0
        else:
            i = int(rng.choice(len(self.children), p=self.weights))
        return [Sample(s.q, (i, *s.source), s.key) for s in self.children[i].sample(rng)]

    def project(self, q_previous: np.ndarray, q_proposed: np.ndarray) -> np.ndarray | None:
        self._require(SetProjector)
        if len(self.children) == 1:
            return self.children[0].project(q_previous, q_proposed)
        best = None
        best_dist = float("inf")
        for c in self.children:
            q = c.project(q_previous, q_proposed)
            if q is None:
                continue
            d = self.metric(q_proposed, q)
            if d < best_dist:
                best, best_dist = q, d
        return best


class AllOf(_Composite):
    """Intersection: ``q`` is a member iff it is a member of every child.

    Capabilities:
        distance: max over children, if every child supports distance. This
            is a geometric summary, not a membership test.
        violation: max over children, if every child supports violation
            (zero iff every child contains the point).
        sample: single child delegates. Multiple children require an explicit
            ``sampling`` strategy (see ``RejectionSampling``). Candidates keep
            the sampled child's provenance; ``AllOf`` adds no index.
        project: single child delegates. Multiple children require an
            explicit ``projection`` strategy (see ``MostViolatedProjection``);
            no generic intersection projector is implied.
    """

    _kind = "AllOf"

    def __init__(
        self,
        children: Sequence[StateSet],
        projection: IntersectionProjection | None = None,
        sampling: IntersectionSampling | None = None,
    ):
        super().__init__(children)
        self.projection = projection
        self.sampling = sampling

        caps = {StateSet}
        if _all_support(self.children, SetDistance):
            caps.add(SetDistance)
        if _all_support(self.children, SetViolation):
            caps.add(SetViolation)
        if len(self.children) == 1:
            if supports(self.children[0], SetSampler):
                caps.add(SetSampler)
            if supports(self.children[0], SetProjector):
                caps.add(SetProjector)
        else:
            if projection is not None:
                if hasattr(projection, "requires"):
                    projection.requires(self.children)
                caps.add(SetProjector)
            if sampling is not None:
                if hasattr(sampling, "requires"):
                    sampling.requires(self.children)
                caps.add(SetSampler)
        self.capabilities = frozenset(caps)

    def _why_unsupported(self, capability: type) -> str:
        if capability is SetProjector:
            return "a multi-child intersection needs an explicit projection strategy (pass projection=)"
        if capability is SetSampler:
            return "a multi-child intersection needs an explicit sampling strategy (pass sampling=)"
        return "every child must support it"

    def contains(self, q: np.ndarray) -> bool:
        return all(c.contains(q) for c in self.children)

    def distance(self, q: np.ndarray) -> float:
        self._require(SetDistance)
        return max(c.distance(q) for c in self.children)

    def violation(self, q: np.ndarray) -> float:
        self._require(SetViolation)
        return max(c.violation(q) for c in self.children)

    def sample(self, rng: np.random.Generator) -> list[Sample]:
        self._require(SetSampler)
        if len(self.children) == 1:
            return self.children[0].sample(rng)
        assert self.sampling is not None
        return self.sampling.sample(self.children, rng)

    def project(self, q_previous: np.ndarray, q_proposed: np.ndarray) -> np.ndarray | None:
        self._require(SetProjector)
        if len(self.children) == 1:
            return self.children[0].project(q_previous, q_proposed)
        assert self.projection is not None
        return self.projection.project(self.children, q_previous, q_proposed)


# ---------------------------------------------------------------------------
# Enumeration of finite sets
# ---------------------------------------------------------------------------


def is_finite(s: StateSet) -> bool:
    """Whether every member of ``s`` can be enumerated by ``members``.

    A ``FiniteSet`` is finite. A union is finite when all children are.
    An intersection is finite when any child is, since intersecting with
    a finite set yields a finite set.
    """
    if isinstance(s, (FiniteSet, EmptySet)):
        return True
    if isinstance(s, AnyOf):
        return all(is_finite(c) for c in s.children)
    if isinstance(s, AllOf):
        return any(is_finite(c) for c in s.children)
    return False


def members(s: StateSet) -> list[Sample]:
    """Enumerate every member of a finite set, with provenance.

    Exhaustive exactly when ``is_finite(s)``; an empty list otherwise. A
    union enumerates every child (all finite). An intersection enumerates a
    child that is itself finite, whichever comes first, and keeps the
    members every other child contains; the resulting configurations do not
    depend on child order, only their provenance does. See ``explicit_samples`` for the
    explicit candidates embedded in a set that is not finite.
    """
    if isinstance(s, FiniteSet):
        return [Sample(c.copy(), (i,)) for i, c in enumerate(s.configs)]
    if isinstance(s, AnyOf):
        if not is_finite(s):
            return []
        out = []
        for i, c in enumerate(s.children):
            out.extend(Sample(m.q, (i, *m.source)) for m in members(c))
        return out
    if isinstance(s, AllOf):
        for i, c in enumerate(s.children):
            if is_finite(c):
                others = s.children[:i] + s.children[i + 1 :]
                return [m for m in members(c) if all(o.contains(m.q) for o in others)]
        return []
    return []


def explicit_samples(s: StateSet) -> list[Sample]:
    """The explicit configurations embedded in a set expression, with provenance.

    Unlike ``members``, this does not require the set to be finite. A union
    of a finite set and a sampleable region still contains the finite set's
    configurations, and they should seed a search regardless of mixture
    weights, which govern random sampling only.

    - ``FiniteSet``: every member.
    - ``AnyOf``: the seeds of every child, with the child index prepended.
    - ``AllOf``: if the intersection is finite, its exhaustive ``members``.
      Otherwise the seeds of every seed-bearing child that the complete
      intersection contains, in child order, with equal configurations
      deduplicated by keeping the first occurrence (``AllOf`` adds no
      provenance, so the kept provenance is the earliest child's).
    - Other leaves: none.
    """
    if isinstance(s, FiniteSet):
        return [Sample(c.copy(), (i,)) for i, c in enumerate(s.configs)]
    if isinstance(s, AnyOf):
        out = []
        for i, c in enumerate(s.children):
            out.extend(Sample(m.q, (i, *m.source)) for m in explicit_samples(c))
        return out
    if isinstance(s, AllOf):
        if is_finite(s):
            return members(s)
        out: list[Sample] = []
        for c in s.children:
            for m in explicit_samples(c):
                if s.contains(m.q) and not any(np.array_equal(m.q, kept.q) for kept in out):
                    out.append(m)
        return out
    return []


# ---------------------------------------------------------------------------
# Named intersection strategies
# ---------------------------------------------------------------------------


class RejectionSampling:
    """Sample one child of an intersection and keep what the others contain.

    ``source`` is the index of the child to draw from; it must support
    sampling. Every candidate of a draw is tested against the remaining
    children by membership. This is exact but wasteful when the
    intersection is a small part of the source: expect many empty draws.
    """

    def __init__(self, source: int = 0):
        self.source = source

    def requires(self, children: Sequence[StateSet]) -> None:
        if not 0 <= self.source < len(children):
            raise ValueError(f"RejectionSampling source index {self.source} out of range for {len(children)} children")
        if not supports(children[self.source], SetSampler):
            raise UnsupportedCapability(
                f"RejectionSampling requires child {self.source} ({children[self.source]!r}) to support sampling"
            )

    def sample(self, children: Sequence[StateSet], rng: np.random.Generator) -> list[Sample]:
        others = [c for i, c in enumerate(children) if i != self.source]
        return [s for s in children[self.source].sample(rng) if all(o.contains(s.q) for o in others)]


class MostViolatedProjection:
    """Heuristic: repeatedly project onto the unsatisfied child with the largest violation.

    Each iteration evaluates every child's ``violation`` (zero for children
    that already contain the point), stops if every child contains the
    configuration, and otherwise projects onto the child with the largest
    violation. A satisfied child is never selected.

    Termination: success when every child is satisfied; failure when the
    selected projection returns None or leaves the configuration unchanged
    while its child is still violated; failure after a full sweep
    (``len(children)`` consecutive iterations) without progress, where
    progress means the violation profile, the violations sorted in
    descending order, decreased lexicographically by at least
    ``progress_tolerance`` in some position. Satisfying one of several
    equally violated children therefore counts as progress even though the
    maximum is unchanged, so tied plateaus are traversed rather than
    mistaken for stagnation. ``max_iters`` bounds everything.

    This is a heuristic for intersections, not a true projection. Every child
    must support both violation and projection, and the children's violations
    must be on a comparable scale (see ``SetViolation``); for an intersection
    of TSR-induced sets in the same units this holds. For heterogeneous
    children use a different strategy or normalize the violations yourself.
    """

    def __init__(self, max_iters: int = 50, progress_tolerance: float = 1e-6):
        self.max_iters = max_iters
        self.progress_tolerance = progress_tolerance

    def requires(self, children: Sequence[StateSet]) -> None:
        for i, c in enumerate(children):
            if not (supports(c, SetViolation) and supports(c, SetProjector)):
                raise UnsupportedCapability(
                    f"MostViolatedProjection requires every child to support violation and projection; "
                    f"child {i} ({c!r}) does not"
                )

    def project(
        self,
        children: Sequence[StateSet],
        q_previous: np.ndarray,
        q_proposed: np.ndarray,
    ) -> np.ndarray | None:
        q = np.array(q_proposed, dtype=float)
        best_profile: list[float] | None = None
        stale = 0
        for _ in range(self.max_iters):
            # Zero for satisfied children by contract, so argmax never picks one unless all are zero
            violations = [0.0 if c.contains(q) else c.violation(q) for c in children]
            worst_idx = int(np.argmax(violations))
            if violations[worst_idx] <= 0.0:
                return q

            profile = sorted(violations, reverse=True)
            if best_profile is None or self._improved(profile, best_profile):
                best_profile = profile
                stale = 0
            else:
                stale += 1
                if stale >= len(children):
                    return None  # a full sweep without progress: cycling or stuck

            q_next = children[worst_idx].project(q_previous, q)
            if q_next is None:
                return None
            q_next = np.asarray(q_next, dtype=float)
            if np.array_equal(q_next, q):
                return None  # the projector cannot move this point while its child is violated
            q = q_next
        return None

    def _improved(self, profile: list[float], best: list[float]) -> bool:
        """Lexicographic decrease of the descending violation profile, at ``progress_tolerance``."""
        for cur, prev in zip(profile, best):
            if prev - cur >= self.progress_tolerance:
                return True
            if cur - prev >= self.progress_tolerance:
                return False
        return False


def seeds(s: StateSet) -> list[Sample]:
    """Deprecated name of :func:`explicit_samples` (``seed`` elsewhere means an RNG seed)."""
    import warnings

    warnings.warn("sscbirrt.seeds is deprecated; use explicit_samples", DeprecationWarning, stacklevel=2)
    return explicit_samples(s)
