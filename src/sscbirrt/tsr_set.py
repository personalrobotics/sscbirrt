# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The configuration-space set induced by a Task Space Region or a TSR chain.

A TSR is a set of end-effector poses; a TSR chain is one too, built by
serially composing a pose from each link of the chain (a door handle whose
door swings on a hinge). Either kind of pose region induces a set of
configurations through forward kinematics, ``{q : FK(q) in region}``. This
module adapts such a region to the state-set protocols in ``sscbirrt.sets``
so the planner can treat it like any other set. It is the only planner
module that imports from the ``tsr`` package besides the legacy lowering.

A chain is **one** set. It is not the intersection of its component TSRs'
world-frame sets: the components constrain successive frames, and only their
serial composition describes the handle's reachable poses (see
``docs/design.md``). Independent TSRs on different robot links are a different
object and compose as ``AllOf``; that is tracked separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from tsr import TSR, TSRChain
from tsr.sampling import _interval_sum, weights_from_tsrs

from sscbirrt.interfaces import IKSolver, RobotModel
from sscbirrt.sets import Sample
from sscbirrt.space import JointSpace


@runtime_checkable
class PoseRegion(Protocol):
    """What the adapter needs from a region of end-effector poses.

    ``tsr.TSR`` and ``tsr.TSRChain`` both satisfy it. ``distance`` returns the
    distance of a world-frame transform to the region (plus region-specific
    coordinates the adapter ignores); ``closest_transform`` returns that
    distance with the closest in-region world-frame transform; ``sample``
    draws a world-frame transform with the given generator.
    """

    def distance(self, trans: np.ndarray) -> tuple[float, Any]: ...
    def closest_transform(self, trans: np.ndarray) -> tuple[float, np.ndarray]: ...
    def sample(self, *args: Any, rng: np.random.Generator | None = None) -> np.ndarray: ...


def full_turn_key(q: np.ndarray, revolute: Sequence[bool]) -> tuple[int, ...]:
    """Equal for configurations that differ only by full turns of the joints the IK declares revolute (#200).

    Each revolute joint is reduced to [-pi, pi]; every joint is rounded to nanoradians. Two windings that round
    apart only cost an extra validator call; two different configurations never share a key.
    """
    w = np.where(np.asarray(revolute, dtype=bool), np.remainder(q + np.pi, 2 * np.pi) - np.pi, q)
    return tuple(int(v) for v in np.round(w * 1e9))


class TSRConfigurationSet:
    """Configurations whose end-effector pose lies in a TSR or a TSR chain.

    Capabilities:
        contains: region distance of ``FK(q)`` is within ``tolerance``.
        distance: the region distance of ``FK(q)``. For a TSR this is the
            closed form of Berenson et al. 2011, Sec. 4.2. For a multi-TSR
            chain it is the best residual a bounded numerical solve found, an
            upper bound on the true distance, so ``contains`` can be a false
            negative on a hard chain; sstsr documents this (its #85).
        violation: ``max(0, distance - tolerance)``; comparable across regions
            that use the same units and rotation weighting.
        sample: draw a pose from the region, solve IK, and return every
            solution within joint limits as candidates. The caller validates
            them; returning all branches matters because which branch is
            collision-free is not knowable here.
        project: iteratively move the pose to the closest point of the region
            and solve IK, choosing the solution nearest the current
            configuration under the joint-space metric. Gives up when the
            violation stops decreasing by ``progress_tolerance`` or after
            ``max_projection_iters``. ``q_previous`` is currently unused.

    Cost: for a single TSR every operation is closed form. For a chain of two
    or more TSRs, ``distance``, ``contains``, and each projection step run a
    bounded multi-start solve (a few milliseconds each). Sampling is cheap for
    both. A chain as a goal costs little; a chain as a path constraint pays
    the solve on every edge sample.

    Collision is not part of this set. Samples and projections are filtered
    by joint limits only; the planner applies its validator.
    """

    def __init__(
        self,
        tsr: PoseRegion,
        robot: RobotModel,
        ik: IKSolver,
        space: JointSpace,
        tolerance: float = 1e-3,
        max_projection_iters: int = 50,
        progress_tolerance: float = 1e-6,
    ):
        if not isinstance(tsr, PoseRegion):
            raise TypeError(
                f"TSRConfigurationSet needs a pose region with distance, closest_transform, and sample "
                f"(a tsr.TSR or tsr.TSRChain), got {type(tsr).__name__}"
            )
        self.tsr = tsr
        self.robot = robot
        self.ik = ik
        self.space = space
        self.tolerance = tolerance
        self.max_projection_iters = max_projection_iters
        self.progress_tolerance = progress_tolerance

    def __repr__(self) -> str:
        return f"TSRConfigurationSet({self.tsr!r}, tolerance={self.tolerance})"

    # -- membership and distance ------------------------------------------

    def distance(self, q: np.ndarray) -> float:
        dist, _ = self.tsr.distance(self.robot.forward_kinematics(q))
        return float(dist)

    def violation(self, q: np.ndarray) -> float:
        """TSR distance beyond the membership tolerance; zero iff ``contains``."""
        return max(0.0, self.distance(q) - self.tolerance)

    def contains(self, q: np.ndarray) -> bool:
        return self.distance(q) <= self.tolerance

    # -- sampling ------------------------------------------------------------

    def sample_pose(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a world-frame end-effector pose uniformly from the TSR bounds with ``rng``."""
        return self.tsr.sample(rng=rng)

    def sample(self, rng: np.random.Generator) -> list[Sample]:
        pose = self.sample_pose(rng)
        revolute = getattr(self.ik, "revolute_joints", None)
        out = []
        for q in self.ik.solve(pose):
            if self.space.within_limits(q):
                q = np.array(q, dtype=float)
                out.append(Sample(q, (), None if revolute is None else full_turn_key(q, revolute)))
        return out

    # -- projection ----------------------------------------------------------

    def project(self, q_previous: np.ndarray, q_proposed: np.ndarray) -> np.ndarray | None:
        q = np.array(q_proposed, dtype=float)
        prev_dist = float("inf")
        for _ in range(self.max_projection_iters):
            # Closest in-bounds pose, already composed with T0_w and Tw_e (#28)
            dist, target = self.tsr.closest_transform(self.robot.forward_kinematics(q))
            if dist <= self.tolerance:
                return q
            if prev_dist - dist < self.progress_tolerance:
                return None
            prev_dist = dist

            best, best_d = None, float("inf")
            for sol in self.ik.solve(target, q_init=q):
                if not self.space.within_limits(sol):
                    continue
                d = self.space.distance(q, sol)
                if d < best_d:
                    best, best_d = sol, d
            if best is None:
                return None
            q = np.array(best, dtype=float)
        return None


def region_volume(region: PoseRegion) -> float:
    """The legacy volume measure of a region: the sum of its bound widths.

    Rotational widths are clamped to one turn. A chain's volume is the sum
    over its component TSRs.
    """
    if isinstance(region, TSRChain):
        return float(sum(_interval_sum(t.Bw) for t in region.TSRs))
    return float(_interval_sum(region.Bw))


def tsr_weights(sets: Sequence[TSRConfigurationSet]) -> np.ndarray:
    """Mixture weights for an ``AnyOf`` of TSR or chain sets, proportional to volume.

    Matches the legacy planner's policy: the sum of bound widths, falling
    back to uniform when every region has zero volume.
    """
    regions = [s.tsr for s in sets]
    if all(isinstance(r, TSR) for r in regions):
        return weights_from_tsrs(regions)
    w = np.array([region_volume(r) for r in regions], dtype=float)
    if not np.any(w > 0.0):
        w = np.ones_like(w)
    return w
