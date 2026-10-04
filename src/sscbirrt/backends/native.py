# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The native backend: lower a Python ``PlanningProblem`` to ``sscbirrt._native`` and solve there.

Every component must have a native form, because the native solve releases the
GIL and calls no Python after entry. ``lower`` walks the problem and raises
``NativeUnsupported`` listing every component that blocked it; ``CBiRRT`` with
``backend="auto"`` then selects the Python backend and records the reasons.
See docs/native-design.md, "Python binding".
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sscbirrt.config import CBiRRTConfig
from sscbirrt.exceptions import (
    AllGoalConfigurationsInCollision,
    AllGoalConfigurationsInvalid,
    AllStartConfigurationsInCollision,
    AllStartConfigurationsInvalid,
    MotionContractError,
    NativeUnsupported,
    UnsupportedCapability,
)
from sscbirrt.legacy import legacy_index
from sscbirrt.problem import PlanningProblem
from sscbirrt.sets import AllOf, AnyOf, EmptySet, FiniteSet, MostViolatedProjection, RejectionSampling, euclidean
from sscbirrt.space import JointSpace
from sscbirrt.testing import NoCollision, Wall
from sscbirrt.tree import RRTree
from sscbirrt.tsr_set import TSRConfigurationSet

FK_AGREEMENT_ATOL = 1e-6  # the robot model's FK must match the SSIK adapter's on the problem's explicit configurations

try:
    from sscbirrt import _native
except ImportError:  # pragma: no cover - exercised only in builds without the extension
    _native = None


@runtime_checkable
class ValidatorIntegration(Protocol):
    """A Python ``CollisionChecker`` that also has a native form.

    ``lower`` recognizes validators by this protocol, never by type, so any collision backend with a
    ``sscbirrt::StateValidator`` in its own extension module plugs in without a change here. The MuJoCo
    scene's ``NativeCollisionChecker`` is one implementation.
    """

    def fresh(self) -> Any:
        """A native ``StateValidator`` for one solve, with any per-solve state (scratch data) of its own."""

    @property
    def provenance(self) -> dict[str, Any]:
        """What the validator checked against, for ``PlanResult.provenance``; may be empty."""


@runtime_checkable
class KinematicsIntegration(Protocol):
    """A Python ``IKSolver`` that also has a native form.

    ``native_kinematics`` returns one object that is both a native ``ForwardKinematics`` and a native
    ``IKSolver`` for the same arm, or raises ``NativeUnsupported`` naming why this instance has none. The
    SSIK adapter's ``SSIKSolver`` is one implementation; the lowering itself knows no IK library.
    """

    def native_kinematics(self) -> Any: ...

    @property
    def provenance(self) -> dict[str, Any]:
        """Which kinematic model answered, for ``PlanResult.provenance``; may be empty."""


def available() -> bool:
    return _native is not None


@dataclass
class Lowered:
    problem: Any  # _native.PlanningProblem
    config: Any  # _native.PlannerConfig
    space: Any  # _native.JointSpace
    provenance: dict = field(default_factory=dict)  # merged from every integration's ``provenance``


def _space_metric(metric, space: JointSpace) -> bool:
    """Whether ``metric`` is ``space.distance`` bound to this very space."""
    return getattr(metric, "__self__", None) is space and getattr(metric, "__func__", None) is JointSpace.distance


def _lower_space(space: JointSpace):
    angular = [] if space.continuous_joints is None else [bool(a) for a in space.continuous_joints]
    return _native.JointSpace(list(space.lower), list(space.upper), angular)


class _Lowering:
    """Per-lowering state: one native kinematics object per Python IK instance, checked once."""

    def __init__(self, problem: PlanningProblem, nspace):
        self.problem = problem
        self.nspace = nspace
        self.arms: dict[int, Any] = {}
        self.arm_reasons: dict[int, str] = {}
        self.provenance: dict[str, Any] = {}

    def arm_for(self, tsr_set: TSRConfigurationSet, where: str, reasons: list[str]):
        ik = tsr_set.ik
        key = id(ik)
        if key in self.arms:
            return self.arms[key]
        if key in self.arm_reasons:
            return None
        why = self._arm_reason(tsr_set, where)
        if why is not None:
            self.arm_reasons[key] = why
            reasons.append(why)
            return None
        return self.arms[key]

    def _arm_reason(self, tsr_set: TSRConfigurationSet, where: str) -> str | None:
        ik = tsr_set.ik
        if not isinstance(ik, KinematicsIntegration):
            return (
                f"{where}: IK {type(ik).__name__} is a Python object; native needs an IK solver with a native form "
                "(sscbirrt.backends.native.KinematicsIntegration)"
            )
        try:
            arm = ik.native_kinematics()
        except NativeUnsupported as e:
            return f"{where}: " + "; ".join(e.reasons)
        if not (isinstance(arm, _native.ForwardKinematics) and isinstance(arm, _native.IKSolver)):
            return (
                f"{where}: {type(ik).__name__}.native_kinematics() returned {type(arm).__name__}, "
                "not a sscbirrt ForwardKinematics and IKSolver"
            )
        # The native set uses the integration's FK where the Python set used the robot model's: they must agree.
        from sscbirrt.sets import explicit_samples

        for smp in explicit_samples(self.problem.start) + explicit_samples(self.problem.goal):
            q = np.asarray(smp.q, dtype=float)
            if q.shape != (arm.dof,):
                continue
            expected = tsr_set.robot.forward_kinematics(q)
            got = np.array(arm.fk(list(map(float, q))))
            err = float(np.abs(expected - got).max())
            if err > FK_AGREEMENT_ATOL:
                return (
                    f"{where}: robot model FK disagrees with the native IK model's by {err:.3g} at an explicit "
                    f"configuration (tolerance {FK_AGREEMENT_ATOL:g}); pass T_base/T_ee so they agree"
                )
        self.arms[id(ik)] = arm
        self.provenance.update(dict(ik.provenance))
        return None


def _lower_tsr_set(s: TSRConfigurationSet, where: str, ctx: _Lowering, reasons: list[str]):
    from tsr import TSR, TSRChain

    if not isinstance(s.tsr, (TSR, TSRChain)):
        reasons.append(f"{where}: {type(s.tsr).__name__} has no native form")
        return None
    arm = ctx.arm_for(s, where, reasons)
    if arm is None:
        return None
    try:
        # TSRs and TSR chains are sstsr's own C++ (sstsr >= 3.3, #184): the same rules as the Python sstsr; a chain's
        # cold inverse is held to sstsr's properties (an upper bound; "not found" is not a certificate).
        if isinstance(s.tsr, TSRChain):
            links = [_native.TSR(t.T0_w.tolist(), t.Tw_e.tolist(), t.Bw.tolist()) for t in s.tsr.TSRs]
            region = _native.TSRChain(links)
        else:
            region = _native.TSR(s.tsr.T0_w.tolist(), s.tsr.Tw_e.tolist(), s.tsr.Bw.tolist())
    except ValueError as e:
        reasons.append(f"{where}: {type(s.tsr).__name__} rejected by the native constructor: {e}")
        return None
    return _native.TSRConfigurationSet(
        region, arm, arm, ctx.nspace, float(s.tolerance), int(s.max_projection_iters), float(s.progress_tolerance)
    )


def _lower_set(s, where: str, space: JointSpace, nspace, reasons: list[str], ctx: "_Lowering | None" = None):
    if isinstance(s, TSRConfigurationSet):
        if ctx is None:
            reasons.append(f"{where}: TSRConfigurationSet needs a lowering context")
            return None
        return _lower_tsr_set(s, where, ctx, reasons)
    if isinstance(s, FiniteSet):
        configs = [list(map(float, c)) for c in s.configs]
        if s.metric is euclidean:
            return _native.FiniteSet(configs, float(s.tolerance))
        if _space_metric(s.metric, space):
            return _native.FiniteSet(configs, float(s.tolerance), nspace)
        reasons.append(f"{where}: FiniteSet metric is a Python callable; native supports euclidean or the space metric")
        return None
    if isinstance(s, EmptySet):
        return _native.EmptySet()
    if isinstance(s, AnyOf):
        children = [_lower_set(c, f"{where}[{i}]", space, nspace, reasons, ctx) for i, c in enumerate(s.children)]
        if any(c is None for c in children):
            return None
        weights = None if s.weights is None else [float(w) for w in s.weights]
        if s.metric is euclidean:
            return _native.AnyOf(children, weights)
        if _space_metric(s.metric, space):
            return _native.AnyOf(children, weights, nspace)
        reasons.append(f"{where}: AnyOf metric is a Python callable")
        return None
    if isinstance(s, AllOf):
        children = [_lower_set(c, f"{where}[{i}]", space, nspace, reasons, ctx) for i, c in enumerate(s.children)]
        if any(c is None for c in children):
            return None
        projection = sampling = None
        if s.projection is not None:
            if isinstance(s.projection, MostViolatedProjection):
                projection = _native.MostViolatedProjection(
                    int(s.projection.max_iters), float(s.projection.progress_tolerance)
                )
            else:
                reasons.append(f"{where}: projection strategy {type(s.projection).__name__} has no native form")
                return None
        if s.sampling is not None:
            if isinstance(s.sampling, RejectionSampling):
                sampling = _native.RejectionSampling(int(s.sampling.source))
            else:
                reasons.append(f"{where}: sampling strategy {type(s.sampling).__name__} has no native form")
                return None
        return _native.AllOf(children, projection, sampling)
    reasons.append(f"{where}: {type(s).__name__} has no native form")
    return None


def _lower_validator(v, dof: int, reasons: list[str]):
    if isinstance(v, ValidatorIntegration):
        native = v.fresh()  # its own per-solve state (docs/native-design.md, v1.7.0: one validator per solve)
        if isinstance(native, _native.StateValidator):
            return native
        reasons.append(
            f"validator: {type(v).__name__}.fresh() returned {type(native).__name__}, not a sscbirrt StateValidator"
        )
        return None
    if isinstance(v, NoCollision):
        return _native.AcceptAll()
    if isinstance(v, Wall):
        lo, hi = [-np.inf] * dof, [np.inf] * dof
        lo[v.axis], hi[v.axis] = float(v.lo), float(v.hi)
        if v.extent is not None:
            other = 1 - v.axis
            lo[other], hi[other] = -float(v.extent), float(v.extent)
        return _native.JointBoxObstacles([(lo, hi)])
    reasons.append(
        f"validator: {type(v).__name__} is a Python object; native needs a validator with a native form "
        "(sscbirrt.backends.native.ValidatorIntegration)"
    )
    return None


def _lower_config(config: CBiRRTConfig):
    c = _native.PlannerConfig()
    c.timeout_seconds = float(config.timeout)
    c.max_iterations = int(config.max_iterations)
    c.connection_tolerance = float(config.connection_tolerance)
    c.edge_resolution = None if config.edge_resolution is None else float(config.edge_resolution)
    c.progress_tolerance = float(config.progress_tolerance)
    c.step_size = float(config.step_size)
    c.start_sample_probability = float(config.start_sample_probability)
    c.goal_sample_probability = float(config.goal_sample_probability)
    c.extend_steps = config.extend_steps
    c.connect_steps = config.connect_steps
    c.sample_draws = int(config.sample_draws)
    c.num_tree_roots = int(config.num_tree_roots)
    c.max_per_draw = int(config.max_per_draw)
    c.smooth_path = bool(config.smooth_path)
    c.smoothing_iterations = int(config.smoothing_iterations)
    c.smoothing_patience = int(config.smoothing_patience)
    return c


def lower(problem: PlanningProblem, config: CBiRRTConfig) -> Lowered:
    """Map a Python problem and config to their native forms, or raise ``NativeUnsupported`` with every blocker."""
    if _native is None:
        raise NativeUnsupported(["sscbirrt._native is not built; install sscbirrt from a wheel with the extension"])
    reasons: list[str] = []
    nspace = _lower_space(problem.space)
    ctx = _Lowering(problem, nspace)
    start = _lower_set(problem.start, "start", problem.space, nspace, reasons, ctx)
    goal = _lower_set(problem.goal, "goal", problem.space, nspace, reasons, ctx)
    constraint = None
    if problem.path_constraint is not None:
        constraint = _lower_set(problem.path_constraint, "path_constraint", problem.space, nspace, reasons, ctx)
    validator = _lower_validator(problem.validator, problem.space.dof, reasons)
    if problem.motion_validator is not None:
        reasons.append(f"motion_validator: {type(problem.motion_validator).__name__} is a Python object")
    if problem.sampler is not None and problem.sampler is not problem.space:
        reasons.append(f"sampler: {type(problem.sampler).__name__} is a Python object")
    if reasons:
        raise NativeUnsupported(reasons)

    np_problem = _native.PlanningProblem()
    np_problem.space = nspace
    np_problem.start = start
    np_problem.goal = goal
    np_problem.validator = validator
    np_problem.path_constraint = constraint
    provenance: dict = dict(ctx.provenance)
    if isinstance(problem.validator, ValidatorIntegration):
        provenance.update(dict(problem.validator.provenance))
    return Lowered(np_problem, _lower_config(config), nspace, provenance)


def _rebuild_tree(tree) -> RRTree | None:
    if tree is None:
        return None
    nodes = tree.nodes()
    roots = [(np.array(q, dtype=float), tuple(src)) for q, parent, src in nodes if parent < 0]
    out = RRTree([q for q, _ in roots], source_indices=[s for _, s in roots])
    for q, parent, _ in nodes:
        if parent >= 0:
            out.add_node(np.array(q, dtype=float), parent)
    return out


def _raise_no_roots(e) -> None:
    report = e.report
    only = report.only_collisions()
    cls = {
        ("start", True): AllStartConfigurationsInCollision,
        ("start", False): AllStartConfigurationsInvalid,
        ("goal", True): AllGoalConfigurationsInCollision,
        ("goal", False): AllGoalConfigurationsInvalid,
    }[(e.role, only)]
    details = list(report.details)
    if report.draws:
        details.append("sampling: " + report.summary())
    raise cls(report.candidates(), details) from None


def solve(lowered: Lowered, seed: int | None, abort_fn: Callable[[], bool] | None):
    """Run the native solve and return a Python ``PlanResult``. Exceptions map to their Python namesakes."""
    token = _native.CancellationToken()
    stop = threading.Event()
    poller = None
    if abort_fn is not None:
        if abort_fn():  # the cancellation case in the artifact: aborted before any iteration
            token.cancel()
        else:

            def poll():
                while not stop.is_set():
                    if abort_fn():
                        token.cancel()
                        return
                    time.sleep(0.001)

            poller = threading.Thread(target=poll, daemon=True)
            poller.start()

    planner = _native.Planner(lowered.config)
    try:
        r = planner.solve(lowered.problem, None if seed is None else int(seed), token, True)
    except _native.NoRoots as e:
        _raise_no_roots(e)
    except _native.UnsupportedCapability as e:
        raise UnsupportedCapability(str(e)) from None
    except _native.ContractError as e:
        raise MotionContractError(str(e)) from None
    finally:
        stop.set()
        if poller is not None:
            poller.join()
    return convert(r, lowered.provenance)


def convert(r, provenance: dict | None = None):
    """A native PlanResult as the Python PlanResult."""
    from sscbirrt.planner import PlanResult, versions

    s = r.stats
    stats = {
        "state_checks": s.state_checks,
        "edge_checks": s.edge_checks,
        "set_samples": s.set_samples,
        "set_projections": s.set_projections,
        "search_roots": s.search_roots,
        "seconds_state_checks": s.seconds_state_checks,
        "seconds_edge_checks": s.seconds_edge_checks,
        "seconds_set_samples": s.seconds_set_samples,
        "seconds_set_projections": s.seconds_set_projections,
        "seconds_roots": s.seconds_roots,
        "seconds_search": s.seconds_search,
        "seconds_smoothing": s.seconds_smoothing,
    }
    success = r.success
    start_source, goal_source = tuple(r.start_source), tuple(r.goal_source)
    return PlanResult(
        path=[np.array(q, dtype=float) for q in r.path] if success else None,
        start_index=legacy_index(start_source),
        goal_index=legacy_index(goal_source),
        iterations=r.iterations,
        planning_time=r.planning_seconds,
        tree_sizes=tuple(r.tree_sizes),
        success=success,
        failure_reason=None if success else r.reason,
        start_source=start_source,
        goal_source=goal_source,
        tree_start=_rebuild_tree(r.tree_start),
        tree_goal=_rebuild_tree(r.tree_goal),
        backend="native",
        provenance={**versions(), "backend": "native", **(provenance or {})},
        stats=stats,
    )
