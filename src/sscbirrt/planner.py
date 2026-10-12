# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import functools
import itertools
import logging
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from sscbirrt.config import CBiRRTConfig
from sscbirrt.exceptions import (
    AllGoalConfigurationsInCollision,
    AllGoalConfigurationsInvalid,
    AllStartConfigurationsInCollision,
    AllStartConfigurationsInvalid,
    MotionContractError,
    UnsupportedCapability,
)
from sscbirrt.interfaces import CollisionChecker, IKSolver, RobotModel
from sscbirrt.legacy import legacy_index, legacy_problem
from sscbirrt.motion import DiscreteMotionValidator, MotionValidator
from sscbirrt.problem import PlanningProblem
from sscbirrt.sets import Sample, SetProjector, SetSampler, StateSet, explicit_samples, is_finite, supports
from sscbirrt.space import JointSpace
from sscbirrt.trajectory import Trajectory, TrajectoryOptions, coalesce, initial_trajectory, limits, shortcut
from sscbirrt.tree import RRTree

logger = logging.getLogger(__name__)

# Endpoint windings (#212): unwind a full turn over the first quarter, half, or all of the path's length, in that order.
UNWIND_BY = (0.25, 0.5, 1.0)


@functools.lru_cache(maxsize=1)
def _versions_cached() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in ("sscbirrt", "sstsr", "ssik", "mujoco"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


def versions() -> dict[str, str]:
    """Installed versions of sscbirrt and its geometric dependencies, for result provenance (cached; a copy)."""
    return dict(_versions_cached())


@dataclass
class PlanResult:
    """Result of a planning query.

    Attributes:
        path: Joint configurations from start to goal, or None if failed.
        start_index: Legacy index into the start input (config list or TSR
            list) that the path begins from. 0 for a single start.
        goal_index: Legacy index into the goal input that the path ends at.
        iterations: Number of RRT iterations used.
        planning_time: Wall-clock time in seconds.
        tree_sizes: Tuple of (start_tree_size, goal_tree_size).
        success: Whether a path was found.
        failure_reason: Human-readable reason for failure, or None if success.
        start_source: Provenance of the start root (see ``Sample.source``).
        goal_source: Provenance of the goal root.
        backend: Which implementation produced this result, "python" or "native".
        backend_reasons: With the default backend="auto", why the native backend
            was not used (one entry per blocking component); empty otherwise.
        tree_start: The search tree rooted at the start set, for inspection
            and visualization. Shares memory with the planner's run; do not
            mutate.
        tree_goal: The search tree rooted at the goal set.
        trajectory: With ``trajectory=TrajectoryOptions(...)``, the timed trajectory along the path
            (``sscbirrt.trajectory.Trajectory``, #214); ``path`` is unchanged. None otherwise, or if planning failed.
    """

    path: list[np.ndarray] | None
    start_index: int
    goal_index: int
    iterations: int
    planning_time: float
    tree_sizes: tuple[int, int]
    success: bool
    failure_reason: str | None = None
    start_source: tuple[int, ...] = field(default=())
    goal_source: tuple[int, ...] = field(default=())
    tree_start: RRTree | None = field(default=None, repr=False)
    tree_goal: RRTree | None = field(default=None, repr=False)
    backend: str = "python"
    backend_reasons: tuple[str, ...] = field(default=())
    provenance: dict = field(
        default_factory=dict, repr=False
    )  # versions; scene, MJB and snapshot hashes when native MuJoCo
    stats: dict = field(default_factory=dict, repr=False)  # cost breakdown: counts and seconds per component
    trajectory: Trajectory | None = field(default=None, repr=False)


def as_configurations(value, dof: int, name: str) -> list[np.ndarray]:
    """Read ``value`` as a list of configurations: one configuration (length ``dof``) or several (``n x dof``).

    A flat list of numbers is one configuration, not a list of scalars (#171). Anything else raises a
    ``ValueError`` that names the argument and the shape it expected.
    """
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}: expected one configuration of length {dof} or a list of them, got a ragged or non-numeric value"
        ) from None
    if arr.ndim == 1 and arr.shape[0] == dof:
        return [arr]
    if arr.ndim == 2 and arr.shape[1] == dof:
        return list(arr)
    if arr.size == 0:
        return []
    raise ValueError(f"{name}: expected one configuration of length {dof} or a list of them, got shape {arr.shape}")


class _AbortedDuringRoots(Exception):
    """Raised inside root collection when ``abort_fn`` fires; carries the roots gathered so far."""

    def __init__(self, role: str, roots: list[Sample]):
        super().__init__(role)
        self.role = role
        self.roots = roots


class CBiRRT:
    """Constrained Bi-directional RRT planner.

    Plans between a start set and a goal set through a path-admissible set,
    as described by a ``PlanningProblem``. The legacy ``plan(...)`` entry
    point accepts fixed configurations and TSRs and lowers them into a
    problem; ``solve(problem)`` is the general interface.
    """

    def __init__(
        self,
        robot: RobotModel,
        ik: IKSolver | None = None,
        collision_checker: CollisionChecker | None = None,
        config: CBiRRTConfig | None = None,
        backend: str = "auto",
        *,
        ik_solver: IKSolver | None = None,
    ):
        """Initialize the CBiRRT planner.

        Args:
            robot: Robot model providing FK and joint limits
            ik: Inverse kinematics solver; needed only for TSR (pose-region) starts, goals and constraints.
                ``ik_solver=`` is its deprecated name.
            collision_checker: Collision checking interface (required)
            config: Planner configuration (uses defaults if None)
            backend: "auto" (the default since 2.0): the native core when every
                component of the problem has a native form, else the Python
                reference with the reasons on ``PlanResult.backend_reasons`` and
                in the log. "native": the native core, or ``NativeUnsupported``
                naming every blocker; never a silent fallback. "python": the
                reference implementation, always.
        """
        if ik_solver is not None:
            if ik is not None:
                raise TypeError("CBiRRT got both ik and ik_solver (deprecated); pass only ik")
            warnings.warn("CBiRRT(ik_solver=...) is deprecated; use ik=", DeprecationWarning, stacklevel=2)
            ik = ik_solver
        if collision_checker is None:
            raise TypeError("CBiRRT needs a collision_checker")
        self.robot = robot
        self.ik = ik
        self.collision = collision_checker
        self.config = config or CBiRRTConfig()
        if backend not in ("python", "native", "auto"):
            raise ValueError(f'backend must be "python", "native", or "auto", got {backend!r}')
        # Selection is decided by the problem's components at solve time (native.lower), never by which
        # optional packages happen to import: a missing extension or adapter is itself a stated reason.
        self.backend = backend

        # Joint-space geometry: limits, metric, interpolation, sampling.
        # Raises ValueError if continuous_joints length does not match robot DOF.
        lower, upper = robot.joint_limits
        self.space = JointSpace(
            lower,
            upper,
            continuous_joints=self.config.continuous_joints,
            joint_names=getattr(robot, "joint_names", None),
        )

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start=None,
        goal=None,
        goal_tsrs=None,
        start_tsrs=None,
        constraint_tsrs=None,
        seed: int | None = None,
        return_details: bool = False,
        trajectory: TrajectoryOptions | None = None,
    ) -> list[np.ndarray] | None | PlanResult:
        """Plan a path from start to goal with optional TSR constraints.

        Multiple TSRs in goal_tsrs or start_tsrs are treated as a union - the
        planner will find a path to ANY of the goal TSRs. TSRs are sampled
        proportionally to their volume (sum of Bw bounds), so TSRs with more
        freedom are explored more frequently.

        Multiple discrete configurations can be provided as lists - all become
        tree roots and are explored simultaneously.

        Args:
            start: Start configuration(s). Can be:
                   - Single config: np.ndarray
                   - Multiple configs: list[np.ndarray]
                   - None (must provide start_tsrs)
            goal: Goal configuration(s). Can be:
                  - Single config: np.ndarray
                  - Multiple configs: list[np.ndarray]
                  - None (must provide goal_tsrs)
            goal_tsrs: Optional TSRs defining goal region(s) (union).
                      Each TSR is sampled proportionally to its volume.
            start_tsrs: Optional TSRs defining valid start regions (union).
            constraint_tsrs: Optional TSRs that constrain the entire path.
                            Every configuration along the path must satisfy ALL of these.
            seed: Random seed for reproducibility
            return_details: If True, return PlanResult with trees; otherwise just path
            trajectory: ``TrajectoryOptions`` to also time the path under velocity and acceleration limits
                (``result.trajectory``; see ``smooth_trajectory``). Needs ``return_details=True``.

        Returns:
            If return_details=False: List of joint configurations or None
            If return_details=True: PlanResult with path, trees, and debug info

        Examples:
            # Single to single
            path = planner.plan(start=q1, goal=q2)

            # Multiple starts to single goal
            path = planner.plan(start=[q1, q2, q3], goal=q_goal)

            # Mix configs and TSRs
            path = planner.plan(start=[q1], goal_tsrs=[tsr1, tsr2])
        """
        if trajectory is not None and not return_details:
            raise ValueError("trajectory=... puts the trajectory on the PlanResult: pass return_details=True")
        if self.ik is None and (start_tsrs or goal_tsrs or constraint_tsrs):
            raise ValueError("TSR starts, goals and constraints need IK: CBiRRT(robot, ik=..., collision_checker=...)")
        start_configs = None if start is None else as_configurations(start, self.space.dof, "start")
        goal_configs = None if goal is None else as_configurations(goal, self.space.dof, "goal")

        problem = legacy_problem(
            self.robot,
            self.ik,
            self.collision,
            self.space,
            self.config,
            start_configs,
            goal_configs,
            start_tsrs,
            goal_tsrs,
            constraint_tsrs,
        )
        result = self.solve(problem, seed=seed, trajectory=trajectory)
        if return_details:
            return result
        return result.path

    def solve(
        self, problem: PlanningProblem, seed: int | None = None, trajectory: TrajectoryOptions | None = None
    ) -> PlanResult:
        """Solve a planning problem with the configured backend (see ``backend`` on the constructor).

        With ``trajectory=TrajectoryOptions(...)``, a found path is also timed under the given limits and attached
        as ``result.trajectory`` (see ``smooth_trajectory``). ``result.path`` stays the planned path. Unless the
        options set a seed, the smoother uses ``seed`` (or 0).
        """
        if trajectory is not None:
            if not isinstance(trajectory, TrajectoryOptions):
                raise TypeError(f"trajectory must be TrajectoryOptions or None, got {type(trajectory).__name__}")
            limits(trajectory, problem.space.dof)  # bad limits fail before planning, not after
        result, lowered = self._solve(problem, seed)
        if trajectory is not None and result.success:
            smooth_seed = trajectory.seed if trajectory.seed is not None else (seed if seed is not None else 0)
            result.trajectory = self._smooth_trajectory(
                problem, result.path, trajectory, smooth_seed, lowered, result.backend_reasons
            )
        return result

    def smooth_trajectory(
        self, problem: PlanningProblem, path: list[np.ndarray], options: TrajectoryOptions
    ) -> Trajectory:
        """A timed trajectory along a valid ``path`` for ``problem``, under ``options``' velocity and acceleration
        limits (#214).

        It starts from the path at rest at every corner, then replaces random windows ``[t1, t2]`` with the
        synchronized time-optimal motion between the states there whenever that is faster and admissible. Each
        shortcut is validated through the problem's boundary: samples at most ``edge_resolution`` apart, each
        admissible (joint space, validator, path constraint), every chord accepted by the problem's motion
        validator. That is the planner's discrete edge guarantee, no stronger. Each piece of the result records how
        it was validated. Never fails on a valid path: at worst the result is the path with a stop at every corner.

        Uses the backend selection of ``solve``: native when the problem lowers (with ``backend="auto"``), else the
        Python reference with the reasons on ``Trajectory.backend_reasons``.
        """
        seed = options.seed if options.seed is not None else 0
        return self._smooth_trajectory(problem, path, options, seed, None)

    def _smooth_trajectory(
        self, problem, path, options: TrajectoryOptions, seed: int, lowered, reasons: tuple[str, ...] = ()
    ) -> Trajectory:
        """``lowered`` from a native solve is reused; ``reasons`` from a solve that could not lower skip lowering."""
        if not isinstance(options, TrajectoryOptions):
            raise TypeError(f"options must be TrajectoryOptions, got {type(options).__name__}")
        vmax, amax = limits(options, problem.space.dof)
        if self.backend != "python" and not reasons:
            from sscbirrt.backends import native

            if lowered is None:
                try:
                    lowered = native.lower(problem, self.config)
                except native.NativeUnsupported as e:
                    if self.backend == "native":
                        raise
                    reasons = tuple(e.reasons)
                    logger.info("native backend not used; smoothing in Python: %s", "; ".join(reasons))
            if lowered is not None:
                traj = native.smooth_trajectory(lowered, path, vmax, amax, options, seed, self.config.abort_fn)
                traj.joint_names = self._joint_names()
                return traj
        traj = shortcut(
            initial_trajectory(coalesce(path), vmax, amax),
            motion_validator=self._motion_validator(problem),
            admissible=lambda q: self._admissible(problem, q),
            resolution=self.config.edge_resolution or self.config.step_size,
            options=options,
            seed=seed,
            abort_fn=self.config.abort_fn,
        )
        traj.joint_names, traj.backend, traj.backend_reasons = self._joint_names(), "python", reasons
        return traj

    def _joint_names(self) -> tuple[str, ...] | None:
        names = getattr(self.robot, "joint_names", None)
        return None if names is None else tuple(str(n) for n in names)

    def _solve(self, problem: PlanningProblem, seed: int | None):
        """The result, and the lowered problem when the native backend solved it (reused for smoothing)."""
        reasons: tuple[str, ...] = ()
        if self.backend != "python":
            from sscbirrt.backends import native

            try:
                lowered = native.lower(problem, self.config)
            except native.NativeUnsupported as e:
                if self.backend == "native":
                    raise
                reasons = tuple(e.reasons)
                logger.info("native backend not used; planning in Python: %s", "; ".join(reasons))
            else:
                return native.solve(lowered, seed, self.config.abort_fn), lowered
        self._counts = {"state_checks": 0, "edge_checks": 0, "search_roots": 0, "reused_verdicts": 0}
        result = self._solve_python(problem, seed)
        result.backend = "python"
        result.backend_reasons = reasons
        result.provenance = {**versions(), "backend": "python"}
        result.stats = dict(self._counts)
        return result, None

    def _solve_python(self, problem: PlanningProblem, seed: int | None = None) -> PlanResult:
        """The reference implementation of ``solve``.

        Args:
            problem: The problem to solve. ``start`` and ``goal`` must each be
                finite, sampleable, or both.
            seed: Random seed for reproducibility.

        Returns:
            PlanResult. ``start_source`` and ``goal_source`` give the
            provenance of the roots the path connects; ``start_index`` and
            ``goal_index`` are their last components (0 if empty).

        Raises:
            UnsupportedCapability: If a start or goal set can neither be
                enumerated nor sampled.
            AllStartConfigurationsInCollision, AllStartConfigurationsInvalid,
            AllGoalConfigurationsInCollision, AllGoalConfigurationsInvalid:
                If no valid root could be found for a role.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        for name, s in (("start", problem.start), ("goal", problem.goal)):
            if not is_finite(s) and not supports(s, SetSampler):
                raise UnsupportedCapability(f"{name} set must be finite or sampleable: {s!r}")

        # Validation errors (no valid roots) propagate; only search failures return a result.
        # Cancellation during root collection is a search failure: Aborted, with what was gathered.
        start_roots: list[Sample] = []
        goal_roots: list[Sample] = []
        try:
            start_roots = self._roots(problem, problem.start, "Start")
            goal_roots = self._roots(problem, problem.goal, "Goal")
        except _AbortedDuringRoots as aborted:
            if aborted.role == "Start":
                start_roots = aborted.roots
            else:
                goal_roots = aborted.roots
            return PlanResult(
                path=None,
                start_index=0,
                goal_index=0,
                iterations=0,
                planning_time=0.0,
                tree_sizes=(len(start_roots), len(goal_roots)),
                success=False,
                failure_reason=f"Aborted by user during {aborted.role.lower()} root collection",
                tree_start=RRTree([r.q for r in start_roots], source_indices=[r.source for r in start_roots]),
                tree_goal=RRTree([r.q for r in goal_roots], source_indices=[r.source for r in goal_roots]),
            )

        tree_start = RRTree([r.q for r in start_roots], source_indices=[r.source for r in start_roots])
        tree_goal = RRTree([r.q for r in goal_roots], source_indices=[r.source for r in goal_roots])

        start_time = time.monotonic()

        def _failure(iteration: int, reason: str) -> PlanResult:
            return PlanResult(
                path=None,
                start_index=0,
                goal_index=0,
                iterations=iteration,
                planning_time=time.monotonic() - start_time,
                tree_sizes=(len(tree_start), len(tree_goal)),
                success=False,
                failure_reason=reason,
                tree_start=tree_start,
                tree_goal=tree_goal,
            )

        # CBiRRT's P_sample (#196): a set that is not finite and can be sampled keeps adding roots during the search.
        # A finite set's members are all roots already, so its tree never tosses the coin.
        roles = {
            id(tree_start): (problem.start, self.config.start_sample_probability),
            id(tree_goal): (problem.goal, self.config.goal_sample_probability),
        }
        seeds = {key: {m.source for m in explicit_samples(s)} for key, (s, _) in roles.items()}
        grows = {key: not is_finite(s) and supports(s, SetSampler) for key, (s, _) in roles.items()}
        counts = getattr(self, "_counts", None)

        for iteration in range(self.config.max_iterations):
            if self._aborted():
                return _failure(iteration, "Aborted by user")

            if time.monotonic() - start_time > self.config.timeout:
                return _failure(
                    iteration,
                    f"Timeout after {self.config.timeout:.1f}s, {iteration} iterations. "
                    f"Trees: start={len(tree_start)} nodes, goal={len(tree_goal)} nodes. "
                    f"Trees could not connect.",
                )

            # Alternate which tree we extend
            if iteration % 2 == 0:
                tree_a, tree_b = tree_start, tree_goal
            else:
                tree_a, tree_b = tree_goal, tree_start

            # Heads: this turn adds roots to tree_a from its own set, and that is the whole turn
            own, p_sample = roles[id(tree_a)]
            if grows[id(tree_a)] and self._rng.random() < p_sample:
                for smp in self._draw_roots(problem, own, seeds[id(tree_a)], None):
                    tree_a.add_root(smp.q, smp.source)
                    if counts is not None:
                        counts["search_roots"] += 1
                continue

            # Tails: an ordinary turn toward a random configuration
            q_sample = (problem.sampler or problem.space).sample(self._rng)

            # Extend tree_a toward the sample (EXT), then connect tree_b to where it got (CON)
            grow_idx, _ = self._grow(problem, tree_a, q_sample, self.config.extend_steps)
            q_reached = tree_a.nodes[grow_idx].config
            connect_idx, connected = self._grow(problem, tree_b, q_reached, self.config.connect_steps)

            if connected:
                path = self._extract_path(tree_start, tree_goal, tree_a, tree_b, grow_idx, connect_idx)
                if self.config.smooth_path:
                    path = self._smooth_path(problem, path)
                    if self.config.shorten_endpoints:
                        shortened = self._shorten_endpoints(problem, path)
                        if shortened is not path:  # an endpoint moved: shortcut the unwound path too
                            path = self._smooth_path(problem, shortened)
                # Angular joints: same physical path, continuous in raw joint values for executors (#77)
                path = problem.space.unwrap_path(path)

                if tree_a is tree_start:
                    start_source = tree_start.get_root_source_index(grow_idx)
                    goal_source = tree_goal.get_root_source_index(connect_idx)
                else:
                    start_source = tree_start.get_root_source_index(connect_idx)
                    goal_source = tree_goal.get_root_source_index(grow_idx)
                start_source = tuple(start_source or ())
                goal_source = tuple(goal_source or ())

                return PlanResult(
                    path=path,
                    start_index=legacy_index(start_source),
                    goal_index=legacy_index(goal_source),
                    iterations=iteration + 1,
                    planning_time=time.monotonic() - start_time,
                    tree_sizes=(len(tree_start), len(tree_goal)),
                    success=True,
                    start_source=start_source,
                    goal_source=goal_source,
                    tree_start=tree_start,
                    tree_goal=tree_goal,
                )

        return _failure(
            self.config.max_iterations,
            f"Max iterations ({self.config.max_iterations}) reached. "
            f"Trees: start={len(tree_start)} nodes, goal={len(tree_goal)} nodes. "
            f"Trees could not connect.",
        )

    # ------------------------------------------------------------------
    # Roots and admissibility
    # ------------------------------------------------------------------

    def _aborted(self) -> bool:
        """Whether ``abort_fn`` asks to stop. Polled once per iteration, per root draw, and per smoothing attempt."""
        return self.config.abort_fn is not None and bool(self.config.abort_fn())

    def _admissible(
        self, problem: PlanningProblem, q: np.ndarray, valid: bool | None = None
    ) -> tuple[bool, str | None]:
        """Whether ``q`` may appear on a path, and if not, why.

        Checked in order: membership in the ambient joint space (shape,
        finiteness, limits), then the validator, then the path constraint.
        Every root, sample, projected extension, and edge sample goes through
        this, so nothing outside ``problem.space`` is ever stored in a tree.
        ``valid`` is the validator's verdict when it is already known (#200).
        """
        counts = getattr(self, "_counts", None)
        if counts is not None:
            counts["state_checks"] += 1
        why = problem.space.why_invalid(q)
        if why is not None:
            return False, f"outside joint space ({why})"
        if not (problem.validator.is_valid(q) if valid is None else valid):
            return False, "in collision"
        if problem.path_constraint is not None and not problem.path_constraint.contains(q):
            return False, "violates path constraints"
        return True, None

    def _draw_roots(
        self,
        problem: PlanningProblem,
        s: StateSet,
        seed_sources: set,
        room: int | None,
        stats: dict[str, int] | None = None,
    ) -> list[Sample]:
        """One sampling draw from ``s``: its admissible candidates, at most ``min(max_per_draw, room)`` of them
        (``None`` for either means no limit from it).

        A draw may yield several candidates (for example every IK solution of one pose). When it yields more than
        can be kept they are visited in a uniformly random order, so the kept ones are a random subset rather than
        the first ones the sampler listed (#168). A candidate that repeats an explicit seed of the set (same
        provenance) is skipped: it is already a root or already rejected. ``stats`` counts why candidates failed.
        The same step builds the trees before the search and adds roots during it (#196).
        """
        candidates = s.sample(self._rng)
        if not candidates:
            if stats is not None:
                stats["sample_failed"] += 1
            return []
        limits = [n for n in (self.config.max_per_draw, room) if n is not None]
        limit = min(limits) if limits else len(candidates)
        if len(candidates) > limit:
            # A draw can yield many more candidates than are kept (an IK solver that enumerates branches and joint
            # windings returns hundreds, in a fixed order); visit them in a random order (#168).
            candidates = [candidates[k] for k in self._rng.permutation(len(candidates))]
        # Windings of one physical configuration share the validator's verdict when it declares that full turns
        # cannot change it (#200); the joint-space and path-constraint checks stay per candidate.
        share = bool(getattr(problem.validator, "full_turn_invariant", False))
        verdicts: dict[tuple[int, ...], bool] = {}
        counts = getattr(self, "_counts", None)
        kept: list[Sample] = []
        for smp in candidates:
            if len(kept) >= limit:
                break
            if smp.source in seed_sources:
                continue
            keyed = share and smp.key is not None
            if keyed and smp.key in verdicts:
                ok, reason = self._admissible(problem, smp.q, valid=verdicts[smp.key])
                if counts is not None:
                    counts["reused_verdicts"] += 1
            else:
                ok, reason = self._admissible(problem, smp.q)
                # The validator ran unless the joint-space check failed first
                if keyed and not (reason or "").startswith("outside joint space"):
                    verdicts[smp.key] = reason != "in collision"
            if ok:
                kept.append(smp)
            elif stats is None:
                pass
            elif reason == "in collision":
                stats["in_collision"] += 1
            elif reason.startswith("outside joint space"):
                stats["outside_space"] += 1
            else:
                stats["constraint_violated"] += 1
        return kept

    def _roots(self, problem: PlanningProblem, s: StateSet, role: str) -> list[Sample]:
        """Collect tree roots for a start or goal set.

        Every explicit configuration embedded in the set (``explicit_samples``: the
        members of finite sets, also inside unions with sampleable regions)
        is a candidate root, validated and filtered with a warning.
        ``abort_fn`` is polled before each sampling draw; if it fires,
        ``_AbortedDuringRoots`` carries the roots gathered so far and
        ``solve`` returns an Aborted result. If the
        set is not finite and supports sampling, admissible samples are
        added until ``num_tree_roots`` roots exist or the sample budget
        (``sample_draws`` draws) is spent; a sampled candidate that repeats
        an explicit seed (same provenance) is skipped. Each draw may yield
        several candidates (for example the IK branches of one pose); at
        most ``max_per_draw`` admissible ones per draw are kept, for
        diversity. A draw with more candidates than that is visited in a
        uniformly random order, so the kept ones are a random subset rather
        than the first ones the sampler listed. Mixture weights govern sampling only, never whether
        explicit seeds are roots.

        Raises:
            AllStartConfigurationsInCollision / AllGoalConfigurationsInCollision:
                every candidate was rejected by the validator.
            AllStartConfigurationsInvalid / AllGoalConfigurationsInvalid:
                every candidate was rejected, for mixed reasons.
        """
        roots: list[Sample] = []
        invalid_details: list[str] = []
        all_in_collision = True

        explicit = explicit_samples(s)
        seed_sources = {m.source for m in explicit}
        for m in explicit:
            ok, reason = self._admissible(problem, m.q)
            if ok:
                roots.append(m)
            else:
                invalid_details.append(f"{role}[{legacy_index(m.source)}]: {reason}")
                if reason != "in collision":
                    all_in_collision = False

        stats = None
        if not is_finite(s) and supports(s, SetSampler):
            stats = {"sample_failed": 0, "outside_space": 0, "in_collision": 0, "constraint_violated": 0}
            for _ in range(self.config.sample_draws):
                if len(roots) >= self.config.num_tree_roots:
                    break
                if self._aborted():
                    raise _AbortedDuringRoots(role, roots)
                roots += self._draw_roots(problem, s, seed_sources, self.config.num_tree_roots - len(roots), stats)

        if roots:
            if invalid_details:
                logger.warning(
                    f"Filtered {len(invalid_details)} invalid {role.lower()} configuration(s): "
                    f"{'; '.join(invalid_details)}"
                )
            return roots

        in_collision_ex = AllStartConfigurationsInCollision if role == "Start" else AllGoalConfigurationsInCollision
        invalid_ex = AllStartConfigurationsInvalid if role == "Start" else AllGoalConfigurationsInvalid

        if explicit:
            if stats is not None and sum(stats.values()) > 0:
                invalid_details.append(f"sampling: {self._sampling_summary(stats)}")
                all_in_collision = all_in_collision and stats["in_collision"] == sum(stats.values())
            ex = in_collision_ex if all_in_collision else invalid_ex
            raise ex(len(explicit), invalid_details)

        if stats is not None and sum(stats.values()) > 0:
            summary = self._sampling_summary(stats)
            only_collisions = stats["in_collision"] == sum(stats.values())
            if only_collisions:
                raise in_collision_ex(stats["in_collision"], [summary])
            raise invalid_ex(sum(stats.values()), [summary])

        raise ValueError(
            f"No valid {role.lower()} configurations available. Provide either {role.lower()} or {role.lower()}_tsrs."
        )

    @staticmethod
    def _sampling_summary(stats: dict[str, int]) -> str:
        details = []
        if stats["sample_failed"]:
            details.append(f"{stats['sample_failed']} IK unreachable")
        if stats["outside_space"]:
            details.append(f"{stats['outside_space']} outside joint space")
        if stats["in_collision"]:
            details.append(f"{stats['in_collision']} in collision")
        if stats["constraint_violated"]:
            details.append(f"{stats['constraint_violated']} constraint violated")
        return ", ".join(details)

    # ------------------------------------------------------------------
    # Tree growth
    # ------------------------------------------------------------------

    def _nearest_node(self, space: JointSpace, tree: RRTree, q_target: np.ndarray) -> int:
        """Find nearest node in tree under the query space's metric.

        ``space`` is the problem's space, never the planner's default: a
        direct ``solve(problem)`` may use a different topology, and every
        geometric operation in a query must agree.
        """
        if space.continuous_joints is None:
            # Use tree's built-in nearest (faster); Euclidean matches the space metric
            return tree.nearest(q_target)

        best_idx = 0
        best_dist = float("inf")
        for i, node in enumerate(tree.nodes):
            dist = space.distance(node.config, q_target)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _grow(
        self,
        problem: PlanningProblem,
        tree: RRTree,
        q_target: np.ndarray,
        max_steps: int | None = None,
    ) -> tuple[int, bool]:
        """Grow tree toward target using EXT or CON behavior.

        Each new configuration is projected onto the path constraint if it
        supports projection, then connected by a checked edge that validates
        every sample including its endpoint (and rejects it if it leaves a
        rejection-only constraint). Once the target is within
        ``connection_tolerance``, the exact target is connected through the
        same checked edge; ``reached`` is True only after that edge exists,
        so every consecutive pair on any returned path has been validated.

        Args:
            problem: The planning problem
            tree: Tree to grow
            q_target: Target configuration to grow toward
            max_steps: Maximum steps (None = CON/unlimited, int = EXT/limited)

        Returns:
            Tuple of (node_index, reached) where:
            - node_index: Index of the furthest node reached toward target
            - reached: True if the tree now contains the exact target,
              connected by a validated edge
        """
        space = problem.space
        constraint = problem.path_constraint
        projector = constraint if constraint is not None and supports(constraint, SetProjector) else None

        if not space.contains(q_target):
            return self._nearest_node(space, tree, q_target), False

        current_idx = self._nearest_node(space, tree, q_target)
        steps_taken = 0
        prev_distance = float("inf")

        while True:
            q_current = tree.nodes[current_idx].config

            direction = space.direction(q_current, q_target)
            distance = float(np.linalg.norm(direction))

            if distance == 0.0:
                return current_idx, True  # already there; no duplicate node

            if distance < self.config.connection_tolerance:
                # Close enough to connect: validate the exact final segment and add the target
                return self._extend_along_edge(problem, tree, current_idx, q_target)

            if prev_distance - distance < self.config.progress_tolerance:
                break
            prev_distance = distance

            if max_steps is not None and steps_taken >= max_steps:
                break

            step = direction / distance * min(distance, self.config.step_size)
            q_new = q_current + step

            if not space.contains(q_new):
                break

            if projector is not None:
                q_projected = projector.project(q_current, q_new)
                if q_projected is None:
                    break
                q_new = np.asarray(q_projected, dtype=float)
                if not space.contains(q_new):
                    break  # a projector may move the point anywhere

            current_idx, reached = self._extend_along_edge(problem, tree, current_idx, q_new)
            steps_taken += 1
            if not reached:
                break

        return current_idx, False

    def _extend_along_edge(
        self,
        problem: PlanningProblem,
        tree: RRTree,
        start_idx: int,
        q_target: np.ndarray,
    ) -> tuple[int, bool]:
        """Extend the tree along the local motion to ``q_target``.

        This is the single local-motion validation boundary, used for
        ordinary growth, the final connection between trees, and shortcut
        smoothing. The problem's ``motion_validator`` (or the default
        ``DiscreteMotionValidator``) returns the validated configurations to
        store; they are added as consecutive nodes. A custom validator
        replaces the default and owns the motion's interior; the planner only
        guarantees that every stored configuration is admissible (it re-checks
        a custom validator's configurations) and that the ``LocalMotion``
        contract holds. See ``sscbirrt.motion`` for the full contract.

        ``reached_target`` is True only if the motion was valid all the way
        and the tree now contains the exact target. With angular joints the
        last raw step may differ by 2π from the target's representation; that
        is the same physical motion (#35).

        The returned ``LocalMotion`` is checked before the tree is touched:

        - A zero-length motion (source already the exact target) succeeds
          without adding a node, whatever the validator returned.
        - ``reached=True`` on a nonzero motion with no configurations, or
          with a final configuration that is not the exact target, is a
          validator bug and raises ``MotionContractError``.
        - ``reached=True`` with any inadmissible configuration (custom
          validators only; the default is built from admissibility) is
          rejected whole: nothing is stored and the result is not reached.
        - ``reached=False`` keeps its admissible prefix: configurations are
          stored up to the first inadmissible one. This is deliberate so that
          partial extensions retain their progress.

        Returns:
            Tuple of (final_idx, reached_target).
        """
        space = problem.space
        if not space.contains(q_target):
            return start_idx, False
        q_from = tree.nodes[start_idx].config
        q_target = np.asarray(q_target, dtype=float)
        if space.distance(q_from, q_target) == 0.0:
            return start_idx, True  # already there; never add a duplicate

        validator = self._motion_validator(problem)
        counts = getattr(self, "_counts", None)
        if counts is not None:
            counts["edge_checks"] += 1
        motion = validator.validate(q_from, q_target)
        configs = [np.asarray(q, dtype=float) for q in motion.configs]

        if motion.reached:
            if not configs:
                raise MotionContractError(
                    f"{type(validator).__name__} reported reached=True with no configurations for a motion of length "
                    f"{space.distance(q_from, q_target):.4g}"
                )
            if configs[-1].shape != q_target.shape or not np.array_equal(configs[-1], q_target):
                raise MotionContractError(
                    f"{type(validator).__name__} reported reached=True but ended at {configs[-1]} instead of the "
                    f"exact target {q_target}"
                )

        custom = problem.motion_validator is not None
        if custom:
            admissible = [self._admissible(problem, q)[0] for q in configs]
            if motion.reached and not all(admissible):
                return start_idx, False  # a claimed success with an inadmissible state: reject whole
            if not motion.reached:
                first_bad = next((i for i, ok in enumerate(admissible) if not ok), len(configs))
                configs = configs[:first_bad]

        current_idx = start_idx
        for q in configs:
            current_idx = tree.add_node(q, current_idx)
        return current_idx, bool(motion.reached)

    def _motion_validator(self, problem: PlanningProblem) -> MotionValidator:
        """The problem's motion validator, or the default discretized one."""
        if problem.motion_validator is not None:
            return problem.motion_validator
        return self.default_motion_validator(problem)

    def default_motion_validator(self, problem: PlanningProblem) -> DiscreteMotionValidator:
        """The validator used when ``problem.motion_validator`` is None.

        Straight segments sampled every ``edge_resolution`` (or ``step_size``),
        each sample checked for admissibility in ``problem``. Use it as the
        base of a ``RestrictedMotionValidator`` to add a restriction while
        keeping the default checks.
        """
        resolution = self.config.edge_resolution or self.config.step_size
        return DiscreteMotionValidator(problem.space, lambda q: self._admissible(problem, q)[0], resolution)

    # ------------------------------------------------------------------
    # Path extraction and smoothing
    # ------------------------------------------------------------------

    def _extract_path(
        self,
        tree_start: RRTree,
        tree_goal: RRTree,
        tree_a: RRTree,
        tree_b: RRTree,
        idx_a: int,
        idx_b: int,
    ) -> list[np.ndarray]:
        """Extract path from connected trees, start to goal.

        The connecting tree holds the exact configuration the other tree
        reached, so the join would repeat it; that duplicate is dropped.
        """
        # get_path_to_root returns path from ROOT to the specified node
        if tree_a is tree_start:
            path_from_start = tree_start.get_path_to_root(idx_a)
            path_from_goal = tree_goal.get_path_to_root(idx_b)
        else:
            path_from_start = tree_start.get_path_to_root(idx_b)
            path_from_goal = tree_goal.get_path_to_root(idx_a)
        path = path_from_start + list(reversed(path_from_goal))
        deduped = [path[0]]
        for q in path[1:]:
            if not np.array_equal(q, deduped[-1]):
                deduped.append(q)
        return deduped

    def _shorten_endpoints(self, problem: PlanningProblem, path: list[np.ndarray]) -> list[np.ndarray]:
        """End the path on the nearest winding of its goal, then its start, when that role is a set (#212).

        The member the trees reached can be a full turn of a bounded joint away from another member that is the same
        pose. The planner proposes the reached endpoint's windings, ``q + 2*pi*k`` on joints whose limits admit a
        full turn (never on continuous joints, whose distance already wraps), and the role's set decides by
        ``contains``, as for every member. A finite set is skipped: its members are explicit and ``goal_index``
        names them. Candidates are tried in order of their straight-line bound on the path length. Each is reached by
        unwinding the path (``_unwind``), which keeps its route; every new edge passes the shortcut boundary every
        shortcut uses (``_try_shortcut``). The first that makes the path shorter is kept: at most one move per
        endpoint, at most ``smoothing_iterations`` attempts each. Deterministic: no random draws.
        """
        for role in ("goal", "start"):
            if self._aborted():
                break
            role_set = problem.goal if role == "goal" else problem.start
            if is_finite(role_set):
                continue
            moved = self._shorten_end(problem, path[::-1] if role == "start" else path, role_set)
            if moved is not None:
                path = moved[::-1] if role == "start" else moved
        return path

    def _shorten_end(
        self, problem: PlanningProblem, path: list[np.ndarray], role_set: StateSet
    ) -> list[np.ndarray] | None:
        """``_shorten_endpoints`` for the last waypoint of ``path``: the shorter path, or None."""
        space = problem.space
        if len(path) < 2:
            return None
        dist = space.distance
        prefix = [0.0]  # path length up to each waypoint
        for a, b in zip(path[:-1], path[1:]):
            prefix.append(prefix[-1] + dist(a, b))
        total = prefix[-1]
        attempts = 0
        candidates = sorted(self._windings(space, path[-1], path[0]), key=lambda c: dist(path[0], c))  # stable
        for target in candidates:
            if attempts >= self.config.smoothing_iterations:
                break
            if dist(path[0], target) >= total - 1e-9:
                break  # sorted by this bound: no later candidate can be shorter either
            if not role_set.contains(target) or not self._admissible(problem, target)[0]:
                continue
            for done in UNWIND_BY:
                if attempts >= self.config.smoothing_iterations or self._aborted():
                    return None
                attempts += 1
                unwound = self._unwind(problem, path, prefix, target, done)
                if unwound is None:
                    continue
                length = sum(dist(a, b) for a, b in zip(unwound[:-1], unwound[1:]))
                if length < total - 1e-9:
                    return unwound
        return None

    def _unwind(
        self,
        problem: PlanningProblem,
        path: list[np.ndarray],
        prefix: list[float],
        target: np.ndarray,
        done: float,
    ) -> list[np.ndarray] | None:
        """The path ending at ``target``, another winding of its end. The change ``delta = target - path[-1]`` (whole
        turns) is made along the first ``done`` of the path's length, in proportion to length; every waypoint after
        that is moved by all of ``delta``, which is the same pose. So the route is kept, and the end of the path, where
        the motion is usually tightest, is physically unchanged. Every edge is re-validated through the shortcut
        boundary; None if any fails."""
        reach = done * prefix[-1]
        delta = target - path[-1]
        moved = [q + min(1.0, prefix[i] / reach) * delta for i, q in enumerate(path)]
        moved[0], moved[-1] = path[0], target  # the start exactly; the end is the checked candidate
        if not all(self._admissible(problem, q)[0] for q in moved[1:-1]):
            return None  # every edge below would end at an inadmissible waypoint; states are cheaper than edges
        # Past the turn, an edge is the path's own validated edge moved by whole turns. When the validator declares
        # its verdict unchanged by full turns (#200) and nothing else judges motion, that edge needs no new check.
        same = (
            bool(getattr(problem.validator, "full_turn_invariant", False))
            and problem.path_constraint is None
            and problem.motion_validator is None
        )
        out = [moved[0]]
        for i, (a, b) in enumerate(zip(moved[:-1], moved[1:])):
            if same and prefix[i] >= reach:
                out.append(b)
                continue
            edge = self._try_shortcut(problem, a, b)
            if edge is None:
                return None
            out.extend(edge[1:])
        return out

    @staticmethod
    def _windings(space: JointSpace, q: np.ndarray, ref: np.ndarray) -> list[np.ndarray]:
        """Other windings of ``q`` that unwind a full turn: ``q + 2*pi*k``, lexicographic over joints. Joint ``j`` is
        offset by ``+-2*pi`` only where that brings it nearer ``ref`` (the path's other end), which is when
        ``|q_j - ref_j| > pi``, and only within its limits; never on a continuous joint. A path with no such joint
        proposes nothing. With more than 10 such joints none are proposed (up to 2^10 candidates)."""
        two_pi = 2.0 * np.pi
        continuous = space.continuous_joints if space.continuous_joints is not None else np.zeros(len(q), dtype=bool)
        choices = []
        for j in range(len(q)):
            options = [0.0]
            if not continuous[j]:
                options += [
                    o
                    for o in (-two_pi, two_pi)
                    if space.lower[j] <= q[j] + o <= space.upper[j]
                    and abs(q[j] + o - ref[j]) < abs(q[j] - ref[j]) - 1e-9
                ]
            choices.append(options)
        if sum(len(c) > 1 for c in choices) > 10:
            return []
        out = []
        for offsets in itertools.product(*choices):
            if any(o != 0.0 for o in offsets):
                candidate = q + np.array(offsets)
                if space.contains(candidate):
                    out.append(candidate)
        return out

    def _smooth_path(self, problem: PlanningProblem, path: list[np.ndarray]) -> list[np.ndarray]:
        """Smooth path by shortcutting with the grow function.

        Picks two random points on the path and attempts to grow from one to
        the other. Stops early, keeping the path as smoothed so far, if
        ``abort_fn`` fires; a valid path exists and is returned as success.
        A shortcut replaces the segment between them only if it is shorter
        in joint-space path length (under ``problem.space``), as
        in the original CBiRRT; a shortcut with fewer waypoints can still be
        longer, for example after projection. Every shortcut is a validated
        edge sequence ending exactly at its target, so the path's first and
        last waypoints are preserved exactly. Stops early after
        ``smoothing_patience`` attempts without improvement.
        """
        if len(path) <= 2:
            return path

        space = problem.space

        def length(segment: list[np.ndarray]) -> float:
            return sum(space.distance(a, b) for a, b in zip(segment[:-1], segment[1:]))

        smoothed = list(path)
        attempts_without_improvement = 0

        for _ in range(self.config.smoothing_iterations):
            if len(smoothed) <= 2:
                break
            if attempts_without_improvement >= self.config.smoothing_patience:
                break
            if self._aborted():
                break  # a valid path exists; return it as smoothed so far

            i = int(self._rng.integers(0, len(smoothed) - 2))
            j = int(self._rng.integers(i + 2, len(smoothed)))

            improved = False
            shortcut = self._try_shortcut(problem, smoothed[i], smoothed[j])
            if shortcut is not None and length(shortcut) < length(smoothed[i : j + 1]) - 1e-9:
                smoothed = smoothed[:i] + shortcut + smoothed[j + 1 :]
                improved = True

            if improved:
                attempts_without_improvement = 0
            else:
                attempts_without_improvement += 1

        return smoothed

    def _try_shortcut(
        self,
        problem: PlanningProblem,
        q_from: np.ndarray,
        q_to: np.ndarray,
    ) -> list[np.ndarray] | None:
        """Try to connect two configurations directly using grow.

        Returns:
            Configurations from ``q_from`` to ``q_to`` inclusive, every
            consecutive pair validated, or None if the connection failed.
            ``q_from == q_to`` gives the single-element path ``[q_from]``.
        """
        temp_tree = RRTree(q_from)
        final_idx, reached = self._grow(problem, temp_tree, q_to, max_steps=None)
        if not reached:
            return None
        return temp_tree.get_path_to_root(final_idx)
