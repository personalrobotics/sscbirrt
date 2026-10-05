# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CBiRRTConfig:
    """Configuration for CBiRRT planner.

    Extension behavior (EXT vs CON):
    - CON (connect): March until blocked or target reached (extension_steps=None)
    - EXT (extend): Take at most X steps toward target (extension_steps=X)

    The planner supports 4 variants based on extend_steps and connect_steps:
    - CON-CON: Both trees march until blocked (default, like RRT-Connect)
    - EXT-EXT: Both trees take limited steps
    - EXT-CON: Extend tree takes limited steps, connect tree marches
    - CON-EXT: Extend tree marches, connect tree takes limited steps
    """

    # Termination
    timeout: float = 30.0  # Wall-clock timeout in seconds
    max_iterations: int = 100000  # Safety limit (timeout is the primary control)

    # Tolerances. Each has one meaning. membership_tolerance, projection_progress_tolerance and
    # max_projection_iters configure the sets plan() builds from its arguments; a set passed to solve() in a
    # PlanningProblem keeps its own constructor values (FiniteSet tolerance 1e-6, TSRConfigurationSet 1e-3).
    membership_tolerance: float = 1e-3  # A configuration is in a TSR-induced set if its TSR distance is within this
    connection_tolerance: float = 1e-3  # Tree growth counts as reaching its target within this joint-space distance
    # Spacing of collision checks along an edge, a joint-space distance in step_size's units (#204). Independent of
    # step_size: on a UR5e a point can move ~1.2 m per rad, so 0.05 rad bounds the motion between checks to ~6 cm.
    # None means step_size, the default before 3.4.
    edge_resolution: float | None = 0.05
    progress_tolerance: float = 1e-6  # Tree growth stops when the distance to target shrinks by less than this
    projection_progress_tolerance: float = 1e-6  # Projection gives up when the violation shrinks by less than this

    # Tree growth parameters
    step_size: float = 0.1  # Maximum joint space step

    # CBiRRT's P_sample (#196): on a tree's turn, the probability that the turn adds roots drawn from that tree's own
    # set instead of extending toward a random configuration. A finite set never does: its members are all roots.
    start_sample_probability: float = 0.1
    goal_sample_probability: float = 0.1

    # Extension behavior (None = CON, int = EXT with X steps)
    extend_steps: int | None = None  # Steps when growing toward random sample
    connect_steps: int | None = None  # Steps when growing toward other tree

    # Constraint projection
    max_projection_iters: int = 50  # Max iterations for projecting onto constraint manifold

    # Roots: how many configurations each tree starts from, drawn from a sampleable start or goal set
    sample_draws: int = 100  # Sampling draws per role (a TSR set's draw is one pose and its IK solutions)
    num_tree_roots: int = 100  # Target number of root configs to seed each tree with
    max_per_draw: int = 3  # Candidates kept per draw, a random subset when a draw has more (for diversity)

    # Smoothing
    smooth_path: bool = True
    smoothing_iterations: int = 50
    smoothing_patience: int = 15  # Stop early if no improvement in this many attempts

    # Continuous joints: True marks a joint with no limits (a turntable, the UR3e's
    # wrist 3), whose distance wraps at 2*pi and whose limit check is skipped.
    # A limited joint is a bounded interval however wide its range: a +-2*pi joint
    # like the UR5e's can reach any angle, but q and q + 2*pi are different states
    # and moving between them is a real full rotation. Leave such joints False
    # (or continuous_joints None) so the planner respects their limits. Every
    # revolute joint is angular; only an unlimited one is continuous.
    continuous_joints: tuple[bool, ...] | None = None

    # Abort callback — return True to stop planning early
    abort_fn: Callable[[], bool] | None = field(default=None, repr=False)

    def __post_init__(self):
        self._validate_ranges()

    def _validate_ranges(self) -> None:
        """Reject malformed values at construction; the native contract applies the same ranges (#108)."""

        def check(name, ok, requirement):
            if not ok:
                raise ValueError(f"{name} must be {requirement}, got {getattr(self, name)!r}")

        for name in ("timeout", "step_size", "progress_tolerance", "projection_progress_tolerance"):
            check(name, getattr(self, name) > 0, "positive")
        for name in ("membership_tolerance", "connection_tolerance"):
            check(name, getattr(self, name) >= 0, "nonnegative")
        for name in ("max_iterations", "sample_draws", "num_tree_roots", "max_per_draw", "max_projection_iters"):
            check(name, getattr(self, name) >= 1, "at least 1")
        for name in ("smoothing_iterations", "smoothing_patience"):
            check(name, getattr(self, name) >= 0, "nonnegative")
        for name in ("edge_resolution",):
            check(name, getattr(self, name) is None or getattr(self, name) > 0, "None or positive")
        for name in ("extend_steps", "connect_steps"):
            check(name, getattr(self, name) is None or getattr(self, name) >= 1, "None or at least 1")
        for name in ("start_sample_probability", "goal_sample_probability"):
            check(name, 0.0 <= getattr(self, name) <= 1.0, "within [0, 1]")


# Names renamed in 3.1.0 (#176). The old keyword still works in the constructor and the old attribute still reads and
# writes, each with a DeprecationWarning, until 4.0. They are translated here rather than kept as dataclass fields, so
# dataclasses.replace() and the constructor see only the current fields: a stored alias would copy a stale value back
# over the one being replaced.
_RENAMED = {
    "tsr_samples": "sample_draws",
    "max_ik_per_pose": "max_per_draw",
    "angular_joints": "continuous_joints",
    # 3.3.0 (#196): the biases steered a turn's extension toward a member of the other tree's set; the probabilities
    # add roots to the tree's own set. The alias maps each to its own role's probability.
    "start_bias": "start_sample_probability",
    "goal_bias": "goal_sample_probability",
}
_MEANING_CHANGED = {"start_bias", "goal_bias"}
_dataclass_init = CBiRRTConfig.__init__


def _warn_renamed(old: str, new: str, stacklevel: int) -> None:
    note = ""
    if old in _MEANING_CHANGED:
        note = " (since 3.3.0 it adds roots to its own tree on that tree's turn instead of steering the other, #196)"
    warnings.warn(f"CBiRRTConfig.{old} is deprecated; use {new}{note}", DeprecationWarning, stacklevel=stacklevel)


def _init(self, *args, tsr_tolerance: float | None = None, **kwargs):
    for old, new in _RENAMED.items():
        if old in kwargs:
            if new in kwargs:
                raise TypeError(f"CBiRRTConfig got both {old} (deprecated) and {new}; pass only {new}")
            _warn_renamed(old, new, 3)
            kwargs[new] = kwargs.pop(old)
    if tsr_tolerance is not None:
        warnings.warn(
            "CBiRRTConfig.tsr_tolerance is deprecated; set membership_tolerance and connection_tolerance separately",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("membership_tolerance", tsr_tolerance)
        kwargs.setdefault("connection_tolerance", tsr_tolerance)
    _dataclass_init(self, *args, **kwargs)


_init.__signature__ = __import__("inspect").signature(_dataclass_init)
CBiRRTConfig.__init__ = _init


def _renamed_property(old: str, new: str) -> property:
    def get(self):
        _warn_renamed(old, new, 3)
        return getattr(self, new)

    def set_(self, value):
        _warn_renamed(old, new, 3)
        setattr(self, new, value)

    return property(get, set_, doc=f"Deprecated alias of ``{new}``.")


for _old, _new in _RENAMED.items():
    setattr(CBiRRTConfig, _old, _renamed_property(_old, _new))


def _tsr_tolerance_get(self):
    warnings.warn(
        "CBiRRTConfig.tsr_tolerance is deprecated; read membership_tolerance", DeprecationWarning, stacklevel=2
    )
    return self.membership_tolerance


def _tsr_tolerance_set(self, value):
    warnings.warn(
        "CBiRRTConfig.tsr_tolerance is deprecated; set membership_tolerance and connection_tolerance",
        DeprecationWarning,
        stacklevel=2,
    )
    self.membership_tolerance = self.connection_tolerance = value


CBiRRTConfig.tsr_tolerance = property(_tsr_tolerance_get, _tsr_tolerance_set, doc="Deprecated alias.")
