# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Custom exceptions for sscbirrt planner."""


class PlanningError(Exception):
    """Base exception for planning errors."""

    pass


class AllStartConfigurationsInCollision(PlanningError):
    """Raised when all provided start configurations are in collision."""

    def __init__(self, n_configs: int, details: list[str] | None = None):
        self.n_configs = n_configs
        self.details = details or []
        msg = f"All {n_configs} start configuration(s) in collision"
        if details:
            msg += f": {'; '.join(details)}"
        super().__init__(msg)


class AllGoalConfigurationsInCollision(PlanningError):
    """Raised when all provided goal configurations are in collision."""

    def __init__(self, n_configs: int, details: list[str] | None = None):
        self.n_configs = n_configs
        self.details = details or []
        msg = f"All {n_configs} goal configuration(s) in collision"
        if details:
            msg += f": {'; '.join(details)}"
        super().__init__(msg)


class AllStartConfigurationsInvalid(PlanningError):
    """Raised when all provided start configurations are invalid (collision or constraint violation)."""

    def __init__(self, n_configs: int, details: list[str] | None = None):
        self.n_configs = n_configs
        self.details = details or []
        msg = f"All {n_configs} start configuration(s) invalid"
        if details:
            msg += f": {'; '.join(details)}"
        super().__init__(msg)


class AllGoalConfigurationsInvalid(PlanningError):
    """Raised when all provided goal configurations are invalid (collision or constraint violation)."""

    def __init__(self, n_configs: int, details: list[str] | None = None):
        self.n_configs = n_configs
        self.details = details or []
        msg = f"All {n_configs} goal configuration(s) invalid"
        if details:
            msg += f": {'; '.join(details)}"
        super().__init__(msg)


class InvalidPath(PlanningError):
    """Raised when a given path is not valid under the planner's own rules, so it cannot be timed (#267).

    ``CBiRRT.time_path`` raises it, before any timing, for the expected operational failure: a path whose state or
    straight motion breaks the joint limits, collides, or leaves the path constraint, such as a rebase bridge that is
    blocked. ``report`` is the path's ``ValidationReport`` under the validator ``plan`` applies (clearance included);
    its ``witness`` names the failure, the segment (0 is the path's first motion) and fraction, the state, and the
    contacts when the checker can name them.
    """

    def __init__(self, report):
        self.report = report
        w = report.witness
        where = "" if w is None else f": {w.failure} in segment {w.segment} at fraction {w.fraction:.4g} ({w.detail})"
        super().__init__(f"path is not valid under the planner's rules{where}")


class UnsupportedCapability(TypeError):
    """Raised when a state set is asked for a capability it does not provide.

    For example: sampling a multi-child ``AnyOf`` without a mixture policy,
    or projecting onto a multi-child ``AllOf`` without a projection strategy.
    """

    pass


class MotionContractError(ValueError):
    """Raised when a MotionValidator returns a LocalMotion that violates its contract.

    For example: ``reached=True`` with no configurations on a nonzero motion,
    or a final configuration that is not the exact target. These are bugs in
    the validator, not planning failures, so they are raised rather than
    treated as an unreachable motion.
    """

    pass


class NativeUnsupported(Exception):
    """The native backend cannot take this problem, or is not installed. ``reasons`` lists every blocker.

    Raised by ``backend="native"`` and by the native MuJoCo scene; ``backend="auto"`` records the same reasons in
    ``PlanResult.backend_reasons`` and plans in Python instead.
    """

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons) if self.reasons else "native backend unavailable")
