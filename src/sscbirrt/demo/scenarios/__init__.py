# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The demo scenarios. Each makes one claim a viewer can check in its video."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mujoco

from sscbirrt.demo.render import Camera, Clip
from sscbirrt.demo.scene import ACCELERATION_LIMIT, VELOCITY_LIMIT
from sscbirrt.trajectory import TrajectoryOptions, coalesce, initial_trajectory

# Every demo motion is timed under the same limits (#214): ``plan(..., trajectory=TIMING)``.
TIMING = TrajectoryOptions(VELOCITY_LIMIT, ACCELERATION_LIMIT)


def timing_report(result) -> str:
    """One report line: the timed duration against stopping at every corner, and the smoothing cost."""
    traj = result.trajectory
    stops = initial_trajectory(coalesce(result.path), traj.velocity_limits, traj.acceleration_limits).duration
    return (
        f"timed: {traj.duration:.2f} s at {VELOCITY_LIMIT} rad/s and {ACCELERATION_LIMIT} rad/s^2 "
        f"({stops:.2f} s with a stop at every corner), smoothed in {traj.diagnostics['seconds']:.3f} s ({traj.backend})"
    )


@dataclass
class Outcome:
    """What a scenario produced: report lines, the clips to render, and the world to render them in."""

    ok: bool
    report: list[str]
    model: mujoco.MjModel
    data: mujoco.MjData
    clips: list[Clip] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)


@dataclass
class Problem:
    """A scenario's main planning call, for benchmarks: ``plan(model, data, arm, config=config, seed=s, **kwargs)``."""

    model: mujoco.MjModel
    data: mujoco.MjData
    arm: Any  # sscbirrt.mujoco.Arm
    config: Any  # sscbirrt.CBiRRTConfig
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class Scenario:
    name: str
    claim: str
    run: Callable[[int], Outcome]  # seed -> outcome
    problem: Callable[[], Problem] | None = None  # the main planning call, without running it


def all_scenarios() -> dict[str, Scenario]:
    from sscbirrt.demo.scenarios import door, pick, transport

    return {s.name: s for s in (pick.SCENARIO, transport.SCENARIO, door.SCENARIO)}
