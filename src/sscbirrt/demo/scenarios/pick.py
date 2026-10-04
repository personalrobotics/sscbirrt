# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""pick: the goal is a set, every side grasp of any of three cans, and one call plans to it around obstacles."""

from __future__ import annotations

import mujoco

from sscbirrt import CBiRRTConfig
from sscbirrt.demo.grasps import side_grasps
from sscbirrt.demo.render import Camera, Clip
from sscbirrt.demo.scenarios import Outcome, Problem, Scenario
from sscbirrt.demo.scene import EE_SITE, HOME, TABLE_TOP_Z, UR5E_JOINTS, build_scene, can_position, set_arm, ur5e_xml
from sscbirrt.mujoco import Arm, plan

CANS = {
    "green can": can_position((0.45, 0.22)),
    "blue can": can_position((0.62, -0.05)),
    "yellow can": can_position((0.40, -0.25)),
}

# Boxes across the direct routes: one resting in the middle of the table, one hovering above the cans, one over
# the near table edge, and two beside the arm.
OBSTACLES = {
    "box_on_table": ((0.52, 0.08, TABLE_TOP_Z + 0.06), (0.05, 0.05, 0.06)),
    "box_over_cans": ((0.50, 0.00, 0.78), (0.12, 0.14, 0.04)),
    "box_near_edge": ((0.22, -0.02, 0.62), (0.05, 0.10, 0.06)),
    "box_left": ((0.25, 0.48, 0.70), (0.07, 0.07, 0.07)),
    "box_right": ((-0.10, -0.45, 0.60), (0.08, 0.08, 0.08)),
}


def owners() -> list[str]:
    """The can each goal region belongs to, in the goal list's order (``result.goal_index`` indexes it)."""
    return [name for name, center in CANS.items() for _ in side_grasps(center)]


def problem() -> Problem:
    """Plan from HOME to any side grasp of any can."""
    model = build_scene({name.replace(" ", "_"): pos for name, pos in CANS.items()}, OBSTACLES)
    data = mujoco.MjData(model)
    set_arm(model, data, HOME)
    arm = Arm(model, UR5E_JOINTS, EE_SITE, mjcf=ur5e_xml())
    goals = [region for center in CANS.values() for region in side_grasps(center)]
    config = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, num_tree_roots=20, timeout=30.0)
    return Problem(model, data, arm, config, {"goal": goals})


def run(seed: int) -> Outcome:
    p = problem()
    model, data, goals, owner = p.model, p.data, p.kwargs["goal"], owners()
    result = plan(model, data, p.arm, config=p.config, seed=seed, **p.kwargs)

    report = [f"goal set: {len(goals)} side-grasp regions of 3 cans (tsr.Robotiq2F85 cylinder primitive)"]
    if not result.success:
        return Outcome(False, report + [f"no path: {result.failure_reason}"], model, data)
    reached = owner[result.goal_index]
    report += [
        f"reached: the {reached} (goal_index {result.goal_index})",
        f"backend: {result.backend}, {result.planning_time:.2f} s, {len(result.path)} waypoints, seed {seed}",
    ]
    report += [f"  not native because: {r}" for r in result.backend_reasons]
    if result.backend == "native":
        prov = result.provenance
        report.append(
            f"provenance: SSIK {prov.get('ssik_solver_name')}, snapshot {prov.get('snapshot_sha256', '')[:12]}"
        )

    clip = Clip(
        path=result.path,
        title="One goal set, many ways to reach it",
        lines=[
            f"Goal: any side grasp of any can ({len(goals)} regions)",
            f"Planner chose: the {reached}",
            f"{result.backend} backend, planned in {result.planning_time:.2f} s",
        ],
    )
    return Outcome(True, report, model, data, [clip], Camera())


SCENARIO = Scenario(
    name="pick",
    claim="a goal can be a set: one call plans to whichever side grasp of whichever can is reachable",
    run=run,
    problem=problem,
)
