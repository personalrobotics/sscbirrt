# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""door: open a door by its handle. The gripper must follow the handle's arc about the hinge: a TSR chain."""

from __future__ import annotations

import mujoco
import numpy as np
from tsr import TSR, Robotiq2F85, TSRChain

from sscbirrt import CBiRRTConfig
from sscbirrt.backends.mujoco import site_offset_in_body
from sscbirrt.demo.render import Camera, Clip
from sscbirrt.demo.scenarios import Outcome, Problem, Scenario
from sscbirrt.demo.scene import (
    DOOR_BODY,
    DOOR_HINGE,
    EE_SITE,
    GRIPPER_BODY,
    HANDLE_LENGTH,
    HANDLE_OFFSET,
    HANDLE_RADIUS,
    HOME,
    UR5E_JOINTS,
    build_scene,
    set_arm,
    ur5e_xml,
)
from sscbirrt.mujoco import Arm, plan

OPEN = np.radians(60.0)  # at 70 the wrist nears the UR5e's shoulder singularity (over the base axis)


def _translation(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


def door_chain(lo: float, hi: float) -> TSRChain:
    """Grasps of the handle with the door open between ``lo`` and ``hi`` radians: the hinge, then the handle grasp.

    Link 1 turns about the hinge's vertical axis and carries the frame to the handle's base; link 2 is one fixed
    side grasp of the handle bar from the robot's side, fingers level, from the tsr package's Robotiq 2F-85.
    """
    templates = Robotiq2F85().grasp_cylinder_side(cylinder_radius=HANDLE_RADIUS, cylinder_height=HANDLE_LENGTH)
    mid_level = next(t for t in templates if "mid" in t.name and "roll 0" in t.name)
    # A chain composes each link on the previous link's end frame and ignores the T0_w of every link after the first
    # (Berenson et al. 2011, sec. 5.1), so the template's frame on the bar goes into the hinge link's Tw_e.
    hinge = TSR(
        T0_w=_translation(DOOR_HINGE),
        Tw_e=_translation(HANDLE_OFFSET) @ mid_level.T_ref_tsr,
        Bw=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [lo, hi]],
    )
    facing_robot = np.pi / 2  # the gripper on the handle's +y side, approaching the door
    grasp = TSR(
        T0_w=np.eye(4),
        Tw_e=mid_level.Tw_e,
        Bw=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [facing_robot, facing_robot]],
    )
    return TSRChain(TSRs=[hinge, grasp])


def _held(model: mujoco.MjModel) -> np.ndarray:
    """The door's pose in the gripper body's frame while its handle is grasped (the same at every opening)."""
    T_world_grasp = door_chain(0.0, 0.0).sample()
    T_world_gripper = T_world_grasp @ np.linalg.inv(site_offset_in_body(model, EE_SITE))
    return np.linalg.inv(T_world_gripper) @ _translation(DOOR_HINGE)


def opening(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> float:
    """How far the door is open at ``q``, in radians, from where the gripper holds the handle."""
    set_arm(model, data, q)
    T_gripper = np.eye(4)
    T_gripper[:3, :3] = data.body(GRIPPER_BODY).xmat.reshape(3, 3)
    T_gripper[:3, 3] = data.body(GRIPPER_BODY).xpos
    T_door = T_gripper @ _held(model)
    return float(np.arctan2(T_door[1, 0], T_door[0, 0]))


def problem() -> Problem:
    """Open the door along the chain: start holding the handle of the closed door, end at OPEN, follow the arc."""
    model = build_scene(table=False, door=True)
    data = mujoco.MjData(model)
    set_arm(model, data, HOME)
    arm = Arm(model, UR5E_JOINTS, EE_SITE, mjcf=ur5e_xml())
    # Every IK solution of the fixed grasp becomes a root: with the grasp fixed, the arm's set is a curve per IK
    # branch, and start and goal roots must share a branch. Keeping the first few candidates per draw (#168) left
    # them on different branches and the trees never met.
    config = CBiRRTConfig(
        step_size=0.1, edge_resolution=0.05, num_tree_roots=400, max_per_draw=400, sample_draws=2, timeout=120.0
    )
    held = {DOOR_BODY: (GRIPPER_BODY, _held(model))}
    kwargs = {
        "start": door_chain(0.0, 0.0),
        "goal": door_chain(OPEN, OPEN),
        "constraint": door_chain(-0.02, OPEN + 0.02),
        "holding": held,
    }
    return Problem(model, data, arm, config, kwargs)


def run(seed: int) -> Outcome:
    p = problem()
    model, data, arm, config, held = p.model, p.data, p.arm, p.config, p.kwargs["holding"]
    # Reach the handle from HOME first (the door closed), to any grasp of it, then open the door from wherever the
    # reach arrived. The other order planned the opening from any grasp and then had to reach that one IK solution,
    # which can lie on a joint winding HOME cannot reach within the limits (#198).
    reach = plan(model, data, arm, goal=p.kwargs["start"], config=config, seed=seed)
    if not reach.success:
        return Outcome(False, [f"reach: no path: {reach.failure_reason}"], model, data)
    opened = plan(model, data, arm, config=config, seed=seed, **{**p.kwargs, "start": reach.path[-1]})
    if not opened.success:
        return Outcome(False, [f"open: no path: {opened.failure_reason}"], model, data)

    view = mujoco.MjData(model)
    angles = [opening(model, view, q) for q in opened.path]
    report = [
        f"opened to {np.degrees(angles[-1]):.0f} deg along the handle's arc (door angle stayed within "
        f"[{np.degrees(min(angles)):.1f}, {np.degrees(max(angles)):.1f}] deg)",
        f"as a TSR chain: {opened.backend}, {opened.planning_time:.2f} s, {len(opened.path)} waypoints",
    ]
    report += [f"  not native because: {r}" for r in dict.fromkeys(opened.backend_reasons)]

    clips = [
        Clip(path=reach.path, title="Reach the handle", lines=["The door is closed"]),
        Clip(
            path=opened.path,
            title="Open the door: a TSR chain",
            lines=["The gripper follows the handle's arc about the hinge"],
            held=(DOOR_BODY, held[DOOR_BODY][1]),
            grip=2 * HANDLE_RADIUS,
        ),
    ]
    return Outcome(True, report, model, data, clips, Camera(lookat=(0.35, -0.3, 0.5), distance=2.0, azimuth=-120.0))


SCENARIO = Scenario(
    name="door",
    claim="a constraint can be a chain of regions: the gripper opens a door by following its handle's arc",
    run=run,
    problem=problem,
)
