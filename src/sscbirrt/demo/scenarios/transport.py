# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""transport: carry a held can across the table, kept upright the whole way by a path constraint."""

from __future__ import annotations

import mujoco
import numpy as np
from tsr import TSR, Robotiq2F85

from sscbirrt import CBiRRTConfig
from sscbirrt.demo.render import Camera, Clip
from sscbirrt.demo.scenarios import Outcome, Problem, Scenario
from sscbirrt.demo.scene import (
    CAN_HALF_HEIGHT,
    CAN_RADIUS,
    EE_SITE,
    GRIPPER_BODY,
    HOME,
    TABLE_TOP_Z,
    UR5E_JOINTS,
    build_scene,
    can_position,
    set_arm,
    ur5e_xml,
)
from sscbirrt.mujoco import Arm, plan

CAN = "green_can"
PICK_AT = can_position((0.45, 0.25))
PLACE_AT = can_position((0.50, -0.25))
# A box between pick and place, and a pillar behind the robot so the upright carry cannot simply swing the can
# around the base at table height: it has to lift it over the box, upright.
OBSTACLES = {
    "box_on_table": ((0.52, 0.0, TABLE_TOP_Z + 0.09), (0.06, 0.06, 0.09)),
    "pillar_behind": ((-0.50, 0.0, 0.65), (0.08, 0.30, 0.65)),
}
TILT_LIMIT = 0.05  # rad


def _grasp(center: np.ndarray) -> TSR:
    """One side grasp of a can at ``center``: mid depth, fingers level (roll 0), at mid height, any approach angle.

    The same grasp at pick and place, because the held can keeps its pose in the gripper. The height is pinned so
    that every member holds the can the same way: the can is symmetric about its axis, so the approach angle does
    not change the grasp, and ``HELD`` is exact for all of them.
    """
    T_bottom = np.eye(4)
    T_bottom[:3, 3] = np.asarray(center) - [0.0, 0.0, CAN_HALF_HEIGHT]
    templates = Robotiq2F85().grasp_cylinder_side(cylinder_radius=CAN_RADIUS, cylinder_height=2 * CAN_HALF_HEIGHT)
    mid_level = next(t for t in templates if "mid" in t.name and "roll 0" in t.name)
    region = mid_level.instantiate(T_bottom)
    bounds = np.array(region.Bw, dtype=float)
    bounds[2] = 0.0  # mid height only
    return TSR(T0_w=region.T0_w, Tw_e=region.Tw_e, Bw=bounds)


def _held(model: mujoco.MjModel) -> np.ndarray:
    """The can's pose in the gripper body's frame for this grasp, from the grasp geometry alone."""
    from sscbirrt.backends.mujoco import site_offset_in_body

    region = _grasp(np.zeros(3))
    T_grasp_can = np.linalg.inv(region.T0_w @ region.Tw_e)  # the can's center is the region's origin here
    return site_offset_in_body(model, EE_SITE) @ T_grasp_can


def upright() -> TSR:
    """Held can upright: the grasp frame's x axis (the can's axis) within TILT_LIMIT of vertical, any heading,
    above the table."""
    Tw_e = np.eye(4)
    Tw_e[:3, :3] = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]  # grasp x -> world z, as in a roll-0 grasp
    bounds = [
        [-2, 2],
        [-2, 2],
        [TABLE_TOP_Z, 1.5],
        [-TILT_LIMIT, TILT_LIMIT],
        [-TILT_LIMIT, TILT_LIMIT],
        [-np.pi, np.pi],
    ]
    return TSR(T0_w=np.eye(4), Tw_e=Tw_e, Bw=bounds)


def tilt(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> float:
    """How far the held can leans from vertical at ``q``, in radians."""
    set_arm(model, data, q)
    axis = data.site(EE_SITE).xmat.reshape(3, 3)[:, 0]
    return float(np.arccos(np.clip(axis[2], -1.0, 1.0)))


def problem() -> Problem:
    """The upright carry: from any grasp of the can at PICK_AT to the same grasp at PLACE_AT, upright throughout.

    Starting from the set of all grasps of the can lets the planner pick a grasp configuration from which an upright
    carry exists (from some windings of the wrist there is none within the joint limits).
    """
    model = build_scene({CAN: PICK_AT}, OBSTACLES)
    data = mujoco.MjData(model)
    set_arm(model, data, HOME)
    arm = Arm(model, UR5E_JOINTS, EE_SITE, mjcf=ur5e_xml())
    config = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, num_tree_roots=20, timeout=30.0)
    held = {CAN: (GRIPPER_BODY, _held(model))}
    kwargs = {"start": _grasp(PICK_AT), "goal": _grasp(PLACE_AT), "constraint": upright(), "holding": held}
    return Problem(model, data, arm, config, kwargs)


def run(seed: int) -> Outcome:
    p = problem()
    model, data, held = p.model, p.data, p.kwargs["holding"]
    view = mujoco.MjData(model)

    def max_tilt(path) -> float:
        return max(tilt(model, view, q) for q in path)

    def length(path) -> float:
        return float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))

    carried = plan(model, data, p.arm, config=p.config, seed=seed, **p.kwargs)
    if not carried.success:
        return Outcome(False, [f"upright: no path: {carried.failure_reason}"], model, data)

    view_tilts = [tilt(model, view, q) for q in carried.path]
    report = [
        f"upright carry: max tilt {np.degrees(max(view_tilts)):.1f} deg, {length(carried.path):.2f} rad of joint "
        f"travel, {carried.backend}, {carried.planning_time:.2f} s",
        f"constraint: the can's roll and pitch each within {np.degrees(TILT_LIMIT):.2f} deg "
        f"(so its tilt within {np.degrees(np.hypot(TILT_LIMIT, TILT_LIMIT)):.2f} deg)",
        "start: chosen from every grasp of the can, so that an upright carry exists",
    ]
    clips = [
        Clip(
            path=carried.path,
            title="Carried upright: a path constraint",
            lines=["Constraint: keep the can upright"],
            caption=lambda q: f"tilt {np.degrees(tilt(model, view, q)):5.1f} deg",
            held=(CAN, held[CAN][1]),
            grip=2 * CAN_RADIUS,
        )
    ]
    set_arm(model, data, carried.path[0])
    return Outcome(True, report, model, data, clips, Camera())


SCENARIO = Scenario(
    name="transport",
    claim="a path constraint holds everywhere: the can is carried over a box and kept upright the whole way",
    run=run,
    problem=problem,
)
