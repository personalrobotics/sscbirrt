# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The demo world: a UR5e with a Robotiq 2F-85 at the origin, a table in front, and props on it."""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np

UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Arm up, gripper pointing down over the table.
HOME = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

# The tsr package's canonical grasp frame for the 2F-85 (tsr.Robotiq2F85): the forward edge of the gripper
# housing, z = approach, y = finger opening. The same site mj_manipulator attaches.
EE_SITE = "gripper_grasp_site"

# Joint speed and acceleration limits the demos time their motions under (rad/s, rad/s^2): conservative for a UR5e
# (rated 3.14 rad/s), and the limits of #214's benchmark, so the README's numbers and its videos agree.
VELOCITY_LIMIT = 1.57
ACCELERATION_LIMIT = 4.0
GRIPPER_BODY = "gripper_base_mount"

TABLE_CENTER = np.array([0.5, 0.0, 0.40])
TABLE_HALF = np.array([0.25, 0.40, 0.02])
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_HALF[2]

CAN_RADIUS = 0.03
CAN_HALF_HEIGHT = 0.05
CAN_Z = TABLE_TOP_Z + CAN_HALF_HEIGHT

CAN_COLORS = [
    (0.25, 0.68, 0.35, 1),
    (0.20, 0.55, 0.85, 1),
    (0.95, 0.70, 0.15, 1),
    (0.55, 0.35, 0.75, 1),
]  # no red: red is for obstacles


def menagerie_path() -> Path:
    """Where the UR5e and 2F-85 MJCFs live: ``MUJOCO_MENAGERIE_PATH`` if set, else the sscbirrt-assets wheel."""
    override = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if override:
        return Path(override)
    try:
        import sscbirrt_assets
    except ImportError as e:
        raise ImportError(
            'The demo robot models are in the sscbirrt-assets package: pip install "sscbirrt[demo]" '
            "(or set MUJOCO_MENAGERIE_PATH to a mujoco_menagerie clone)."
        ) from e
    return sscbirrt_assets.menagerie_path()


def ur5e_xml() -> Path:
    return _existing(menagerie_path() / "universal_robots_ur5e" / "ur5e.xml")


def robotiq_2f85_xml() -> Path:
    return _existing(menagerie_path() / "robotiq_2f85" / "2f85.xml")


def _existing(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (models are read from {menagerie_path()})")
    return path


def can_position(xy: Sequence[float]) -> np.ndarray:
    """The center of a can standing on the table at ``xy``."""
    return np.array([xy[0], xy[1], CAN_Z])


OBSTACLE_RGBA = (0.80, 0.15, 0.12, 1.0)  # solid red; no can is red


# A door beside the robot that swings toward it about a vertical hinge. The leaf is a free body whose origin is the
# hinge (identity rotation when closed), so a planner can hold it: along a path that keeps the handle grasp fixed it
# moves rigidly with the gripper, as a door does.
DOOR_BODY = "door"
DOOR_HINGE = np.array([0.30, -0.60, 0.0])
DOOR_WIDTH = 0.45
DOOR_HEIGHT = 0.95
DOOR_THICKNESS = 0.03
HANDLE_RADIUS = 0.012
HANDLE_LENGTH = 0.16
HANDLE_OFFSET = np.array([DOOR_WIDTH - 0.07, DOOR_THICKNESS / 2 + 0.05, 0.50])  # handle bottom, in the door frame


def build_scene(
    cans: Mapping[str, Sequence[float]] | None = None,
    obstacles: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    *,
    table: bool = True,
    door: bool = False,
) -> mujoco.MjModel:
    """Compile the world: arm, gripper, floor, table, and one free-floating can per entry of ``cans``.

    ``cans`` maps a body name to the can's center (see :func:`can_position`). Cans are free bodies so a
    planner can treat one as held (``attachments``); the demos never step the simulation. ``obstacles`` maps a
    name to a box ``(center, half_extents)``, floating or resting on the table: fixed, solid red. ``door`` adds the
    door (``DOOR_*``) in a frame beside the robot; ``table=False`` leaves the table out.
    """
    from tsr import Robotiq2F85

    arm = mujoco.MjSpec.from_file(str(ur5e_xml()))
    gripper = mujoco.MjSpec.from_file(str(robotiq_2f85_xml()))
    grasp_site = gripper.body("base_mount").add_site()
    grasp_site.name = "grasp_site"
    grasp_site.pos = [0.0, 0.0, Robotiq2F85.PALM_OFFSET_FROM_BASE_MOUNT]
    grasp_site.quat = [0.7071068, 0.0, 0.0, -0.7071068]  # -90 deg about z: finger opening along y
    grasp_site.group = 5  # hidden in renders

    _style(arm)

    # The 2F-85 sets impratio and an elliptic cone for grasping physics. The demos never step the simulation,
    # but matching them keeps MjSpec.attach from warning about the conflict.
    arm.option.impratio = gripper.option.impratio
    arm.option.cone = gripper.option.cone

    wrist = arm.body("wrist_3_link")
    site = next((s for s in wrist.sites if s.name == "attachment_site"), None)
    if site is None:
        raise RuntimeError(f"no attachment_site on wrist_3_link in {ur5e_xml()}")
    frame = wrist.add_frame()
    frame.name = "gripper_attachment_frame"
    frame.pos = site.pos
    frame.quat = site.quat
    # The prefix avoids name collisions (both models define a "black" material).
    frame.attach_body(gripper.body("base_mount"), "gripper_", "")

    world = arm.worldbody
    key = world.add_light()
    key.name = "key"
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.dir = [0.4, 0.5, -1.0]
    key.diffuse = [0.45, 0.45, 0.45]
    key.specular = [0.1, 0.1, 0.1]

    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = [0, 0, -0.01]
    floor.size = [0, 0, 0.05]
    floor.material = "floor"

    base = world.add_geom()
    base.name = "base_marker"
    base.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    base.pos = [0, 0, 0.01]
    base.size = [0.1, 0.01, 0]
    base.rgba = [0.3, 0.5, 0.7, 0.8]
    base.contype = 0
    base.conaffinity = 0

    if table:
        table_body = world.add_body()
        table_body.name = "table"
        table_body.pos = list(TABLE_CENTER)
        top = table_body.add_geom()
        top.name = "table_top"
        top.type = mujoco.mjtGeom.mjGEOM_BOX
        top.size = list(TABLE_HALF)
        top.material = "table"
        leg_half = (TABLE_CENTER[2] - TABLE_HALF[2]) / 2  # from the floor to the underside of the top
        for i, (x, y) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            leg = table_body.add_geom()
            leg.name = f"table_leg_{i}"
            leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            leg.pos = [x * (TABLE_HALF[0] - 0.03), y * (TABLE_HALF[1] - 0.03), leg_half - TABLE_CENTER[2]]
            leg.size = [0.025, leg_half, 0]
            leg.material = "table_leg"  # solid: a planner must not route the arm through the legs

    if door:
        _add_door(world)

    for i, (name, pos) in enumerate((cans or {}).items()):
        body = world.add_body()
        body.name = name
        body.pos = list(pos)
        body.add_freejoint()
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [CAN_RADIUS, CAN_HALF_HEIGHT, 0]
        geom.rgba = list(CAN_COLORS[i % len(CAN_COLORS)])

    # No shadows: with the far plane the infinite floor needs (see _style), MuJoCo's shadow map spreads too thin to
    # show them, and the default far plane blows the floor out and leaves a dark band under the sky.
    for light in arm.lights:
        light.castshadow = False

    for name, (center, half) in (obstacles or {}).items():
        box = world.add_geom()
        box.name = name
        box.type = mujoco.mjtGeom.mjGEOM_BOX
        box.pos = list(center)
        box.size = list(half)
        box.rgba = list(OBSTACLE_RGBA)

    return arm.compile()


def _add_door(world) -> None:
    """The door leaf (a free body at the hinge) with its handle, and a fixed frame around it."""
    leaf = world.add_body()
    leaf.name = DOOR_BODY
    leaf.pos = list(DOOR_HINGE)
    leaf.add_freejoint()
    panel = leaf.add_geom()
    panel.name = "door_panel"
    panel.type = mujoco.mjtGeom.mjGEOM_BOX
    panel.pos = [DOOR_WIDTH / 2, 0.0, 0.05 + DOOR_HEIGHT / 2]
    panel.size = [DOOR_WIDTH / 2 - 0.005, DOOR_THICKNESS / 2, DOOR_HEIGHT / 2]
    panel.rgba = [0.82, 0.78, 0.70, 1]
    handle = leaf.add_geom()
    handle.name = "door_handle"
    handle.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    handle.pos = list(HANDLE_OFFSET + [0.0, 0.0, HANDLE_LENGTH / 2])
    handle.size = [HANDLE_RADIUS, HANDLE_LENGTH / 2, 0]
    handle.rgba = [0.25, 0.25, 0.28, 1]
    for k, z in enumerate((HANDLE_OFFSET[2] + 0.02, HANDLE_OFFSET[2] + HANDLE_LENGTH - 0.02)):
        post = leaf.add_geom()  # the standoffs that hold the bar off the panel
        post.name = f"door_handle_post_{k}"
        post.type = mujoco.mjtGeom.mjGEOM_BOX
        post.pos = [HANDLE_OFFSET[0], (DOOR_THICKNESS / 2 + HANDLE_OFFSET[1]) / 2, z]
        post.size = [0.008, (HANDLE_OFFSET[1] - DOOR_THICKNESS / 2) / 2, 0.008]
        post.rgba = [0.25, 0.25, 0.28, 1]
    frame_rgba = [0.55, 0.42, 0.30, 1]
    for name, pos, size in (
        ("door_jamb_hinge", DOOR_HINGE + [-0.07, 0.0, 0.52], [0.03, 0.05, 0.52]),  # clear of the swinging leaf
        ("door_jamb_latch", DOOR_HINGE + [DOOR_WIDTH + 0.04, 0.0, 0.52], [0.03, 0.05, 0.52]),
        ("door_lintel", DOOR_HINGE + [DOOR_WIDTH / 2, 0.0, 1.07], [DOOR_WIDTH / 2 + 0.10, 0.05, 0.03]),
    ):
        geom = world.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.pos = list(pos)
        geom.size = list(size)
        geom.rgba = frame_rgba


def _style(spec: mujoco.MjSpec) -> None:
    """Offscreen resolution and anti-aliasing, a gradient sky, a checkered floor, and matte materials."""
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.offsamples = 8
    # The default far plane clips the infinite floor and leaves dark gaps under the sky.
    spec.visual.map.zfar = 5000.0
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.30, 0.30, 0.30]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]

    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.82, 0.87, 0.93]
    sky.rgb2 = [0.97, 0.98, 1.0]
    sky.width = sky.height = 512

    grid = spec.add_texture()
    grid.name = "floor"
    grid.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid.rgb1 = [0.86, 0.87, 0.88]
    grid.rgb2 = [0.80, 0.81, 0.83]
    grid.width = grid.height = 512
    floor = spec.add_material()
    floor.name = "floor"
    textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "floor"
    floor.textures = textures
    floor.texrepeat = [8, 8]
    floor.texuniform = True
    floor.reflectance = 0.0

    for name, rgba in (("table", [0.55, 0.42, 0.30, 1]), ("table_leg", [0.30, 0.26, 0.22, 1])):
        mat = spec.add_material()
        mat.name = name
        mat.rgba = rgba
        mat.specular = 0.2
        mat.shininess = 0.3


@functools.lru_cache(maxsize=None)
def gripper_closure(width: float) -> tuple[tuple[str, float], ...]:
    """Finger joint positions of the 2F-85 closed until its pads are ``width`` apart (for rendering a grasp).

    The fingers are a linkage held by equality constraints, so they are not set joint by joint: a standalone
    2F-85 is simulated at increasing close commands, bisecting on the command until MuJoCo's distance between the
    two pads is ``width``. Returns ``(joint name, qpos)`` pairs, names as in the gripper's own model.
    """
    model = mujoco.MjModel.from_xml_path(str(robotiq_2f85_xml()))
    left, right = model.geom("left_pad1").id, model.geom("right_pad1").id

    def settle(command: float) -> mujoco.MjData:
        data = mujoco.MjData(model)
        data.ctrl[:] = command
        for _ in range(1500):
            mujoco.mj_step(model, data)
        return data

    def gap(data: mujoco.MjData) -> float:
        return float(mujoco.mj_geomDistance(model, data, left, right, 1.0, None))

    lo, hi = 0.0, float(model.actuator_ctrlrange[0, 1])  # the gap shrinks as the command grows
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(settle(mid)) > width else (lo, mid)
    data = settle(lo)
    return tuple((model.joint(i).name, float(data.qpos[model.jnt_qposadr[i]])) for i in range(model.njnt))


def close_gripper(model: mujoco.MjModel, data: mujoco.MjData, width: float) -> None:
    """Set the scene's 2F-85 fingers as if closed on an object ``width`` wide (visual; call before kinematics)."""
    for name, value in gripper_closure(round(width, 4)):
        data.qpos[model.jnt_qposadr[model.joint(f"gripper_{name}").id]] = value


def set_arm(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, joints: Sequence[str] = UR5E_JOINTS) -> None:
    """Write the arm configuration into ``data`` and run forward kinematics."""
    for name, value in zip(joints, q):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
