# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Offscreen rendering of planned paths to MP4, with a text overlay and the gripper's trail."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from sscbirrt.demo.scene import EE_SITE, UR5E_JOINTS, close_gripper, set_arm
from sscbirrt.trajectory import Trajectory


class RenderUnavailable(RuntimeError):
    """No OpenGL context for offscreen rendering; the message says what to install or set."""


@dataclass
class Camera:
    lookat: tuple[float, float, float] = (0.35, 0.0, 0.50)
    distance: float = 1.75
    azimuth: float = -135.0
    elevation: float = -25.0


@dataclass
class Clip:
    """One path to show, with its overlay.

    ``caption(q)`` is an optional per-frame readout (for example the gripper's tilt). ``held`` names a free
    body that moves with the gripper: ``(body, T_gripper_body)``.
    """

    path: Sequence[np.ndarray]
    title: str
    lines: Sequence[str] = ()
    caption: Callable[[np.ndarray], str] | None = None
    held: tuple[str, np.ndarray] | None = None
    grip: float | None = None  # close the gripper's fingers on an object this wide (visual only)
    trail_rgba: tuple[float, float, float, float] = (0.1, 0.4, 0.9, 0.9)
    joints: Sequence[str] = field(default_factory=lambda: list(UR5E_JOINTS))
    trajectory: Trajectory | None = None  # when set, the clip plays this timed motion in real time, not ``path``


def _clip_configurations(clip: Clip, fps: int, joint_speed: float, hold_start: float, hold_end: float):
    """Frame-by-frame configurations: a hold, the motion, a hold. A timed clip plays its trajectory in real time;
    otherwise the path plays at a bounded joint speed."""
    path = [np.asarray(q, dtype=float) for q in clip.path]
    qs = [path[0]] * int(hold_start * fps)
    if clip.trajectory is not None:
        traj = clip.trajectory
        q, _, _ = traj.evaluate(np.arange(math.ceil(traj.duration * fps) + 1) / fps)  # clamped at the end
        qs.extend(q)
    else:
        per_frame = joint_speed / fps
        for a, b in zip(path, path[1:]):
            n = max(1, math.ceil(np.max(np.abs(b - a)) / per_frame))
            qs.extend(a + (b - a) * t for t in np.arange(n) / n)
    qs.extend([path[-1]] * max(1, int(hold_end * fps)))
    return qs


def _add_segment(scene: mujoco.MjvScene, a, b, width: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32)
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has only the fixed bitmap font
        return ImageFont.load_default()


def _overlay(frame: np.ndarray, title: str, lines: Sequence[str], caption: str | None) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = image.size[1] / 720
    title_font, body_font = _font(int(34 * scale)), _font(int(22 * scale))
    pad, x, y = int(14 * scale), int(24 * scale), int(22 * scale)

    rows = [(title, title_font)] + [(line, body_font) for line in lines]
    widths = [draw.textbbox((0, 0), text, font=font)[2] for text, font in rows]
    heights = [draw.textbbox((0, 0), text, font=font)[3] + int(8 * scale) for text, font in rows]
    draw.rounded_rectangle(
        (x - pad, y - pad, x + max(widths) + pad, y + sum(heights) + pad), radius=int(10 * scale), fill=(0, 0, 0, 150)
    )
    for (text, font), h in zip(rows, heights):
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        y += h

    if caption:
        box = draw.textbbox((0, 0), caption, font=body_font)
        cy = image.size[1] - int(24 * scale) - box[3]
        draw.rounded_rectangle(
            (x - pad, cy - pad, x + box[2] + pad, cy + box[3] + pad), radius=int(10 * scale), fill=(0, 0, 0, 150)
        )
        draw.text((x, cy), caption, font=body_font, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def _open_renderer(model: mujoco.MjModel, width: int, height: int) -> mujoco.Renderer:
    try:
        renderer = mujoco.Renderer(model, width=width, height=height)
        renderer.render()  # warm-up: the first frame from a fresh context can come out overexposed
        return renderer
    except Exception as e:  # MuJoCo raises plain Exceptions for GL context failures
        gl = os.environ.get("MUJOCO_GL", "the platform default")
        raise RenderUnavailable(
            f"Could not create an OpenGL context for offscreen rendering (MUJOCO_GL={gl}): {e}. "
            "On a headless Linux machine install EGL (for example `apt install libegl1`) or set MUJOCO_GL=osmesa; "
            "or run with --no-video to plan and report without rendering."
        ) from e


def render_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clips: Sequence[Clip],
    out_path: str | Path,
    *,
    camera: Camera | None = None,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    joint_speed: float = 1.2,
    max_frames: int | None = None,
    overlay: bool = True,
) -> int:
    """Render ``clips`` one after another into ``out_path`` (H.264 MP4). Returns the number of frames written.

    A clip with a ``trajectory`` plays it in real time; ``joint_speed`` (rad/s) paces the others along their paths.
    ``max_frames`` subsamples evenly (for smoke tests); the video then plays faster, not shorter. ``overlay=False``
    leaves out the text boxes (for small renders, where the caption goes alongside instead).
    """
    import imageio.v2 as imageio

    camera = camera or Camera()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = camera.lookat
    cam.distance, cam.azimuth, cam.elevation = camera.distance, camera.azimuth, camera.elevation

    plan = [(clip, q) for clip in clips for q in _clip_configurations(clip, fps, joint_speed, 0.8, 1.2)]
    if max_frames is not None and len(plan) > max_frames:
        keep = np.linspace(0, len(plan) - 1, max_frames).round().astype(int)
        plan = [plan[i] for i in keep]

    site = model.site(EE_SITE).id
    renderer = _open_renderer(model, width, height)
    out_path = Path(out_path)
    written = 0
    try:
        with imageio.get_writer(
            out_path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=7, ffmpeg_log_level="error"
        ) as writer:
            trail: list[np.ndarray] = []
            current = None
            for clip, q in plan:
                if clip is not current:
                    current, trail = clip, []
                if clip.grip is not None:
                    close_gripper(model, data, clip.grip)
                set_arm(model, data, q, clip.joints)
                if clip.held is not None:
                    _place_held(model, data, *clip.held)
                trail.append(data.site_xpos[site].copy())
                renderer.update_scene(data, camera=cam)
                stride = max(1, len(trail) // 400)
                points = trail[::stride] + [trail[-1]]
                for a, b in zip(points, points[1:]):
                    if np.linalg.norm(b - a) > 1e-6:
                        _add_segment(renderer.scene, a, b, 0.004, clip.trail_rgba)
                frame = renderer.render()
                if overlay:
                    frame = _overlay(frame, clip.title, clip.lines, clip.caption(q) if clip.caption else None)
                writer.append_data(frame)
                written += 1
    finally:
        renderer.close()
    return written


def _place_held(model: mujoco.MjModel, data: mujoco.MjData, body: str, T_gripper_body: np.ndarray) -> None:
    """Move a free body so it stays rigidly attached to the gripper (visual only)."""
    from sscbirrt.demo.scene import GRIPPER_BODY

    g = model.body(GRIPPER_BODY).id
    T_world_gripper = np.eye(4)
    T_world_gripper[:3, :3] = data.xmat[g].reshape(3, 3)
    T_world_gripper[:3, 3] = data.xpos[g]
    T = T_world_gripper @ T_gripper_body
    joint = model.body(body).jntadr[0]
    adr = model.jnt_qposadr[joint]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
    data.qpos[adr : adr + 3] = T[:3, 3]
    data.qpos[adr + 3 : adr + 7] = quat
    mujoco.mj_forward(model, data)
