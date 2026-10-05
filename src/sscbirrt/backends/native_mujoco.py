# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The owned MuJoCo scene and immutable snapshots (docs/native-design.md, v1.7.0 addendum).

``import sscbirrt`` never imports MuJoCo. This module imports ``sscbirrt._native_mujoco`` lazily and
turns a missing module or a version mismatch into ``NativeUnsupported`` with a reason.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from sscbirrt.backends.native import NativeUnsupported

_module: Any = None


def _load():
    """The native scene module, or raise NativeUnsupported with the reason."""
    global _module
    if _module is not None:
        return _module
    try:
        import mujoco
    except ImportError:
        raise NativeUnsupported(["mujoco is not installed; install sscbirrt[mujoco] (mujoco==3.14.0)"]) from None
    try:
        import importlib

        _native_mujoco = importlib.import_module("sscbirrt._native_mujoco")
    except ImportError as e:
        raise NativeUnsupported(
            [f"sscbirrt._native_mujoco is not available: built without MuJoCo support, or it failed to load ({e})"]
        ) from None
    compiled, loaded, installed = (
        _native_mujoco.compiled_mujoco_version(),
        _native_mujoco.loaded_mujoco_version(),
        mujoco.__version__,
    )
    if not (compiled == loaded == installed):
        raise NativeUnsupported(
            [
                f"MuJoCo version mismatch: sscbirrt._native_mujoco was built against {compiled}, loaded library "
                f"{loaded}, installed mujoco package {installed}; install mujoco=={compiled}"
            ]
        )
    _module = _native_mujoco
    return _module


def available() -> bool:
    try:
        _load()
        return True
    except NativeUnsupported:
        return False


def unavailable_reason() -> str | None:
    try:
        _load()
        return None
    except NativeUnsupported as e:
        return e.reasons[0]


def export_mjb(model) -> bytes:
    """The MJB bytes of a compiled ``mujoco.MjModel``."""
    import mujoco

    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return buf.tobytes()


class NativeScene:
    """An owned native ``mjModel`` for the controlled joints, built once from a Python model.

    The Python model may be destroyed or mutated afterward; the scene does not observe it. ``from_model``
    caches by the SHA-256 of the model's MJB bytes, the joint list, and the extra bodies, so any change to
    the source model, structural or numeric, yields a new scene on the next call. MuJoCo's ``signature``
    is not the key: it hashes structure only (a geom resized in place keeps its signature), and is
    recorded as provenance.
    """

    _cache: dict[tuple, "NativeScene"] = {}

    def __init__(self, native, joint_names: tuple[str, ...], extra_arm_bodies: tuple[str, ...]):
        self.native = native
        self.joint_names = joint_names
        self.extra_arm_bodies = extra_arm_bodies

    @classmethod
    def from_model(cls, model, joint_names, extra_arm_bodies=()) -> "NativeScene":
        mod = _load()
        mjb = export_mjb(model)
        key = (hashlib.sha256(mjb).hexdigest(), tuple(joint_names), tuple(extra_arm_bodies))
        scene = cls._cache.get(key)
        if scene is None:
            native = mod.Scene(mjb, list(joint_names), list(extra_arm_bodies), int(model.signature))
            scene = cls(native, tuple(joint_names), tuple(extra_arm_bodies))
            cls._cache[key] = scene
        return scene

    @property
    def dof(self) -> int:
        return self.native.dof

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.native.lower, dtype=float), np.array(self.native.upper, dtype=float)

    @property
    def provenance(self) -> dict[str, Any]:
        p = self.native.provenance
        return {"mujoco": p.mujoco_version, "model_signature": p.model_signature, "mjb_sha256": p.mjb_sha256}


def gripper_allowed_bodies(scene: NativeScene, gripper_body_name: str) -> list[int]:
    """mj_manipulator's gripper rule as data: the subtree of ``<prefix>/base`` when that body exists, else of
    the attachment body itself. Contacts between the grasped object and these bodies are allowed."""
    native = scene.native
    base = None
    if "/" in gripper_body_name:
        candidate = gripper_body_name.rsplit("/", 1)[0] + "/base"
        if native.body_id(candidate) >= 0:
            base = candidate
    root = native.body_id(base if base is not None else gripper_body_name)
    if root < 0:
        raise ValueError(f"gripper body '{gripper_body_name}' not found in the scene")
    return list(native.subtree(root))


class Snapshot:
    """An immutable copy of what a geometric query reads from a live ``MjData``: qpos, mocap poses, attachments."""

    def __init__(self, native, scene: NativeScene):
        self.native = native
        self.scene = scene

    @classmethod
    def capture(
        cls, scene: NativeScene, data, attachments: dict[str, tuple[str, np.ndarray]] | None = None
    ) -> "Snapshot":
        """Copy the live state now. ``attachments`` is mj_manipulator's ``{object: (gripper_body, T_gripper_object)}``.
        Call on the thread that owns ``data``."""
        mod = _load()
        native = scene.native
        atts = []
        for obj, (gripper, T) in (attachments or {}).items():
            ob, gb = native.body_id(obj), native.body_id(gripper)
            if ob < 0:
                raise ValueError(f"attached object body '{obj}' not found in the scene")
            if gb < 0:
                raise ValueError(f"gripper body '{gripper}' not found in the scene")
            atts.append(
                mod.Attachment(ob, gb, np.asarray(T, dtype=float).tolist(), gripper_allowed_bodies(scene, gripper))
            )
        snap = mod.Snapshot(
            native,
            np.asarray(data.qpos, dtype=float).tolist(),
            np.asarray(data.mocap_pos, dtype=float).reshape(-1).tolist(),
            np.asarray(data.mocap_quat, dtype=float).reshape(-1).tolist(),
            atts,
        )
        return cls(snap, scene)

    @property
    def sha256(self) -> str:
        return self.native.sha256

    @property
    def qpos(self) -> np.ndarray:
        return np.array(self.native.qpos, dtype=float)


class NativeCollisionChecker:
    """A sscbirrt ``CollisionChecker`` over a scene and a snapshot, with mj_manipulator's contact policy.

    ``is_valid`` calls the native validator, so the Python backend exercises the same implementation the
    native backend lowers to. It is a ``sscbirrt.backends.native.ValidatorIntegration``: ``fresh()`` returns a
    new validator on the same scene and snapshot with its own ``mjData``, and lowering calls it so that every
    solve owns its validator.
    """

    def __init__(self, scene: NativeScene, snapshot: Snapshot):
        if snapshot.scene is not scene:
            raise ValueError("the snapshot was captured for a different scene")
        self.scene = scene
        self.snapshot = snapshot
        self.native = _load().SceneValidator(scene.native, snapshot.native)

    def fresh(self):
        """A new native validator with its own mjData, for one solve."""
        return _load().SceneValidator(self.scene.native, self.snapshot.native)

    @property
    def provenance(self) -> dict[str, Any]:
        """The scene and snapshot this checker validates against (``ValidatorIntegration``)."""
        return {
            "scene_model_signature": self.scene.provenance["model_signature"],
            "scene_mjb_sha256": self.scene.provenance["mjb_sha256"],
            "snapshot_sha256": self.snapshot.sha256,
        }

    @property
    def full_turn_invariant(self) -> bool:
        """Every controlled joint is a hinge, so no verdict depends on winding (#200)."""
        return bool(self.native.full_turn_invariant())

    def is_valid(self, q) -> bool:
        return bool(self.native.is_valid([float(x) for x in np.asarray(q, dtype=float)]))

    def invalid_contacts(self, q) -> list[dict[str, Any]]:
        """Every contact the policy counts against ``q``, with body names, for diagnostics."""
        out = []
        for c in self.native.invalid_contacts([float(x) for x in np.asarray(q, dtype=float)]):
            out.append(
                {
                    "kind": c.kind,
                    "body1": self.scene.native.body_name(c.body1),
                    "body2": self.scene.native.body_name(c.body2),
                    "geom1": c.geom1,
                    "geom2": c.geom2,
                    "dist": c.dist,
                }
            )
        return out


class _NoIK:
    """Stands in for the IK solver when a problem has no pose regions; ``plan_native`` rejects pose regions
    without an ``ik`` before planning, so any use is a programming error."""

    def solve(self, pose, q_init=None):
        raise RuntimeError("internal error: plan_native planned pose regions without an IK solver")


def plan_native(
    model,
    data,
    joint_names,
    *,
    start,
    goal=None,
    goal_tsrs=None,
    constraint_tsrs=None,
    ik=None,
    robot=None,
    ee_site: str = "attachment_site",
    attachments: dict[str, tuple[str, np.ndarray]] | None = None,
    extra_arm_bodies=(),
    config=None,
    seed: int | None = None,
    fallback: bool = False,
):
    """One call from a live MuJoCo world to a native solve (docs/native-design.md, v1.7.0).

    Deprecated since 3.1.0: use ``sscbirrt.mujoco.plan`` with an ``sscbirrt.mujoco.Arm``.

    Builds or reuses the owned scene for ``model``, captures a snapshot of ``data`` now (call on the
    thread that owns it), plans with ``backend="native"`` (or ``"auto"`` when ``fallback`` is true), and
    returns a ``PlanResult`` whose ``provenance`` names the scene, the snapshot, and every dependency.

    ``ik`` is an ``SSIKSolver`` around an ``ssik.Manipulator`` built from the same MJCF (needed only when
    ``goal_tsrs`` or ``constraint_tsrs`` are given); ``robot`` defaults to ``MuJoCoRobotModel`` on
    ``ee_site``, and lowering checks its forward kinematics against SSIK's at the start configurations.
    ``attachments`` is mj_manipulator's ``{object_body: (gripper_body, T_gripper_object)}``.
    """
    import warnings

    from sscbirrt import CBiRRT, CBiRRTConfig
    from sscbirrt.backends.mujoco import MuJoCoRobotModel

    warnings.warn(
        "plan_native is deprecated and will be removed in 4.0; use sscbirrt.mujoco.plan(model, data, Arm(...), ...)",
        DeprecationWarning,
        stacklevel=2,
    )
    # Check the arguments before any work, so the first error is about what the caller passed (#172).
    if ik is None and (goal_tsrs or constraint_tsrs):
        raise ValueError(
            "plan_native: goal_tsrs and constraint_tsrs need an IK solver; pass ik= "
            "(for example SSIKSolver(ssik.Manipulator.from_mjcf(...), T_ee=...))"
        )
    if robot is None:
        robot = MuJoCoRobotModel(model, data, ee_site, list(joint_names))  # checks the site and every joint name
    try:
        scene = NativeScene.from_model(model, joint_names, extra_arm_bodies)
    except NativeUnsupported as e:
        # There is no Python form of the owned scene, so fallback cannot cover it.
        raise NativeUnsupported(
            [f"plan_native needs the native MuJoCo scene ({r}); fallback=True covers planning components, not the scene"
             for r in e.reasons]
        ) from None  # fmt: skip
    snapshot = Snapshot.capture(scene, data, attachments)
    checker = NativeCollisionChecker(scene, snapshot)
    planner = CBiRRT(
        robot,
        ik if ik is not None else _NoIK(),
        checker,
        config or CBiRRTConfig(),
        backend="auto" if fallback else "native",
    )
    return planner.plan(
        start=start, goal=goal, goal_tsrs=goal_tsrs, constraint_tsrs=constraint_tsrs, seed=seed, return_details=True
    )
