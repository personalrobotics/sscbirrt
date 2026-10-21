# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The owned MuJoCo scene and immutable snapshots (docs/native-design.md, v1.7.0 addendum).

``import sscbirrt`` never imports MuJoCo. This module imports ``sscbirrt._native_mujoco`` lazily and
turns a missing module or a version mismatch into ``NativeUnsupported`` with a reason.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Union

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

    ``from_model`` captures the compiled model on first use and does not observe later in-place edits:
    treat an ``MjModel`` as immutable after first use. A repeated call with the same model object, joints
    and extra bodies returns the same scene without inspecting the model; the cache holds the model by weak
    reference, so an entry dies with its model and cannot be inherited through object-ID reuse. A model
    object seen for the first time is exported and its MJB hashed in full, and that SHA-256 is the
    cross-object key, so two model objects with the same content share a scene. After an intentional edit,
    pass ``refresh=True`` or a new model object. MuJoCo's ``signature`` is not a key: it hashes structure
    only (a geom resized in place keeps its signature), and is recorded as provenance.
    """

    _cache: dict[tuple, "NativeScene"] = {}
    _by_model: "weakref.WeakKeyDictionary[Any, dict[tuple, NativeScene]]" = weakref.WeakKeyDictionary()

    def __init__(self, native, joint_names: tuple[str, ...], extra_arm_bodies: tuple[str, ...]):
        self.native = native
        self.joint_names = joint_names
        self.extra_arm_bodies = extra_arm_bodies

    @classmethod
    def from_model(cls, model, joint_names, extra_arm_bodies=(), *, refresh: bool = False) -> "NativeScene":
        """The scene for ``model``. ``refresh=True`` re-exports and rehashes the model after an in-place edit."""
        mod = _load()
        params = (tuple(joint_names), tuple(extra_arm_bodies))
        per_model = cls._by_model.setdefault(model, {})
        scene = None if refresh else per_model.get(params)
        if scene is None:
            mjb = export_mjb(model)
            key = (hashlib.sha256(mjb).hexdigest(), *params)
            scene = cls._cache.get(key)
            if scene is None:
                native = mod.Scene(mjb, list(params[0]), list(params[1]), int(model.signature))
                scene = cls(native, *params)
                cls._cache[key] = scene
            per_model[params] = scene
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


BodyRef = Union[str, int]
"""A body of the scene: its name, or its body id. Either denotes that exact body, not its subtree."""


def _check_ref(ref, field: str):
    if isinstance(ref, (bool, np.bool_)) or not isinstance(ref, (str, int, np.integer)):
        raise TypeError(f"{field}: expected a body name or body id, got {type(ref).__name__}")
    return ref if isinstance(ref, str) else int(ref)


def _real_array(value, field: str, shape: tuple[int, ...]) -> np.ndarray:
    """``value`` as a float array of ``shape``; booleans, strings and objects are refused, not converted."""
    a = np.asarray(value)
    if a.dtype.kind not in "iuf":
        raise TypeError(f"{field}: expected real numbers, got an array of dtype {a.dtype}")
    if a.shape != shape:
        raise ValueError(f"{field}: expected shape {shape}, got {a.shape}")
    return a.astype(float)


@dataclass(frozen=True)
class Attachment:
    """An object held by the arm for a snapshot: it rides on ``gripper_body`` at ``T_gripper_object``.

    ``allowed_bodies`` are the bodies the object may touch, in the arm or the environment (the grasp, a support it
    is lifted from, a surface it is lowered onto, a container):

    - ``None``: the gripper's subtree, by mj_manipulator's rule (``gripper_allowed_bodies``);
    - ``()``: nothing;
    - otherwise exactly those bodies.

    Each name or id denotes that exact body; to allow a subtree, list it (``scene.native.subtree(id)``). Contacts
    between two held objects are always checked. Body references are resolved against the scene when a snapshot is
    built; a reference that is not a name or an id, or a malformed transform, is refused here.
    """

    object_body: BodyRef
    gripper_body: BodyRef
    T_gripper_object: Any
    allowed_bodies: Sequence[BodyRef] | None = None

    def __post_init__(self):
        object.__setattr__(self, "object_body", _check_ref(self.object_body, "object_body"))
        object.__setattr__(self, "gripper_body", _check_ref(self.gripper_body, "gripper_body"))
        T = _real_array(self.T_gripper_object, "T_gripper_object", (4, 4))
        if not np.all(np.isfinite(T)):
            raise ValueError("T_gripper_object: has a non-finite entry")
        object.__setattr__(self, "T_gripper_object", tuple(map(tuple, T.tolist())))
        allowed = self.allowed_bodies
        if allowed is not None:
            if isinstance(allowed, (str, bytes, Mapping)) or not hasattr(allowed, "__iter__"):
                raise TypeError(
                    f"allowed_bodies: expected None or a sequence of body names or ids, got {type(allowed).__name__}"
                )
            allowed = tuple(_check_ref(b, f"allowed_bodies[{j}]") for j, b in enumerate(allowed))
        object.__setattr__(self, "allowed_bodies", allowed)


def _resolve_body(native, ref: BodyRef, field: str) -> int:
    if isinstance(ref, str):
        b = native.body_id(ref)
        if b < 0:
            raise ValueError(f"{field}: body '{ref}' not found in the scene")
        return b
    if not 0 <= ref < native.nbody:
        raise ValueError(f"{field}: body id {ref} is out of range [0, {native.nbody})")
    return ref


def _native_attachments(scene: "NativeScene", attachments) -> list:
    """Public attachments, or the legacy ``{object: (gripper, T_gripper_object)}`` mapping, as native values."""
    mod, native = _load(), scene.native
    if attachments is None:
        return []
    if isinstance(attachments, Mapping):  # legacy: the gripper's subtree is allowed
        items = []
        for obj, value in attachments.items():
            if not (isinstance(value, Sequence) and len(value) == 2):
                raise TypeError(f"attachments[{obj!r}]: expected (gripper_body, T_gripper_object)")
            items.append((f"attachments[{obj!r}]", Attachment(obj, value[0], value[1])))
    elif isinstance(attachments, (str, bytes)) or not hasattr(attachments, "__iter__"):
        raise TypeError(f"attachments: expected a sequence of Attachment, got {type(attachments).__name__}")
    else:
        items = []
        for i, a in enumerate(attachments):
            if not isinstance(a, Attachment):
                raise TypeError(f"attachments[{i}]: expected an Attachment, got {type(a).__name__}")
            items.append((f"attachments[{i}]", a))
    out = []
    for where, a in items:
        ob = _resolve_body(native, a.object_body, f"{where}.object_body")
        if not native.body_has_free_joint(ob):
            raise ValueError(f"{where}.object_body: '{native.body_name(ob)}' has no free joint as its first joint")
        gb = _resolve_body(native, a.gripper_body, f"{where}.gripper_body")
        if not native.is_arm_body(gb):
            raise ValueError(f"{where}.gripper_body: '{native.body_name(gb)}' is not part of the arm")
        if a.allowed_bodies is None:
            allowed = gripper_allowed_bodies(scene, native.body_name(gb))
        else:
            allowed = [_resolve_body(native, b, f"{where}.allowed_bodies[{j}]") for j, b in enumerate(a.allowed_bodies)]
        # The frame check (rotation, homogeneous row) and duplicate objects are the snapshot's own validation.
        out.append(mod.Attachment(ob, gb, [list(r) for r in a.T_gripper_object], allowed))
    return out


class Snapshot:
    """An immutable copy of what a geometric query reads: qpos, mocap poses, and attachments.

    Build one with ``capture`` from a live ``MjData``, or with ``from_state`` from state the caller already owns;
    the same state gives the same snapshot and ``sha256``. Each attachment's allowed bodies are part of the snapshot
    (and its hash), so a new allowance set is a new snapshot over the same ``NativeScene``.
    """

    def __init__(self, native, scene: NativeScene):
        self.native = native
        self.scene = scene

    @classmethod
    def capture(cls, scene: NativeScene, data, attachments=None) -> "Snapshot":
        """Copy the live state of ``data`` now. Call on the thread that owns ``data``.

        ``attachments`` is a sequence of ``Attachment``, or mj_manipulator's ``{object: (gripper_body,
        T_gripper_object)}``, which allows the gripper's subtree (``allowed_bodies=None``)."""
        return cls.from_state(scene, data.qpos, data.mocap_pos, data.mocap_quat, attachments)

    @classmethod
    def from_state(cls, scene: NativeScene, qpos, mocap_pos=None, mocap_quat=None, attachments=None) -> "Snapshot":
        """A snapshot of explicit state: ``qpos`` of shape ``(nq,)``, ``mocap_pos`` ``(nmocap, 3)`` and
        ``mocap_quat`` ``(nmocap, 4)`` as in ``MjData`` (omitted only when the scene has no mocap bodies), and
        ``attachments`` as for ``capture``. Malformed input is refused with an error naming the field."""
        native = scene.native
        nmocap = native.nmocap
        q = _real_array(qpos, "qpos", (native.nq,))
        mocap = []
        for field, value, width in (("mocap_pos", mocap_pos, 3), ("mocap_quat", mocap_quat, 4)):
            if value is None:
                if nmocap:
                    raise ValueError(f"{field}: the scene has {nmocap} mocap bodies; pass their poses")
                value = np.zeros((0, width))
            mocap.append(_real_array(value, field, (nmocap, width)).reshape(-1).tolist())
        snap = _load().Snapshot(native, q.tolist(), mocap[0], mocap[1], _native_attachments(scene, attachments))
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
