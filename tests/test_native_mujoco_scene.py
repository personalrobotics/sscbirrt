# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The owned MuJoCo scene and immutable snapshots from Python (#93, #137)."""

import gc

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

mujoco = pytest.importorskip("mujoco")
from sscbirrt.backends import native_mujoco  # noqa: E402
from sscbirrt.backends.native import NativeUnsupported  # noqa: E402

if not native_mujoco.available():
    pytest.skip(native_mujoco.unavailable_reason(), allow_module_level=True)

XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="robot/base" pos="0 0 0.1">
      <joint name="j0" type="hinge" axis="0 0 1" limited="true" range="-2 2"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0"/>
      <body name="robot/link1" pos="0.3 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0"/>
        <body name="robot/gripper/base" pos="0.3 0 0">
          <geom type="box" size="0.02 0.02 0.02"/>
          <body name="robot/gripper/finger" pos="0.03 0 0"><geom type="box" size="0.01 0.01 0.02"/></body>
        </body>
      </body>
    </body>
    <body name="can" pos="0.8 0 0.2"><freejoint/><geom type="cylinder" size="0.03 0.06"/></body>
    <body name="target" mocap="true" pos="1 1 1"><geom type="sphere" size="0.02" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>
"""
JOINTS = ["j0", "j1"]


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(XML)


class TestScene:
    def test_versions_and_provenance(self, model):
        scene = native_mujoco.NativeScene.from_model(model, JOINTS)
        p = scene.provenance
        assert p["mujoco"] == mujoco.__version__ == native_mujoco._load().compiled_mujoco_version()
        assert p["model_signature"] == model.signature and len(p["mjb_sha256"]) == 64
        lo, hi = scene.joint_limits
        assert (lo[0], hi[0]) == (-2.0, 2.0) and lo[1] == -np.inf and hi[1] == np.inf

    def test_scene_is_captured_per_model_and_survives_it(self, model):
        def valid(scene, q=(0.0, 0.0)):
            data = mujoco.MjData(mujoco.MjModel.from_xml_string(XML))
            data.qpos[:2] = q
            snap = native_mujoco.Snapshot.capture(scene, data)
            return native_mujoco.NativeCollisionChecker(scene, snap).is_valid(np.array(q))

        a = native_mujoco.NativeScene.from_model(model, JOINTS)
        assert native_mujoco.NativeScene.from_model(model, JOINTS) is a and valid(a)
        sig = model.signature
        # A numeric change in place that grows the can into the gripper (with the bounds MuJoCo's broad phase
        # reads). MuJoCo's signature does not see it.
        can = model.body_geomadr[model.body("can").id]
        model.geom_size[can], model.geom_aabb[can] = [0.2, 0.12, 0], [0, 0, 0, 0.2, 0.2, 0.12]
        model.geom_rbound[can] = np.hypot(0.2, 0.12)
        assert model.signature == sig
        # The model was captured on first use: without refresh the edit is not seen.
        assert native_mujoco.NativeScene.from_model(model, JOINTS) is a and valid(a)
        c = native_mujoco.NativeScene.from_model(model, JOINTS, refresh=True)
        assert c is not a and c.provenance["mjb_sha256"] != a.provenance["mjb_sha256"] and not valid(c)
        assert native_mujoco.NativeScene.from_model(model, JOINTS) is c  # the refreshed scene is the cached one
        # Scene parameters never alias, for the same model or across models.
        assert native_mujoco.NativeScene.from_model(model, ["j0"]) is not c
        assert native_mujoco.NativeScene.from_model(model, JOINTS, ["can"]) is not c
        # A new model object is hashed in full: same content shares the scene, different content never does.
        assert native_mujoco.NativeScene.from_model(mujoco.MjModel.from_xml_string(XML), JOINTS) is a
        del model
        gc.collect()
        assert a.native.body_id("can") >= 0 and a.provenance["model_signature"] == sig
        other = mujoco.MjModel.from_xml_string(XML.replace('name="can"', 'name="tin"'))  # structural: renamed body
        assert native_mujoco.NativeScene.from_model(other, JOINTS) is not a

    def test_a_dead_models_entry_is_not_inherited_through_id_reuse(self):
        # CPython reuses a freed object's address, so id(model) repeats; the cache must not. Every model here has
        # different content, so a reused id must still get a different scene.
        dead = {}
        for i in range(50):
            model = mujoco.MjModel.from_xml_string(XML.replace('pos="0.8 0 0.2"', f'pos="{0.8 + i / 1000} 0 0.2"'))
            scene = native_mujoco.NativeScene.from_model(model, JOINTS)
            if id(model) in dead:
                assert scene is not dead[id(model)]
                assert scene.provenance["mjb_sha256"] != dead[id(model)].provenance["mjb_sha256"]
                return
            dead[id(model)] = scene
            del model
            gc.collect()
        pytest.skip("the interpreter never reused a model's id")

    def test_rejections(self, model):
        mod = native_mujoco._load()
        mjb = native_mujoco.export_mjb(model)
        with pytest.raises(ValueError, match="duplicate"):
            mod.Scene(mjb, ["j0", "j0"])
        with pytest.raises(ValueError, match="not found"):
            mod.Scene(mjb, ["nope"])
        with pytest.raises(ValueError, match="could not be loaded"):
            mod.Scene(mjb[: len(mjb) // 2], JOINTS)
        with pytest.raises(ValueError, match="at least one"):
            mod.Scene(mjb, [])


class TestSnapshot:
    def test_capture_copies_and_later_changes_do_not_reach_it(self, model):
        scene = native_mujoco.NativeScene.from_model(model, JOINTS)
        data = mujoco.MjData(model)
        data.qpos[0] = 0.7
        snap = native_mujoco.Snapshot.capture(scene, data)
        before = snap.sha256
        data.qpos[0] = -1.0
        data.mocap_pos[0] = [5.0, 5.0, 5.0]
        assert snap.sha256 == before and snap.qpos[0] == 0.7

    def test_attachments_resolve_the_gripper_rule_to_body_ids(self, model):
        scene = native_mujoco.NativeScene.from_model(model, JOINTS)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        snap = native_mujoco.Snapshot.capture(scene, data, attachments={"can": ("robot/gripper/finger", np.eye(4))})
        att = snap.native.attachments[0]
        base = scene.native.body_id("robot/gripper/base")
        finger = scene.native.body_id("robot/gripper/finger")
        assert att.object_body == scene.native.body_id("can") and att.gripper_body == finger
        assert att.allowed_bodies == sorted([base, finger])  # <prefix>/base exists: its subtree is allowed
        with pytest.raises(ValueError, match="not found"):
            native_mujoco.Snapshot.capture(scene, data, attachments={"nope": ("robot/gripper/finger", np.eye(4))})
        with pytest.raises(ValueError, match="free joint"):
            native_mujoco.Snapshot.capture(
                scene, data, attachments={"robot/link1": ("robot/gripper/finger", np.eye(4))}
            )
        with pytest.raises(ValueError, match="rotation|orthonormal|determinant|homogeneous"):
            T = np.eye(4)
            T[0, 0] = 2.0
            native_mujoco.Snapshot.capture(scene, data, attachments={"can": ("robot/gripper/finger", T)})

    @settings(max_examples=25, deadline=None)
    @given(
        q=st.lists(st.floats(-2, 2, allow_nan=False), min_size=2, max_size=2), scale=st.floats(-3, 3, allow_nan=False)
    )
    def test_snapshot_hash_is_a_function_of_content_only(self, q, scale):
        model = mujoco.MjModel.from_xml_string(XML)
        scene = native_mujoco.NativeScene.from_model(model, JOINTS)
        d1, d2 = mujoco.MjData(model), mujoco.MjData(model)
        for d in (d1, d2):
            d.qpos[:2] = q
            d.mocap_pos[0] = [scale, 0.0, 1.0]
        a = native_mujoco.Snapshot.capture(scene, d1)
        b = native_mujoco.Snapshot.capture(scene, d2)
        assert a.sha256 == b.sha256
        d2.qpos[0] = q[0] + 1e-9 if q[0] < 1 else q[0] - 1e-9
        assert native_mujoco.Snapshot.capture(scene, d2).sha256 != a.sha256


def test_unavailable_reason_is_a_string_when_the_module_is_missing(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "sscbirrt._native_mujoco":
            raise ImportError("simulated missing module")
        return real(name, *a, **k)

    monkeypatch.setattr(native_mujoco, "_module", None)
    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(NativeUnsupported, match="built without MuJoCo support"):
        native_mujoco._load()
    monkeypatch.setattr(importlib, "import_module", real)
    monkeypatch.setattr(native_mujoco, "_module", None)
    assert native_mujoco.available()


def test_version_mismatch_is_refused_with_all_three_versions(monkeypatch):
    monkeypatch.setattr(native_mujoco, "_module", None)
    monkeypatch.setattr(mujoco, "__version__", "9.9.9")
    with pytest.raises(NativeUnsupported, match="built against 3.14.0.*installed mujoco package 9.9.9"):
        native_mujoco._load()
    monkeypatch.undo()
    monkeypatch.setattr(native_mujoco, "_module", None)
    assert native_mujoco.available()
