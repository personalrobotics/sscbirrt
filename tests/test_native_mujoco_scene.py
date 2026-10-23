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


FINGER, BASE = "robot/gripper/finger", "robot/gripper/base"


class TestPublicConstruction:
    """#206: snapshots with explicit allowed bodies through the public API; malformed input is refused at ingress."""

    @pytest.fixture
    def scene(self, model):
        return native_mujoco.NativeScene.from_model(model, JOINTS)

    @staticmethod
    def _state(model):
        data = mujoco.MjData(model)
        data.qpos[:2] = [0.4, -0.3]
        data.mocap_pos[0] = [0.5, 0.5, 0.5]
        return data

    def _snap(self, scene, model, allowed, **kw):
        data = self._state(model)
        att = native_mujoco.Attachment("can", FINGER, np.eye(4), allowed_bodies=allowed)  # the can on the finger
        return native_mujoco.Snapshot.from_state(scene, data.qpos, data.mocap_pos, data.mocap_quat, [att], **kw)

    def test_none_empty_and_explicit_allowed_bodies_mean_three_things(self, scene, model):
        # The can, held on the finger, contains the finger (no contact) and touches the gripper base and link1's end.
        q = np.array([0.4, -0.3])

        def touching(snapshot):
            contacts = native_mujoco.NativeCollisionChecker(scene, snapshot).invalid_contacts(q)
            assert {c["kind"] for c in contacts} <= {"self_collision"}
            return {c["body1"] if c["body2"] == "can" else c["body2"] for c in contacts}

        derived = self._snap(scene, model, None)
        legacy = native_mujoco.Snapshot.capture(scene, self._state(model), {"can": (FINGER, np.eye(4))})
        assert derived.sha256 == legacy.sha256  # None is the existing derived gripper-subtree rule
        assert touching(derived) == {"robot/link1"}
        assert touching(self._snap(scene, model, ())) == {"robot/link1", BASE}  # () allows nothing
        # Exact bodies, not subtrees: link1 alone leaves the base; link1 and the base make the state valid.
        assert touching(self._snap(scene, model, ["robot/link1"])) == {BASE}
        assert touching(self._snap(scene, model, ["robot/link1", BASE])) == set()
        assert self._snap(scene, model, [FINGER, BASE]).sha256 == derived.sha256

    @settings(max_examples=25, deadline=None)
    @given(st.permutations([FINGER, BASE, "robot/link1", "world"]), st.lists(st.booleans(), min_size=4, max_size=4))
    def test_names_ids_order_and_duplicates_give_one_snapshot(self, order, as_id):
        model = mujoco.MjModel.from_xml_string(XML)
        scene = native_mujoco.NativeScene.from_model(model, JOINTS)
        refs = [scene.native.body_id(b) if i else b for b, i in zip(order, as_id)]
        canonical = self._snap(scene, model, [BASE, FINGER, "robot/link1", "world"])
        assert self._snap(scene, model, refs + refs[:2]).sha256 == canonical.sha256
        assert self._snap(scene, model, refs[:3]).sha256 != canonical.sha256

    def test_capture_and_from_state_give_the_same_snapshot_and_decisions(self, scene, model):
        data = self._state(model)
        atts = [native_mujoco.Attachment("can", FINGER, np.eye(4), allowed_bodies=[FINGER])]
        a = native_mujoco.Snapshot.capture(scene, data, atts)
        b = native_mujoco.Snapshot.from_state(
            scene, data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy(), atts
        )
        assert a.sha256 == b.sha256
        ca, cb = native_mujoco.NativeCollisionChecker(scene, a), native_mujoco.NativeCollisionChecker(scene, b)
        for q in ([0.4, -0.3], [0.0, 0.0], [1.5, 1.0], [-1.0, -2.5]):
            assert ca.is_valid(np.array(q)) == cb.is_valid(np.array(q))
            assert ca.invalid_contacts(np.array(q)) == cb.invalid_contacts(np.array(q))

    def test_malformed_input_is_refused_naming_the_field(self, scene, model):
        A, eye = native_mujoco.Attachment, np.eye(4)
        data = self._state(model)

        def snap(atts=None, qpos=data.qpos, mocap_pos=data.mocap_pos, mocap_quat=data.mocap_quat):
            return native_mujoco.Snapshot.from_state(scene, qpos, mocap_pos, mocap_quat, atts)

        scaled = np.diag([2.0, 1.0, 1.0, 1.0])
        cases = [
            (lambda: A("can", FINGER, eye, allowed_bodies=BASE), TypeError, "allowed_bodies"),
            (lambda: A("can", FINGER, eye, allowed_bodies={BASE: 1}), TypeError, "allowed_bodies"),
            (lambda: A("can", FINGER, eye, allowed_bodies=[BASE, True]), TypeError, r"allowed_bodies\[1\]"),
            (lambda: A("can", FINGER, eye, allowed_bodies=[1.0]), TypeError, r"allowed_bodies\[0\]"),
            (lambda: A(None, FINGER, eye), TypeError, "object_body"),
            (lambda: A("can", False, eye), TypeError, "gripper_body"),
            (lambda: A("can", FINGER, np.eye(3)), ValueError, "T_gripper_object"),
            (lambda: A("can", FINGER, np.full((4, 4), np.nan)), ValueError, "T_gripper_object"),
            (lambda: A("can", FINGER, np.eye(4, dtype=bool)), TypeError, "T_gripper_object"),
            (lambda: snap([A("can", FINGER, eye, allowed_bodies=[BASE, "nope"])]), ValueError,
             r"attachments\[0\]\.allowed_bodies\[1\]: body 'nope' not found"),
            (lambda: snap([A("can", FINGER, eye, allowed_bodies=[-1])]), ValueError, "out of range"),
            (lambda: snap([A("can", FINGER, eye, allowed_bodies=[10_000])]), ValueError, "out of range"),
            (lambda: snap([A("nope", FINGER, eye)]), ValueError, r"attachments\[0\]\.object_body"),
            (lambda: snap([A("robot/link1", FINGER, eye)]), ValueError, r"object_body: .*free joint"),
            (lambda: snap([A("can", "target", eye)]), ValueError, r"gripper_body: .*not part of the arm"),
            (lambda: snap([A("can", FINGER, scaled)]), ValueError, "T_gripper_object"),
            (lambda: snap([A("can", FINGER, eye), A(scene.native.body_id("can"), BASE, eye)]), ValueError,
             "attached twice"),
            (lambda: snap([("can", FINGER, eye)]), TypeError, r"attachments\[0\]"),
            (lambda: snap("can"), TypeError, "attachments"),
            (lambda: snap(qpos=data.qpos[:-1]), ValueError, "qpos"),
            (lambda: snap(qpos=["0"] * model.nq), TypeError, "qpos"),
            (lambda: snap(qpos=np.zeros(model.nq, dtype=bool)), TypeError, "qpos"),
            (lambda: snap(qpos=np.full(model.nq, np.inf)), ValueError, "qpos"),
            (lambda: snap(mocap_pos=None), ValueError, "mocap_pos"),
            (lambda: snap(mocap_quat=np.ones((1, 3))), ValueError, "mocap_quat"),
            (lambda: snap(mocap_quat=np.full((1, 4), 0.9)), ValueError, "mocap_quat"),
        ]  # fmt: skip
        for i, (build, error, match) in enumerate(cases):
            with pytest.raises(error, match=match):
                build()
                pytest.fail(f"case {i} was accepted")


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


class TestClearanceIngress:
    """#256: a clearance is a finite real number of metres >= 0; anything else is refused where it enters."""

    @pytest.fixture
    def parts(self, model):
        scene = native_mujoco.NativeScene.from_model(model, JOINTS)
        return scene, native_mujoco.Snapshot.capture(scene, mujoco.MjData(model))

    @pytest.mark.parametrize(
        "bad, error",
        [
            (True, TypeError),
            (np.bool_(False), TypeError),
            ("0.005", TypeError),
            (None, TypeError),
            ([0.005], TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (-0.001, ValueError),
            (-0.0 - 1e-12, ValueError),
        ],
    )
    def test_malformed_clearance_is_refused(self, parts, bad, error):
        from sscbirrt import ValidationPolicy

        scene, snap = parts
        with pytest.raises(error, match="clearance"):
            native_mujoco.NativeCollisionChecker(scene, snap, clearance=bad)
        with pytest.raises(error, match="clearance"):
            ValidationPolicy(clearance_m=bad)

    def test_numbers_are_accepted_as_metres(self, parts):
        scene, snap = parts
        for good, want in ((0, 0.0), (np.float32(0.005), float(np.float32(0.005))), (np.int64(0), 0.0), (0.01, 0.01)):
            assert native_mujoco.NativeCollisionChecker(scene, snap, clearance=good).clearance_m == want

    def test_a_policy_stating_another_clearance_than_its_checker_is_refused(self, parts):
        from sscbirrt import JointSpace, ValidationPolicy, Validator

        scene, snap = parts
        checker = native_mujoco.NativeCollisionChecker(scene, snap, clearance=0.005)
        space = JointSpace(np.array([-2.0, -3.0]), np.array([2.0, 3.0]))
        assert Validator(space, checker).policy.clearance_m == 0.005  # taken from the checker
        assert Validator(space, checker, policy=ValidationPolicy(clearance_m=0.005)).policy.clearance_m == 0.005
        with pytest.raises(ValueError, match="clearance"):
            Validator(space, checker, policy=ValidationPolicy(clearance_m=0.01))
