# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The native validator agrees with mj_manipulator's CollisionChecker on the checked-in corpus (#84, #138)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

mujoco = pytest.importorskip("mujoco")
from sscbirrt.backends import native_mujoco  # noqa: E402

if not native_mujoco.available():
    pytest.skip(native_mujoco.unavailable_reason(), allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "tests" / "reference" / "mujoco_collision_corpus.json"


@pytest.fixture(scope="module")
def corpus():
    return json.loads(ARTIFACT.read_text())


def _checker(scenario):
    model = mujoco.MjModel.from_xml_string(scenario["xml"])
    scene = native_mujoco.NativeScene.from_model(model, scenario["joints"])
    data = mujoco.MjData(model)
    data.qpos[:] = scenario["qpos"]
    if model.nmocap:
        data.mocap_pos[:] = np.array(scenario["mocap_pos"]).reshape(-1, 3)
        data.mocap_quat[:] = np.array(scenario["mocap_quat"]).reshape(-1, 4)
    attachments = {k: (v[0], np.array(v[1])) for k, v in scenario["attachments"].items()}
    snap = native_mujoco.Snapshot.capture(scene, data, attachments=attachments)
    return model, native_mujoco.NativeCollisionChecker(scene, snap)


def test_corpus_is_checked_in(corpus):
    assert corpus["scenarios"] and corpus["versions"]["mujoco"] == mujoco.__version__
    names = {s["name"] for s in corpus["scenarios"]}
    assert {
        "free_arm_no_margin",
        "free_arm_table_margin_2cm",
        "mocap_ball_in_the_workspace",
        "holding_the_can",
    } <= names


def test_native_decisions_equal_mj_manipulators(corpus):
    total = 0
    for scenario in corpus["scenarios"]:
        _, checker = _checker(scenario)
        for i, rec in enumerate(scenario["configurations"]):
            got = checker.is_valid(np.array(rec["q"]))
            assert got == rec["mj_manipulator_valid"], (
                f"{scenario['name']}[{i}] native={got} mj_manipulator={rec['mj_manipulator_valid']} "
                f"contacts={checker.invalid_contacts(np.array(rec['q']))} mjm={rec['invalid_contacts_mjm']}"
            )
            if rec["sscbirrt_valid"]:
                # The legacy checker counts environment-environment contacts too, so it is stricter; it may reject
                # what the policy accepts, never the reverse.
                assert got, f"{scenario['name']}[{i}] native rejects a configuration sscbirrt's legacy checker accepts"
            total += 1
    assert total > 200


def test_kinematics_plus_collision_equals_forward_on_the_corpus(corpus):
    for scenario in corpus["scenarios"]:
        assert all(r["forward_vs_kinematics_agree"] for r in scenario["configurations"]), scenario["name"]


def test_every_coverage_item_has_an_invalid_and_a_valid_configuration(corpus):
    by_name = {s["name"]: s for s in corpus["scenarios"]}
    for name in ("free_arm_no_margin", "free_arm_table_margin_2cm", "mocap_ball_in_the_workspace", "holding_the_can"):
        decisions = {r["mj_manipulator_valid"] for r in by_name[name]["configurations"]}
        assert decisions == {True, False}, name
    # Margins matter: on the margin scenario's own configurations, the same world without the margin accepts
    # something the margin world rejects (a 2 cm standoff from the table), never the reverse.
    margin = by_name["free_arm_table_margin_2cm"]
    _, with_margin = _checker(margin)
    plain_scenario = dict(margin, xml=margin["xml"].replace('margin="0.02"', 'margin="0.0"'))
    _, without = _checker(plain_scenario)
    qs = [np.array(r["q"]) for r in margin["configurations"]]
    assert all(without.is_valid(q) or not with_margin.is_valid(q) for q in qs)
    assert any(without.is_valid(q) and not with_margin.is_valid(q) for q in qs)


def test_regeneration_matches_when_mj_manipulator_is_available(corpus):
    pytest.importorskip("mj_manipulator")
    spec = importlib.util.spec_from_file_location(
        "mujoco_collision_corpus", ROOT / "tools" / "mujoco_collision_corpus.py"
    )
    tool = importlib.util.module_from_spec(spec)
    sys.modules["mujoco_collision_corpus"] = tool
    spec.loader.exec_module(tool)
    assert tool.main(["--check"]) == 0


class TestProperties:
    """Invariants that need no oracle."""

    @settings(max_examples=30, deadline=None)
    @given(q=st.lists(st.floats(-2.5, 2.5, allow_nan=False), min_size=3, max_size=3))
    def test_same_decision_across_calls_and_validators(self, q, corpus):
        scenario = corpus["scenarios"][3]  # holding_the_can
        _, a = _checker(scenario)
        b = native_mujoco.NativeCollisionChecker(a.scene, a.snapshot)
        qa = np.array(q)
        assert a.is_valid(qa) == a.is_valid(qa) == b.is_valid(qa) == bool(a.fresh().is_valid(list(qa)))

    def test_allowed_bodies_are_monotone(self, corpus):
        """Allowing more gripper-object contacts can only turn invalid into valid, never the reverse."""
        scenario = corpus["scenarios"][3]  # holding_the_can
        model = mujoco.MjModel.from_xml_string(scenario["xml"])
        scene = native_mujoco.NativeScene.from_model(model, scenario["joints"])
        mod = native_mujoco._load()
        data = mujoco.MjData(model)
        data.qpos[:] = scenario["qpos"]
        data.mocap_pos[:] = np.array(scenario["mocap_pos"]).reshape(-1, 3)
        gripper, T = scenario["attachments"]["can"]
        can, grip = scene.native.body_id("can"), scene.native.body_id(gripper)

        def checker(allowed):
            snap = mod.Snapshot(
                scene.native,
                np.asarray(data.qpos, dtype=float).tolist(),
                np.asarray(data.mocap_pos, dtype=float).reshape(-1).tolist(),
                np.asarray(data.mocap_quat, dtype=float).reshape(-1).tolist(),
                [mod.Attachment(can, grip, np.array(T).tolist(), allowed)],
            )
            return mod.SceneValidator(scene.native, snap)

        strict = checker([])
        normal = checker(list(scene.native.subtree(scene.native.body_id("arm/gripper/base"))))
        permissive = checker(list(scene.native.arm_bodies))
        seen_difference = False
        for r in scenario["configurations"]:
            q = list(map(float, r["q"]))
            a, b, c = strict.is_valid(q), normal.is_valid(q), permissive.is_valid(q)
            assert (not a or b) and (not b or c), (a, b, c)
            seen_difference = seen_difference or (a != b)
        assert seen_difference  # the grasp itself is an allowed contact somewhere in the corpus


class TestOnlyPairsThatCanChange:
    """#254: the validator collides only pairs involving the arm or a held object. Its answers must be exactly those
    of a full ``mj_collision`` (``filter_passive_pairs=False``, the reference path): every contact that counts, with
    its bodies, geoms, distance, kind and order, on every corpus world, at clearance 0 and 5 mm (#256: margins on
    every environment geom, in the validator's view), with and without the held can."""

    @staticmethod
    def _validators(scenario, clearance, attached):
        model = mujoco.MjModel.from_xml_string(scenario["xml"])
        scene = native_mujoco.NativeScene.from_model(model, scenario["joints"])
        data = mujoco.MjData(model)
        data.qpos[:] = scenario["qpos"]
        if model.nmocap:
            data.mocap_pos[:] = np.array(scenario["mocap_pos"]).reshape(-1, 3)
            data.mocap_quat[:] = np.array(scenario["mocap_quat"]).reshape(-1, 4)
        held = {k: (v[0], np.array(v[1])) for k, v in scenario["attachments"].items()} if attached else None
        snap = native_mujoco.Snapshot.capture(scene, data, attachments=held)
        mod = native_mujoco._load()
        return (
            scene,
            mod.SceneValidator(scene.native, snap.native, filter_passive_pairs=True, clearance=clearance),
            mod.SceneValidator(scene.native, snap.native, filter_passive_pairs=False, clearance=clearance),
        )

    @staticmethod
    def _contacts(validator, q):
        return [(c.body1, c.body2, c.geom1, c.geom2, c.dist, c.kind) for c in validator.invalid_contacts(q)]

    def test_filtered_contacts_equal_a_full_collision_on_the_corpus(self, corpus):
        compared, kinds, held_contacts, fewer_raw = 0, set(), 0, 0
        for scenario in corpus["scenarios"]:
            for clearance in (0.0, 0.005):
                for attached in {False, bool(scenario["attachments"])}:
                    scene, filtered, full = self._validators(scenario, clearance, attached)
                    can = scene.native.body_id("can")
                    for rec in scenario["configurations"]:
                        q = list(map(float, rec["q"]))
                        got, want = self._contacts(filtered, q), self._contacts(full, q)
                        assert got == want, (scenario["name"], clearance, attached, q)
                        assert filtered.is_valid(q) == full.is_valid(q)
                        kinds |= {c[5] for c in want}
                        held_contacts += attached and any(can in c[:2] for c in want)
                        fewer_raw += filtered.raw_contact_count(q) < full.raw_contact_count(q)
                        compared += 1
        # every configuration, at both clearances, with the can held and free where the world holds it
        assert compared == sum(2 * (1 + bool(s["attachments"])) * len(s["configurations"]) for s in corpus["scenarios"])
        assert kinds == {"self_collision", "robot_environment"}  # both kinds of contact were compared
        assert held_contacts > 0  # and contacts of the held can
        assert fewer_raw > 0  # the filter skipped environment pairs somewhere: the comparison is not vacuous

    @settings(max_examples=60, deadline=None)
    @given(
        q=st.lists(st.floats(-2.5, 2.5, allow_nan=False), min_size=3, max_size=3),
        which=st.integers(0, 4),
        clearance=st.sampled_from([0.0, 0.005]),
        attached=st.booleans(),
    )
    def test_filtered_contacts_equal_a_full_collision_anywhere(self, q, which, clearance, attached, corpus):
        scenario = corpus["scenarios"][which % len(corpus["scenarios"])]
        _, filtered, full = self._validators(scenario, clearance, attached)
        assert self._contacts(filtered, q) == self._contacts(full, q)
