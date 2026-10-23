# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""#230's evidence: a fresh run of tools/validation_artifact.py agrees with the stored artifact."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
from sscbirrt.backends import native_mujoco  # noqa: E402

if not native_mujoco.available():
    pytest.skip(native_mujoco.unavailable_reason(), allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("validation_artifact", ROOT / "tools" / "validation_artifact.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_a_fresh_run_has_the_stored_verdicts():
    stored = json.loads(tool.ARTIFACT.read_text())
    fresh = json.loads(json.dumps(tool.generate()))
    assert tool.semantic_view(fresh) == tool.semantic_view(stored)


def test_the_evidence_shows_what_230_asks():
    a = json.loads(tool.ARTIFACT.read_text())
    assert all(a["python_equals_native_verdicts"].values())
    for b in a["backends"].values():
        assert b["planning_equals_public"] == {"path": True, "trajectory": True}
        assert b["repeated_equal"] and b["reordered_equal"]
        public = b["public"]
        assert public["accepted_path"]["valid"] and public["reparameterization"]["valid"]
        w = public["rejected_path"]["witness"]
        assert (w["failure"], w["segment"]) == ("collision", 0) and w["contacts"]
        assert public["trace_changing"]["witness"]["time_s"] > 0


def test_the_geodude_edge_evidence_is_consistent():
    """ssrobot#132's edge, made inside ssrobot (tools/geodude_132_evidence.py): planning and the public call agree."""
    e = json.loads((ROOT / "tests" / "reference" / "geodude_132_evidence.json").read_text())
    rec = e["recorded_plan"]
    assert rec["result_equals_ci_artifact"] and rec["snapshot_fingerprint_matches"]
    assert rec["smoother_3_7_rerun_reproduces_trajectory"] and rec["python_equals_native"]
    assert rec["dense_recheck_3_7"]["invalid"] > 0 and rec["dense_recheck_3_7"]["pieces"] == [3]
    assert all(why is None for why in rec["rule_3_7_per_piece"])  # the 3.7 rule accepted every piece ...
    assert rec["this_validator"]["native"]["trajectory"]["valid"]  # ... and so does this one: the policy (#222)
    planned = e["planned_with_this_sscbirrt"]
    assert planned["planning_equals_public"] and planned["smoother_rerun_reproduces_trajectory"]


# #247's rule, stated per case: whether each snapshot's state is valid, and the disallowed pairs in MuJoCo's order.
# The pairs cover both orderings: the held can first against the box table, second against the sphere post.
TABLE = [("can", "table", "robot_environment")]
POST = [("post", "can", "robot_environment")]
OTHER = [("can", "other/link", "robot_environment")]
EXPECTED_247 = {
    "can_on_table": {"strict": TABLE, "table": [], "strict_again": TABLE},
    "can_on_post": {"strict": POST, "table": POST, "post": []},  # allowing the table does not allow the post
    "can_on_stationary_robot": {"strict": OTHER, "other": []},
    "arm_on_table": {k: [("table", "arm/gripper/base", "robot_environment")] for k in ("strict", "table")},
    "held_on_held": {k: [("can", "box2", "self_collision")] for k in ("strict", "each_other")},
    "excluded_pair": {"strict": [], "bin": []},
    "crate_not_excluded": {k: [("arm/link", "crate", "robot_environment")] for k in ("strict", "crate")},
}


def test_the_evidence_shows_what_247_asks():
    a = json.loads(tool.ARTIFACT.read_text())
    view = tool._allowance_view(a["allowances"])
    assert view["one_scene_provenance"]  # every snapshot is checked against one native scene
    for name, expected in EXPECTED_247.items():
        verdicts = view[name]["verdicts"]
        assert list(verdicts) == list(expected), name
        for label, pairs in expected.items():
            for backend in ("python", "native"):
                got = verdicts[label][backend]
                assert got == {"valid": not pairs, "contacts": [list(p) for p in pairs]}, (name, label, backend)
    # Snapshots that differ only in their allowances differ in hash; the same allowances give the same hash.
    assert view["can_on_table"]["same_hash"] == [["strict", "strict_again"], ["table"]]
    assert all(len(group) == 1 for name in EXPECTED_247 if name != "can_on_table" for group in view[name]["same_hash"])


# #256's contract, per case: whether the state is valid at clearance 0 and at 5 mm (both backends).
EXPECTED_256 = {
    "primitive_inside": (True, False),
    "primitive_outside": (True, True),
    "mesh_inside": (True, False),  # a two-geom body: reached only through BVH traversal, so body_margin matters
    "mesh_outside": (True, True),
    "stationary_robot_inside": (True, False),  # a robot body outside the planned joints is environment
    "stationary_robot_outside": (True, True),
    "held_near_environment_inside": (True, False),
    "held_near_environment_outside": (True, True),
    "held_near_own_arm": (True, True),  # 3 mm from its own arm, nothing allowed: no clearance within the moving set
    "held_on_allowed_body": (True, True),  # touching an allowed body stays allowed
    "held_on_body_not_allowed": (False, False),
    "predefined_pair_inside": (True, False),  # a predefined moving-environment pair gets the clearance
    "predefined_pair_outside": (True, True),
}


def test_the_evidence_shows_what_256_asks():
    a = json.loads(tool.ARTIFACT.read_text())
    v = tool._clearance_view(a["clearance"])
    assert set(v["cases"]) == set(EXPECTED_256)
    for name, (at_zero, at_5mm) in EXPECTED_256.items():
        for clearance, want in (("0", at_zero), ("0.005", at_5mm)):
            got = v["cases"][name][clearance]
            assert got["python"] == got["native"] and got["python"]["valid"] is want, (name, clearance, got)
            if not want and name != "held_on_body_not_allowed":
                assert {k for _, _, k in got["python"]["contacts"]} == {"robot_environment"}, (name, got)
    # The validators' own view: what moves, and the margins they collide with (environment raised, moving set not).
    held = v["classification"]["holding_can"]
    assert held["moving_bodies"] == ["arm", "can"]
    for geom, body, margin in held["geoms"]:
        assert margin["0"] == 0.0 and margin["0.005"] == (0.0 if body in ("arm", "can") else 0.005), geom
    assert held["pair_margin"] == {"0": [0.0], "0.005": [0.005]}
    assert v["classification"]["empty_hand"]["moving_bodies"] == ["arm"]
    # Refused before any check, and only for a positive clearance: override, and a flex that can touch the arm.
    assert v["refusals"] == {
        "override": (True, True),
        "flex_that_can_touch_the_moving_set": (True, True),
        "flex_with_collisions_off": (True, False),
    }
    assert v["interleaved_equal_alone"] and v["oracle_agrees"]
    assert a["clearance"]["model_margin_oracle"]["invalid_configurations"] > 0  # the oracle compared real contacts
    # Plans: the policy and provenance record the clearance; one scene serves both.
    for backend in ("python", "native"):
        for c, text in ((0.0, "0"), (0.005, "0.005")):
            success, parameters, recorded = v["plans"][f"{backend}_{c:g}"]
            assert success and ["clearance_m", text] in parameters and recorded == c
    assert v["one_scene_for_every_plan"]
    assert v["mutation"] == {"caller_model_unchanged": True, "scene_model_unchanged": True}
